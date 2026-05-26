from math import inf, sqrt
from copy import deepcopy
import os
import signal
import sys

import torch
from tqdm import tqdm

from ..utils.misc import relative_errors, save


def _get_batch_errors(model, kind, batch):
    actual = batch["u"]
    weights_x = batch["u/weights_x"]
    weights_y = batch["u/weights_y"]
    if kind == "model":
        predicted = model(batch)
    elif kind == "branch":
        predicted = model.call_system_net(batch)
    else:
        raise ValueError(f'"kind" must be one of "model", "branch" (got "{kind}")')

    return relative_errors(predicted, actual, weights_x, weights_y)


def _get_overall_error(model, kind, dataset, *, device="cuda", batch_size=20):
    model.eval()
    N = len(dataset)
    indexloader = torch.utils.data.DataLoader(range(N), batch_size=batch_size)
    total = 0.0
    with torch.no_grad():
        for indices in indexloader:
            batch = dataset.get_batch(indices, device)
            rel_errs = _get_batch_errors(model, kind, batch)
            loss_batch = rel_errs.sum()
            total += loss_batch

    return (total / len(dataset)).sqrt().item()


def train_model(
    *,
    model,
    kind,
    device="cuda",
    data_train,
    data_validate=None,
    nr_epochs,
    update_every=1,
    output_name,
    learning_rate=0.001,
    batch_size=100,
    attributes={},
    optimiser_state=None,
    batch_size_eval=100,
    opt="adam",
):
    # Before we do anything, check if the location where the output will be stored exists or not.
    # If it does, then we should obviously return an error immediately, instead of after training
    # has completed
    prefix = f"output/{output_name}/"
    abort = False
    try:
        os.makedirs(prefix)
    except FileExistsError:
        if len(os.listdir(prefix)) == 0:
            print(f"Output directory {prefix} was empty.", file=sys.stderr)
        else:
            abort = True

    if abort:
        raise FileExistsError(
            f"output directory {prefix} already exists and is nonempty. Delete or empty the "
            "existing folder, or choose a different destination"
        )

    # Set up a signal handler so that we can send USR1 to the python process to save a snapshot of
    # the training model. This is useful if we want to obtain a partially-trained model while still
    # letting the model train until completion
    pid = os.getpid()
    got_USR1 = False

    def handle_usr1(_signal, _stack):
        nonlocal got_USR1
        got_USR1 = True

    signal.signal(signal.SIGUSR1, handle_usr1)

    model_before = deepcopy(model)
    model.to(device)
    model.train()
    optimiser = None
    if opt == "adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif opt == "soap":
        optimiser = SOAP(model.parameters())
    else:
        raise ValueError(f'unknown optimiser "{opt}"')

    if optimiser_state is not None:
        optimiser.load_state_dict(optimiser_state)

    history_train = []
    history_validate = None if data_validate is None else []

    best_loss = inf
    best_params = deepcopy(model.state_dict())
    iterable = tqdm(range(nr_epochs), ascii=True, dynamic_ncols=True)
    n_samples = len(data_train)
    indexloader = torch.utils.data.DataLoader(range(n_samples), batch_size=batch_size, shuffle=True)
    try:
        for i in iterable:
            loss_train = 0.0
            for indices in indexloader:
                batch = data_train.get_batch(indices, device)
                rel_errs = _get_batch_errors(model, kind, batch)
                loss_train_batch = rel_errs.sum()
                optimiser.zero_grad()
                loss_train_batch.backward()
                optimiser.step()
                loss_train += loss_train_batch.item()

            loss_train /= n_samples
            loss_train = sqrt(loss_train)
            history_train.append(loss_train)
            if (i + 1) % update_every == 0:
                if data_validate is None:
                    loss = loss_train
                else:
                    loss = _get_overall_error(
                        model, kind, data_validate, device=device, batch_size=batch_size_eval
                    )
                    history_validate.append(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_params = deepcopy(model.state_dict())

                if data_validate is None:
                    iterable.set_description_str(f"pid={pid} rms={loss:.4e} best={best_loss:.4e}")
                else:
                    iterable.set_description_str(
                        f"pid={pid} train={loss_train:.4e} val={loss:.4e} best={best_loss:.4e}"
                    )

            # If we got a USR1 signal, save an intermediate result and print that we did
            if got_USR1:
                prefix_i = prefix + f"epoch-{i}/"
                os.makedirs(prefix_i)
                save_all(
                    model,
                    model_before,
                    best_params,
                    history_train,
                    history_validate,
                    prefix_i,
                    optimiser,
                    attributes,
                )
                iterable.write(f"Stored intermediate result in '{prefix_i}'")
                got_USR1 = False

    except KeyboardInterrupt:
        epochs_done = len(history_train)
        print(f"Training interrupted after {epochs_done} out of {nr_epochs} epochs")
        pass

    save_all(
        model,
        model_before,
        best_params,
        history_train,
        history_validate,
        prefix,
        optimiser,
        attributes,
    )


def save_all(
    model,
    model_before,
    best_params,
    history_train,
    history_validate,
    prefix,
    optimiser,
    attributes,
):
    # Training is finished. So now restore the best parameters (based on validation loss) and save
    # everything.
    model_best = deepcopy(model)
    model_before = deepcopy(model_before)
    model_last = deepcopy(model)
    model_best.load_state_dict(best_params)
    # Somewhat weirdly, obj.to(device) works in-place on torch.nn.Modules, but out of place on
    # torch.Tensors
    model_last.to("cpu")
    model_best.to("cpu")
    model_before.to("cpu")

    history_train = torch.tensor(history_train, device="cpu")
    history_validate = (
        None if history_validate is None else torch.tensor(history_validate, device="cpu")
    )

    save(model_best, prefix + "model.pt")
    save(model_before, prefix + "model_before.pt")
    save(model_last, prefix + "model_last.pt")
    save(
        {
            "model_description": str(model),
            "state_dict": model_best.state_dict(),
            "state_dict_before": model_before.state_dict(),
            "state_dict_last": model_last.state_dict(),
        },
        prefix + "model_description.pt",
    )
    save({"train": history_train, "validate": history_validate}, prefix + "history.pt")
    save(
        {"description": str(optimiser), "state_dict": optimiser.state_dict()},
        prefix + "optimiser.pt",
    )

    # Finally, save the attributes, which are just arbitrary other data that I wanted to keep. Note
    # that I often include the script in this, but the script is more useful to store separately
    attributes = dict(attributes)
    if "script" in attributes:
        with open(prefix + "script.py", "w") as f:
            f.write(attributes["script"])

        del attributes["script"]

    if len(attributes) != 0:
        save(attributes, prefix + "attributes.pt")
