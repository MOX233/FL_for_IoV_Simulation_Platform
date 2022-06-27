"""lstm_utils.py contains utility functions for running LSTM Baselines."""

import torch
import time
import os
import tempfile
import shutil
import utils.baseline_utils as baseline_utils
import utils.baseline_config as config
import torch.nn.functional as F
import torch.nn as nn
import pickle as pkl
import pandas as pd
import numpy as np

from typing import Any, Dict, List, Tuple, Union
from termcolor import cprint
from torch.utils.data import Dataset
from utils.logger import Logger
from utils.baseline_utils import viz_predictions
from utils.baseline_config import FEATURE_FORMAT
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics


class LSTMDataset(Dataset):
    """PyTorch Dataset for LSTM Baselines."""

    def __init__(self, data_dict: Dict[str, Any], args: Any, mode: str):
        """Initialize the Dataset.

        Args:
            data_dict: Dict containing all the data
            args: Arguments passed to the baseline code
            mode: train/val/test mode

        """
        self.data_dict = data_dict
        self.args = args
        self.mode = mode

        # Get input
        self.input_data = data_dict["{}_input".format(mode)]
        if mode != "test":
            self.output_data = data_dict["{}_output".format(mode)]
        self.data_size = self.input_data.shape[0]

        # Get helpers
        self.helpers = self.get_helpers()
        self.helpers = list(zip(*self.helpers))

    def __len__(self):
        """Get length of dataset.

        Returns:
            Length of dataset

        """
        return self.data_size

    def __getitem__(self, idx: int
                    ) -> Tuple[torch.FloatTensor, Any, Dict[str, np.ndarray]]:
        """Get the element at the given index.

        Args:
            idx: Query index

        Returns:
            A list containing input Tensor, Output Tensor (Empty if test) and viz helpers. 

        """
        return (
            torch.FloatTensor(self.input_data[idx]),
            torch.empty(1) if self.mode == "test" else torch.FloatTensor(
                self.output_data[idx]),
            self.helpers[idx],
        )

    def get_helpers(self) -> Tuple[Any]:
        """Get helpers for running baselines.

        Returns:
            helpers: Tuple in the format specified by LSTM_HELPER_DICT_IDX

        Note: We need a tuple because DataLoader needs to index across all these helpers simultaneously.

        """
        helper_df = self.data_dict[f"{self.mode}_helpers"]
        candidate_centerlines = helper_df["CANDIDATE_CENTERLINES"].values
        candidate_nt_distances = helper_df["CANDIDATE_NT_DISTANCES"].values
        xcoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["X"]].astype("float")
        ycoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["Y"]].astype("float")
        centroids = np.stack((xcoord, ycoord), axis=2)
        _DEFAULT_HELPER_VALUE = np.full((centroids.shape[0]), None)
        city_names = np.stack(helper_df["FEATURES"].values
                              )[:, :, config.FEATURE_FORMAT["CITY_NAME"]]
        seq_paths = helper_df["SEQUENCE"].values
        translation = (helper_df["TRANSLATION"].values
                       if self.args.normalize else _DEFAULT_HELPER_VALUE)
        rotation = (helper_df["ROTATION"].values
                    if self.args.normalize else _DEFAULT_HELPER_VALUE)

        use_candidates = self.args.use_map and self.mode == "test"

        candidate_delta_references = (
            helper_df["CANDIDATE_DELTA_REFERENCES"].values
            if self.args.use_map and use_candidates else _DEFAULT_HELPER_VALUE)
        delta_reference = (helper_df["DELTA_REFERENCE"].values
                           if self.args.use_delta and not use_candidates else
                           _DEFAULT_HELPER_VALUE)

        helpers = [None for i in range(len(config.LSTM_HELPER_DICT_IDX))]

        # Name of the variables should be the same as keys in LSTM_HELPER_DICT_IDX
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers[v] = locals()[k.lower()]

        return tuple(helpers)


class ModelUtils:
    """Utils for LSTM baselines."""

    def save_checkpoint(self, save_dir: str, state: Dict[str, Any]) -> None:
        """Save checkpoint file.

        Args:
            save_dir: Directory where model is to be saved
            state: State of the model

        """
        filename = "{}/LSTM_rollout{}.pth.tar".format(save_dir,
                                                      state["rollout_len"])
        torch.save(state, filename)

    def load_checkpoint(
            self,
            checkpoint_file: str,
            encoder: Any,
            decoder: Any,
            encoder_optimizer: Any,
            decoder_optimizer: Any,
    ) -> Tuple[int, int, float]:
        """Load the checkpoint.

        Args:
            checkpoint_file: Path to checkpoint file
            encoder: Encoder model
            decoder: Decoder model 

        Returns:
            round: round when the model was saved.
            rollout_len: horizon used
            best_loss: loss when the checkpoint was saved

        """
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            round = checkpoint["round"]
            best_loss = checkpoint["best_loss"]
            rollout_len = checkpoint["rollout_len"]
            """
            if use_cuda:
                encoder.module.load_state_dict(
                    checkpoint["encoder_state_dict"])
                decoder.module.load_state_dict(
                    checkpoint["decoder_state_dict"])
            else:
                encoder.load_state_dict(checkpoint["encoder_state_dict"])
                decoder.load_state_dict(checkpoint["decoder_state_dict"])
            """
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            decoder.load_state_dict(checkpoint["decoder_state_dict"])
            encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
            decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
            print(
                f"=> loaded checkpoint {checkpoint_file} (round: {round}, loss: {best_loss})"
            )
        else:
            print(f"=> no checkpoint found at {checkpoint_file}")

        return round, rollout_len, best_loss

    def my_collate_fn(self, batch: List[Any]) -> List[Any]:
        """Collate function for PyTorch DataLoader.

        Args:
            batch: Batch data

        Returns: 
            input, output and helpers in the format expected by DataLoader

        """
        _input, output, helpers = [], [], []

        for item in batch:
            _input.append(item[0])
            output.append(item[1])
            helpers.append(item[2])
        _input = torch.stack(_input)
        output = torch.stack(output)
        return [_input, output, helpers]

    def init_hidden(self, batch_size: int,
                    hidden_size: int, device) -> Tuple[Any, Any]:
        """Get initial hidden state for LSTM.

        Args:
            batch_size: Batch size
            hidden_size: Hidden size of LSTM

        Returns:
            Initial hidden states

        """
        return (
            torch.zeros(batch_size, hidden_size).to(device),
            torch.zeros(batch_size, hidden_size).to(device),
        )


class EncoderRNN(nn.Module):
    """Encoder Network."""

    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 16):
        """Initialize the encoder network.

        Args:
            input_size: number of features in the input
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM

        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        return hidden


class DecoderRNN(nn.Module):
    """Decoder Network."""

    def __init__(self, embedding_size=8, hidden_size=16, output_size=2):
        """Initialize the decoder network.

        Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output

        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            output: output from lstm
            hidden: final hidden state

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden


def train(
        args,
        device,
        train_loader: Any,
        round: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        decoder: Any,
        encoder_optimizer: Any,
        decoder_optimizer: Any,
        model_utils: ModelUtils,
        rollout_len: int = 30,
):
    """Train the lstm network.

    Args:
        train_loader: DataLoader for the train set
        round: round number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        rollout_len: current prediction horizon

    """
    loss_list = []
    total_iter = len(train_loader)
    start = time.perf_counter()
    for i, (_input, target, helpers) in enumerate(train_loader):
        # show progress bar
        progress_bar_len = 50
        c = (i / total_iter) * 100
        a = "*" * int(i / total_iter * progress_bar_len)
        b = "." * (progress_bar_len-int(i / total_iter * progress_bar_len))
        dur = time.perf_counter() - start
        print("\rTraining Round {}: {}/{} {:^3.0f}%[{}->{}]{:.2f}s".format(
            round, i+1, total_iter, c, a, b, dur), end="")

        _input = _input.to(device)
        target = target.to(device)

        # Set to train mode
        encoder.train()
        decoder.train()

        # Zero the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        output_length = target.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size, encoder.hidden_size, device)

        # Initialize losses
        loss = 0

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(target.shape).to(device)

        # Decode hidden state in future trajectory
        for di in range(rollout_len):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Update loss
            loss += criterion(decoder_output[:, :2], target[:, di, :2])

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get average loss for pred_len
        loss = loss / rollout_len
        loss_list.append(loss)
        # Backpropagate
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    # Log results
    average_loss = torch.as_tensor(loss_list).mean()
    print(
        f" -- average loss:{average_loss:.5f}")

    logger.scalar_summary(tag="Train/loss",
                          value=torch.as_tensor(loss_list).mean(),
                          step=round)
    return average_loss


def validate(
        args,
        device,
        val_loader: Any,
        round: int,
        criterion: Any,
        encoder: Any,
        decoder: Any,
        model_utils: ModelUtils,
        best_loss: float,
        save_address_id: str,
        rollout_len: int = 30,
) -> float:
    """Validate the lstm network.

    Args:
        val_loader: DataLoader for the train set
        round: round number
        criterion: Loss criterion
        encoder: Encoder network instance
        decoder: Decoder network instance
        model_utils: instance for ModelUtils class
        best_loss:
        save_address_id:
        rollout_len: current prediction horizon

    """
    total_loss = []

    for i, (_input, target, helpers) in enumerate(val_loader):

        _input = _input.to(device)
        target = target.to(device)

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        output_length = target.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size, encoder.hidden_size, device)

        # Initialize loss
        loss = 0
        with torch.no_grad():
            # Encode observed trajectory
            for ei in range(input_length):
                encoder_input = _input[:, ei, :]
                encoder_hidden = encoder(encoder_input, encoder_hidden)

            # Initialize decoder input with last coordinate in encoder
            decoder_input = encoder_input[:, :2]

            # Initialize decoder hidden state as encoder hidden state
            decoder_hidden = encoder_hidden

            decoder_outputs = torch.zeros(target.shape).to(device)

            # Decode hidden state in future trajectory
            for di in range(output_length):
                decoder_output, decoder_hidden = decoder(decoder_input,
                                                         decoder_hidden)
                decoder_outputs[:, di, :] = decoder_output

                # Update losses for all benchmarks
                loss += criterion(decoder_output[:, :2], target[:, di, :2])

                # Use own predictions as inputs at next step
                decoder_input = decoder_output

        # Get average loss for pred_len
        loss = loss / output_length
        total_loss.append(loss)

    # Save
    val_loss = sum(total_loss) / len(total_loss)
    cprint(
        f"Val -- Round:{round}, loss:{val_loss:.5f}",
        color="green",
    )

    if best_loss!=None and val_loss <= best_loss:
        if args.use_map:
            save_dir = "saved_models/lstm_map"
        elif args.use_social:
            save_dir = "saved_models/lstm_social"
        else:
            save_dir = "saved_models/lstm"
        save_dir = os.path.join(save_dir, save_address_id)
        os.makedirs(save_dir, exist_ok=True)
        model_utils.save_checkpoint(
            save_dir,
            {
                "round": round + 1,
                "rollout_len": rollout_len,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "best_loss": val_loss,
            },
        )

    return val_loss


def evaluate(args, device, round, data_dict, encoder, decoder, model_utils):
    temp_save_dir = tempfile.mkdtemp()
    test_data_dict = {}
    for k, v in data_dict.items():
        if k in ["test_input", "test_helpers"]:
            test_data_dict[k] = v
    infer_helper(args, device, test_data_dict, 0, encoder, decoder,
                 model_utils, temp_save_dir)

    traj_save_path = os.path.join(
        args.traj_save_path, args.save_address_id+".pth.tar")
    baseline_utils.merge_saved_traj(temp_save_dir, traj_save_path)
    shutil.rmtree(temp_save_dir)

    print(
        f"Round {round}: forecasted trajectories for the test set are saved at {traj_save_path}")

    # Evaluating stage
    with open(args.gt, "rb") as f:
        gt_trajectories: Dict[int, np.ndarray] = pkl.load(f)

    with open(traj_save_path, "rb") as f:
        forecasted_trajectories: Dict[int, List[np.ndarray]] = pkl.load(f)

    with open(args.test_features, "rb") as f:
        features_df: pd.DataFrame = pkl.load(f)

    metric_results = None
    if args.metrics:

        city_names = get_city_names_from_features(features_df)

        # Get displacement error and dac on multiple guesses along each centerline
        if not args.prune_n_guesses and args.n_cl:
            forecasted_trajectories = get_m_trajectories_along_n_cl(
                args,
                forecasted_trajectories)
            num_trajectories = args.n_cl * args.n_guesses_cl

        # Get displacement error and dac on pruned guesses
        elif args.prune_n_guesses:
            forecasted_trajectories = get_pruned_guesses(
                args,
                forecasted_trajectories, city_names, gt_trajectories)
            num_trajectories = args.prune_n_guesses

        # Normal case
        else:
            num_trajectories = args.max_n_guesses

        metric_results = compute_forecasting_metrics(
            forecasted_trajectories,
            gt_trajectories,
            city_names,
            num_trajectories,
            args.pred_len,
            args.miss_threshold,
        )

    if args.viz:
        id_for_viz = None
        if args.viz_seq_id:
            with open(args.viz_seq_id, "rb") as f:
                id_for_viz = pkl.load(f)
        viz_predictions_helper(args, forecasted_trajectories, gt_trajectories,
                               features_df, id_for_viz)

    return metric_results


def infer_absolute(
        args,
        device,
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
):
    """Infer function for non-map LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance

    """
    forecasted_trajectories = {}

    for i, (_input, target, helpers) in enumerate(test_loader):

        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size, encoder.hidden_size, device)

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(
            (batch_size, args.pred_len, 2)).to(device)

        # Decode hidden state in future trajectory
        for di in range(args.pred_len):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get absolute trajectory
        abs_helpers = {}
        abs_helpers["REFERENCE"] = np.array(helpers_dict["DELTA_REFERENCE"])
        abs_helpers["TRANSLATION"] = np.array(helpers_dict["TRANSLATION"])
        abs_helpers["ROTATION"] = np.array(helpers_dict["ROTATION"])
        abs_inputs, abs_outputs = baseline_utils.get_abs_traj(
            _input.clone().cpu().numpy(),
            decoder_outputs.detach().clone().cpu().numpy(),
            args,
            abs_helpers,
        )

        for i in range(abs_outputs.shape[0]):
            seq_id = int(helpers_dict["SEQ_PATHS"][i])
            forecasted_trajectories[seq_id] = [abs_outputs[i]]

    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)


def infer_map(
        args,
        device,
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
):
    """Infer function for map-based LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance

    """
    forecasted_trajectories = {}
    for i, (_input, target, helpers) in enumerate(test_loader):

        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]

        # Iterate over every element in the batch
        for batch_idx in range(batch_size):
            num_candidates = len(
                helpers_dict["CANDIDATE_CENTERLINES"][batch_idx])
            curr_centroids = helpers_dict["CENTROIDS"][batch_idx]
            seq_id = int(helpers_dict["SEQ_PATHS"][batch_idx])
            abs_outputs = []

            # Predict using every centerline candidate for the current trajectory
            for candidate_idx in range(num_candidates):
                curr_centerline = helpers_dict["CANDIDATE_CENTERLINES"][
                    batch_idx][candidate_idx]
                curr_nt_dist = helpers_dict["CANDIDATE_NT_DISTANCES"][
                    batch_idx][candidate_idx]

                _input = torch.FloatTensor(
                    np.expand_dims(curr_nt_dist[:args.obs_len].astype(float),
                                   0)).to(device)

                # Initialize encoder hidden state
                encoder_hidden = model_utils.init_hidden(
                    1, encoder.hidden_size, device)

                # Encode observed trajectory
                for ei in range(input_length):
                    encoder_input = _input[:, ei, :]
                    encoder_hidden = encoder(encoder_input, encoder_hidden)

                # Initialize decoder input with last coordinate in encoder
                decoder_input = encoder_input[:, :2]

                # Initialize decoder hidden state as encoder hidden state
                decoder_hidden = encoder_hidden

                decoder_outputs = torch.zeros((1, args.pred_len, 2)).to(device)

                # Decode hidden state in future trajectory
                for di in range(args.pred_len):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    decoder_outputs[:, di, :] = decoder_output

                    # Use own predictions as inputs at next step
                    decoder_input = decoder_output

                # Get absolute trajectory
                abs_helpers = {}
                abs_helpers["REFERENCE"] = np.expand_dims(
                    np.array(helpers_dict["CANDIDATE_DELTA_REFERENCES"]
                             [batch_idx][candidate_idx]),
                    0,
                )
                abs_helpers["CENTERLINE"] = np.expand_dims(curr_centerline, 0)

                abs_input, abs_output = baseline_utils.get_abs_traj(
                    _input.clone().cpu().numpy(),
                    decoder_outputs.detach().clone().cpu().numpy(),
                    args,
                    abs_helpers,
                )

                # array of shape (1,30,2) to list of (30,2)
                abs_outputs.append(abs_output[0])
            forecasted_trajectories[seq_id] = abs_outputs

    os.makedirs(forecasted_save_dir, exist_ok=True)
    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)


def infer_helper(
        args,
        device,
        curr_data_dict: Dict[str, Any],
        start_idx: int,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        model_utils: ModelUtils,
        forecasted_save_dir: str,
):
    """Run inference on the current joblib batch.

    Args:
        curr_data_dict: Data dictionary for the current joblib batch
        start_idx: Start idx of the current joblib batch
        encoder: Encoder network instance
        decoder: Decoder network instance
        model_utils: ModelUtils instance
        forecasted_save_dir: Directory where forecasted trajectories are to be saved

    """
    curr_test_dataset = LSTMDataset(curr_data_dict, args, "test")
    curr_test_loader = torch.utils.data.DataLoader(
        curr_test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        collate_fn=model_utils.my_collate_fn,
    )

    if args.use_map:
        print(f"#### LSTM+map inference at index {start_idx} ####")
        infer_map(
            args,
            device,
            curr_test_loader,
            encoder,
            decoder,
            start_idx,
            forecasted_save_dir,
            model_utils,
        )

    else:
        print(f"#### LSTM+social inference at {start_idx} ####"
              ) if args.use_social else print(
                  f"#### LSTM inference at {start_idx} ####")
        infer_absolute(
            args,
            device,
            curr_test_loader,
            encoder,
            decoder,
            start_idx,
            forecasted_save_dir,
            model_utils,
        )


def get_city_names_from_features(features_df: pd.DataFrame) -> Dict[int, str]:
    """Get sequence id to city name mapping from the features.

    Args:
        features_df: DataFrame containing the features
    Returns:
        city_names: Dict mapping sequence id to city name

    """
    city_names = {}
    for index, row in features_df.iterrows():
        city_names[row["SEQUENCE"]] = row["FEATURES"][0][
            FEATURE_FORMAT["CITY_NAME"]]
    return city_names


def get_pruned_guesses(
        args,
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        city_names: Dict[int, str],
        gt_trajectories: Dict[int, np.ndarray],
) -> Dict[int, List[np.ndarray]]:
    """Prune the number of guesses using map.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.
        city_names: Dict mapping sequence id to city name.
        gt_trajectories: Ground Truth trajectories.

    Returns:
        Pruned number of forecasted trajectories.

    """
    avm = ArgoverseMap()

    pruned_guesses = {}

    for seq_id, trajectories in forecasted_trajectories.items():

        city_name = city_names[seq_id]
        da_points = []
        for trajectory in trajectories:
            raster_layer = avm.get_raster_layer_points_boolean(
                trajectory, city_name, "driveable_area")
            da_points.append(np.sum(raster_layer))

        sorted_idx = np.argsort(da_points)[::-1]
        pruned_guesses[seq_id] = [
            trajectories[i] for i in sorted_idx[:args.prune_n_guesses]
        ]

    return pruned_guesses


def get_m_trajectories_along_n_cl(
        args,
        forecasted_trajectories: Dict[int, List[np.ndarray]]
) -> Dict[int, List[np.ndarray]]:
    """Given forecasted trajectories, get <args.n_guesses_cl> trajectories along each of <args.n_cl> centerlines.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.

    Returns:
        <args.n_guesses_cl> trajectories along each of <args.n_cl> centerlines.

    """
    selected_trajectories = {}
    for seq_id, trajectories in forecasted_trajectories.items():
        curr_selected_trajectories = []
        max_predictions_along_cl = min(len(forecasted_trajectories[seq_id]),
                                       args.n_cl * args.max_neighbors_cl)
        for i in range(0, max_predictions_along_cl, args.max_neighbors_cl):
            for j in range(i, i + args.n_guesses_cl):
                curr_selected_trajectories.append(
                    forecasted_trajectories[seq_id][j])
        selected_trajectories[seq_id] = curr_selected_trajectories
    return selected_trajectories


def viz_predictions_helper(
        args,
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        gt_trajectories: Dict[int, np.ndarray],
        features_df: pd.DataFrame,
        viz_seq_id: Union[None, List[int]],
) -> None:
    """Visualize predictions.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.
        gt_trajectories: Ground Truth trajectories.
        features_df: DataFrame containing the features
        viz_seq_id: Sequence ids to be visualized

    """
    seq_ids = gt_trajectories.keys() if viz_seq_id is None else viz_seq_id
    for seq_id in seq_ids:
        gt_trajectory = gt_trajectories[seq_id]
        curr_features_df = features_df[features_df["SEQUENCE"] == seq_id]
        input_trajectory = (
            curr_features_df["FEATURES"].values[0]
            [:args.obs_len, [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]].astype(
                "float"))
        output_trajectories = forecasted_trajectories[seq_id]
        candidate_centerlines = curr_features_df[
            "CANDIDATE_CENTERLINES"].values[0]
        city_name = curr_features_df["FEATURES"].values[0][
            0, FEATURE_FORMAT["CITY_NAME"]]

        gt_trajectory = np.expand_dims(gt_trajectory, 0)
        input_trajectory = np.expand_dims(input_trajectory, 0)
        output_trajectories = np.expand_dims(np.array(output_trajectories), 0)
        candidate_centerlines = np.expand_dims(np.array(candidate_centerlines),
                                               0)
        city_name = np.array([city_name])

        viz_predictions(
            input_trajectory,
            output_trajectories,
            gt_trajectory,
            candidate_centerlines,
            city_name,
            idx=seq_id,
            show=True,
        )
