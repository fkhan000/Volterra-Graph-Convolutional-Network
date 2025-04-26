import mne
import numpy as np
import os
from mne_connectivity import spectral_connectivity_epochs
import json
from tqdm import tqdm
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    """
    Dataset class for extracting EEG connectivity graphs from OpenMIIR patient data.

    This class processes raw EEG recordings into coherence-based connectivity matrices
    segmented into fixed-length epochs. It can either generate and save the processed
    data or load previously saved graphs and labels.
    """
    def __init__(self,
                 patient_ids: list[float],
                 split: str,
                 generate: bool = True,
                 epoch_duration: float=3.0,
                 fmin: float=8,
                 fmax: float=12):
        """
        Initializes the EEGDataset.

        Args:
            patient_ids (list of float): List of patient IDs to load.
            split (str): Dataset split name ("train" or "test") for file naming.
            generate (bool): Whether to generate data from raw EEG or load preprocessed files.
            epoch_duration (float): Length (in seconds) of each epoch (each trial was split into chunks of length epoch duration).
            fmin (float): Minimum frequency (Hz) for spectral connectivity computation (most relevant eeg data falls into a relatively small frequency band usually 8-12 Hz)
            fmax (float): Maximum frequency (Hz) for spectral connectivity computation 
        """
        
        # load in events.json that contains information on each song like name and duration
        with open(os.path.join("data","OpenMIIR", "events.json"), "r") as f:
            self.song_data = json.load(f)
        
        self.coherence_matrices = []
        self.labels = []
        self.patient_ids = patient_ids
        self.split = split
        
        # if we need to generate the coherence matrices for the first time
        if generate:
            # call on load_graph_data
            self.load_graph_data(epoch_duration, fmin, fmax)
        else:
            # else we just load them in from the data directory
            dir_path = os.path.join("data", "OpenMIIR")
            self.coherence_matrices = np.load(os.path.join(dir_path, f"{split}_trial_graphs.npy"))
            with open(os.path.join(dir_path, f"{split}_labels.csv"), "r") as f:
                self.labels = f.readlines()
    
    def process_patient(self, filename: str, epoch_duration, fmin: float, fmax: float, include_cue=False):
        """
        Process EEG data for a single patient and extract connectivity graphs.

        Args:
            filename (str): Path to the raw FIF file for the patient.
            epoch_duration (float): Duration (in seconds) of each chunked epoch.
            fmin (float): Minimum frequency for spectral connectivity computation.
            fmax (float): Maximum frequency for spectral connectivity computation.
            include_cue (bool): Whether to include cue durations for perception and cued imagination trials.
        
        Notes:
            - Skips system events (event IDs >= 1000).
            - Extracts 3-second fixed-length epochs from each stimulus trial.
            - Computes coherence-based spectral connectivity graphs for each epoch.
            - Stores resulting matrices and associated song labels.
        """
        # read in the fif file using the standard 10-20 sytem that was used for measuring the EEG data
        raw = mne.io.read_raw_fif(filename, preload=True)
        channels_to_exclude = ['EXG5', 'EXG6']
        # some patients (patients 9 through 14 I think) had two additional electrodes (EXG5 and EXG6)
        # which caused an error when reading them in so those channels were dropped
        try:
            raw.drop_channels(channels_to_exclude)
        except:
            pass
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        # we then locate the events, each fif file contains all of the trials that were conducted for the patient
        # and all of the events with an event id less than 1000 correspond to trials
        events = mne.find_events(raw, stim_channel='STI 014', shortest_event=1)

        # for each event
        for event in tqdm(events):
            event_sample, _, event_id = event
            # if the event id's greater than 1000 this isn't a trial and we can skip
            if event_id >= 1000:
                continue
            # each event id follows the format {song_id}{condition_id} where condition id is a single digit number
            # and song id can be either one or two digits
            # we extract the song and condition id
            song_id = str(event_id // 10)
            condition_id = event_id % 10

            # and get the relevant information
            song_name = self.song_data[song_id]["name"]
            # if there's a cue then that affects the duration of the event
            # by default the cue was excluded
            if condition_id in [1,2] and include_cue:
                duration = self.song_data[song_id]["duration_cue"]
            else:
                duration = self.song_data[song_id]["duration_no_cue"]
            
            # we get the start and end time
            start_time = event_sample / raw.info["sfreq"]
            tmin = start_time
            tmax = start_time + duration
            
            # and get the snippet corresponding to the trial
            trial = raw.copy().crop(tmin=tmin, tmax=tmax)
            # we then split the event into several chunks
            events_fixed = mne.make_fixed_length_events(trial,
                                                        duration=epoch_duration)
            
            epochs = mne.Epochs(trial,
                                events_fixed,
                                tmin=0,
                                tmax=epoch_duration,
                                baseline=None,
                                preload=True,
                                verbose=False)
            
            if len(epochs) == 0:
                continue
            # we calculate the spectral connectivity matrix between each of the nodes for each chunk
            # and filter for frequencies between 8-12 Hz since most relevant information from EEG data falls into this band
            spec_con = spectral_connectivity_epochs(
                epochs, method='coh', sfreq=raw.info['sfreq'],
                fmin=fmin, fmax=fmax, faverage=True, verbose=False
                )
            
            coherence_matrices = spec_con.get_data(output='dense')
            
            # for each chunk
            for chunk_idx in range(coherence_matrices.shape[2]):
                # we add in the associated coherence matrix
                coherence_matrix = coherence_matrices[:, :, chunk_idx]
                self.coherence_matrices.append(coherence_matrix)
            # as well as the label
            self.labels += [song_name]*coherence_matrices.shape[2]
    
    def load_graph_data(self, epoch_duration: float, fmin: float, fmax: float):
        """
        Load the training and testing EEG datasets.

        Returns:
            tuple: (train_dataset, test_dataset)
                - train_dataset (EEGDataset): EEGDataset instance for training data.
                - test_dataset (EEGDataset): EEGDataset instance for testing data.
        """
        
        data_dir = os.path.join("data", "OpenMIIR", "OpenMIIR-RawEEG_v1")
        for patient_id in self.patient_ids:
            print(f"Processing Patient {patient_id}...")
            filename = os.path.join(data_dir, f"P{patient_id}-raw.fif")
            self.process_patient(filename, epoch_duration, fmin, fmax)

        self.coherence_matrices = np.asarray(self.coherence_matrices)
        np.save(os.path.join("data", "OpenMIIR", f"{self.split}_trial_graphs.npy"),
                self.coherence_matrices)
        
        with open(os.path.join("data", "OpenMIIR", f"{self.split}_labels.csv"), "w+") as f:
            f.write("Song Name\n")
            for label in self.labels:
                f.write(f"{label}\n")
    
    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (coherence_matrix, song_label)
                - coherence_matrix (np.ndarray): The connectivity graph.
                - song_label (str): Name of the associated song.
        """
        return self.coherence_matrices[index], self.labels[index]
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

def load_data():
    
    train_patient_ids = ["01", "04", "05", "06", "07", "09", "11"]
    test_patient_ids = ["12", "13", "14"]
    train_eeg_dataset = EEGDataset(train_patient_ids, "train", generate=True)
    test_eeg_dataset = EEGDataset(test_patient_ids, "test", generate=True)
    
    return train_eeg_dataset, test_eeg_dataset

if __name__ == "__main__":
    load_data()