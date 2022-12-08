clear;
EEG = pop_loadset('filename','sample_eeg_data.set','filepath','');

EEG = pop_eegfiltnew(EEG, 'locutoff',1,'hicutoff',45,'plotfreqz',0);

EEG = SPA_EEG(EEG,100,2,2);

EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'stop',0.1,'interrupt','off');
EEG = pop_iclabel(EEG, 'default');%identify artifacts

%remove ocular artifacts
eye_i = find(ismember(EEG.etc.ic_classification.ICLabel.classes,'Eye'));
eye_ic = find(EEG.etc.ic_classification.ICLabel.classifications(:,eye_i)>.5);
EEG = pop_subcomp(EEG,eye_ic,0);


