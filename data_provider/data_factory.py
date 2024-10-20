from data_provider.data_loader import (
    DependentLoader,
    ADSZIndependentLoader,
    APAVAIndependentLoader,
    ADFDIndependentLoader,
    CNBPMIndependentLoader,
    COGERPIndependentLoader,
    COGrsEEGIndependentLoader,
    ADFDBinaryIndependentLoader,
    CNBPMBinaryIndependentLoader,
    ADFD7ChannelsIndependentLoader,
    CNBPM7ChannelsIndependentLoader,
    ADFDLeaveSubjectsOutLoader,
    CNBPMLeaveSubjectsOutLoader,
)
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    # subject-dependent
    'ADSZDep': DependentLoader,  # ADSZ
    'APAVADep': DependentLoader,  # APAVA
    'ADFDDep': DependentLoader,  # ADFD
    'CNBPMDep': DependentLoader,  # CNBPM
    'COGERPDep': DependentLoader,  # COGERP
    'COGrsEEGDep': DependentLoader,  # COGrsEEG
    # subject-independent
    'ADSZIndep': ADSZIndependentLoader,  # ADSZ
    'APAVAIndep': APAVAIndependentLoader,  # APAVA
    'ADFDIndep': ADFDIndependentLoader,  # ADFD
    'CNBPMIndep': CNBPMIndependentLoader,  # CNBPM
    'COGERPIndep': COGERPIndependentLoader,  # COGERP
    'COGrsEEGIndep': COGrsEEGIndependentLoader,  # COGrsEEG
    # subject-independent, only use 2 classes
    'ADFDBinaryIndep': ADFDBinaryIndependentLoader,  # ADFD
    'CNBPMBinaryIndep': CNBPMBinaryIndependentLoader,  # CNBPM
    # subject-independent, only use 2 classes and 7 channels
    'ADFD7CIndep': ADFD7ChannelsIndependentLoader,  # ADFD
    'CNBPM7CIndep': CNBPM7ChannelsIndependentLoader,  # CNBPM
    # Leave-Subjects-Out, only use 2 classes
    'ADFDLSO': ADFDLeaveSubjectsOutLoader,  # ADFD,  72 : 8 : 8 (train : valid : test)
    'CNBPMLSO': CNBPMLeaveSubjectsOutLoader,  # CNBPM, 149 : 20 : 20 (train : valid : test)
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' \
                or args.task_name == 'classification'\
                or args.task_name == 'classification_contrastive':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification' or args.task_name == 'classification_contrastive':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            args=args,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)  # only called when yeilding batches
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
