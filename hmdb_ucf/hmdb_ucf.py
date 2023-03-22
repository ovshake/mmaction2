import copy
import os.path as osp

import mmcv
import os

from .base import BaseDataset
from .builder import DATASETS
import numpy as np
import os.path as osp
from .pipelines import Compose
import pandas as pd

#---------------------


#-------------------


@DATASETS.register_module()
class UCF_toHMDB_two_pathway(BaseDataset):
    def __init__(self,
                 domain,
                 pathway_A,
                 pathway_B,
                 clip_len,
                 test_mode=False,
                 sample_by_class=False,
                filename_tmpl='img_{:05}.jpg'):

        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []


        if not isinstance(domain, list):
            domain = [domain]
        for d in domain:
            metadata_path = f"/data/shinpaul14/projects/mmaction2/data_list/{self.split}_{d}.pkl"
            self.metadata_paths.append(metadata_path)

        # print( self.metadata_paths)
            self.datapath = f'/local_datasets/{d}/rawframes'
     
        super().__init__(ann_file=None, pipeline=pathway_A, test_mode=test_mode, sample_by_class=sample_by_class)
        self.filename_tmpl = filename_tmpl
        self.pathway_A = Compose(pathway_A)
        self.pathway_B = Compose(pathway_B)
        self.clip_len = clip_len


    def load_annotations(self):
        video_infos = []
        for metadata_path in self.metadata_paths:
            df = pd.read_pickle(metadata_path)
            for _, line in df.iterrows():
                start_frame = int(line['start_frame'])
                end_frame = int(line['stop_frame'])
                label = line['verb_class']
                frame_dir =  line['frame_dir']
                total_frames = end_frame - start_frame 

                label = int(label)
                #print('start', start_frame, 'end', end_frame)
                if total_frames < 29:
                    print(frame_dir)
                    # print(total_frames)
                    pass
                else:
                    video_infos.append(
                    dict(
                        frame_dir=frame_dir,
                        total_frames=total_frames,
                        label=label,
                        start_index=start_frame,
                        end_index=end_frame
                    )
                )

        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        # print(results.keys())
        #dict_keys(['frame_dir', 'total_frames', 'label', 'start_index', 'end_index'])
        start_index = results['start_index']
        end_index = results['end_index']
        num_frames = self.clip_len
        #print("num_frames", num_frames)
        # print('start_index', start_index, 'end_index', end_index)
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        pathway_A_start_index = np.random.randint(1, max(end_index - num_frames, 1 + 1))
        pathway_B_start_index=pathway_A_start_index
   
        if pathway_A_start_index + 28 > end_index:
            pathway_A_start_index = end_index - 31
            if pathway_A_start_index == 0:
                pathway_A_start_index = 1
        pathway_A_results = copy.deepcopy(results)
        pathway_B_results = copy.deepcopy(results)
        pathway_A_results['start_index'] = pathway_A_start_index
        pathway_A_results['total_frames'] = self.clip_len
        pathway_B_results['start_index'] = pathway_B_start_index
        pathway_B_results['total_frames'] = self.clip_len
       # how can we implement speed augmentation
       #both pathways has to have the same start frame or idx

        return self.pathway_A(pathway_A_results), self.pathway_B(pathway_B_results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pathway_A(results)



@DATASETS.register_module()
class UCF_to_HMDB(BaseDataset):

    def __init__(self,
                 domain,
                 pipeline,
                #  clip_len,
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='img_{:05}.jpg'):

        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []


        if not isinstance(domain, list):
            domain = [domain]
        for d in domain:
            metadata_path = f"/data/shinpaul14/projects/mmaction2/data_list/{self.split}_{d}.pkl"
            self.metadata_paths.append(metadata_path)
            self.datapath = f'/local_datasets/{d}/rawframes'
        super(UCF_to_HMDB, self).__init__(ann_file=None,
                                                pipeline=pipeline,
                                                test_mode=test_mode,
                                                sample_by_class=sample_by_class)
        self.filename_tmpl = filename_tmpl



    def load_annotations(self):
        video_infos = []
        for metadata_path in self.metadata_paths:
            df = pd.read_pickle(metadata_path)
            missing_frames = []
            for _, line in df.iterrows():
                if int(line['start_frame']) == 1:
                    start_frame = int(line['start_frame'])
                else:
                    start_frame = 1
                end_frame = int(line['stop_frame'])
                label = line['verb_class']
                frame_dir =  line['frame_dir']
                total_frames = end_frame - start_frame 

                label = int(label)
                #print('start', start_frame, 'end', end_frame)
                if total_frames < 8:
                    missing_frames.append(frame_dir)
                    print(frame_dir)
                    # print(total_frames)
                    pass
                else:
                    video_infos.append(
                    dict(
                        frame_dir=frame_dir,
                        total_frames=total_frames,
                        label=label,
                        start_index=start_frame,
                        end_index=end_frame
                    )
                )
        print(len(missing_frames))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pipeline(results)

#----------------------------------------------


@DATASETS.register_module()
class UCF_to_HMDB_ensemble(BaseDataset):

    def __init__(self,
                 domain,
                 pathway_A,
                 clip_len,
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='img_{:05}.jpg'):

        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []


        if not isinstance(domain, list):
            domain = [domain]
        for d in domain:
            metadata_path = f"/data/shinpaul14/projects/mmaction2/data_list/{self.split}_{d}.pkl"
            self.metadata_paths.append(metadata_path)
            self.datapath = f'/local_datasets/{d}/rawframes'
        super().__init__(ann_file=None,
                                                pipeline=pathway_A,
                                                test_mode=test_mode,
                                                sample_by_class=sample_by_class)
        self.filename_tmpl = filename_tmpl
        self.pathway_A = Compose(pathway_A)



    def load_annotations(self):
        video_infos = []
        for metadata_path in self.metadata_paths:
            df = pd.read_pickle(metadata_path)
            missing_frames = []
            for _, line in df.iterrows():
                if int(line['start_frame']) == 1:
                    start_frame = int(line['start_frame'])
                else:
                    start_frame = 1
                end_frame = int(line['stop_frame'])
                label = line['verb_class']
                frame_dir =  line['frame_dir']
                total_frames = end_frame - start_frame 

                label = int(label)
                #print('start', start_frame, 'end', end_frame)
                if total_frames < 8:
                    missing_frames.append(frame_dir)
                    print(frame_dir)
                    # print(total_frames)
                    pass
                else:
                    video_infos.append(
                    dict(
                        frame_dir=frame_dir,
                        total_frames=total_frames,
                        label=label,
                        start_index=start_frame,
                        end_index=end_frame
                    )
                )
        print(len(missing_frames))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pathway_A(results), self.pathway_A(results), self.pathway_A(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pathway_A(results)

#----------------------------------------------