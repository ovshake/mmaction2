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

@DATASETS.register_module()
class EpicKitchensMMSADA(BaseDataset):

    def __init__(self,
                 domain,
                 pipeline,
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='frame_{:010d}.jpg'):

        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []

        if not isinstance(domain, list):
            domain = [domain]
        for d in domain:
            metadata_path = f"/data/{os.path.expanduser('~').split('/')[-1]}/projects/MM-SADA_Domain_Adaptation_Splits/{d.upper()}_{self.split}.pkl"
            self.metadata_paths.append(metadata_path)

        if osp.exists('/local_datasets/EPIC_KITCHENS_UDA'):
            self.datapath = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        else:
            self.datapath = '/data/dataset/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        self.domain_to_participant_map = {"P08": "D1",  "P01": "D2", "P22": "D3"}
        super(EpicKitchensMMSADA, self).__init__(ann_file=None,
                                                pipeline=pipeline,
                                                test_mode=test_mode,
                                                sample_by_class=sample_by_class)
        self.filename_tmpl = filename_tmpl



    def load_annotations(self):
        video_infos = []
        for metadata_path in self.metadata_paths:
            df = pd.read_pickle(metadata_path)
            for _, line in df.iterrows():
                participant_id = line['participant_id']
                video_id = line['video_id']
                start_frame = int(line['start_frame'])
                end_frame = int(line['stop_frame'])
                label = line['verb_class']
                frame_dir =  f"{self.datapath}/{self.split}/{self.domain_to_participant_map[participant_id]}/{video_id}"
                total_frames = end_frame - start_frame + 1
                label = int(label)

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
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pipeline(results)



@DATASETS.register_module()
class EpicKitchensSlowFastMMSADA(BaseDataset):
    def __init__(self,
                 domain,
                 slow_pipeline,
                 fast_pipeline,
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='frame_{:010d}.jpg'):

        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []

        if not isinstance(domain, list):
            domain = [domain]
        for d in domain:
            metadata_path = f"/data/shinpaul14/projects/MM-SADA_Domain_Adaptation_Splits/{d.upper()}_{self.split}.pkl"
            self.metadata_paths.append(metadata_path)

        if osp.exists('/local_datasets/EPIC_KITCHENS_UDA'):
            self.datapath = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        else:
            self.datapath = '/data/dataset/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        self.domain_to_participant_map = {"P08": "D1",  "P01": "D2", "P22": "D3"}
        super().__init__(ann_file=None, pipeline=slow_pipeline, test_mode=test_mode, sample_by_class=sample_by_class)
        self.filename_tmpl = filename_tmpl
        self.slow_pipeline = Compose(slow_pipeline)
        self.fast_pipeline = Compose(fast_pipeline)


    def load_annotations(self):
        video_infos = []
        for metadata_path in self.metadata_paths:
            df = pd.read_pickle(metadata_path)
            for _, line in df.iterrows():
                participant_id = line['participant_id']
                video_id = line['video_id']
                start_frame = int(line['start_frame'])
                end_frame = int(line['stop_frame'])
                label = line['verb_class']
                frame_dir =  f"{self.datapath}/{self.split}/{self.domain_to_participant_map[participant_id]}/{video_id}"
                total_frames = end_frame - start_frame + 1
                label = int(label)

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
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.slow_pipeline(results), self.fast_pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.slow_pipeline(results)


@DATASETS.register_module()
class EpicKitchensTemporalSpatialMMSADA(BaseDataset):
    def __init__(self,
                 domain,
                 pathway_A,
                 pathway_B,
                 clip_len,
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='frame_{:010d}.jpg'):

        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []


        if not isinstance(domain, list):
            domain = [domain]
        for d in domain:
            metadata_path = f"/data/{os.path.expanduser('~').split('/')[-1]}/projects/MM-SADA_Domain_Adaptation_Splits/{d.upper()}_{self.split}.pkl"
            self.metadata_paths.append(metadata_path)

        if osp.exists('/local_datasets/EPIC_KITCHENS_UDA') and False:
            self.datapath = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        else:
            self.datapath = '/data/dataset/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        self.domain_to_participant_map = {"P08": "D1",  "P01": "D2", "P22": "D3"}
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
                participant_id = line['participant_id']
                video_id = line['video_id']
                start_frame = int(line['start_frame'])

                end_frame = int(line['stop_frame'])

                label = line['verb_class']
                frame_dir =  f"{self.datapath}/{self.split}/{self.domain_to_participant_map[participant_id]}/{video_id}"
                total_frames = end_frame - start_frame + 1

                label = int(label)
                #print('start', start_frame, 'end', end_frame)
                if total_frames < 29:
                    print(video_id)
                    print(total_frames)
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
        pathway_A_start_index = np.random.randint(start_index, max(end_index - num_frames, start_index + 1))
        pathway_B_start_index=pathway_A_start_index
        # if self.not_same_start_frame:

        #     pathway_B_start_index = np.random.randint(start_index, max(end_index - num_frames, start_index + 1))
        # else:
        #     pathway_B_start_index = pathway_A_start_index
        # print('pathway_A_start_index', pathway_A_start_index)
        if pathway_A_start_index + 28 > end_index:
            pathway_A_start_index = end_index - 31
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
class EpicKitchensTemporalSpatialMMSADA_not_same_start(BaseDataset):
    def __init__(self,
                 domain,
                 pathway_A,
                 pathway_B,
                 clip_len,
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='frame_{:010d}.jpg'):

        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []


        if not isinstance(domain, list):
            domain = [domain]
        for d in domain:
            metadata_path = f"/data/{os.path.expanduser('~').split('/')[-1]}/projects/MM-SADA_Domain_Adaptation_Splits/{d.upper()}_{self.split}.pkl"
            self.metadata_paths.append(metadata_path)

        if osp.exists('/local_datasets/EPIC_KITCHENS_UDA') and False:
            self.datapath = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        else:
            self.datapath = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        self.domain_to_participant_map = {"P08": "D1",  "P01": "D2", "P22": "D3"}
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
                participant_id = line['participant_id']
                video_id = line['video_id']
                start_frame = int(line['start_frame'])

                end_frame = int(line['stop_frame'])

                label = line['verb_class']
                frame_dir =  f"{self.datapath}/{self.split}/{self.domain_to_participant_map[participant_id]}/{video_id}"
                total_frames = end_frame - start_frame + 1

                label = int(label)
                #print('start', start_frame, 'end', end_frame)
                if total_frames < 29:
                    print(video_id)
                    print(total_frames)
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
        pathway_A_start_index = np.random.randint(start_index, max(end_index - num_frames, start_index + 1))
        # pathway_B_start_index=pathway_A_start_index


        pathway_B_start_index = np.random.randint(start_index, max(end_index - num_frames, start_index + 1))
        # else:
        #     pathway_B_start_index = pathway_A_start_index
        # print('pathway_A_start_index', pathway_A_start_index)
        # if pathway_A_start_index + 28 > end_index:
        #     pathway_A_start_index = end_index - 31
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
#--------------------------------------------------------------

@DATASETS.register_module()
class EpicKitchensTemporalSpatialMMSADA_val_temp(BaseDataset):
    '''Epic Kitchens dataset D1 val dataset for train and test to check our loss fuctions
    if it can overfit or not on a small dataset 
     '''
    def __init__(self,
                 domain,
                 pathway_A,
                 pathway_B,
                 clip_len,
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='frame_{:010d}.jpg'):

        self.split = 'test' if not test_mode else 'test' # changed it to test not train
        self.test_mode = test_mode
        self.metadata_paths = []


        if not isinstance(domain, list):
            domain = [domain]
        for d in domain:
            metadata_path = f"/data/shinpaul14/projects/MM-SADA_Domain_Adaptation_Splits/{d.upper()}_{self.split}.pkl"
            self.metadata_paths.append(metadata_path)

        if osp.exists('/local_datasets/EPIC_KITCHENS_UDA') and False:
            self.datapath = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        else:
            self.datapath = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        self.domain_to_participant_map = {"P08": "D1",  "P01": "D2", "P22": "D3"}
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
                participant_id = line['participant_id']
                video_id = line['video_id']
                start_frame = int(line['start_frame'])
         
                end_frame = int(line['stop_frame'])
             
                label = line['verb_class']
                frame_dir =  f"{self.datapath}/{self.split}/{self.domain_to_participant_map[participant_id]}/{video_id}"
                total_frames = end_frame - start_frame + 1
                
                label = int(label)
                #print('start', start_frame, 'end', end_frame)
                if total_frames < 29:
                    print(video_id)
                    print(total_frames)
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
        pathway_A_start_index = np.random.randint(start_index, max(end_index - num_frames, start_index + 1))
        pathway_B_start_index=pathway_A_start_index
        # if self.not_same_start_frame:

        #     pathway_B_start_index = np.random.randint(start_index, max(end_index - num_frames, start_index + 1))
        # else:
        #     pathway_B_start_index = pathway_A_start_index
        # print('pathway_A_start_index', pathway_A_start_index)
        if pathway_A_start_index + 28 > end_index:
            pathway_A_start_index = end_index - 31
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


#--------------------------------------------------------------

@DATASETS.register_module()
class EpicKitchensMultipleContrastiveSpaces(BaseDataset):
    def __init__(self,
                 domain,
                 pipelines,
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='frame_{:010d}.jpg'):

        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []

        if not isinstance(domain, list):
            domain = [domain]
        for d in domain:
            metadata_path = f"/data/shinpaul14/projects/MM-SADA_Domain_Adaptation_Splits/{d.upper()}_{self.split}.pkl"
            self.metadata_paths.append(metadata_path)

        if osp.exists('/local_datasets/EPIC_KITCHENS_UDA'):
            self.datapath = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        else:
            self.datapath = '/data/dataset/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        self.domain_to_participant_map = {"P08": "D1",  "P01": "D2", "P22": "D3"}

        msg = 'Atleast one pathway is necessary'
        assert len(pipelines) > 0, msg
        super().__init__(ann_file=None, pipeline=pipelines[0], test_mode=test_mode, sample_by_class=sample_by_class)
        self.filename_tmpl = filename_tmpl
        self.pipelines = []
        for pipeline in pipelines:
            self.pipelines.append(Compose(pipeline))



    def load_annotations(self):
        video_infos = []
        for metadata_path in self.metadata_paths:
            df = pd.read_pickle(metadata_path)
            for _, line in df.iterrows():
                participant_id = line['participant_id']
                video_id = line['video_id']
                start_frame = int(line['start_frame'])
                end_frame = int(line['stop_frame'])
                label = line['verb_class']
                frame_dir =  f"{self.datapath}/{self.split}/{self.domain_to_participant_map[participant_id]}/{video_id}"
                total_frames = end_frame - start_frame + 1
                label = int(label)

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
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        pathways = []
        for pipeline in self.pipelines:
            pathways.append(pipeline(results))
        return pathways


    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        pathways = []
        for pipeline in self.pipelines:
            pathways.append(pipeline(results))
        return pathways