import copy
import os.path as osp

import mmcv

from .base import BaseDataset
from .builder import DATASETS
import numpy as np 
import os.path as osp 
from .pipelines import Compose
import pandas as pd 

@DATASETS.register_module()
class Kinetics400UCFHMDB(BaseDataset):

    def __init__(self,
                 domain,
                 pipeline, 
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='img_{:05d}.jpg'):
        
        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []
        
        if not isinstance(domain, list):
            domain = [domain] 
        for d in domain:
            metadata_path = f"/data/adithya/data/{d}_{self.split}.csv"
            self.metadata_paths.append(metadata_path) 
        
        
        super(Kinetics400UCFHMDB, self).__init__(ann_file=None, 
                                                pipeline=pipeline, 
                                                test_mode=test_mode, 
                                                sample_by_class=sample_by_class)
        self.filename_tmpl = filename_tmpl

    

    def load_annotations(self):
        video_infos = [] 
        for metadata_path in self.metadata_paths:
            df = pd.read_csv(metadata_path) 
            for _, line in df.iterrows():
                frame_dir = line['path']                  
                start_frame = int(line['start_frame']) 
                end_frame = int(line['stop_frame'])
                label = line['class']
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
class Kinetics400UCFHMDBTwoPathway(BaseDataset):
    def __init__(self,
                 domain,
                 pathway_A_pipeline,
                 pathway_B_pipeline, 
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='img_{:05d}.jpg'):
        
        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []
        
        if not isinstance(domain, list):
            domain = [domain] 
        for d in domain:
            metadata_path = f"/data/adithya/data/{d}_{self.split}.csv"
            self.metadata_paths.append(metadata_path) 
        
        super().__init__(ann_file=None, pipeline=pathway_A_pipeline, test_mode=test_mode, sample_by_class=sample_by_class)
        self.filename_tmpl = filename_tmpl
        self.pathway_A_pipeline = Compose(pathway_A_pipeline) 
        self.pathway_B_pipeline = Compose(pathway_B_pipeline)


    def load_annotations(self):
        video_infos = [] 
        for metadata_path in self.metadata_paths:
            df = pd.read_csv(metadata_path) 
            for _, line in df.iterrows():
                frame_dir = line['path']                  
                start_frame = int(line['start_frame']) 
                end_frame = int(line['stop_frame'])
                label = line['class']
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
        return self.pathway_A_pipeline(results), self.pathway_B_pipeline(results) 

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pathway_A_pipeline(results)
    