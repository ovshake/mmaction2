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
class EpicKitchensMultiContrastiveNonBinaryAugSpaces(BaseDataset):
    def __init__(self,
                 domain,
                 pipeline,
                 test_mode=False,
                 sample_by_class=False,
                 filename_tmpl='frame_{:010d}.jpg', 
                 valid_contrastive_augmentations=['RandomSampleColorJitter', 
                                            'RandomFrequencySampleFrames']):
        
        self.split = 'train' if not test_mode else 'test'
        self.test_mode = test_mode
        self.metadata_paths = []
        
        if not isinstance(domain, list):
            domain = [domain] 
        for d in domain:
            metadata_path = f"/data/shinpaul14/projects/MM-SADA_Domain_Adaptation_Splits/{d.upper()}_{self.split}.pkl"
            self.metadata_paths.append(metadata_path) 
        
        self.datapath = '/local_datasets/EPIC_KITCHENS_UDA/frames_rgb_flow/rgb'
        self.domain_to_participant_map = {"P08": "D1",  "P01": "D2", "P22": "D3"}

        super().__init__(ann_file=None, pipeline=pipeline, test_mode=test_mode, sample_by_class=sample_by_class)
        self.filename_tmpl = filename_tmpl
        self.pipeline = Compose(pipeline)
        self.valid_contrastive_augmentations = valid_contrastive_augmentations
        self.contrastive_augmentations = []
        for aug in self.pipeline.transforms:
            if type(aug).__name__ in self.valid_contrastive_augmentations:
                self.contrastive_augmentations.append(aug) 
        


    
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
    
    def sample_params(self, augmentation_type):
        params = None
        if augmentation_type == 'RandomSampleColorJitter':
            brightness = np.random.uniform(low=0., high=0.5) 
            contrast = np.random.uniform(low=0., high=0.5)
            saturation = np.random.uniform(low=0., high=0.5) 
            hue = np.random.uniform(low=0., high=0.2) 
            params = {'brightness': brightness, 
                    'contrast': contrast, 
                    'saturation': saturation, 
                    'hue': hue} 

        elif augmentation_type == 'RandomFrequencySampleFrames':
            sample_rate_choices = [1, 2, 3, 4] 
            params = {'frame_interval': np.random.choice(sample_rate_choices)} 
        
        assert params is not None, 'params cannot be returned None'
        return params 

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality

        q_params = {} 
        k_0_params = {} 
        for aug in self.contrastive_augmentations:
            q_params[type(aug).__name__] = self.sample_params(type(aug).__name__)
            k_0_params[type(aug).__name__] = self.sample_params(type(aug).__name__)

        
        k_params = [] 
        for k_idx in range(len(self.contrastive_augmentations)):
            invariant_aug = self.contrastive_augmentations[k_idx] 
            k_param = {type(invariant_aug).__name__: copy.deepcopy(q_params[type(invariant_aug).__name__])}
            for rk_idx in range(len(self.contrastive_augmentations)):
                if k_idx != rk_idx:
                    variant_aug = self.contrastive_augmentations[rk_idx] 
                    k_param[type(variant_aug).__name__] = self.sample_params(type(variant_aug).__name__) 
                
            k_params.append(k_param) 

        
        q_results = self.process_data(results, q_params)
        k_0_results = self.process_data(results, k_0_params)
        k_results = [] 
        for idx in range(len(self.contrastive_augmentations)):
            k_results.append(self.process_data(results, k_params[idx])) 
        
        results = [k_0_results] + k_results + [q_results] 
        return results 


    
    def process_data(self, result, contrastive_params):
        for aug in self.pipeline.transforms:
            aug_name = type(aug).__name__
            if aug_name in self.valid_contrastive_augmentations:
                result = aug(result, **contrastive_params[aug_name]) 
            else:
                result = aug(result) 
        
        return result 


            
        


        
if __name__ == '__main__':
    contrastive_pipeline = contrastive_pipeline = [
        dict(type='RandomFrequencySampleFrames', clip_len=8, frame_interval=1, num_clips=1),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='RandomCrop', size=224),
        dict(type='RandomSampleColorJitter'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]

    dataset = EpicKitchensMultiContrastiveNonBinaryAugSpaces(
        domain='D1',
        pipeline=contrastive_pipeline,
        test_mode=False,
        sample_by_class=True
    )
    import ipdb; ipdb.set_trace() 
        