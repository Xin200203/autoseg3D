# Copied from mmdet3d/models/data_preprocessors/data_preprocessor.py
from mmdet3d.models.data_preprocessors.data_preprocessor import \
    Det3DDataPreprocessor
from mmdet3d.registry import MODELS


@MODELS.register_module()
class Det3DDataPreprocessor_(Det3DDataPreprocessor):
    """
    Custom preprocessor for AutoSeg3D.

    In addition to MinkowskiEngine inputs, we keep several python-object fields
    required by online 2D-3D fusion modules (GDINO point-fusion / DACA-2D):
    - points_raw: raw points captured before 3D aug (projection space)
    - img_paths / poses / cam_info: per-frame camera metadata (list structures)
    """
    def simple_process(self, data, training=False):
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()
        batch_size = len(data_samples) if data_samples is not None else 0

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict

        if 'elastic_coords' in inputs:
            batch_inputs['elastic_coords'] = inputs['elastic_coords']
        if 'points_raw' in inputs:
            batch_inputs['points_raw'] = inputs['points_raw']

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if hasattr(self, 'boxtype2tensor') and self.boxtype2tensor:
                    from mmdet.models.utils.misc import \
                        samplelist_boxtype2tensor
                    samplelist_boxtype2tensor(data_samples)
                elif hasattr(self, 'boxlist2tensor') and self.boxlist2tensor:
                    from mmdet.models.utils.misc import \
                        samplelist_boxlist2tensor
                    samplelist_boxlist2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)

                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs
        
        def _maybe_time_major_to_batch_first(v):
            if not isinstance(v, list) or batch_size <= 0:
                return v
            if len(v) == 0:
                return v
            # time-major: T x B
            if isinstance(v[0], list) and len(v[0]) == batch_size:
                T = len(v)
                return [[v[t][b] for t in range(T)] for b in range(batch_size)]
            return v

        # Keep python-object metadata required by online 2D-3D fusion modules.
        # Note: default_collate transposes list fields; convert back to batch-first.
        if 'img_paths' in inputs:
            batch_inputs['img_paths'] = _maybe_time_major_to_batch_first(inputs['img_paths'])
        if 'poses' in inputs:
            batch_inputs['poses'] = _maybe_time_major_to_batch_first(inputs['poses'])
        if 'cam_info' in inputs:
            batch_inputs['cam_info'] = _maybe_time_major_to_batch_first(inputs['cam_info'])
        
        if 'img_path' in inputs:
            batch_inputs['img_path'] = inputs['img_path']
        return {'inputs': batch_inputs, 'data_samples': data_samples}
