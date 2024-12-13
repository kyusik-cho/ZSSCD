from builtins import breakpoint

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.object_utils import convert_json_dict_to_objects_info
from deva.inference.frame_utils import FrameInfo
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import get_model_and_config

def mask_track_clip(meta_loader, cross, clip_len=60, detection_every = 5, num_voting_frames = 3, args=None):

    network, config, args = get_model_and_config(args)
    l = meta_loader.dataset.preprocess_sequence_frames(clip_len=clip_len) 
    pbar = tqdm(meta_loader.dataset.get_datasets_clip(cross=cross, 
                                                # multitemp_offset=0,
                                                ), 
                                                total=l)
                                                        
    for vid_reader in pbar:
        network.pixel_encoder.init_styles() # AdaIN style initialization
        
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=1)  
        curr_num_voting_frames = min(num_voting_frames, len(vid_reader) // 2 )   

        vid_name = vid_reader.vid_name
        pbar.set_description(vid_name)
        vid_length = len(loader)
        next_voting_frame = curr_num_voting_frames - 1
        # no need to count usage for LT if the video is not that long anyway
        config['enable_long_term_count_usage'] = (
            config['enable_long_term']
            and (vid_length / (config['max_mid_term_frames'] - config['min_mid_term_frames']) *
                config['num_prototypes']) >= config['max_long_term_elements'])

        try:
            processor = DEVAInferenceCore(network, config=config)
            processor.enabled_long_id()
            result_saver = ResultSaver(meta_loader.dataset.masktrack_dir,
                                    vid_name,
                                    # dataset=dataset_name,
                                    palette=vid_reader.palette,
                                    object_manager=processor.object_manager)

            source_loader_iter = enumerate(loader)
            cross_buffer = []
            for ti in range(len(loader) // 2):
                _, track_data = source_loader_iter.__next__()
                _, cross_data = source_loader_iter.__next__()
                with torch.cuda.amp.autocast(enabled=args.amp):
                    track_image = track_data['rgb'].cuda()[0]
                    track_mask = track_data.get('mask')
                    if track_mask is not None:
                        track_mask = track_mask.cuda()[0]
                    track_info = track_data['info']
                    track_frame = track_info['frame'][0]
                    track_shape = track_info['shape']
                    track_path_to_image = track_info['path_to_image'][0]
                    track_info['save'][0] = True                        
                    track_segments_info = convert_json_dict_to_objects_info(track_mask,None)                        
                    track_frame_info = FrameInfo(track_image, track_mask, track_segments_info, ti, track_info)
                    need_resize = track_info['need_resize'][0]

                    cross_image = cross_data['rgb'].cuda()[0]
                    cross_mask = cross_data.get('mask')
                    if cross_mask is not None:
                        cross_mask = cross_mask.cuda()[0]
                    cross_info = cross_data['info']
                    cross_frame = cross_info['frame'][0]
                    cross_shape = cross_info['shape']
                    cross_path_to_image = cross_info['path_to_image'][0]
                    cross_info['save'][0] = True                        
                    cross_segments_info = convert_json_dict_to_objects_info(cross_mask,None)                        
                    cross_frame_info = FrameInfo(cross_image, cross_mask, cross_segments_info, ti, cross_info)    
            
                    # clip_len > 2
                    if curr_num_voting_frames > 1:                            
                        if ti + curr_num_voting_frames > next_voting_frame:
                            # wait for more frames before proceeding
                            processor.add_to_temporary_buffer(track_frame_info)
                            cross_buffer.append(cross_frame_info)
                            
                            if (ti == next_voting_frame) or (ti == len(loader)//2 - 1) :
                                # process this clip
                                processor.network.pixel_encoder.set_style_update(True)
                                this_image = processor.frame_buffer[0].image
                                this_ti = processor.frame_buffer[0].ti
                                this_frame_name = processor.frame_buffer[0].name
                                save_this_frame = processor.frame_buffer[0].save_needed
                                path_to_image = processor.frame_buffer[0].path_to_image
                                _, mask, new_segments_info = processor.vote_in_temporary_buffer(keyframe_selection='first')
                                try:
                                    prob = processor.incorporate_detection(this_image, mask, new_segments_info)
                                except:
                                    breakpoint()
                                next_voting_frame += detection_every
                                if next_voting_frame >= vid_length:
                                    next_voting_frame = vid_length + curr_num_voting_frames

                                if save_this_frame:
                                    result_saver.save_mask(
                                        prob,
                                        this_frame_name,
                                        need_resize=need_resize,
                                        shape=track_shape,
                                        path_to_image=path_to_image,
                                    )

                                ### Process cross frame
                                processor.network.pixel_encoder.set_style_update(False)
                                prob2 = processor.step(cross_buffer[0].image, None, None, end=False, update_sensory=False)

                                if cross_buffer[0].save_needed:
                                    result_saver.save_mask(prob2, cross_buffer[0].name, need_resize=need_resize, shape=cross_shape, path_to_image=cross_buffer[0].path_to_image)

                                for frame_info, frame_info2 in zip(processor.frame_buffer[1:], cross_buffer[1:]):
                                    processor.network.pixel_encoder.set_style_update(True)
                                    prob = processor.step(frame_info.image,
                                                        None,
                                                        None,
                                                        end=False,
                                                        update_sensory=True)

                                    if frame_info.save_needed:
                                        result_saver.save_mask(prob,
                                                            frame_info.name,
                                                            need_resize=need_resize,
                                                            shape=track_shape,
                                                            path_to_image=frame_info.path_to_image)

                                    processor.network.pixel_encoder.set_style_update(False)
                                    prob2 = processor.step(frame_info2.image,
                                                        None,
                                                        None,
                                                        end=False,
                                                        update_sensory=False)

                                    if frame_info2.save_needed:
                                        result_saver.save_mask(prob2,
                                                            frame_info2.name,
                                                            need_resize=need_resize,
                                                            shape=track_shape,
                                                            path_to_image=frame_info2.path_to_image)

                                processor.clear_buffer()
                                cross_buffer = []
                            else:
                                ...

                        else:
                            processor.network.pixel_encoder.set_style_update(True)
                            prob = processor.step(track_image, None, None, end=False, update_sensory=True)
                            if track_info['save'][0]:
                                result_saver.save_mask(prob,
                                                    track_frame,
                                                    need_resize=need_resize,
                                                    shape=track_shape,
                                                    path_to_image=path_to_image)

                            processor.network.pixel_encoder.set_style_update(False)
                            prob2 = processor.step(cross_image, None, None, end=False, update_sensory=False)

                            if cross_info['save'][0]:
                                result_saver.save_mask(prob2,
                                                    cross_frame,
                                                    need_resize=need_resize,
                                                    shape=cross_shape,
                                                    path_to_image=path_to_image)

                    else: # online or clip len == 2
                        if ti % detection_every == 0:
                            # incorporate new detections
                            assert track_mask is not None
                            processor.network.pixel_encoder.set_style_update(True)
                            prob = processor.incorporate_detection(track_image, track_mask, track_segments_info)
                            processor.network.pixel_encoder.set_style_update(False)
                            prob2 = processor.step(cross_image, None, None, end=False, update_sensory=False)

                        else:
                            # Run the model on this frame
                            processor.network.pixel_encoder.set_style_update(True)
                            prob = processor.step(track_image, None, None, end=False, update_sensory=True)
                            processor.network.pixel_encoder.set_style_update(False)
                            prob2 = processor.step(cross_image, None, None, end=False, update_sensory=False)


                        if track_info['save'][0]:
                            result_saver.save_mask(prob,
                                                track_frame,
                                                need_resize=need_resize,
                                                shape=track_shape,
                                                path_to_image=track_path_to_image)
                        if cross_info['save'][0]:
                            result_saver.save_mask(prob2,
                                                cross_frame,
                                                need_resize=need_resize,
                                                shape=cross_shape,
                                                path_to_image=cross_path_to_image)

            result_saver.end()
            del processor

        except Exception as e:
            print(f'Runtime error at {vid_name}')
            print(e)
            raise e  # comment this out if you want

