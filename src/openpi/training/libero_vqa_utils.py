import numpy as np
from collections import OrderedDict


_OBJECT_NAME_MAP = OrderedDict()

_OBJECT_NAME_MAP['the black bowl between the plate and the ramekin'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl from table center'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl in the top drawer of the wooden cabinet'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl next to the cookie box'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl next to the plate'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl next to the ramekin'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl on the cookie box'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl on the ramekin'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl on the stove'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl on the wooden cabinet'] = 'akita_black_bowl_1_main'

_OBJECT_NAME_MAP['the front black bowl'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the middle black bowl'] = 'akita_black_bowl_2_main'
_OBJECT_NAME_MAP['the back black bowl'] = 'akita_black_bowl_3_main'
_OBJECT_NAME_MAP['the black bowl at the front'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl in the middle'] = 'akita_black_bowl_2_main'
_OBJECT_NAME_MAP['the black bowl at the back'] = 'akita_black_bowl_3_main'
_OBJECT_NAME_MAP['the black bowl on the left'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the black bowl'] = 'akita_black_bowl_1_main'


_OBJECT_NAME_MAP['the left bowl'] = 'akita_black_bowl_1_main'
_OBJECT_NAME_MAP['the right bowl'] = 'akita_black_bowl_2_main'
_OBJECT_NAME_MAP['the bowl'] = 'akita_black_bowl_1_main'

_OBJECT_NAME_MAP['the wine_bottle'] = 'wine_bottle_1_main'
_OBJECT_NAME_MAP['the ketchup'] = 'ketchup_1_main'
_OBJECT_NAME_MAP['the frying pan'] = 'chefmate_8_frypan_1_main'

_OBJECT_NAME_MAP['the left moka pot'] = 'moka_pot_2_main'
_OBJECT_NAME_MAP['the right moka pot'] = 'moka_pot_1_main'
_OBJECT_NAME_MAP['the moka pot'] = 'moka_pot_1_main'

_OBJECT_NAME_MAP['the yellow and white mug'] = 'white_yellow_mug_1_main'
_OBJECT_NAME_MAP['the white mug'] = 'porcelain_mug_1_main'
_OBJECT_NAME_MAP['the red mug'] = 'red_coffee_mug_1_main'

_OBJECT_NAME_MAP['the white bowl'] = 'white_bowl_1_main'

_OBJECT_NAME_MAP['the butter at the back'] = 'butter_2_main'
_OBJECT_NAME_MAP['the butter at the front'] = 'butter_1_main'
_OBJECT_NAME_MAP['the butter'] = 'butter_1_main'

_OBJECT_NAME_MAP['the chocolate pudding'] = 'chocolate_pudding_1_main'

_OBJECT_NAME_MAP['the alphabet soup'] = 'alphabet_soup_1_main'
_OBJECT_NAME_MAP['the cream cheese'] = 'cream_cheese_1_main'
_OBJECT_NAME_MAP['the cream cheese box'] = 'cream_cheese_1_main'
_OBJECT_NAME_MAP['the tomato sauce'] = 'tomato_sauce_1_main'
_OBJECT_NAME_MAP['the milk'] = 'milk_1_main'
_OBJECT_NAME_MAP['the orange juice'] = 'orange_juice_1_main'
_OBJECT_NAME_MAP['the salad dressing'] = ['new_salad_dressing_1_main', 'salad_dressing_1_main']
_OBJECT_NAME_MAP['the bbq sauce'] = 'bbq_sauce_1_main'

_OBJECT_NAME_MAP['the book on the left'] = 'yellow_book_2_main'
_OBJECT_NAME_MAP['the book on the right'] = 'yellow_book_1_main'
_OBJECT_NAME_MAP['the book'] = 'black_book_1_main'

_OBJECT_NAME_MAP['the left plate'] = 'plate_1_main'
_OBJECT_NAME_MAP['the right plate'] = 'plate_2_main'
_OBJECT_NAME_MAP['the plate'] = 'plate_1_main'

_OBJECT_NAME_MAP['the top drawer of the cabinet'] = ['white_cabinet_1_cabinet_top', 'wooden_cabinet_1_cabinet_top']
_OBJECT_NAME_MAP['the middle drawer of the cabinet'] = ['white_cabinet_1_cabinet_top', 'wooden_cabinet_1_cabinet_top']
_OBJECT_NAME_MAP['the bottom drawer of the cabinet'] = ['white_cabinet_1_cabinet_bottom', 'wooden_cabinet_1_cabinet_bottom']

_OBJECT_NAME_MAP['top of the cabinet'] = ['white_cabinet_1_main', 'wooden_cabinet_1_main', 'wooden_two_layer_shelf_1_main']

_OBJECT_NAME_MAP['on top of the shelf'] = 'wooden_two_layer_shelf_1_main'
_OBJECT_NAME_MAP['on the cabinet shelf'] = 'wooden_two_layer_shelf_1_main'
_OBJECT_NAME_MAP['under the cabinet shelf'] = 'wooden_two_layer_shelf_1_main'

_OBJECT_NAME_MAP['turn on the stove'] = 'flat_stove_1_button'
_OBJECT_NAME_MAP['turn off the stove'] = 'flat_stove_1_button'
_OBJECT_NAME_MAP['the stove'] = 'flat_stove_1_burner'

_OBJECT_NAME_MAP['the tray'] = 'wooden_tray_1_main'
_OBJECT_NAME_MAP['the wine rack'] = 'wine_rack_1_main'
_OBJECT_NAME_MAP['the rack'] = 'wine_rack_1_main'
_OBJECT_NAME_MAP['the microwave'] = 'microwave_1_main'
_OBJECT_NAME_MAP['the basket'] = 'basket_1_main'
_OBJECT_NAME_MAP['the caddy'] = 'desk_caddy_1_main'

_OBJECT_NAME_MAP['on it'] = 'flat_stove_1_burner'


def find_target_objects(lang):
    if lang == 'put both moka pots on the stove':
        return ['the left moka pot', 'the right moka pot', 'the stove']
    target_objects = []
    for obj in _OBJECT_NAME_MAP:
        if obj in lang:
            target_objects.append(obj)
            lang = lang.replace(obj, '')
    return target_objects


def post_process_object(obj):
    return obj.replace('turn on the stove', 'the stove button') \
              .replace('turn off the stove', 'the stove button') \
              .replace('on top of the shelf', 'top of the shelf') \
              .replace('on the cabinet shelf', 'top of the cabinet shelf') \
              .replace('under the cabinet shelf', 'bottom of the cabinet self') \
              .replace('on it', 'the stove')


def get_relation_to_robot(obj, obj_info, thresh=0.06, check_catch=True, check_close=True, mode='coarse_direction'):
    if mode in obj_info:
        return obj_info[mode][obj]
    
    aliases = _OBJECT_NAME_MAP[obj]
    if isinstance(aliases, str):
        alias = aliases
    else:
        alias = None
        for a in aliases:
            if a in obj_info['gripper_to_obj']:
                alias = a
                break
        if alias is None:
            raise ValueError(f"Cannot find alias for {obj} in object info, aliases: {aliases}, available objects: {list(obj_info.get('gripper_to_obj').keys())}.")
    
    gripper_to_obj = obj_info['gripper_to_obj'][alias]
    
    if alias in ['white_cabinet_1_cabinet_top', 'wooden_cabinet_1_cabinet_top', 'white_cabinet_1_main', 'wooden_cabinet_1_main']:
        gripper_to_obj[2] = gripper_to_obj[2] + 0.22152
    
    if alias == 'basket_1_main':
        gripper_to_obj[2] = gripper_to_obj[2] + 0.07185
    
    if alias == 'wine_rack_1_main':
        gripper_to_obj[2] = gripper_to_obj[2] + 0.05903
        
    if alias == 'wooden_two_layer_shelf_1_main' and obj in ['on top of the shelf', 'on the cabinet shelf', 'top of the cabinet']:
        gripper_to_obj[2] = gripper_to_obj[2] + 0.22152
    
    if alias.endswith('_main'):
        is_grasp = obj_info['is_grasp'].get(alias[:-5], False)
    elif alias in obj_info['is_grasp']:
        is_grasp = obj_info['is_grasp'][alias]
    else:
        is_grasp = False
    
    if check_catch and is_grasp:
        return 'catch'

    if not check_close:
        thresh = 0

    if mode == 'coarse_direction':
        # return the direction with the largest absolute distance of each axis
        max_idx = np.argmax(np.abs(gripper_to_obj))
        if max_idx == 0:
            if gripper_to_obj[0] < -thresh:
                return 'back'
            if gripper_to_obj[0] > thresh:
                return 'front'
            return 'close'

        if max_idx == 1:
            if gripper_to_obj[1] < -thresh:
                return 'left'
            if gripper_to_obj[1] > thresh:
                return 'right'

        if gripper_to_obj[2] < -thresh:
            return 'down'
        if gripper_to_obj[2] > thresh:
            return 'up'
        return 'close'
    
    elif mode == 'coarse_direction_3d':
        # return the direction of x, y, z
        outputs = []
        if gripper_to_obj[0] < -thresh:
            outputs.append('back')
        elif gripper_to_obj[0] > thresh:
            outputs.append('front')
        else:
            outputs.append('close')

        if gripper_to_obj[1] < -thresh:
            outputs.append('left')
        elif gripper_to_obj[1] > thresh:
            outputs.append('right')
        else:
            outputs.append('close')
            
        if gripper_to_obj[2] < -thresh:
            outputs.append('down')
        elif gripper_to_obj[2] > thresh:
            outputs.append('up')
        else:
            outputs.append('close')
        return ' '.join(outputs)
    
    elif mode == 'fine_direction_3d':
        # return the accurate direction of x, y, z
        outputs = []
        outputs.append('{:.1f}'.format(gripper_to_obj[0]))
        outputs.append('{:.1f}'.format(gripper_to_obj[1]))
        outputs.append('{:.1f}'.format(gripper_to_obj[2]))
        return ' '.join(outputs)
    
    elif mode == 'coarse_distance':
        # return near, middle or far
        distance = np.linalg.norm(gripper_to_obj)
        if distance < thresh:
            return 'close'
        elif distance < 0.2:
            return 'near'
        elif distance < 0.3:
            return 'middle'
        return 'far'
    
    elif mode == 'fine_distance':
        # return accurate distance
        distance = np.linalg.norm(gripper_to_obj)
        return '{:.1f}'.format(distance)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_question_template(mode):
    if mode == 'coarse_direction':
        return 'In which direction is {} relative to the robot?'
    if mode == 'coarse_direction_3d':
        return 'In which direction is {} relative to the robot? x, y, z:'
    if mode == 'fine_direction_3d':
        return 'What is the accurate position of {} relative to the robot? x, y, z:'
    if mode == 'coarse_distance':
        return 'What is the distance between the robot and {}?'
    if mode == 'fine_distance':
        return 'What is the accurate distance between the robot and {}?'
    raise ValueError(f"Unknown mode: {mode}")


def get_object_info(env, obs):
    pos_keys = [key for key in obs.keys() if key.endswith('pos')]
    positions = {key: list(obs[key]) for key in pos_keys}

    objects_dict = {**env.env.objects_dict, **env.env.fixtures_dict}
    objects = env.env.sim.data.model.body_names
    
    is_grasps = {}
    for obj_name, obj in objects_dict.items():
        is_grasps[obj_name] = env.env._check_grasp(env.env.robots[0].gripper, obj)
    
    gripper_to_objs = {}
    # for obj_name, obj in objects_dict.items():
    #     gripper_to_objs[obj_name] = list(env.env._gripper_to_target(env.env.robots[0].gripper, obj, return_distance=False))
    for obj in objects:
        gripper_to_objs[obj] = list(env.env._gripper_to_target(env.env.robots[0].gripper, obj, return_distance=False))

    gripper_to_obj_distances = {}
    # for obj_name, obj in objects_dict.items():
    #     gripper_to_obj_distances[obj_name] = env.env._gripper_to_target(env.env.robots[0].gripper, obj, return_distance=True)
    for obj in objects:
        gripper_to_obj_distances[obj] = env.env._gripper_to_target(env.env.robots[0].gripper, obj, return_distance=True)
    
    return {
        'position': positions,
        'is_grasp': is_grasps,
        'gripper_to_obj': gripper_to_objs,
        'gripper_to_obj_distance': gripper_to_obj_distances,
        # 'name_to_seg_id': obj_name_to_seg_id
    }


def get_vqa_instruction_prompt(prompt, object_info):
    vqa_prompt = ''
    objects = find_target_objects(prompt)
    
    for obj in objects:
        vqa_prompt = vqa_prompt + get_question_template('coarse_direction').format(post_process_object(obj)) + ' '
        vqa_prompt = vqa_prompt + get_relation_to_robot(obj, object_info, check_catch=True, check_close=False, mode='coarse_direction') + '. '

    return vqa_prompt


def get_vqa_questions(prompt):
    questions = []
    objects = find_target_objects(prompt)
    
    for obj in objects:
        questions.append(get_question_template('coarse_direction').format(post_process_object(obj)))

    return questions
