_MUJOCO_GL = 'glfw'

import os
os.environ['MUJOCO_GL'] = _MUJOCO_GL

import OpenGL
OpenGL.ERROR_CHECKING = False

import sys
sys.path.append('/home-ssd/Users/nsgm_lx/wushihan/Codes/openpi')

import argparse
import collections
import imageio
import math
import multiprocessing
import numpy as np
import os
import pathlib
import traceback

from openpi_client import image_tools
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as training_config
from openpi.training.libero_vqa_utils import get_vqa_questions, get_vqa_instruction_prompt

from utils.logger import Logger, reset_logging


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _get_libero_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class ParallelLiberoEvalutor:
    def __init__(
        self, 
        config_name: str,
        checkpoint_dir: str,
        task_suite_name: str = "libero_90",
        resize_size: int = 224,
        num_steps_wait: int = 10 ,
        num_trials_per_task: int = 10,
        seed: int = 7,
        num_processes: int = 16,
        save_root: str = 'results'
    ):
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        self.task_suite_name = task_suite_name
        self.resize_size = resize_size
        self.num_steps_wait = num_steps_wait
        self.num_trials_per_task = num_trials_per_task
        self.seed = seed
        self.num_processes = num_processes
        self.save_root = os.path.join(save_root, config_name, task_suite_name)
        self.video_out_path = os.path.join(self.save_root, 'videos')

        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.video_out_path, exist_ok=True)

        if self.task_suite_name == "libero_spatial":
            self.max_steps = 220  # longest training demo has 193 steps
        elif self.task_suite_name == "libero_object":
            self.max_steps = 280  # longest training demo has 254 steps
        elif self.task_suite_name == "libero_goal":
            self.max_steps = 300  # longest training demo has 270 steps
        elif self.task_suite_name == "libero_10":
            self.max_steps = 520  # longest training demo has 505 steps
        elif self.task_suite_name == "libero_90":
            self.max_steps = 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task suite: {self.task_suite_name}")
    
    def evaluate(self):
        from libero.libero import benchmark

        reset_logging()
        self._build_logger()
        np.random.seed(self.seed)

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = self.task_suite.n_tasks

        gpus = self._check_free_gpus()
        task_ids_and_episodes_all_processes = [[] for _ in range(self.num_processes)]
        idx = 0
        for task_id in range(num_tasks_in_suite):
            for episode in range(self.num_trials_per_task):
                task_ids_and_episodes_all_processes[idx % self.num_processes].append((task_id, episode))
                idx += 1

        processes = []
        manager = multiprocessing.Manager()
        summaries = manager.list()
        
        for idx, task_ids_and_episodes in enumerate(task_ids_and_episodes_all_processes):
            gpu = gpus[idx % len(gpus)]
            self.logger.info(f'GPU {gpu}: {task_ids_and_episodes}')
            process = multiprocessing.Process(target=self.evaluate_episodes,
                                              args=(gpu, task_ids_and_episodes, idx == 0, summaries))
            processes.append(process)
            
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        # summaries = []
        # self.evaluate_episodes(gpus[0], task_ids_and_episodes_all_processes[0], True, summaries)

        reset_logging()
        self._build_logger('a')

        task_ids = set([summary["task_id"] for summary in summaries])
        for task_id in task_ids:
            task_summaries = [summary for summary in summaries if summary["task_id"] == task_id]
            success_rate = sum([summary["success"] for summary in task_summaries]) / len(task_summaries)
            task_description = task_summaries[0]['task']
            self.logger.info(f"Task {task_id} {task_description} success rate: {success_rate:.2f}")
        
        success_rate = sum([summary["success"] for summary in summaries]) / len(summaries)
        self.logger.info(f"Overall success rate: {success_rate:.2f}")
        self.logger.info("Evaluation finished.")
        
    def evaluate_episodes(self, gpu, task_ids_and_episodes, show_detail, summaries):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ['MUJOCO_GL'] = _MUJOCO_GL
        import OpenGL
        OpenGL.ERROR_CHECKING = False
        try:
            policy = self._build_policy()
            reset_logging()
            self._build_logger('a')

            for task_id, episode in task_ids_and_episodes:
                self.logger.info(f"GPU {gpu}: task {task_id} episode {episode}")
                summary = self.evaluate_single(policy, task_id, episode, show_detail)
                summaries.append(summary)
        
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            with open(os.path.join(self.save_root, f'error_gpu{gpu}.log'), 'w') as f:
                f.write(str(e) + '\n')
                f.write(traceback.format_exc())
    
    def evaluate_single(self, policy, task_id, episode, show_detail):
        task = self.task_suite.get_task(task_id)
        initial_states = self.task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, self.seed)
        env.reset()
        action_plan = collections.deque()

        if self.task_suite_name != 'libero_object':
            obs = env.set_init_state(initial_states[episode])

        t = 0
        replay_images = []
        while t < self.max_steps + self.num_steps_wait:
            if t < self.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, self.resize_size, self.resize_size))
            wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, self.resize_size, self.resize_size))
            replay_images.append(img)

            if not action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }

                if 'vqa' in self.config_name:
                    element["vqa"] = get_vqa_questions(str(task_description))
                
                outputs = policy.infer(element)
                action_chunk = outputs['actions']
                if show_detail:
                    if 'vqa' in outputs:
                        vqa = outputs['vqa']
                        self.logger.info(f'vqa: {vqa}')
                    self.logger.info(f'actions: {action_chunk}')
                action_plan.extend(action_chunk)

            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            if done:
                break
            t += 1

        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_")
        imageio.mimwrite(
            pathlib.Path(self.video_out_path) / f"{task_id}_{task_segment}_episode{episode}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=30,
        )
        self.logger.info(f'Task {task_id} {task_description} episode {episode} {suffix}.')
        return {"task_id": task_id, "task": task_description, "episode": episode, "success": done}

    def _build_policy(self):
        config = training_config.get_config(self.config_name)
        checkpoint_dir = download.maybe_download(self.checkpoint_dir)
        if 'vqa' not in self.config_name:
            return _policy_config.create_trained_policy(config, checkpoint_dir)
        elif 'v2' not in self.config_name:
            return _policy_config.create_fast_vqa_trained_policy(config, checkpoint_dir)
        else:
            return _policy_config.create_fast_vqa_trained_policy_v2(config, checkpoint_dir)

    def _build_logger(self, mode='w'):
        self.logger = Logger(os.path.join(self.save_root, '000.log'), mode=mode)       

    def _check_free_gpus(self):
        used_memorys = os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader").readlines()
        used_memorys = [int(memory.strip()) for memory in used_memorys]
        return [i for i, memory in enumerate(used_memorys) if memory < 1000]


def main(args):
    evaluator = ParallelLiberoEvalutor(args.config_name, args.checkpoint_dir, args.task_suite_name)
    evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', default='pi0_fast_libero_only_front_chunk5')
    parser.add_argument('--checkpoint-dir', default='checkpoints/pi0_fast_libero_only_front_chunk5/pi0_fast_libero_only_front_chunk5/29999')
    parser.add_argument('--task-suite-name', default='libero_90')
    args = parser.parse_args()
    main(args)
