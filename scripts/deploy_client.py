import dataclasses
import enum
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro



@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000

    num_steps: int = 10


def main(args: Args) -> None:
    from PIL import Image
    
    main_img = Image.open('main.png').convert("RGB").resize((224, 224))
    wrist_img = Image.open('wrist.png').convert("RGB").resize((224, 224))
    main_img = np.array(main_img)
    wrist_img = np.array(wrist_img)
    
    obs_fn = {
        "image": np.random.randint(256, size=(500, 500, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(500, 500, 3), dtype=np.uint8),
        "state": np.random.rand(7),
        # "actions": np.random.rand(5,7),
        "prompt": "pick blue cup",
    }

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {policy.get_server_metadata()}")

    # Send 1 observation to make sure the model is loaded.
    policy.infer(obs_fn)

    start = time.time()
    for _ in range(args.num_steps):
        print(policy.infer(obs_fn))
    end = time.time()

    print(f"Total time taken: {end - start:.2f} s")
    print(f"Average inference time: {1000 * (end - start) / args.num_steps:.2f} ms")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
