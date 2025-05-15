prompt = ''
main_image = None
wrist_image = None

def update_prompt(new_prompt: str):
    global prompt
    prompt = new_prompt

def get_prompt() -> str:
    return prompt

def update_images(new_main_image: np.ndarray, new_wrist_image: np.ndarray):
    global main_image, wrist_image
    main_image = new_main_image
    wrist_image = new_wrist_image

def get_images() -> tuple[np.ndarray, np.ndarray]:
    return main_image, wrist_image