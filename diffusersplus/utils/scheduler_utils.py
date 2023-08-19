from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

scheduler_mapping = {
    "DDIM": DDIMScheduler,
    "DDPMScheduler": DDPMScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
    "EulerAncestralDiscrete": EulerAncestralDiscreteScheduler,
    "EulerDiscrete": EulerDiscreteScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler,
    "KDPM2Discrete": KDPM2DiscreteScheduler,
    "PNDMScheduler": PNDMScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
}


def get_scheduler(pipe, scheduler_name):
    if scheduler_name in scheduler_mapping:
        SchedulerClass = scheduler_mapping[scheduler_name]
        pipe.scheduler = SchedulerClass.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Invalid scheduler name {scheduler_name}")

    return pipe
