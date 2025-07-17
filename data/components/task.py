from enum import Enum


class TaskType(Enum):
    CLASSIFY = 'classify'
    DETECT = 'detect'
    POSE = 'pose'
    SEGMENT = 'segment'

    @staticmethod
    def from_str(s: str) -> 'TaskType':
        tasks: list[TaskType] = [
            TaskType.CLASSIFY,
            TaskType.DETECT,
            TaskType.POSE,
            TaskType.SEGMENT,
        ]
        for task in tasks:
            if s == task.value:
                return task

        raise Exception(f'unknown task type: {s}')