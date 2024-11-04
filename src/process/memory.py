from .preprocess import ProcessMemoryContent
from .augment import AugmentContext

class Memory():
    def __init__(self,
                 raw_folder,
                 processed_folder) -> None:
        self.preprocess_memory = ProcessMemoryContent(
            raw_data_folder=raw_folder,
            processed_folder=processed_folder
            )
        
    def preprocess(self):
        self.preprocess_memory.process()
        self.memory_content_processed = self.preprocess_memory.memory_content_processed

    def augment(self):
        if self.memory_content_processed is None:
            raise ValueError("Memory content is not processed yet.")
        self.augment_context = AugmentContext(
            memory_content_processed=self.memory_content_processed
        )
