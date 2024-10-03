# callbacks.py
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from gcs_utils import upload_blob
import threading
import time
from gcs_utils import upload_blob
import os

class GCSCheckpointCallback(ModelCheckpoint):
    def __init__(self, bucket_name, gcs_ckpt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        self.gcs_ckpt_path = gcs_ckpt_path

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Call the superclass method to ensure checkpoint is saved
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        # Get the path of the last saved checkpoint
        last_ckpt_path = self.last_model_path

        if last_ckpt_path:
            # Construct the destination path in GCS
            filename = os.path.basename(last_ckpt_path)
            gcs_checkpoint_path = os.path.join(self.gcs_ckpt_path, filename).replace("\\", "/")

            # Upload the checkpoint to GCS
            upload_blob(self.bucket_name, last_ckpt_path, gcs_checkpoint_path)
            print(f"Uploaded checkpoint to gs://{self.bucket_name}/{gcs_checkpoint_path}")


class GCSTensorBoardLoggerCallback(Callback):
    def __init__(self, bucket_name, local_tb_log_dir, gcs_tb_log_dir, upload_interval=300):
        super().__init__()
        self.bucket_name = bucket_name
        self.local_tb_log_dir = local_tb_log_dir
        self.gcs_tb_log_dir = gcs_tb_log_dir
        self.upload_interval = upload_interval  # In seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.upload_logs_periodically)
        self.thread.daemon = True  # Daemon thread will exit when main program exits

    def on_train_start(self, trainer, pl_module):
        # Start the upload thread
        self.thread.start()

    def on_train_end(self, trainer, pl_module):
        # Stop the upload thread
        self.stop_event.set()
        self.thread.join()
        # Perform a final upload to ensure all logs are uploaded
        self.upload_logs()

    def upload_logs_periodically(self):
        while not self.stop_event.is_set():
            self.upload_logs()
            time.sleep(self.upload_interval)

    def upload_logs(self):
        for root, dirs, files in os.walk(self.local_tb_log_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, self.local_tb_log_dir)
                gcs_file_path = os.path.join(self.gcs_tb_log_dir, relative_path).replace("\\", "/")
                try:
                    upload_blob(self.bucket_name, local_file_path, gcs_file_path)
                except Exception as e:
                    print(f"Error uploading {local_file_path}: {e}")