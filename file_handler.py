import os
import shutil
import uuid

from fastapi import UploadFile
from werkzeug.utils import secure_filename


class FileHandler:
    def __init__(self, uploads_folder="uploads"):
        self.uploads_folder = uploads_folder

    def save_file(self, file: UploadFile, user_id):
        filename = secure_filename(file.filename)
        file_parent_path = os.path.join(self.uploads_folder, user_id)
        if not os.path.exists(file_parent_path):
            os.makedirs(file_parent_path)

        file_extension = filename.rsplit(".", 1)[1].lower()
        document_id = (
            filename[: -len(file_extension) - 1]
            + "-"
            + str(uuid.uuid4())
            + "."
            + file_extension
        )
        filepath = os.path.join(self.uploads_folder, user_id, document_id)
        with open(filepath, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        return document_id

    def delete_file(self, user_id, document_id):
        filepath = os.path.join(self.uploads_folder, user_id, document_id)
        if os.path.exists(filepath):
            os.remove(filepath)

    def delete_user_files(self, user_id):
        user_folder_path = os.path.join(self.uploads_folder, user_id)
        if os.path.exists(user_folder_path):
            shutil.rmtree(user_folder_path)

    def file_exists(self, user_id, document_id):
        filepath = os.path.join(self.uploads_folder, user_id, document_id)
        return os.path.exists(filepath)

    def get_file(self, user_id, document_id):
        filepath = os.path.join(self.uploads_folder, user_id, document_id)
        return filepath

    def list_files(self, user_id):
        file_parent_path = os.path.join(self.uploads_folder, user_id)
        if not os.path.exists(file_parent_path):
            return []
        filepaths = os.listdir(file_parent_path)
        return [
            {
                "name": filepath,
                "size": os.path.getsize(os.path.join(file_parent_path, filepath)),
            }
            for filepath in filepaths
        ]
