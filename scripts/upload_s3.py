import logging
import boto3
from botocore.exceptions import ClientError
import os
import sys
import threading

BUCKET = 'aicitychallenge2023'

class ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(os.path.join(os.path.dirname(__file__), filename)))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


def upload_file(file_name, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    dirname = os.path.dirname(__file__)
    filePath = os.path.join(dirname, file_name)

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        # object_name = os.path.basename(file_name)
        object_name = file_name

    # Upload the file
    s3r = boto3.resource('s3')
    try:
        s3r.Bucket(BUCKET).upload_file(filePath, object_name,
        Callback=ProgressPercentage(file_name))
    except ClientError as e:
        logging.error(e)
        return False
    return True

def main(dataset, split):
    dirname = os.path.dirname(__file__)
    rel_path = f"{dataset}/{split}"
    root_path = os.path.join(dirname, rel_path)

    # seq_list = os.listdir(root_path)
    seq_list = ['c036']
    for seq_name in seq_list:
        path_data = f"{rel_path}/{seq_name}"
        # print(path_data)
        path_roi = f"{path_data}/roi.jpg"
        path_images = f"{path_data}/img1"

        upload_file(path_roi)
        for img_name in os.listdir(os.path.join(dirname, path_images)):
            upload_file(f"{path_images}/{img_name}")

if __name__ == '__main__':
    main('validation', 'S05')
