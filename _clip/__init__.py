import gdown, os

res = [
    ('bpe_simple_vocab_16e6.txt.gz', '933b7abbbbde62c36f02f0e6ccde464f', '1rfEQqTWOpfSC7VxoQ08WPGdUlI4ddO4N')
]

def md5sum(filename):
    import hashlib
    md5 = hashlib.md5()

    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b''):
            md5.update(chunk)

    return md5.hexdigest()

for filename, md5, file_id in res:
    if not os.path.exists(filename):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False)
    elif md5sum(filename) != md5:
        os.remove(filename)
        gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False)

    assert md5sum(filename) == md5, f'{filename} is corrupted!'

from .clip import *