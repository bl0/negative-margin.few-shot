from collections import defaultdict
from os import listdir
from os.path import join
import os
import re

cwd = os.getcwd()
data_path = join(cwd, 'ILSVRC2015/Data/CLS-LOC/train')

sorted_fnames = defaultdict(list)
fids = defaultdict(list)
with open('train.csv', 'r') as lines:
    for i, line in enumerate(lines):
        if i == 0:
            continue
        fid, _, label = re.split(r',|\.', line.strip())
        if label not in sorted_fnames.keys():
            fnames = listdir(join(data_path, label))
            fname_number = [int(re.split(r'_|\.', fname)[1]) for fname in fnames]
            sorted_fnames[label] = list(zip(*sorted(zip(fnames, fname_number), key=lambda f_tuple: f_tuple[1])))[0]

        fids[label].append(int(fid[-5:]) - 1)

filelists_flat = []
labellists_flat = []
for cl, label in enumerate(fids.keys()):
    diff = set(range(len(sorted_fnames[label]))) - set(fids[label])
    for fid in diff:
        fname = join(data_path, label, sorted_fnames[label][fid])
        filelists_flat.append(fname)
        labellists_flat.append(cl)
print(len(labellists_flat))

with open("base_val.json", "w") as fo:
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in fids.keys()])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item for item in filelists_flat])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item for item in labellists_flat])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
