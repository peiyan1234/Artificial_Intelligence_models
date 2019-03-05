"""
This tool is created by Alvin Pei-Yan Li.
Please contact him, d05548014@ntu.edu.tw / Alvin.Li@acer.com / alvin_li@acctom.com / a4624393@gmail.com,
for futher authorization of the use of this tool.
"""

import gzip
import glob
import os.path

source_dir = "C:/Users/Alvin.Li/Desktop/small_project/dataset_zipped"
dest_dir = "C:/Users/Alvin.Li/Desktop/small_project/dataset"

for src_name in glob.glob(os.path.join(source_dir, '*.gz')):
    base = os.path.basename(src_name)
    dest_name = os.path.join(dest_dir, base[:-3])
    with gzip.open(src_name, 'rb') as infile:
        with open(dest_name, 'wb') as outfile:
            for line in infile:
                outfile.write(line)
