# Copyright 2020 TalentedSoft ( Author: Shipeng XIA )

# Sum duration obtained by using .tsv

import sys
from tensorflow_asr.utils.utils import sum_duration

file = sys.argv[1]

duration = float(sum_duration(file))
h = duration/3600
print (f"seconds: {duration:.3f}\nhours: {h:.3f}")


