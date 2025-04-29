#!/usr/bin bash
export PYDEVD_DISABLE_FILE_VALIDATION=1
jupyter nbconvert --execute --to markdown quickstart.ipynb
mv quickstart.md ../README.md
mv quickstart_files/* ../quickstart_files
