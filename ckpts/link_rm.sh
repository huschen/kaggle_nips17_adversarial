#!/bin/bash
#
# Scripts to remove all checkpoints links.
#

sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$sdir"/..
find . -type l -name '*.ckpt*' | xargs rm
find . -type f -name '.DS_Store' | xargs rm
find . -type d -name '__pycache__' | xargs rm -r
