#!/bin/sh

# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

# The mvifdiff.sh script works like the mv(1) command, except
# that it does not touch the destination file if its contents
# are the same as the source file.

if cmp -s "$1" "$2" ; then
  rm "$1"
else
  mv "$1" "$2"
fi
