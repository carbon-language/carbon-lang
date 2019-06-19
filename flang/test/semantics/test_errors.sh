#!/usr/bin/env bash
# Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Compile a source file and check errors against those listed in the file.
# Change the compiler by setting the F18 environment variable.

PATH=/usr/bin:/bin
srcdir=$(dirname $0)
CMD="${F18:-../../../tools/f18/f18} -fdebug-resolve-names -fparse-only -Mstandard"

if [[ $# != 1 ]]; then
  echo "Usage: $0 <fortran-source>"
  exit 1
fi
src=$srcdir/$1
[[ ! -f $src ]] && echo "File not found: $src" && exit 1

temp=temp-$1
rm -rf $temp
mkdir $temp
[[ $KEEP ]] || trap "rm -rf $temp" EXIT

log=$temp/log
actual=$temp/actual
expect=$temp/expect
diffs=$temp/diffs
options=$temp/options

# See if there are additional options
sed -n 's/^ *! *OPTIONS: *//p' $src > $options
cat $options

cmd="$CMD `cat $options` $src"
( cd $temp; $cmd ) > $log 2>&1
if [[ $? -ge 128 ]]; then
  cat $log
  exit 1
fi

# $actual has errors from the compiler; $expect has them from !ERROR comments in source
# Format both as "<line>: <text>" so they can be diffed.
sed -n 's=^[^:]*:\([^:]*\):[^:]*: error: =\1: =p' $log > $actual
awk '
  BEGIN { FS = "!ERROR: "; }
  /^ *!ERROR: / { errors[nerrors++] = $2; next; }
  { for (i = 0; i < nerrors; ++i) printf "%d: %s\n", NR, errors[i]; nerrors = 0; }
' $src > $expect

if diff -U0 $actual $expect > $diffs; then
  echo PASS
else
  echo "$cmd"
  < $diffs \
    sed -n -e 's/^-\([0-9]\)/actual at \1/p' -e 's/^+\([0-9]\)/expect at \1/p' \
    | sort -n -k 2
  echo FAIL
  exit 1
fi
