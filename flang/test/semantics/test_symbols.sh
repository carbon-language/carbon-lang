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

# Compile a source file with '-funparse-with-symbols' and verify
# we get the right symbols in the output, i.e. the output should be
# the same as the input, except for the copyright comment.
# Change the compiler by setting the F18 environment variable.

PATH=/usr/bin:/bin
srcdir=$(dirname $0)
CMD="${F18:-../../../tools/f18/f18} -funparse-with-symbols"

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

src1=$temp/1.f90
src2=$temp/2.f90
src3=$temp/3.f90
diffs=$temp/diffs

# Strip out blank lines and all comments except "!DEF:" and "!REF:"
sed -e 's/!\([DR]EF:\)/KEEP \1/' \
  -e 's/!.*//' -e 's/ *$//' -e '/^$/d' -e 's/KEEP \([DR]EF:\)/!\1/' \
  $src > $src1
egrep -v '^ *!' $src1 > $src2  # strip out meaningful comments
( cd $temp; $CMD $(basename $src2) ) > $src3  # compile, inserting comments for symbols

if diff -w -U999999 $src1 $src3 > $diffs; then
  echo PASS
else
  sed '1,/^\@\@/d' $diffs
  echo
  echo FAIL
  exit 1
fi
