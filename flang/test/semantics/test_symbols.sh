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

F18_OPTIONS="-funparse-with-symbols"
srcdir=$(dirname $0)
source $srcdir/common.sh
[[ ! -f $src ]] && echo "File not found: $src" && exit 1

src1=$temp/1.f90
src2=$temp/2.f90
src3=$temp/3.f90
diffs=$temp/diffs

# Strip out blank lines and all comments except "!DEF:", "!REF:", and "!$omp"
sed -e 's/!\([DR]EF:\)/KEEP \1/' -e 's/!\($omp\)/KEEP \1/' \
  -e 's/!.*//' -e 's/ *$//' -e '/^$/d' -e 's/KEEP \([DR]EF:\)/!\1/' \
  -e 's/KEEP \($omp\)/!\1/' \
  $src > $src1
egrep -v '![DR]EF:' $src1 > $src2  # strip out DEF and REF comments
# compile, inserting comments for symbols:
( cd $temp; $F18 $F18_OPTIONS $USER_OPTIONS $(basename $src2) ) > $src3

if diff -w -U999999 $src1 $src3 > $diffs; then
  echo PASS
else
  sed '1,/^\@\@/d' $diffs
  echo
  echo FAIL
  exit 1
fi
