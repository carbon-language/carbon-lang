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

# Compile a source file and compare generated .mod files against expected.

set -e
F18_OPTIONS="-fdebug-resolve-names -fparse-only"
srcdir=$(dirname $0)
source $srcdir/common.sh

actual=$temp/actual.mod
expect=$temp/expect.mod
actual_files=$temp/actual_files
prev_files=$temp/prev_files
diffs=$temp/diffs

set $src

touch $actual
for src in "$@"; do
  [[ ! -f $src ]] && echo "File not found: $src" && exit 1
  path=$(git ls-files --full-name $src 2>/dev/null || echo $src)
  (
    cd $temp
    ls -1 *.mod > prev_files
    $F18 $F18_OPTIONS $src
    ls -1 *.mod | comm -13 prev_files -
  ) > $actual_files
  expected_files=$(sed -n 's/^!Expect: \(.*\)/\1/p' $src | sort)
  extra_files=$(echo "$expected_files" | comm -23 $actual_files -)
  if [[ ! -z "$extra_files" ]]; then
    echo "Unexpected .mod files produced:" $extra_files
    die FAIL $path
  fi
  for mod in $expected_files; do
    if [[ ! -f $temp/$mod ]]; then
      echo "Compilation did not produce expected mod file: $mod"
      die FAIL $path
    fi
    # The first three bytes of the file are a UTF-8 BOM
    sed '/^[^!]*!mod\$/d' $temp/$mod > $actual
    sed '1,/^!Expect: '"$mod"'/d' $src | sed -e '/^$/,$d' -e 's/^! *//' > $expect
    if ! diff -U999999 $expect $actual > $diffs; then
      echo "Module file $mod differs from expected:"
      sed '1,2d' $diffs
      die FAIL $path
    fi
  done
  rm -f $actual $expect
done
echo PASS
