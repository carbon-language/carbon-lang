#!/usr/bin/env bash
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
F18=${F18:=../../tools/f18/f18}
FileCheck=${FileCheck:=internal_check}

function internal_check() {
  r=true
  linput="$1"
  lstdin=`mktemp`
  lcheck=`mktemp`
  cat - > ${lstdin}
  egrep '^[[:space:]]*![[:space:]]*CHECK:[[:space:]]*' ${linput} | sed -e 's/^[[:space:]]*![[:space:]]*CHECK:[[:space:]]*//' > ${lcheck} 2>/dev/null
  while read p; do
    if egrep "${p}" ${lstdin} >/dev/null 2>&1; then
      true
    else
      echo "Not found: ${p}" >&2
      r=false
    fi
  done < ${lcheck}
  egrep '^[[:space:]]*![[:space:]]*CHECK-NOT:[[:space:]]*' ${linput} | sed -e 's/^[[:space:]]*![[:space:]]*CHECK-NOT:[[:space:]]*//' > ${lcheck} 2>/dev/null
  while read p; do
    if egrep ${p} ${lstdin} >/dev/null 2>&1; then
      echo "Found: ${p}" >&2
      r=false
    fi
  done < ${lcheck}
  rm -f ${lstdin} ${lcheck}
  ${r}
}

r=0
for input in $*; do
  finput="${srcdir}/${input}"
  CMD=$(cat ${finput} | egrep '^[[:space:]]*![[:space:]]*RUN:[[:space:]]*' | sed -e 's/^[[:space:]]*![[:space:]]*RUN:[[:space:]]*//')
  CMD=$(echo ${CMD} | sed -e "s:%s:${finput}:g")
  eval "( ${CMD} )" || (echo "test ${finput} failed"; r=1)
done
exit $r
