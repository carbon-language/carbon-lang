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

# Common functionality for test scripts
# Process arguments, expecting source file as 1st; optional path to f18 as 2nd
# Set: $F18 to the path to f18; $temp to an empty temp directory; and $src
# to the full path of the single source argument

PATH=/usr/bin:/bin

function die {
  echo "$(basename $0): $*" >&2
  exit 1
}

case $# in
  (1) ;;
  (2) F18=$2 ;;
  (*) echo "Usage: $(basename $0) <fortran-source> [<f18-executable>]"; exit 1
esac
[[ -z ${F18+x} ]] && die "Path to f18 must be second argument or in F18 environment variable"
[[ ! -f $F18 ]] && die "f18 executable not found: $F18"
case $1 in
  (/*) src=$1 ;;
  (*) src=$(dirname $0)/$1 ;;
esac
temp=`mktemp -d ./tmp.XXXXXX`
[[ $KEEP ]] || trap "rm -rf $temp" EXIT
