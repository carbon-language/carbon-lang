#!/bin/sh
# This script prints out some of the source files that are useful when 
# editing.  I use this like this: xemacs `./getsomesrcs.sh` &
./getsrcs.sh | grep -v Assembly | grep -v Byte | grep -v \\.ll | grep -v Optimization | grep -v llvm/Support | grep -v llvm/CodeGen | grep -v '/i[^/]*$' | grep -v Holder | grep -v Analysis | grep -v html | grep include/llvm


