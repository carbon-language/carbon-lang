#!/bin/sh
# This script prints out some of the source files that are useful when 
# editing.  I use this like this: xemacs `./getsomesrcs.sh` &
./getsrcs.sh | grep -v Assembly | grep -v Byte | grep -v \\.ll | grep -v tools | grep -v Makefile | grep -v Opt | grep -v llvm/Tools | grep -v '/i[^/]*$' | grep -v VMCore | grep -v Holder | grep -v Analysis | grep -v html


