#!/bin/sh
# This script prints out some of the source files that are useful when 
# editing.  I use this like this: xemacs `./getsomesrcs.sh` &
./getsrcs.sh | grep -v Assembly | grep -v Byte | grep -v \\.ll | grep -v tools | grep -v Makefile | grep -v Opt | grep -v CommandLi | grep -v String | grep -v DataType | grep -v '/i[^/]*$' | grep -v SlotCalcul | grep -v VMCore


