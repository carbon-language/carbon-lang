#!/bin/sh

LLFILE=`echo $1 | sed -e 's/\.c/.ll/g'`

clang -c -S -emit-llvm -O0 $1 -o ${LLFILE}

opt -correlated-propagation -mem2reg -instcombine -loop-simplify -indvars \
-instnamer ${LLFILE} -S -o ${LLFILE}
