! RUN: not llvm-mc %s -arch=sparc   -show-encoding 2>&1 | FileCheck %s
! RUN: not llvm-mc %s -arch=sparcv9 -show-encoding 2>&1 | FileCheck %s

! Test the lower and upper bounds of 'set'
        ! CHECK: argument must be between
        set -2147483649, %o1
        ! CHECK: argument must be between
        set 4294967296, %o1
