# REQUIRES: x86
# RUN: llvm-mc --triple=x86_64-pc-linux --filetype=obj -o %t.o %s
# RUN: not ld.lld -z pac-plt -z force-bti %t.o -o /dev/null 2>&1 | FileCheck %s
#
## Check that we error if -z pac-plt and -z force-bti are used when target is not
## aarch64

# CHECK: error: -z pac-plt only supported on AArch64
# CHECK-NEXT: error: -z force-bti only supported on AArch64

        .globl start
start:  ret
