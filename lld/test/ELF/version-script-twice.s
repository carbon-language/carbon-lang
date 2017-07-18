# REQUIRES: x86

# RUN: echo "FBSD_1.2 {};" > %t.ver
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so --version-script=%t.ver
# RUN: llvm-nm --dynamic %t.so | FileCheck %s

        .weak	openat
openat:
openat@@FBSD_1.2 = openat

# CHECK: 0000000000001000 W openat
# CHECK-NEXT: 0000000000001000 W openat
