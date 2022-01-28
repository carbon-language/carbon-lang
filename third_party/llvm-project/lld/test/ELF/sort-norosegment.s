# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

# RUN: ld.lld --hash-style=sysv --no-rosegment -o %t %t.o -shared
# RUN: llvm-readelf -S %t | FileCheck %s

# CHECK:      .dynsym  {{.*}}   A
# CHECK-NEXT: .hash    {{.*}}   A
# CHECK-NEXT: .dynstr  {{.*}}   A
# CHECK-NEXT: .text    {{.*}}   AX
# CHECK-NEXT: .dynamic {{.*}}  WA
# CHECK-NEXT: foo      {{.*}}  WA

.section foo, "aw"
.byte 0
