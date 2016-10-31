# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:                  .text : { bar = ALIGNOF(.text); *(.text) } \
# RUN:                };" > %t.script
# RUN: ld.lld -o %t.so --script %t.script %t.o -shared
# RUN: llvm-readobj -t %t.so | FileCheck %s

# CHECK:      Symbol {
# CHECK:        Name: bar
# CHECK-NEXT:   Value: 0x4
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: None
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Absolute
# CHECK-NEXT: }
