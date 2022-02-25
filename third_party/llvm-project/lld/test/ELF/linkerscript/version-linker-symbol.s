# REQUIRES: x86

# RUN: echo "VER1 { global: _end; foo ; local: * ; } ;" > %t.script
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld --version-script %t.script -shared %t.o -o %t.so
# RUN: llvm-readobj --dyn-syms %t.so | FileCheck %s

# CHECK:      Name: foo@@VER1
# CHECK-NEXT: Value: 0
# CHECK-NEXT: Size: 0
# CHECK-NEXT: Binding: Global
# CHECK-NEXT: Type: None
# CHECK-NEXT: Other: 0
# CHECK-NEXT: Section: .text

# CHECK:      Name: _end@@VER1
# CHECK-NEXT: Value: 0
# CHECK-NEXT: Size: 0
# CHECK-NEXT: Binding: Global
# CHECK-NEXT: Type: None
# CHECK-NEXT: Other: 0
# CHECK-NEXT: Section: .data

.global foo
foo:
        .data
        .quad _end
        .quad foo
