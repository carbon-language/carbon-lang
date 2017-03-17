# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { foo = ADDR(.text) + 1; bar = 1 + ADDR(.text); };" > %t.script
# RUN: ld.lld -o %t.so --script %t.script %t.o -shared
# RUN: llvm-readobj -t -s %t.so | FileCheck %s

# CHECK:      Section {
# CHECK:        Index:
# CHECK:        Name: .text
# CHECK-NEXT:   Type:
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:     SHF_EXECINSTR
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x0

# CHECK:      Symbol {
# CHECK:        Name: foo
# CHECK-NEXT:   Value: 0x1
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: None
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: .text
# CHECK-NEXT: }

# CHECK:      Symbol {
# CHECK:        Name: bar
# CHECK-NEXT:   Value: 0x1
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: None
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: .text
# CHECK-NEXT: }
