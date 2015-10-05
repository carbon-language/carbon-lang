// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld2 -shared %t.o -o %t.so
// RUN: llvm-readobj -t -r -dyn-symbols %t.so | FileCheck %s

// Test that we create R_X86_64_RELATIVE relocations but don't put any
// symbols in the dynamic symbol table.

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     [[FOO_ADDR:.*]] R_X86_64_RELATIVE - [[FOO_ADDR]]
// CHECK-NEXT:     [[BAR_ADDR:.*]] R_X86_64_RELATIVE - [[BAR_ADDR]]
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK:      Symbols [
// CHECK:        Name: foo
// CHECK-NEXT:   Value: [[FOO_ADDR]]
// CHECK:        Name: bar
// CHECK-NEXT:   Value: [[BAR_ADDR]]
// CHECK:      ]

// CHECK:      DynamicSymbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: @ (0)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined
// CHECK-NEXT:   }
// CHECK-NEXT: ]

foo:
        .quad foo

        .hidden bar
        .global bar
bar:
        .quad bar
