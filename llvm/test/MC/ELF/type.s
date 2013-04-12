// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -t | FileCheck %s

// Test that both % and @ are accepted.
        .global foo
        .type foo,%function
foo:

        .global bar
        .type bar,@object
bar:

// Test that gnu_unique_object is accepted.
        .type zed,@gnu_unique_object

obj:
        .global obj
        .type obj,@object
        .type obj,@notype

func:
        .global func
        .type func,@function
        .type func,@object

ifunc:
        .global ifunc
        .type ifunc,@gnu_indirect_function

tls:
        .global tls
        .type tls,@tls_object
        .type tls,@gnu_indirect_function

// CHECK:        Symbol {
// CHECK:          Name: bar
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text (0x1)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Function
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text (0x1)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: func
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Function
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text (0x1)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: ifunc
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: GNU_IFunc
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text (0x1)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: obj
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text (0x1)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: tls
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text (0x1)
// CHECK-NEXT:   }
