// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -t | FileCheck %s

// Test that all symbols are of type STT_TLS.

	leaq	foo1@TLSGD(%rip), %rdi
        leaq    foo2@GOTTPOFF(%rip), %rdi
        leaq    foo3@TLSLD(%rip), %rdi
	.long foo4@GOTTPOFF
	.long foo5@TLSLD
	.long foo6@TLSGD
	.section	.zed,"awT",@progbits
foobar:
	.long	43

// CHECK:        Symbol {
// CHECK:          Name: foobar (31)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .zed (0x5)
// CHECK-NEXT:   }

// CHECK:        Symbol {
// CHECK:          Name: foo1 (1)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo2 (6)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo3 (11)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo4 (16)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo5 (21)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo6 (26)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: TLS
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: (0x0)
// CHECK-NEXT:   }
