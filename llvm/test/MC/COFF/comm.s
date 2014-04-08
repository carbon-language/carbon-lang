// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj -t | FileCheck %s

.lcomm _a,4,4
.comm	_b, 4


// CHECK:       Symbol {
// CHECK:         Name: _a
// CHECK-NEXT:    Value:
// CHECK-NEXT:    Section: .bss
// CHECK-NEXT:    BaseType: Null
// CHECK-NEXT:    ComplexType: Null
// CHECK-NEXT:    StorageClass: Static
// CHECK-NEXT:    AuxSymbolCount: 0
// CHECK-NEXT:  }

// CHECK:       Symbol {
// CHECK:         Name: _b
// CHECK-NEXT:    Value: 4
// CHECK-NEXT:    Section:  (0)
// CHECK-NEXT:    BaseType: Null
// CHECK-NEXT:    ComplexType: Null
// CHECK-NEXT:    StorageClass: External
// CHECK-NEXT:    AuxSymbolCount: 0
// CHECK-NEXT:  }
