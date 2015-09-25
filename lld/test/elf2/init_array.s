// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %t2
// RUN: llvm-readobj -symbols -sections %t2 | FileCheck %s
// RUN: llvm-objdump -d %t2 | FileCheck --check-prefix=DISASM %s
// REQUIRES: x86

.globl _start
_start:
  call __init_array_start
  call __init_array_end


.section .init_array,"aw",@init_array
  .quad 0


// CHECK:      Name: .init_array
// CHECK-NEXT: Type: SHT_INIT_ARRAY
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x12000
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 8

// CHECK:        Name: __init_array_end
// CHECK-NEXT:   Value: 0x12008
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .init_array
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: __init_array_start
// CHECK-NEXT:   Value: 0x12000
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .init_array
// CHECK-NEXT: }

// 0x12000 - (0x11000 + 5) = 4091
// 0x12008 - (0x11005 + 5) = 4094
// DISASM:      _start:
// DISASM-NEXT:   11000:  e8 fb 0f 00 00  callq  4091
// DISASM-NEXT:   11005:  e8 fe 0f 00 00  callq  4094
