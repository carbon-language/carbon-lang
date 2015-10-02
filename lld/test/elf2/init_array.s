// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2
// RUN: lld -flavor gnu2 %t2 -o t2.so -shared
// RUN: lld -flavor gnu2 %t t2.so -o %t2
// RUN: llvm-readobj -symbols -sections -dynamic-table %t2 | FileCheck %s
// RUN: llvm-objdump -d %t2 | FileCheck --check-prefix=DISASM %s
// REQUIRES: x86

.globl _start
_start:
  call __init_array_start
  call __init_array_end


.section .init_array,"aw",@init_array
  .quad 0

.section .preinit_array,"aw",@preinit_array
        .quad 0
        .byte 0

.section .fini_array,"aw",@fini_array
        .quad 0
        .short 0

// CHECK:      Name: .init_array
// CHECK-NEXT: Type: SHT_INIT_ARRAY
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: [[INIT_ADDR:.*]]
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: [[INIT_SIZE:.*]]


// CHECK:     Name: .preinit_array
// CHECK-NEXT: Type: SHT_PREINIT_ARRAY
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT:    ]
// CHECK-NEXT: Address: [[PREINIT_ADDR:.*]]
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: [[PREINIT_SIZE:.*]]


// CHECK:      Name: .fini_array
// CHECK-NEXT: Type: SHT_FINI_ARRAY
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: [[FINI_ADDR:.*]]
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: [[FINI_SIZE:.*]]

// CHECK:        Name: __init_array_end
// CHECK-NEXT:   Value: 0x13008
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .init_array
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: __init_array_start
// CHECK-NEXT:   Value: [[INIT_ADDR]]
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .init_array
// CHECK-NEXT: }


// CHECK: DynamicSection
// CHECK: PREINIT_ARRAY        [[PREINIT_ADDR]]
// CHECK: PREINIT_ARRAYSZ      [[PREINIT_SIZE]] (bytes)
// CHECK: INIT_ARRAY           [[INIT_ADDR]]
// CHECK: INIT_ARRAYSZ         [[INIT_SIZE]] (bytes)
// CHECK: FINI_ARRAY           [[FINI_ADDR]]
// CHECK: FINI_ARRAYSZ         [[FINI_SIZE]] (bytes)


// 0x13000 - (0x12000 + 5) = 4091
// 0x13008 - (0x12005 + 5) = 4094
// DISASM:      _start:
// DISASM-NEXT:   12000:  e8 fb 0f 00 00  callq  4091
// DISASM-NEXT:   12005:  e8 fe 0f 00 00  callq  4094
