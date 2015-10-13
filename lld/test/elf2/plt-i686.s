// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld2 -shared %t2.o -o %t2.so
// RUN: ld.lld2 %t.o %t2.so -o %t
// RUN: llvm-readobj -s -r %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s
// REQUIRES: x86

// CHECK:      Name: .plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x11010
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 48
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 16

// CHECK:      Name: .got.plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x12058
// CHECK-NEXT: Offset: 0x2058
// CHECK-NEXT: Size: 20
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 4
// CHECK-NEXT: EntrySize: 0

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rel.plt {
// CHECK-NEXT:     0x12064 R_386_JUMP_SLOT bar 0x0
// CHECK-NEXT:     0x12068 R_386_JUMP_SLOT zed 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Unfortunately FileCheck can't do math, so we have to check for explicit
// values:

// 16 is the size of PLT[0]
// (0x11010 + 16) - (0x11000 + 1) - 4 = 27
// (0x11010 + 16) - (0x11005 + 1) - 4 = 22
// (0x11020 + 16) - (0x1100a + 1) - 4 = 33

// DISASM:      _start:
// DISASM-NEXT:   11000:  e9 1b 00 00 00  jmp  27
// DISASM-NEXT:   11005:  e9 16 00 00 00  jmp  22
// DISASM-NEXT:   1100a:  e9 21 00 00 00  jmp  33

// 0x12064 - 0x11020 - 6 = 4158
// 0x12068 - 0x11030 - 6 = 4146
// 0x11010 - 0x1102b - 5 = -32
// 0x11010 - 0x1103b - 5 = -48
// DISASM:      Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:   11010:  ff 35 4a 10 00 00 pushl 4170
// DISASM-NEXT:   11016:  ff 25 4c 10 00 00 jmpl *4172
// DISASM-NEXT:   1101c:  00 00  addb %al, (%eax)
// DISASM-NEXT:   1101e:  00 00  addb %al, (%eax)
// DISASM-NEXT:   11020:  ff 25 3e 10 00 00 jmpl *4158
// DISASM-NEXT:   11026:  68 00 00 00 00 pushl $0
// DISASM-NEXT:   1102b:  e9 e0 ff ff ff jmp -32
// DISASM-NEXT:   11030:  ff 25 32 10 00 00 jmpl *4146
// DISASM-NEXT:   11036:  68 01 00 00 00 pushl $1
// DISASM-NEXT:   1103b:  e9 d0 ff ff ff jmp -48
   
   
.global _start
_start:
  jmp bar@PLT
  jmp bar@PLT
  jmp zed@PLT
