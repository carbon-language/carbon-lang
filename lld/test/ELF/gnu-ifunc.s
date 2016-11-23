// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld -static %t.o -o %tout
// RUN: llvm-objdump -d %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readobj -r -symbols -sections %tout | FileCheck %s
// REQUIRES: x86

// CHECK:      Sections [
// CHECK:       Section {
// CHECK:       Index: 1
// CHECK-NEXT:  Name: .rela.plt
// CHECK-NEXT:  Type: SHT_RELA
// CHECK-NEXT:  Flags [
// CHECK-NEXT:    SHF_ALLOC
// CHECK-NEXT:  ]
// CHECK-NEXT:  Address: [[RELA:.*]]
// CHECK-NEXT:  Offset: 0x158
// CHECK-NEXT:  Size: 48
// CHECK-NEXT:  Link: 6
// CHECK-NEXT:  Info: 0
// CHECK-NEXT:  AddressAlignment: 8
// CHECK-NEXT:  EntrySize: 24
// CHECK-NEXT: }
// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.plt {
// CHECK-NEXT:     0x12018 R_X86_64_IRELATIVE
// CHECK-NEXT:     0x12020 R_X86_64_IRELATIVE
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK:      Symbols [
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name:
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: Undefined
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: __rela_iplt_end
// CHECK-NEXT:    Value: 0x10188
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other [
// CHECK-NEXT:      STV_HIDDEN
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: .rela.plt
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: __rela_iplt_start
// CHECK-NEXT:    Value: [[RELA]]
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other [
// CHECK-NEXT:      STV_HIDDEN
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: .rela.plt
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: _start
// CHECK-NEXT:    Value: 0x11002
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: bar
// CHECK-NEXT:    Value: 0x11001
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: foo
// CHECK-NEXT:    Value: 0x11000
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT: ]

// DISASM:      Disassembly of section .text:
// DISASM-NEXT: foo:
// DISASM-NEXT:    11000: {{.*}} retq
// DISASM:      bar:
// DISASM-NEXT:    11001: {{.*}} retq
// DISASM:      _start:
// DISASM-NEXT:    11002: {{.*}} callq 41
// DISASM-NEXT:    11007: {{.*}} callq 52
// DISASM-NEXT:    1100c: {{.*}} movl $65880, %edx
// DISASM-NEXT:    11011: {{.*}} movl $65928, %edx
// DISASM-NEXT:    11016: {{.*}} movl $65929, %edx
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:    11020: {{.*}} pushq 4066(%rip)
// DISASM-NEXT:    11026: {{.*}} jmpq *4068(%rip)
// DISASM-NEXT:    1102c: {{.*}} nopl (%rax)
// DISASM-NEXT:    11030: {{.*}} jmpq *4066(%rip)
// DISASM-NEXT:    11036: {{.*}} pushq $0
// DISASM-NEXT:    1103b: {{.*}} jmp -32
// DISASM-NEXT:    11040: {{.*}} jmpq *4058(%rip)
// DISASM-NEXT:    11046: {{.*}} pushq $1
// DISASM-NEXT:    1104b: {{.*}} jmp -48

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 ret

.type bar STT_GNU_IFUNC
.globl bar
bar:
 ret

.globl _start
_start:
 call foo
 call bar
 movl $__rela_iplt_start,%edx
 movl $__rela_iplt_end,%edx
 movl $__rela_iplt_end + 1,%edx
