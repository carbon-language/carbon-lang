// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld2 -shared %t -o %tout
// RUN: llvm-readobj -sections -relocations %tout | FileCheck %s
// RUN: llvm-objdump -d %tout | FileCheck %s --check-prefix=DIS

  leaq  a@tlsld(%rip), %rdi
  callq __tls_get_addr@PLT
  leaq  b@tlsld(%rip), %rdi
  callq __tls_get_addr@PLT

  .global	a
	.section	.tbss,"awT",@nobits
  .align 4
a:
	.long	0
  
  .global	b
	.section	.tbss,"awT",@nobits
  .align 4
b:
	.long	0

// Get the address of the got, and check that it has two entries.

// CHECK:      Sections [
// CHECK:          Name: .got
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x20D0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 16

// CHECK:      Relocations [
// CHECK:        Section ({{.+}}) .rela.dyn {
// CHECK-NEXT:     0x20D0 R_X86_64_DTPMOD64 - 0x0
// CHECK-NEXT:   }

// 4297 = (0x20D0 + -4) - (0x1000 + 3) // PC relative offset to got entry.

// DIS:      Disassembly of section .text:
// DIS-NEXT: .text:
// DIS-NEXT:     1000: {{.+}} leaq    4297(%rip), %rdi
// DIS-NEXT:     1007: {{.+}} callq
// DIS-NEXT:     100c: {{.+}} leaq    4285(%rip), %rdi
