// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj  -t -r --expand-relocs | FileCheck %s

// Test that we can forward reference a section.

mov .rodata, %rsi
.section .rodata

// CHECK:Relocations [
// CHECK:  Section (2) .rela.text {
// CHECK:    Relocation {
// CHECK:      Offset: 0x4
// CHECK:      Type: R_X86_64_32S (11)
// CHECK:      Symbol: .rodata
// CHECK:      Addend: 0x0
// CHECK:    }
// CHECK:  }
// CHECK:]

// There is only one .rodata symbol

// CHECK:Symbols [
// CHECK-NOT:    Name: .rodata
// CHECK:        Name: .rodata
// CHECK-NEXT:   Value: 0x0
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local (0x0)
// CHECK-NEXT:   Type: Section (0x3)
// CHECK-NOT:    Name: .rodata
