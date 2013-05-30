// RUN: llvm-mc -g -triple  i686-pc-linux-gnu %s -filetype=obj -o - | llvm-readobj -r | FileCheck %s


// Test that on ELF:
// 1. the debug info has a relocation to debug_abbrev and one to to debug_line.
// 2. the debug_aranges has relocations to text and debug_line.


    .text
    .globl foo
    .type foo, @function
    .align 4
foo:
    ret
    .size foo, .-foo

// CHECK:      Relocations [
// CHECK:        Section ({{[^ ]+}}) .rel.debug_info {
// CHECK-NEXT:     0x6 R_386_32 .debug_abbrev 0x0
// CHECK-NEXT:     0xC R_386_32 .debug_line 0x0
// CHECK:        }
// CHECK-NEXT:   Section ({{[^ ]+}}) .rel.debug_aranges {
// CHECK-NEXT:     0x6 R_386_32 .debug_info 0x0
// CHECK-NEXT:     0x10 R_386_32 .text 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]
