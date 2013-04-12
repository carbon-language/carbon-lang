// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sr -t | FileCheck %s

// When doing a rename, all the checks for where the relocation should go
// should be performed with the original symbol. Only if we decide to relocate
// with the symbol we should then use the renamed one.

// This is a regression test for a bug where we used bar5@@@zed when deciding
// if we should relocate with the symbol or with the section and we would then
// not produce a relocation with .text.

defined1:
defined3:
        .symver defined3, bar5@@@zed
        .long defined3

        .global defined1

// Section 1 is .text
// CHECK:        Section {
// CHECK:          Index: 1
// CHECK-NEXT:     Name: .text
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_EXECINSTR
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:       0x0 R_X86_64_32 .text 0x0
// CHECK-NEXT:     ]
// CHECK-NEXT:   }

// Symbol 2 is section 1
// CHECK:        Symbol {
// CHECK:          Name: .text (0)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Section
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text (0x1)
// CHECK-NEXT:   }
