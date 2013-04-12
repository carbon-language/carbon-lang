// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s | FileCheck %s

// Test that like gnu as we create text, data and bss by default. Also test
// that shstrtab, symtab and strtab are listed in that order.

// CHECK:        Section {
// CHECK:          Name: .text
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_EXECINSTR
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .data
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .bss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .shstrtab
// CHECK-NEXT:     Type: SHT_STRTAB
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 44
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .symtab
// CHECK-NEXT:     Type: SHT_SYMTAB
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 96
// CHECK-NEXT:     Link: 6
// CHECK-NEXT:     Info: 4
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 24
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .strtab
// CHECK-NEXT:     Type: SHT_STRTAB
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 1
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
