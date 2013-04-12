// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sd -sr | FileCheck  %s

        call bar
bar:

// CHECK:        Section {
// CHECK:          Name: .text
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [ (0x6)
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_EXECINSTR
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 5
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:     ]
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: E8000000 00
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK: shstrtab
