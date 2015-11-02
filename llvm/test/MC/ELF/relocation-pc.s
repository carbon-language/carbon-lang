// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sr | FileCheck  %s

// Test that we produce the correct relocation.

        loope   0                 # R_X86_64_PC8
        jmp     -256              # R_X86_64_PC32

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .rela.text
// CHECK-NEXT:     Type: SHT_RELA
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 48
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 24
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:       0x1 R_X86_64_PC8 - 0xFFFFFFFFFFFFFFFF
// CHECK-NEXT:       0x3 R_X86_64_PC32 - 0xFFFFFFFFFFFFFEFC
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
