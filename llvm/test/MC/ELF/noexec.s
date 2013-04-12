// RUN: llvm-mc -mc-no-exec-stack -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -t | FileCheck  %s

// CHECK:        Section {
// CHECK:          Index: 4
// CHECK-NEXT:     Name: .note.GNU-stack
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }

// CHECK:        Symbol {
// CHECK:          Name: .note.GNU-stack (0)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Section
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .note.GNU-stack (0x4)
// CHECK-NEXT:   }
