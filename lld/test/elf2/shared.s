// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t.o
// RUN: lld -flavor gnu2 %t.o %p/Inputs/i686-simple-library.so -o %t
// RUN: llvm-readobj --program-headers --dynamic-table -t -s %t | FileCheck %s
// REQUIRES: x86

// CHECK:        Name: .dynamic
// CHECK-NEXT:   Type: SHT_DYNAMIC
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_ALLOC
// CHECK-NEXT:     SHF_WRITE
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: [[ADDR:.*]]
// CHECK-NEXT:   Offset: [[OFFSET:.*]]
// CHECK-NEXT:   Size: [[SIZE:.*]]
// CHECK-NEXT:   Link: [[DYNSTR:.*]]
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: [[ALIGN:.*]]
// CHECK-NEXT:   EntrySize: 8
// CHECK-NEXT: }

// CHECK:        Index: [[DYNSTR]]
// CHECK-NEXT:   Name: .dynstr
// CHECK-NEXT:   Type: SHT_STRTAB
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_ALLOC
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: [[DYNSTRADDR:.*]]
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size:
// CHECK-NEXT:   Link: 0
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: 1
// CHECK-NEXT:   EntrySize: 0
// CHECK-NEXT: }


// CHECK:      Symbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name:
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: _start
// CHECK-NEXT:     Value: 0x1000
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: bar
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Function
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK:      DynamicSection [
// CHECK-NEXT:   Tag        Type                 Name/Value
// CHECK-NEXT:   0x00000005 STRTAB               [[DYNSTRADDR]]
// CHECK-NEXT:   0x0000000A STRSZ
// CHECK-NEXT:   0x00000001 NEEDED               SharedLibrary ({{.*}}/Inputs/i686-simple-library.so)
// CHECK-NEXT:   0x00000000 NULL                 0x0
// CHECK-NEXT: ]

// CHECK:      ProgramHeader {
// CHECK:        Type: PT_DYNAMIC
// CHECK-NEXT:   Offset: [[OFFSET]]
// CHECK-NEXT:   VirtualAddress: [[ADDR]]
// CHECK-NEXT:   PhysicalAddress: [[ADDR]]
// CHECK-NEXT:   FileSize: [[SIZE]]
// CHECK-NEXT:   MemSize: [[SIZE]]
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     PF_R
// CHECK-NEXT:     PF_W
// CHECK-NEXT:   ]
// CHECK-NEXT:   Alignment: [[ALIGN]]
// CHECK-NEXT: }

.global _start
_start:
.long bar
