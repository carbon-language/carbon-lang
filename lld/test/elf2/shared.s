// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: lld -flavor gnu2 -shared %t2.o -o %t2.so
// RUN: lld -flavor gnu2 -dynamic-linker /lib64/ld-linux-x86-64.so.2 -rpath foo -rpath bar %t.o %t2.so -o %t
// RUN: llvm-readobj --program-headers --dynamic-table -t -s -dyn-symbols -section-data %t | FileCheck %s
// REQUIRES: x86

// CHECK:        Name: .interp
// CHECK-NEXT:   Type: SHT_PROGBITS
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_ALLOC
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: [[INTERPADDR:.*]]
// CHECK-NEXT:   Offset: [[INTERPOFFSET:.*]]
// CHECK-NEXT:   Size: [[INTERPSIZE:.*]]
// CHECK-NEXT:   Link: 0
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: 1
// CHECK-NEXT:   EntrySize: 0
// CHECK-NEXT:   SectionData (
// CHECK-NEXT:     0000: 2F6C6962 36342F6C 642D6C69 6E75782D  |/lib64/ld-linux-|
// CHECK-NEXT:     0010: 7838362D 36342E73 6F2E3200           |x86-64.so.2.|
// CHECK-NEXT:   )
// CHECK-NEXT: }

// CHECK:        Name: .dynsym
// CHECK-NEXT:   Type: SHT_DYNSYM
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_ALLOC
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: [[DYNSYMADDR:.*]]
// CHECK-NEXT:   Offset: 0x3000
// CHECK-NEXT:   Size: 48
// CHECK-NEXT:   Link: [[DYNSTR:.*]]
// CHECK-NEXT:   Info: 1
// CHECK-NEXT:   AddressAlignment: 4
// CHECK-NEXT:   EntrySize: 16
// CHECK-NEXT:   SectionData (
// CHECK-NEXT:     0000:
// CHECK-NEXT:     0010:
// CHECK-NEXT:     0020:
// CHECK-NEXT:   )
// CHECK-NEXT: }

// CHECK:        Name: .dynamic
// CHECK-NEXT:   Type: SHT_DYNAMIC
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_ALLOC
// CHECK-NEXT:     SHF_WRITE
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: [[ADDR:.*]]
// CHECK-NEXT:   Offset: [[OFFSET:.*]]
// CHECK-NEXT:   Size: [[SIZE:.*]]
// CHECK-NEXT:   Link: [[DYNSTR]]
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: [[ALIGN:.*]]
// CHECK-NEXT:   EntrySize: 8
// CHECK-NEXT:   SectionData (
// CHECK-NEXT:     0000:
// CHECK-NEXT:     0010:
// CHECK-NEXT:     0020:
// CHECK-NEXT:   )
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
// CHECK-NEXT:   SectionData (
// CHECK:        )
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
// CHECK-NEXT:     Value: 0x11000
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

// CHECK:      DynamicSymbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: @ (0)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: _start@
// CHECK-NEXT:     Value: 0x11000
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Non
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: bar@
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
// CHECK-NEXT:   0x00000006 SYMTAB               [[DYNSYMADDR]]
// CHECK-NEXT:   0x00000005 STRTAB               [[DYNSTRADDR]]
// CHECK-NEXT:   0x0000000A STRSZ
// CHECK-NEXT:   0x0000001D RUNPATH              foo:bar
// CHECK-NEXT:   0x00000001 NEEDED               SharedLibrary ({{.*}}2.so)
// CHECK-NEXT:   0x00000000 NULL                 0x0
// CHECK-NEXT: ]

// CHECK:      ProgramHeaders [
// CHECK-NEXT:   ProgramHeader {
// CHECK-NEXT:   Type: PT_INTERP
// CHECK-NEXT:   Offset: [[INTERPOFFSET]]
// CHECK-NEXT:   VirtualAddress: [[INTERPADDR]]
// CHECK-NEXT:   PhysicalAddress: [[INTERPADDR]]
// CHECK-NEXT:   FileSize: [[INTERPSIZE]]
// CHECK-NEXT:   MemSize: [[INTERPSIZE]]
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     PF_R
// CHECK-NEXT:   ]
// CHECK-NEXT:   Alignment: 1
// CHECK-NEXT: }
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
