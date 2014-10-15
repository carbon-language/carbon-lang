// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -t -r --expand-relocs | FileCheck %s

.section foo, "aG", @progbits, f1, comdat
.section foo, "G", @progbits, f2, comdat
.section bar
.long foo

// Test that the relocation points to the first section foo.

// The first seciton foo has index 6
// CHECK:      Section {
// CHECK:        Index:   6
// CHECK-NEXT:   Name:    foo (28)
// CHECK-NEXT:   Type:    SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x202)
// CHECK-NEXT:     SHF_ALLOC (0x2)
// CHECK-NEXT:     SHF_GROUP (0x200)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address:         0x0
// CHECK-NEXT:   Offset:  0x50
// CHECK-NEXT:   Size:    0
// CHECK-NEXT:   Link:    0
// CHECK-NEXT:   Info:    0
// CHECK-NEXT:   AddressAlignment:        1
// CHECK-NEXT:   EntrySize:       0
// CHECK-NEXT: }
// CHECK-NEXT: Section {
// CHECK-NEXT:   Index:   7
// CHECK-NEXT:   Name:    foo (28)
// CHECK-NEXT:   Type:    SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x200)
// CHECK-NEXT:     SHF_GROUP (0x200)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address:         0x0
// CHECK-NEXT:   Offset:  0x50
// CHECK-NEXT:   Size:    0
// CHECK-NEXT:   Link:    0
// CHECK-NEXT:   Info:    0
// CHECK-NEXT:   AddressAlignment:        1
// CHECK-NEXT:   EntrySize:       0
// CHECK-NEXT: }

// The relocation points to symbol 6
// CHECK:      Relocations [
// CHECK-NEXT:   Section (9) .relabar {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset:  0x0
// CHECK-NEXT:       Type:    R_X86_64_32 (10)
// CHECK-NEXT:       Symbol:  foo (6)
// CHECK-NEXT:       Addend:  0x0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]


// The symbol 6 corresponds section 6
// CHECK: Symbols [

// symbol 0
// CHECK-NOT: Name
// CHECK: Name:

// symbol 1
// CHECK-NOT: Name
// CHECK: Name:    f1

// symbol 2
// CHECK-NOT: Name
// CHECK: Name:    f2

// symbol 3
// CHECK-NOT: Name
// CHECK: Name:    .text

// symbol 4
// CHECK-NOT: Name
// CHECK: Name:    .data

// symbol 5
// CHECK-NOT: Name
// CHECK: Name:    .bss

// symbol 6
// CHECK-NOT: Name
// CHECK: Name:    foo
// CHECK: Section: foo (0x6)

// symbol 7
// CHECK-NOT: Name
// CHECK: Name:    foo
// CHECK: Section: foo (0x7)
