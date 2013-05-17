
# RUN: llvm-mc -triple powerpc64-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -s -sd | FileCheck %s

# This checks that fixups that can be resolved within the same
# object file are applied correctly.

.text

addi 1, 1, target
addis 1, 1, target

.set target, 0x1234

.data

.quad v1
.word v2
.short v3
.byte v4

.set v1, 0x123456789abcdef0
.set v2, 0x87654321
.set v3, 0xbeef
.set v4, 0x42

# CHECK:       Section {
# CHECK:         Name: .text
# CHECK-NEXT:    Type: SHT_PROGBITS
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      SHF_ALLOC
# CHECK-NEXT:      SHF_EXECINSTR
# CHECK-NEXT:    ]
# CHECK-NEXT:    Address: 0x0
# CHECK-NEXT:    Offset:
# CHECK-NEXT:    Size: 8
# CHECK-NEXT:    Link: 0
# CHECK-NEXT:    Info: 0
# CHECK-NEXT:    AddressAlignment: 4
# CHECK-NEXT:    EntrySize: 0
# CHECK-NEXT:    SectionData (
# CHECK-NEXT:      0000: 38211234 3C211234
# CHECK-NEXT:    )
# CHECK-NEXT:  }

# CHECK:        Section {
# CHECK:          Name: .data
# CHECK-NEXT:     Type: SHT_PROGBITS
# CHECK-NEXT:     Flags [
# CHECK-NEXT:       SHF_ALLOC
# CHECK-NEXT:       SHF_WRITE
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset:
# CHECK-NEXT:     Size: 15
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 4
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:       0000: 12345678 9ABCDEF0 87654321 BEEF42
# CHECK-NEXT:     )
# CHECK-NEXT:   }

