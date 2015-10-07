// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux -save-temp-labels %s -o %t
// RUN: ld.lld2 -discard-none -shared %t -o %t2
// RUN: llvm-readobj -s -sd -t %t2 | FileCheck %s
// REQUIRES: x86

.text
.Lmyvar:
.Lmyothervar:

// CHECK:   Section {
// CHECK:     Name: .strtab
// CHECK-NEXT:     Type: SHT_STRTAB
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address:
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 002E7465 7874002E 62737300 2E64796E  |..text..bss..dyn|
// CHECK-NEXT:       0010: 73747200 2E4C6D79 76617200 2E4C6D79  |str..Lmyvar..Lmy|
// CHECK-NEXT:       0020: 6F746865 72766172 002E6479 6E73796D  |othervar..dynsym|
// CHECK-NEXT:       0030: 002E6861 7368002E 64796E61 6D696300  |..hash..dynamic.|
// CHECK-NEXT:       0040: 2E737472 74616200 2E73796D 74616200  |.strtab..symtab.|
// CHECK-NEXT:       0050: 2E646174 6100                        |.data.|
// CHECK-NEXT:     )
// CHECK-NEXT:   }

// CHECK:   Symbol {
// CHECK-NEXT:     Name:
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: .Lmyothervar
// CHECK-NEXT:     Value: 0x102C
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: .Lmyvar
// CHECK-NEXT:     Value: 0x102C
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
