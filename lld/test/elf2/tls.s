// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld2 %t -o %tout
// RUN: llvm-readobj -sections -program-headers %tout | FileCheck %s

.global _start
_start:

	.section	.tbss,"awT",@nobits
	.long	0

	.section	.tdata,"awT",@progbits
	.long	1

	.section	.thread_bss,"awT",@nobits
	.long	0

	.section	.thread_data,"awT",@progbits
	.long	2

// CHECK:          Name: .tdata
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_TLS
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: [[TDATA_ADDR:0x.*]]
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: .thread_data
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_TLS
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address:
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: .tbss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_TLS
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: [[TBSS_ADDR:0x.*]]
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: .thread_bss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_TLS
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]

// 0x1100C = TBSS_ADDR + 4

// CHECK-NEXT:     Address: 0x1100C
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:   }

// Check that the TLS NOBITS sections weren't added to the R/W PT_LOAD's size.

// CHECK:      ProgramHeaders [
// CHECK:          Type: PT_LOAD
// CHECK:          Type: PT_LOAD
// CHECK:          FileSize: 8
// CHECK-NEXT:     MemSize: 8
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       PF_R
// CHECK-NEXT:       PF_W
// CHECK-NEXT:     ]
// CHECK:          Type: PT_TLS
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     VirtualAddress: [[TDATA_ADDR]]
// CHECK-NEXT:     PhysicalAddress: [[TDATA_ADDR]]
// CHECK-NEXT:     FileSize: 8
// CHECK-NEXT:     MemSize: 16
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       PF_R
// CHECK-NEXT:     ]
// CHECK-NEXT:     Alignment:
// CHECK-NEXT:   }
