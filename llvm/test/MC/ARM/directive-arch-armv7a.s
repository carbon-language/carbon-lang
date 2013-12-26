@ Test the .arch directive for armv7-a

@ This test case will check the default .ARM.attributes value for the
@ armv7-a architecture when using the armv7a alias.

@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=asm \
@ RUN:   | FileCheck %s --check-prefix=CHECK-ASM
@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=obj \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CHECK-OBJ

	.syntax	unified
	.arch	armv7a

@ CHECK-ASM: 	.arch	armv7-a

@ CHECK-OBJ:    Name: .ARM.attributes
@ CHECK-OBJ:    Type: SHT_ARM_ATTRIBUTES (0x70000003)
@ CHECK-OBJ:    Flags [ (0x0)
@ CHECK-OBJ:    ]
@ CHECK-OBJ:    Address: 0x0
@ CHECK-OBJ:    Offset: 0x34
@ CHECK-OBJ:    Size: 29
@ CHECK-OBJ:    Link: 0
@ CHECK-OBJ:    Info: 0
@ CHECK-OBJ:    AddressAlignment: 1
@ CHECK-OBJ:    EntrySize: 0
@ CHECK-OBJ:    SectionData (
@ CHECK-OBJ:      0000: 411C0000 00616561 62690001 12000000  |A....aeabi......|
@ CHECK-OBJ:      0010: 05372D41 00060A07 41080109 02        |.7-A....A....|
@ CHECK-OBJ:    )
