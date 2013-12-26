@ Test the .arch directive for armv7-m

@ This test case will check the default .ARM.attributes value for the
@ armv7-m architecture when using the armv7m alias.

@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=asm \
@ RUN:   | FileCheck %s --check-prefix=CHECK-ASM
@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=obj \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CHECK-OBJ

	.syntax	unified
	.arch	armv7m

@ CHECK-ASM: 	.arch	armv7-m

@ CHECK-OBJ:    Name: .ARM.attributes
@ CHECK-OBJ:    Type: SHT_ARM_ATTRIBUTES (0x70000003)
@ CHECK-OBJ:    Flags [ (0x0)
@ CHECK-OBJ:    ]
@ CHECK-OBJ:    Address: 0x0
@ CHECK-OBJ:    Offset: 0x34
@ CHECK-OBJ:    Size: 27
@ CHECK-OBJ:    Link: 0
@ CHECK-OBJ:    Info: 0
@ CHECK-OBJ:    AddressAlignment: 1
@ CHECK-OBJ:    EntrySize: 0
@ CHECK-OBJ:    SectionData (
@ CHECK-OBJ:      0000: 411A0000 00616561 62690001 10000000  |A....aeabi......|
@ CHECK-OBJ:      0010: 05372D4D 00060A07 4D0902             |.7-M....M..|
@ CHECK-OBJ:    )
