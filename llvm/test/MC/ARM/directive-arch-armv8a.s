@ Test the .arch directive for armv8-a

@ This test case will check the default .ARM.attributes value for the
@ armv8-a architecture when using the armv8a alias.

@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=asm \
@ RUN:   | FileCheck %s --check-prefix=CHECK-ASM
@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=obj \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CHECK-OBJ

	.syntax	unified
	.arch	armv8a

@ CHECK-ASM: 	.arch	armv8-a

@ CHECK-OBJ:    Name: .ARM.attributes
@ CHECK-OBJ:    Type: SHT_ARM_ATTRIBUTES (0x70000003)
@ CHECK-OBJ:    Flags [ (0x0)
@ CHECK-OBJ:    ]
@ CHECK-OBJ:    Address: 0x0
@ CHECK-OBJ:    Offset: 0x34
@ CHECK-OBJ:    Size: 33
@ CHECK-OBJ:    Link: 0
@ CHECK-OBJ:    Info: 0
@ CHECK-OBJ:    AddressAlignment: 1
@ CHECK-OBJ:    EntrySize: 0
@ CHECK-OBJ:    SectionData (
@ CHECK-OBJ:      0000: 41200000 00616561 62690001 16000000  |A ...aeabi......|
@ CHECK-OBJ:      0010: 05382D41 00060E07 41080109 022A0144  |.8-A....A....*.D|
@ CHECK-OBJ:      0020: 03                                   |.|
@ CHECK-OBJ:    )
