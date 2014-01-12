@ Test the .arch directive for armv4

@ This test case will check the default .ARM.attributes value for the
@ armv4 architecture.

@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=asm \
@ RUN:   | FileCheck %s --check-prefix=CHECK-ASM
@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=obj \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CHECK-OBJ

	.syntax	unified
	.arch	armv4

@ CHECK-ASM: 	.arch	armv4

@ CHECK-OBJ:    Name: .ARM.attributes
@ CHECK-OBJ:    Type: SHT_ARM_ATTRIBUTES (0x70000003)
@ CHECK-OBJ:    Flags [ (0x0)
@ CHECK-OBJ:    ]
@ CHECK-OBJ:    Address: 0x0
@ CHECK-OBJ:    Offset: 0x{{[0-9A-F]*}}
@ CHECK-OBJ:    Size: 23
@ CHECK-OBJ:    Link: 0
@ CHECK-OBJ:    Info: 0
@ CHECK-OBJ:    AddressAlignment: 1
@ CHECK-OBJ:    EntrySize: 0
@ CHECK-OBJ:    SectionData (
@ CHECK-OBJ:      0000: 41160000 00616561 62690001 0C000000  |A....aeabi......|
@ CHECK-OBJ:      0010: 05340006 010801                      |.4.....|
@ CHECK-OBJ:    )


@ Check that multiplication is supported
	mul r4, r5, r6
	smull r4, r5, r6, r3
	umull r4, r5, r6, r3
	smlal r4, r5, r6, r3
	umlal r4, r5, r6, r3
