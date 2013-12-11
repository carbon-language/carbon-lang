@ Test the .arch directive for iwmmxt2

@ This test case will check the default .ARM.attributes value for the
@ iwmmxt2 architecture.

@ RUN: llvm-mc < %s -arch=arm -filetype=asm \
@ RUN:   | FileCheck %s --check-prefix=CHECK-ASM
@ RUN: llvm-mc < %s -arch=arm -filetype=obj \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CHECK-OBJ

	.syntax	unified
	.arch	iwmmxt2

@ CHECK-ASM: 	.arch	iwmmxt2

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
@ CHECK-OBJ:      0010: 0549574D 4D585432 00060408 0109010B  |.IWMMXT2........|
@ CHECK-OBJ:      0020: 02                                   |.|
@ CHECK-OBJ:    )
