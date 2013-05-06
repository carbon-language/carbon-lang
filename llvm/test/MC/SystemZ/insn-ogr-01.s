# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ogr	%r0, %r0                # encoding: [0xb9,0x81,0x00,0x00]
#CHECK: ogr	%r0, %r15               # encoding: [0xb9,0x81,0x00,0x0f]
#CHECK: ogr	%r15, %r0               # encoding: [0xb9,0x81,0x00,0xf0]
#CHECK: ogr	%r7, %r8                # encoding: [0xb9,0x81,0x00,0x78]

	ogr	%r0,%r0
	ogr	%r0,%r15
	ogr	%r15,%r0
	ogr	%r7,%r8
