@ RUN: llvm-mc < %s -triple armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s

@ CHECK: Name: .ARM.attribute
@ CHECK: SectionData (

@ <format-version>
@ CHECK: 41

@ <section-length>
@ CHECK: 250000 00

@ <vendor-name> "aeabi\0"
@ CHECK: 616561 626900

@ <file-tag>
@ CHECK: 01

@ <size>
@ CHECK: 1B000000

@ <attribute>*

	.eabi_attribute 6, 10
@ CHECK: 060A

	.eabi_attribute 7, 65
@ CHECK: 0741

	.eabi_attribute 8, 1
@ CHECK: 0801

	.eabi_attribute 9, 2
@ CHECK: 0902

	.eabi_attribute 10, 3
@ CHECK: 0A03

	.eabi_attribute 12, 1
@ CHECK: 0C01

	.eabi_attribute 20, 1
@ CHECK: 1401

	.eabi_attribute 21, 1
@ CHECK: 1501

	.eabi_attribute 23, 3
@ CHECK: 1703

	.eabi_attribute 24, 1
@ CHECK: 1801

	.eabi_attribute 25, 1
@ CHECK: 1901
@ CHECK: )
