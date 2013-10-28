@ RUN: llvm-mc < %s -triple armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s

@ CHECK: Name: .ARM.attribute
@ CHECK: SectionData (

@ <format-version>
@ CHECK: 41

@ <section-length>
@ CHECK: 1A0000 00

@ <vendor-name> "aeabi\0"
@ CHECK: 616561 626900

@ <file-tag>
@ CHECK: 01

@ <size>
@ CHECK: 10000000

	.cpu	cortex-a8
@ CHECK: 05
@ CHECK: 434F52 5445582D 413800

@ CHECK: )
