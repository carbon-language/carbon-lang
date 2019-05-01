# RUN: llvm-mc -triple i686-windows -g -filetype obj -o - %s \
# RUN:   | llvm-readobj -S -t | FileCheck %s

	.section .rdata

	.align 8
	.global data
data:
	.quad 0

# CHECK: Sections [
# CHECK:  Section {
# CHECK:    Name: .text
# CHECK:  }
# CHECK: ]

