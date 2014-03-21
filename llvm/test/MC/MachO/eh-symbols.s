// RUN: llvm-mc -filetype=obj -triple i686-apple-darwin %s  -o - | llvm-readobj -t | FileCheck %s

// Make sure that the exception handling data has the same visibility as the
// function it's generated for.

	.private_extern	_main
	.globl	_main
_main:
	.cfi_startproc
	retl
	.cfi_endproc

"_-[NSString(local) isNullOrNil]":
	.cfi_startproc
	retl
	.cfi_endproc

// CHECK: Name: _-[NSString(local) isNullOrNil].eh

// CHECK:       Name: _main
// CHECK-NEXT:  PrivateExtern

// CHECK:       Name: _main.eh
// CHECK-NEXT:  PrivateExtern

