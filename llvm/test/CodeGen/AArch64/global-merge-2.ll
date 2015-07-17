; RUN: llc %s -mtriple=aarch64-none-linux-gnu -aarch64-global-merge -global-merge-on-external -o - | FileCheck %s
; RUN: llc %s -mtriple=aarch64-linux-gnuabi -aarch64-global-merge -global-merge-on-external -o - | FileCheck %s
; RUN: llc %s -mtriple=aarch64-apple-ios -aarch64-global-merge -global-merge-on-external -o - | FileCheck %s --check-prefix=CHECK-APPLE-IOS

@x = global i32 0, align 4
@y = global i32 0, align 4
@z = global i32 0, align 4

define void @f1(i32 %a1, i32 %a2) {
;CHECK-APPLE-IOS-LABEL: _f1:
;CHECK-APPLE-IOS-NOT: adrp
;CHECK-APPLE-IOS: adrp	x8, __MergedGlobals_x@PAGE
;CHECK-APPLE-IOS: add	x8, x8, __MergedGlobals_x@PAGEOFF
;CHECK-APPLE-IOS-NOT: adrp
  store i32 %a1, i32* @x, align 4
  store i32 %a2, i32* @y, align 4
  ret void
}

define void @g1(i32 %a1, i32 %a2) {
;CHECK-APPLE-IOS-LABEL: _g1:
;CHECK-APPLE-IOS: adrp	x8, __MergedGlobals_x@PAGE
;CHECK-APPLE-IOS: add	x8, x8, __MergedGlobals_x@PAGEOFF
;CHECK-APPLE-IOS-NOT: adrp
  store i32 %a1, i32* @y, align 4
  store i32 %a2, i32* @z, align 4
  ret void
}

;CHECK:	.type	_MergedGlobals_x,@object // @_MergedGlobals_x
;CHECK:	.globl	_MergedGlobals_x
;CHECK:	.align	3
;CHECK: _MergedGlobals_x:
;CHECK:	.size	_MergedGlobals_x, 12

;CHECK:	.globl	x
;CHECK: x = _MergedGlobals_x
;CHECK: .size x, 4
;CHECK:	.globl	y
;CHECK: y = _MergedGlobals_x+4
;CHECK: .size y, 4
;CHECK:	.globl	z
;CHECK: z = _MergedGlobals_x+8
;CHECK: .size z, 4

;CHECK-APPLE-IOS: .globl	__MergedGlobals_x       ; @_MergedGlobals_x
;CHECK-APPLE-IOS: .zerofill __DATA,__common,__MergedGlobals_x,12,3

;CHECK-APPLE-IOS: .globl	_x
;CHECK-APPLE-IOS: _x = __MergedGlobals_x
;CHECK-APPLE-IOS: .globl	_y
;CHECK-APPLE-IOS: _y = __MergedGlobals_x+4
;CHECK-APPLE-IOS: .globl	_z
;CHECK-APPLE-IOS: _z = __MergedGlobals_x+8
;CHECK-APPLE-IOS: .subsections_via_symbols
