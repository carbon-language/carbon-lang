; RUN: llc %s -mtriple=aarch64-none-linux-gnu -aarch64-global-merge -global-merge-on-external -o - | FileCheck %s
; RUN: llc %s -mtriple=aarch64-linux-gnuabi -aarch64-global-merge -global-merge-on-external -o - | FileCheck %s
; RUN: llc %s -mtriple=aarch64-apple-ios -aarch64-global-merge -global-merge-on-external -o - | FileCheck %s --check-prefix=CHECK-APPLE-IOS

@x = global [1000 x i32] zeroinitializer, align 1
@y = global [1000 x i32] zeroinitializer, align 1
@z = internal global i32 1, align 4

define void @f1(i32 %a1, i32 %a2, i32 %a3) {
;CHECK-APPLE-IOS: adrp	x8, __MergedGlobals_x@PAGE
;CHECK-APPLE-IOS-NOT: adrp
;CHECK-APPLE-IOS: add	x8, x8, __MergedGlobals_x@PAGEOFF
;CHECK-APPLE-IOS: adrp	x9, __MergedGlobals_y@PAGE
;CHECK-APPLE-IOS: add	x9, x9, __MergedGlobals_y@PAGEOFF
  %x3 = getelementptr inbounds [1000 x i32], [1000 x i32]* @x, i32 0, i64 3
  %y3 = getelementptr inbounds [1000 x i32], [1000 x i32]* @y, i32 0, i64 3
  store i32 %a1, i32* %x3, align 4
  store i32 %a2, i32* %y3, align 4
  store i32 %a3, i32* @z, align 4
  ret void
}

;CHECK:	.type	_MergedGlobals_x,@object // @_MergedGlobals_x
;CHECK: .globl	_MergedGlobals_x
;CHECK: .align	4
;CHECK: _MergedGlobals_x:
;CHECK: .size	_MergedGlobals_x, 4004

;CHECK: .type	_MergedGlobals_y,@object // @_MergedGlobals_y
;CHECK: .globl	_MergedGlobals_y
;CHECK: _MergedGlobals_y:
;CHECK: .size	_MergedGlobals_y, 4000

;CHECK-APPLE-IOS: .globl	__MergedGlobals_x       ; @_MergedGlobals_x
;CHECK-APPLE-IOS: .align	4
;CHECK-APPLE-IOS:  __MergedGlobals_x:
;CHECK-APPLE-IOS: .long 1
;CHECK-APPLE-IOS: .space	4000

;CHECK-APPLE-IOS: .globl	__MergedGlobals_y       ; @_MergedGlobals_y
;CHECK-APPLE-IOS: .zerofill __DATA,__common,__MergedGlobals_y,4000,4

;CHECK:	.globl	x
;CHECK: x = _MergedGlobals_x+4
;CHECK:	.globl	y
;CHECK: y = _MergedGlobals_y

;CHECK-APPLE-IOS:.globl	_x
;CHECK-APPLE-IOS: _x = __MergedGlobals_x+4
;CHECK-APPLE-IOS:.globl	_y
;CHECK-APPLE-IOS: _y = __MergedGlobals_y
