; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin9 -mattr=sse41 | FileCheck %s -check-prefix=X32
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin9 -mattr=sse41 | FileCheck %s -check-prefix=X64


define <4 x i32> @pinsrd(i32 %s, <4 x i32> %tmp) nounwind {
        %tmp1 = insertelement <4 x i32> %tmp, i32 %s, i32 1
        ret <4 x i32> %tmp1
; X32: pinsrd:
; X32:    pinsrd $1, 4(%esp), %xmm0

; X64: pinsrd:
; X64:    pinsrd $1, %edi, %xmm0
}

define <16 x i8> @pinsrb(i8 %s, <16 x i8> %tmp) nounwind {
        %tmp1 = insertelement <16 x i8> %tmp, i8 %s, i32 1
        ret <16 x i8> %tmp1
; X32: pinsrb:
; X32:    pinsrb $1, 4(%esp), %xmm0

; X64: pinsrb:
; X64:    pinsrb $1, %edi, %xmm0
}
