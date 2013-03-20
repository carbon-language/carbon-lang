; RUN: llc < %s -mtriple=i686-pc-linux -mcpu=corei7-avx | FileCheck %s

define <8 x i32> @shiftInput___vyuunu(<8 x i32> %input, i32 %shiftval, <8 x i32> %__mask) nounwind {
allocas:
  %smear.0 = insertelement <8 x i32> undef, i32 %shiftval, i32 0
  %smear.1 = insertelement <8 x i32> %smear.0, i32 %shiftval, i32 1
  %smear.2 = insertelement <8 x i32> %smear.1, i32 %shiftval, i32 2
  %smear.3 = insertelement <8 x i32> %smear.2, i32 %shiftval, i32 3
  %smear.4 = insertelement <8 x i32> %smear.3, i32 %shiftval, i32 4
  %smear.5 = insertelement <8 x i32> %smear.4, i32 %shiftval, i32 5
  %smear.6 = insertelement <8 x i32> %smear.5, i32 %shiftval, i32 6
  %smear.7 = insertelement <8 x i32> %smear.6, i32 %shiftval, i32 7
  %bitop = lshr <8 x i32> %input, %smear.7
  ret <8 x i32> %bitop
}

; CHECK: shiftInput___vyuunu
; CHECK: psrld
; CHECK: psrld
; CHECK: ret

define <8 x i32> @shiftInput___canonical(<8 x i32> %input, i32 %shiftval, <8 x i32> %__mask) nounwind {
allocas:
  %smear.0 = insertelement <8 x i32> undef, i32 %shiftval, i32 0
  %smear.7 = shufflevector <8 x i32> %smear.0, <8 x i32> undef, <8 x i32> zeroinitializer
  %bitop = lshr <8 x i32> %input, %smear.7
  ret <8 x i32> %bitop
}

; CHECK: shiftInput___canonical
; CHECK: psrld
; CHECK: psrld
; CHECK: ret

define <4 x i64> @shiftInput___64in32bitmode(<4 x i64> %input, i64 %shiftval, <4 x i64> %__mask) nounwind {
allocas:
  %smear.0 = insertelement <4 x i64> undef, i64 %shiftval, i32 0
  %smear.7 = shufflevector <4 x i64> %smear.0, <4 x i64> undef, <4 x i32> zeroinitializer
  %bitop = lshr <4 x i64> %input, %smear.7
  ret <4 x i64> %bitop
}

; CHECK: shiftInput___64in32bitmode
; CHECK: psrlq
; CHECK: psrlq
; CHECK: ret
