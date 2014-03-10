; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon
; Bug: i8 type in FRP8 register but not registering with register class causes segmentation fault.
; Fix: Removed i8 type from FPR8 register class.

define void @test_concatvector_v8i8() {
entry.split:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry.split
  unreachable

if.end:                                           ; preds = %entry.split
  br i1 undef, label %if.then9, label %if.end18

if.then9:                                         ; preds = %if.end
  unreachable

if.end18:                                         ; preds = %if.end
  br label %for.body

for.body:                                         ; preds = %for.inc, %if.end18
  br i1 false, label %if.then30, label %for.inc

if.then30:                                        ; preds = %for.body
  unreachable

for.inc:                                          ; preds = %for.body
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  br label %for.body77

for.body77:                                       ; preds = %for.body77, %for.end
  br i1 undef, label %for.end106, label %for.body77

for.end106:                                       ; preds = %for.body77
  br i1 undef, label %for.body130.us.us, label %stmt.for.body130.us.us

stmt.for.body130.us.us:                     ; preds = %stmt.for.body130.us.us, %for.end106
  %_p_splat.us = shufflevector <1 x i8> zeroinitializer, <1 x i8> undef, <8 x i32> zeroinitializer
  store <8 x i8> %_p_splat.us, <8 x i8>* undef, align 1
  br label %stmt.for.body130.us.us

for.body130.us.us:                                ; preds = %for.body130.us.us, %for.end106
  br label %for.body130.us.us
}

declare <1 x i16> @llvm.aarch64.neon.vuqrshrn.v1i16(<1 x i32>, i32)

define <8 x i16> @test_splat(i32 %l) nounwind {
; CHECK-LABEL: test_splat:
; CHECK: ret
  %lhs = insertelement <1 x i32> undef, i32 %l, i32 0
  %shift = tail call <1 x i16> @llvm.aarch64.neon.vuqrshrn.v1i16(<1 x i32> %lhs, i32 11)
  %vec = shufflevector <1 x i16> %shift, <1 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %vec
}


define <8 x i16> @test_notsplat(<8 x i16> %a, <8 x i16> %b, i32 %l) nounwind {
; CHECK-LABEL: test_notsplat:
; CHECK: ret
entry:
  %lhs = insertelement <1 x i32> undef, i32 %l, i32 0
  %shift = tail call <1 x i16> @llvm.aarch64.neon.vuqrshrn.v1i16(<1 x i32> %lhs, i32 11)
  %vec = shufflevector <1 x i16> %shift, <1 x i16> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i16> %vec
}
