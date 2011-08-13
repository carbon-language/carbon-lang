; RUN: llc %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a9 -O0
; The following test is supposed to produce a VMOVQQQQ pseudo instruction.
; Make sure that it gets expanded; otherwise, the compile fails when trying
; to print the pseudo-instruction.

define void @test_vmovqqqq_pseudo() nounwind ssp {
entry:
  %vld3_lane = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3lane.v8i16(i8* undef, <8 x i16> undef, <8 x i16> undef, <8 x i16> zeroinitializer, i32 7, i32 2)
  store { <8 x i16>, <8 x i16>, <8 x i16> } %vld3_lane, { <8 x i16>, <8 x i16>, <8 x i16> }* undef
  ret void
}

declare { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3lane.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32) nounwind readonly
