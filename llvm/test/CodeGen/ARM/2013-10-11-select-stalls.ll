; REQUIRES: asserts
; RUN: llc < %s -mtriple=thumbv7-apple-ios -stats 2>&1 | not grep "Number of pipeline stalls"
; Evaluate the two vld1.8 instructions in separate MBB's,
; instead of stalling on one and conditionally overwriting its result.

define <16 x i8> @multiselect(i32 %avail, i8* %foo, i8* %bar) {
entry:
  %vld1 = call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* %foo, i32 1)
  %vld2 = call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* %bar, i32 1)
  %and = and i32 %avail, 1
  %tobool = icmp eq i32 %and, 0
  %retv = select i1 %tobool, <16 x i8> %vld1, <16 x i8> %vld2
  ret <16 x i8> %retv
}

declare <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* , i32 )
