;RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s

;CHECK: IMAGE_SAMPLE
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE_C
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE_C
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE_C
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE_C
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE_C
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE_C
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE
;CHECK-NEXT: S_WAITCNT 1792
;CHECK-NEXT: IMAGE_SAMPLE

define void @test() {
   %res1 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 1)
   %res2 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 2)
   %res3 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 3)
   %res4 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 4)
   %res5 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 5)
   %res6 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 6)
   %res7 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 7)
   %res8 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 8)
   %res9 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 9)
   %res10 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 10)
   %res11 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 11)
   %res12 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 12)
   %res13 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 13)
   %res14 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 14)
   %res15 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 15)
   %res16 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> undef,
      <8 x i32> undef, <4 x i32> undef, i32 16)
   ret void
}

declare <4 x float> @llvm.SI.sample.(i32, <4 x i32>, <8 x i32>, <4 x i32>, i32)
