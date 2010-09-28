; RUN: llc < %s -mtriple=arm-apple-darwin -mcpu=cortex-a9 | FileCheck %s
; rdar://8402126
; Make sure if-converter is not predicating vldmia and ldmia. These are
; micro-coded and would have long issue latency even if predicated on
; false predicate.

%0 = type { float, float, float, float }
%pln = type { %vec, float }
%vec = type { [4 x float] }

define arm_aapcs_vfpcc float @aaa(%vec* nocapture %ustart, %vec* nocapture %udir, %vec* nocapture %vstart, %vec* nocapture %vdir, %vec* %upoint, %vec* %vpoint) {
; CHECK: aaa:
; CHECK: vldr.32
; CHECK-NOT: vldrne
; CHECK-NOT: vpopne
; CHECK-NOT: popne
; CHECK: vpop
; CHECK: pop
entry:
  br i1 undef, label %bb81, label %bb48

bb48:                                             ; preds = %entry
  %0 = call arm_aapcs_vfpcc  %0 @bbb(%pln* undef, %vec* %vstart, %vec* undef) nounwind ; <%0> [#uses=0]
  ret float 0.000000e+00

bb81:                                             ; preds = %entry
  ret float 0.000000e+00
}

declare arm_aapcs_vfpcc %0 @bbb(%pln* nocapture, %vec* nocapture, %vec* nocapture) nounwind
