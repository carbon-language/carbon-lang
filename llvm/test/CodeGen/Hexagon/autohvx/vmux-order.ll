; RUN: llc -march=hexagon < %s | FileCheck %s

; The generated code isn't great, the vdeltas are no-ops (controls are all 0).
; Check for the correct order of vmux operands as is, when the code improves
; fix the checking as well.

; CHECK-DAG: v[[V0:[0-9]+]] = vdelta(v0,v{{[0-9]+}})
; CHECK-DAG: v[[V1:[0-9]+]] = vdelta(v1,v{{[0-9]+}})
; CHECK: vmux(q{{[0-3]+}},v[[V1]],v[[V0]])
define <16 x i32> @fred(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %p = shufflevector <16 x i32> %v0, <16 x i32> %v1, <16 x i32> <i32 0,i32 17,i32 2,i32 19,i32 4,i32 21,i32 6,i32 23,i32 8,i32 25,i32 10,i32 27,i32 12,i32 29,i32 14,i32 31>
  ret <16 x i32> %p
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
