; RUN: llc -march=hexagon -disable-hexagon-misched < %s | FileCheck %s
; Check that we generate dual stores in one packet in V4

; CHECK: memw(r{{[0-9]+}}{{ *}}+{{ *}}#{{[0-9]+}}){{ *}}=
; CHECK-NEXT: memw(r{{[0-9]+}}{{ *}}+{{ *}}#{{[0-9]+}}){{ *}}=

define i32 @main(i32 %v, i32* %p1, i32* %p2) nounwind {
entry:
  store i32 %v, i32* %p1, align 4
  store i32 %v, i32* %p2, align 4
  ret i32 0
}
