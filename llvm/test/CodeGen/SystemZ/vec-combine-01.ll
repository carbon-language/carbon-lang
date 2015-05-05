; Test various target-specific DAG combiner patterns.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Check that an extraction followed by a truncation is effectively treated
; as a bitcast.
define void @f1(<4 x i32> %v1, <4 x i32> %v2, i8 *%ptr1, i8 *%ptr2) {
; CHECK-LABEL: f1:
; CHECK: vaf [[REG:%v[0-9]+]], %v24, %v26
; CHECK-DAG: vsteb [[REG]], 0(%r2), 3
; CHECK-DAG: vsteb [[REG]], 0(%r3), 15
; CHECK: br %r14
  %add = add <4 x i32> %v1, %v2
  %elem1 = extractelement <4 x i32> %add, i32 0
  %elem2 = extractelement <4 x i32> %add, i32 3
  %trunc1 = trunc i32 %elem1 to i8
  %trunc2 = trunc i32 %elem2 to i8
  store i8 %trunc1, i8 *%ptr1
  store i8 %trunc2, i8 *%ptr2
  ret void
}

; Test a case where a pack-type shuffle can be eliminated.
define i16 @f2(<4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3) {
; CHECK-LABEL: f2:
; CHECK-NOT: vpk
; CHECK-DAG: vaf [[REG1:%v[0-9]+]], %v24, %v26
; CHECK-DAG: vaf [[REG2:%v[0-9]+]], %v26, %v28
; CHECK-DAG: vlgvh {{%r[0-5]}}, [[REG1]], 3
; CHECK-DAG: vlgvh {{%r[0-5]}}, [[REG2]], 7
; CHECK: br %r14
  %add1 = add <4 x i32> %v1, %v2
  %add2 = add <4 x i32> %v2, %v3
  %shuffle = shufflevector <4 x i32> %add1, <4 x i32> %add2,
                           <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %bitcast = bitcast <4 x i32> %shuffle to <8 x i16>
  %elem1 = extractelement <8 x i16> %bitcast, i32 1
  %elem2 = extractelement <8 x i16> %bitcast, i32 7
  %res = add i16 %elem1, %elem2
  ret i16 %res
}

; ...and again in a case where there's also a splat and a bitcast.
define i16 @f3(<4 x i32> %v1, <4 x i32> %v2, <2 x i64> %v3) {
; CHECK-LABEL: f3:
; CHECK-NOT: vrepg
; CHECK-NOT: vpk
; CHECK-DAG: vaf [[REG:%v[0-9]+]], %v24, %v26
; CHECK-DAG: vlgvh {{%r[0-5]}}, [[REG]], 6
; CHECK-DAG: vlgvh {{%r[0-5]}}, %v28, 3
; CHECK: br %r14
  %add = add <4 x i32> %v1, %v2
  %splat = shufflevector <2 x i64> %v3, <2 x i64> undef,
                         <2 x i32> <i32 0, i32 0>
  %splatcast = bitcast <2 x i64> %splat to <4 x i32>
  %shuffle = shufflevector <4 x i32> %add, <4 x i32> %splatcast,
                           <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %bitcast = bitcast <4 x i32> %shuffle to <8 x i16>
  %elem1 = extractelement <8 x i16> %bitcast, i32 2
  %elem2 = extractelement <8 x i16> %bitcast, i32 7
  %res = add i16 %elem1, %elem2
  ret i16 %res
}

; ...and again with a merge low instead of a pack.
define i16 @f4(<4 x i32> %v1, <4 x i32> %v2, <2 x i64> %v3) {
; CHECK-LABEL: f4:
; CHECK-NOT: vrepg
; CHECK-NOT: vmr
; CHECK-DAG: vaf [[REG:%v[0-9]+]], %v24, %v26
; CHECK-DAG: vlgvh {{%r[0-5]}}, [[REG]], 6
; CHECK-DAG: vlgvh {{%r[0-5]}}, %v28, 3
; CHECK: br %r14
  %add = add <4 x i32> %v1, %v2
  %splat = shufflevector <2 x i64> %v3, <2 x i64> undef,
                         <2 x i32> <i32 0, i32 0>
  %splatcast = bitcast <2 x i64> %splat to <4 x i32>
  %shuffle = shufflevector <4 x i32> %add, <4 x i32> %splatcast,
                           <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  %bitcast = bitcast <4 x i32> %shuffle to <8 x i16>
  %elem1 = extractelement <8 x i16> %bitcast, i32 4
  %elem2 = extractelement <8 x i16> %bitcast, i32 7
  %res = add i16 %elem1, %elem2
  ret i16 %res
}

; ...and again with a merge high.
define i16 @f5(<4 x i32> %v1, <4 x i32> %v2, <2 x i64> %v3) {
; CHECK-LABEL: f5:
; CHECK-NOT: vrepg
; CHECK-NOT: vmr
; CHECK-DAG: vaf [[REG:%v[0-9]+]], %v24, %v26
; CHECK-DAG: vlgvh {{%r[0-5]}}, [[REG]], 2
; CHECK-DAG: vlgvh {{%r[0-5]}}, %v28, 3
; CHECK: br %r14
  %add = add <4 x i32> %v1, %v2
  %splat = shufflevector <2 x i64> %v3, <2 x i64> undef,
                         <2 x i32> <i32 0, i32 0>
  %splatcast = bitcast <2 x i64> %splat to <4 x i32>
  %shuffle = shufflevector <4 x i32> %add, <4 x i32> %splatcast,
                           <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %bitcast = bitcast <4 x i32> %shuffle to <8 x i16>
  %elem1 = extractelement <8 x i16> %bitcast, i32 4
  %elem2 = extractelement <8 x i16> %bitcast, i32 7
  %res = add i16 %elem1, %elem2
  ret i16 %res
}
