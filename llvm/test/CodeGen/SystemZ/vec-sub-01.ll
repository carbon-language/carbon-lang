; Test vector subtraction.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 subtraction.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vsb %v24, %v26, %v28
; CHECK: br %r14
  %ret = sub <16 x i8> %val1, %val2
  ret <16 x i8> %ret
}

; Test a v8i16 subtraction.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vsh %v24, %v26, %v28
; CHECK: br %r14
  %ret = sub <8 x i16> %val1, %val2
  ret <8 x i16> %ret
}

; Test a v4i32 subtraction.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vsf %v24, %v26, %v28
; CHECK: br %r14
  %ret = sub <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v2i64 subtraction.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK: vsg %v24, %v26, %v28
; CHECK: br %r14
  %ret = sub <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}

; Test a v4f32 subtraction, as an example of an operation that needs to be
; scalarized and reassembled.  At present there's an unnecessary move that
; could be avoided with smarter ordering.  It also isn't important whether
; the VSLDBs use the result of the VLRs or use %v24 and %v26 directly.
define <4 x float> @f5(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f5:
; CHECK-DAG: vlr %v[[A1:[0-5]]], %v24
; CHECK-DAG: vlr %v[[A2:[0-5]]], %v26
; CHECK-DAG: vrepf %v[[B1:[0-5]]], %v[[A1]], 1
; CHECK-DAG: vrepf %v[[B2:[0-5]]], %v[[A2]], 1
; CHECK-DAG: vrepf %v[[C1:[0-5]]], %v[[A1]], 2
; CHECK-DAG: vrepf %v[[C2:[0-5]]], %v[[A2]], 2
; CHECK-DAG: vrepf %v[[D1:[0-5]]], %v[[A1]], 3
; CHECK-DAG: vrepf %v[[D2:[0-5]]], %v[[A2]], 3
; CHECK-DAG: ler %f[[A1copy:[0-5]]], %f[[A1]]
; CHECK-DAG: sebr %f[[A1copy]], %f[[A2]]
; CHECK-DAG: sebr %f[[B1]], %f[[B2]]
; CHECK-DAG: sebr %f[[C1]], %f[[C2]]
; CHECK-DAG: sebr %f[[D1]], %f[[D2]]
; CHECK-DAG: vmrhf [[HIGH:%v[0-9]+]], %v[[A1copy]], %v[[B1]]
; CHECK-DAG: vmrhf [[LOW:%v[0-9]+]], %v[[C1]], %v[[D1]]
; CHECK: vmrhg %v24, [[HIGH]], [[LOW]]
; CHECK: br %r14
  %ret = fsub <4 x float> %val1, %val2
  ret <4 x float> %ret
}

; Test a v2f64 subtraction.
define <2 x double> @f6(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) {
; CHECK-LABEL: f6:
; CHECK: vfsdb %v24, %v26, %v28
; CHECK: br %r14
  %ret = fsub <2 x double> %val1, %val2
  ret <2 x double> %ret
}
