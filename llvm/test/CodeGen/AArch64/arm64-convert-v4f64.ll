; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -march=arm64 | FileCheck %s


define <4 x i16> @fptosi_v4f64_to_v4i16(<4 x double>* %ptr) {
; CHECK: fptosi_v4f64_to_v4i16
; CHECK-DAG: fcvtzs  v[[LHS:[0-9]+]].2d, v1.2d
; CHECK-DAG: fcvtzs  v[[RHS:[0-9]+]].2d, v0.2d
; CHECK-DAG: xtn  v[[LHS_NA:[0-9]+]].2s, v[[LHS]].2d
; CHECK-DAG: xtn  v[[RHS_NA:[0-9]+]].2s, v[[RHS]].2d
; CHECK:     uzp1  v0.4h, v[[RHS_NA]].4h, v[[LHS_NA]].4h
  %tmp1 = load <4 x double>, <4 x double>* %ptr
  %tmp2 = fptosi <4 x double> %tmp1 to <4 x i16>
  ret <4 x i16> %tmp2
}

define <8 x i8> @fptosi_v4f64_to_v4i8(<8 x double>* %ptr) {
; CHECK: fptosi_v4f64_to_v4i8
; CHECK-DAG:  fcvtzs  v[[CONV3:[0-9]+]].2d, v3.2d
; CHECK-DAG:  fcvtzs  v[[CONV2:[0-9]+]].2d, v2.2d
; CHECK-DAG:  fcvtzs  v[[CONV1:[0-9]+]].2d, v1.2d
; CHECK-DAG:  fcvtzs  v[[CONV0:[0-9]+]].2d, v0.2d
; CHECK-DAG:  xtn  v[[NA3:[0-9]+]].2s, v[[CONV3]].2d
; CHECK-DAG:  xtn  v[[NA2:[0-9]+]].2s, v[[CONV2]].2d
; CHECK-DAG:  xtn  v[[NA1:[0-9]+]].2s, v[[CONV1]].2d
; CHECK-DAG:  xtn  v[[NA0:[0-9]+]].2s, v[[CONV0]].2d
; CHECK-DAG:  uzp1  v[[TMP1:[0-9]+]].4h, v[[CONV2]].4h, v[[CONV3]].4h
; CHECK-DAG:  uzp1  v[[TMP2:[0-9]+]].4h, v[[CONV0]].4h, v[[CONV1]].4h
; CHECK:      uzp1  v0.8b, v[[TMP2]].8b, v[[TMP1]].8b
  %tmp1 = load <8 x double>, <8 x double>* %ptr
  %tmp2 = fptosi <8 x double> %tmp1 to <8 x i8>
  ret <8 x i8> %tmp2
}

