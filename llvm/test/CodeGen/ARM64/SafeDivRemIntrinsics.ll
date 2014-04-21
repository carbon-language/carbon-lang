; RUN: llc < %s -march=arm64 | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

%divovf8  = type { i8, i1 }
%divovf16 = type { i16, i1 }
%divovf32 = type { i32, i1 }
%divovf64 = type { i64, i1 }

declare %divovf8  @llvm.safe.sdiv.i8(i8, i8) nounwind readnone
declare %divovf16 @llvm.safe.sdiv.i16(i16, i16) nounwind readnone
declare %divovf32 @llvm.safe.sdiv.i32(i32, i32) nounwind readnone
declare %divovf64 @llvm.safe.sdiv.i64(i64, i64) nounwind readnone

declare %divovf8  @llvm.safe.srem.i8(i8, i8) nounwind readnone
declare %divovf16 @llvm.safe.srem.i16(i16, i16) nounwind readnone
declare %divovf32 @llvm.safe.srem.i32(i32, i32) nounwind readnone
declare %divovf64 @llvm.safe.srem.i64(i64, i64) nounwind readnone

declare %divovf8  @llvm.safe.udiv.i8(i8, i8) nounwind readnone
declare %divovf16 @llvm.safe.udiv.i16(i16, i16) nounwind readnone
declare %divovf32 @llvm.safe.udiv.i32(i32, i32) nounwind readnone
declare %divovf64 @llvm.safe.udiv.i64(i64, i64) nounwind readnone

declare %divovf8  @llvm.safe.urem.i8(i8, i8) nounwind readnone
declare %divovf16 @llvm.safe.urem.i16(i16, i16) nounwind readnone
declare %divovf32 @llvm.safe.urem.i32(i32, i32) nounwind readnone
declare %divovf64 @llvm.safe.urem.i64(i64, i64) nounwind readnone

; CHECK-LABEL: sdiv8
; CHECK: sdiv{{[ 	]}}
define %divovf8 @sdiv8(i8 %x, i8 %y) {
entry:
  %divr = call %divovf8 @llvm.safe.sdiv.i8(i8 %x, i8 %y)
  ret %divovf8 %divr
}
; CHECK-LABEL: sdiv16
; CHECK: sdiv{{[ 	]}}
define %divovf16 @sdiv16(i16 %x, i16 %y) {
entry:
  %divr = call %divovf16 @llvm.safe.sdiv.i16(i16 %x, i16 %y)
  ret %divovf16 %divr
}
; CHECK-LABEL: sdiv32
; CHECK: sdiv{{[ 	]}}
define %divovf32 @sdiv32(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.sdiv.i32(i32 %x, i32 %y)
  ret %divovf32 %divr
}
; CHECK-LABEL: sdiv64
; CHECK: sdiv{{[ 	]}}
define %divovf64 @sdiv64(i64 %x, i64 %y) {
entry:
  %divr = call %divovf64 @llvm.safe.sdiv.i64(i64 %x, i64 %y)
  ret %divovf64 %divr
}
; CHECK-LABEL: udiv8
; CHECK: udiv{{[ 	]}}
define %divovf8 @udiv8(i8 %x, i8 %y) {
entry:
  %divr = call %divovf8 @llvm.safe.udiv.i8(i8 %x, i8 %y)
  ret %divovf8 %divr
}
; CHECK-LABEL: udiv16
; CHECK: udiv{{[ 	]}}
define %divovf16 @udiv16(i16 %x, i16 %y) {
entry:
  %divr = call %divovf16 @llvm.safe.udiv.i16(i16 %x, i16 %y)
  ret %divovf16 %divr
}
; CHECK-LABEL: udiv32
; CHECK: udiv{{[ 	]}}
define %divovf32 @udiv32(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.udiv.i32(i32 %x, i32 %y)
  ret %divovf32 %divr
}
; CHECK-LABEL: udiv64
; CHECK: udiv{{[ 	]}}
define %divovf64 @udiv64(i64 %x, i64 %y) {
entry:
  %divr = call %divovf64 @llvm.safe.udiv.i64(i64 %x, i64 %y)
  ret %divovf64 %divr
}
; CHECK-LABEL: srem8
; CHECK: sdiv{{[ 	]}}
; CHECK: msub{{[ 	]}}
define %divovf8 @srem8(i8 %x, i8 %y) {
entry:
  %remr = call %divovf8 @llvm.safe.srem.i8(i8 %x, i8 %y)
  ret %divovf8 %remr
}
; CHECK-LABEL: srem16
; CHECK: sdiv{{[ 	]}}
; CHECK: msub{{[ 	]}}
define %divovf16 @srem16(i16 %x, i16 %y) {
entry:
  %remr = call %divovf16 @llvm.safe.srem.i16(i16 %x, i16 %y)
  ret %divovf16 %remr
}
; CHECK-LABEL: srem32
; CHECK: sdiv{{[ 	]}}
; CHECK: msub{{[ 	]}}
define %divovf32 @srem32(i32 %x, i32 %y) {
entry:
  %remr = call %divovf32 @llvm.safe.srem.i32(i32 %x, i32 %y)
  ret %divovf32 %remr
}
; CHECK-LABEL: srem64
; CHECK: sdiv{{[ 	]}}
; CHECK: msub{{[ 	]}}
define %divovf64 @srem64(i64 %x, i64 %y) {
entry:
  %remr = call %divovf64 @llvm.safe.srem.i64(i64 %x, i64 %y)
  ret %divovf64 %remr
}
; CHECK-LABEL: urem8
; CHECK: udiv{{[ 	]}}
; CHECK: msub{{[ 	]}}
define %divovf8 @urem8(i8 %x, i8 %y) {
entry:
  %remr = call %divovf8 @llvm.safe.urem.i8(i8 %x, i8 %y)
  ret %divovf8 %remr
}
; CHECK-LABEL: urem16
; CHECK: udiv{{[ 	]}}
; CHECK: msub{{[ 	]}}
define %divovf16 @urem16(i16 %x, i16 %y) {
entry:
  %remr = call %divovf16 @llvm.safe.urem.i16(i16 %x, i16 %y)
  ret %divovf16 %remr
}
; CHECK-LABEL: urem32
; CHECK: udiv{{[ 	]}}
; CHECK: msub{{[ 	]}}
define %divovf32 @urem32(i32 %x, i32 %y) {
entry:
  %remr = call %divovf32 @llvm.safe.urem.i32(i32 %x, i32 %y)
  ret %divovf32 %remr
}
; CHECK-LABEL: urem64
; CHECK: udiv{{[ 	]}}
; CHECK: msub{{[ 	]}}
define %divovf64 @urem64(i64 %x, i64 %y) {
entry:
  %remr = call %divovf64 @llvm.safe.urem.i64(i64 %x, i64 %y)
  ret %divovf64 %remr
}

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.5.0 "}
