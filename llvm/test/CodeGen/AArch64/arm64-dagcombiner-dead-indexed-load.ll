; RUN: llc -mcpu=cyclone < %s | FileCheck %s

; r208640 broke ppc64/Linux self-hosting; xfailing while this is worked on.
; XFAIL: *

target datalayout = "e-i64:64-n32:64-S128"
target triple = "arm64-apple-ios"

%"struct.SU" = type { i32, %"struct.SU"*, i32*, i32, i32, %"struct.BO", i32, [5 x i8] }
%"struct.BO" = type { %"struct.RE" }

%"struct.RE" = type { i32, i32, i32, i32 }

; This is a read-modify-write of some bifields combined into an i48.  It gets
; legalized into i32 and i16 accesses.  Only a single store of zero to the low
; i32 part should be live.

; CHECK-LABEL: test:
; CHECK-NOT: ldr
; CHECK: str wzr
; CHECK-NOT: str
define void @test(%"struct.SU"* nocapture %su) {
entry:
  %r1 = getelementptr inbounds %"struct.SU"* %su, i64 1, i32 5
  %r2 = bitcast %"struct.BO"* %r1 to i48*
  %r3 = load i48* %r2, align 8
  %r4 = and i48 %r3, -4294967296
  %r5 = or i48 0, %r4
  store i48 %r5, i48* %r2, align 8

  ret void
}
