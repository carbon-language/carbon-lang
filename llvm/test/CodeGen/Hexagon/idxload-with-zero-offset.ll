; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that we generate load instruction with (base + register offset << 0)

; load word

define i32 @load_w(i32* nocapture %a, i32 %n) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memw(r{{[0-9]+}}+r{{[0-9]+}}<<#0)
entry:
  %tmp = shl i32 %n, 4
  %scevgep9 = getelementptr i32* %a, i32 %tmp
  %val = load i32* %scevgep9, align 4
  ret i32 %val
}

; load unsigned half word

define i16 @load_uh(i16* nocapture %a, i32 %n) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memuh(r{{[0-9]+}}+r{{[0-9]+}}<<#0)
entry:
  %tmp = shl i32 %n, 4
  %scevgep9 = getelementptr i16* %a, i32 %tmp
  %val = load i16* %scevgep9, align 2
  ret i16 %val
}

; load signed half word

define i32 @load_h(i16* nocapture %a, i32 %n) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memh(r{{[0-9]+}}+r{{[0-9]+}}<<#0)
entry:
  %tmp = shl i32 %n, 4
  %scevgep9 = getelementptr i16* %a, i32 %tmp
  %val = load i16* %scevgep9, align 2
  %conv = sext i16 %val to i32
  ret i32 %conv
}

; load unsigned byte

define i8 @load_ub(i8* nocapture %a, i32 %n) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memub(r{{[0-9]+}}+r{{[0-9]+}}<<#0)
entry:
  %tmp = shl i32 %n, 4
  %scevgep9 = getelementptr i8* %a, i32 %tmp
  %val = load i8* %scevgep9, align 1
  ret i8 %val
}

; load signed byte

define i32 @foo_2(i8* nocapture %a, i32 %n) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memb(r{{[0-9]+}}+r{{[0-9]+}}<<#0)
entry:
  %tmp = shl i32 %n, 4
  %scevgep9 = getelementptr i8* %a, i32 %tmp
  %val = load i8* %scevgep9, align 1
  %conv = sext i8 %val to i32
  ret i32 %conv
}

; load doubleword

define i64 @load_d(i64* nocapture %a, i32 %n) nounwind {
; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}memd(r{{[0-9]+}}+r{{[0-9]+}}<<#0)
entry:
  %tmp = shl i32 %n, 4
  %scevgep9 = getelementptr i64* %a, i32 %tmp
  %val = load i64* %scevgep9, align 8
  ret i64 %val
}
