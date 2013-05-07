; RUN: llc -march=hexagon -mcpu=hexagonv5 -hexagon-small-data-threshold=0 < %s | FileCheck %s

; Check that CONST32/CONST64 instructions are 'not' generated when
; small-data-threshold is set to 0.

; with immediate value.
@a = external global i32
@b = external global i32
@la = external global i64
@lb = external global i64

define void @test1() nounwind {
; CHECK-NOT: CONST32 
entry:
  store i32 12345670, i32* @a, align 4
  store i32 12345670, i32* @b, align 4
  ret void
}

define void @test2() nounwind {
; CHECK-NOT: CONST64
entry:
  store i64 1234567890123, i64* @la, align 8
  store i64 1234567890123, i64* @lb, align 8
  ret void
}
