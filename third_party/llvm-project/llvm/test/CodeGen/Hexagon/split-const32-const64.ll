; RUN: llc -march=hexagon -hexagon-small-data-threshold=0 < %s | FileCheck %s

; Check that CONST32/CONST64 instructions are 'not' generated when the
; small data threshold is set to 0.

@a = external global i32
@b = external global i32
@la = external global i64
@lb = external global i64

; CHECK-LABEL: test1:
; CHECK-NOT: CONST32
define void @test1() nounwind {
entry:
  br label %block
block:
  store i32 12345670, i32* @a, align 4
  %q = ptrtoint i8* blockaddress (@test1, %block) to i32
  store i32 %q, i32* @b, align 4
  ret void
}

; CHECK-LABEL: test2:
; CHECK-NOT: CONST64
define void @test2() nounwind {
entry:
  store i64 1234567890123, i64* @la, align 8
  store i64 1234567890123, i64* @lb, align 8
  ret void
}
