; RUN: llc < %s -mtriple=x86_64-linux-gnu | FileCheck %s

; fold (shl (zext (lshr (A, X))), X) -> (zext (shl (lshr (A, X)), X))

; Canolicalize the sequence shl/zext/lshr performing the zeroextend
; as the last instruction of the sequence.
; This will help DAGCombiner to identify and then fold the sequence
; of shifts into a single AND.
; This transformation is profitable if the shift amounts are the same
; and if there is only one use of the zext.

define i16 @fun1(i8 zeroext %v) {
entry:
  %shr = lshr i8 %v, 4
  %ext = zext i8 %shr to i16
  %shl = shl i16 %ext, 4
  ret i16 %shl
}

; CHECK-LABEL: @fun1
; CHECK: and
; CHECK-NOT: shr
; CHECK-NOT: shl
; CHECK: ret

define i32 @fun2(i8 zeroext %v) {
entry:
  %shr = lshr i8 %v, 4
  %ext = zext i8 %shr to i32
  %shl = shl i32 %ext, 4
  ret i32 %shl
}

; CHECK-LABEL: @fun2
; CHECK: and
; CHECK-NOT: shr
; CHECK-NOT: shl
; CHECK: ret

define i32 @fun3(i16 zeroext %v) {
entry:
  %shr = lshr i16 %v, 4
  %ext = zext i16 %shr to i32
  %shl = shl i32 %ext, 4
  ret i32 %shl
}

; CHECK-LABEL: @fun3
; CHECK: and
; CHECK-NOT: shr
; CHECK-NOT: shl
; CHECK: ret

define i64 @fun4(i8 zeroext %v) {
entry:
  %shr = lshr i8 %v, 4
  %ext = zext i8 %shr to i64
  %shl = shl i64 %ext, 4
  ret i64 %shl
}

; CHECK-LABEL: @fun4
; CHECK: and
; CHECK-NOT: shr
; CHECK-NOT: shl
; CHECK: ret

define i64 @fun5(i16 zeroext %v) {
entry:
  %shr = lshr i16 %v, 4
  %ext = zext i16 %shr to i64
  %shl = shl i64 %ext, 4
  ret i64 %shl
}

; CHECK-LABEL: @fun5
; CHECK: and
; CHECK-NOT: shr
; CHECK-NOT: shl
; CHECK: ret

define i64 @fun6(i32 zeroext %v) {
entry:
  %shr = lshr i32 %v, 4
  %ext = zext i32 %shr to i64
  %shl = shl i64 %ext, 4
  ret i64 %shl
}

; CHECK-LABEL: @fun6
; CHECK: and
; CHECK-NOT: shr
; CHECK-NOT: shl
; CHECK: ret

; Don't fold the pattern if we use arithmetic shifts.

define i64 @fun7(i8 zeroext %v) {
entry:
  %shr = ashr i8 %v, 4
  %ext = zext i8 %shr to i64
  %shl = shl i64 %ext, 4
  ret i64 %shl
}

; CHECK-LABEL: @fun7
; CHECK: sar
; CHECK: shl
; CHECK: ret

define i64 @fun8(i16 zeroext %v) {
entry:
  %shr = ashr i16 %v, 4
  %ext = zext i16 %shr to i64
  %shl = shl i64 %ext, 4
  ret i64 %shl
}

; CHECK-LABEL: @fun8
; CHECK: sar
; CHECK: shl
; CHECK: ret

define i64 @fun9(i32 zeroext %v) {
entry:
  %shr = ashr i32 %v, 4
  %ext = zext i32 %shr to i64
  %shl = shl i64 %ext, 4
  ret i64 %shl
}

; CHECK-LABEL: @fun9
; CHECK: sar
; CHECK: shl
; CHECK: ret

; Don't fold the pattern if there is more than one use of the
; operand in input to the shift left.

define i64 @fun10(i8 zeroext %v) {
entry:
  %shr = lshr i8 %v, 4
  %ext = zext i8 %shr to i64
  %shl = shl i64 %ext, 4
  %add = add i64 %shl, %ext
  ret i64 %add
}

; CHECK-LABEL: @fun10
; CHECK: shr
; CHECK: shl
; CHECK: ret

define i64 @fun11(i16 zeroext %v) {
entry:
  %shr = lshr i16 %v, 4
  %ext = zext i16 %shr to i64
  %shl = shl i64 %ext, 4
  %add = add i64 %shl, %ext
  ret i64 %add
}

; CHECK-LABEL: @fun11
; CHECK: shr
; CHECK: shl
; CHECK: ret

define i64 @fun12(i32 zeroext %v) {
entry:
  %shr = lshr i32 %v, 4
  %ext = zext i32 %shr to i64
  %shl = shl i64 %ext, 4
  %add = add i64 %shl, %ext
  ret i64 %add
}

; CHECK-LABEL: @fun12
; CHECK: shr
; CHECK: shl
; CHECK: ret

; PR17380
; Make sure that the combined dags are legal if we run the DAGCombiner after
; Legalization took place. The add instruction is redundant and increases by 
; one the number of uses of the zext. This prevents the transformation from
; firing before dags are legalized and optimized.
; Once the add is removed, the number of uses becomes one and therefore the
; dags are canonicalized. After Legalization, we need to make sure that the
; valuetype for the shift count is legal.
; Verify also that we correctly fold the shl-shr sequence into an 
; AND with bitmask.

define void @g(i32 %a) {
  %b = lshr i32 %a, 2
  %c = zext i32 %b to i64
  %d = add i64 %c, 1
  %e = shl i64 %c, 2
  tail call void @f(i64 %e)
  ret void
}

; CHECK-LABEL: @g
; CHECK-NOT: shr
; CHECK-NOT: shl
; CHECK: and
; CHECK-NEXT: jmp

declare void @f(i64)

