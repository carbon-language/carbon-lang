; RUN: llc < %s -mcpu=corei7 -march=x86-64 -mattr=+sse2  | FileCheck %s

; This test makes sure that the compiler does not crash with an
; assertion failure when trying to fold a vector shift left
; by immediate count if the type of the input vector is different
; to the result type.
;
; This happens for example when lowering a shift left of a MVT::v16i8 vector.
; This is custom lowered into the following sequence:
;     count << 5
;     A =  VSHLI(MVT::v8i16, r & (char16)15, 4)
;     B = BITCAST MVT::v16i8, A
;     VSELECT(r, B, count);
;     count += count
;     C = VSHLI(MVT::v8i16, r & (char16)63, 2)
;     D = BITCAST MVT::v16i8, C
;     r = VSELECT(r, C, count);
;     count += count
;     VSELECT(r, r+r, count);
;     count = count << 5;
;
; Where 'r' is a vector of type MVT::v16i8, and
; 'count' is the vector shift count.

define <16 x i8> @do_not_crash(i8*, i32*, i64*, i32, i64, i8) {
entry:
  store i8 %5, i8* %0
  %L5 = load i8* %0
  %I8 = insertelement <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, i8 %L5, i32 7
  %B51 = shl <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, %I8
  ret <16 x i8> %B51
}

; CHECK-LABEL: do_not_crash
; CHECK: ret

