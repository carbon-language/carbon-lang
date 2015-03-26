; RUN: opt < %s -slsr -gvn -dce -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

define i32 @slsr_gep(i32* %input, i64 %s) {
; CHECK-LABEL: @slsr_gep(
  ; v0 = input[0];
  %p0 = getelementptr inbounds i32, i32* %input, i64 0
  %v0 = load i32, i32* %p0

  ; v1 = input[s];
  %p1 = getelementptr inbounds i32, i32* %input, i64 %s
; CHECK: %p1 = getelementptr inbounds i32, i32* %input, i64 %s
  %v1 = load i32, i32* %p1

  ; v2 = input[s * 2];
  %s2 = mul nsw i64 %s, 2
  %p2 = getelementptr inbounds i32, i32* %input, i64 %s2
; CHECK: %p2 = getelementptr inbounds i32, i32* %p1, i64 %s
  %v2 = load i32, i32* %p2

  ; return v0 + v1 + v2;
  %1 = add i32 %v0, %v1
  %2 = add i32 %1, %v2
  ret i32 %2
}

define i32 @slsr_gep_sext(i32* %input, i32 %s) {
; CHECK-LABEL: @slsr_gep_sext(
  ; v0 = input[0];
  %p0 = getelementptr inbounds i32, i32* %input, i64 0
  %v0 = load i32, i32* %p0

  ; v1 = input[(long)s];
  %t = sext i32 %s to i64
  %p1 = getelementptr inbounds i32, i32* %input, i64 %t
; CHECK: %p1 = getelementptr inbounds i32, i32* %input, i64 %t
  %v1 = load i32, i32* %p1

  ; v2 = input[(long)(s * 2)];
  %s2 = mul nsw i32 %s, 2
  %t2 = sext i32 %s2 to i64
  %p2 = getelementptr inbounds i32, i32* %input, i64 %t2
; CHECK: %p2 = getelementptr inbounds i32, i32* %p1, i64 %t
  %v2 = load i32, i32* %p2

  ; return v0 + v1 + v2;
  %1 = add i32 %v0, %v1
  %2 = add i32 %1, %v2
  ret i32 %2
}

define i32 @slsr_gep_2d([10 x [5 x i32]]* %input, i64 %s, i64 %t) {
; CHECK-LABEL: @slsr_gep_2d(
  ; v0 = input[s][t];
  %p0 = getelementptr inbounds [10 x [5 x i32]], [10 x [5 x i32]]* %input, i64 0, i64 %s, i64 %t
  %v0 = load i32, i32* %p0

  ; v1 = input[s * 2][t];
  %s2 = mul nsw i64 %s, 2
; CHECK: [[BUMP:%[a-zA-Z0-9]+]] = mul i64 %s, 5
  %p1 = getelementptr inbounds [10 x [5 x i32]], [10 x [5 x i32]]* %input, i64 0, i64 %s2, i64 %t
; CHECK: %p1 = getelementptr inbounds i32, i32* %p0, i64 [[BUMP]]
  %v1 = load i32, i32* %p1

  ; v2 = input[s * 3][t];
  %s3 = mul nsw i64 %s, 3
  %p2 = getelementptr inbounds [10 x [5 x i32]], [10 x [5 x i32]]* %input, i64 0, i64 %s3, i64 %t
; CHECK: %p2 = getelementptr inbounds i32, i32* %p1, i64 [[BUMP]]
  %v2 = load i32, i32* %p2

  ; return v0 + v1 + v2;
  %1 = add i32 %v0, %v1
  %2 = add i32 %1, %v2
  ret i32 %2
}

%struct.S = type <{ i64, i32 }>

; In this case, the bump
;     = (char *)&input[s * 2][t].f1 - (char *)&input[s][t].f1
;     = 60 * s
; which may not be divisible by typeof(input[s][t].f1) = 8. Therefore, we
; rewrite the candidates using byte offset instead of index offset as in
; @slsr_gep_2d.
define i64 @slsr_gep_uglygep([10 x [5 x %struct.S]]* %input, i64 %s, i64 %t) {
; CHECK-LABEL: @slsr_gep_uglygep(
  ; v0 = input[s][t].f1;
  %p0 = getelementptr inbounds [10 x [5 x %struct.S]], [10 x [5 x %struct.S]]* %input, i64 0, i64 %s, i64 %t, i32 0
  %v0 = load i64, i64* %p0

  ; v1 = input[s * 2][t].f1;
  %s2 = mul nsw i64 %s, 2
; CHECK: [[BUMP:%[a-zA-Z0-9]+]] = mul i64 %s, 60
  %p1 = getelementptr inbounds [10 x [5 x %struct.S]], [10 x [5 x %struct.S]]* %input, i64 0, i64 %s2, i64 %t, i32 0
; CHECK: getelementptr inbounds i8, i8* %{{[0-9]+}}, i64 [[BUMP]]
  %v1 = load i64, i64* %p1

  ; v2 = input[s * 3][t].f1;
  %s3 = mul nsw i64 %s, 3
  %p2 = getelementptr inbounds [10 x [5 x %struct.S]], [10 x [5 x %struct.S]]* %input, i64 0, i64 %s3, i64 %t, i32 0
; CHECK: getelementptr inbounds i8, i8* %{{[0-9]+}}, i64 [[BUMP]]
  %v2 = load i64, i64* %p2

  ; return v0 + v1 + v2;
  %1 = add i64 %v0, %v1
  %2 = add i64 %1, %v2
  ret i64 %2
}
