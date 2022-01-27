; RUN: opt < %s -slsr -gvn -S | FileCheck %s
; RUN: opt < %s -passes='slsr,gvn' -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64-p:64:64:64-p1:32:32:32"

; foo(input[0]);
; foo(input[s]);
; foo(input[s * 2]);
;   =>
; p0 = &input[0];
; foo(*p);
; p1 = p0 + s;
; foo(*p1);
; p2 = p1 + s;
; foo(*p2);
define void @slsr_gep(i32* %input, i64 %s) {
; CHECK-LABEL: @slsr_gep(
  ; v0 = input[0];
  %p0 = getelementptr inbounds i32, i32* %input, i64 0
  call void @foo(i32* %p0)

  ; v1 = input[s];
  %p1 = getelementptr inbounds i32, i32* %input, i64 %s
; CHECK: %p1 = getelementptr inbounds i32, i32* %input, i64 %s
  call void @foo(i32* %p1)

  ; v2 = input[s * 2];
  %s2 = shl nsw i64 %s, 1
  %p2 = getelementptr inbounds i32, i32* %input, i64 %s2
; CHECK: %p2 = getelementptr inbounds i32, i32* %p1, i64 %s
  call void @foo(i32* %p2)

  ret void
}

; foo(input[0]);
; foo(input[(long)s]);
; foo(input[(long)(s * 2)]);
;   =>
; p0 = &input[0];
; foo(*p);
; p1 = p0 + (long)s;
; foo(*p1);
; p2 = p1 + (long)s;
; foo(*p2);
define void @slsr_gep_sext(i32* %input, i32 %s) {
; CHECK-LABEL: @slsr_gep_sext(
  ; v0 = input[0];
  %p0 = getelementptr inbounds i32, i32* %input, i64 0
  call void @foo(i32* %p0)

  ; v1 = input[s];
  %t = sext i32 %s to i64
  %p1 = getelementptr inbounds i32, i32* %input, i64 %t
; CHECK: %p1 = getelementptr inbounds i32, i32* %input, i64 %t
  call void @foo(i32* %p1)

  ; v2 = input[s * 2];
  %s2 = shl nsw i32 %s, 1
  %t2 = sext i32 %s2 to i64
  %p2 = getelementptr inbounds i32, i32* %input, i64 %t2
; CHECK: %p2 = getelementptr inbounds i32, i32* %p1, i64 %t
  call void @foo(i32* %p2)

  ret void
}

; int input[10][5];
; foo(input[s][t]);
; foo(input[s * 2][t]);
; foo(input[s * 3][t]);
;   =>
; p0 = &input[s][t];
; foo(*p0);
; p1 = p0 + 5s;
; foo(*p1);
; p2 = p1 + 5s;
; foo(*p2);
define void @slsr_gep_2d([10 x [5 x i32]]* %input, i64 %s, i64 %t) {
; CHECK-LABEL: @slsr_gep_2d(
  ; v0 = input[s][t];
  %p0 = getelementptr inbounds [10 x [5 x i32]], [10 x [5 x i32]]* %input, i64 0, i64 %s, i64 %t
  call void @foo(i32* %p0)

  ; v1 = input[s * 2][t];
  %s2 = shl nsw i64 %s, 1
; CHECK: [[BUMP:%[a-zA-Z0-9]+]] = mul i64 %s, 5
  %p1 = getelementptr inbounds [10 x [5 x i32]], [10 x [5 x i32]]* %input, i64 0, i64 %s2, i64 %t
; CHECK: %p1 = getelementptr inbounds i32, i32* %p0, i64 [[BUMP]]
  call void @foo(i32* %p1)

  ; v3 = input[s * 3][t];
  %s3 = mul nsw i64 %s, 3
  %p2 = getelementptr inbounds [10 x [5 x i32]], [10 x [5 x i32]]* %input, i64 0, i64 %s3, i64 %t
; CHECK: %p2 = getelementptr inbounds i32, i32* %p1, i64 [[BUMP]]
  call void @foo(i32* %p2)

  ret void
}

%struct.S = type <{ i64, i32 }>

; In this case, the bump
;     = (char *)&input[s * 2][t].f1 - (char *)&input[s][t].f1
;     = 60 * s
; which may not be divisible by typeof(input[s][t].f1) = 8. Therefore, we
; rewrite the candidates using byte offset instead of index offset as in
; @slsr_gep_2d.
define void @slsr_gep_uglygep([10 x [5 x %struct.S]]* %input, i64 %s, i64 %t) {
; CHECK-LABEL: @slsr_gep_uglygep(
  ; v0 = input[s][t].f1;
  %p0 = getelementptr inbounds [10 x [5 x %struct.S]], [10 x [5 x %struct.S]]* %input, i64 0, i64 %s, i64 %t, i32 0
  call void @bar(i64* %p0)

  ; v1 = input[s * 2][t].f1;
  %s2 = shl nsw i64 %s, 1
; CHECK: [[BUMP:%[a-zA-Z0-9]+]] = mul i64 %s, 60
  %p1 = getelementptr inbounds [10 x [5 x %struct.S]], [10 x [5 x %struct.S]]* %input, i64 0, i64 %s2, i64 %t, i32 0
; CHECK: getelementptr inbounds i8, i8* %{{[0-9]+}}, i64 [[BUMP]]
  call void @bar(i64* %p1)

  ; v2 = input[s * 3][t].f1;
  %s3 = mul nsw i64 %s, 3
  %p2 = getelementptr inbounds [10 x [5 x %struct.S]], [10 x [5 x %struct.S]]* %input, i64 0, i64 %s3, i64 %t, i32 0
; CHECK: getelementptr inbounds i8, i8* %{{[0-9]+}}, i64 [[BUMP]]
  call void @bar(i64* %p2)

  ret void
}

define void @slsr_out_of_bounds_gep(i32* %input, i32 %s) {
; CHECK-LABEL: @slsr_out_of_bounds_gep(
  ; v0 = input[0];
  %p0 = getelementptr i32, i32* %input, i64 0
  call void @foo(i32* %p0)

  ; v1 = input[(long)s];
  %t = sext i32 %s to i64
  %p1 = getelementptr i32, i32* %input, i64 %t
; CHECK: %p1 = getelementptr i32, i32* %input, i64 %t
  call void @foo(i32* %p1)

  ; v2 = input[(long)(s * 2)];
  %s2 = shl nsw i32 %s, 1
  %t2 = sext i32 %s2 to i64
  %p2 = getelementptr i32, i32* %input, i64 %t2
; CHECK: %p2 = getelementptr i32, i32* %p1, i64 %t
  call void @foo(i32* %p2)

  ret void
}

define void @slsr_gep_128bit_index(i32* %input, i128 %s) {
; CHECK-LABEL: @slsr_gep_128bit_index(
  ; p0 = &input[0]
  %p0 = getelementptr inbounds i32, i32* %input, i128 0
  call void @foo(i32* %p0)

  ; p1 = &input[s << 125]
  %s125 = shl nsw i128 %s, 125
  %p1 = getelementptr inbounds i32, i32* %input, i128 %s125
; CHECK: %p1 = getelementptr inbounds i32, i32* %input, i128 %s125
  call void @foo(i32* %p1)

  ; p2 = &input[s << 126]
  %s126 = shl nsw i128 %s, 126
  %p2 = getelementptr inbounds i32, i32* %input, i128 %s126
; CHECK: %p2 = getelementptr inbounds i32, i32* %input, i128 %s126
  call void @foo(i32* %p2)

  ret void
}

define void @slsr_gep_32bit_pointer(i32 addrspace(1)* %input, i64 %s) {
; CHECK-LABEL: @slsr_gep_32bit_pointer(
  ; p1 = &input[s]
  %p1 = getelementptr inbounds i32, i32 addrspace(1)* %input, i64 %s
  call void @baz(i32 addrspace(1)* %p1)

  ; p2 = &input[s * 2]
  %s2 = mul nsw i64 %s, 2
  %p2 = getelementptr inbounds i32, i32 addrspace(1)* %input, i64 %s2
  ; %s2 is wider than the pointer size of addrspace(1), so do not factor it.
; CHECK: %p2 = getelementptr inbounds i32, i32 addrspace(1)* %input, i64 %s2
  call void @baz(i32 addrspace(1)* %p2)

  ret void
}

declare void @foo(i32*)
declare void @bar(i64*)
declare void @baz(i32 addrspace(1)*)
