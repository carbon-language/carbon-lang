; RUN: opt -mtriple=amdgcn-amd-amdhsa -load-store-vectorizer -dce -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

define void @base_case(i1 %cnd, i32 addrspace(1)* %a, i32 addrspace(1)* %b, <3 x i32> addrspace(1)* %out) {
; CHECK-LABEL: @base_case
; CHECK: load <3 x i32>
entry:
  %gep1 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 1
  %gep2 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 2
  %gep4 = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 1
  %gep5 = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 2
  %selected = select i1 %cnd, i32 addrspace(1)* %a, i32 addrspace(1)* %b
  %selected14 = select i1 %cnd, i32 addrspace(1)* %gep1, i32 addrspace(1)* %gep4
  %selected25 = select i1 %cnd, i32 addrspace(1)* %gep2, i32 addrspace(1)* %gep5
  %val0 = load i32, i32 addrspace(1)* %selected, align 4
  %val1 = load i32, i32 addrspace(1)* %selected14, align 4
  %val2 = load i32, i32 addrspace(1)* %selected25, align 4
  %t0 = insertelement <3 x i32> undef, i32 %val0, i32 0
  %t1 = insertelement <3 x i32> %t0, i32 %val1, i32 1
  %t2 = insertelement <3 x i32> %t1, i32 %val2, i32 2
  store <3 x i32> %t2, <3 x i32> addrspace(1)* %out
  ret void
}

define void @scev_targeting_complex_case(i1 %cnd, i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %base, <2 x i32> addrspace(1)* %out) {
; CHECK-LABEL: @scev_targeting_complex_case
; CHECK: load <2 x i32>
entry:
  %base.x4 = shl i32 %base, 2
  %base.x4.p1 = add i32 %base.x4, 1
  %base.x4.p2 = add i32 %base.x4, 2
  %base.x4.p3 = add i32 %base.x4, 3
  %zext.x4 = zext i32 %base.x4 to i64
  %zext.x4.p1 = zext i32 %base.x4.p1 to i64
  %zext.x4.p2 = zext i32 %base.x4.p2 to i64
  %zext.x4.p3 = zext i32 %base.x4.p3 to i64
  %base.x16 = mul i64 %zext.x4, 4
  %base.x16.p4 = shl i64 %zext.x4.p1, 2
  %base.x16.p8 = shl i64 %zext.x4.p2, 2
  %base.x16.p12 = mul i64 %zext.x4.p3, 4
  %a.pi8 = bitcast i32 addrspace(1)* %a to i8 addrspace(1)*
  %b.pi8 = bitcast i32 addrspace(1)* %b to i8 addrspace(1)*
  %gep.a.base.x16 = getelementptr inbounds i8, i8 addrspace(1)* %a.pi8, i64 %base.x16
  %gep.b.base.x16.p4 = getelementptr inbounds i8, i8 addrspace(1)* %b.pi8, i64 %base.x16.p4
  %gep.a.base.x16.p8 = getelementptr inbounds i8, i8 addrspace(1)* %a.pi8, i64 %base.x16.p8
  %gep.b.base.x16.p12 = getelementptr inbounds i8, i8 addrspace(1)* %b.pi8, i64 %base.x16.p12
  %a.base.x16 = bitcast i8 addrspace(1)* %gep.a.base.x16 to i32 addrspace(1)*
  %b.base.x16.p4 = bitcast i8 addrspace(1)* %gep.b.base.x16.p4 to i32 addrspace(1)*
  %selected.base.x16.p0.or.4 = select i1 %cnd, i32 addrspace(1)* %a.base.x16, i32 addrspace(1)* %b.base.x16.p4
  %gep.selected.base.x16.p8.or.12 = select i1 %cnd, i8 addrspace(1)* %gep.a.base.x16.p8, i8 addrspace(1)* %gep.b.base.x16.p12
  %selected.base.x16.p8.or.12 = bitcast i8 addrspace(1)* %gep.selected.base.x16.p8.or.12 to i32 addrspace(1)*
  %selected.base.x16.p40.or.44 = getelementptr inbounds i32, i32 addrspace(1)* %selected.base.x16.p0.or.4, i64 10
  %selected.base.x16.p44.or.48 = getelementptr inbounds i32, i32 addrspace(1)* %selected.base.x16.p8.or.12, i64 9
  %val0 = load i32, i32 addrspace(1)* %selected.base.x16.p40.or.44, align 4
  %val1 = load i32, i32 addrspace(1)* %selected.base.x16.p44.or.48, align 4
  %t0 = insertelement <2 x i32> undef, i32 %val0, i32 0
  %t1 = insertelement <2 x i32> %t0, i32 %val1, i32 1
  store <2 x i32> %t1, <2 x i32> addrspace(1)* %out
  ret void
}

define void @nested_selects(i1 %cnd0, i1 %cnd1, i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %base, <2 x i32> addrspace(1)* %out) {
; CHECK-LABEL: @nested_selects
; CHECK: load <2 x i32>
entry:
  %base.p1 = add nsw i32 %base, 1
  %base.p2 = add i32 %base, 2
  %base.p3 = add nsw i32 %base, 3
  %base.x4 = mul i32 %base, 4
  %base.x4.p5 = add i32 %base.x4, 5
  %base.x4.p6 = add i32 %base.x4, 6
  %sext = sext i32 %base to i64
  %sext.p1 = sext i32 %base.p1 to i64
  %sext.p2 = sext i32 %base.p2 to i64
  %sext.p3 = sext i32 %base.p3 to i64
  %sext.x4.p5 = sext i32 %base.x4.p5 to i64
  %sext.x4.p6 = sext i32 %base.x4.p6 to i64
  %gep.a.base = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %sext
  %gep.a.base.p1 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %sext.p1
  %gep.a.base.p2 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %sext.p2
  %gep.a.base.p3 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %sext.p3
  %gep.b.base.x4.p5 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %sext.x4.p5
  %gep.b.base.x4.p6 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %sext.x4.p6
  %selected.1.L = select i1 %cnd1, i32 addrspace(1)* %gep.a.base.p2, i32 addrspace(1)* %gep.b.base.x4.p5
  %selected.1.R = select i1 %cnd1, i32 addrspace(1)* %gep.a.base.p3, i32 addrspace(1)* %gep.b.base.x4.p6
  %selected.0.L = select i1 %cnd0, i32 addrspace(1)* %gep.a.base, i32 addrspace(1)* %selected.1.L
  %selected.0.R = select i1 %cnd0, i32 addrspace(1)* %gep.a.base.p1, i32 addrspace(1)* %selected.1.R
  %val0 = load i32, i32 addrspace(1)* %selected.0.L, align 4
  %val1 = load i32, i32 addrspace(1)* %selected.0.R, align 4
  %t0 = insertelement <2 x i32> undef, i32 %val0, i32 0
  %t1 = insertelement <2 x i32> %t0, i32 %val1, i32 1
  store <2 x i32> %t1, <2 x i32> addrspace(1)* %out
  ret void
}
