; RUN: opt -instcombine -S  < %s | FileCheck %s

%struct1 = type { %struct2*, i32, i32, i32 }
%struct2 = type { i32, i32 }

define i32 @test1(%struct1* %dm, i1 %tmp4, i64 %tmp9, i64 %tmp19) {
bb:
  %tmp = getelementptr inbounds %struct1* %dm, i64 0, i32 0
  %tmp1 = load %struct2** %tmp, align 8
  br i1 %tmp4, label %bb1, label %bb2

bb1:
  %tmp10 = getelementptr inbounds %struct2* %tmp1, i64 %tmp9
  %tmp11 = getelementptr inbounds %struct2* %tmp10, i64 0, i32 0
  store i32 0, i32* %tmp11, align 4
  br label %bb3

bb2:
  %tmp20 = getelementptr inbounds %struct2* %tmp1, i64 %tmp19
  %tmp21 = getelementptr inbounds %struct2* %tmp20, i64 0, i32 0
  store i32 0, i32* %tmp21, align 4
  br label %bb3

bb3:
  %phi = phi %struct2* [ %tmp10, %bb1 ], [ %tmp20, %bb2 ]
  %tmp24 = getelementptr inbounds %struct2* %phi, i64 0, i32 1
  %tmp25 = load i32* %tmp24, align 4
  ret i32 %tmp25

; CHECK-LABEL: @test1(
; CHECK: getelementptr inbounds %struct2* %tmp1, i64 %tmp9, i32 0
; CHECK: getelementptr inbounds %struct2* %tmp1, i64 %tmp19, i32 0
; CHECK: %[[PHI:[0-9A-Za-z]+]] = phi i64 [ %tmp9, %bb1 ], [ %tmp19, %bb2 ]
; CHECK: getelementptr inbounds %struct2* %tmp1, i64 %[[PHI]], i32 1
}

define i32 @test2(%struct1* %dm, i1 %tmp4, i64 %tmp9, i64 %tmp19) {
bb:
  %tmp = getelementptr inbounds %struct1* %dm, i64 0, i32 0
  %tmp1 = load %struct2** %tmp, align 8
  %tmp10 = getelementptr inbounds %struct2* %tmp1, i64 %tmp9
  %tmp11 = getelementptr inbounds %struct2* %tmp10, i64 0, i32 0
  store i32 0, i32* %tmp11, align 4
  %tmp20 = getelementptr inbounds %struct2* %tmp1, i64 %tmp19
  %tmp21 = getelementptr inbounds %struct2* %tmp20, i64 0, i32 0
  store i32 0, i32* %tmp21, align 4
  %tmp24 = getelementptr inbounds %struct2* %tmp10, i64 0, i32 1
  %tmp25 = load i32* %tmp24, align 4
  ret i32 %tmp25

; CHECK-LABEL: @test2(
; CHECK: getelementptr inbounds %struct2* %tmp1, i64 %tmp9, i32 0
; CHECK: getelementptr inbounds %struct2* %tmp1, i64 %tmp19, i32 0
; CHECK: getelementptr inbounds %struct2* %tmp1, i64 %tmp9, i32 1
}

