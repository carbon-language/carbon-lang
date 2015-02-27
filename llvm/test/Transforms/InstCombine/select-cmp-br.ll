; Replace a 'select' with 'or' in 'select - cmp [eq|ne] - br' sequence
; RUN: opt -instcombine -S < %s | FileCheck %s

%C = type <{ %struct.S }>
%struct.S = type { i64*, i32, i32 }

declare void @bar(%struct.S *) #1
declare void @foobar()

define void @test1(%C*) {
entry:
  %1 = getelementptr inbounds %C, %C* %0, i64 0, i32 0, i32 0
  %m = load i64** %1, align 8
  %2 = getelementptr inbounds %C, %C* %0, i64 1, i32 0, i32 0
  %n = load i64** %2, align 8
  %3 = getelementptr inbounds i64, i64* %m, i64 9
  %4 = bitcast i64* %3 to i64 (%C*)**
  %5 = load i64 (%C*)** %4, align 8
  %6 = icmp eq i64* %m, %n
  %7 = select i1 %6, %C* %0, %C* null
  %8 = icmp eq %C* %7, null
  br i1 %8, label %12, label %10

; <label>:9                                       ; preds = %10, %12
  ret void

; <label>:10                                      ; preds = %entry
  %11 = getelementptr inbounds %C, %C* %7, i64 0, i32 0
  tail call void @bar(%struct.S* %11)
  br label %9

; <label>:12                                      ; preds = %entry
  %13 = tail call i64 %5(%C* %0)
  br label %9
; CHECK-LABEL: @test1(
; CHECK-NOT: select
; CHECK: or
; CHECK-NOT: select
}

define void @test2(%C*) {
entry:
  %1 = getelementptr inbounds %C, %C* %0, i64 0, i32 0, i32 0
  %m = load i64** %1, align 8
  %2 = getelementptr inbounds %C, %C* %0, i64 1, i32 0, i32 0
  %n = load i64** %2, align 8
  %3 = getelementptr inbounds i64, i64* %m, i64 9
  %4 = bitcast i64* %3 to i64 (%C*)**
  %5 = load i64 (%C*)** %4, align 8
  %6 = icmp eq i64* %m, %n
  %7 = select i1 %6, %C* null, %C* %0
  %8 = icmp eq %C* %7, null
  br i1 %8, label %12, label %10

; <label>:9                                       ; preds = %10, %12
  ret void

; <label>:10                                      ; preds = %entry
  %11 = getelementptr inbounds %C, %C* %7, i64 0, i32 0
  tail call void @bar(%struct.S* %11)
  br label %9

; <label>:12                                      ; preds = %entry
  %13 = tail call i64 %5(%C* %0)
  br label %9
; CHECK-LABEL: @test2(
; CHECK-NOT: select
; CHECK: or
; CHECK-NOT: select
}

define void @test3(%C*) {
entry:
  %1 = getelementptr inbounds %C, %C* %0, i64 0, i32 0, i32 0
  %m = load i64** %1, align 8
  %2 = getelementptr inbounds %C, %C* %0, i64 1, i32 0, i32 0
  %n = load i64** %2, align 8
  %3 = getelementptr inbounds i64, i64* %m, i64 9
  %4 = bitcast i64* %3 to i64 (%C*)**
  %5 = load i64 (%C*)** %4, align 8
  %6 = icmp eq i64* %m, %n
  %7 = select i1 %6, %C* %0, %C* null
  %8 = icmp ne %C* %7, null
  br i1 %8, label %10, label %12

; <label>:9                                       ; preds = %10, %12
  ret void

; <label>:10                                      ; preds = %entry
  %11 = getelementptr inbounds %C, %C* %7, i64 0, i32 0
  tail call void @bar(%struct.S* %11)
  br label %9

; <label>:12                                      ; preds = %entry
  %13 = tail call i64 %5(%C* %0)
  br label %9
; CHECK-LABEL: @test3(
; CHECK-NOT: select
; CHECK: or
; CHECK-NOT: select
}

define void @test4(%C*) {
entry:
  %1 = getelementptr inbounds %C, %C* %0, i64 0, i32 0, i32 0
  %m = load i64** %1, align 8
  %2 = getelementptr inbounds %C, %C* %0, i64 1, i32 0, i32 0
  %n = load i64** %2, align 8
  %3 = getelementptr inbounds i64, i64* %m, i64 9
  %4 = bitcast i64* %3 to i64 (%C*)**
  %5 = load i64 (%C*)** %4, align 8
  %6 = icmp eq i64* %m, %n
  %7 = select i1 %6, %C* null, %C* %0
  %8 = icmp ne %C* %7, null
  br i1 %8, label %10, label %12

; <label>:9                                       ; preds = %10, %12
  ret void

; <label>:10                                      ; preds = %entry
  %11 = getelementptr inbounds %C, %C* %7, i64 0, i32 0
  tail call void @bar(%struct.S* %11)
  br label %9

; <label>:12                                      ; preds = %entry
  %13 = tail call i64 %5(%C* %0)
  br label %9
; CHECK-LABEL: @test4(
; CHECK-NOT: select
; CHECK: or
; CHECK-NOT: select
}

define void @test5(%C*, i1) {
entry:
  %2 = select i1 %1, %C* null, %C* %0
  %3 = icmp ne %C* %2, null
  br i1 %3, label %5, label %7

; <label>:4                                       ; preds = %10, %12
  ret void

; <label>:5                                      ; preds = %entry
  %6 = getelementptr inbounds %C, %C* %2, i64 0, i32 0
  tail call void @bar(%struct.S* %6)
  br label %4

; <label>:7                                      ; preds = %entry
  tail call void @foobar()
  br label %4
; CHECK-LABEL: @test5(
; CHECK-NOT: select
; CHECK: or
; CHECK-NOT: select
}
