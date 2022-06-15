; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct = type { i32, i32, i32 }

; CHECK-LABEL: test_simple

; CHECK-DAG: MustAlias: %struct* %st, %struct* %sta

; CHECK-DAG: MayAlias: %struct* %st, i32* %x
; CHECK-DAG: MayAlias: %struct* %st, i32* %y
; CHECK-DAG: MayAlias: %struct* %st, i32* %z

; CHECK-DAG: NoAlias: i32* %x, i32* %y
; CHECK-DAG: NoAlias: i32* %x, i32* %z
; CHECK-DAG: NoAlias: i32* %y, i32* %z

; CHECK-DAG: MayAlias: %struct* %st, %struct* %y_12
; CHECK-DAG: MayAlias: i32* %x, %struct* %y_12
; CHECK-DAG: MayAlias: i32* %x, i80* %y_10

; CHECK-DAG: MayAlias: %struct* %st, i64* %y_8
; CHECK-DAG: MayAlias: i64* %y_8, i32* %z
; CHECK-DAG: NoAlias: i32* %x, i64* %y_8

; CHECK-DAG: MustAlias: i32* %y, %struct* %y_12
; CHECK-DAG: MustAlias: i32* %y, i64* %y_8
; CHECK-DAG: MustAlias: i32* %y, i80* %y_10

define void @test_simple(%struct* %st, i64 %i, i64 %j, i64 %k) {
  %x = getelementptr inbounds %struct, %struct* %st, i64 %i, i32 0
  %y = getelementptr inbounds %struct, %struct* %st, i64 %j, i32 1
  %sta = call %struct* @func2(%struct* %st)
  %z = getelementptr inbounds %struct, %struct* %sta, i64 %k, i32 2
  %y_12 = bitcast i32* %y to %struct*
  %y_10 = bitcast i32* %y to i80*
  %ya = call i32* @func1(i32* %y)
  %y_8 = bitcast i32* %ya to i64*
  load %struct, %struct* %st
  load %struct, %struct* %sta
  load i32, i32* %x
  load i32, i32* %y
  load i32, i32* %z
  load %struct, %struct* %y_12
  load i80, i80* %y_10
  load i64, i64* %y_8
  ret void
}

declare i32* @func1(i32* returned) nounwind
declare %struct* @func2(%struct* returned) nounwind

