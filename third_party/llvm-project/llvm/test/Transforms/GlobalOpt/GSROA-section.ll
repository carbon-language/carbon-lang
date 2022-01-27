; This test lets globalopt split the global struct and array into different
; values. The pass needs to preserve section attribute.

; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; Check that the new global values still have their section assignment.
; CHECK: @struct
; CHECK: section ".foo"
; CHECK: @array
; CHECK-NOT: section ".foo"

@struct = internal global { i32, i32 } zeroinitializer, section ".foo"
@array = internal global [ 2 x i32 ] zeroinitializer

define i32 @foo() {
  %A = load i32, i32* getelementptr ({ i32, i32 }, { i32, i32 }* @struct, i32 0, i32 0)
  %B = load i32, i32* getelementptr ([ 2 x i32 ], [ 2 x i32 ]* @array, i32 0, i32 0)
  ; Use the loaded values, so they won't get removed completely
  %R = add i32 %A, %B
  ret i32 %R
}

; We put stores in a different function, so that the global variables won't get
; optimized away completely.
define void @bar(i32 %R) {
  store i32 %R, i32* getelementptr ([ 2 x i32 ], [ 2 x i32 ]* @array, i32 0, i32 0)
  store i32 %R, i32* getelementptr ({ i32, i32 }, { i32, i32 }* @struct, i32 0, i32 0)
  ret void
}


