; This test ensures that alloca instructions in the entry block for an inlined
; function are moved to the top of the function they are inlined into.
;
; RUN: opt -S -inline < %s | FileCheck %s

define i32 @func(i32 %i) {
  %X = alloca i32
  store i32 %i, i32* %X
  ret i32 %i
}

declare void @bar()

define i32 @main(i32 %argc) {
; CHECK-LABEL: @main(
; CHECK-NEXT:  Entry:
; CHECK-NEXT:    [[X_I:%.*]] = alloca i32
;
Entry:
  call void @bar( )
  %X = call i32 @func( i32 7 )
  %Y = add i32 %X, %argc
  ret i32 %Y
}

