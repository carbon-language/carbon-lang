; RUN: opt < %s -globaldce -S | FileCheck %s

; test_function should not be emitted to the .s file.
; CHECK-NOT: @test_function
define available_externally i32 @test_function() {
  ret i32 4
}

; test_global should not be emitted to the .s file.
; CHECK-NOT: @test_global
@test_global = available_externally global i32 4

; CHECK: @x = external constant void ()*
@x = available_externally constant void()* @f
; CHECK: @y = external constant i32
@y = available_externally constant i32 ptrtoint (void()* @g to i32)
; @h is still alive, so don't remove the initializer too eagerly.
; CHECK: @z = available_externally constant i8 ptrtoint (void (i8)* @h to i8)
@z = available_externally constant i8 ptrtoint (void(i8)* @h to i8)

; CHECK-NOT: @f
define linkonce_odr void @f() {
  ret void
}

; CHECK-NOT: @g
define linkonce_odr void @g() {
  ret void
}

; CHECK: define linkonce_odr void @h
define linkonce_odr void @h(i8) {
  ret void
}

define i32 @main() {
  %f = load void()** @x
  call void %f()
  %g = load i32* @y
  %h = load i8* @z
  call void @h(i8 %h)
  ret i32 %g
}
