; Test that llvm-reduce can remove uninteresting functions as well as
; their InstCalls.
;
; RUN: llvm-reduce --test %p/Inputs/remove-funcs.sh --test-arg %lli %s
; RUN: cat reduced.ll | FileCheck %s
; REQUIRES: plugins, shell

@.str = private unnamed_addr constant [4 x i8] c"%i\0A\00", align 1

; CHECK-NOT: uninteresting1()
define i32 @uninteresting1() {
entry:
  ret i32 25
}

; CHECK: interesting()
define i32 @interesting() {
entry:
  ret i32 10
}

; CHECK-NOT: uninteresting2()
define i32 @uninteresting2() {
entry:
  ret i32 2000
}

; CHECK: main()
define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %number = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %number, align 4
  ; CHECK-NOT: %call = call i32 @uninteresting1()
  %call = call i32 @uninteresting1()
  ; CHECK: %call1 = call i32 @interesting()
  %call1 = call i32 @interesting()
  store i32 %call1, i32* %number, align 4
  %0 = load i32, i32* %number, align 4
  %call2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32 %0)
  ; CHECK-NOT: %call3 = call i32 @uninteresting1()
  %call3 = call i32 @uninteresting1()
  ; CHECK-NOT: %call4 = call i32 @uninteresting2()
  %call4 = call i32 @uninteresting2()
  ret i32 0
}

declare i32 @printf(i8*, ...)
