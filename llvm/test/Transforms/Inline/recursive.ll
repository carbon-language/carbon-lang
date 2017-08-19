; Inlining in the presence of recursion presents special challenges that we
; test here.
;
; RUN: opt -inline -S < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s

define i32 @large_stack_callee(i32 %param) {
; CHECK-LABEL: define i32 @large_stack_callee(
entry:
 %yyy = alloca [100000 x i8]
 %r = bitcast [100000 x i8]* %yyy to i8*
 call void @bar(i8* %r)
 ret i32 4
}

; Test a recursive function which calls another function with a large stack. In
; addition to not inlining the recursive call, we should also not inline the
; large stack allocation into a potentially recursive frame.
define i32 @large_stack_recursive_caller(i32 %param) {
; CHECK-LABEL: define i32 @large_stack_recursive_caller(
entry:
; CHECK-NEXT: entry:
; CHECK-NOT: alloca
  %t = call i32 @foo(i32 %param)
  %cmp = icmp eq i32 %t, -1
  br i1 %cmp, label %exit, label %cont

cont:
  %r = call i32 @large_stack_recursive_caller(i32 %t)
; CHECK: call i32 @large_stack_recursive_caller
  %f = call i32 @large_stack_callee(i32 %r)
; CHECK: call i32 @large_stack_callee
  br label %exit

exit:
  ret i32 4
}

declare void @bar(i8* %in)

declare i32 @foo(i32 %param)

