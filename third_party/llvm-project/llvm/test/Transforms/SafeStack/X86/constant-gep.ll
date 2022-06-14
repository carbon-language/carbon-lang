; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

%class.A = type { [2 x i8] }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; [2 x i8] in a class
;  safestack attribute
; Requires no protector.
; CHECK-LABEL: @foo(
define signext i8 @foo() nounwind uwtable safestack {
entry:
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  %a = alloca %class.A, align 1
  %array = getelementptr inbounds %class.A, %class.A* %a, i32 0, i32 0
  %arrayidx = getelementptr inbounds [2 x i8], [2 x i8]* %array, i32 0, i64 0
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}
