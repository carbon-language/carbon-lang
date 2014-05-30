; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s
define i32 @main() #0 {
entry:
  %a = alloca i32, align 4
  store i32 1, i32* %a, align 4
  %0 = load i32* %a, align 4
  %or = or i32 1, %0
  %and = and i32 1, %or
  %rem = urem i32 %and, 1
  %add = add i32 %rem, 1
  ret i32 %add
}
; CHECK: $1, %eax
; CHECK-NEXT: retq
