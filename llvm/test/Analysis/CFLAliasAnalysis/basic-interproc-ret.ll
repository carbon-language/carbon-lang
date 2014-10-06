; This testcase ensures that CFL AA gives conservative answers on variables
; that involve arguments.

; RUN: opt < %s -cfl-aa -aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

; CHECK:     Function: test
; CHECK: 4 Total Alias Queries Performed
; CHECK: 3 no alias responses
; ^ The 1 MayAlias is due to %arg1. Sadly, we don't currently have machinery
; in place to check whether %arg1 aliases %a, because BasicAA takes care of 
; that for us.

define i32* @test2(i32* %arg1) {
  store i32 0, i32* %arg1

  %a = alloca i32, align 4
  ret i32* %a
}

define void @test() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = call i32* @test2(i32* %a)

  ret void
}
