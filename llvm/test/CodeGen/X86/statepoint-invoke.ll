; RUN: llc < %s 2>&1 | FileCheck %s

target triple = "x86_64-pc-linux-gnu"

declare i64 addrspace(1)* @"some_other_call"(i64 addrspace(1)*)

declare i32 @"personality_function"()

define i64 addrspace(1)* @test_result(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1) {
entry:
  ; CHECK: .Ltmp{{[0-9]+}}:
  ; CHECK: callq some_other_call
  ; CHECK: .Ltmp{{[0-9]+}}:
  %0 = invoke i32 (i64 addrspace(1)* (i64 addrspace(1)*)*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_p1i64p1i64f(i64 addrspace(1)* (i64 addrspace(1)*)* @some_other_call, i32 1, i32 0, i64 addrspace(1)* %obj, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1)
          to label %normal_return unwind label %exceptional_return

normal_return:
  ; CHECK: popq 
  ; CHECK: retq
  %ret_val = call i64 addrspace(1)* @llvm.experimental.gc.result.p1i64(i32 %0)
  ret i64 addrspace(1)* %ret_val

exceptional_return:
  ; CHECK: .Ltmp{{[0-9]+}}:
  ; CHECK: popq
  ; CHECK: retq
  %landing_pad = landingpad { i8*, i32 } personality i32 ()* @personality_function
          cleanup
  ret i64 addrspace(1)* %obj
}
; CHECK-LABEL: GCC_except_table{{[0-9]+}}:
; CHECK: .long .Ltmp{{[0-9]+}}-.Ltmp{{[0-9]+}}
; CHECK: .long .Ltmp{{[0-9]+}}-.Lfunc_begin{{[0-9]+}}
; CHECK: .byte 0
; CHECK: .align 4

declare i32 @llvm.experimental.gc.statepoint.p0f_p1i64p1i64f(i64 addrspace(1)* (i64 addrspace(1)*)*, i32, i32, ...)
declare i64 addrspace(1)* @llvm.experimental.gc.result.p1i64(i32)
