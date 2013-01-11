; RUN: llc < %s -O1 -mcpu=atom -mtriple=i686-linux  | FileCheck %s

declare void @external_function(...)

define i32 @test_return_val(i32 %a) nounwind {
; CHECK: test_return_val
; CHECK: movl
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: ret
  ret i32 %a
}

define i32 @test_optsize(i32 %a) nounwind optsize {
; CHECK: test_optsize
; CHECK: movl
; CHECK-NEXT: ret
  ret i32 %a
}

define i32 @test_minsize(i32 %a) nounwind minsize {
; CHECK: test_minsize
; CHECK: movl
; CHECK-NEXT: ret
  ret i32 %a
}

define i32 @test_add(i32 %a, i32 %b) nounwind {
; CHECK: test_add
; CHECK: addl
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: ret
  %result = add i32 %a, %b
  ret i32 %result
}

define i32 @test_multiple_ret(i32 %a, i32 %b, i1 %c) nounwind {
; CHECK: @test_multiple_ret
; CHECK: je

; CHECK: nop
; CHECK: nop
; CHECK: ret

; CHECK: nop
; CHECK: nop
; CHECK: ret

  br i1 %c, label %bb1, label %bb2

bb1:
  ret i32 %a

bb2:
  ret i32 %b
}

define void @test_call_others(i32 %x) nounwind
{
; CHECK: test_call_others
; CHECK: je
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %if.end, label %true.case

; CHECK: jmp external_function
true.case:
  tail call void bitcast (void (...)* @external_function to void ()*)() nounwind
  br label %if.end

; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: ret
if.end:
  ret void

}

define void @test_branch_to_same_bb(i32 %x, i32 %y) nounwind {
; CHECK: @test_branch_to_same_bb
  %cmp = icmp sgt i32 %x, 0
  br i1 %cmp, label %while.cond, label %while.end

while.cond:
  br label %while.cond

; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: ret
while.end:
  ret void
}

