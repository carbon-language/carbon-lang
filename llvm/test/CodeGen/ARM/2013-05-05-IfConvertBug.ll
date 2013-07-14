; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 | FileCheck %s
; rdar://13782395

define i32 @t1(i32 %a, i32 %b, i8** %retaddr) {
; CHECK-LABEL: t1:
; CHECK: Block address taken
; CHECK-NOT: Address of block that was removed by CodeGen
  store i8* blockaddress(@t1, %cond_true), i8** %retaddr
  %tmp2 = icmp eq i32 %a, 0
  br i1 %tmp2, label %cond_false, label %cond_true

cond_true:
  %tmp5 = add i32 %b, 1
  ret i32 %tmp5

cond_false:
  %tmp7 = add i32 %b, -1
  ret i32 %tmp7
}

define i32 @t2(i32 %a, i32 %b, i32 %c, i32 %d, i8** %retaddr) {
; CHECK-LABEL: t2:
; CHECK: Block address taken
; CHECK: %cond_true
; CHECK: add
; CHECK: bx lr
  store i8* blockaddress(@t2, %cond_true), i8** %retaddr
  %tmp2 = icmp sgt i32 %c, 10
  %tmp5 = icmp slt i32 %d, 4
  %tmp8 = and i1 %tmp5, %tmp2
  %tmp13 = add i32 %b, %a
  br i1 %tmp8, label %cond_true, label %UnifiedReturnBlock

cond_true:
  %tmp15 = add i32 %tmp13, %c
  %tmp1821 = sub i32 %tmp15, %d
  ret i32 %tmp1821

UnifiedReturnBlock:
  ret i32 %tmp13
}

define hidden fastcc void @t3(i8** %retaddr) {
; CHECK-LABEL: t3:
; CHECK: Block address taken
; CHECK-NOT: Address of block that was removed by CodeGen
bb:
  store i8* blockaddress(@t3, %KBBlockZero_return_1), i8** %retaddr
  br i1 undef, label %bb77, label %bb7.i

bb7.i:                                            ; preds = %bb35
  br label %bb2.i

KBBlockZero_return_1:                             ; preds = %KBBlockZero.exit
  unreachable

KBBlockZero_return_0:                             ; preds = %KBBlockZero.exit
  unreachable

bb77:                                             ; preds = %bb26, %bb12, %bb
  ret void

bb2.i:                                            ; preds = %bb6.i350, %bb7.i
  br i1 undef, label %bb6.i350, label %KBBlockZero.exit

bb6.i350:                                         ; preds = %bb2.i
  br label %bb2.i

KBBlockZero.exit:                                 ; preds = %bb2.i
  indirectbr i8* undef, [label %KBBlockZero_return_1, label %KBBlockZero_return_0]
}
