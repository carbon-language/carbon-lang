; RUN: llc < %s -mtriple=x86_64-pc-linux -mattr=+bmi,+bmi2,+popcnt | FileCheck %s
declare void @foo(i32)
declare void @foo64(i64)

; CHECK-LABEL: neg:
; CHECK: negl %edi
; CHECK-NEXT: je
; CHECK: jmp foo
; CHECK: ret
define void @neg(i32 %x) nounwind {
  %sub = sub i32 0, %x
  %cmp = icmp eq i32 %sub, 0
  br i1 %cmp, label %return, label %bb

bb:
  tail call void @foo(i32 %sub)
  br label %return

return:
  ret void
}

; CHECK-LABEL: sar:
; CHECK: sarl %edi
; CHECK-NEXT: je
; CHECK: jmp foo
; CHECK: ret
define void @sar(i32 %x) nounwind {
  %ashr = ashr i32 %x, 1
  %cmp = icmp eq i32 %ashr, 0
  br i1 %cmp, label %return, label %bb

bb:
  tail call void @foo(i32 %ashr)
  br label %return

return:
  ret void
}

; CHECK-LABEL: shr:
; CHECK: shrl %edi
; CHECK-NEXT: je
; CHECK: jmp foo
; CHECK: ret
define void @shr(i32 %x) nounwind {
  %ashr = lshr i32 %x, 1
  %cmp = icmp eq i32 %ashr, 0
  br i1 %cmp, label %return, label %bb

bb:
  tail call void @foo(i32 %ashr)
  br label %return

return:
  ret void
}

; CHECK-LABEL: shri:
; CHECK: shrl $3, %edi
; CHECK-NEXT: je
; CHECK: jmp foo
; CHECK: ret
define void @shri(i32 %x) nounwind {
  %ashr = lshr i32 %x, 3
  %cmp = icmp eq i32 %ashr, 0
  br i1 %cmp, label %return, label %bb

bb:
  tail call void @foo(i32 %ashr)
  br label %return

return:
  ret void
}

; CHECK-LABEL: shl:
; CHECK: addl %edi, %edi
; CHECK-NEXT: je
; CHECK: jmp foo
; CHECK: ret
define void @shl(i32 %x) nounwind {
  %shl = shl i32 %x, 1
  %cmp = icmp eq i32 %shl, 0
  br i1 %cmp, label %return, label %bb

bb:
  tail call void @foo(i32 %shl)
  br label %return

return:
  ret void
}

; CHECK-LABEL: shli:
; CHECK: shll $4, %edi
; CHECK-NEXT: je
; CHECK: jmp foo
; CHECK: ret
define void @shli(i32 %x) nounwind {
  %shl = shl i32 %x, 4
  %cmp = icmp eq i32 %shl, 0
  br i1 %cmp, label %return, label %bb

bb:
  tail call void @foo(i32 %shl)
  br label %return

return:
  ret void
}

; CHECK-LABEL: adc:
; CHECK: movabsq $-9223372036854775808, %rax
; CHECK-NEXT: addq  %rdi, %rax
; CHECK-NEXT: adcq  $0, %rsi
; CHECK-NEXT: sete  %al
; CHECK: ret
define zeroext i1 @adc(i128 %x) nounwind {
  %add = add i128 %x, 9223372036854775808
  %cmp = icmp ult i128 %add, 18446744073709551616
  ret i1 %cmp
}

; CHECK-LABEL: sbb:
; CHECK: cmpq  %rdx, %rdi
; CHECK-NEXT: sbbq  %rcx, %rsi
; CHECK-NEXT: setns %al
; CHECK: ret
define zeroext i1 @sbb(i128 %x, i128 %y) nounwind {
  %sub = sub i128 %x, %y
  %cmp = icmp sge i128 %sub, 0
  ret i1 %cmp
}

; CHECK-LABEL: andn:
; CHECK: andnl   %esi, %edi, %edi
; CHECK-NEXT: je
; CHECK: jmp foo
; CHECK: ret
define void @andn(i32 %x, i32 %y) nounwind {
  %not = xor i32 %x, -1
  %andn = and i32 %y, %not
  %cmp = icmp eq i32 %andn, 0
  br i1 %cmp, label %return, label %bb

bb:
  tail call void @foo(i32 %andn)
  br label %return

return:
  ret void
}

; CHECK-LABEL: bextr:
; CHECK: bextrl   %esi, %edi, %edi
; CHECK-NEXT: je
; CHECK: jmp foo
; CHECK: ret
declare i32 @llvm.x86.bmi.bextr.32(i32, i32) nounwind readnone
define void @bextr(i32 %x, i32 %y) nounwind {
  %bextr = tail call i32 @llvm.x86.bmi.bextr.32(i32 %x, i32 %y)
  %cmp = icmp eq i32 %bextr, 0
  br i1 %cmp, label %return, label %bb

bb:
  tail call void @foo(i32 %bextr)
  br label %return

return:
  ret void
}

; CHECK-LABEL: popcnt:
; CHECK: popcntl
; CHECK-NEXT: je
; CHECK: jmp foo
; CHECK: ret
declare i32 @llvm.ctpop.i32(i32) nounwind readnone
define void @popcnt(i32 %x) nounwind {
  %popcnt = tail call i32 @llvm.ctpop.i32(i32 %x)
  %cmp = icmp eq i32 %popcnt, 0
  br i1 %cmp, label %return, label %bb
;
bb:
  tail call void @foo(i32 %popcnt)
  br label %return
;
return:
  ret void
}
