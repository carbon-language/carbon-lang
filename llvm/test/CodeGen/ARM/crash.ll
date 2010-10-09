; RUN: llc < %s -mtriple=thumbv7-apple-darwin10

; <rdar://problem/8529919>
%struct.foo = type { i32, i32 }

define void @func() nounwind {
entry:
  %tmp = load i32* undef, align 4
  br label %bb1

bb1:
  %tmp1 = and i32 %tmp, 16
  %tmp2 = icmp eq i32 %tmp1, 0
  %invok.1.i = select i1 %tmp2, i32 undef, i32 0
  %tmp119 = add i32 %invok.1.i, 0
  br i1 undef, label %bb2, label %exit

bb2:
  %tmp120 = add i32 %tmp119, 0
  %scevgep810.i = getelementptr %struct.foo* null, i32 %tmp120, i32 1
  store i32 undef, i32* %scevgep810.i, align 4
  br i1 undef, label %bb2, label %bb3

bb3:
  br i1 %tmp2, label %bb2, label %bb2

exit:
  ret void
}
