; RUN: llvm-as < %s | opt -ssi-everything -disable-output

define void @test(i32 %x) {
entry:
  br label %label1
label1:
  %A = phi i32 [ 0, %entry ], [ %A.1, %label2 ]
  %B = icmp slt i32 %A, %x
  br i1 %B, label %label2, label %label2
label2:
  %A.1 = add i32 %A, 1
  br label %label1
label3:  ; No predecessors!
  ret void
}
