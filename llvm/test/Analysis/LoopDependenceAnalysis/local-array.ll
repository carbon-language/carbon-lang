; RUN: llvm-as < %s | opt -disable-output -analyze -lda > %t
; RUN: grep {instructions: 2} %t | count 1
; RUN: grep {0,1: dependent} %t | count 1

; x[5] = x[6] // with x being an array on the stack

define void @foo(...) nounwind {
entry:
  %xptr = alloca [256 x i32], align 4
  %x.ld.addr = getelementptr [256 x i32]* %xptr, i64 0, i64 6
  %x.st.addr = getelementptr [256 x i32]* %xptr, i64 0, i64 5
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %x = load i32* %x.ld.addr
  store i32 %x, i32* %x.st.addr
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
