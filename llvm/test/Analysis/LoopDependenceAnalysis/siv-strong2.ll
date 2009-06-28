; RUN: llvm-as < %s | opt -disable-output -analyze -lda > %t
; RUN: grep {instructions: 3} %t | count 1
; RUN: grep {0,2: dependent} %t | count 1
; RUN: grep {1,2: dependent} %t | count 1

; for (i = 0; i < 256; i++)
;   x[i+1] = x[i] + y[i]

@x = common global [256 x i32] zeroinitializer, align 4
@y = common global [256 x i32] zeroinitializer, align 4

define void @foo(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %y.ld.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i
  %i.next = add i64 %i, 1
  %x.st.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i.next
  %x = load i32* %x.ld.addr
  %y = load i32* %y.ld.addr
  %r = add i32 %y, %x
  store i32 %r, i32* %x.st.addr
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
