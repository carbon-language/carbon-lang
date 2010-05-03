; RUN: llvm-as < %s | opt -loop-index-split | llvm-dis | not grep clone

declare void @f()

define i32 @main() {
entry:
        br label %head
head:
        %i = phi i32 [0, %entry], [%i1, %tail]
        call void @f()
        %splitcond = icmp slt i32 %i, 2
        br i1 %splitcond, label %yes, label %no
yes:
        br label %tail
no:
        br label %tail
tail:
        %i1 = add i32 %i, 1
        %exitcond = icmp slt i32 %i1, 4
        br i1 %exitcond, label %head, label %exit
exit:
        ret i32 0
}
