; This testcase was failing because without merging the return blocks, ADCE
; didn't know that it could get rid of the then.0 block.

; RUN: llvm-as < %s | opt -adce | llvm-dis | not grep load


define void @main(i32 %argc, i8** %argv) {
entry:
        call void @__main( )
        %tmp.1 = icmp ule i32 %argc, 5          ; <i1> [#uses=1]
        br i1 %tmp.1, label %then.0, label %return

then.0:         ; preds = %entry
        %tmp.8 = load i8** %argv                ; <i8*> [#uses=1]
        %tmp.10 = load i8* %tmp.8               ; <i8> [#uses=1]
        %tmp.11 = icmp eq i8 %tmp.10, 98                ; <i1> [#uses=1]
        br i1 %tmp.11, label %then.1, label %return

then.1:         ; preds = %then.0
        ret void

return:         ; preds = %then.0, %entry
        ret void
}

declare void @__main()

