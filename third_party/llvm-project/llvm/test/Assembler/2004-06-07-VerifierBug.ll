; RUN: llvm-as < %s > /dev/null
; RUN: verify-uselistorder %s

define void @t() {
entry:
     ret void

loop:           ; preds = %loop
     %tmp.4.i9 = getelementptr i32, i32* null, i32 %tmp.5.i10             ; <i32*> [#uses=1]
     %tmp.5.i10 = load i32, i32* %tmp.4.i9                ; <i32> [#uses=1]
     br label %loop
}
