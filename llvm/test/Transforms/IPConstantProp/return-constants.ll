; RUN: llvm-as < %s | opt -ipconstprop | llvm-dis | grep {add i32 21, 21}

define internal {i32, i32} @foo(i1 %C) {
        br i1 %C, label %T, label %F

T:              ; preds = %0
        ret i32 21, i32 21

F:              ; preds = %0
        ret i32 21, i32 21
}

define i32 @caller(i1 %C) {
        %X = call {i32, i32} @foo( i1 %C )
        %A = getresult {i32, i32} %X, 0
        %B = getresult {i32, i32} %X, 1
        %Y = add i32 %A, %B
        ret i32 %Y
}

