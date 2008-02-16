; RUN: llvm-as < %s | opt -inline | llvm-dis | \
; RUN:   not grep {%G = alloca int}

; In this testcase, %bar stores to the global G.  Make sure that inlining does
; not cause it to store to the G in main instead.
@G = global i32 7               ; <i32*> [#uses=1]

define i32 @main() {
    %G = alloca i32         ; <i32*> [#uses=2]
    store i32 0, i32* %G
    call void @bar( )
    %RV = load i32* %G              ; <i32> [#uses=1]
    ret i32 %RV
}

define internal void @bar() {
    store i32 123, i32* @G
    ret void
}


