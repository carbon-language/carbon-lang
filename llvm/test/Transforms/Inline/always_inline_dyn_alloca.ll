; RUN: opt < %s -inline -S | not grep callee
; rdar://6655932

; If callee is marked alwaysinline, inline it! Even if callee has dynamic
; alloca and caller does not,

define internal void @callee(i32 %N) alwaysinline {
        %P = alloca i32, i32 %N
        ret void
}

define void @foo(i32 %N) {
        call void @callee( i32 %N )
        ret void
}
