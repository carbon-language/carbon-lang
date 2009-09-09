; RUN: llc < %s -march=alpha | grep .weak.*f
; RUN: llc < %s -march=alpha | grep .weak.*h

define weak i32 @f() {
entry:
        unreachable
}

define void @g() {
entry:
        tail call void @h( )
        ret void
}

declare extern_weak void @h()

