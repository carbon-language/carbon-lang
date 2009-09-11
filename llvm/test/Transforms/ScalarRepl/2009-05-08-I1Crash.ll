; RUN: opt < %s -scalarrepl | llvm-dis
; PR4146

 %wrapper = type { i1 }

define void @f() {
entry:
        %w = alloca %wrapper, align 8           ; <%wrapper*> [#uses=1]
        %0 = getelementptr %wrapper* %w, i64 0, i32 0           ; <i1*>
        store i1 true, i1* %0
        ret void
}
