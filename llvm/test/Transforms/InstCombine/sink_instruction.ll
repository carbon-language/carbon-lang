; RUN: opt -instcombine %s -S | FileCheck %s

;; This tests that the instructions in the entry blocks are sunk into each
;; arm of the 'if'.

define i32 @foo(i1 %C, i32 %A, i32 %B) {
entry:
        %tmp.2 = sdiv i32 %A, %B                ; <i32> [#uses=1]
        %tmp.9 = add i32 %B, %A         ; <i32> [#uses=1]
        br i1 %C, label %then, label %endif

then:           ; preds = %entry
        ret i32 %tmp.9

endif:          ; preds = %entry
; CHECK: sdiv i32
; CHECK-NEXT: ret i32
        ret i32 %tmp.2
}

