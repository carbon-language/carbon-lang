; RUN: not llvm-as < %s |& FileCheck %s
; PR4170

; CHECK: invalid indices for extractvalue

define void @test() {
entry:
        extractvalue [0 x i32] undef, 0
        ret void
}
