; RUN: not llvm-as < %s |& grep {invalid indices for extractvalue}
; PR4170

define void @test() {
entry:
        extractvalue [0 x i32] undef, 0
        ret void
}
