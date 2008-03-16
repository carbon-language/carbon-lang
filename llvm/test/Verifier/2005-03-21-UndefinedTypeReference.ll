; RUN: not llvm-as < %s |& grep {Reference to an undefined type}

define void @test() {
        malloc %InvalidType
        ret void
}

