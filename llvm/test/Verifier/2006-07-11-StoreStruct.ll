; RUN: not llvm-as < %s |& grep {Instruction operands must be first-class}
; PR826

        %struct_4 = type { i32 }

define void @test() {
        store %struct_4 zeroinitializer, %struct_4* null
        unreachable
}
