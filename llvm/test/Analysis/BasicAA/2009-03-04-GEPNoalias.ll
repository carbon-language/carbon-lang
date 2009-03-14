; RUN: llvm-as < %s | opt -aa-eval -basicaa |& grep {0 no alias}

declare noalias i32* @noalias()

define void @test(i32 %x) {
  %a = call i32* @noalias()
  %b = getelementptr i32* %a, i32 %x
  ret void
}
