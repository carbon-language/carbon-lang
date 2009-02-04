; RUN: llvm-as < %s | opt -basicaa -gvn | llvm-dis | grep load | count 1

@flag0 = internal global i32 zeroinitializer
@turn = internal global i32 zeroinitializer


define i32 @main() {
  %a = load i32* @flag0
	%b = tail call i32 @llvm.atomic.swap.i32.p0i32(i32* @turn, i32 1)
  %c = load i32* @flag0
	ret i32 %c
}

declare i32 @llvm.atomic.swap.i32.p0i32(i32*, i32) nounwind