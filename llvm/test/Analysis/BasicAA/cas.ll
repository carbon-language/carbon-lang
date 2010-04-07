; RUN: opt < %s -basicaa -gvn -instcombine -S | grep {ret i32 0}

@flag0 = internal global i32 zeroinitializer
@turn = internal global i32 zeroinitializer


define i32 @main() {
  %a = load i32* @flag0
  %b = tail call i32 @llvm.atomic.swap.i32.p0i32(i32* @turn, i32 1)
  %c = load i32* @flag0
  %d = sub i32 %a, %c
  ret i32 %d
}

declare i32 @llvm.atomic.swap.i32.p0i32(i32*, i32) nounwind
