; RUN: not llvm-as < %s |& grep {invalid cast opcode}

define i8 @foo(<4 x i64> %x) {
  %y = trunc <4 x i64> %x to i8
  ret i8 %y
}
