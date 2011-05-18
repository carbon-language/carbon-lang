; RUN: not llvm-as < %s |& grep {invalid cast opcode}

define <3 x i8> @foo(<4 x i64> %x) {
  %y = trunc <4 x i64> %x to <3 x i8>
  ret <3 x i8> %y
}
