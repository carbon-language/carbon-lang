; RUN: not llvm-as < %s |& FileCheck %s

; CHECK: invalid cast opcode for cast from '<4 x i64>' to '<3 x i8>'

define <3 x i8> @foo(<4 x i64> %x) {
  %y = trunc <4 x i64> %x to <3 x i8>
  ret <3 x i8> %y
}
