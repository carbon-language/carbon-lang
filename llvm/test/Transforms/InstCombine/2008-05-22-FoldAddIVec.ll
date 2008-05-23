; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep mul

define <3 x i8> @f(<3 x i8> %i) {
  %A = add <3 x i8> %i, %i
  ret <3 x i8> %A
}
