; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: invalid cast opcode for cast from '<4 x i32*>' to '<2 x i32*>'
define <2 x i32*> @illegal_vector_pointer_bitcast_num_elements(<4 x i32*> %c) {
  %bc = bitcast <4 x i32*> %c to <2 x i32*>
  ret <2 x i32*> %bc
}
