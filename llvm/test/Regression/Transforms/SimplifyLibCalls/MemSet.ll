; Test that the LLVMMemSetOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*llvm.memset'

declare void %llvm.memset(sbyte*,ubyte,uint,uint)

implementation   ; Functions:

int %main () {
  %target = alloca [1024 x sbyte]
  %target_p = getelementptr [1024 x sbyte]* %target, int 0, int 0
  call void %llvm.memset(sbyte* %target_p, ubyte 1, uint 0, uint 1)
  call void %llvm.memset(sbyte* %target_p, ubyte 1, uint 1, uint 1)
  call void %llvm.memset(sbyte* %target_p, ubyte 1, uint 2, uint 2)
  call void %llvm.memset(sbyte* %target_p, ubyte 1, uint 4, uint 4)
  call void %llvm.memset(sbyte* %target_p, ubyte 1, uint 8, uint 8)
  ret int 0
}
