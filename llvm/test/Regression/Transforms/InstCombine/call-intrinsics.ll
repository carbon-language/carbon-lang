; RUN: llvm-as < %s | opt -instcombine | llvm-dis

declare void %llvm.memmove(sbyte*, sbyte*, uint, uint)
declare void %llvm.memcpy(sbyte*, sbyte*, uint, uint)
declare void %llvm.memset(sbyte*, ubyte, uint, uint)

%X = global sbyte 0
%Y = global sbyte 12

void %zero_byte_test() {
  ; These process zero bytes, so they are a noop.
  call void %llvm.memmove(sbyte* %X, sbyte* %Y, uint 0, uint 100)
  call void %llvm.memcpy(sbyte* %X, sbyte* %Y, uint 0, uint 100)
  call void %llvm.memset(sbyte* %X, ubyte 123, uint 0, uint 100)
  ret void
}

