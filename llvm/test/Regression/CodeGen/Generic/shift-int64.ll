; RUN: llvm-as < %s | llc

long %test_imm(long %X) {
   %Y = shr long %X, ubyte 17 
   ret long %Y
}

long %test_variable(long %X, ubyte %Amt) {
   %Y = shr long %X, ubyte %Amt
   ret long %Y
}
