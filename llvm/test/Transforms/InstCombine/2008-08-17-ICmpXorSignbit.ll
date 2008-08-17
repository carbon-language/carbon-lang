; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep -v xor

define i1 @test1(i8 %x, i8 %y) {
  %X = xor i8 %x, 128
  %Y = xor i8 %y, 128
  %tmp = icmp slt i8 %X, %Y
  ret i1 %tmp
}

define i1 @test2(i8 %x, i8 %y) {
  %X = xor i8 %x, 128
  %Y = xor i8 %y, 128
  %tmp = icmp ult i8 %X, %Y
  ret i1 %tmp
}

define i1 @test3(i8 %x) {
  %X = xor i8 %x, 128
  %tmp = icmp uge i8 %X, 15
  ret i1 %tmp
}

