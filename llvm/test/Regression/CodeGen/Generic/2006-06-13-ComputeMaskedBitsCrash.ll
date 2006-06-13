; RUN: llvm-as < %s | llc -fast

uint %test1(uint %tmp1) {
  %tmp2 = or uint %tmp1, 2147483648
  %tmp3 = shr uint %tmp2, ubyte 31
  %tmp4 = and uint %tmp3, 2147483648
  %tmp5 = seteq uint %tmp4, 0
  br bool %tmp5, label %cond_true, label %cond_false
  
cond_true:
  ret uint %tmp1
  
cond_false:

  ret uint %tmp2
}


uint %test2(uint %tmp1) {
  %tmp2 = or uint %tmp1, 2147483648
  %tmp3 = cast uint %tmp2 to int
  %tmp4 = shr int %tmp3, ubyte 31
  %tmp5 = cast int %tmp4 to uint
  %tmp6 = and uint %tmp5, 2147483648
  %tmp7 = seteq uint %tmp6, 0
  br bool %tmp7, label %cond_true, label %cond_false
  
cond_true:
  ret uint %tmp1
  
cond_false:

  ret uint %tmp2
}


uint %test3(uint %tmp1) {
  %tmp2 = or uint %tmp1, 1
  %tmp3 = shl uint %tmp2, ubyte 31
  %tmp4 = and uint %tmp3, 1
  %tmp5 = seteq uint %tmp4, 0
  br bool %tmp5, label %cond_true, label %cond_false
  
cond_true:
  ret uint %tmp1
  
cond_false:

  ret uint %tmp2
}
