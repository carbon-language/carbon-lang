; Test that the IsDigitOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call' 

declare int %isdigit(int)
declare int %isascii(int)

implementation   ; Functions:

int %main () {
  %val1 = call int %isdigit(int 47)
  %val2 = call int %isdigit(int 48)
  %val3 = call int %isdigit(int 57)
  %val4 = call int %isdigit(int 58)
  %rslt1 = add int %val1, %val2
  %rslt2 = add int %val3, %val4
  %sum = add int %rslt1, %rslt2
  %rslt = call int %isdigit(int %sum)
  %tmp = call int %isascii(int %rslt)
  ret int %tmp
}
