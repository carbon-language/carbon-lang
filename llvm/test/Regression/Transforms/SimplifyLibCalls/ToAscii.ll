; Test that the ToAsciiOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*toascii'

declare int %toascii(int)

implementation   ; Functions:

int %main () {
  %val1 = call int %toascii(int 1)
  %val2 = call int %toascii(int 0)
  %val3 = call int %toascii(int 127)
  %val4 = call int %toascii(int 128)
  %val5 = call int %toascii(int 255)
  %val6 = call int %toascii(int 256)
  %rslt1 = add int %val1, %val2
  %rslt2 = add int %val3, %val4
  %rslt3 = add int %val5, %val6
  %rslt4 = add int %rslt1, %rslt2
  %rslt5 = add int %rslt4, %rslt3
  ret int %rslt5
}
