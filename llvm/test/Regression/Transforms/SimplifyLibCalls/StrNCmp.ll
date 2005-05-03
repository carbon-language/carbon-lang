; Test that the StrCmpOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*strcmp'

declare int %strncmp(sbyte*,sbyte*,int)
declare int %puts(sbyte*)
%hello = constant [6 x sbyte] c"hello\00"
%hell = constant [5 x sbyte] c"hell\00"
%null = constant [1 x sbyte] c"\00"

implementation   ; Functions:

int %main () {
  %hello_p = getelementptr [6 x sbyte]* %hello, int 0, int 0
  %hell_p  = getelementptr [5 x sbyte]* %hell, int 0, int 0
  %null_p  = getelementptr [1 x sbyte]* %null, int 0, int 0
  %temp1 = call int %strncmp(sbyte* %hello_p, sbyte* %hello_p,int 5)
  %temp2 = call int %strncmp(sbyte* %null_p, sbyte* %null_p,int 0)
  %temp3 = call int %strncmp(sbyte* %hello_p, sbyte* %null_p,int 0)
  %temp4 = call int %strncmp(sbyte* %null_p, sbyte* %hello_p,int 0)
  %temp5 = call int %strncmp(sbyte* %hell_p, sbyte* %hello_p,int 4)
  %rslt1 = add int %temp1, %temp2
  %rslt2 = add int %rslt1, %temp3
  %rslt3 = add int %rslt2, %temp4
  %rslt4 = add int %rslt3, %temp5
  ret int %rslt4
}
