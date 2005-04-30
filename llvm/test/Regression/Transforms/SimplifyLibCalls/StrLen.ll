; Test that the StrCatOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*strlen'

declare uint %strlen(sbyte*)
%hello      = constant [6 x sbyte] c"hello\00"
%null       = constant [1 x sbyte] c"\00"
%null_hello = constant [7 x sbyte] c"\00hello\00"

implementation   ; Functions:

int %main () {
  %hello_p      = getelementptr [6 x sbyte]* %hello, int 0, int 0
  %null_p       = getelementptr [1 x sbyte]* %null, int 0, int 0
  %null_hello_p = getelementptr [7 x sbyte]* %null_hello, int 0, int 0
  %hello_l      = call uint %strlen(sbyte* %hello_p)
  %null_l       = call uint %strlen(sbyte* %null_p)
  %null_hello_l = call uint %strlen(sbyte* %null_hello_p)
  %sum1 = add uint %hello_l, %null_l
  %sum2 = add uint %sum1, %null_hello_l
  %result = cast uint %sum2 to int
  ret int %result
}
