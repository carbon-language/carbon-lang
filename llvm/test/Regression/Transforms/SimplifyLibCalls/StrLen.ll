; Test that the StrCatOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*strlen'

declare uint %strlen(sbyte*)
%hello      = constant [6 x sbyte] c"hello\00"
%null       = constant [1 x sbyte] c"\00"
%null_hello = constant [7 x sbyte] c"\00hello\00"

implementation   ; Functions:

uint %test1() {
  %hello_p      = getelementptr [6 x sbyte]* %hello, int 0, int 0
  %hello_l      = call uint %strlen(sbyte* %hello_p)
  ret uint %hello_l
}

uint %test2() {
  %null_p       = getelementptr [1 x sbyte]* %null, int 0, int 0
  %null_l       = call uint %strlen(sbyte* %null_p)
  ret uint %null_l
}

uint %test3() {
  %null_hello_p = getelementptr [7 x sbyte]* %null_hello, int 0, int 0
  %null_hello_l = call uint %strlen(sbyte* %null_hello_p)
  ret uint %null_hello_l
}

bool %test4() {
  %hello_p      = getelementptr [6 x sbyte]* %hello, int 0, int 0
  %hello_l      = call uint %strlen(sbyte* %hello_p)
  %eq_hello     = seteq uint %hello_l, 0
  ret bool %eq_hello
}

bool %test5() {
  %null_p       = getelementptr [1 x sbyte]* %null, int 0, int 0
  %null_l       = call uint %strlen(sbyte* %null_p)
  %eq_null      = seteq uint %null_l, 0
  ret bool %eq_null
}

bool %test6() {
  %hello_p      = getelementptr [6 x sbyte]* %hello, int 0, int 0
  %hello_l      = call uint %strlen(sbyte* %hello_p)
  %ne_hello     = setne uint %hello_l, 0
  ret bool %ne_hello
}

bool %test7() {
  %null_p       = getelementptr [1 x sbyte]* %null, int 0, int 0
  %null_l       = call uint %strlen(sbyte* %null_p)
  %ne_null      = setne uint %null_l, 0
  ret bool %ne_null
}
