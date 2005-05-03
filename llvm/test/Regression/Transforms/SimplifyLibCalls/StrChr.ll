; Test that the StrChrOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*strchr'

declare sbyte* %strchr(sbyte*,int)
declare int %puts(sbyte*)
%hello = constant [14 x sbyte] c"hello world\n\00"
%null = constant [1 x sbyte] c"\00"

implementation   ; Functions:

int %main () {
  %hello_p = getelementptr [14 x sbyte]* %hello, int 0, int 0
  %null_p = getelementptr [1 x sbyte]* %null, int 0, int 0

  %world  = call sbyte* %strchr(sbyte* %hello_p, int 119 )
  %ignore = call sbyte* %strchr(sbyte* %null_p, int 119 )
  %result = call int %puts(sbyte* %world)
  ret int %result
}
