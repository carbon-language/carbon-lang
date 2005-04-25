; Test that the StrCatOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | grep strlen
; XFAIL: *

declare sbyte* %strcat(sbyte*,sbyte*)
%hello = constant [6 x sbyte] c"hello\00"

implementation   ; Functions:

int %main () {
  %target = alloca [1024 x sbyte]
  %arg1 = getelementptr [1024 x sbyte]* %target, int 0, int 0
  store sbyte 0, sbyte* %arg1
  %arg2 = getelementptr [6 x sbyte]* %hello, int 0, int 0
  %rslt = call sbyte* %strcat(sbyte* %arg1, sbyte* %arg2)
  ret int 0
}
