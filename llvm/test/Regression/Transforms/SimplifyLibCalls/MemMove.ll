; Test that the StrCatOptimizer works correctly
; RUN: llvm-as < %s | opt -constprop -simplify-libcalls -disable-output &&
; RUN: llvm-as < %s | opt -constprop -simplify-libcalls | llvm-dis | not grep 'call.*llvm.memmove'

declare sbyte* %llvm.memmove(sbyte*,sbyte*,int,int)
%h = constant [2 x sbyte] c"h\00"
%hel = constant [4 x sbyte] c"hel\00"
%hello_u = constant [8 x sbyte] c"hello_u\00"

implementation   ; Functions:

int %main () {
  %h_p = getelementptr [2 x sbyte]* %h, int 0, int 0
  %hel_p = getelementptr [4 x sbyte]* %hel, int 0, int 0
  %hello_u_p = getelementptr [8 x sbyte]* %hello_u, int 0, int 0
  %target = alloca [1024 x sbyte]
  %target_p = getelementptr [1024 x sbyte]* %target, int 0, int 0
  call sbyte* %llvm.memmove(sbyte* %target_p, sbyte* %h_p, int 2, int 2)
  call sbyte* %llvm.memmove(sbyte* %target_p, sbyte* %hel_p, int 4, int 4)
  call sbyte* %llvm.memmove(sbyte* %target_p, sbyte* %hello_u_p, int 8, int 8)
  ret int 0
}
