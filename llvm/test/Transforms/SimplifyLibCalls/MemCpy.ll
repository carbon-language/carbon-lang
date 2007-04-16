; Test that the StrCatOptimizer works correctly
; RUN: llvm-upgrade < %s | llvm-as | opt -constprop -simplify-libcalls | \
; RUN:   llvm-dis | not grep {call.*llvm.memcpy.i32}

declare void %llvm.memcpy.i32(sbyte*,sbyte*,uint,uint)
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
  call void %llvm.memcpy.i32(sbyte* %target_p, sbyte* %h_p, uint 2, uint 2)
  call void %llvm.memcpy.i32(sbyte* %target_p, sbyte* %hel_p, uint 4, uint 4)
  call void %llvm.memcpy.i32(sbyte* %target_p, sbyte* %hello_u_p, uint 8, uint 8)
  ret int 0
}
