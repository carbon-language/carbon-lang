; Test that the SPrintFOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls -disable-output &&
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*sprintf'

declare int %sprintf(sbyte*,sbyte*,...)
declare int %puts(sbyte*)
%hello = constant [6 x sbyte] c"hello\00"
%null = constant [1 x sbyte] c"\00"
%null_hello = constant [7 x sbyte] c"\00hello\00"
%fmt1 = constant [3 x sbyte] c"%s\00"
%fmt2 = constant [3 x sbyte] c"%c\00"

implementation   ; Functions:

int %foo (sbyte* %p) {
  %target = alloca [1024 x sbyte]
  %target_p = getelementptr [1024 x sbyte]* %target, int 0, int 0
  %hello_p = getelementptr [6 x sbyte]* %hello, int 0, int 0
  %null_p = getelementptr [1 x sbyte]* %null, int 0, int 0
  %nh_p = getelementptr [7 x sbyte]* %null_hello, int 0, int 0
  %fmt1_p = getelementptr [3 x sbyte]* %fmt1, int 0, int 0
  %fmt2_p = getelementptr [3 x sbyte]* %fmt2, int 0, int 0
  store sbyte 0, sbyte* %target_p
  %r1 = call int (sbyte*,sbyte*,...)* %sprintf(sbyte* %target_p, sbyte* %hello_p)
  %r2 = call int (sbyte*,sbyte*,...)* %sprintf(sbyte* %target_p, sbyte* %null_p)
  %r3 = call int (sbyte*,sbyte*,...)* %sprintf(sbyte* %target_p, sbyte* %nh_p)
  %r4 = call int (sbyte*,sbyte*,...)* %sprintf(sbyte* %target_p, sbyte* %fmt1_p, sbyte* %hello_p)
  %r4.1 = call int (sbyte*,sbyte*,...)* %sprintf(sbyte* %target_p, sbyte* %fmt1_p, sbyte* %p)
  %r5 = call int (sbyte*,sbyte*,...)* %sprintf(sbyte* %target_p, sbyte* %fmt2_p, int 82)
  %r6 = add int %r1, %r2
  %r7 = add int %r3, %r6
  %r8 = add int %r5, %r7
  %r9 = add int %r8, %r4
  %r10 = add int %r9, %r4.1
  ret int %r10
}
