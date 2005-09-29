; Test that the memcmpOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*memcmp' &&
; RUN: llvm-as < %s | opt -simplify-libcalls -disable-output

declare int %memcmp(sbyte*,sbyte*,int)
%h = constant [2 x sbyte] c"h\00"
%hel = constant [4 x sbyte] c"hel\00"
%hello_u = constant [8 x sbyte] c"hello_u\00"

implementation

void %test(sbyte *%P, sbyte *%Q, int %N, int* %IP, bool *%BP) {
  %A = call int %memcmp(sbyte *%P, sbyte* %P, int %N)
  volatile store int %A, int* %IP
  %B = call int %memcmp(sbyte *%P, sbyte* %Q, int 0)
  volatile store int %B, int* %IP
  %C = call int %memcmp(sbyte *%P, sbyte* %Q, int 1)
  volatile store int %C, int* %IP
  %D = call int %memcmp(sbyte *%P, sbyte* %Q, int 2)
  %E = seteq int %D, 0
  volatile store bool %E, bool* %BP
  ret void
}
