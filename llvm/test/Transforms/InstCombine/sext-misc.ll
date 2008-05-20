; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep sext
; RUN: llvm-as < %s | llc -march=x86-64 | not grep movslq
; RUN: llvm-as < %s | llc -march=x86 | not grep sar

declare i32 @llvm.ctpop.i32(i32)
declare i32 @llvm.ctlz.i32(i32)
declare i32 @llvm.cttz.i32(i32)

define i64 @foo(i32 %x) {
  %t = call i32 @llvm.ctpop.i32(i32 %x)
  %s = sext i32 %t to i64
  ret i64 %s
}
define i64 @boo(i32 %x) {
  %t = call i32 @llvm.ctlz.i32(i32 %x)
  %s = sext i32 %t to i64
  ret i64 %s
}
define i64 @zoo(i32 %x) {
  %t = call i32 @llvm.cttz.i32(i32 %x)
  %s = sext i32 %t to i64
  ret i64 %s
}
define i64 @coo(i32 %x) {
  %t = udiv i32 %x, 3
  %s = sext i32 %t to i64
  ret i64 %s
}
define i64 @moo(i32 %x) {
  %t = urem i32 %x, 30000
  %s = sext i32 %t to i64
  ret i64 %s
}
define i64 @yoo(i32 %x) {
  %u = lshr i32 %x, 3
  %t = mul i32 %u, 3
  %s = sext i32 %t to i64
  ret i64 %s
}
define i64 @voo(i32 %x) {
  %t = and i32 %x, 511
  %u = sub i32 20000, %t
  %s = sext i32 %u to i64
  ret i64 %s
}
define i32 @woo(i8 %a, i32 %f, i1 %p, i32* %z) {
  %d = lshr i32 %f, 24
  %e = select i1 %p, i32 %d, i32 0
  %s = trunc i32 %e to i16
  %n = sext i16 %s to i32
  ret i32 %n
}
