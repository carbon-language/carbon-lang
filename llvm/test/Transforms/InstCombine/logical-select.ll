; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep select | count 2

define i32 @foo(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
  %e = icmp slt i32 %a, %b
  %f = sext i1 %e to i32
  %g = and i32 %c, %f
  %h = xor i32 %f, -1
  %i = and i32 %d, %h
  %j = or i32 %g, %i
  ret i32 %j
}
define i32 @bar(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
  %e = icmp slt i32 %a, %b
  %f = sext i1 %e to i32
  %g = and i32 %c, %f
  %h = xor i32 %f, -1
  %i = and i32 %d, %h
  %j = or i32 %i, %g
  ret i32 %j
}
