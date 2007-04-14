; Tests to make sure bit counts of constants are folded
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i32 19}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep -v declare | not grep llvm.ct

declare i32 @llvm.ctpop.i31(i31 %val) 
declare i32 @llvm.cttz.i32(i32 %val) 
declare i32 @llvm.ctlz.i33(i33 %val) 

define i32 @test(i32 %A) {
  %c1 = call i32 @llvm.ctpop.i31(i31 12415124)
  %c2 = call i32 @llvm.cttz.i32(i32 87359874)
  %c3 = call i32 @llvm.ctlz.i33(i33 87359874)
  %r1 = add i32 %c1, %c2
  %r2 = add i32 %r1, %c3
  ret i32 %r2
}
