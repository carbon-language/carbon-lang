; Checks to see that instcombine can handle a sign extension of i1

; RUN: opt < %s -instcombine -S | FileCheck %s

define void @test(<2 x i16> %srcA, <2 x i16> %srcB, <2 x i16>* %dst) nounwind {
entry:
; CHECK-NOT: tmask
; CHECK: ret
  %cmp = icmp eq <2 x i16> %srcB, %srcA;
  %sext = sext <2 x i1> %cmp to <2 x i16>;
  %tmask = ashr <2 x i16> %sext, <i16 15, i16 15> ;
  store <2 x i16> %tmask, <2 x i16>* %dst;                                                                   
  ret void                                                                                                                      
}                                                                                                                               
