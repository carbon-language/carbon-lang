; RUN: llc < %s -mtriple=x86_64-applecl-darwin11 | FileCheck %s
; 8193553

define void @__math_kernel_Vectorized_wrapper(<4 x double> addrspace(1)* %a, <4 x double> addrspace(1)* %b, i64 addrspace(1)* %c, i64 addrspace(1)* %d) nounwind {
entry.i:                                          ; preds = %entry.i, %loop
; CHECK: math_kernel_Vectorized_wrapper
; CHECK-NOT: cmpordsd (%rsi),
  %0 = alloca i8
  %1 = alloca i8
  %2 = alloca i8
  %tmp213.i = load <4 x double> addrspace(1)* %a ; <<4 x double>> [#uses=4]
  %extract25.i = extractelement <4 x double> %tmp213.i, i32 1 ; <double> [#uses=1]
  %tmp723.i = load <4 x double> addrspace(1)* %b ; <<4 x double>> [#uses=4]
  %extract29.i = extractelement <4 x double> %tmp723.i, i32 1 ; <double> [#uses=1]
  %tmp2.i26 = insertelement <2 x double> undef, double %extract25.i, i32 0 ; <<2 x double>> [#uses=1]
  %tmp5.i27 = insertelement <2 x double> undef, double %extract29.i, i32 1 ; <<2 x double>> [#uses=1]
  %cmpsd.i.i28 = call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %tmp2.i26, <2 x double> %tmp5.i27, i8 7) nounwind ; <<2 x double>> [#uses=1]
  %3 = bitcast <2 x double> %cmpsd.i.i28 to <4 x i32> ; <<4 x i32>> [#uses=1]
  %tmp12.i29 = extractelement <4 x i32> %3, i32 0 ; <i32> [#uses=1]
  %and.i30 = and i32 %tmp12.i29, 1                ; <i32> [#uses=1]
  %conv937.i36 = zext i32 %and.i30 to i64         ; <i64> [#uses=1]
  store i64 %conv937.i36, i64 addrspace(1)* %d
  ret void
; CHECK: ret
}

declare <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double>, <2 x double>, i8) nounwind readnone
