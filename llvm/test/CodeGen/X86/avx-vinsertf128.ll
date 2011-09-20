; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck -check-prefix=CHECK-SSE %s

; CHECK-NOT: vunpck
; CHECK: vinsertf128 $1
define <8 x float> @A(<8 x float> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> undef, <8 x i32> <i32 8, i32 8, i32 8, i32 8, i32 0, i32 1, i32 2, i32 3>
  ret <8 x float> %shuffle
}

; CHECK-NOT: vunpck
; CHECK: vinsertf128 $1
define <4 x double> @B(<4 x double> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x double> %a, <4 x double> undef, <4 x i32> <i32 4, i32 4, i32 0, i32 1>
  ret <4 x double> %shuffle
}

declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

declare <2 x double> @llvm.x86.sse2.min.sd(<2 x double>, <2 x double>) nounwind readnone

; Just check that no crash happens
; CHECK-SSE: _insert_crash
define void @insert_crash() nounwind {
allocas:
  %v1.i.i451 = shufflevector <4 x double> zeroinitializer, <4 x double> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %ret_0a.i.i.i452 = shufflevector <4 x double> %v1.i.i451, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %vret_0.i.i.i454 = tail call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %ret_0a.i.i.i452, <2 x double> undef) nounwind
  %ret_val.i.i.i463 = tail call <2 x double> @llvm.x86.sse2.min.sd(<2 x double> %vret_0.i.i.i454, <2 x double> undef) nounwind
  %ret.i1.i.i464 = extractelement <2 x double> %ret_val.i.i.i463, i32 0
  %double2float = fptrunc double %ret.i1.i.i464 to float
  %smearinsert50 = insertelement <4 x float> undef, float %double2float, i32 3
  %blendAsInt.i503 = bitcast <4 x float> %smearinsert50 to <4 x i32>
  store <4 x i32> %blendAsInt.i503, <4 x i32>* undef, align 4
  ret void
}

