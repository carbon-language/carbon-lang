; RUN: llvm-as < %s | opt -instcombine | llvm-dis | FileCheck %s

; The load replacing the extract element must occur before the call
; that may modify local array a.

declare void @mod_a_func(<4 x float>* %a);

; CHECK: load float* %arraydecay1, align 16
; CHECK: call void @mod_a_func

define void @cl_jpegenc_k2(<4 x float> addrspace(1)* %src, float addrspace(1)* %dst) {
  %a = alloca [2 x <4 x float>], align 16
  %arraydecay = getelementptr [2 x <4 x float>]* %a, i32 0, i32 0
  %arrayidx31 = getelementptr <4 x float> addrspace(1)* %src, i32 0
  %tmp32 = load <4 x float> addrspace(1)* %arrayidx31
  store <4 x float> %tmp32, <4 x float>* %arraydecay, align 16
  %tmp86 = load <4 x float>* %arraydecay, align 16
  call void @mod_a_func(<4 x float>* %arraydecay)
  %arrayidx132 = getelementptr float addrspace(1)* %dst, i32 0
  %tmp236 = extractelement <4 x float> %tmp86, i32 0
  store float %tmp236, float addrspace(1)* %arrayidx132
  ret void
}