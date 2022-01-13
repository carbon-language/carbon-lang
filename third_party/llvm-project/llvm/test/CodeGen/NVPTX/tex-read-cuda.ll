; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=SM20
; RUN: llc < %s -march=nvptx -mcpu=sm_30 | FileCheck %s --check-prefix=SM30


target triple = "nvptx-unknown-cuda"

declare { float, float, float, float } @llvm.nvvm.tex.unified.1d.v4f32.s32(i64, i32)
declare i64 @llvm.nvvm.texsurf.handle.internal.p1i64(i64 addrspace(1)*)

; SM20-LABEL: .entry foo
; SM30-LABEL: .entry foo
define void @foo(i64 %img, float* %red, i32 %idx) {
; SM20: ld.param.u64    %rd[[TEXREG:[0-9]+]], [foo_param_0];
; SM20: tex.1d.v4.f32.s32 {%f[[RED:[0-9]+]], %f[[GREEN:[0-9]+]], %f[[BLUE:[0-9]+]], %f[[ALPHA:[0-9]+]]}, [%rd[[TEXREG]], {%r{{[0-9]+}}}]
; SM30: ld.param.u64    %rd[[TEXREG:[0-9]+]], [foo_param_0];
; SM30: tex.1d.v4.f32.s32 {%f[[RED:[0-9]+]], %f[[GREEN:[0-9]+]], %f[[BLUE:[0-9]+]], %f[[ALPHA:[0-9]+]]}, [%rd[[TEXREG]], {%r{{[0-9]+}}}]
  %val = tail call { float, float, float, float } @llvm.nvvm.tex.unified.1d.v4f32.s32(i64 %img, i32 %idx)
  %ret = extractvalue { float, float, float, float } %val, 0
; SM20: st.global.f32 [%r{{[0-9]+}}], %f[[RED]]
; SM30: st.global.f32 [%r{{[0-9]+}}], %f[[RED]]
  store float %ret, float* %red
  ret void
}


@tex0 = internal addrspace(1) global i64 0, align 8

; SM20-LABEL: .entry bar
; SM30-LABEL: .entry bar
define void @bar(float* %red, i32 %idx) {
; SM30: mov.u64 %rd[[TEXHANDLE:[0-9]+]], tex0 
  %texHandle = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1i64(i64 addrspace(1)* @tex0)
; SM20: tex.1d.v4.f32.s32 {%f[[RED:[0-9]+]], %f[[GREEN:[0-9]+]], %f[[BLUE:[0-9]+]], %f[[ALPHA:[0-9]+]]}, [tex0, {%r{{[0-9]+}}}]
; SM30: tex.1d.v4.f32.s32 {%f[[RED:[0-9]+]], %f[[GREEN:[0-9]+]], %f[[BLUE:[0-9]+]], %f[[ALPHA:[0-9]+]]}, [%rd[[TEXHANDLE]], {%r{{[0-9]+}}}]
  %val = tail call { float, float, float, float } @llvm.nvvm.tex.unified.1d.v4f32.s32(i64 %texHandle, i32 %idx)
  %ret = extractvalue { float, float, float, float } %val, 0
; SM20: st.global.f32 [%r{{[0-9]+}}], %f[[RED]]
; SM30: st.global.f32 [%r{{[0-9]+}}], %f[[RED]]
  store float %ret, float* %red
  ret void
}

!nvvm.annotations = !{!1, !2, !3}
!1 = !{void (i64, float*, i32)* @foo, !"kernel", i32 1}
!2 = !{void (float*, i32)* @bar, !"kernel", i32 1}
!3 = !{i64 addrspace(1)* @tex0, !"texture", i32 1}
