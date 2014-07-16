; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64

; Ensure we access the local stack properly

; PTX32:        mov.u32          %r{{[0-9]+}}, __local_depot{{[0-9]+}};
; PTX32:        cvta.local.u32   %SP, %r{{[0-9]+}};
; PTX32:        ld.param.u32     %r{{[0-9]+}}, [foo_param_0];
; PTX32:        st.volatile.u32  [%SP+0], %r{{[0-9]+}};
; PTX64:        mov.u64          %rd{{[0-9]+}}, __local_depot{{[0-9]+}};
; PTX64:        cvta.local.u64   %SP, %rd{{[0-9]+}};
; PTX64:        ld.param.u32     %r{{[0-9]+}}, [foo_param_0];
; PTX64:        st.volatile.u32  [%SP+0], %r{{[0-9]+}};
define void @foo(i32 %a) {
  %local = alloca i32, align 4
  store volatile i32 %a, i32* %local
  ret void
}
