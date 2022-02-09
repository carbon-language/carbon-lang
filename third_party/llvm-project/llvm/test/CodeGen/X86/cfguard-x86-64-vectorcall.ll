; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s -check-prefix=X64
; Control Flow Guard is currently only available on Windows


; Test that Control Flow Guard checks are correctly added for x86_64 vector calls.
define void @func_cf_vector_x64(void (%struct.HVA)* %0, %struct.HVA* %1) #0 {
entry:
  %2 = alloca %struct.HVA, align 8
  %3 = bitcast %struct.HVA* %2 to i8*
  %4 = bitcast %struct.HVA* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %3, i8* align 8 %4, i64 32, i1 false)
  %5 = load %struct.HVA, %struct.HVA* %2, align 8
  call x86_vectorcallcc void %0(%struct.HVA inreg %5)
  ret void

  ; X64-LABEL: func_cf_vector_x64
  ; X64:       movq	%rcx, %rax
  ; X64:       movups (%rdx), %xmm0
  ; X64:       movups 16(%rdx), %xmm1
  ; X64:       movaps %xmm0, 32(%rsp)
  ; X64:       movaps %xmm1, 48(%rsp)
  ; X64:       movsd 32(%rsp), %xmm0         # xmm0 = mem[0],zero
  ; X64:       movsd 40(%rsp), %xmm1         # xmm1 = mem[0],zero
  ; X64:       movsd 48(%rsp), %xmm2         # xmm2 = mem[0],zero
  ; X64:       movsd 56(%rsp), %xmm3         # xmm3 = mem[0],zero
  ; X64:       callq *__guard_dispatch_icall_fptr(%rip)
  ; X64-NOT:   callq
}
attributes #0 = { "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }

%struct.HVA = type { double, double, double, double }

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #1
attributes #1 = { argmemonly nounwind willreturn }


!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
