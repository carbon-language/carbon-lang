; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix PTX
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix PTX
; RUN: opt < %s -S -nvptx-favor-non-generic -dce | FileCheck %s --check-prefix IR

@array = internal addrspace(3) global [10 x float] zeroinitializer, align 4
@scalar = internal addrspace(3) global float 0.000000e+00, align 4

; Verifies nvptx-favor-non-generic correctly optimizes generic address space
; usage to non-generic address space usage for the patterns we claim to handle:
; 1. load cast
; 2. store cast
; 3. load gep cast
; 4. store gep cast
; gep and cast can be an instruction or a constant expression. This function
; tries all possible combinations.
define float @ld_st_shared_f32(i32 %i, float %v) {
; IR-LABEL: @ld_st_shared_f32
; IR-NOT: addrspacecast
; PTX-LABEL: ld_st_shared_f32(
  ; load cast
  %1 = load float, float* addrspacecast (float addrspace(3)* @scalar to float*), align 4
; PTX: ld.shared.f32 %f{{[0-9]+}}, [scalar];
  ; store cast
  store float %v, float* addrspacecast (float addrspace(3)* @scalar to float*), align 4
; PTX: st.shared.f32 [scalar], %f{{[0-9]+}};
  ; use syncthreads to disable optimizations across components
  call void @llvm.cuda.syncthreads()
; PTX: bar.sync 0;

  ; cast; load
  %2 = addrspacecast float addrspace(3)* @scalar to float*
  %3 = load float, float* %2, align 4
; PTX: ld.shared.f32 %f{{[0-9]+}}, [scalar];
  ; cast; store
  store float %v, float* %2, align 4
; PTX: st.shared.f32 [scalar], %f{{[0-9]+}};
  call void @llvm.cuda.syncthreads()
; PTX: bar.sync 0;

  ; load gep cast
  %4 = load float, float* getelementptr inbounds ([10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i32 0, i32 5), align 4
; PTX: ld.shared.f32 %f{{[0-9]+}}, [array+20];
  ; store gep cast
  store float %v, float* getelementptr inbounds ([10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i32 0, i32 5), align 4
; PTX: st.shared.f32 [array+20], %f{{[0-9]+}};
  call void @llvm.cuda.syncthreads()
; PTX: bar.sync 0;

  ; gep cast; load
  %5 = getelementptr inbounds [10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i32 0, i32 5
  %6 = load float, float* %5, align 4
; PTX: ld.shared.f32 %f{{[0-9]+}}, [array+20];
  ; gep cast; store
  store float %v, float* %5, align 4
; PTX: st.shared.f32 [array+20], %f{{[0-9]+}};
  call void @llvm.cuda.syncthreads()
; PTX: bar.sync 0;

  ; cast; gep; load
  %7 = addrspacecast [10 x float] addrspace(3)* @array to [10 x float]*
  %8 = getelementptr inbounds [10 x float], [10 x float]* %7, i32 0, i32 %i
  %9 = load float, float* %8, align 4
; PTX: ld.shared.f32 %f{{[0-9]+}}, [%{{(r|rl|rd)[0-9]+}}];
  ; cast; gep; store
  store float %v, float* %8, align 4
; PTX: st.shared.f32 [%{{(r|rl|rd)[0-9]+}}], %f{{[0-9]+}};
  call void @llvm.cuda.syncthreads()
; PTX: bar.sync 0;

  %sum2 = fadd float %1, %3
  %sum3 = fadd float %sum2, %4
  %sum4 = fadd float %sum3, %6
  %sum5 = fadd float %sum4, %9
  ret float %sum5
}

; When hoisting an addrspacecast between different pointer types, replace the
; addrspacecast with a bitcast.
define i32 @ld_int_from_float() {
; IR-LABEL: @ld_int_from_float
; IR: load i32, i32 addrspace(3)* bitcast (float addrspace(3)* @scalar to i32 addrspace(3)*)
; PTX-LABEL: ld_int_from_float(
; PTX: ld.shared.u{{(32|64)}}
  %1 = load i32, i32* addrspacecast(float addrspace(3)* @scalar to i32*), align 4
  ret i32 %1
}

define i32 @ld_int_from_global_float(float addrspace(1)* %input, i32 %i, i32 %j) {
; IR-LABEL: @ld_int_from_global_float(
; PTX-LABEL: ld_int_from_global_float(
  %1 = addrspacecast float addrspace(1)* %input to float*
  %2 = getelementptr float, float* %1, i32 %i
; IR-NEXT: getelementptr float, float addrspace(1)* %input, i32 %i
  %3 = getelementptr float, float* %2, i32 %j
; IR-NEXT: getelementptr float, float addrspace(1)* {{%[^,]+}}, i32 %j
  %4 = bitcast float* %3 to i32*
; IR-NEXT: bitcast float addrspace(1)* {{%[^ ]+}} to i32 addrspace(1)*
  %5 = load i32, i32* %4
; IR-NEXT: load i32, i32 addrspace(1)* {{%.+}}
; PTX-LABEL: ld.global
  ret i32 %5
}

define void @nested_const_expr() {
; PTX-LABEL: nested_const_expr(
  ; store 1 to bitcast(gep(addrspacecast(array), 0, 1))
  store i32 1, i32* bitcast (float* getelementptr ([10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i64 0, i64 1) to i32*), align 4
; PTX: mov.u32 %r1, 1;
; PTX-NEXT: st.shared.u32 [array+4], %r1;
  ret void
}

define void @rauw(float addrspace(1)* %input) {
  %generic_input = addrspacecast float addrspace(1)* %input to float*
  %addr = getelementptr float, float* %generic_input, i64 10
  %v = load float, float* %addr
  store float %v, float* %addr
  ret void
; IR-LABEL: @rauw(
; IR-NEXT: %1 = getelementptr float, float addrspace(1)* %input, i64 10
; IR-NEXT: %v = load float, float addrspace(1)* %1
; IR-NEXT: store float %v, float addrspace(1)* %1
; IR-NEXT: ret void
}

declare void @llvm.cuda.syncthreads() #3

attributes #3 = { noduplicate nounwind }

