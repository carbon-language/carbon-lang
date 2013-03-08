; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @test_a
; CHECK-NOT: CND
; CHECK: SET{{[NEQGTL]+}}_DX10

define void @test_a(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ult float %in, 0.000000e+00
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  %4 = bitcast i32 %3 to float
  %5 = bitcast float %4 to i32
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %IF, label %ENDIF

IF:
  %7 = getelementptr i32 addrspace(1)* %out, i32 1
  store i32 0, i32 addrspace(1)* %7
  br label %ENDIF

ENDIF:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; Same as test_a, but the branch labels are swapped to produce the inverse cc
; for the icmp instruction

; CHECK: @test_b
; CHECK: SET{{[GTEQN]+}}_DX10
; CHECK-NEXT: PRED_
define void @test_b(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ult float %in, 0.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  %4 = bitcast i32 %3 to float
  %5 = bitcast float %4 to i32
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %ENDIF, label %IF

IF:
  %7 = getelementptr i32 addrspace(1)* %out, i32 1
  store i32 0, i32 addrspace(1)* %7
  br label %ENDIF

ENDIF:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; Test a CND*_INT instruction with float true/false values
; CHECK: @test_c
; CHECK: CND{{[GTE]+}}_INT
define void @test_c(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = icmp sgt i32 %in, 0
  %1 = select i1 %0, float 2.0, float 3.0
  store float %1, float addrspace(1)* %out
  ret void
}
