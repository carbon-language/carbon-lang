; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}test_i64_eq:
; SI: V_CMP_EQ_I64
define void @test_i64_eq(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp eq i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_ne:
; SI: V_CMP_NE_I64
define void @test_i64_ne(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ne i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_slt:
; SI: V_CMP_LT_I64
define void @test_i64_slt(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp slt i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_ult:
; SI: V_CMP_LT_U64
define void @test_i64_ult(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ult i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_sle:
; SI: V_CMP_LE_I64
define void @test_i64_sle(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp sle i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_ule:
; SI: V_CMP_LE_U64
define void @test_i64_ule(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ule i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_sgt:
; SI: V_CMP_GT_I64
define void @test_i64_sgt(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp sgt i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_ugt:
; SI: V_CMP_GT_U64
define void @test_i64_ugt(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ugt i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_sge:
; SI: V_CMP_GE_I64
define void @test_i64_sge(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp sge i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_uge:
; SI: V_CMP_GE_U64
define void @test_i64_uge(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp uge i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

