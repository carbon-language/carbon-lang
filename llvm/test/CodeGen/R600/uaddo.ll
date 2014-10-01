; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs< %s

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone

; FUNC-LABEL: {{^}}uaddo_i64_zext:
; SI: ADD
; SI: ADDC
; SI: ADDC
define void @uaddo_i64_zext(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %uadd = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %uadd, 0
  %carry = extractvalue { i64, i1 } %uadd, 1
  %ext = zext i1 %carry to i64
  %add2 = add i64 %val, %ext
  store i64 %add2, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_uaddo_i32:
; SI: S_ADD_I32
define void @s_uaddo_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 %a, i32 %b) nounwind {
  %uadd = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b) nounwind
  %val = extractvalue { i32, i1 } %uadd, 0
  %carry = extractvalue { i32, i1 } %uadd, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_uaddo_i32:
; SI: V_ADD_I32
define void @v_uaddo_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %a = load i32 addrspace(1)* %aptr, align 4
  %b = load i32 addrspace(1)* %bptr, align 4
  %uadd = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b) nounwind
  %val = extractvalue { i32, i1 } %uadd, 0
  %carry = extractvalue { i32, i1 } %uadd, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}s_uaddo_i64:
; SI: S_ADD_U32
; SI: S_ADDC_U32
define void @s_uaddo_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 %a, i64 %b) nounwind {
  %uadd = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %uadd, 0
  %carry = extractvalue { i64, i1 } %uadd, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_uaddo_i64:
; SI: V_ADD_I32
; SI: V_ADDC_U32
define void @v_uaddo_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) nounwind {
  %a = load i64 addrspace(1)* %aptr, align 4
  %b = load i64 addrspace(1)* %bptr, align 4
  %uadd = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %uadd, 0
  %carry = extractvalue { i64, i1 } %uadd, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}
