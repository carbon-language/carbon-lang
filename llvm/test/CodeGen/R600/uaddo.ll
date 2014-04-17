; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck -check-prefix=SI %s

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone

; SI-LABEL: @uaddo_i64_zext
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
