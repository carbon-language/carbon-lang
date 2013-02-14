; RUN: llc < %s -O3  -mtriple=arm-linux-gnueabi | FileCheck %s

; check if regs are passing correctly
define void @i64_write(i64* %p, i64 %val) nounwind {
; CHECK: i64_write:
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; CHECK: strexd [[REG1]], {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
  %1 = tail call i64 asm sideeffect "1: ldrexd $0, ${0:H}, [$2]\0A strexd $0, $3, ${3:H}, [$2]\0A teq $0, #0\0A bne 1b", "=&r,=*Qo,r,r,~{cc}"(i64* %p, i64* %p, i64 %val) nounwind
  ret void
}

; check if register allocation can reuse the registers
define void @multi_writes(i64* %p, i64 %val1, i64 %val2, i64 %val3, i64 %val4, i64 %val5, i64 %val6) nounwind {
entry:
; CHECK: multi_writes:
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]

; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]

; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; check: strexd {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}, [r{{[0-9]+}}]

  tail call void asm sideeffect " strexd $1, ${1:H}, [$0]\0A strexd $2, ${2:H}, [$0]\0A strexd $3, ${3:H}, [$0]\0A strexd $4, ${4:H}, [$0]\0A strexd $5, ${5:H}, [$0]\0A strexd $6, ${6:H}, [$0]\0A", "r,r,r,r,r,r,r"(i64* %p, i64 %val1, i64 %val2, i64 %val3, i64 %val4, i64 %val5, i64 %val6) nounwind
  %incdec.ptr = getelementptr inbounds i64* %p, i32 1
  tail call void asm sideeffect " strexd $1, ${1:H}, [$0]\0A strexd $2, ${2:H}, [$0]\0A strexd $3, ${3:H}, [$0]\0A strexd $4, ${4:H}, [$0]\0A strexd $5, ${5:H}, [$0]\0A strexd $6, ${6:H}, [$0]\0A", "r,r,r,r,r,r,r"(i64* %incdec.ptr, i64 %val1, i64 %val2, i64 %val3, i64 %val4, i64 %val5, i64 %val6) nounwind
  tail call void asm sideeffect " strexd $1, ${1:H}, [$0]\0A strexd $2, ${2:H}, [$0]\0A strexd $3, ${3:H}, [$0]\0A strexd $4, ${4:H}, [$0]\0A strexd $5, ${5:H}, [$0]\0A strexd $6, ${6:H}, [$0]\0A", "r,r,r,r,r,r,r"(i64* %incdec.ptr, i64 %val1, i64 %val2, i64 %val3, i64 %val4, i64 %val5, i64 %val6) nounwind
  ret void
}


; check if callee-saved registers used by inline asm are saved/restored
define void @foo(i64* %p, i64 %i) nounwind {
; CHECK:foo:
; CHECK: push {{{r[4-9]|r10|r11}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
; CHECK: strexd [[REG1]], {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: pop {{{r[4-9]|r10|r11}}
  %1 = tail call { i64, i64 } asm sideeffect "@ atomic64_set\0A1: ldrexd $0, ${0:H}, [$3]\0Aldrexd $1, ${1:H}, [$3]\0A strexd $0, $4, ${4:H}, [$3]\0A teq $0, #0\0A bne 1b", "=&r,=&r,=*Qo,r,r,~{cc}"(i64* %p, i64* %p, i64 %i) nounwind
  ret void
}
