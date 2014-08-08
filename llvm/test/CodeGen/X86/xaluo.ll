; RUN: llc -mtriple=x86_64-darwin-unknown < %s                             | FileCheck %s --check-prefix=DAG
; RUN: llc -mtriple=x86_64-darwin-unknown -fast-isel -fast-isel-abort < %s | FileCheck %s --check-prefix=FAST
; RUN: llc -mtriple=x86_64-darwin-unknown < %s                             | FileCheck %s
; RUN: llc -mtriple=x86_64-darwin-unknown -fast-isel -fast-isel-abort < %s | FileCheck %s

;
; Get the actual value of the overflow bit.
;
; SADDO reg, reg
define zeroext i1 @saddo.i8(i8 signext %v1, i8 signext %v2, i8* %res) {
entry:
; DAG-LABEL:    saddo.i8
; DAG:          addb %sil, %dil
; DAG-NEXT:     seto %al
; FAST-LABEL:   saddo.i8
; FAST:         addb %sil, %dil
; FAST-NEXT:    seto %al
  %t = call {i8, i1} @llvm.sadd.with.overflow.i8(i8 %v1, i8 %v2)
  %val = extractvalue {i8, i1} %t, 0
  %obit = extractvalue {i8, i1} %t, 1
  store i8 %val, i8* %res
  ret i1 %obit
}

define zeroext i1 @saddo.i16(i16 %v1, i16 %v2, i16* %res) {
entry:
; DAG-LABEL:    saddo.i16
; DAG:          addw %si, %di
; DAG-NEXT:     seto %al
; FAST-LABEL:   saddo.i16
; FAST:         addw %si, %di
; FAST-NEXT:    seto %al
  %t = call {i16, i1} @llvm.sadd.with.overflow.i16(i16 %v1, i16 %v2)
  %val = extractvalue {i16, i1} %t, 0
  %obit = extractvalue {i16, i1} %t, 1
  store i16 %val, i16* %res
  ret i1 %obit
}

define zeroext i1 @saddo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; DAG-LABEL:    saddo.i32
; DAG:          addl %esi, %edi
; DAG-NEXT:     seto %al
; FAST-LABEL:   saddo.i32
; FAST:         addl %esi, %edi
; FAST-NEXT:    seto %al
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define zeroext i1 @saddo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64
; DAG:          addq %rsi, %rdi
; DAG-NEXT:     seto %al
; FAST-LABEL:   saddo.i64
; FAST:         addq %rsi, %rdi
; FAST-NEXT:    seto %al
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; SADDO reg, 1 | INC
define zeroext i1 @saddo.inc.i8(i8 %v1, i8* %res) {
entry:
; CHECK-LABEL: saddo.inc.i8
; CHECK:       incb %dil
; CHECK-NEXT:  seto %al
  %t = call {i8, i1} @llvm.sadd.with.overflow.i8(i8 %v1, i8 1)
  %val = extractvalue {i8, i1} %t, 0
  %obit = extractvalue {i8, i1} %t, 1
  store i8 %val, i8* %res
  ret i1 %obit
}

define zeroext i1 @saddo.inc.i16(i16 %v1, i16* %res) {
entry:
; CHECK-LABEL: saddo.inc.i16
; CHECK:       incw %di
; CHECK-NEXT:  seto %al
  %t = call {i16, i1} @llvm.sadd.with.overflow.i16(i16 %v1, i16 1)
  %val = extractvalue {i16, i1} %t, 0
  %obit = extractvalue {i16, i1} %t, 1
  store i16 %val, i16* %res
  ret i1 %obit
}

define zeroext i1 @saddo.inc.i32(i32 %v1, i32* %res) {
entry:
; CHECK-LABEL: saddo.inc.i32
; CHECK:       incl %edi
; CHECK-NEXT:  seto %al
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %v1, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define zeroext i1 @saddo.inc.i64(i64 %v1, i64* %res) {
entry:
; CHECK-LABEL: saddo.inc.i64
; CHECK:       incq %rdi
; CHECK-NEXT:  seto %al
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; SADDO reg, imm | imm, reg
; FIXME: DAG doesn't optimize immediates on the LHS.
define zeroext i1 @saddo.i64imm1(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm1
; DAG:          mov
; DAG-NEXT:     addq
; DAG-NEXT:     seto
; FAST-LABEL:   saddo.i64imm1
; FAST:         addq $2, %rdi
; FAST-NEXT:    seto %al
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 2, i64 %v1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; Check boundary conditions for large immediates.
define zeroext i1 @saddo.i64imm2(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm2
; DAG:          addq $-2147483648, %rdi
; DAG-NEXT:     seto %al
; FAST-LABEL:   saddo.i64imm2
; FAST:         addq $-2147483648, %rdi
; FAST-NEXT:    seto %al
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 -2147483648)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

define zeroext i1 @saddo.i64imm3(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm3
; DAG:          movabsq $-21474836489, %[[REG:[a-z]+]]
; DAG-NEXT:     addq %rdi, %[[REG]]
; DAG-NEXT:     seto
; FAST-LABEL:   saddo.i64imm3
; FAST:         movabsq $-21474836489, %[[REG:[a-z]+]]
; FAST-NEXT:    addq %rdi, %[[REG]]
; FAST-NEXT:    seto
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 -21474836489)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

define zeroext i1 @saddo.i64imm4(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm4
; DAG:          addq $2147483647, %rdi
; DAG-NEXT:     seto
; FAST-LABEL:   saddo.i64imm4
; FAST:         addq $2147483647, %rdi
; FAST-NEXT:    seto
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 2147483647)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; TODO: FastISel shouldn't use movabsq.
define zeroext i1 @saddo.i64imm5(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm5
; DAG:          movl $2147483648, %ecx
; DAG:          addq %rdi, %rcx
; DAG-NEXT:     seto
; FAST-LABEL:   saddo.i64imm5
; FAST:         movabsq $2147483648, %[[REG:[a-z]+]]
; FAST:         addq %rdi, %[[REG]]
; FAST-NEXT:     seto
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 2147483648)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; UADDO
define zeroext i1 @uaddo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; DAG-LABEL:    uaddo.i32
; DAG:          addl %esi, %edi
; DAG-NEXT:     setb %al
; FAST-LABEL:   uaddo.i32
; FAST:         addl %esi, %edi
; FAST-NEXT:    setb %al
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define zeroext i1 @uaddo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; DAG-LABEL:    uaddo.i64
; DAG:          addq %rsi, %rdi
; DAG-NEXT:     setb %al
; FAST-LABEL:   uaddo.i64
; FAST:         addq %rsi, %rdi
; FAST-NEXT:    setb %al
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; UADDO reg, 1 | NOT INC
define zeroext i1 @uaddo.inc.i8(i8 %v1, i8* %res) {
entry:
; CHECK-LABEL: uaddo.inc.i8
; CHECK-NOT:   incb %dil
  %t = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 %v1, i8 1)
  %val = extractvalue {i8, i1} %t, 0
  %obit = extractvalue {i8, i1} %t, 1
  store i8 %val, i8* %res
  ret i1 %obit
}

define zeroext i1 @uaddo.inc.i16(i16 %v1, i16* %res) {
entry:
; CHECK-LABEL: uaddo.inc.i16
; CHECK-NOT:   incw %di
  %t = call {i16, i1} @llvm.uadd.with.overflow.i16(i16 %v1, i16 1)
  %val = extractvalue {i16, i1} %t, 0
  %obit = extractvalue {i16, i1} %t, 1
  store i16 %val, i16* %res
  ret i1 %obit
}

define zeroext i1 @uaddo.inc.i32(i32 %v1, i32* %res) {
entry:
; CHECK-LABEL: uaddo.inc.i32
; CHECK-NOT:   incl %edi
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define zeroext i1 @uaddo.inc.i64(i64 %v1, i64* %res) {
entry:
; CHECK-LABEL: uaddo.inc.i64
; CHECK-NOT:   incq %rdi
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %v1, i64 1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; SSUBO
define zeroext i1 @ssubo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; DAG-LABEL:    ssubo.i32
; DAG:          subl %esi, %edi
; DAG-NEXT:     seto %al
; FAST-LABEL:   ssubo.i32
; FAST:         subl %esi, %edi
; FAST-NEXT:    seto %al
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define zeroext i1 @ssubo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; DAG-LABEL:    ssubo.i64
; DAG:          subq %rsi, %rdi
; DAG-NEXT:     seto %al
; FAST-LABEL:   ssubo.i64
; FAST:         subq %rsi, %rdi
; FAST-NEXT:    seto %al
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; USUBO
define zeroext i1 @usubo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; DAG-LABEL:    usubo.i32
; DAG:          subl %esi, %edi
; DAG-NEXT:     setb %al
; FAST-LABEL:   usubo.i32
; FAST:         subl %esi, %edi
; FAST-NEXT:    setb %al
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define zeroext i1 @usubo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; DAG-LABEL:    usubo.i64
; DAG:          subq %rsi, %rdi
; DAG-NEXT:     setb %al
; FAST-LABEL:   usubo.i64
; FAST:         subq %rsi, %rdi
; FAST-NEXT:    setb %al
  %t = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; SMULO
define zeroext i1 @smulo.i8(i8 %v1, i8 %v2, i8* %res) {
entry:
; FAST-LABEL:   smulo.i8
; FAST:         movb %dil, %al
; FAST-NEXT:    imulb %sil
; FAST-NEXT:    seto %cl
  %t = call {i8, i1} @llvm.smul.with.overflow.i8(i8 %v1, i8 %v2)
  %val = extractvalue {i8, i1} %t, 0
  %obit = extractvalue {i8, i1} %t, 1
  store i8 %val, i8* %res
  ret i1 %obit
}

define zeroext i1 @smulo.i16(i16 %v1, i16 %v2, i16* %res) {
entry:
; DAG-LABEL:    smulo.i16
; DAG:          imulw %si, %di
; DAG-NEXT:     seto %al
; FAST-LABEL:   smulo.i16
; FAST:         imulw %si, %di
; FAST-NEXT:    seto %al
  %t = call {i16, i1} @llvm.smul.with.overflow.i16(i16 %v1, i16 %v2)
  %val = extractvalue {i16, i1} %t, 0
  %obit = extractvalue {i16, i1} %t, 1
  store i16 %val, i16* %res
  ret i1 %obit
}

define zeroext i1 @smulo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; DAG-LABEL:    smulo.i32
; DAG:          imull %esi, %edi
; DAG-NEXT:     seto %al
; FAST-LABEL:   smulo.i32
; FAST:         imull %esi, %edi
; FAST-NEXT:    seto %al
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define zeroext i1 @smulo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; DAG-LABEL:    smulo.i64
; DAG:          imulq %rsi, %rdi
; DAG-NEXT:     seto %al
; FAST-LABEL:   smulo.i64
; FAST:         imulq %rsi, %rdi
; FAST-NEXT:    seto %al
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; UMULO
define zeroext i1 @umulo.i8(i8 %v1, i8 %v2, i8* %res) {
entry:
; FAST-LABEL:   umulo.i8
; FAST:         movb %dil, %al
; FAST-NEXT:    mulb %sil
; FAST-NEXT:    seto %cl
  %t = call {i8, i1} @llvm.umul.with.overflow.i8(i8 %v1, i8 %v2)
  %val = extractvalue {i8, i1} %t, 0
  %obit = extractvalue {i8, i1} %t, 1
  store i8 %val, i8* %res
  ret i1 %obit
}

define zeroext i1 @umulo.i16(i16 %v1, i16 %v2, i16* %res) {
entry:
; DAG-LABEL:    umulo.i16
; DAG:          mulw %si
; DAG-NEXT:     seto
; FAST-LABEL:   umulo.i16
; FAST:         mulw %si
; FAST-NEXT:    seto
  %t = call {i16, i1} @llvm.umul.with.overflow.i16(i16 %v1, i16 %v2)
  %val = extractvalue {i16, i1} %t, 0
  %obit = extractvalue {i16, i1} %t, 1
  store i16 %val, i16* %res
  ret i1 %obit
}

define zeroext i1 @umulo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; DAG-LABEL:    umulo.i32
; DAG:          mull %esi
; DAG-NEXT:     seto
; FAST-LABEL:   umulo.i32
; FAST:         mull %esi
; FAST-NEXT:    seto
  %t = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define zeroext i1 @umulo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; DAG-LABEL:    umulo.i64
; DAG:          mulq %rsi
; DAG-NEXT:     seto
; FAST-LABEL:   umulo.i64
; FAST:         mulq %rsi
; FAST-NEXT:    seto
  %t = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

;
; Check the use of the overflow bit in combination with a select instruction.
;
define i32 @saddo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    saddo.select.i32
; CHECK:          addl   %esi, %eax
; CHECK-NEXT:     cmovol %edi, %esi
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @saddo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    saddo.select.i64
; CHECK:          addq   %rsi, %rax
; CHECK-NEXT:     cmovoq %rdi, %rsi
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}

define i32 @uaddo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    uaddo.select.i32
; CHECK:          addl   %esi, %eax
; CHECK-NEXT:     cmovbl %edi, %esi
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @uaddo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    uaddo.select.i64
; CHECK:          addq   %rsi, %rax
; CHECK-NEXT:     cmovbq %rdi, %rsi
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}

define i32 @ssubo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    ssubo.select.i32
; CHECK:          cmpl   %esi, %edi
; CHECK-NEXT:     cmovol %edi, %esi
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @ssubo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    ssubo.select.i64
; CHECK:          cmpq   %rsi, %rdi
; CHECK-NEXT:     cmovoq %rdi, %rsi
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}

define i32 @usubo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    usubo.select.i32
; CHECK:          cmpl   %esi, %edi
; CHECK-NEXT:     cmovbl %edi, %esi
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @usubo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    usubo.select.i64
; CHECK:          cmpq   %rsi, %rdi
; CHECK-NEXT:     cmovbq %rdi, %rsi
  %t = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}

define i32 @smulo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    smulo.select.i32
; CHECK:          imull  %esi, %eax
; CHECK-NEXT:     cmovol %edi, %esi
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @smulo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    smulo.select.i64
; CHECK:          imulq  %rsi, %rax
; CHECK-NEXT:     cmovoq %rdi, %rsi
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}

define i32 @umulo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    umulo.select.i32
; CHECK:          mull   %esi
; CHECK-NEXT:     cmovol %edi, %esi
  %t = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @umulo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    umulo.select.i64
; CHECK:          mulq   %rsi
; CHECK-NEXT:     cmovoq %rdi, %rsi
  %t = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}


;
; Check the use of the overflow bit in combination with a branch instruction.
;
define zeroext i1 @saddo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    saddo.br.i32
; CHECK:          addl   %esi, %edi
; CHECK-NEXT:     jo
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @saddo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    saddo.br.i64
; CHECK:          addq   %rsi, %rdi
; CHECK-NEXT:     jo
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @uaddo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    uaddo.br.i32
; CHECK:          addl   %esi, %edi
; CHECK-NEXT:     jb
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @uaddo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    uaddo.br.i64
; CHECK:          addq   %rsi, %rdi
; CHECK-NEXT:     jb
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @ssubo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    ssubo.br.i32
; CHECK:          cmpl   %esi, %edi
; CHECK-NEXT:     jo
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @ssubo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    ssubo.br.i64
; CHECK:          cmpq   %rsi, %rdi
; CHECK-NEXT:     jo
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @usubo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    usubo.br.i32
; CHECK:          cmpl   %esi, %edi
; CHECK-NEXT:     jb
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @usubo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    usubo.br.i64
; CHECK:          cmpq   %rsi, %rdi
; CHECK-NEXT:     jb
  %t = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @smulo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    smulo.br.i32
; CHECK:          imull  %esi, %edi
; CHECK-NEXT:     jo
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @smulo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    smulo.br.i64
; CHECK:          imulq  %rsi, %rdi
; CHECK-NEXT:     jo
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @umulo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:    umulo.br.i32
; CHECK:          mull  %esi
; CHECK-NEXT:     jo
  %t = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

define zeroext i1 @umulo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:    umulo.br.i64
; CHECK:          mulq  %rsi
; CHECK-NEXT:     jo
  %t = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue, !prof !0

overflow:
  ret i1 false

continue:
  ret i1 true
}

declare {i8,  i1} @llvm.sadd.with.overflow.i8 (i8,  i8 ) nounwind readnone
declare {i16, i1} @llvm.sadd.with.overflow.i16(i16, i16) nounwind readnone
declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.sadd.with.overflow.i64(i64, i64) nounwind readnone
declare {i8,  i1} @llvm.uadd.with.overflow.i8 (i8,  i8 ) nounwind readnone
declare {i16, i1} @llvm.uadd.with.overflow.i16(i16, i16) nounwind readnone
declare {i32, i1} @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.ssub.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.ssub.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.usub.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.usub.with.overflow.i64(i64, i64) nounwind readnone
declare {i8,  i1} @llvm.smul.with.overflow.i8 (i8,  i8 ) nounwind readnone
declare {i16, i1} @llvm.smul.with.overflow.i16(i16, i16) nounwind readnone
declare {i32, i1} @llvm.smul.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.smul.with.overflow.i64(i64, i64) nounwind readnone
declare {i8,  i1} @llvm.umul.with.overflow.i8 (i8,  i8 ) nounwind readnone
declare {i16, i1} @llvm.umul.with.overflow.i16(i16, i16) nounwind readnone
declare {i32, i1} @llvm.umul.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.umul.with.overflow.i64(i64, i64) nounwind readnone

!0 = metadata !{metadata !"branch_weights", i32 0, i32 2147483647}
