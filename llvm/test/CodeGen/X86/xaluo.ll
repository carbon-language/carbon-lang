; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s -check-prefix=DAG
; RUN: llc -mtriple=x86_64-unknown-unknown -fast-isel -fast-isel-abort < %s | FileCheck %s --check-prefix=FAST

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

; SADDO reg, imm | imm, reg
; FIXME: INC isn't supported in FastISel yet
define zeroext i1 @saddo.i64imm1(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm1
; DAG:          incq %rdi
; DAG-NEXT:     seto %al
; FAST-LABEL:   saddo.i64imm1
; FAST:         addq $1, %rdi
; FAST-NEXT:    seto %al
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; FIXME: DAG doesn't optimize immediates on the LHS.
define zeroext i1 @saddo.i64imm2(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm2
; DAG:          mov
; DAG-NEXT:     addq
; DAG-NEXT:     seto
; FAST-LABEL:   saddo.i64imm2
; FAST:         addq $1, %rdi
; FAST-NEXT:    seto %al
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 1, i64 %v1)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; Check boundary conditions for large immediates.
define zeroext i1 @saddo.i64imm3(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm3
; DAG:          addq $-2147483648, %rdi
; DAG-NEXT:     seto %al
; FAST-LABEL:   saddo.i64imm3
; FAST:         addq $-2147483648, %rdi
; FAST-NEXT:    seto %al
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 -2147483648)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

define zeroext i1 @saddo.i64imm4(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm4
; DAG:          movabsq $-21474836489, %[[REG:[a-z]+]]
; DAG-NEXT:     addq %rdi, %[[REG]]
; DAG-NEXT:     seto
; FAST-LABEL:   saddo.i64imm4
; FAST:         movabsq $-21474836489, %[[REG:[a-z]+]]
; FAST-NEXT:    addq %rdi, %[[REG]]
; FAST-NEXT:    seto
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 -21474836489)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

define zeroext i1 @saddo.i64imm5(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm5
; DAG:          addq $2147483647, %rdi
; DAG-NEXT:     seto
; FAST-LABEL:   saddo.i64imm5
; FAST:         addq $2147483647, %rdi
; FAST-NEXT:    seto
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 2147483647)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

; TODO: FastISel shouldn't use movabsq.
define zeroext i1 @saddo.i64imm6(i64 %v1, i64* %res) {
entry:
; DAG-LABEL:    saddo.i64imm6
; DAG:          movl $2147483648, %ecx
; DAG:          addq %rdi, %rcx
; DAG-NEXT:     seto
; FAST-LABEL:   saddo.i64imm6
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

declare {i8, i1} @llvm.sadd.with.overflow.i8(i8, i8) nounwind readnone
declare {i16, i1} @llvm.sadd.with.overflow.i16(i16, i16) nounwind readnone
declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.sadd.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.ssub.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.ssub.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.usub.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.usub.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.smul.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.smul.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.umul.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.umul.with.overflow.i64(i64, i64) nounwind readnone

