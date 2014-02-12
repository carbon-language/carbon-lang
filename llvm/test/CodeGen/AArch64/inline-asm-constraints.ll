;RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

define i64 @test_inline_constraint_r(i64 %base, i32 %offset) {
; CHECK-LABEL: test_inline_constraint_r:
  %val = call i64 asm "add $0, $1, $2, sxtw", "=r,r,r"(i64 %base, i32 %offset)
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{w[0-9]+}}, sxtw
  ret i64 %val
}

define i16 @test_small_reg(i16 %lhs, i16 %rhs) {
; CHECK-LABEL: test_small_reg:
  %val = call i16 asm sideeffect "add $0, $1, $2, sxth", "=r,r,r"(i16 %lhs, i16 %rhs)
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, sxth
  ret i16 %val
}

define i64 @test_inline_constraint_r_imm(i64 %base, i32 %offset) {
; CHECK-LABEL: test_inline_constraint_r_imm:
  %val = call i64 asm "add $0, $1, $2, sxtw", "=r,r,r"(i64 4, i32 12)
; CHECK: movz [[FOUR:x[0-9]+]], #4
; CHECK: movz [[TWELVE:w[0-9]+]], #12
; CHECK: add {{x[0-9]+}}, [[FOUR]], [[TWELVE]], sxtw
  ret i64 %val
}

; m is permitted to have a base/offset form. We don't do that
; currently though.
define i32 @test_inline_constraint_m(i32 *%ptr) {
; CHECK-LABEL: test_inline_constraint_m:
  %val = call i32 asm "ldr $0, $1", "=r,m"(i32 *%ptr)
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}]
  ret i32 %val
}

@arr = global [8 x i32] zeroinitializer

; Q should *never* have base/offset form even if given the chance.
define i32 @test_inline_constraint_Q(i32 *%ptr) {
; CHECK-LABEL: test_inline_constraint_Q:
  %val = call i32 asm "ldr $0, $1", "=r,Q"(i32* getelementptr([8 x i32]* @arr, i32 0, i32 1))
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}]
  ret i32 %val
}

@dump = global fp128 zeroinitializer

define void @test_inline_constraint_w(<8 x i8> %vec64, <4 x float> %vec128, half %hlf, float %flt, double %dbl, fp128 %quad) {
; CHECK: test_inline_constraint_w:
  call <8 x i8> asm sideeffect "add $0.8b, $1.8b, $1.8b", "=w,w"(<8 x i8> %vec64)
  call <8 x i8> asm sideeffect "fadd $0.4s, $1.4s, $1.4s", "=w,w"(<4 x float> %vec128)
; CHECK: add {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK: fadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s

  ; Arguably semantically dodgy to output "vN", but it's what GCC does
  ; so purely for compatibility we want vector registers to be output.
  call float asm sideeffect "fcvt ${0:s}, ${1:h}", "=w,w"(half undef)
  call float asm sideeffect "fadd $0.2s, $0.2s, $0.2s", "=w,w"(float %flt)
  call double asm sideeffect "fadd $0.2d, $0.2d, $0.2d", "=w,w"(double %dbl)
  call fp128 asm sideeffect "fadd $0.2d, $0.2d, $0.2d", "=w,w"(fp128 %quad)
; CHECK: fcvt {{s[0-9]+}}, {{h[0-9]+}}
; CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
; CHECK: fadd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
; CHECK: fadd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
  ret void
}

define void @test_inline_constraint_I() {
; CHECK-LABEL: test_inline_constraint_I:
  call void asm sideeffect "add x0, x0, $0", "I"(i32 0)
  call void asm sideeffect "add x0, x0, $0", "I"(i64 4095)
; CHECK: add x0, x0, #0
; CHECK: add x0, x0, #4095

  ret void
}

; Skip J because it's useless

define void @test_inline_constraint_K() {
; CHECK-LABEL: test_inline_constraint_K:
  call void asm sideeffect "and w0, w0, $0", "K"(i32 2863311530) ; = 0xaaaaaaaa
  call void asm sideeffect "and w0, w0, $0", "K"(i32 65535)
; CHECK: and w0, w0, #-1431655766
; CHECK: and w0, w0, #65535

  ret void
}

define void @test_inline_constraint_L() {
; CHECK-LABEL: test_inline_constraint_L:
  call void asm sideeffect "and x0, x0, $0", "L"(i64 4294967296) ; = 0xaaaaaaaa
  call void asm sideeffect "and x0, x0, $0", "L"(i64 65535)
; CHECK: and x0, x0, #4294967296
; CHECK: and x0, x0, #65535

  ret void
}

; Skip M and N because we don't support MOV pseudo-instructions yet.

@var = global i32 0

define void @test_inline_constraint_S() {
; CHECK-LABEL: test_inline_constraint_S:
  call void asm sideeffect "adrp x0, $0", "S"(i32* @var)
  call void asm sideeffect "adrp x0, ${0:A}", "S"(i32* @var)
  call void asm sideeffect "add x0, x0, ${0:L}", "S"(i32* @var)
; CHECK: adrp x0, var
; CHECK: adrp x0, var
; CHECK: add x0, x0, #:lo12:var
  ret void
}

define i32 @test_inline_constraint_S_label(i1 %in) {
; CHECK-LABEL: test_inline_constraint_S_label:
  call void asm sideeffect "adr x0, $0", "S"(i8* blockaddress(@test_inline_constraint_S_label, %loc))
; CHECK: adr x0, .Ltmp{{[0-9]+}}
  br i1 %in, label %loc, label %loc2
loc:
  ret i32 0
loc2:
  ret i32 42
}

define void @test_inline_constraint_Y() {
; CHECK-LABEL: test_inline_constraint_Y:
  call void asm sideeffect "fcmp s0, $0", "Y"(float 0.0)
; CHECK: fcmp s0, #0.0
  ret void
}

define void @test_inline_constraint_Z() {
; CHECK-LABEL: test_inline_constraint_Z:
  call void asm sideeffect "cmp w0, $0", "Z"(i32 0)
; CHECK: cmp w0, #0
  ret void
}
