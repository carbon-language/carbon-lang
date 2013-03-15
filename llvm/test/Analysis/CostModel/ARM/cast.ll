; RUN: opt < %s  -cost-model -analyze -mtriple=thumbv7-apple-ios6.0.0 -mcpu=swift | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios6.0.0"

define i32 @casts() {

    ; -- scalars --
  ; CHECK: cost of 1 {{.*}} sext
  %r0 = sext i1 undef to i8
  ; CHECK: cost of 1 {{.*}} zext
  %r1 = zext i1 undef to i8
  ; CHECK: cost of 1 {{.*}} sext
  %r2 = sext i1 undef to i16
  ; CHECK: cost of 1 {{.*}} zext
  %r3 = zext i1 undef to i16
  ; CHECK: cost of 1 {{.*}} sext
  %r4 = sext i1 undef to i32
  ; CHECK: cost of 1 {{.*}} zext
  %r5 = zext i1 undef to i32
  ; CHECK: cost of 1 {{.*}} sext
  %r6 = sext i1 undef to i64
  ; CHECK: cost of 1 {{.*}} zext
  %r7 = zext i1 undef to i64
  ; CHECK: cost of 0 {{.*}} trunc
  %r8 = trunc i8 undef to i1
  ; CHECK: cost of 1 {{.*}} sext
  %r9 = sext i8 undef to i16
  ; CHECK: cost of 1 {{.*}} zext
  %r10 = zext i8 undef to i16
  ; CHECK: cost of 1 {{.*}} sext
  %r11 = sext i8 undef to i32
  ; CHECK: cost of 1 {{.*}} zext
  %r12 = zext i8 undef to i32
  ; CHECK: cost of 1 {{.*}} sext
  %r13 = sext i8 undef to i64
  ; CHECK: cost of 1 {{.*}} zext
  %r14 = zext i8 undef to i64
  ; CHECK: cost of 0 {{.*}} trunc
  %r15 = trunc i16 undef to i1
  ; CHECK: cost of 0 {{.*}} trunc
  %r16 = trunc i16 undef to i8
  ; CHECK: cost of 1 {{.*}} sext
  %r17 = sext i16 undef to i32
  ; CHECK: cost of 1 {{.*}} zext
  %r18 = zext i16 undef to i32
  ; CHECK: cost of 2 {{.*}} sext
  %r19 = sext i16 undef to i64
  ; CHECK: cost of 1 {{.*}} zext
  %r20 = zext i16 undef to i64
  ; CHECK: cost of 0 {{.*}} trunc
  %r21 = trunc i32 undef to i1
  ; CHECK: cost of 0 {{.*}} trunc
  %r22 = trunc i32 undef to i8
  ; CHECK: cost of 0 {{.*}} trunc
  %r23 = trunc i32 undef to i16
  ; CHECK: cost of 1 {{.*}} sext
  %r24 = sext i32 undef to i64
  ; CHECK: cost of 1 {{.*}} zext
  %r25 = zext i32 undef to i64
  ; CHECK: cost of 0 {{.*}} trunc
  %r26 = trunc i64 undef to i1
  ; CHECK: cost of 0 {{.*}} trunc
  %r27 = trunc i64 undef to i8
  ; CHECK: cost of 0 {{.*}} trunc
  %r28 = trunc i64 undef to i16
  ; CHECK: cost of 0 {{.*}} trunc
  %r29 = trunc i64 undef to i32

    ; -- floating point conversions --
  ; Moves between scalar and NEON registers.
  ; CHECK: cost of 2 {{.*}} fptoui
  %r30 = fptoui float undef to i1
  ; CHECK: cost of 2 {{.*}} fptosi
  %r31 = fptosi float undef to i1
  ; CHECK: cost of 2 {{.*}} fptoui
  %r32 = fptoui float undef to i8
  ; CHECK: cost of 2 {{.*}} fptosi
  %r33 = fptosi float undef to i8
  ; CHECK: cost of 2 {{.*}} fptoui
  %r34 = fptoui float undef to i16
  ; CHECK: cost of 2 {{.*}} fptosi
  %r35 = fptosi float undef to i16
  ; CHECK: cost of 2 {{.*}} fptoui
  %r36 = fptoui float undef to i32
  ; CHECK: cost of 2 {{.*}} fptosi
  %r37 = fptosi float undef to i32
  ; CHECK: cost of 10 {{.*}} fptoui
  %r38 = fptoui float undef to i64
  ; CHECK: cost of 10 {{.*}} fptosi
  %r39 = fptosi float undef to i64
  ; CHECK: cost of 2 {{.*}} fptoui
  %r40 = fptoui double undef to i1
  ; CHECK: cost of 2 {{.*}} fptosi
  %r41 = fptosi double undef to i1
  ; CHECK: cost of 2 {{.*}} fptoui
  %r42 = fptoui double undef to i8
  ; CHECK: cost of 2 {{.*}} fptosi
  %r43 = fptosi double undef to i8
  ; CHECK: cost of 2 {{.*}} fptoui
  %r44 = fptoui double undef to i16
  ; CHECK: cost of 2 {{.*}} fptosi
  %r45 = fptosi double undef to i16
  ; CHECK: cost of 2 {{.*}} fptoui
  %r46 = fptoui double undef to i32
  ; CHECK: cost of 2 {{.*}} fptosi
  %r47 = fptosi double undef to i32
  ; Function call
  ; CHECK: cost of 10 {{.*}} fptoui
  %r48 = fptoui double undef to i64
  ; CHECK: cost of 10 {{.*}} fptosi
  %r49 = fptosi double undef to i64

  ; CHECK: cost of 2 {{.*}} sitofp
  %r50 = sitofp i1 undef to float
  ; CHECK: cost of 2 {{.*}} uitofp
  %r51 = uitofp i1 undef to float
  ; CHECK: cost of 2 {{.*}} sitofp
  %r52 = sitofp i1 undef to double
  ; CHECK: cost of 2 {{.*}} uitofp
  %r53 = uitofp i1 undef to double
  ; CHECK: cost of 2 {{.*}} sitofp
  %r54 = sitofp i8 undef to float
  ; CHECK: cost of 2 {{.*}} uitofp
  %r55 = uitofp i8 undef to float
  ; CHECK: cost of 2 {{.*}} sitofp
  %r56 = sitofp i8 undef to double
  ; CHECK: cost of 2 {{.*}} uitofp
  %r57 = uitofp i8 undef to double
  ; CHECK: cost of 2 {{.*}} sitofp
  %r58 = sitofp i16 undef to float
  ; CHECK: cost of 2 {{.*}} uitofp
  %r59 = uitofp i16 undef to float
  ; CHECK: cost of 2 {{.*}} sitofp
  %r60 = sitofp i16 undef to double
  ; CHECK: cost of 2 {{.*}} uitofp
  %r61 = uitofp i16 undef to double
  ; CHECK: cost of 2 {{.*}} sitofp
  %r62 = sitofp i32 undef to float
  ; CHECK: cost of 2 {{.*}} uitofp
  %r63 = uitofp i32 undef to float
  ; CHECK: cost of 2 {{.*}} sitofp
  %r64 = sitofp i32 undef to double
  ; CHECK: cost of 2 {{.*}} uitofp
  %r65 = uitofp i32 undef to double
  ; Function call
  ; CHECK: cost of 10 {{.*}} sitofp
  %r66 = sitofp i64 undef to float
  ; CHECK: cost of 10 {{.*}} uitofp
  %r67 = uitofp i64 undef to float
  ; CHECK: cost of 10 {{.*}} sitofp
  %r68 = sitofp i64 undef to double
  ; CHECK: cost of 10 {{.*}} uitofp
  %r69 = uitofp i64 undef to double

  ; Vector cast cost of instructions lowering the cast to the stack.
  ; CHECK: cost of 24 {{.*}} sext
  %r70 = sext <8 x i8> undef to <8 x i32>
  ; CHECK: cost of 48 {{.*}} sext
  %r71 = sext <16 x i8> undef to <16 x i32>
  ; CHECK: cost of 22 {{.*}} zext
  %r72 = zext <8 x i8> undef to <8 x i32>
  ; CHECK: cost of 44 {{.*}} zext
  %r73 = zext <16 x i8> undef to <16 x i32>
  ; CHECK: cost of 19 {{.*}} trunc
  %r74 = trunc <8 x i32> undef to <8 x i8>
  ; CHECK: cost of 38 {{.*}} trunc
  %r75 = trunc <16 x i32> undef to <16 x i8>

  ; Floating point truncation costs.
  ; CHECK: cost of 1 {{.*}} fptrunc double
  %r80 = fptrunc double undef to float
  ; CHECK: cost of 2 {{.*}} fptrunc <2 x double
  %r81 = fptrunc <2 x double> undef to <2 x float>
  ; CHECK: cost of 4 {{.*}} fptrunc <4 x double
  %r82 = fptrunc <4 x double> undef to <4 x float>
  ; CHECK: cost of 8 {{.*}} fptrunc <8 x double
  %r83 = fptrunc <8 x double> undef to <8 x float>
  ; CHECK: cost of 16 {{.*}} fptrunc <16 x double
  %r84 = fptrunc <16 x double> undef to <16 x float>

  ; Floating point extension costs.
  ; CHECK: cost of 1 {{.*}} fpext float
  %r85 = fpext float undef to double
  ; CHECK: cost of 2 {{.*}} fpext <2 x float
  %r86 = fpext <2 x float> undef to <2 x double>
  ; CHECK: cost of 4 {{.*}} fpext <4 x float
  %r87 = fpext <4 x float> undef to <4 x double>
  ; CHECK: cost of 8 {{.*}} fpext <8 x float
  %r88 = fpext <8 x float> undef to <8 x double>
  ; CHECK: cost of 16 {{.*}} fpext <16 x float
  %r89 = fpext <16 x float> undef to <16 x double>

  ;CHECK: cost of 0 {{.*}} ret
  ret i32 undef
}

