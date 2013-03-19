; RUN: opt < %s  -cost-model -analyze -mtriple=thumbv7-apple-ios6.0.0 -mcpu=cortex-a8 | FileCheck %s
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

  ; CHECK: cost of 3 {{.*}} sext
  %r70 = sext <8 x i8> undef to <8 x i32>
  ; CHECK: cost of 6 {{.*}} sext
  %r71 = sext <16 x i8> undef to <16 x i32>
  ; CHECK: cost of 3 {{.*}} zext
  %r72 = zext <8 x i8> undef to <8 x i32>
  ; CHECK: cost of 6 {{.*}} zext
  %r73 = zext <16 x i8> undef to <16 x i32>

  ; CHECK: cost of 7 {{.*}} sext
  %rext_0 = sext <8 x i8> undef to <8 x i64>
  ; CHECK: cost of 7 {{.*}} zext
  %rext_1 = zext <8 x i8> undef to <8 x i64>
  ; CHECK: cost of 6 {{.*}} sext
  %rext_2 = sext <8 x i16> undef to <8 x i64>
  ; CHECK: cost of 6 {{.*}} zext
  %rext_3 = zext <8 x i16> undef to <8 x i64>
  ; CHECK: cost of 3 {{.*}} sext
  %rext_4 = sext <4 x i16> undef to <4 x i64>
  ; CHECK: cost of 3 {{.*}} zext
  %rext_5 = zext <4 x i16> undef to <4 x i64>

  ; Vector cast cost of instructions lowering the cast to the stack.
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

  ;; Floating point to integer vector casts.
  ; CHECK: cost of 1 {{.*}} fptoui
  %r90 = fptoui <2 x float> undef to <2 x i1>
  ; CHECK: cost of 1 {{.*}} fptosi
  %r91 = fptosi <2 x float> undef to <2 x i1>
  ; CHECK: cost of 1 {{.*}} fptoui
  %r92 = fptoui <2 x float> undef to <2 x i8>
  ; CHECK: cost of 1 {{.*}} fptosi
  %r93 = fptosi <2 x float> undef to <2 x i8>
  ; CHECK: cost of 1 {{.*}} fptoui
  %r94 = fptoui <2 x float> undef to <2 x i16>
  ; CHECK: cost of 1 {{.*}} fptosi
  %r95 = fptosi <2 x float> undef to <2 x i16>
  ; CHECK: cost of 1 {{.*}} fptoui
  %r96 = fptoui <2 x float> undef to <2 x i32>
  ; CHECK: cost of 1 {{.*}} fptosi
  %r97 = fptosi <2 x float> undef to <2 x i32>
  ; CHECK: cost of 24 {{.*}} fptoui
  %r98 = fptoui <2 x float> undef to <2 x i64>
  ; CHECK: cost of 24 {{.*}} fptosi
  %r99 = fptosi <2 x float> undef to <2 x i64>

  ; CHECK: cost of 8 {{.*}} fptoui
  %r100 = fptoui <2 x double> undef to <2 x i1>
  ; CHECK: cost of 8 {{.*}} fptosi
  %r101 = fptosi <2 x double> undef to <2 x i1>
  ; CHECK: cost of 8 {{.*}} fptoui
  %r102 = fptoui <2 x double> undef to <2 x i8>
  ; CHECK: cost of 8 {{.*}} fptosi
  %r103 = fptosi <2 x double> undef to <2 x i8>
  ; CHECK: cost of 8 {{.*}} fptoui
  %r104 = fptoui <2 x double> undef to <2 x i16>
  ; CHECK: cost of 8 {{.*}} fptosi
  %r105 = fptosi <2 x double> undef to <2 x i16>
  ; CHECK: cost of 2 {{.*}} fptoui
  %r106 = fptoui <2 x double> undef to <2 x i32>
  ; CHECK: cost of 2 {{.*}} fptosi
  %r107 = fptosi <2 x double> undef to <2 x i32>
  ; CHECK: cost of 24 {{.*}} fptoui
  %r108 = fptoui <2 x double> undef to <2 x i64>
  ; CHECK: cost of 24 {{.*}} fptosi
  %r109 = fptosi <2 x double> undef to <2 x i64>

  ; CHECK: cost of 16 {{.*}} fptoui
  %r110 = fptoui <4 x float> undef to <4 x i1>
  ; CHECK: cost of 16 {{.*}} fptosi
  %r111 = fptosi <4 x float> undef to <4 x i1>
  ; CHECK: cost of 3 {{.*}} fptoui
  %r112 = fptoui <4 x float> undef to <4 x i8>
  ; CHECK: cost of 3 {{.*}} fptosi
  %r113 = fptosi <4 x float> undef to <4 x i8>
  ; CHECK: cost of 2 {{.*}} fptoui
  %r114 = fptoui <4 x float> undef to <4 x i16>
  ; CHECK: cost of 2 {{.*}} fptosi
  %r115 = fptosi <4 x float> undef to <4 x i16>
  ; CHECK: cost of 1 {{.*}} fptoui
  %r116 = fptoui <4 x float> undef to <4 x i32>
  ; CHECK: cost of 1 {{.*}} fptosi
  %r117 = fptosi <4 x float> undef to <4 x i32>
  ; CHECK: cost of 48 {{.*}} fptoui
  %r118 = fptoui <4 x float> undef to <4 x i64>
  ; CHECK: cost of 48 {{.*}} fptosi
  %r119 = fptosi <4 x float> undef to <4 x i64>

  ; CHECK: cost of 16 {{.*}} fptoui
  %r120 = fptoui <4 x double> undef to <4 x i1>
  ; CHECK: cost of 16 {{.*}} fptosi
  %r121 = fptosi <4 x double> undef to <4 x i1>
  ; CHECK: cost of 16 {{.*}} fptoui
  %r122 = fptoui <4 x double> undef to <4 x i8>
  ; CHECK: cost of 16 {{.*}} fptosi
  %r123 = fptosi <4 x double> undef to <4 x i8>
  ; CHECK: cost of 16 {{.*}} fptoui
  %r124 = fptoui <4 x double> undef to <4 x i16>
  ; CHECK: cost of 16 {{.*}} fptosi
  %r125 = fptosi <4 x double> undef to <4 x i16>
  ; CHECK: cost of 16 {{.*}} fptoui
  %r126 = fptoui <4 x double> undef to <4 x i32>
  ; CHECK: cost of 16 {{.*}} fptosi
  %r127 = fptosi <4 x double> undef to <4 x i32>
  ; CHECK: cost of 48 {{.*}} fptoui
  %r128 = fptoui <4 x double> undef to <4 x i64>
  ; CHECK: cost of 48 {{.*}} fptosi
  %r129 = fptosi <4 x double> undef to <4 x i64>

  ; CHECK: cost of 32 {{.*}} fptoui
  %r130 = fptoui <8 x float> undef to <8 x i1>
  ; CHECK: cost of 32 {{.*}} fptosi
  %r131 = fptosi <8 x float> undef to <8 x i1>
  ; CHECK: cost of 32 {{.*}} fptoui
  %r132 = fptoui <8 x float> undef to <8 x i8>
  ; CHECK: cost of 32 {{.*}} fptosi
  %r133 = fptosi <8 x float> undef to <8 x i8>
  ; CHECK: cost of 4 {{.*}} fptoui
  %r134 = fptoui <8 x float> undef to <8 x i16>
  ; CHECK: cost of 4 {{.*}} fptosi
  %r135 = fptosi <8 x float> undef to <8 x i16>
  ; CHECK: cost of 2 {{.*}} fptoui
  %r136 = fptoui <8 x float> undef to <8 x i32>
  ; CHECK: cost of 2 {{.*}} fptosi
  %r137 = fptosi <8 x float> undef to <8 x i32>
  ; CHECK: cost of 96 {{.*}} fptoui
  %r138 = fptoui <8 x float> undef to <8 x i64>
  ; CHECK: cost of 96 {{.*}} fptosi
  %r139 = fptosi <8 x float> undef to <8 x i64>

  ; CHECK: cost of 32 {{.*}} fptoui
  %r140 = fptoui <8 x double> undef to <8 x i1>
  ; CHECK: cost of 32 {{.*}} fptosi
  %r141 = fptosi <8 x double> undef to <8 x i1>
  ; CHECK: cost of 32 {{.*}} fptoui
  %r142 = fptoui <8 x double> undef to <8 x i8>
  ; CHECK: cost of 32 {{.*}} fptosi
  %r143 = fptosi <8 x double> undef to <8 x i8>
  ; CHECK: cost of 32 {{.*}} fptoui
  %r144 = fptoui <8 x double> undef to <8 x i16>
  ; CHECK: cost of 32 {{.*}} fptosi
  %r145 = fptosi <8 x double> undef to <8 x i16>
  ; CHECK: cost of 32 {{.*}} fptoui
  %r146 = fptoui <8 x double> undef to <8 x i32>
  ; CHECK: cost of 32 {{.*}} fptosi
  %r147 = fptosi <8 x double> undef to <8 x i32>
  ; CHECK: cost of 96 {{.*}} fptoui
  %r148 = fptoui <8 x double> undef to <8 x i64>
  ; CHECK: cost of 96 {{.*}} fptosi
  %r149 = fptosi <8 x double> undef to <8 x i64>

  ; CHECK: cost of 64 {{.*}} fptoui
  %r150 = fptoui <16 x float> undef to <16 x i1>
  ; CHECK: cost of 64 {{.*}} fptosi
  %r151 = fptosi <16 x float> undef to <16 x i1>
  ; CHECK: cost of 64 {{.*}} fptoui
  %r152 = fptoui <16 x float> undef to <16 x i8>
  ; CHECK: cost of 64 {{.*}} fptosi
  %r153 = fptosi <16 x float> undef to <16 x i8>
  ; CHECK: cost of 8 {{.*}} fptoui
  %r154 = fptoui <16 x float> undef to <16 x i16>
  ; CHECK: cost of 8 {{.*}} fptosi
  %r155 = fptosi <16 x float> undef to <16 x i16>
  ; CHECK: cost of 4 {{.*}} fptoui
  %r156 = fptoui <16 x float> undef to <16 x i32>
  ; CHECK: cost of 4 {{.*}} fptosi
  %r157 = fptosi <16 x float> undef to <16 x i32>
  ; CHECK: cost of 192 {{.*}} fptoui
  %r158 = fptoui <16 x float> undef to <16 x i64>
  ; CHECK: cost of 192 {{.*}} fptosi
  %r159 = fptosi <16 x float> undef to <16 x i64>

  ; CHECK: cost of 64 {{.*}} fptoui
  %r160 = fptoui <16 x double> undef to <16 x i1>
  ; CHECK: cost of 64 {{.*}} fptosi
  %r161 = fptosi <16 x double> undef to <16 x i1>
  ; CHECK: cost of 64 {{.*}} fptoui
  %r162 = fptoui <16 x double> undef to <16 x i8>
  ; CHECK: cost of 64 {{.*}} fptosi
  %r163 = fptosi <16 x double> undef to <16 x i8>
  ; CHECK: cost of 64 {{.*}} fptoui
  %r164 = fptoui <16 x double> undef to <16 x i16>
  ; CHECK: cost of 64 {{.*}} fptosi
  %r165 = fptosi <16 x double> undef to <16 x i16>
  ; CHECK: cost of 64 {{.*}} fptoui
  %r166 = fptoui <16 x double> undef to <16 x i32>
  ; CHECK: cost of 64 {{.*}} fptosi
  %r167 = fptosi <16 x double> undef to <16 x i32>
  ; CHECK: cost of 192 {{.*}} fptoui
  %r168 = fptoui <16 x double> undef to <16 x i64>
  ; CHECK: cost of 192 {{.*}} fptosi
  %r169 = fptosi <16 x double> undef to <16 x i64>

  ; CHECK: cost of 8 {{.*}} uitofp
  %r170 = uitofp <2 x i1> undef to <2 x float>
  ; CHECK: cost of 8 {{.*}} sitofp
  %r171 = sitofp <2 x i1> undef to <2 x float>
  ; CHECK: cost of 3 {{.*}} uitofp
  %r172 = uitofp <2 x i8> undef to <2 x float>
  ; CHECK: cost of 3 {{.*}} sitofp
  %r173 = sitofp <2 x i8> undef to <2 x float>
  ; CHECK: cost of 2 {{.*}} uitofp
  %r174 = uitofp <2 x i16> undef to <2 x float>
  ; CHECK: cost of 2 {{.*}} sitofp
  %r175 = sitofp <2 x i16> undef to <2 x float>
  ; CHECK: cost of 1 {{.*}} uitofp
  %r176 = uitofp <2 x i32> undef to <2 x float>
  ; CHECK: cost of 1 {{.*}} sitofp
  %r177 = sitofp <2 x i32> undef to <2 x float>
  ; CHECK: cost of 24 {{.*}} uitofp
  %r178 = uitofp <2 x i64> undef to <2 x float>
  ; CHECK: cost of 24 {{.*}} sitofp
  %r179 = sitofp <2 x i64> undef to <2 x float>

  ; CHECK: cost of 8 {{.*}} uitofp
  %r180 = uitofp <2 x i1> undef to <2 x double>
  ; CHECK: cost of 8 {{.*}} sitofp
  %r181 = sitofp <2 x i1> undef to <2 x double>
  ; CHECK: cost of 4 {{.*}} uitofp
  %r182 = uitofp <2 x i8> undef to <2 x double>
  ; CHECK: cost of 4 {{.*}} sitofp
  %r183 = sitofp <2 x i8> undef to <2 x double>
  ; CHECK: cost of 3 {{.*}} uitofp
  %r184 = uitofp <2 x i16> undef to <2 x double>
  ; CHECK: cost of 3 {{.*}} sitofp
  %r185 = sitofp <2 x i16> undef to <2 x double>
  ; CHECK: cost of 2 {{.*}} uitofp
  %r186 = uitofp <2 x i32> undef to <2 x double>
  ; CHECK: cost of 2 {{.*}} sitofp
  %r187 = sitofp <2 x i32> undef to <2 x double>
  ; CHECK: cost of 24 {{.*}} uitofp
  %r188 = uitofp <2 x i64> undef to <2 x double>
  ; CHECK: cost of 24 {{.*}} sitofp
  %r189 = sitofp <2 x i64> undef to <2 x double>

  ; CHECK: cost of 3 {{.*}} uitofp
  %r190 = uitofp <4 x i1> undef to <4 x float>
  ; CHECK: cost of 3 {{.*}} sitofp
  %r191 = sitofp <4 x i1> undef to <4 x float>
  ; CHECK: cost of 3 {{.*}} uitofp
  %r192 = uitofp <4 x i8> undef to <4 x float>
  ; CHECK: cost of 3 {{.*}} sitofp
  %r193 = sitofp <4 x i8> undef to <4 x float>
  ; CHECK: cost of 2 {{.*}} uitofp
  %r194 = uitofp <4 x i16> undef to <4 x float>
  ; CHECK: cost of 2 {{.*}} sitofp
  %r195 = sitofp <4 x i16> undef to <4 x float>
  ; CHECK: cost of 1 {{.*}} uitofp
  %r196 = uitofp <4 x i32> undef to <4 x float>
  ; CHECK: cost of 1 {{.*}} sitofp
  %r197 = sitofp <4 x i32> undef to <4 x float>
  ; CHECK: cost of 48 {{.*}} uitofp
  %r198 = uitofp <4 x i64> undef to <4 x float>
  ; CHECK: cost of 48 {{.*}} sitofp
  %r199 = sitofp <4 x i64> undef to <4 x float>

  ; CHECK: cost of 16 {{.*}} uitofp
  %r200 = uitofp <4 x i1> undef to <4 x double>
  ; CHECK: cost of 16 {{.*}} sitofp
  %r201 = sitofp <4 x i1> undef to <4 x double>
  ; CHECK: cost of 16 {{.*}} uitofp
  %r202 = uitofp <4 x i8> undef to <4 x double>
  ; CHECK: cost of 16 {{.*}} sitofp
  %r203 = sitofp <4 x i8> undef to <4 x double>
  ; CHECK: cost of 16 {{.*}} uitofp
  %r204 = uitofp <4 x i16> undef to <4 x double>
  ; CHECK: cost of 16 {{.*}} sitofp
  %r205 = sitofp <4 x i16> undef to <4 x double>
  ; CHECK: cost of 16 {{.*}} uitofp
  %r206 = uitofp <4 x i32> undef to <4 x double>
  ; CHECK: cost of 16 {{.*}} sitofp
  %r207 = sitofp <4 x i32> undef to <4 x double>
  ; CHECK: cost of 48 {{.*}} uitofp
  %r208 = uitofp <4 x i64> undef to <4 x double>
  ; CHECK: cost of 48 {{.*}} sitofp
  %r209 = sitofp <4 x i64> undef to <4 x double>

  ; CHECK: cost of 32 {{.*}} uitofp
  %r210 = uitofp <8 x i1> undef to <8 x float>
  ; CHECK: cost of 32 {{.*}} sitofp
  %r211 = sitofp <8 x i1> undef to <8 x float>
  ; CHECK: cost of 32 {{.*}} uitofp
  %r212 = uitofp <8 x i8> undef to <8 x float>
  ; CHECK: cost of 32 {{.*}} sitofp
  %r213 = sitofp <8 x i8> undef to <8 x float>
  ; CHECK: cost of 4 {{.*}} uitofp
  %r214 = uitofp <8 x i16> undef to <8 x float>
  ; CHECK: cost of 4 {{.*}} sitofp
  %r215 = sitofp <8 x i16> undef to <8 x float>
  ; CHECK: cost of 2 {{.*}} uitofp
  %r216 = uitofp <8 x i32> undef to <8 x float>
  ; CHECK: cost of 2 {{.*}} sitofp
  %r217 = sitofp <8 x i32> undef to <8 x float>
  ; CHECK: cost of 96 {{.*}} uitofp
  %r218 = uitofp <8 x i64> undef to <8 x float>
  ; CHECK: cost of 96 {{.*}} sitofp
  %r219 = sitofp <8 x i64> undef to <8 x float>

  ; CHECK: cost of 32 {{.*}} uitofp
  %r220 = uitofp <8 x i1> undef to <8 x double>
  ; CHECK: cost of 32 {{.*}} sitofp
  %r221 = sitofp <8 x i1> undef to <8 x double>
  ; CHECK: cost of 32 {{.*}} uitofp
  %r222 = uitofp <8 x i8> undef to <8 x double>
  ; CHECK: cost of 32 {{.*}} sitofp
  %r223 = sitofp <8 x i8> undef to <8 x double>
  ; CHECK: cost of 32 {{.*}} uitofp
  %r224 = uitofp <8 x i16> undef to <8 x double>
  ; CHECK: cost of 32 {{.*}} sitofp
  %r225 = sitofp <8 x i16> undef to <8 x double>
  ; CHECK: cost of 32 {{.*}} uitofp
  %r226 = uitofp <8 x i16> undef to <8 x double>
  ; CHECK: cost of 32 {{.*}} sitofp
  %r227 = sitofp <8 x i16> undef to <8 x double>
  ; CHECK: cost of 96 {{.*}} uitofp
  %r228 = uitofp <8 x i64> undef to <8 x double>
  ; CHECK: cost of 96 {{.*}} sitofp
  %r229 = sitofp <8 x i64> undef to <8 x double>

  ; CHECK: cost of 64 {{.*}} uitofp
  %r230 = uitofp <16 x i1> undef to <16 x float>
  ; CHECK: cost of 64 {{.*}} sitofp
  %r231 = sitofp <16 x i1> undef to <16 x float>
  ; CHECK: cost of 64 {{.*}} uitofp
  %r232 = uitofp <16 x i8> undef to <16 x float>
  ; CHECK: cost of 64 {{.*}} sitofp
  %r233 = sitofp <16 x i8> undef to <16 x float>
  ; CHECK: cost of 8 {{.*}} uitofp
  %r234 = uitofp <16 x i16> undef to <16 x float>
  ; CHECK: cost of 8 {{.*}} sitofp
  %r235 = sitofp <16 x i16> undef to <16 x float>
  ; CHECK: cost of 4 {{.*}} uitofp
  %r236 = uitofp <16 x i32> undef to <16 x float>
  ; CHECK: cost of 4 {{.*}} sitofp
  %r237 = sitofp <16 x i32> undef to <16 x float>
  ; CHECK: cost of 192 {{.*}} uitofp
  %r238 = uitofp <16 x i64> undef to <16 x float>
  ; CHECK: cost of 192 {{.*}} sitofp
  %r239 = sitofp <16 x i64> undef to <16 x float>

  ; CHECK: cost of 64 {{.*}} uitofp
  %r240 = uitofp <16 x i1> undef to <16 x double>
  ; CHECK: cost of 64 {{.*}} sitofp
  %r241 = sitofp <16 x i1> undef to <16 x double>
  ; CHECK: cost of 64 {{.*}} uitofp
  %r242 = uitofp <16 x i8> undef to <16 x double>
  ; CHECK: cost of 64 {{.*}} sitofp
  %r243 = sitofp <16 x i8> undef to <16 x double>
  ; C4ECK: cost of 64 {{.*}} uitofp
  %r244 = uitofp <16 x i16> undef to <16 x double>
  ; CHECK: cost of 64 {{.*}} sitofp
  %r245 = sitofp <16 x i16> undef to <16 x double>
  ; CHECK: cost of 64 {{.*}} uitofp
  %r246 = uitofp <16 x i16> undef to <16 x double>
  ; CHECK: cost of 64 {{.*}} sitofp
  %r247 = sitofp <16 x i16> undef to <16 x double>
  ; CHECK: cost of 192 {{.*}} uitofp
  %r248 = uitofp <16 x i64> undef to <16 x double>
  ; CHECK: cost of 192 {{.*}} sitofp
  %r249 = sitofp <16 x i64> undef to <16 x double>

  ;CHECK: cost of 0 {{.*}} ret
  ret i32 undef
}

