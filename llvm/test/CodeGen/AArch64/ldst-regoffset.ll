; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-none-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=arm64-none-linux-gnu -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP %s

@var_8bit = global i8 0
@var_16bit = global i16 0
@var_32bit = global i32 0
@var_64bit = global i64 0

@var_float = global float 0.0
@var_double = global double 0.0

define void @ldst_8bit(i8* %base, i32 %off32, i64 %off64) minsize {
; CHECK-LABEL: ldst_8bit:

   %addr8_sxtw = getelementptr i8* %base, i32 %off32
   %val8_sxtw = load volatile i8* %addr8_sxtw
   %val32_signed = sext i8 %val8_sxtw to i32
   store volatile i32 %val32_signed, i32* @var_32bit
; CHECK: ldrsb {{w[0-9]+}}, [{{x[0-9]+}}, {{[wx][0-9]+}}, sxtw]

  %addr_lsl = getelementptr i8* %base, i64 %off64
  %val8_lsl = load volatile i8* %addr_lsl
  %val32_unsigned = zext i8 %val8_lsl to i32
  store volatile i32 %val32_unsigned, i32* @var_32bit
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}]

  %addrint_uxtw = ptrtoint i8* %base to i64
  %offset_uxtw = zext i32 %off32 to i64
  %addrint1_uxtw = add i64 %addrint_uxtw, %offset_uxtw
  %addr_uxtw = inttoptr i64 %addrint1_uxtw to i8*
  %val8_uxtw = load volatile i8* %addr_uxtw
  %newval8 = add i8 %val8_uxtw, 1
  store volatile i8 %newval8, i8* @var_8bit
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw]

   ret void
}


define void @ldst_16bit(i16* %base, i32 %off32, i64 %off64) minsize {
; CHECK-LABEL: ldst_16bit:

   %addr8_sxtwN = getelementptr i16* %base, i32 %off32
   %val8_sxtwN = load volatile i16* %addr8_sxtwN
   %val32_signed = sext i16 %val8_sxtwN to i32
   store volatile i32 %val32_signed, i32* @var_32bit
; CHECK: ldrsh {{w[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #1]

  %addr_lslN = getelementptr i16* %base, i64 %off64
  %val8_lslN = load volatile i16* %addr_lslN
  %val32_unsigned = zext i16 %val8_lslN to i32
  store volatile i32 %val32_unsigned, i32* @var_32bit
; CHECK: ldrh {{w[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}, lsl #1]

  %addrint_uxtw = ptrtoint i16* %base to i64
  %offset_uxtw = zext i32 %off32 to i64
  %addrint1_uxtw = add i64 %addrint_uxtw, %offset_uxtw
  %addr_uxtw = inttoptr i64 %addrint1_uxtw to i16*
  %val8_uxtw = load volatile i16* %addr_uxtw
  %newval8 = add i16 %val8_uxtw, 1
  store volatile i16 %newval8, i16* @var_16bit
; CHECK: ldrh {{w[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw]

  %base_sxtw = ptrtoint i16* %base to i64
  %offset_sxtw = sext i32 %off32 to i64
  %addrint_sxtw = add i64 %base_sxtw, %offset_sxtw
  %addr_sxtw = inttoptr i64 %addrint_sxtw to i16*
  %val16_sxtw = load volatile i16* %addr_sxtw
  %val64_signed = sext i16 %val16_sxtw to i64
  store volatile i64 %val64_signed, i64* @var_64bit
; CHECK: ldrsh {{x[0-9]+}}, [{{x[0-9]+}}, {{[wx][0-9]+}}, sxtw]


  %base_lsl = ptrtoint i16* %base to i64
  %addrint_lsl = add i64 %base_lsl, %off64
  %addr_lsl = inttoptr i64 %addrint_lsl to i16*
  %val16_lsl = load volatile i16* %addr_lsl
  %val64_unsigned = zext i16 %val16_lsl to i64
  store volatile i64 %val64_unsigned, i64* @var_64bit
; CHECK: ldrh {{w[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}]

  %base_uxtwN = ptrtoint i16* %base to i64
  %offset_uxtwN = zext i32 %off32 to i64
  %offset2_uxtwN = shl i64 %offset_uxtwN, 1
  %addrint_uxtwN = add i64 %base_uxtwN, %offset2_uxtwN
  %addr_uxtwN = inttoptr i64 %addrint_uxtwN to i16*
  %val32 = load volatile i32* @var_32bit
  %val16_trunc32 = trunc i32 %val32 to i16
  store volatile i16 %val16_trunc32, i16* %addr_uxtwN
; CHECK: strh {{w[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw #1]
   ret void
}

define void @ldst_32bit(i32* %base, i32 %off32, i64 %off64) minsize {
; CHECK-LABEL: ldst_32bit:

   %addr_sxtwN = getelementptr i32* %base, i32 %off32
   %val_sxtwN = load volatile i32* %addr_sxtwN
   store volatile i32 %val_sxtwN, i32* @var_32bit
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #2]

  %addr_lslN = getelementptr i32* %base, i64 %off64
  %val_lslN = load volatile i32* %addr_lslN
  store volatile i32 %val_lslN, i32* @var_32bit
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}, lsl #2]

  %addrint_uxtw = ptrtoint i32* %base to i64
  %offset_uxtw = zext i32 %off32 to i64
  %addrint1_uxtw = add i64 %addrint_uxtw, %offset_uxtw
  %addr_uxtw = inttoptr i64 %addrint1_uxtw to i32*
  %val_uxtw = load volatile i32* %addr_uxtw
  %newval8 = add i32 %val_uxtw, 1
  store volatile i32 %newval8, i32* @var_32bit
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw]


  %base_sxtw = ptrtoint i32* %base to i64
  %offset_sxtw = sext i32 %off32 to i64
  %addrint_sxtw = add i64 %base_sxtw, %offset_sxtw
  %addr_sxtw = inttoptr i64 %addrint_sxtw to i32*
  %val16_sxtw = load volatile i32* %addr_sxtw
  %val64_signed = sext i32 %val16_sxtw to i64
  store volatile i64 %val64_signed, i64* @var_64bit
; CHECK: ldrsw {{x[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw]


  %base_lsl = ptrtoint i32* %base to i64
  %addrint_lsl = add i64 %base_lsl, %off64
  %addr_lsl = inttoptr i64 %addrint_lsl to i32*
  %val16_lsl = load volatile i32* %addr_lsl
  %val64_unsigned = zext i32 %val16_lsl to i64
  store volatile i64 %val64_unsigned, i64* @var_64bit
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}]

  %base_uxtwN = ptrtoint i32* %base to i64
  %offset_uxtwN = zext i32 %off32 to i64
  %offset2_uxtwN = shl i64 %offset_uxtwN, 2
  %addrint_uxtwN = add i64 %base_uxtwN, %offset2_uxtwN
  %addr_uxtwN = inttoptr i64 %addrint_uxtwN to i32*
  %val32 = load volatile i32* @var_32bit
  store volatile i32 %val32, i32* %addr_uxtwN
; CHECK: str {{w[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw #2]
   ret void
}

define void @ldst_64bit(i64* %base, i32 %off32, i64 %off64) minsize {
; CHECK-LABEL: ldst_64bit:

   %addr_sxtwN = getelementptr i64* %base, i32 %off32
   %val_sxtwN = load volatile i64* %addr_sxtwN
   store volatile i64 %val_sxtwN, i64* @var_64bit
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #3]

  %addr_lslN = getelementptr i64* %base, i64 %off64
  %val_lslN = load volatile i64* %addr_lslN
  store volatile i64 %val_lslN, i64* @var_64bit
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}, lsl #3]

  %addrint_uxtw = ptrtoint i64* %base to i64
  %offset_uxtw = zext i32 %off32 to i64
  %addrint1_uxtw = add i64 %addrint_uxtw, %offset_uxtw
  %addr_uxtw = inttoptr i64 %addrint1_uxtw to i64*
  %val8_uxtw = load volatile i64* %addr_uxtw
  %newval8 = add i64 %val8_uxtw, 1
  store volatile i64 %newval8, i64* @var_64bit
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw]

  %base_sxtw = ptrtoint i64* %base to i64
  %offset_sxtw = sext i32 %off32 to i64
  %addrint_sxtw = add i64 %base_sxtw, %offset_sxtw
  %addr_sxtw = inttoptr i64 %addrint_sxtw to i64*
  %val64_sxtw = load volatile i64* %addr_sxtw
  store volatile i64 %val64_sxtw, i64* @var_64bit
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw]

  %base_lsl = ptrtoint i64* %base to i64
  %addrint_lsl = add i64 %base_lsl, %off64
  %addr_lsl = inttoptr i64 %addrint_lsl to i64*
  %val64_lsl = load volatile i64* %addr_lsl
  store volatile i64 %val64_lsl, i64* @var_64bit
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}]

  %base_uxtwN = ptrtoint i64* %base to i64
  %offset_uxtwN = zext i32 %off32 to i64
  %offset2_uxtwN = shl i64 %offset_uxtwN, 3
  %addrint_uxtwN = add i64 %base_uxtwN, %offset2_uxtwN
  %addr_uxtwN = inttoptr i64 %addrint_uxtwN to i64*
  %val64 = load volatile i64* @var_64bit
  store volatile i64 %val64, i64* %addr_uxtwN
; CHECK: str {{x[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw #3]
   ret void
}

define void @ldst_float(float* %base, i32 %off32, i64 %off64) minsize {
; CHECK-LABEL: ldst_float:

   %addr_sxtwN = getelementptr float* %base, i32 %off32
   %val_sxtwN = load volatile float* %addr_sxtwN
   store volatile float %val_sxtwN, float* @var_float
; CHECK: ldr {{s[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #2]
; CHECK-NOFP-NOT: ldr {{s[0-9]+}},

  %addr_lslN = getelementptr float* %base, i64 %off64
  %val_lslN = load volatile float* %addr_lslN
  store volatile float %val_lslN, float* @var_float
; CHECK: ldr {{s[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}, lsl #2]
; CHECK-NOFP-NOT: ldr {{s[0-9]+}},

  %addrint_uxtw = ptrtoint float* %base to i64
  %offset_uxtw = zext i32 %off32 to i64
  %addrint1_uxtw = add i64 %addrint_uxtw, %offset_uxtw
  %addr_uxtw = inttoptr i64 %addrint1_uxtw to float*
  %val_uxtw = load volatile float* %addr_uxtw
  store volatile float %val_uxtw, float* @var_float
; CHECK: ldr {{s[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw]
; CHECK-NOFP-NOT: ldr {{s[0-9]+}},

  %base_sxtw = ptrtoint float* %base to i64
  %offset_sxtw = sext i32 %off32 to i64
  %addrint_sxtw = add i64 %base_sxtw, %offset_sxtw
  %addr_sxtw = inttoptr i64 %addrint_sxtw to float*
  %val64_sxtw = load volatile float* %addr_sxtw
  store volatile float %val64_sxtw, float* @var_float
; CHECK: ldr {{s[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw]
; CHECK-NOFP-NOT: ldr {{s[0-9]+}},

  %base_lsl = ptrtoint float* %base to i64
  %addrint_lsl = add i64 %base_lsl, %off64
  %addr_lsl = inttoptr i64 %addrint_lsl to float*
  %val64_lsl = load volatile float* %addr_lsl
  store volatile float %val64_lsl, float* @var_float
; CHECK: ldr {{s[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}]
; CHECK-NOFP-NOT: ldr {{s[0-9]+}},

  %base_uxtwN = ptrtoint float* %base to i64
  %offset_uxtwN = zext i32 %off32 to i64
  %offset2_uxtwN = shl i64 %offset_uxtwN, 2
  %addrint_uxtwN = add i64 %base_uxtwN, %offset2_uxtwN
  %addr_uxtwN = inttoptr i64 %addrint_uxtwN to float*
  %val64 = load volatile float* @var_float
  store volatile float %val64, float* %addr_uxtwN
; CHECK: str {{s[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw #2]
; CHECK-NOFP-NOT: ldr {{s[0-9]+}},
   ret void
}

define void @ldst_double(double* %base, i32 %off32, i64 %off64) minsize {
; CHECK-LABEL: ldst_double:

   %addr_sxtwN = getelementptr double* %base, i32 %off32
   %val_sxtwN = load volatile double* %addr_sxtwN
   store volatile double %val_sxtwN, double* @var_double
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #3]
; CHECK-NOFP-NOT: ldr {{d[0-9]+}},

  %addr_lslN = getelementptr double* %base, i64 %off64
  %val_lslN = load volatile double* %addr_lslN
  store volatile double %val_lslN, double* @var_double
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}, lsl #3]
; CHECK-NOFP-NOT: ldr {{d[0-9]+}},

  %addrint_uxtw = ptrtoint double* %base to i64
  %offset_uxtw = zext i32 %off32 to i64
  %addrint1_uxtw = add i64 %addrint_uxtw, %offset_uxtw
  %addr_uxtw = inttoptr i64 %addrint1_uxtw to double*
  %val_uxtw = load volatile double* %addr_uxtw
  store volatile double %val_uxtw, double* @var_double
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw]
; CHECK-NOFP-NOT: ldr {{d[0-9]+}},

  %base_sxtw = ptrtoint double* %base to i64
  %offset_sxtw = sext i32 %off32 to i64
  %addrint_sxtw = add i64 %base_sxtw, %offset_sxtw
  %addr_sxtw = inttoptr i64 %addrint_sxtw to double*
  %val64_sxtw = load volatile double* %addr_sxtw
  store volatile double %val64_sxtw, double* @var_double
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw]
; CHECK-NOFP-NOT: ldr {{d[0-9]+}},

  %base_lsl = ptrtoint double* %base to i64
  %addrint_lsl = add i64 %base_lsl, %off64
  %addr_lsl = inttoptr i64 %addrint_lsl to double*
  %val64_lsl = load volatile double* %addr_lsl
  store volatile double %val64_lsl, double* @var_double
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}]
; CHECK-NOFP-NOT: ldr {{d[0-9]+}},

  %base_uxtwN = ptrtoint double* %base to i64
  %offset_uxtwN = zext i32 %off32 to i64
  %offset2_uxtwN = shl i64 %offset_uxtwN, 3
  %addrint_uxtwN = add i64 %base_uxtwN, %offset2_uxtwN
  %addr_uxtwN = inttoptr i64 %addrint_uxtwN to double*
  %val64 = load volatile double* @var_double
  store volatile double %val64, double* %addr_uxtwN
; CHECK: str {{d[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw #3]
; CHECK-NOFP-NOT: ldr {{d[0-9]+}},
   ret void
}


define void @ldst_128bit(fp128* %base, i32 %off32, i64 %off64) minsize {
; CHECK-LABEL: ldst_128bit:

   %addr_sxtwN = getelementptr fp128* %base, i32 %off32
   %val_sxtwN = load volatile fp128* %addr_sxtwN
   store volatile fp128 %val_sxtwN, fp128* %base
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #4]
; CHECK-NOFP-NOT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #4]

  %addr_lslN = getelementptr fp128* %base, i64 %off64
  %val_lslN = load volatile fp128* %addr_lslN
  store volatile fp128 %val_lslN, fp128* %base
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}, lsl #4]
; CHECK-NOFP-NOT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #4]

  %addrint_uxtw = ptrtoint fp128* %base to i64
  %offset_uxtw = zext i32 %off32 to i64
  %addrint1_uxtw = add i64 %addrint_uxtw, %offset_uxtw
  %addr_uxtw = inttoptr i64 %addrint1_uxtw to fp128*
  %val_uxtw = load volatile fp128* %addr_uxtw
  store volatile fp128 %val_uxtw, fp128* %base
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw]
; CHECK-NOFP-NOT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #4]

  %base_sxtw = ptrtoint fp128* %base to i64
  %offset_sxtw = sext i32 %off32 to i64
  %addrint_sxtw = add i64 %base_sxtw, %offset_sxtw
  %addr_sxtw = inttoptr i64 %addrint_sxtw to fp128*
  %val64_sxtw = load volatile fp128* %addr_sxtw
  store volatile fp128 %val64_sxtw, fp128* %base
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw]
; CHECK-NOFP-NOT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #4]

  %base_lsl = ptrtoint fp128* %base to i64
  %addrint_lsl = add i64 %base_lsl, %off64
  %addr_lsl = inttoptr i64 %addrint_lsl to fp128*
  %val64_lsl = load volatile fp128* %addr_lsl
  store volatile fp128 %val64_lsl, fp128* %base
; CHECK: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}]
; CHECK-NOFP-NOT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #4]

  %base_uxtwN = ptrtoint fp128* %base to i64
  %offset_uxtwN = zext i32 %off32 to i64
  %offset2_uxtwN = shl i64 %offset_uxtwN, 4
  %addrint_uxtwN = add i64 %base_uxtwN, %offset2_uxtwN
  %addr_uxtwN = inttoptr i64 %addrint_uxtwN to fp128*
  %val64 = load volatile fp128* %base
  store volatile fp128 %val64, fp128* %addr_uxtwN
; CHECK: str {{q[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, uxtw #4]
; CHECK-NOFP-NOT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, {{[xw][0-9]+}}, sxtw #4]
   ret void
}
