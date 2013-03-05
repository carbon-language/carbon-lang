; RUN: llc -mtriple=mips64el-unknown-unknown -mcpu=mips64 -soft-float -O1 \
; RUN: -disable-mips-delay-filler < %s | FileCheck %s

@gld0 = external global fp128
@gld1 = external global fp128
@gld2 = external global fp128
@gf1 = external global float
@gd1 = external global double

; CHECK: addLD:
; CHECK: ld $25, %call16(__addtf3)

define fp128 @addLD() {
entry:
  %0 = load fp128* @gld0, align 16
  %1 = load fp128* @gld1, align 16
  %add = fadd fp128 %0, %1
  ret fp128 %add
}

; CHECK: subLD:
; CHECK: ld $25, %call16(__subtf3)

define fp128 @subLD() {
entry:
  %0 = load fp128* @gld0, align 16
  %1 = load fp128* @gld1, align 16
  %sub = fsub fp128 %0, %1
  ret fp128 %sub
}

; CHECK: mulLD:
; CHECK: ld $25, %call16(__multf3)

define fp128 @mulLD() {
entry:
  %0 = load fp128* @gld0, align 16
  %1 = load fp128* @gld1, align 16
  %mul = fmul fp128 %0, %1
  ret fp128 %mul
}

; CHECK: divLD:
; CHECK: ld $25, %call16(__divtf3)

define fp128 @divLD() {
entry:
  %0 = load fp128* @gld0, align 16
  %1 = load fp128* @gld1, align 16
  %div = fdiv fp128 %0, %1
  ret fp128 %div
}

; CHECK: conv_LD_char:
; CHECK: ld $25, %call16(__floatsitf)

define fp128 @conv_LD_char(i8 signext %a) {
entry:
  %conv = sitofp i8 %a to fp128
  ret fp128 %conv
}

; CHECK: conv_LD_short:
; CHECK: ld $25, %call16(__floatsitf)

define fp128 @conv_LD_short(i16 signext %a) {
entry:
  %conv = sitofp i16 %a to fp128
  ret fp128 %conv
}

; CHECK: conv_LD_int:
; CHECK: ld $25, %call16(__floatsitf)

define fp128 @conv_LD_int(i32 %a) {
entry:
  %conv = sitofp i32 %a to fp128
  ret fp128 %conv
}

; CHECK: conv_LD_LL:
; CHECK: ld $25, %call16(__floatditf)

define fp128 @conv_LD_LL(i64 %a) {
entry:
  %conv = sitofp i64 %a to fp128
  ret fp128 %conv
}

; CHECK: conv_LD_UChar:
; CHECK: ld $25, %call16(__floatunsitf)

define fp128 @conv_LD_UChar(i8 zeroext %a) {
entry:
  %conv = uitofp i8 %a to fp128
  ret fp128 %conv
}

; CHECK: conv_LD_UShort:
; CHECK: ld $25, %call16(__floatunsitf)

define fp128 @conv_LD_UShort(i16 zeroext %a) {
entry:
  %conv = uitofp i16 %a to fp128
  ret fp128 %conv
}

; CHECK: conv_LD_UInt:
; CHECK: ld $25, %call16(__floatunsitf)

define fp128 @conv_LD_UInt(i32 %a) {
entry:
  %conv = uitofp i32 %a to fp128
  ret fp128 %conv
}

; CHECK: conv_LD_ULL:
; CHECK: ld $25, %call16(__floatunditf)

define fp128 @conv_LD_ULL(i64 %a) {
entry:
  %conv = uitofp i64 %a to fp128
  ret fp128 %conv
}

; CHECK: conv_char_LD:
; CHECK: ld $25, %call16(__fixtfsi)

define signext i8 @conv_char_LD(fp128 %a) {
entry:
  %conv = fptosi fp128 %a to i8
  ret i8 %conv
}

; CHECK: conv_short_LD:
; CHECK: ld $25, %call16(__fixtfsi)

define signext i16 @conv_short_LD(fp128 %a) {
entry:
  %conv = fptosi fp128 %a to i16
  ret i16 %conv
}

; CHECK: conv_int_LD:
; CHECK: ld $25, %call16(__fixtfsi)

define i32 @conv_int_LD(fp128 %a) {
entry:
  %conv = fptosi fp128 %a to i32
  ret i32 %conv
}

; CHECK: conv_LL_LD:
; CHECK: ld $25, %call16(__fixtfdi)

define i64 @conv_LL_LD(fp128 %a) {
entry:
  %conv = fptosi fp128 %a to i64
  ret i64 %conv
}

; CHECK: conv_UChar_LD:
; CHECK: ld $25, %call16(__fixtfsi)

define zeroext i8 @conv_UChar_LD(fp128 %a) {
entry:
  %conv = fptoui fp128 %a to i8
  ret i8 %conv
}

; CHECK: conv_UShort_LD:
; CHECK: ld $25, %call16(__fixtfsi)

define zeroext i16 @conv_UShort_LD(fp128 %a) {
entry:
  %conv = fptoui fp128 %a to i16
  ret i16 %conv
}

; CHECK: conv_UInt_LD:
; CHECK: ld $25, %call16(__fixunstfsi)

define i32 @conv_UInt_LD(fp128 %a) {
entry:
  %conv = fptoui fp128 %a to i32
  ret i32 %conv
}

; CHECK: conv_ULL_LD:
; CHECK: ld $25, %call16(__fixunstfdi)

define i64 @conv_ULL_LD(fp128 %a) {
entry:
  %conv = fptoui fp128 %a to i64
  ret i64 %conv
}

; CHECK: conv_LD_float:
; CHECK: ld $25, %call16(__extendsftf2)

define fp128 @conv_LD_float(float %a) {
entry:
  %conv = fpext float %a to fp128
  ret fp128 %conv
}

; CHECK: conv_LD_double:
; CHECK: ld $25, %call16(__extenddftf2)

define fp128 @conv_LD_double(double %a) {
entry:
  %conv = fpext double %a to fp128
  ret fp128 %conv
}

; CHECK: conv_float_LD:
; CHECK: ld $25, %call16(__trunctfsf2)

define float @conv_float_LD(fp128 %a) {
entry:
  %conv = fptrunc fp128 %a to float
  ret float %conv
}

; CHECK: conv_double_LD:
; CHECK: ld $25, %call16(__trunctfdf2)

define double @conv_double_LD(fp128 %a) {
entry:
  %conv = fptrunc fp128 %a to double
  ret double %conv
}

; CHECK: libcall1_fabsl:
; CHECK: ld      $[[R0:[0-9]+]], 8($[[R4:[0-9]+]])
; CHECK: daddiu  $[[R1:[0-9]+]], $zero, 1
; CHECK: dsll    $[[R2:[0-9]+]], $[[R1]], 63
; CHECK: daddiu  $[[R3:[0-9]+]], $[[R2]], -1
; CHECK: and     $4, $[[R0]], $[[R3]]
; CHECK: ld      $2, 0($[[R4]])

define fp128 @libcall1_fabsl() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @fabsl(fp128 %0) nounwind readnone
  ret fp128 %call
}

declare fp128 @fabsl(fp128) #1

; CHECK: libcall1_ceill:
; CHECK: ld $25, %call16(ceill)

define fp128 @libcall1_ceill() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @ceill(fp128 %0) nounwind readnone
  ret fp128 %call
}

declare fp128 @ceill(fp128) #1

; CHECK: libcall1_sinl:
; CHECK: ld $25, %call16(sinl)

define fp128 @libcall1_sinl() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @sinl(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @sinl(fp128) #2

; CHECK: libcall1_cosl:
; CHECK: ld $25, %call16(cosl)

define fp128 @libcall1_cosl() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @cosl(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @cosl(fp128) #2

; CHECK: libcall1_expl:
; CHECK: ld $25, %call16(expl)

define fp128 @libcall1_expl() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @expl(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @expl(fp128) #2

; CHECK: libcall1_exp2l:
; CHECK: ld $25, %call16(exp2l)

define fp128 @libcall1_exp2l() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @exp2l(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @exp2l(fp128) #2

; CHECK: libcall1_logl:
; CHECK: ld $25, %call16(logl)

define fp128 @libcall1_logl() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @logl(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @logl(fp128) #2

; CHECK: libcall1_log2l:
; CHECK: ld $25, %call16(log2l)

define fp128 @libcall1_log2l() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @log2l(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @log2l(fp128) #2

; CHECK: libcall1_log10l:
; CHECK: ld $25, %call16(log10l)

define fp128 @libcall1_log10l() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @log10l(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @log10l(fp128) #2

; CHECK: libcall1_nearbyintl:
; CHECK: ld $25, %call16(nearbyintl)

define fp128 @libcall1_nearbyintl() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @nearbyintl(fp128 %0) nounwind readnone
  ret fp128 %call
}

declare fp128 @nearbyintl(fp128) #1

; CHECK: libcall1_floorl:
; CHECK: ld $25, %call16(floorl)

define fp128 @libcall1_floorl() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @floorl(fp128 %0) nounwind readnone
  ret fp128 %call
}

declare fp128 @floorl(fp128) #1

; CHECK: libcall1_sqrtl:
; CHECK: ld $25, %call16(sqrtl)

define fp128 @libcall1_sqrtl() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @sqrtl(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @sqrtl(fp128) #2

; CHECK: libcall1_rintl:
; CHECK: ld $25, %call16(rintl)

define fp128 @libcall1_rintl() {
entry:
  %0 = load fp128* @gld0, align 16
  %call = tail call fp128 @rintl(fp128 %0) nounwind readnone
  ret fp128 %call
}

declare fp128 @rintl(fp128) #1

; CHECK: libcall_powil:
; CHECK: ld $25, %call16(__powitf2)

define fp128 @libcall_powil(fp128 %a, i32 %b) {
entry:
  %0 = tail call fp128 @llvm.powi.f128(fp128 %a, i32 %b)
  ret fp128 %0
}

declare fp128 @llvm.powi.f128(fp128, i32) #3

; CHECK: libcall2_copysignl:
; CHECK: daddiu $[[R2:[0-9]+]], $zero, 1
; CHECK: dsll   $[[R3:[0-9]+]], $[[R2]], 63
; CHECK: ld     $[[R0:[0-9]+]], %got_disp(gld1)
; CHECK: ld     $[[R1:[0-9]+]], 8($[[R0]])
; CHECK: and    $[[R4:[0-9]+]], $[[R1]], $[[R3]]
; CHECK: ld     $[[R5:[0-9]+]], %got_disp(gld0)
; CHECK: ld     $[[R6:[0-9]+]], 8($[[R5]])
; CHECK: daddiu $[[R7:[0-9]+]], $[[R3]], -1
; CHECK: and    $[[R8:[0-9]+]], $[[R6]], $[[R7]]
; CHECK: or     $4, $[[R8]], $[[R4]]
; CHECK: ld     $2, 0($[[R5]])

define fp128 @libcall2_copysignl() {
entry:
  %0 = load fp128* @gld0, align 16
  %1 = load fp128* @gld1, align 16
  %call = tail call fp128 @copysignl(fp128 %0, fp128 %1) nounwind readnone
  ret fp128 %call
}

declare fp128 @copysignl(fp128, fp128) #1

; CHECK: libcall2_powl:
; CHECK: ld $25, %call16(powl)

define fp128 @libcall2_powl() {
entry:
  %0 = load fp128* @gld0, align 16
  %1 = load fp128* @gld1, align 16
  %call = tail call fp128 @powl(fp128 %0, fp128 %1) nounwind
  ret fp128 %call
}

declare fp128 @powl(fp128, fp128) #2

; CHECK: libcall2_fmodl:
; CHECK: ld $25, %call16(fmodl)

define fp128 @libcall2_fmodl() {
entry:
  %0 = load fp128* @gld0, align 16
  %1 = load fp128* @gld1, align 16
  %call = tail call fp128 @fmodl(fp128 %0, fp128 %1) nounwind
  ret fp128 %call
}

declare fp128 @fmodl(fp128, fp128) #2

; CHECK: libcall3_fmal:
; CHECK: ld $25, %call16(fmal)

define fp128 @libcall3_fmal() {
entry:
  %0 = load fp128* @gld0, align 16
  %1 = load fp128* @gld2, align 16
  %2 = load fp128* @gld1, align 16
  %3 = tail call fp128 @llvm.fma.f128(fp128 %0, fp128 %2, fp128 %1)
  ret fp128 %3
}

declare fp128 @llvm.fma.f128(fp128, fp128, fp128) #4

; CHECK: cmp_lt:
; CHECK: ld $25, %call16(__lttf2)

define i32 @cmp_lt(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp olt fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK: cmp_le:
; CHECK: ld $25, %call16(__letf2)

define i32 @cmp_le(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp ole fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK: cmp_gt:
; CHECK: ld $25, %call16(__gttf2)

define i32 @cmp_gt(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp ogt fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK: cmp_ge:
; CHECK: ld $25, %call16(__getf2)

define i32 @cmp_ge(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp oge fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK: cmp_eq:
; CHECK: ld $25, %call16(__eqtf2)

define i32 @cmp_eq(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp oeq fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK: cmp_ne:
; CHECK: ld $25, %call16(__netf2)

define i32 @cmp_ne(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp une fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK: load_LD_LD:
; CHECK: ld $[[R0:[0-9]+]], %got_disp(gld1)
; CHECK: ld $2, 0($[[R0]])
; CHECK: ld $4, 8($[[R0]])

define fp128 @load_LD_LD() {
entry:
  %0 = load fp128* @gld1, align 16
  ret fp128 %0
}

; CHECK: load_LD_float:
; CHECK: ld   $[[R0:[0-9]+]], %got_disp(gf1)
; CHECK: lw   $4, 0($[[R0]])
; CHECK: ld   $25, %call16(__extendsftf2)
; CHECK: jalr $25

define fp128 @load_LD_float() {
entry:
  %0 = load float* @gf1, align 4
  %conv = fpext float %0 to fp128
  ret fp128 %conv
}

; CHECK: load_LD_double:
; CHECK: ld   $[[R0:[0-9]+]], %got_disp(gd1)
; CHECK: ld   $4, 0($[[R0]])
; CHECK: ld   $25, %call16(__extenddftf2)
; CHECK: jalr $25

define fp128 @load_LD_double() {
entry:
  %0 = load double* @gd1, align 8
  %conv = fpext double %0 to fp128
  ret fp128 %conv
}

; CHECK: store_LD_LD:
; CHECK: ld $[[R0:[0-9]+]], %got_disp(gld1)
; CHECK: ld $[[R1:[0-9]+]], 0($[[R0]])
; CHECK: ld $[[R2:[0-9]+]], 8($[[R0]])
; CHECK: ld $[[R3:[0-9]+]], %got_disp(gld0)
; CHECK: sd $[[R2]], 8($[[R3]])
; CHECK: sd $[[R1]], 0($[[R3]])

define void @store_LD_LD() {
entry:
  %0 = load fp128* @gld1, align 16
  store fp128 %0, fp128* @gld0, align 16
  ret void
}

; CHECK: store_LD_float:
; CHECK: ld   $[[R0:[0-9]+]], %got_disp(gld1)
; CHECK: ld   $4, 0($[[R0]])
; CHECK: ld   $5, 8($[[R0]])
; CHECK: ld   $25, %call16(__trunctfsf2)
; CHECK: jalr $25
; CHECK: ld   $[[R1:[0-9]+]], %got_disp(gf1)
; CHECK: sw   $2, 0($[[R1]])

define void @store_LD_float() {
entry:
  %0 = load fp128* @gld1, align 16
  %conv = fptrunc fp128 %0 to float
  store float %conv, float* @gf1, align 4
  ret void
}

; CHECK: store_LD_double:
; CHECK: ld   $[[R0:[0-9]+]], %got_disp(gld1)
; CHECK: ld   $4, 0($[[R0]])
; CHECK: ld   $5, 8($[[R0]])
; CHECK: ld   $25, %call16(__trunctfdf2)
; CHECK: jalr $25
; CHECK: ld   $[[R1:[0-9]+]], %got_disp(gd1)
; CHECK: sd   $2, 0($[[R1]])

define void @store_LD_double() {
entry:
  %0 = load fp128* @gld1, align 16
  %conv = fptrunc fp128 %0 to double
  store double %conv, double* @gd1, align 8
  ret void
}

; CHECK: select_LD:
; CHECK: movn $8, $6, $4
; CHECK: movn $9, $7, $4
; CHECK: move $2, $8
; CHECK: move $4, $9

define fp128 @select_LD(i32 %a, i64, fp128 %b, fp128 %c) {
entry:
  %tobool = icmp ne i32 %a, 0
  %cond = select i1 %tobool, fp128 %b, fp128 %c
  ret fp128 %cond
}

; CHECK: selectCC_LD:
; CHECK: move $[[R0:[0-9]+]], $11
; CHECK: move $[[R1:[0-9]+]], $10
; CHECK: move $[[R2:[0-9]+]], $9
; CHECK: move $[[R3:[0-9]+]], $8
; CHECK: ld   $25, %call16(__gttf2)($gp)
; CHECK: jalr $25
; CHECK: slti $1, $2, 1
; CHECK: movz $[[R1]], $[[R3]], $1
; CHECK: movz $[[R0]], $[[R2]], $1
; CHECK: move $2, $[[R1]]
; CHECK: move $4, $[[R0]]

define fp128 @selectCC_LD(fp128 %a, fp128 %b, fp128 %c, fp128 %d) {
entry:
  %cmp = fcmp ogt fp128 %a, %b
  %cond = select i1 %cmp, fp128 %c, fp128 %d
  ret fp128 %cond
}
