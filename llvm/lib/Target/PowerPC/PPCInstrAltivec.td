//===- PPCInstrAltivec.td - The PowerPC Altivec Extension --*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the Altivec extension to the PowerPC instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Altivec transformation functions and pattern fragments.
//

// VSPLT_get_imm xform function: convert vector_shuffle mask to VSPLT* imm.
def VSPLT_get_imm : SDNodeXForm<build_vector, [{
  return getI32Imm(PPC::getVSPLTImmediate(N));
}]>;

def VSPLT_shuffle_mask : PatLeaf<(build_vector), [{
  return PPC::isSplatShuffleMask(N);
}], VSPLT_get_imm>;

def vecimm0 : PatLeaf<(build_vector), [{
  return PPC::isZeroVector(N);
}]>;


// VSPLTISB_get_imm xform function: convert build_vector to VSPLTISB imm.
def VSPLTISB_get_imm : SDNodeXForm<build_vector, [{
  char Val;
  PPC::isVecSplatImm(N, 1, &Val);
  return getI32Imm(Val);
}]>;
def vecspltisb : PatLeaf<(build_vector), [{
  return PPC::isVecSplatImm(N, 1);
}], VSPLTISB_get_imm>;

// VSPLTISH_get_imm xform function: convert build_vector to VSPLTISH imm.
def VSPLTISH_get_imm : SDNodeXForm<build_vector, [{
  char Val;
  PPC::isVecSplatImm(N, 2, &Val);
  return getI32Imm(Val);
}]>;
def vecspltish : PatLeaf<(build_vector), [{
  return PPC::isVecSplatImm(N, 2);
}], VSPLTISH_get_imm>;

// VSPLTISW_get_imm xform function: convert build_vector to VSPLTISW imm.
def VSPLTISW_get_imm : SDNodeXForm<build_vector, [{
  char Val;
  PPC::isVecSplatImm(N, 4, &Val);
  return getI32Imm(Val);
}]>;
def vecspltisw : PatLeaf<(build_vector), [{
  return PPC::isVecSplatImm(N, 4);
}], VSPLTISW_get_imm>;



//===----------------------------------------------------------------------===//
// Instruction Definitions.

def IMPLICIT_DEF_VRRC : Pseudo<(ops VRRC:$rD), "; $rD = IMPLICIT_DEF_VRRC",
                               [(set VRRC:$rD, (v4f32 (undef)))]>;

let isLoad = 1, PPC970_Unit = 2 in {  // Loads.
def LVEBX: XForm_1<31,   7, (ops VRRC:$vD, memrr:$src),
                   "lvebx $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (v16i8 (PPClve_x xoaddr:$src)))]>;
def LVEHX: XForm_1<31,  39, (ops VRRC:$vD,  memrr:$src),
                   "lvehx $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (v8i16 (PPClve_x xoaddr:$src)))]>;
def LVEWX: XForm_1<31,  71, (ops VRRC:$vD,  memrr:$src),
                   "lvewx $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (v4f32 (PPClve_x xoaddr:$src)))]>;
def LVX  : XForm_1<31, 103, (ops VRRC:$vD,  memrr:$src),
                   "lvx $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (v4f32 (load xoaddr:$src)))]>;
}

def LVSL : XForm_1<31,   6, (ops VRRC:$vD,  GPRC:$base, GPRC:$rA),
                   "lvsl $vD, $base, $rA", LdStGeneral,
                   []>, PPC970_Unit_LSU;
def LVSR : XForm_1<31,  38, (ops VRRC:$vD,  GPRC:$base, GPRC:$rA),
                   "lvsl $vD, $base, $rA", LdStGeneral,
                   []>, PPC970_Unit_LSU;

let isStore = 1, noResults = 1, PPC970_Unit = 2 in {   // Stores.
def STVEBX: XForm_8<31, 135, (ops VRRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stvebx $rS, $rA, $rB", LdStGeneral,
                   []>;
def STVEHX: XForm_8<31, 167, (ops VRRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stvehx $rS, $rA, $rB", LdStGeneral,
                   []>;
def STVEWX: XForm_8<31, 199, (ops VRRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stvewx $rS, $rA, $rB", LdStGeneral,
                   []>;
def STVX  : XForm_8<31, 231, (ops VRRC:$rS, memrr:$dst),
                   "stvx $rS, $dst", LdStGeneral,
                   [(store (v4f32 VRRC:$rS), xoaddr:$dst)]>;
}

let PPC970_Unit = 5 in {  // VALU Operations.
// VA-Form instructions.  3-input AltiVec ops.
def VMADDFP : VAForm_1<46, (ops VRRC:$vD, VRRC:$vA, VRRC:$vC, VRRC:$vB),
                       "vmaddfp $vD, $vA, $vC, $vB", VecFP,
                       [(set VRRC:$vD, (fadd (fmul VRRC:$vA, VRRC:$vC),
                                             VRRC:$vB))]>,
                       Requires<[FPContractions]>;
def VNMSUBFP: VAForm_1<47, (ops VRRC:$vD, VRRC:$vA, VRRC:$vC, VRRC:$vB),
                       "vnmsubfp $vD, $vA, $vC, $vB", VecFP,
                       [(set VRRC:$vD, (fneg (fsub (fmul VRRC:$vA, VRRC:$vC),
                                                   VRRC:$vB)))]>,
                       Requires<[FPContractions]>;

def VPERM   : VAForm_1<43, (ops VRRC:$vD, VRRC:$vA, VRRC:$vC, VRRC:$vB),
                       "vperm $vD, $vA, $vB, $vC", VecPerm,
                       [(set VRRC:$vD,
                             (PPCvperm (v4f32 VRRC:$vA), VRRC:$vB, VRRC:$vC))]>;


// VX-Form instructions.  AltiVec arithmetic ops.
def VADDCUW : VXForm_1<384, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vaddcuw $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vaddcuw VRRC:$vA, VRRC:$vB))]>;
def VADDFP : VXForm_1<10, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vaddfp $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (fadd VRRC:$vA, VRRC:$vB))]>;
def VADDSBS : VXForm_1<768, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vaddsbs $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vaddsbs VRRC:$vA, VRRC:$vB))]>;
def VADDSHS : VXForm_1<832, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vaddshs $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vaddshs VRRC:$vA, VRRC:$vB))]>;
def VADDSWS : VXForm_1<896, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vaddsws $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vaddsws VRRC:$vA, VRRC:$vB))]>;
def VADDUBS : VXForm_1<512, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vaddubs $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vaddubs VRRC:$vA, VRRC:$vB))]>;
def VADDUHS : VXForm_1<576, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vadduhs $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vadduhs VRRC:$vA, VRRC:$vB))]>;
def VADDUWM : VXForm_1<128, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vadduwm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (add (v4i32 VRRC:$vA), VRRC:$vB))]>;
def VADDUWS : VXForm_1<640, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vadduws $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vadduws VRRC:$vA, VRRC:$vB))]>;
def VAND : VXForm_1<1028, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                    "vand $vD, $vA, $vB", VecFP,
                    [(set VRRC:$vD, (and (v4i32 VRRC:$vA), VRRC:$vB))]>;
def VANDC : VXForm_1<1092, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                     "vandc $vD, $vA, $vB", VecFP,
                     []>;

def VCFSX  : VXForm_1<842, (ops VRRC:$vD, u5imm:$UIMM, VRRC:$vB),
                      "vcfsx $vD, $vB, $UIMM", VecFP,
                      [(set VRRC:$vD,
                             (int_ppc_altivec_vcfsx VRRC:$vB, imm:$UIMM))]>;
def VCFUX  : VXForm_1<778, (ops VRRC:$vD, u5imm:$UIMM, VRRC:$vB),
                      "vcfux $vD, $vB, $UIMM", VecFP,
                      [(set VRRC:$vD,
                             (int_ppc_altivec_vcfux VRRC:$vB, imm:$UIMM))]>;
def VCTSXS : VXForm_1<970, (ops VRRC:$vD, u5imm:$UIMM, VRRC:$vB),
                      "vctsxs $vD, $vB, $UIMM", VecFP,
                      []>;
def VCTUXS : VXForm_1<906, (ops VRRC:$vD, u5imm:$UIMM, VRRC:$vB),
                      "vctuxs $vD, $vB, $UIMM", VecFP,
                      []>;
def VEXPTEFP : VXForm_2<394, (ops VRRC:$vD, VRRC:$vB),
                        "vexptefp $vD, $vB", VecFP,
                        []>;
def VLOGEFP  : VXForm_2<458, (ops VRRC:$vD, VRRC:$vB),
                        "vlogefp $vD, $vB", VecFP,
                        []>;
def VMAXFP : VXForm_1<1034, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vmaxfp $vD, $vA, $vB", VecFP,
                      []>;
def VMINFP : VXForm_1<1098, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vminfp $vD, $vA, $vB", VecFP,
                      []>;
def VREFP  : VXForm_2<266, (ops VRRC:$vD, VRRC:$vB),
                      "vrefp $vD, $vB", VecFP,
                      []>;
def VRFIM  : VXForm_2<714, (ops VRRC:$vD, VRRC:$vB),
                      "vrfim $vD, $vB", VecFP,
                      []>;
def VRFIN  : VXForm_2<522, (ops VRRC:$vD, VRRC:$vB),
                      "vrfin $vD, $vB", VecFP,
                      []>;
def VRFIP  : VXForm_2<650, (ops VRRC:$vD, VRRC:$vB),
                      "vrfip $vD, $vB", VecFP,
                      []>;
def VRFIZ  : VXForm_2<586, (ops VRRC:$vD, VRRC:$vB),
                      "vrfiz $vD, $vB", VecFP,
                      []>;
def VRSQRTEFP : VXForm_2<330, (ops VRRC:$vD, VRRC:$vB),
                         "vrsqrtefp $vD, $vB", VecFP,
                         [(set VRRC:$vD,(int_ppc_altivec_vrsqrtefp VRRC:$vB))]>;
def VSUBFP : VXForm_1<74, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vsubfp $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (fsub VRRC:$vA, VRRC:$vB))]>;
def VNOR : VXForm_1<1284, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                    "vnor $vD, $vA, $vB", VecFP,
                    []>;
def VOR : VXForm_1<1156, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vor $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (or (v4i32 VRRC:$vA), VRRC:$vB))]>;
def VXOR : VXForm_1<1220, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vxor $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (xor (v4i32 VRRC:$vA), VRRC:$vB))]>;

def VSPLTB : VXForm_1<524, (ops VRRC:$vD, u5imm:$UIMM, VRRC:$vB),
                      "vspltb $vD, $vB, $UIMM", VecPerm,
                      []>;
def VSPLTH : VXForm_1<588, (ops VRRC:$vD, u5imm:$UIMM, VRRC:$vB),
                      "vsplth $vD, $vB, $UIMM", VecPerm,
                      []>;
def VSPLTW : VXForm_1<652, (ops VRRC:$vD, u5imm:$UIMM, VRRC:$vB),
                      "vspltw $vD, $vB, $UIMM", VecPerm,
                      [(set VRRC:$vD, (vector_shuffle (v4f32 VRRC:$vB), (undef),
                                      VSPLT_shuffle_mask:$UIMM))]>;

def VSPLTISB : VXForm_1<780, (ops VRRC:$vD, s5imm:$SIMM),
                      "vspltisb $vD, $SIMM", VecPerm,
                      [(set VRRC:$vD, (v4f32 vecspltisb:$SIMM))]>;
def VSPLTISH : VXForm_1<844, (ops VRRC:$vD, s5imm:$SIMM),
                      "vspltish $vD, $SIMM", VecPerm,
                      [(set VRRC:$vD, (v4f32 vecspltish:$SIMM))]>;
def VSPLTISW : VXForm_1<908, (ops VRRC:$vD, s5imm:$SIMM),
                      "vspltisw $vD, $SIMM", VecPerm,
                      [(set VRRC:$vD, (v4f32 vecspltisw:$SIMM))]>;

                      
// VX-Form Pseudo Instructions

def V_SET0 : VXForm_setzero<1220, (ops VRRC:$vD),
                      "vxor $vD, $vD, $vD", VecFP,
                      [(set VRRC:$vD, (v4f32 vecimm0))]>;
}

//===----------------------------------------------------------------------===//
// Additional Altivec Patterns
//

// Undef/Zero.
def : Pat<(v16i8 (undef)), (v16i8 (IMPLICIT_DEF_VRRC))>;
def : Pat<(v8i16 (undef)), (v8i16 (IMPLICIT_DEF_VRRC))>;
def : Pat<(v4i32 (undef)), (v4i32 (IMPLICIT_DEF_VRRC))>;
def : Pat<(v16i8 vecimm0), (v16i8 (V_SET0))>;
def : Pat<(v8i16 vecimm0), (v8i16 (V_SET0))>;
def : Pat<(v4i32 vecimm0), (v4i32 (V_SET0))>;

// Loads.
def : Pat<(v16i8 (load xoaddr:$src)), (v16i8 (LVX xoaddr:$src))>;
def : Pat<(v8i16 (load xoaddr:$src)), (v8i16 (LVX xoaddr:$src))>;
def : Pat<(v4i32 (load xoaddr:$src)), (v4i32 (LVX xoaddr:$src))>;

// Stores.
def : Pat<(store (v16i8 VRRC:$rS), xoaddr:$dst),
          (STVX (v16i8 VRRC:$rS), xoaddr:$dst)>;
def : Pat<(store (v8i16 VRRC:$rS), xoaddr:$dst),
          (STVX (v8i16 VRRC:$rS), xoaddr:$dst)>;
def : Pat<(store (v4i32 VRRC:$rS), xoaddr:$dst),
          (STVX (v4i32 VRRC:$rS), xoaddr:$dst)>;

// Bit conversions.
def : Pat<(v16i8 (bitconvert (v8i16 VRRC:$src))), (v16i8 VRRC:$src)>;
def : Pat<(v16i8 (bitconvert (v4i32 VRRC:$src))), (v16i8 VRRC:$src)>;
def : Pat<(v16i8 (bitconvert (v4f32 VRRC:$src))), (v16i8 VRRC:$src)>;

def : Pat<(v8i16 (bitconvert (v16i8 VRRC:$src))), (v8i16 VRRC:$src)>;
def : Pat<(v8i16 (bitconvert (v4i32 VRRC:$src))), (v8i16 VRRC:$src)>;
def : Pat<(v8i16 (bitconvert (v4f32 VRRC:$src))), (v8i16 VRRC:$src)>;

def : Pat<(v4i32 (bitconvert (v16i8 VRRC:$src))), (v4i32 VRRC:$src)>;
def : Pat<(v4i32 (bitconvert (v8i16 VRRC:$src))), (v4i32 VRRC:$src)>;
def : Pat<(v4i32 (bitconvert (v4f32 VRRC:$src))), (v4i32 VRRC:$src)>;

def : Pat<(v4f32 (bitconvert (v16i8 VRRC:$src))), (v4f32 VRRC:$src)>;
def : Pat<(v4f32 (bitconvert (v8i16 VRRC:$src))), (v4f32 VRRC:$src)>;
def : Pat<(v4f32 (bitconvert (v4i32 VRRC:$src))), (v4f32 VRRC:$src)>;

// Immediate vector formation with vsplti*.
def : Pat<(v16i8 vecspltisb:$invec), (v16i8 (VSPLTISB vecspltisb:$invec))>;
def : Pat<(v16i8 vecspltish:$invec), (v16i8 (VSPLTISH vecspltish:$invec))>;
def : Pat<(v16i8 vecspltisw:$invec), (v16i8 (VSPLTISW vecspltisw:$invec))>;

def : Pat<(v8i16 vecspltisb:$invec), (v8i16 (VSPLTISB vecspltisb:$invec))>;
def : Pat<(v8i16 vecspltish:$invec), (v8i16 (VSPLTISH vecspltish:$invec))>;
def : Pat<(v8i16 vecspltisw:$invec), (v8i16 (VSPLTISW vecspltisw:$invec))>;

def : Pat<(v4i32 vecspltisb:$invec), (v4i32 (VSPLTISB vecspltisb:$invec))>;
def : Pat<(v4i32 vecspltish:$invec), (v4i32 (VSPLTISH vecspltish:$invec))>;
def : Pat<(v4i32 vecspltisw:$invec), (v4i32 (VSPLTISW vecspltisw:$invec))>;

// Logical Operations
def : Pat<(v16i8 (and VRRC:$A, VRRC:$B)), (v16i8 (VAND VRRC:$A, VRRC:$B))>;
def : Pat<(v8i16 (and VRRC:$A, VRRC:$B)), (v8i16 (VAND VRRC:$A, VRRC:$B))>;
def : Pat<(v16i8 (or  VRRC:$A, VRRC:$B)), (v16i8 (VOR  VRRC:$A, VRRC:$B))>;
def : Pat<(v8i16 (or  VRRC:$A, VRRC:$B)), (v8i16 (VOR  VRRC:$A, VRRC:$B))>;
def : Pat<(v16i8 (xor VRRC:$A, VRRC:$B)), (v16i8 (VXOR VRRC:$A, VRRC:$B))>;
def : Pat<(v8i16 (xor VRRC:$A, VRRC:$B)), (v8i16 (VXOR VRRC:$A, VRRC:$B))>;


def : Pat<(fmul VRRC:$vA, VRRC:$vB),
          (VMADDFP VRRC:$vA, VRRC:$vB, (V_SET0))>; 

// Fused multiply add and multiply sub for packed float.  These are represented
// separately from the real instructions above, for operations that must have
// the additional precision, such as Newton-Rhapson (used by divide, sqrt)
def : Pat<(PPCvmaddfp VRRC:$A, VRRC:$B, VRRC:$C),
          (VMADDFP VRRC:$A, VRRC:$B, VRRC:$C)>;
def : Pat<(PPCvnmsubfp VRRC:$A, VRRC:$B, VRRC:$C),
          (VNMSUBFP VRRC:$A, VRRC:$B, VRRC:$C)>;

def : Pat<(int_ppc_altivec_vmaddfp VRRC:$A, VRRC:$B, VRRC:$C),
          (VMADDFP VRRC:$A, VRRC:$B, VRRC:$C)>;
def : Pat<(int_ppc_altivec_vnmsubfp VRRC:$A, VRRC:$B, VRRC:$C),
          (VNMSUBFP VRRC:$A, VRRC:$B, VRRC:$C)>;

def : Pat<(vector_shuffle (v4i32 VRRC:$vB), (undef), VSPLT_shuffle_mask:$UIMM),
          (v4i32 (VSPLTW VSPLT_shuffle_mask:$UIMM, VRRC:$vB))>;

def : Pat<(PPCvperm (v4i32 VRRC:$vA), VRRC:$vB, VRRC:$vC),
          (v4i32 (VPERM VRRC:$vA, VRRC:$vB, VRRC:$vC))>;

def : Pat<(v4i32 (PPClve_x xoaddr:$src)),
          (v4i32 (LVEWX xoaddr:$src))>;


