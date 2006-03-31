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

class isVDOT {   // vector dot instruction.
  list<Register> Defs = [CR6];
  bit RC = 1;
}

//===----------------------------------------------------------------------===//
// Helpers for defining instructions that directly correspond to intrinsics.

// VA1a_Int - A VAForm_1a intrinsic definition.
class VA1a_Int<bits<6> xo, string opc, Intrinsic IntID>
  : VAForm_1a<xo, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB, VRRC:$vC),
              !strconcat(opc, " $vD, $vA, $vB, $vC"), VecFP,
                       [(set VRRC:$vD, (IntID VRRC:$vA, VRRC:$vB, VRRC:$vC))]>;

// VX1_Int - A VXForm_1 intrinsic definition.
class VX1_Int<bits<11> xo, string opc, Intrinsic IntID>
  : VXForm_1<xo, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
             !strconcat(opc, " $vD, $vA, $vB"), VecFP,
             [(set VRRC:$vD, (IntID VRRC:$vA, VRRC:$vB))]>;

// VX2_Int - A VXForm_2 intrinsic definition.
class VX2_Int<bits<11> xo, string opc, Intrinsic IntID>
  : VXForm_2<xo, (ops VRRC:$vD, VRRC:$vB),
             !strconcat(opc, " $vD, $vB"), VecFP,
             [(set VRRC:$vD, (IntID VRRC:$vB))]>;

//===----------------------------------------------------------------------===//
// Instruction Definitions.

def IMPLICIT_DEF_VRRC : Pseudo<(ops VRRC:$rD), "; $rD = IMPLICIT_DEF_VRRC",
                               [(set VRRC:$rD, (v4f32 (undef)))]>;

let isLoad = 1, PPC970_Unit = 2 in {  // Loads.
def LVEBX: XForm_1<31,   7, (ops VRRC:$vD, memrr:$src),
                   "lvebx $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (int_ppc_altivec_lvebx xoaddr:$src))]>;
def LVEHX: XForm_1<31,  39, (ops VRRC:$vD, memrr:$src),
                   "lvehx $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (int_ppc_altivec_lvehx xoaddr:$src))]>;
def LVEWX: XForm_1<31,  71, (ops VRRC:$vD, memrr:$src),
                   "lvewx $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (int_ppc_altivec_lvewx xoaddr:$src))]>;
def LVX  : XForm_1<31, 103, (ops VRRC:$vD, memrr:$src),
                   "lvx $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (int_ppc_altivec_lvx xoaddr:$src))]>;
def LVXL : XForm_1<31, 359, (ops VRRC:$vD, memrr:$src),
                   "lvxl $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (int_ppc_altivec_lvxl xoaddr:$src))]>;
}

def LVSL : XForm_1<31,   6, (ops VRRC:$vD, memrr:$src),
                   "lvsl $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (int_ppc_altivec_lvsl xoaddr:$src))]>,
                   PPC970_Unit_LSU;
def LVSR : XForm_1<31,  38, (ops VRRC:$vD, memrr:$src),
                   "lvsl $vD, $src", LdStGeneral,
                   [(set VRRC:$vD, (int_ppc_altivec_lvsr xoaddr:$src))]>,
                   PPC970_Unit_LSU;

let isStore = 1, noResults = 1, PPC970_Unit = 2 in {   // Stores.
def STVEBX: XForm_8<31, 135, (ops VRRC:$rS, memrr:$dst),
                   "stvebx $rS, $dst", LdStGeneral,
                   [(int_ppc_altivec_stvebx VRRC:$rS, xoaddr:$dst)]>;
def STVEHX: XForm_8<31, 167, (ops VRRC:$rS, memrr:$dst),
                   "stvehx $rS, $dst", LdStGeneral,
                   [(int_ppc_altivec_stvehx VRRC:$rS, xoaddr:$dst)]>;
def STVEWX: XForm_8<31, 199, (ops VRRC:$rS, memrr:$dst),
                   "stvewx $rS, $dst", LdStGeneral,
                   [(int_ppc_altivec_stvewx VRRC:$rS, xoaddr:$dst)]>;
def STVX  : XForm_8<31, 231, (ops VRRC:$rS, memrr:$dst),
                   "stvx $rS, $dst", LdStGeneral,
                   [(int_ppc_altivec_stvx VRRC:$rS, xoaddr:$dst)]>;
def STVXL : XForm_8<31, 487, (ops VRRC:$rS, memrr:$dst),
                   "stvxl $rS, $dst", LdStGeneral,
                   [(int_ppc_altivec_stvxl VRRC:$rS, xoaddr:$dst)]>;
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
def VMHADDSHS  : VA1a_Int<32, "vmhaddshs",  int_ppc_altivec_vmhaddshs>;
def VMHRADDSHS : VA1a_Int<33, "vmhraddshs", int_ppc_altivec_vmhraddshs>;
def VPERM      : VA1a_Int<43, "vperm",      int_ppc_altivec_vperm>;
def VSEL       : VA1a_Int<42, "vsel",       int_ppc_altivec_vsel>;

def VSLDOI  : VAForm_2<44, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB, u5imm:$SH),
                       "vsldoi $vD, $vA, $vB, $SH", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vsldoi VRRC:$vA, VRRC:$vB,
                                                     imm:$SH))]>;

// VX-Form instructions.  AltiVec arithmetic ops.
def VADDCUW : VXForm_1<384, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vaddcuw $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vaddcuw VRRC:$vA, VRRC:$vB))]>;
def VADDFP : VXForm_1<10, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vaddfp $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (fadd VRRC:$vA, VRRC:$vB))]>;
                      
def VADDUBM : VXForm_1<0, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vaddubm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (add (v16i8 VRRC:$vA), VRRC:$vB))]>;
def VADDUHM : VXForm_1<64, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vadduhm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (add (v8i16 VRRC:$vA), VRRC:$vB))]>;
def VADDUWM : VXForm_1<128, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vadduwm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (add (v4i32 VRRC:$vA), VRRC:$vB))]>;
                      
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
def VADDUWS : VXForm_1<640, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vadduws $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vadduws VRRC:$vA, VRRC:$vB))]>;
def VAND : VXForm_1<1028, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                    "vand $vD, $vA, $vB", VecFP,
                    [(set VRRC:$vD, (and (v4i32 VRRC:$vA), VRRC:$vB))]>;
def VANDC : VXForm_1<1092, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                     "vandc $vD, $vA, $vB", VecFP,
                     [(set VRRC:$vD, (and (v4i32 VRRC:$vA), (vnot VRRC:$vB)))]>;

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
                        [(set VRRC:$vD, (int_ppc_altivec_vexptefp VRRC:$vB))]>;
def VLOGEFP  : VXForm_2<458, (ops VRRC:$vD, VRRC:$vB),
                        "vlogefp $vD, $vB", VecFP,
                        [(set VRRC:$vD, (int_ppc_altivec_vlogefp VRRC:$vB))]>;
def VMAXFP : VXForm_1<1034, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vmaxfp $vD, $vA, $vB", VecFP,
                      []>;
def VMINFP : VXForm_1<1098, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vminfp $vD, $vA, $vB", VecFP,
                      []>;

def VMRGHH : VX1_Int<76 , "vmrghh", int_ppc_altivec_vmrghh>;
def VMRGHW : VX1_Int<140, "vmrghw", int_ppc_altivec_vmrghw>;
def VMRGLH : VX1_Int<332, "vmrglh", int_ppc_altivec_vmrglh>;
def VMRGLW : VX1_Int<396, "vmrglw", int_ppc_altivec_vmrglw>;

def VMSUMMBM : VA1a_Int<37, "vmsummbm", int_ppc_altivec_vmsummbm>;
def VMSUMSHM : VA1a_Int<40, "vmsumshm", int_ppc_altivec_vmsumshm>;
def VMSUMSHS : VA1a_Int<41, "vmsumshs", int_ppc_altivec_vmsumshs>;
def VMSUMUBM : VA1a_Int<36, "vmsumubm", int_ppc_altivec_vmsumubm>;
def VMSUMUHM : VA1a_Int<38, "vmsumuhm", int_ppc_altivec_vmsumuhm>;
def VMSUMUHS : VA1a_Int<39, "vmsumuhs", int_ppc_altivec_vmsumuhs>;

def VMULESB : VX1_Int<776, "vmulesb", int_ppc_altivec_vmulesb>;
def VMULESH : VX1_Int<840, "vmulesh", int_ppc_altivec_vmulesh>;
def VMULEUB : VX1_Int<520, "vmuleub", int_ppc_altivec_vmuleub>;
def VMULEUH : VX1_Int<584, "vmuleuh", int_ppc_altivec_vmuleuh>;
def VMULOSB : VX1_Int<264, "vmulosb", int_ppc_altivec_vmulosb>;
def VMULOSH : VX1_Int<328, "vmulosh", int_ppc_altivec_vmulosh>;
def VMULOUB : VX1_Int<  8, "vmuloub", int_ppc_altivec_vmuloub>;
def VMULOUH : VX1_Int< 72, "vmulouh", int_ppc_altivec_vmulouh>;
                       
def VREFP     : VX2_Int<266, "vrefp",     int_ppc_altivec_vrefp>;
def VRFIM     : VX2_Int<714, "vrfim",     int_ppc_altivec_vrfim>;
def VRFIN     : VX2_Int<522, "vrfin",     int_ppc_altivec_vrfin>;
def VRFIP     : VX2_Int<650, "vrfip",     int_ppc_altivec_vrfip>;
def VRFIZ     : VX2_Int<586, "vrfiz",     int_ppc_altivec_vrfiz>;
def VRSQRTEFP : VX2_Int<330, "vrsqrtefp", int_ppc_altivec_vrsqrtefp>;

def VSUBCUW : VX1_Int<74, "vsubcuw", int_ppc_altivec_vsubcuw>;

def VSUBFP  : VXForm_1<74, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vsubfp $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (fsub VRRC:$vA, VRRC:$vB))]>;
def VSUBUBM : VXForm_1<1024, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vsububm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (sub (v16i8 VRRC:$vA), VRRC:$vB))]>;
def VSUBUHM : VXForm_1<1088, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vsubuhm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (sub (v8i16 VRRC:$vA), VRRC:$vB))]>;
def VSUBUWM : VXForm_1<1152, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vsubuwm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (sub (v4i32 VRRC:$vA), VRRC:$vB))]>;
                      
def VSUBSBS : VX1_Int<1792, "vsubsbs" , int_ppc_altivec_vsubsbs>;
def VSUBSHS : VX1_Int<1856, "vsubshs" , int_ppc_altivec_vsubshs>;
def VSUBSWS : VX1_Int<1920, "vsubsws" , int_ppc_altivec_vsubsws>;
def VSUBUBS : VX1_Int<1536, "vsububs" , int_ppc_altivec_vsububs>;
def VSUBUHS : VX1_Int<1600, "vsubuhs" , int_ppc_altivec_vsubuhs>;
def VSUBUWS : VX1_Int<1664, "vsubuws" , int_ppc_altivec_vsubuws>;
def VSUMSWS : VX1_Int<1928, "vsumsws" , int_ppc_altivec_vsumsws>;
def VSUM2SWS: VX1_Int<1672, "vsum2sws", int_ppc_altivec_vsum2sws>;
def VSUM4SBS: VX1_Int<1672, "vsum4sbs", int_ppc_altivec_vsum4sbs>;
def VSUM4SHS: VX1_Int<1608, "vsum4shs", int_ppc_altivec_vsum4shs>;
def VSUM4UBS: VX1_Int<1544, "vsum4ubs", int_ppc_altivec_vsum4ubs>;

def VNOR : VXForm_1<1284, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                    "vnor $vD, $vA, $vB", VecFP,
                    [(set VRRC:$vD, (vnot (or (v4i32 VRRC:$vA), VRRC:$vB)))]>;
def VOR : VXForm_1<1156, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vor $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (or (v4i32 VRRC:$vA), VRRC:$vB))]>;
def VXOR : VXForm_1<1220, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                      "vxor $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (xor (v4i32 VRRC:$vA), VRRC:$vB))]>;

def VRLB   : VX1_Int<   4, "vrlb", int_ppc_altivec_vrlb>;
def VRLH   : VX1_Int<  68, "vrlh", int_ppc_altivec_vrlh>;
def VRLW   : VX1_Int< 132, "vrlw", int_ppc_altivec_vrlw>;
def VSLO   : VX1_Int<1036, "vslo", int_ppc_altivec_vslo>;
def VSLB   : VX1_Int< 260, "vslb", int_ppc_altivec_vslb>;
def VSLH   : VX1_Int< 324, "vslh", int_ppc_altivec_vslh>;
def VSLW   : VX1_Int< 388, "vslw", int_ppc_altivec_vslw>;

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

def VSR    : VX1_Int< 708, "vsr"  , int_ppc_altivec_vsr>;
def VSRO   : VX1_Int<1100, "vsro" , int_ppc_altivec_vsro>;
def VSRAB  : VX1_Int< 772, "vsrab", int_ppc_altivec_vsrab>;
def VSRAH  : VX1_Int< 836, "vsrah", int_ppc_altivec_vsrah>;
def VSRAW  : VX1_Int< 900, "vsraw", int_ppc_altivec_vsraw>;
def VSRB   : VX1_Int< 516, "vsrb" , int_ppc_altivec_vsrb>;
def VSRH   : VX1_Int< 580, "vsrh" , int_ppc_altivec_vsrh>;
def VSRW   : VX1_Int< 644, "vsrw" , int_ppc_altivec_vsrw>;


def VSPLTISB : VXForm_3<780, (ops VRRC:$vD, s5imm:$SIMM),
                       "vspltisb $vD, $SIMM", VecPerm,
                       [(set VRRC:$vD, (v4f32 vecspltisb:$SIMM))]>;
def VSPLTISH : VXForm_3<844, (ops VRRC:$vD, s5imm:$SIMM),
                       "vspltish $vD, $SIMM", VecPerm,
                       [(set VRRC:$vD, (v4f32 vecspltish:$SIMM))]>;
def VSPLTISW : VXForm_3<908, (ops VRRC:$vD, s5imm:$SIMM),
                       "vspltisw $vD, $SIMM", VecPerm,
                       [(set VRRC:$vD, (v4f32 vecspltisw:$SIMM))]>;

// Vector Pack.
def VPKPX   : VX1_Int<782, "vpkpx", int_ppc_altivec_vpkpx>;
def VPKSHSS : VX1_Int<398, "vpkshss", int_ppc_altivec_vpkshss>;
def VPKSHUS : VX1_Int<270, "vpkshus", int_ppc_altivec_vpkshus>;
def VPKSWSS : VX1_Int<462, "vpkswss", int_ppc_altivec_vpkswss>;
def VPKSWUS : VX1_Int<334, "vpkswus", int_ppc_altivec_vpkswus>;
def VPKUHUM : VXForm_1<14, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vpkuhum $vD, $vA, $vB", VecFP,
                       [/*TODO*/]>;
def VPKUHUS : VX1_Int<142, "vpkuhus", int_ppc_altivec_vpkuhus>;
def VPKUWUM : VXForm_1<78, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB),
                       "vpkuwum $vD, $vA, $vB", VecFP,
                       [/*TODO*/]>;
def VPKUWUS : VX1_Int<206, "vpkuwus", int_ppc_altivec_vpkuwus>;

// Vector Unpack.
def VUPKHPX : VXForm_2<846, (ops VRRC:$vD, VRRC:$vB),
                       "vupkhpx $vD, $vB", VecFP,
                       [(set VRRC:$vD, (int_ppc_altivec_vupkhpx VRRC:$vB))]>;
def VUPKHSB : VXForm_2<526, (ops VRRC:$vD, VRRC:$vB),
                       "vupkhsb $vD, $vB", VecFP,
                       [(set VRRC:$vD, (int_ppc_altivec_vupkhsb VRRC:$vB))]>;
def VUPKHSH : VXForm_2<590, (ops VRRC:$vD, VRRC:$vB),
                       "vupkhsh $vD, $vB", VecFP,
                       [(set VRRC:$vD, (int_ppc_altivec_vupkhsh VRRC:$vB))]>;
def VUPKLPX : VXForm_2<974, (ops VRRC:$vD, VRRC:$vB),
                       "vupklpx $vD, $vB", VecFP,
                       [(set VRRC:$vD, (int_ppc_altivec_vupklpx VRRC:$vB))]>;
def VUPKLSB : VXForm_2<654, (ops VRRC:$vD, VRRC:$vB),
                       "vupklsb $vD, $vB", VecFP,
                       [(set VRRC:$vD, (int_ppc_altivec_vupklsb VRRC:$vB))]>;
def VUPKLSH : VXForm_2<718, (ops VRRC:$vD, VRRC:$vB),
                       "vupklsh $vD, $vB", VecFP,
                       [(set VRRC:$vD, (int_ppc_altivec_vupklsh VRRC:$vB))]>;


// Altivec Comparisons.

class VCMP<bits<10> xo, string asmstr, ValueType Ty>
  : VXRForm_1<xo, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB), asmstr, VecFPCompare,
              [(set VRRC:$vD, (Ty (PPCvcmp VRRC:$vA, VRRC:$vB, xo)))]>;
class VCMPo<bits<10> xo, string asmstr, ValueType Ty>
  : VXRForm_1<xo, (ops VRRC:$vD, VRRC:$vA, VRRC:$vB), asmstr, VecFPCompare,
              [(set VRRC:$vD, (Ty (PPCvcmp_o VRRC:$vA, VRRC:$vB, xo)))]>,isVDOT;

// f32 element comparisons.0
def VCMPBFP   : VCMP <966, "vcmpbfp $vD, $vA, $vB"  , v4f32>;
def VCMPBFPo  : VCMPo<966, "vcmpbfp. $vD, $vA, $vB" , v4f32>;
def VCMPEQFP  : VCMP <198, "vcmpeqfp $vD, $vA, $vB" , v4f32>;
def VCMPEQFPo : VCMPo<198, "vcmpeqfp. $vD, $vA, $vB", v4f32>;
def VCMPGEFP  : VCMP <454, "vcmpgefp $vD, $vA, $vB" , v4f32>;
def VCMPGEFPo : VCMPo<454, "vcmpgefp. $vD, $vA, $vB", v4f32>;
def VCMPGTFP  : VCMP <710, "vcmpgtfp $vD, $vA, $vB" , v4f32>;
def VCMPGTFPo : VCMPo<710, "vcmpgtfp. $vD, $vA, $vB", v4f32>;

// i8 element comparisons.
def VCMPEQUB  : VCMP <  6, "vcmpequb $vD, $vA, $vB" , v16i8>;
def VCMPEQUBo : VCMPo<  6, "vcmpequb. $vD, $vA, $vB", v16i8>;
def VCMPGTSB  : VCMP <774, "vcmpgtsb $vD, $vA, $vB" , v16i8>;
def VCMPGTSBo : VCMPo<774, "vcmpgtsb. $vD, $vA, $vB", v16i8>;
def VCMPGTUB  : VCMP <518, "vcmpgtub $vD, $vA, $vB" , v16i8>;
def VCMPGTUBo : VCMPo<518, "vcmpgtub. $vD, $vA, $vB", v16i8>;

// i16 element comparisons.
def VCMPEQUH  : VCMP < 70, "vcmpequh $vD, $vA, $vB" , v8i16>;
def VCMPEQUHo : VCMPo< 70, "vcmpequh. $vD, $vA, $vB", v8i16>;
def VCMPGTSH  : VCMP <838, "vcmpgtsh $vD, $vA, $vB" , v8i16>;
def VCMPGTSHo : VCMPo<838, "vcmpgtsh. $vD, $vA, $vB", v8i16>;
def VCMPGTUH  : VCMP <582, "vcmpgtuh $vD, $vA, $vB" , v8i16>;
def VCMPGTUHo : VCMPo<582, "vcmpgtuh. $vD, $vA, $vB", v8i16>;

// i32 element comparisons.
def VCMPEQUW  : VCMP <134, "vcmpequw $vD, $vA, $vB" , v4i32>;
def VCMPEQUWo : VCMPo<134, "vcmpequw. $vD, $vA, $vB", v4i32>;
def VCMPGTSW  : VCMP <902, "vcmpgtsw $vD, $vA, $vB" , v4i32>;
def VCMPGTSWo : VCMPo<902, "vcmpgtsw. $vD, $vA, $vB", v4i32>;
def VCMPGTUW  : VCMP <646, "vcmpgtuw $vD, $vA, $vB" , v4i32>;
def VCMPGTUWo : VCMPo<646, "vcmpgtuw. $vD, $vA, $vB", v4i32>;
                      
def V_SET0 : VXForm_setzero<1220, (ops VRRC:$vD),
                      "vxor $vD, $vD, $vD", VecFP,
                      [(set VRRC:$vD, (v4f32 immAllZerosV))]>;
}

//===----------------------------------------------------------------------===//
// Additional Altivec Patterns
//

// Undef/Zero.
def : Pat<(v16i8 (undef)), (v16i8 (IMPLICIT_DEF_VRRC))>;
def : Pat<(v8i16 (undef)), (v8i16 (IMPLICIT_DEF_VRRC))>;
def : Pat<(v4i32 (undef)), (v4i32 (IMPLICIT_DEF_VRRC))>;
def : Pat<(v16i8 immAllZerosV), (v16i8 (V_SET0))>;
def : Pat<(v8i16 immAllZerosV), (v8i16 (V_SET0))>;
def : Pat<(v4i32 immAllZerosV), (v4i32 (V_SET0))>;

// Loads.
def : Pat<(v16i8 (load xoaddr:$src)), (v16i8 (LVX xoaddr:$src))>;
def : Pat<(v8i16 (load xoaddr:$src)), (v8i16 (LVX xoaddr:$src))>;
def : Pat<(v4i32 (load xoaddr:$src)), (v4i32 (LVX xoaddr:$src))>;
def : Pat<(v4f32 (load xoaddr:$src)), (v4f32 (LVX xoaddr:$src))>;

// Stores.
def : Pat<(store (v16i8 VRRC:$rS), xoaddr:$dst),
          (STVX (v16i8 VRRC:$rS), xoaddr:$dst)>;
def : Pat<(store (v8i16 VRRC:$rS), xoaddr:$dst),
          (STVX (v8i16 VRRC:$rS), xoaddr:$dst)>;
def : Pat<(store (v4i32 VRRC:$rS), xoaddr:$dst),
          (STVX (v4i32 VRRC:$rS), xoaddr:$dst)>;
def : Pat<(store (v4f32 VRRC:$rS), xoaddr:$dst),
          (STVX (v4f32 VRRC:$rS), xoaddr:$dst)>;

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
def : Pat<(v16i8 (vnot (or VRRC:$A, VRRC:$B))),(v16i8 (VNOR VRRC:$A, VRRC:$B))>;
def : Pat<(v8i16 (vnot (or VRRC:$A, VRRC:$B))),(v8i16 (VNOR VRRC:$A, VRRC:$B))>;
def : Pat<(v16i8 (and VRRC:$A, (vnot VRRC:$B))),
          (v16i8 (VANDC VRRC:$A, VRRC:$B))>;
def : Pat<(v8i16 (and VRRC:$A, (vnot VRRC:$B))),
          (v8i16 (VANDC VRRC:$A, VRRC:$B))>;

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
def : Pat<(PPCvperm (v4f32 VRRC:$vA), VRRC:$vB, VRRC:$vC),
          (v4f32 (VPERM VRRC:$vA, VRRC:$vB, VRRC:$vC))>;
def : Pat<(PPCvperm (v8i16 VRRC:$vA), VRRC:$vB, VRRC:$vC),
          (v8i16 (VPERM VRRC:$vA, VRRC:$vB, VRRC:$vC))>;
def : Pat<(PPCvperm (v16i8 VRRC:$vA), VRRC:$vB, VRRC:$vC),
          (v16i8 (VPERM VRRC:$vA, VRRC:$vB, VRRC:$vC))>;
