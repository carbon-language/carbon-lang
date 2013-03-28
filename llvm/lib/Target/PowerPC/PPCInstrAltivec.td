//===-- PPCInstrAltivec.td - The PowerPC Altivec Extension -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Altivec extension to the PowerPC instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Altivec transformation functions and pattern fragments.
//

// Since we canonicalize buildvectors to v16i8, all vnots "-1" operands will be
// of that type.
def vnot_ppc : PatFrag<(ops node:$in),
                       (xor node:$in, (bitconvert (v16i8 immAllOnesV)))>;

def vpkuhum_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                              (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVPKUHUMShuffleMask(cast<ShuffleVectorSDNode>(N), false);
}]>;
def vpkuwum_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                              (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVPKUWUMShuffleMask(cast<ShuffleVectorSDNode>(N), false);
}]>;
def vpkuhum_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                    (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVPKUHUMShuffleMask(cast<ShuffleVectorSDNode>(N), true);
}]>;
def vpkuwum_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                    (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVPKUWUMShuffleMask(cast<ShuffleVectorSDNode>(N), true);
}]>;


def vmrglb_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 1, false);
}]>;
def vmrglh_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 2, false);
}]>;
def vmrglw_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 4, false);
}]>;
def vmrghb_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 1, false);
}]>;
def vmrghh_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 2, false);
}]>;
def vmrghw_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 4, false);
}]>;


def vmrglb_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                               (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 1, true);
}]>;
def vmrglh_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 2, true);
}]>;
def vmrglw_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 4, true);
}]>;
def vmrghb_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 1, true);
}]>;
def vmrghh_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 2, true);
}]>;
def vmrghw_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 4, true);
}]>;


def VSLDOI_get_imm : SDNodeXForm<vector_shuffle, [{
  return getI32Imm(PPC::isVSLDOIShuffleMask(N, false));
}]>;
def vsldoi_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVSLDOIShuffleMask(N, false) != -1;
}], VSLDOI_get_imm>;


/// VSLDOI_unary* - These are used to match vsldoi(X,X), which is turned into
/// vector_shuffle(X,undef,mask) by the dag combiner.
def VSLDOI_unary_get_imm : SDNodeXForm<vector_shuffle, [{
  return getI32Imm(PPC::isVSLDOIShuffleMask(N, true));
}]>;
def vsldoi_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVSLDOIShuffleMask(N, true) != -1;
}], VSLDOI_unary_get_imm>;


// VSPLT*_get_imm xform function: convert vector_shuffle mask to VSPLT* imm.
def VSPLTB_get_imm : SDNodeXForm<vector_shuffle, [{
  return getI32Imm(PPC::getVSPLTImmediate(N, 1));
}]>;
def vspltb_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isSplatShuffleMask(cast<ShuffleVectorSDNode>(N), 1);
}], VSPLTB_get_imm>;
def VSPLTH_get_imm : SDNodeXForm<vector_shuffle, [{
  return getI32Imm(PPC::getVSPLTImmediate(N, 2));
}]>;
def vsplth_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isSplatShuffleMask(cast<ShuffleVectorSDNode>(N), 2);
}], VSPLTH_get_imm>;
def VSPLTW_get_imm : SDNodeXForm<vector_shuffle, [{
  return getI32Imm(PPC::getVSPLTImmediate(N, 4));
}]>;
def vspltw_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isSplatShuffleMask(cast<ShuffleVectorSDNode>(N), 4);
}], VSPLTW_get_imm>;


// VSPLTISB_get_imm xform function: convert build_vector to VSPLTISB imm.
def VSPLTISB_get_imm : SDNodeXForm<build_vector, [{
  return PPC::get_VSPLTI_elt(N, 1, *CurDAG);
}]>;
def vecspltisb : PatLeaf<(build_vector), [{
  return PPC::get_VSPLTI_elt(N, 1, *CurDAG).getNode() != 0;
}], VSPLTISB_get_imm>;

// VSPLTISH_get_imm xform function: convert build_vector to VSPLTISH imm.
def VSPLTISH_get_imm : SDNodeXForm<build_vector, [{
  return PPC::get_VSPLTI_elt(N, 2, *CurDAG);
}]>;
def vecspltish : PatLeaf<(build_vector), [{
  return PPC::get_VSPLTI_elt(N, 2, *CurDAG).getNode() != 0;
}], VSPLTISH_get_imm>;

// VSPLTISW_get_imm xform function: convert build_vector to VSPLTISW imm.
def VSPLTISW_get_imm : SDNodeXForm<build_vector, [{
  return PPC::get_VSPLTI_elt(N, 4, *CurDAG);
}]>;
def vecspltisw : PatLeaf<(build_vector), [{
  return PPC::get_VSPLTI_elt(N, 4, *CurDAG).getNode() != 0;
}], VSPLTISW_get_imm>;

//===----------------------------------------------------------------------===//
// Helpers for defining instructions that directly correspond to intrinsics.

// VA1a_Int - A VAForm_1a intrinsic definition of generic type.
class VA1a_Int<bits<6> xo, string opc, Intrinsic IntID>
  : VAForm_1a<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB, VRRC:$vC),
              !strconcat(opc, " $vD, $vA, $vB, $vC"), VecFP,
                       [(set VRRC:$vD, (IntID VRRC:$vA, VRRC:$vB, VRRC:$vC))]>;

// VA1a_Int_Ty - A VAForm_1a intrinsic definition of specific type.
class VA1a_Int_Ty<bits<6> xo, string opc, Intrinsic IntID, ValueType Ty>
  : VAForm_1a<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB, VRRC:$vC),
              !strconcat(opc, " $vD, $vA, $vB, $vC"), VecFP,
                       [(set Ty:$vD, (IntID Ty:$vA, Ty:$vB, Ty:$vC))]>;

// VA1a_Int_Ty2 - A VAForm_1a intrinsic definition where the type of the
// inputs doesn't match the type of the output.
class VA1a_Int_Ty2<bits<6> xo, string opc, Intrinsic IntID, ValueType OutTy,
                   ValueType InTy>
  : VAForm_1a<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB, VRRC:$vC),
              !strconcat(opc, " $vD, $vA, $vB, $vC"), VecFP,
                       [(set OutTy:$vD, (IntID InTy:$vA, InTy:$vB, InTy:$vC))]>;

// VA1a_Int_Ty3 - A VAForm_1a intrinsic definition where there are two
// input types and an output type.
class VA1a_Int_Ty3<bits<6> xo, string opc, Intrinsic IntID, ValueType OutTy,
                   ValueType In1Ty, ValueType In2Ty>
  : VAForm_1a<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB, VRRC:$vC),
              !strconcat(opc, " $vD, $vA, $vB, $vC"), VecFP,
                       [(set OutTy:$vD,
                         (IntID In1Ty:$vA, In1Ty:$vB, In2Ty:$vC))]>;

// VX1_Int - A VXForm_1 intrinsic definition of generic type.
class VX1_Int<bits<11> xo, string opc, Intrinsic IntID>
  : VXForm_1<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
             !strconcat(opc, " $vD, $vA, $vB"), VecFP,
             [(set VRRC:$vD, (IntID VRRC:$vA, VRRC:$vB))]>;

// VX1_Int_Ty - A VXForm_1 intrinsic definition of specific type.
class VX1_Int_Ty<bits<11> xo, string opc, Intrinsic IntID, ValueType Ty>
  : VXForm_1<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
             !strconcat(opc, " $vD, $vA, $vB"), VecFP,
             [(set Ty:$vD, (IntID Ty:$vA, Ty:$vB))]>;

// VX1_Int_Ty2 - A VXForm_1 intrinsic definition where the type of the
// inputs doesn't match the type of the output.
class VX1_Int_Ty2<bits<11> xo, string opc, Intrinsic IntID, ValueType OutTy,
                  ValueType InTy>
  : VXForm_1<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
             !strconcat(opc, " $vD, $vA, $vB"), VecFP,
             [(set OutTy:$vD, (IntID InTy:$vA, InTy:$vB))]>;

// VX1_Int_Ty3 - A VXForm_1 intrinsic definition where there are two
// input types and an output type.
class VX1_Int_Ty3<bits<11> xo, string opc, Intrinsic IntID, ValueType OutTy,
                  ValueType In1Ty, ValueType In2Ty>
  : VXForm_1<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
             !strconcat(opc, " $vD, $vA, $vB"), VecFP,
             [(set OutTy:$vD, (IntID In1Ty:$vA, In2Ty:$vB))]>;

// VX2_Int - A VXForm_1 intrinsic definition of generic type.
class VX2_Int<bits<11> xo, string opc, Intrinsic IntID>
  : VXForm_2<xo, (outs VRRC:$vD), (ins VRRC:$vB),
             !strconcat(opc, " $vD, $vB"), VecFP,
             [(set VRRC:$vD, (IntID VRRC:$vB))]>;

// VX2_Int_SP - A VXForm_2 intrinsic definition of vector single-precision type.
class VX2_Int_SP<bits<11> xo, string opc, Intrinsic IntID>
  : VXForm_2<xo, (outs VRRC:$vD), (ins VRRC:$vB),
             !strconcat(opc, " $vD, $vB"), VecFP,
             [(set v4f32:$vD, (IntID v4f32:$vB))]>;

// VX2_Int_Ty2 - A VXForm_2 intrinsic definition where the type of the
// inputs doesn't match the type of the output.
class VX2_Int_Ty2<bits<11> xo, string opc, Intrinsic IntID, ValueType OutTy,
                  ValueType InTy>
  : VXForm_2<xo, (outs VRRC:$vD), (ins VRRC:$vB),
             !strconcat(opc, " $vD, $vB"), VecFP,
             [(set OutTy:$vD, (IntID InTy:$vB))]>;

//===----------------------------------------------------------------------===//
// Instruction Definitions.

def HasAltivec : Predicate<"PPCSubTarget.hasAltivec()">;
let Predicates = [HasAltivec] in {

let isCodeGenOnly = 1 in {
def DSS      : DSS_Form<822, (outs),
                        (ins u5imm:$ZERO0, u5imm:$STRM,u5imm:$ZERO1,u5imm:$ZERO2),
                        "dss $STRM", LdStLoad /*FIXME*/, []>;
def DSSALL   : DSS_Form<822, (outs),
                        (ins u5imm:$ONE, u5imm:$ZERO0,u5imm:$ZERO1,u5imm:$ZERO2),
                        "dssall", LdStLoad /*FIXME*/, []>;
def DST      : DSS_Form<342, (outs),
                        (ins u5imm:$ZERO, u5imm:$STRM, GPRC:$rA, GPRC:$rB),
                        "dst $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTT     : DSS_Form<342, (outs),
                        (ins u5imm:$ONE, u5imm:$STRM, GPRC:$rA, GPRC:$rB),
                        "dstt $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTST    : DSS_Form<374, (outs),
                        (ins u5imm:$ZERO, u5imm:$STRM, GPRC:$rA, GPRC:$rB),
                        "dstst $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTSTT   : DSS_Form<374, (outs),
                        (ins u5imm:$ONE, u5imm:$STRM, GPRC:$rA, GPRC:$rB),
                        "dststt $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;

def DST64    : DSS_Form<342, (outs),
                        (ins u5imm:$ZERO, u5imm:$STRM, G8RC:$rA, GPRC:$rB),
                        "dst $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTT64   : DSS_Form<342, (outs),
                        (ins u5imm:$ONE, u5imm:$STRM, G8RC:$rA, GPRC:$rB),
                        "dstt $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTST64  : DSS_Form<374, (outs),
                        (ins u5imm:$ZERO, u5imm:$STRM, G8RC:$rA, GPRC:$rB),
                        "dstst $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTSTT64 : DSS_Form<374, (outs),
                        (ins u5imm:$ONE, u5imm:$STRM, G8RC:$rA, GPRC:$rB),
                        "dststt $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
}

def MFVSCR : VXForm_4<1540, (outs VRRC:$vD), (ins),
                      "mfvscr $vD", LdStStore,
                      [(set v8i16:$vD, (int_ppc_altivec_mfvscr))]>; 
def MTVSCR : VXForm_5<1604, (outs), (ins VRRC:$vB),
                      "mtvscr $vB", LdStLoad,
                      [(int_ppc_altivec_mtvscr v4i32:$vB)]>; 

let canFoldAsLoad = 1, PPC970_Unit = 2 in {  // Loads.
def LVEBX: XForm_1<31,   7, (outs VRRC:$vD), (ins memrr:$src),
                   "lvebx $vD, $src", LdStLoad,
                   [(set v16i8:$vD, (int_ppc_altivec_lvebx xoaddr:$src))]>;
def LVEHX: XForm_1<31,  39, (outs VRRC:$vD), (ins memrr:$src),
                   "lvehx $vD, $src", LdStLoad,
                   [(set v8i16:$vD, (int_ppc_altivec_lvehx xoaddr:$src))]>;
def LVEWX: XForm_1<31,  71, (outs VRRC:$vD), (ins memrr:$src),
                   "lvewx $vD, $src", LdStLoad,
                   [(set v4i32:$vD, (int_ppc_altivec_lvewx xoaddr:$src))]>;
def LVX  : XForm_1<31, 103, (outs VRRC:$vD), (ins memrr:$src),
                   "lvx $vD, $src", LdStLoad,
                   [(set v4i32:$vD, (int_ppc_altivec_lvx xoaddr:$src))]>;
def LVXL : XForm_1<31, 359, (outs VRRC:$vD), (ins memrr:$src),
                   "lvxl $vD, $src", LdStLoad,
                   [(set v4i32:$vD, (int_ppc_altivec_lvxl xoaddr:$src))]>;
}

def LVSL : XForm_1<31,   6, (outs VRRC:$vD), (ins memrr:$src),
                   "lvsl $vD, $src", LdStLoad,
                   [(set v16i8:$vD, (int_ppc_altivec_lvsl xoaddr:$src))]>,
                   PPC970_Unit_LSU;
def LVSR : XForm_1<31,  38, (outs VRRC:$vD), (ins memrr:$src),
                   "lvsr $vD, $src", LdStLoad,
                   [(set v16i8:$vD, (int_ppc_altivec_lvsr xoaddr:$src))]>,
                   PPC970_Unit_LSU;

let PPC970_Unit = 2 in {   // Stores.
def STVEBX: XForm_8<31, 135, (outs), (ins VRRC:$rS, memrr:$dst),
                   "stvebx $rS, $dst", LdStStore,
                   [(int_ppc_altivec_stvebx v16i8:$rS, xoaddr:$dst)]>;
def STVEHX: XForm_8<31, 167, (outs), (ins VRRC:$rS, memrr:$dst),
                   "stvehx $rS, $dst", LdStStore,
                   [(int_ppc_altivec_stvehx v8i16:$rS, xoaddr:$dst)]>;
def STVEWX: XForm_8<31, 199, (outs), (ins VRRC:$rS, memrr:$dst),
                   "stvewx $rS, $dst", LdStStore,
                   [(int_ppc_altivec_stvewx v4i32:$rS, xoaddr:$dst)]>;
def STVX  : XForm_8<31, 231, (outs), (ins VRRC:$rS, memrr:$dst),
                   "stvx $rS, $dst", LdStStore,
                   [(int_ppc_altivec_stvx v4i32:$rS, xoaddr:$dst)]>;
def STVXL : XForm_8<31, 487, (outs), (ins VRRC:$rS, memrr:$dst),
                   "stvxl $rS, $dst", LdStStore,
                   [(int_ppc_altivec_stvxl v4i32:$rS, xoaddr:$dst)]>;
}

let PPC970_Unit = 5 in {  // VALU Operations.
// VA-Form instructions.  3-input AltiVec ops.
def VMADDFP : VAForm_1<46, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vC, VRRC:$vB),
                       "vmaddfp $vD, $vA, $vC, $vB", VecFP,
                       [(set v4f32:$vD,
                        (fma v4f32:$vA, v4f32:$vC, v4f32:$vB))]>;
def VNMSUBFP: VAForm_1<47, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vC, VRRC:$vB),
                       "vnmsubfp $vD, $vA, $vC, $vB", VecFP,
                       [(set v4f32:$vD, (fneg (fma v4f32:$vA, v4f32:$vC,
                                                  (fneg v4f32:$vB))))]>; 

def VMHADDSHS  : VA1a_Int_Ty<32, "vmhaddshs", int_ppc_altivec_vmhaddshs, v8i16>;
def VMHRADDSHS : VA1a_Int_Ty<33, "vmhraddshs", int_ppc_altivec_vmhraddshs,
                             v8i16>;
def VMLADDUHM  : VA1a_Int_Ty<34, "vmladduhm", int_ppc_altivec_vmladduhm, v8i16>;

def VPERM      : VA1a_Int_Ty3<43, "vperm", int_ppc_altivec_vperm,
                              v4i32, v4i32, v16i8>;
def VSEL       : VA1a_Int_Ty<42, "vsel",  int_ppc_altivec_vsel, v4i32>;

// Shuffles.
def VSLDOI  : VAForm_2<44, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB, u5imm:$SH),
                       "vsldoi $vD, $vA, $vB, $SH", VecFP,
                       [(set VRRC:$vD, 
                         (vsldoi_shuffle:$SH (v16i8 VRRC:$vA), VRRC:$vB))]>;

// VX-Form instructions.  AltiVec arithmetic ops.
def VADDFP : VXForm_1<10, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vaddfp $vD, $vA, $vB", VecFP,
                      [(set v4f32:$vD, (fadd v4f32:$vA, v4f32:$vB))]>;
                      
def VADDUBM : VXForm_1<0, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vaddubm $vD, $vA, $vB", VecGeneral,
                      [(set v16i8:$vD, (add v16i8:$vA, v16i8:$vB))]>;
def VADDUHM : VXForm_1<64, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vadduhm $vD, $vA, $vB", VecGeneral,
                      [(set v8i16:$vD, (add v8i16:$vA, v8i16:$vB))]>;
def VADDUWM : VXForm_1<128, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vadduwm $vD, $vA, $vB", VecGeneral,
                      [(set v4i32:$vD, (add v4i32:$vA, v4i32:$vB))]>;
                      
def VADDCUW : VX1_Int_Ty<384, "vaddcuw", int_ppc_altivec_vaddcuw, v4i32>;
def VADDSBS : VX1_Int_Ty<768, "vaddsbs", int_ppc_altivec_vaddsbs, v16i8>;
def VADDSHS : VX1_Int_Ty<832, "vaddshs", int_ppc_altivec_vaddshs, v8i16>;
def VADDSWS : VX1_Int_Ty<896, "vaddsws", int_ppc_altivec_vaddsws, v4i32>;
def VADDUBS : VX1_Int_Ty<512, "vaddubs", int_ppc_altivec_vaddubs, v16i8>;
def VADDUHS : VX1_Int_Ty<576, "vadduhs", int_ppc_altivec_vadduhs, v8i16>;
def VADDUWS : VX1_Int_Ty<640, "vadduws", int_ppc_altivec_vadduws, v4i32>;
                             
                             
def VAND : VXForm_1<1028, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                    "vand $vD, $vA, $vB", VecFP,
                    [(set VRRC:$vD, (and (v4i32 VRRC:$vA), VRRC:$vB))]>;
def VANDC : VXForm_1<1092, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                     "vandc $vD, $vA, $vB", VecFP,
                     [(set VRRC:$vD, (and (v4i32 VRRC:$vA),
                                          (vnot_ppc VRRC:$vB)))]>;

def VCFSX  : VXForm_1<842, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vcfsx $vD, $vB, $UIMM", VecFP,
                      [(set v4f32:$vD,
                             (int_ppc_altivec_vcfsx v4i32:$vB, imm:$UIMM))]>;
def VCFUX  : VXForm_1<778, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vcfux $vD, $vB, $UIMM", VecFP,
                      [(set v4f32:$vD,
                             (int_ppc_altivec_vcfux v4i32:$vB, imm:$UIMM))]>;
def VCTSXS : VXForm_1<970, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vctsxs $vD, $vB, $UIMM", VecFP,
                      [(set v4i32:$vD,
                             (int_ppc_altivec_vctsxs v4f32:$vB, imm:$UIMM))]>;
def VCTUXS : VXForm_1<906, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vctuxs $vD, $vB, $UIMM", VecFP,
                      [(set v4i32:$vD,
                             (int_ppc_altivec_vctuxs v4f32:$vB, imm:$UIMM))]>;

// Defines with the UIM field set to 0 for floating-point
// to integer (fp_to_sint/fp_to_uint) conversions and integer
// to floating-point (sint_to_fp/uint_to_fp) conversions.
let VA = 0 in {
def VCFSX_0 : VXForm_1<842, (outs VRRC:$vD), (ins VRRC:$vB),
                       "vcfsx $vD, $vB, 0", VecFP,
                       [(set v4f32:$vD,
                             (int_ppc_altivec_vcfsx v4i32:$vB, 0))]>;
def VCTUXS_0 : VXForm_1<906, (outs VRRC:$vD), (ins VRRC:$vB),
                        "vctuxs $vD, $vB, 0", VecFP,
                        [(set v4i32:$vD,
                               (int_ppc_altivec_vctuxs v4f32:$vB, 0))]>;
def VCFUX_0 : VXForm_1<778, (outs VRRC:$vD), (ins VRRC:$vB),
                       "vcfux $vD, $vB, 0", VecFP,
                       [(set v4f32:$vD,
                               (int_ppc_altivec_vcfux v4i32:$vB, 0))]>;
def VCTSXS_0 : VXForm_1<970, (outs VRRC:$vD), (ins VRRC:$vB),
                      "vctsxs $vD, $vB, 0", VecFP,
                      [(set v4i32:$vD,
                             (int_ppc_altivec_vctsxs v4f32:$vB, 0))]>;
}
def VEXPTEFP : VX2_Int_SP<394, "vexptefp", int_ppc_altivec_vexptefp>;
def VLOGEFP  : VX2_Int_SP<458, "vlogefp",  int_ppc_altivec_vlogefp>;

def VAVGSB : VX1_Int_Ty<1282, "vavgsb", int_ppc_altivec_vavgsb, v16i8>;
def VAVGSH : VX1_Int_Ty<1346, "vavgsh", int_ppc_altivec_vavgsh, v8i16>;
def VAVGSW : VX1_Int_Ty<1410, "vavgsw", int_ppc_altivec_vavgsw, v4i32>;
def VAVGUB : VX1_Int_Ty<1026, "vavgub", int_ppc_altivec_vavgub, v16i8>;
def VAVGUH : VX1_Int_Ty<1090, "vavguh", int_ppc_altivec_vavguh, v8i16>;
def VAVGUW : VX1_Int_Ty<1154, "vavguw", int_ppc_altivec_vavguw, v4i32>;

def VMAXFP : VX1_Int_Ty<1034, "vmaxfp", int_ppc_altivec_vmaxfp, v4f32>;
def VMAXSB : VX1_Int_Ty< 258, "vmaxsb", int_ppc_altivec_vmaxsb, v16i8>;
def VMAXSH : VX1_Int_Ty< 322, "vmaxsh", int_ppc_altivec_vmaxsh, v8i16>;
def VMAXSW : VX1_Int_Ty< 386, "vmaxsw", int_ppc_altivec_vmaxsw, v4i32>;
def VMAXUB : VX1_Int_Ty<   2, "vmaxub", int_ppc_altivec_vmaxub, v16i8>;
def VMAXUH : VX1_Int_Ty<  66, "vmaxuh", int_ppc_altivec_vmaxuh, v8i16>;
def VMAXUW : VX1_Int_Ty< 130, "vmaxuw", int_ppc_altivec_vmaxuw, v4i32>;
def VMINFP : VX1_Int_Ty<1098, "vminfp", int_ppc_altivec_vminfp, v4f32>;
def VMINSB : VX1_Int_Ty< 770, "vminsb", int_ppc_altivec_vminsb, v16i8>;
def VMINSH : VX1_Int_Ty< 834, "vminsh", int_ppc_altivec_vminsh, v8i16>;
def VMINSW : VX1_Int_Ty< 898, "vminsw", int_ppc_altivec_vminsw, v4i32>;
def VMINUB : VX1_Int_Ty< 514, "vminub", int_ppc_altivec_vminub, v16i8>;
def VMINUH : VX1_Int_Ty< 578, "vminuh", int_ppc_altivec_vminuh, v8i16>;
def VMINUW : VX1_Int_Ty< 642, "vminuw", int_ppc_altivec_vminuw, v4i32>;

def VMRGHB : VXForm_1< 12, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrghb $vD, $vA, $vB", VecFP,
                      [(set v16i8:$vD, (vmrghb_shuffle v16i8:$vA, v16i8:$vB))]>;
def VMRGHH : VXForm_1< 76, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrghh $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (vmrghh_shuffle VRRC:$vA, VRRC:$vB))]>;
def VMRGHW : VXForm_1<140, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrghw $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (vmrghw_shuffle VRRC:$vA, VRRC:$vB))]>;
def VMRGLB : VXForm_1<268, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrglb $vD, $vA, $vB", VecFP,
                      [(set v16i8:$vD, (vmrglb_shuffle v16i8:$vA, v16i8:$vB))]>;
def VMRGLH : VXForm_1<332, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrglh $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (vmrglh_shuffle VRRC:$vA, VRRC:$vB))]>;
def VMRGLW : VXForm_1<396, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrglw $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (vmrglw_shuffle VRRC:$vA, VRRC:$vB))]>;

def VMSUMMBM : VA1a_Int_Ty3<37, "vmsummbm", int_ppc_altivec_vmsummbm,
                            v4i32, v16i8, v4i32>;
def VMSUMSHM : VA1a_Int_Ty3<40, "vmsumshm", int_ppc_altivec_vmsumshm,
                            v4i32, v8i16, v4i32>;
def VMSUMSHS : VA1a_Int_Ty3<41, "vmsumshs", int_ppc_altivec_vmsumshs,
                            v4i32, v8i16, v4i32>;
def VMSUMUBM : VA1a_Int_Ty3<36, "vmsumubm", int_ppc_altivec_vmsumubm,
                            v4i32, v16i8, v4i32>;
def VMSUMUHM : VA1a_Int_Ty3<38, "vmsumuhm", int_ppc_altivec_vmsumuhm,
                            v4i32, v8i16, v4i32>;
def VMSUMUHS : VA1a_Int_Ty3<39, "vmsumuhs", int_ppc_altivec_vmsumuhs,
                            v4i32, v8i16, v4i32>;

def VMULESB : VX1_Int_Ty2<776, "vmulesb", int_ppc_altivec_vmulesb,
                          v8i16, v16i8>;
def VMULESH : VX1_Int_Ty2<840, "vmulesh", int_ppc_altivec_vmulesh,
                          v4i32, v8i16>;
def VMULEUB : VX1_Int_Ty2<520, "vmuleub", int_ppc_altivec_vmuleub,
                          v8i16, v16i8>;
def VMULEUH : VX1_Int_Ty2<584, "vmuleuh", int_ppc_altivec_vmuleuh,
                          v4i32, v8i16>;
def VMULOSB : VX1_Int_Ty2<264, "vmulosb", int_ppc_altivec_vmulosb,
                          v8i16, v16i8>;
def VMULOSH : VX1_Int_Ty2<328, "vmulosh", int_ppc_altivec_vmulosh,
                          v4i32, v8i16>;
def VMULOUB : VX1_Int_Ty2<  8, "vmuloub", int_ppc_altivec_vmuloub,
                          v8i16, v16i8>;
def VMULOUH : VX1_Int_Ty2< 72, "vmulouh", int_ppc_altivec_vmulouh,
                          v4i32, v8i16>;
                       
def VREFP     : VX2_Int_SP<266, "vrefp",     int_ppc_altivec_vrefp>;
def VRFIM     : VX2_Int_SP<714, "vrfim",     int_ppc_altivec_vrfim>;
def VRFIN     : VX2_Int_SP<522, "vrfin",     int_ppc_altivec_vrfin>;
def VRFIP     : VX2_Int_SP<650, "vrfip",     int_ppc_altivec_vrfip>;
def VRFIZ     : VX2_Int_SP<586, "vrfiz",     int_ppc_altivec_vrfiz>;
def VRSQRTEFP : VX2_Int_SP<330, "vrsqrtefp", int_ppc_altivec_vrsqrtefp>;

def VSUBCUW : VX1_Int_Ty<74, "vsubcuw", int_ppc_altivec_vsubcuw, v4i32>;

def VSUBFP  : VXForm_1<74, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vsubfp $vD, $vA, $vB", VecGeneral,
                      [(set v4f32:$vD, (fsub v4f32:$vA, v4f32:$vB))]>;
def VSUBUBM : VXForm_1<1024, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vsububm $vD, $vA, $vB", VecGeneral,
                      [(set v16i8:$vD, (sub v16i8:$vA, v16i8:$vB))]>;
def VSUBUHM : VXForm_1<1088, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vsubuhm $vD, $vA, $vB", VecGeneral,
                      [(set v8i16:$vD, (sub v8i16:$vA, v8i16:$vB))]>;
def VSUBUWM : VXForm_1<1152, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vsubuwm $vD, $vA, $vB", VecGeneral,
                      [(set v4i32:$vD, (sub v4i32:$vA, v4i32:$vB))]>;
                      
def VSUBSBS : VX1_Int_Ty<1792, "vsubsbs" , int_ppc_altivec_vsubsbs, v16i8>;
def VSUBSHS : VX1_Int_Ty<1856, "vsubshs" , int_ppc_altivec_vsubshs, v8i16>;
def VSUBSWS : VX1_Int_Ty<1920, "vsubsws" , int_ppc_altivec_vsubsws, v4i32>;
def VSUBUBS : VX1_Int_Ty<1536, "vsububs" , int_ppc_altivec_vsububs, v16i8>;
def VSUBUHS : VX1_Int_Ty<1600, "vsubuhs" , int_ppc_altivec_vsubuhs, v8i16>;
def VSUBUWS : VX1_Int_Ty<1664, "vsubuws" , int_ppc_altivec_vsubuws, v4i32>;

def VSUMSWS : VX1_Int_Ty<1928, "vsumsws" , int_ppc_altivec_vsumsws, v4i32>;
def VSUM2SWS: VX1_Int_Ty<1672, "vsum2sws", int_ppc_altivec_vsum2sws, v4i32>;

def VSUM4SBS: VX1_Int_Ty3<1672, "vsum4sbs", int_ppc_altivec_vsum4sbs,
                          v4i32, v16i8, v4i32>;
def VSUM4SHS: VX1_Int_Ty3<1608, "vsum4shs", int_ppc_altivec_vsum4shs,
                          v4i32, v8i16, v4i32>;
def VSUM4UBS: VX1_Int_Ty3<1544, "vsum4ubs", int_ppc_altivec_vsum4ubs,
                          v4i32, v16i8, v4i32>;

def VNOR : VXForm_1<1284, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                    "vnor $vD, $vA, $vB", VecFP,
                    [(set VRRC:$vD, (vnot_ppc (or (v4i32 VRRC:$vA),
                                                  VRRC:$vB)))]>;
def VOR : VXForm_1<1156, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vor $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (or (v4i32 VRRC:$vA), VRRC:$vB))]>;
def VXOR : VXForm_1<1220, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vxor $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (xor (v4i32 VRRC:$vA), VRRC:$vB))]>;

def VRLB   : VX1_Int_Ty<   4, "vrlb", int_ppc_altivec_vrlb, v16i8>;
def VRLH   : VX1_Int_Ty<  68, "vrlh", int_ppc_altivec_vrlh, v8i16>;
def VRLW   : VX1_Int_Ty< 132, "vrlw", int_ppc_altivec_vrlw, v4i32>;

def VSL    : VX1_Int_Ty< 452, "vsl" , int_ppc_altivec_vsl,  v4i32 >;
def VSLO   : VX1_Int_Ty<1036, "vslo", int_ppc_altivec_vslo, v4i32>;

def VSLB   : VX1_Int_Ty< 260, "vslb", int_ppc_altivec_vslb, v16i8>;
def VSLH   : VX1_Int_Ty< 324, "vslh", int_ppc_altivec_vslh, v8i16>;
def VSLW   : VX1_Int_Ty< 388, "vslw", int_ppc_altivec_vslw, v4i32>;

def VSPLTB : VXForm_1<524, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vspltb $vD, $vB, $UIMM", VecPerm,
                      [(set VRRC:$vD,
                        (vspltb_shuffle:$UIMM (v16i8 VRRC:$vB), (undef)))]>;
def VSPLTH : VXForm_1<588, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vsplth $vD, $vB, $UIMM", VecPerm,
                      [(set VRRC:$vD,
                        (vsplth_shuffle:$UIMM (v16i8 VRRC:$vB), (undef)))]>;
def VSPLTW : VXForm_1<652, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vspltw $vD, $vB, $UIMM", VecPerm,
                      [(set VRRC:$vD, 
                        (vspltw_shuffle:$UIMM (v16i8 VRRC:$vB), (undef)))]>;

def VSR    : VX1_Int_Ty< 708, "vsr"  , int_ppc_altivec_vsr,  v4i32>;
def VSRO   : VX1_Int_Ty<1100, "vsro" , int_ppc_altivec_vsro, v4i32>;

def VSRAB  : VX1_Int_Ty< 772, "vsrab", int_ppc_altivec_vsrab, v16i8>;
def VSRAH  : VX1_Int_Ty< 836, "vsrah", int_ppc_altivec_vsrah, v8i16>;
def VSRAW  : VX1_Int_Ty< 900, "vsraw", int_ppc_altivec_vsraw, v4i32>;
def VSRB   : VX1_Int_Ty< 516, "vsrb" , int_ppc_altivec_vsrb , v16i8>;
def VSRH   : VX1_Int_Ty< 580, "vsrh" , int_ppc_altivec_vsrh , v8i16>;
def VSRW   : VX1_Int_Ty< 644, "vsrw" , int_ppc_altivec_vsrw , v4i32>;


def VSPLTISB : VXForm_3<780, (outs VRRC:$vD), (ins s5imm:$SIMM),
                       "vspltisb $vD, $SIMM", VecPerm,
                       [(set v16i8:$vD, (v16i8 vecspltisb:$SIMM))]>;
def VSPLTISH : VXForm_3<844, (outs VRRC:$vD), (ins s5imm:$SIMM),
                       "vspltish $vD, $SIMM", VecPerm,
                       [(set v8i16:$vD, (v8i16 vecspltish:$SIMM))]>;
def VSPLTISW : VXForm_3<908, (outs VRRC:$vD), (ins s5imm:$SIMM),
                       "vspltisw $vD, $SIMM", VecPerm,
                       [(set v4i32:$vD, (v4i32 vecspltisw:$SIMM))]>;

// Vector Pack.
def VPKPX   : VX1_Int_Ty2<782, "vpkpx", int_ppc_altivec_vpkpx,
                          v8i16, v4i32>;
def VPKSHSS : VX1_Int_Ty2<398, "vpkshss", int_ppc_altivec_vpkshss,
                          v16i8, v8i16>;
def VPKSHUS : VX1_Int_Ty2<270, "vpkshus", int_ppc_altivec_vpkshus,
                          v16i8, v8i16>;
def VPKSWSS : VX1_Int_Ty2<462, "vpkswss", int_ppc_altivec_vpkswss,
                          v16i8, v4i32>;
def VPKSWUS : VX1_Int_Ty2<334, "vpkswus", int_ppc_altivec_vpkswus,
                          v8i16, v4i32>;
def VPKUHUM : VXForm_1<14, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                       "vpkuhum $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                         (vpkuhum_shuffle (v16i8 VRRC:$vA), VRRC:$vB))]>;
def VPKUHUS : VX1_Int_Ty2<142, "vpkuhus", int_ppc_altivec_vpkuhus,
                          v16i8, v8i16>;
def VPKUWUM : VXForm_1<78, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                       "vpkuwum $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                         (vpkuwum_shuffle (v16i8 VRRC:$vA), VRRC:$vB))]>;
def VPKUWUS : VX1_Int_Ty2<206, "vpkuwus", int_ppc_altivec_vpkuwus,
                          v8i16, v4i32>;

// Vector Unpack.
def VUPKHPX : VX2_Int_Ty2<846, "vupkhpx", int_ppc_altivec_vupkhpx,
                          v4i32, v8i16>;
def VUPKHSB : VX2_Int_Ty2<526, "vupkhsb", int_ppc_altivec_vupkhsb,
                          v8i16, v16i8>;
def VUPKHSH : VX2_Int_Ty2<590, "vupkhsh", int_ppc_altivec_vupkhsh,
                          v4i32, v8i16>;
def VUPKLPX : VX2_Int_Ty2<974, "vupklpx", int_ppc_altivec_vupklpx,
                          v4i32, v8i16>;
def VUPKLSB : VX2_Int_Ty2<654, "vupklsb", int_ppc_altivec_vupklsb,
                          v8i16, v16i8>;
def VUPKLSH : VX2_Int_Ty2<718, "vupklsh", int_ppc_altivec_vupklsh,
                          v4i32, v8i16>;


// Altivec Comparisons.

class VCMP<bits<10> xo, string asmstr, ValueType Ty>
  : VXRForm_1<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),asmstr,VecFPCompare,
              [(set Ty:$vD, (Ty (PPCvcmp Ty:$vA, Ty:$vB, xo)))]>;
class VCMPo<bits<10> xo, string asmstr, ValueType Ty>
  : VXRForm_1<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),asmstr,VecFPCompare,
              [(set Ty:$vD, (Ty (PPCvcmp_o Ty:$vA, Ty:$vB, xo)))]> {
  let Defs = [CR6];
  let RC = 1;
}

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
                      
let isCodeGenOnly = 1 in
def V_SET0 : VXForm_setzero<1220, (outs VRRC:$vD), (ins),
                      "vxor $vD, $vD, $vD", VecFP,
                      [(set VRRC:$vD, (v4i32 immAllZerosV))]>;
let IMM=-1 in {
def V_SETALLONES : VXForm_3<908, (outs VRRC:$vD), (ins),
                      "vspltisw $vD, -1", VecFP,
                      [(set v4i32:$vD, (v4i32 immAllOnesV))]>;
}
} // VALU Operations.

//===----------------------------------------------------------------------===//
// Additional Altivec Patterns
//

// DS* intrinsics
def : Pat<(int_ppc_altivec_dssall), (DSSALL 1, 0, 0, 0)>;
def : Pat<(int_ppc_altivec_dss imm:$STRM), (DSS 0, imm:$STRM, 0, 0)>;

//  * 32-bit
def : Pat<(int_ppc_altivec_dst i32:$rA, i32:$rB, imm:$STRM),
          (DST 0, imm:$STRM, $rA, $rB)>;
def : Pat<(int_ppc_altivec_dstt i32:$rA, i32:$rB, imm:$STRM),
          (DSTT 1, imm:$STRM, $rA, $rB)>;
def : Pat<(int_ppc_altivec_dstst i32:$rA, i32:$rB, imm:$STRM),
          (DSTST 0, imm:$STRM, $rA, $rB)>;
def : Pat<(int_ppc_altivec_dststt i32:$rA, i32:$rB, imm:$STRM),
          (DSTSTT 1, imm:$STRM, $rA, $rB)>;

//  * 64-bit
def : Pat<(int_ppc_altivec_dst i64:$rA, i32:$rB, imm:$STRM),
          (DST64 0, imm:$STRM, $rA, $rB)>;
def : Pat<(int_ppc_altivec_dstt i64:$rA, i32:$rB, imm:$STRM),
          (DSTT64 1, imm:$STRM, $rA, $rB)>;
def : Pat<(int_ppc_altivec_dstst i64:$rA, i32:$rB, imm:$STRM),
          (DSTST64 0, imm:$STRM, $rA, $rB)>;
def : Pat<(int_ppc_altivec_dststt i64:$rA, i32:$rB, imm:$STRM),
          (DSTSTT64 1, imm:$STRM, $rA, $rB)>;

// Loads.
def : Pat<(v4i32 (load xoaddr:$src)), (LVX xoaddr:$src)>;

// Stores.
def : Pat<(store v4i32:$rS, xoaddr:$dst),
          (STVX $rS, xoaddr:$dst)>;

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

// Shuffles.

// Match vsldoi(x,x), vpkuwum(x,x), vpkuhum(x,x)
def:Pat<(vsldoi_unary_shuffle:$in v16i8:$vA, undef),
        (VSLDOI $vA, $vA, (VSLDOI_unary_get_imm VRRC:$in))>;
def:Pat<(vpkuwum_unary_shuffle v16i8:$vA, undef),
        (VPKUWUM $vA, $vA)>;
def:Pat<(vpkuhum_unary_shuffle v16i8:$vA, undef),
        (VPKUHUM $vA, $vA)>;

// Match vmrg*(x,x)
def:Pat<(vmrglb_unary_shuffle v16i8:$vA, undef),
        (VMRGLB $vA, $vA)>;
def:Pat<(vmrglh_unary_shuffle v16i8:$vA, undef),
        (VMRGLH $vA, $vA)>;
def:Pat<(vmrglw_unary_shuffle v16i8:$vA, undef),
        (VMRGLW $vA, $vA)>;
def:Pat<(vmrghb_unary_shuffle v16i8:$vA, undef),
        (VMRGHB $vA, $vA)>;
def:Pat<(vmrghh_unary_shuffle v16i8:$vA, undef),
        (VMRGHH $vA, $vA)>;
def:Pat<(vmrghw_unary_shuffle v16i8:$vA, undef),
        (VMRGHW $vA, $vA)>;

// Logical Operations
def : Pat<(v4i32 (vnot_ppc VRRC:$vA)), (VNOR $vA, $vA)>;

def : Pat<(v4i32 (vnot_ppc (or VRRC:$A, VRRC:$B))),
          (VNOR $A, $B)>;
def : Pat<(v4i32 (and VRRC:$A, (vnot_ppc VRRC:$B))),
          (VANDC $A, $B)>;

def : Pat<(fmul v4f32:$vA, v4f32:$vB),
          (VMADDFP $vA, $vB,
             (v4i32 (VSLW (V_SETALLONES), (V_SETALLONES))))>; 

// Fused multiply add and multiply sub for packed float.  These are represented
// separately from the real instructions above, for operations that must have
// the additional precision, such as Newton-Rhapson (used by divide, sqrt)
def : Pat<(PPCvmaddfp v4f32:$A, v4f32:$B, v4f32:$C),
          (VMADDFP $A, $B, $C)>;
def : Pat<(PPCvnmsubfp v4f32:$A, v4f32:$B, v4f32:$C),
          (VNMSUBFP $A, $B, $C)>;

def : Pat<(int_ppc_altivec_vmaddfp v4f32:$A, v4f32:$B, v4f32:$C),
          (VMADDFP $A, $B, $C)>;
def : Pat<(int_ppc_altivec_vnmsubfp v4f32:$A, v4f32:$B, v4f32:$C),
          (VNMSUBFP $A, $B, $C)>;

def : Pat<(PPCvperm (v16i8 VRRC:$vA), VRRC:$vB, VRRC:$vC),
          (VPERM $vA, $vB, $vC)>;

// Vector shifts
def : Pat<(v16i8 (shl v16i8:$vA, v16i8:$vB)),
          (v16i8 (VSLB $vA, $vB))>;
def : Pat<(v8i16 (shl v8i16:$vA, v8i16:$vB)),
          (v8i16 (VSLH $vA, $vB))>;
def : Pat<(v4i32 (shl v4i32:$vA, v4i32:$vB)),
          (v4i32 (VSLW $vA, $vB))>;

def : Pat<(v16i8 (srl v16i8:$vA, v16i8:$vB)),
          (v16i8 (VSRB $vA, $vB))>;
def : Pat<(v8i16 (srl v8i16:$vA, v8i16:$vB)),
          (v8i16 (VSRH $vA, $vB))>;
def : Pat<(v4i32 (srl v4i32:$vA, v4i32:$vB)),
          (v4i32 (VSRW $vA, $vB))>;

def : Pat<(v16i8 (sra v16i8:$vA, v16i8:$vB)),
          (v16i8 (VSRAB $vA, $vB))>;
def : Pat<(v8i16 (sra v8i16:$vA, v8i16:$vB)),
          (v8i16 (VSRAH $vA, $vB))>;
def : Pat<(v4i32 (sra v4i32:$vA, v4i32:$vB)),
          (v4i32 (VSRAW $vA, $vB))>;

// Float to integer and integer to float conversions
def : Pat<(v4i32 (fp_to_sint v4f32:$vA)),
           (VCTSXS_0 $vA)>;
def : Pat<(v4i32 (fp_to_uint v4f32:$vA)),
           (VCTUXS_0 $vA)>;
def : Pat<(v4f32 (sint_to_fp v4i32:$vA)),
           (VCFSX_0 $vA)>;
def : Pat<(v4f32 (uint_to_fp v4i32:$vA)),
           (VCFUX_0 $vA)>;

// Floating-point rounding
def : Pat<(v4f32 (ffloor v4f32:$vA)),
          (VRFIM $vA)>;
def : Pat<(v4f32 (fceil v4f32:$vA)),
          (VRFIP $vA)>;
def : Pat<(v4f32 (ftrunc v4f32:$vA)),
          (VRFIZ $vA)>;
def : Pat<(v4f32 (fnearbyint v4f32:$vA)),
          (VRFIN $vA)>;

} // end HasAltivec

