//===- ARMInstrNEON.td - NEON support for ARM -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the ARM NEON instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// NEON-specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDTARMVCMP    : SDTypeProfile<1, 2, [SDTCisInt<0>, SDTCisSameAs<1, 2>]>;

def NEONvceq      : SDNode<"ARMISD::VCEQ", SDTARMVCMP>;
def NEONvcge      : SDNode<"ARMISD::VCGE", SDTARMVCMP>;
def NEONvcgeu     : SDNode<"ARMISD::VCGEU", SDTARMVCMP>;
def NEONvcgt      : SDNode<"ARMISD::VCGT", SDTARMVCMP>;
def NEONvcgtu     : SDNode<"ARMISD::VCGTU", SDTARMVCMP>;
def NEONvtst      : SDNode<"ARMISD::VTST", SDTARMVCMP>;

// Types for vector shift by immediates.  The "SHX" version is for long and
// narrow operations where the source and destination vectors have different
// types.  The "SHINS" version is for shift and insert operations.
def SDTARMVSH     : SDTypeProfile<1, 2, [SDTCisInt<0>, SDTCisSameAs<0, 1>,
                                         SDTCisVT<2, i32>]>;
def SDTARMVSHX    : SDTypeProfile<1, 2, [SDTCisInt<0>, SDTCisInt<1>,
                                         SDTCisVT<2, i32>]>;
def SDTARMVSHINS  : SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisSameAs<0, 1>,
                                         SDTCisSameAs<0, 2>, SDTCisVT<3, i32>]>;

def NEONvshl      : SDNode<"ARMISD::VSHL", SDTARMVSH>;
def NEONvshrs     : SDNode<"ARMISD::VSHRs", SDTARMVSH>;
def NEONvshru     : SDNode<"ARMISD::VSHRu", SDTARMVSH>;
def NEONvshlls    : SDNode<"ARMISD::VSHLLs", SDTARMVSHX>;
def NEONvshllu    : SDNode<"ARMISD::VSHLLu", SDTARMVSHX>;
def NEONvshlli    : SDNode<"ARMISD::VSHLLi", SDTARMVSHX>;
def NEONvshrn     : SDNode<"ARMISD::VSHRN", SDTARMVSHX>;

def NEONvrshrs    : SDNode<"ARMISD::VRSHRs", SDTARMVSH>;
def NEONvrshru    : SDNode<"ARMISD::VRSHRu", SDTARMVSH>;
def NEONvrshrn    : SDNode<"ARMISD::VRSHRN", SDTARMVSHX>;

def NEONvqshls    : SDNode<"ARMISD::VQSHLs", SDTARMVSH>;
def NEONvqshlu    : SDNode<"ARMISD::VQSHLu", SDTARMVSH>;
def NEONvqshlsu   : SDNode<"ARMISD::VQSHLsu", SDTARMVSH>;
def NEONvqshrns   : SDNode<"ARMISD::VQSHRNs", SDTARMVSHX>;
def NEONvqshrnu   : SDNode<"ARMISD::VQSHRNu", SDTARMVSHX>;
def NEONvqshrnsu  : SDNode<"ARMISD::VQSHRNsu", SDTARMVSHX>;

def NEONvqrshrns  : SDNode<"ARMISD::VQRSHRNs", SDTARMVSHX>;
def NEONvqrshrnu  : SDNode<"ARMISD::VQRSHRNu", SDTARMVSHX>;
def NEONvqrshrnsu : SDNode<"ARMISD::VQRSHRNsu", SDTARMVSHX>;

def NEONvsli      : SDNode<"ARMISD::VSLI", SDTARMVSHINS>;
def NEONvsri      : SDNode<"ARMISD::VSRI", SDTARMVSHINS>;

def SDTARMVGETLN  : SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisInt<1>,
                                         SDTCisVT<2, i32>]>;
def NEONvgetlaneu : SDNode<"ARMISD::VGETLANEu", SDTARMVGETLN>;
def NEONvgetlanes : SDNode<"ARMISD::VGETLANEs", SDTARMVGETLN>;

def SDTARMVMOVIMM : SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVT<1, i32>]>;
def NEONvmovImm   : SDNode<"ARMISD::VMOVIMM", SDTARMVMOVIMM>;
def NEONvmvnImm   : SDNode<"ARMISD::VMVNIMM", SDTARMVMOVIMM>;

def NEONvdup      : SDNode<"ARMISD::VDUP", SDTypeProfile<1, 1, [SDTCisVec<0>]>>;

// VDUPLANE can produce a quad-register result from a double-register source,
// so the result is not constrained to match the source.
def NEONvduplane  : SDNode<"ARMISD::VDUPLANE",
                           SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                                SDTCisVT<2, i32>]>>;

def SDTARMVEXT    : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0, 1>,
                                         SDTCisSameAs<0, 2>, SDTCisVT<3, i32>]>;
def NEONvext      : SDNode<"ARMISD::VEXT", SDTARMVEXT>;

def SDTARMVSHUF   : SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisSameAs<0, 1>]>;
def NEONvrev64    : SDNode<"ARMISD::VREV64", SDTARMVSHUF>;
def NEONvrev32    : SDNode<"ARMISD::VREV32", SDTARMVSHUF>;
def NEONvrev16    : SDNode<"ARMISD::VREV16", SDTARMVSHUF>;

def SDTARMVSHUF2  : SDTypeProfile<2, 2, [SDTCisVec<0>, SDTCisSameAs<0, 1>,
                                         SDTCisSameAs<0, 2>,
                                         SDTCisSameAs<0, 3>]>;
def NEONzip       : SDNode<"ARMISD::VZIP", SDTARMVSHUF2>;
def NEONuzp       : SDNode<"ARMISD::VUZP", SDTARMVSHUF2>;
def NEONtrn       : SDNode<"ARMISD::VTRN", SDTARMVSHUF2>;

def SDTARMVMULL   : SDTypeProfile<1, 2, [SDTCisInt<0>, SDTCisInt<1>,
                                         SDTCisSameAs<1, 2>]>;
def NEONvmulls    : SDNode<"ARMISD::VMULLs", SDTARMVMULL>;
def NEONvmullu    : SDNode<"ARMISD::VMULLu", SDTARMVMULL>;

def SDTARMFMAX    : SDTypeProfile<1, 2, [SDTCisVT<0, f32>, SDTCisSameAs<0, 1>,
                                         SDTCisSameAs<0, 2>]>;
def NEONfmax      : SDNode<"ARMISD::FMAX", SDTARMFMAX>;
def NEONfmin      : SDNode<"ARMISD::FMIN", SDTARMFMAX>;

def NEONimmAllZerosV: PatLeaf<(NEONvmovImm (i32 timm)), [{
  ConstantSDNode *ConstVal = cast<ConstantSDNode>(N->getOperand(0));
  unsigned EltBits = 0;
  uint64_t EltVal = ARM_AM::decodeNEONModImm(ConstVal->getZExtValue(), EltBits);
  return (EltBits == 32 && EltVal == 0);
}]>;

def NEONimmAllOnesV: PatLeaf<(NEONvmovImm (i32 timm)), [{
  ConstantSDNode *ConstVal = cast<ConstantSDNode>(N->getOperand(0));
  unsigned EltBits = 0;
  uint64_t EltVal = ARM_AM::decodeNEONModImm(ConstVal->getZExtValue(), EltBits);
  return (EltBits == 8 && EltVal == 0xff);
}]>;

//===----------------------------------------------------------------------===//
// NEON operand definitions
//===----------------------------------------------------------------------===//

def nModImm : Operand<i32> {
  let PrintMethod = "printNEONModImmOperand";
}

//===----------------------------------------------------------------------===//
// NEON load / store instructions
//===----------------------------------------------------------------------===//

// Use VLDM to load a Q register as a D register pair.
// This is a pseudo instruction that is expanded to VLDMD after reg alloc.
def VLDMQ
  : PseudoVFPLdStM<(outs QPR:$dst), (ins GPR:$Rn, ldstm_mode:$mode),
                    IIC_fpLoad_m, "",
                   [(set QPR:$dst, (v2f64 (load GPR:$Rn)))]>;

// Use VSTM to store a Q register as a D register pair.
// This is a pseudo instruction that is expanded to VSTMD after reg alloc.
def VSTMQ
  : PseudoVFPLdStM<(outs), (ins QPR:$src, GPR:$Rn, ldstm_mode:$mode),
                    IIC_fpStore_m, "",
                   [(store (v2f64 QPR:$src), GPR:$Rn)]>;

let mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1 in {

// Classes for VLD* pseudo-instructions with multi-register operands.
// These are expanded to real instructions after register allocation.
class VLDQPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QPR:$dst), (ins addrmode6:$addr), itin, "">;
class VLDQWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QPR:$dst, GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset), itin,
                "$addr.addr = $wb">;
class VLDQQPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QQPR:$dst), (ins addrmode6:$addr), itin, "">;
class VLDQQWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QQPR:$dst, GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset), itin,
                "$addr.addr = $wb">;
class VLDQQQQWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QQQQPR:$dst, GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset, QQQQPR:$src), itin,
                "$addr.addr = $wb, $src = $dst">;

//   VLD1     : Vector Load (multiple single elements)
class VLD1D<bits<4> op7_4, string Dt>
  : NLdSt<0,0b10,0b0111,op7_4, (outs DPR:$Vd),
          (ins addrmode6:$Rn), IIC_VLD1,
          "vld1", Dt, "\\{$Vd\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{4} = Rn{4};
}
class VLD1Q<bits<4> op7_4, string Dt>
  : NLdSt<0,0b10,0b1010,op7_4, (outs DPR:$Vd, DPR:$dst2),
          (ins addrmode6:$Rn), IIC_VLD1x2,
          "vld1", Dt, "\\{$Vd, $dst2\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{5-4} = Rn{5-4};
}

def  VLD1d8   : VLD1D<{0,0,0,?}, "8">;
def  VLD1d16  : VLD1D<{0,1,0,?}, "16">;
def  VLD1d32  : VLD1D<{1,0,0,?}, "32">;
def  VLD1d64  : VLD1D<{1,1,0,?}, "64">;

def  VLD1q8   : VLD1Q<{0,0,?,?}, "8">;
def  VLD1q16  : VLD1Q<{0,1,?,?}, "16">;
def  VLD1q32  : VLD1Q<{1,0,?,?}, "32">;
def  VLD1q64  : VLD1Q<{1,1,?,?}, "64">;

def  VLD1q8Pseudo  : VLDQPseudo<IIC_VLD1x2>;
def  VLD1q16Pseudo : VLDQPseudo<IIC_VLD1x2>;
def  VLD1q32Pseudo : VLDQPseudo<IIC_VLD1x2>;
def  VLD1q64Pseudo : VLDQPseudo<IIC_VLD1x2>;

// ...with address register writeback:
class VLD1DWB<bits<4> op7_4, string Dt>
  : NLdSt<0,0b10,0b0111,op7_4, (outs DPR:$Vd, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm), IIC_VLD1u,
          "vld1", Dt, "\\{$Vd\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{4} = Rn{4}; 
}
class VLD1QWB<bits<4> op7_4, string Dt>
  : NLdSt<0,0b10,0b1010,op7_4, (outs DPR:$Vd, DPR:$dst2, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm), IIC_VLD1x2u,
          "vld1", Dt, "\\{$Vd, $dst2\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{5-4} = Rn{5-4};
}

def VLD1d8_UPD  : VLD1DWB<{0,0,0,?}, "8">;
def VLD1d16_UPD : VLD1DWB<{0,1,0,?}, "16">;
def VLD1d32_UPD : VLD1DWB<{1,0,0,?}, "32">;
def VLD1d64_UPD : VLD1DWB<{1,1,0,?}, "64">;

def VLD1q8_UPD  : VLD1QWB<{0,0,?,?}, "8">;
def VLD1q16_UPD : VLD1QWB<{0,1,?,?}, "16">;
def VLD1q32_UPD : VLD1QWB<{1,0,?,?}, "32">;
def VLD1q64_UPD : VLD1QWB<{1,1,?,?}, "64">;

def VLD1q8Pseudo_UPD  : VLDQWBPseudo<IIC_VLD1x2u>;
def VLD1q16Pseudo_UPD : VLDQWBPseudo<IIC_VLD1x2u>;
def VLD1q32Pseudo_UPD : VLDQWBPseudo<IIC_VLD1x2u>;
def VLD1q64Pseudo_UPD : VLDQWBPseudo<IIC_VLD1x2u>;

// ...with 3 registers (some of these are only for the disassembler):
class VLD1D3<bits<4> op7_4, string Dt>
  : NLdSt<0,0b10,0b0110,op7_4, (outs DPR:$Vd, DPR:$dst2, DPR:$dst3),
          (ins addrmode6:$Rn), IIC_VLD1x3, "vld1", Dt,
          "\\{$Vd, $dst2, $dst3\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{4} = Rn{4};
}
class VLD1D3WB<bits<4> op7_4, string Dt>
  : NLdSt<0,0b10,0b0110,op7_4, (outs DPR:$Vd, DPR:$dst2, DPR:$dst3, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm), IIC_VLD1x3u, "vld1", Dt,
          "\\{$Vd, $dst2, $dst3\\}, $Rn$Rm", "$Rn.addr = $wb", []> {
  let Inst{4} = Rn{4};
}

def VLD1d8T      : VLD1D3<{0,0,0,?}, "8">;
def VLD1d16T     : VLD1D3<{0,1,0,?}, "16">;
def VLD1d32T     : VLD1D3<{1,0,0,?}, "32">;
def VLD1d64T     : VLD1D3<{1,1,0,?}, "64">;

def VLD1d8T_UPD  : VLD1D3WB<{0,0,0,?}, "8">;
def VLD1d16T_UPD : VLD1D3WB<{0,1,0,?}, "16">;
def VLD1d32T_UPD : VLD1D3WB<{1,0,0,?}, "32">;
def VLD1d64T_UPD : VLD1D3WB<{1,1,0,?}, "64">;

def VLD1d64TPseudo     : VLDQQPseudo<IIC_VLD1x3>;
def VLD1d64TPseudo_UPD : VLDQQWBPseudo<IIC_VLD1x3u>;

// ...with 4 registers (some of these are only for the disassembler):
class VLD1D4<bits<4> op7_4, string Dt>
  : NLdSt<0,0b10,0b0010,op7_4,(outs DPR:$Vd, DPR:$dst2, DPR:$dst3, DPR:$dst4),
          (ins addrmode6:$Rn), IIC_VLD1x4, "vld1", Dt,
          "\\{$Vd, $dst2, $dst3, $dst4\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{5-4} = Rn{5-4};
}
class VLD1D4WB<bits<4> op7_4, string Dt>
  : NLdSt<0,0b10,0b0010,op7_4,
          (outs DPR:$Vd, DPR:$dst2, DPR:$dst3, DPR:$dst4, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm), IIC_VLD4, "vld1", Dt,
          "\\{$Vd, $dst2, $dst3, $dst4\\}, $Rn$Rm", "$Rn.addr = $wb",
          []> {
  let Inst{5-4} = Rn{5-4};
}

def VLD1d8Q      : VLD1D4<{0,0,?,?}, "8">;
def VLD1d16Q     : VLD1D4<{0,1,?,?}, "16">;
def VLD1d32Q     : VLD1D4<{1,0,?,?}, "32">;
def VLD1d64Q     : VLD1D4<{1,1,?,?}, "64">;

def VLD1d8Q_UPD  : VLD1D4WB<{0,0,?,?}, "8">;
def VLD1d16Q_UPD : VLD1D4WB<{0,1,?,?}, "16">;
def VLD1d32Q_UPD : VLD1D4WB<{1,0,?,?}, "32">;
def VLD1d64Q_UPD : VLD1D4WB<{1,1,?,?}, "64">;

def VLD1d64QPseudo     : VLDQQPseudo<IIC_VLD1x4>;
def VLD1d64QPseudo_UPD : VLDQQWBPseudo<IIC_VLD1x4u>;

//   VLD2     : Vector Load (multiple 2-element structures)
class VLD2D<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b10, op11_8, op7_4, (outs DPR:$Vd, DPR:$dst2),
          (ins addrmode6:$Rn), IIC_VLD2,
          "vld2", Dt, "\\{$Vd, $dst2\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{5-4} = Rn{5-4};
}
class VLD2Q<bits<4> op7_4, string Dt>
  : NLdSt<0, 0b10, 0b0011, op7_4,
          (outs DPR:$Vd, DPR:$dst2, DPR:$dst3, DPR:$dst4),
          (ins addrmode6:$Rn), IIC_VLD2x2,
          "vld2", Dt, "\\{$Vd, $dst2, $dst3, $dst4\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{5-4} = Rn{5-4};
}

def  VLD2d8   : VLD2D<0b1000, {0,0,?,?}, "8">;
def  VLD2d16  : VLD2D<0b1000, {0,1,?,?}, "16">;
def  VLD2d32  : VLD2D<0b1000, {1,0,?,?}, "32">;

def  VLD2q8   : VLD2Q<{0,0,?,?}, "8">;
def  VLD2q16  : VLD2Q<{0,1,?,?}, "16">;
def  VLD2q32  : VLD2Q<{1,0,?,?}, "32">;

def  VLD2d8Pseudo  : VLDQPseudo<IIC_VLD2>;
def  VLD2d16Pseudo : VLDQPseudo<IIC_VLD2>;
def  VLD2d32Pseudo : VLDQPseudo<IIC_VLD2>;

def  VLD2q8Pseudo  : VLDQQPseudo<IIC_VLD2x2>;
def  VLD2q16Pseudo : VLDQQPseudo<IIC_VLD2x2>;
def  VLD2q32Pseudo : VLDQQPseudo<IIC_VLD2x2>;

// ...with address register writeback:
class VLD2DWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b10, op11_8, op7_4, (outs DPR:$Vd, DPR:$dst2, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm), IIC_VLD2u,
          "vld2", Dt, "\\{$Vd, $dst2\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{5-4} = Rn{5-4};
}
class VLD2QWB<bits<4> op7_4, string Dt>
  : NLdSt<0, 0b10, 0b0011, op7_4,
          (outs DPR:$Vd, DPR:$dst2, DPR:$dst3, DPR:$dst4, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm), IIC_VLD2x2u,
          "vld2", Dt, "\\{$Vd, $dst2, $dst3, $dst4\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{5-4} = Rn{5-4};
}

def VLD2d8_UPD  : VLD2DWB<0b1000, {0,0,?,?}, "8">;
def VLD2d16_UPD : VLD2DWB<0b1000, {0,1,?,?}, "16">;
def VLD2d32_UPD : VLD2DWB<0b1000, {1,0,?,?}, "32">;

def VLD2q8_UPD  : VLD2QWB<{0,0,?,?}, "8">;
def VLD2q16_UPD : VLD2QWB<{0,1,?,?}, "16">;
def VLD2q32_UPD : VLD2QWB<{1,0,?,?}, "32">;

def VLD2d8Pseudo_UPD  : VLDQWBPseudo<IIC_VLD2u>;
def VLD2d16Pseudo_UPD : VLDQWBPseudo<IIC_VLD2u>;
def VLD2d32Pseudo_UPD : VLDQWBPseudo<IIC_VLD2u>;

def VLD2q8Pseudo_UPD  : VLDQQWBPseudo<IIC_VLD2x2u>;
def VLD2q16Pseudo_UPD : VLDQQWBPseudo<IIC_VLD2x2u>;
def VLD2q32Pseudo_UPD : VLDQQWBPseudo<IIC_VLD2x2u>;

// ...with double-spaced registers (for disassembly only):
def VLD2b8      : VLD2D<0b1001, {0,0,?,?}, "8">;
def VLD2b16     : VLD2D<0b1001, {0,1,?,?}, "16">;
def VLD2b32     : VLD2D<0b1001, {1,0,?,?}, "32">;
def VLD2b8_UPD  : VLD2DWB<0b1001, {0,0,?,?}, "8">;
def VLD2b16_UPD : VLD2DWB<0b1001, {0,1,?,?}, "16">;
def VLD2b32_UPD : VLD2DWB<0b1001, {1,0,?,?}, "32">;

//   VLD3     : Vector Load (multiple 3-element structures)
class VLD3D<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b10, op11_8, op7_4, (outs DPR:$Vd, DPR:$dst2, DPR:$dst3),
          (ins addrmode6:$Rn), IIC_VLD3,
          "vld3", Dt, "\\{$Vd, $dst2, $dst3\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{4} = Rn{4};
}

def  VLD3d8   : VLD3D<0b0100, {0,0,0,?}, "8">;
def  VLD3d16  : VLD3D<0b0100, {0,1,0,?}, "16">;
def  VLD3d32  : VLD3D<0b0100, {1,0,0,?}, "32">;

def  VLD3d8Pseudo  : VLDQQPseudo<IIC_VLD3>;
def  VLD3d16Pseudo : VLDQQPseudo<IIC_VLD3>;
def  VLD3d32Pseudo : VLDQQPseudo<IIC_VLD3>;

// ...with address register writeback:
class VLD3DWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b10, op11_8, op7_4,
          (outs DPR:$Vd, DPR:$dst2, DPR:$dst3, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm), IIC_VLD3u,
          "vld3", Dt, "\\{$Vd, $dst2, $dst3\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{4} = Rn{4};
}

def VLD3d8_UPD  : VLD3DWB<0b0100, {0,0,0,?}, "8">;
def VLD3d16_UPD : VLD3DWB<0b0100, {0,1,0,?}, "16">;
def VLD3d32_UPD : VLD3DWB<0b0100, {1,0,0,?}, "32">;

def VLD3d8Pseudo_UPD  : VLDQQWBPseudo<IIC_VLD3u>;
def VLD3d16Pseudo_UPD : VLDQQWBPseudo<IIC_VLD3u>;
def VLD3d32Pseudo_UPD : VLDQQWBPseudo<IIC_VLD3u>;

// ...with double-spaced registers (non-updating versions for disassembly only):
def VLD3q8      : VLD3D<0b0101, {0,0,0,?}, "8">;
def VLD3q16     : VLD3D<0b0101, {0,1,0,?}, "16">;
def VLD3q32     : VLD3D<0b0101, {1,0,0,?}, "32">;
def VLD3q8_UPD  : VLD3DWB<0b0101, {0,0,0,?}, "8">;
def VLD3q16_UPD : VLD3DWB<0b0101, {0,1,0,?}, "16">;
def VLD3q32_UPD : VLD3DWB<0b0101, {1,0,0,?}, "32">;

def VLD3q8Pseudo_UPD  : VLDQQQQWBPseudo<IIC_VLD3u>;
def VLD3q16Pseudo_UPD : VLDQQQQWBPseudo<IIC_VLD3u>;
def VLD3q32Pseudo_UPD : VLDQQQQWBPseudo<IIC_VLD3u>;

// ...alternate versions to be allocated odd register numbers:
def VLD3q8oddPseudo_UPD  : VLDQQQQWBPseudo<IIC_VLD3u>;
def VLD3q16oddPseudo_UPD : VLDQQQQWBPseudo<IIC_VLD3u>;
def VLD3q32oddPseudo_UPD : VLDQQQQWBPseudo<IIC_VLD3u>;

//   VLD4     : Vector Load (multiple 4-element structures)
class VLD4D<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b10, op11_8, op7_4,
          (outs DPR:$Vd, DPR:$dst2, DPR:$dst3, DPR:$dst4),
          (ins addrmode6:$Rn), IIC_VLD4,
          "vld4", Dt, "\\{$Vd, $dst2, $dst3, $dst4\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{5-4} = Rn{5-4};
}

def  VLD4d8   : VLD4D<0b0000, {0,0,?,?}, "8">;
def  VLD4d16  : VLD4D<0b0000, {0,1,?,?}, "16">;
def  VLD4d32  : VLD4D<0b0000, {1,0,?,?}, "32">;

def  VLD4d8Pseudo  : VLDQQPseudo<IIC_VLD4>;
def  VLD4d16Pseudo : VLDQQPseudo<IIC_VLD4>;
def  VLD4d32Pseudo : VLDQQPseudo<IIC_VLD4>;

// ...with address register writeback:
class VLD4DWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b10, op11_8, op7_4,
          (outs DPR:$Vd, DPR:$dst2, DPR:$dst3, DPR:$dst4, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm), IIC_VLD4,
          "vld4", Dt, "\\{$Vd, $dst2, $dst3, $dst4\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{5-4} = Rn{5-4};
}

def VLD4d8_UPD  : VLD4DWB<0b0000, {0,0,?,?}, "8">;
def VLD4d16_UPD : VLD4DWB<0b0000, {0,1,?,?}, "16">;
def VLD4d32_UPD : VLD4DWB<0b0000, {1,0,?,?}, "32">;

def VLD4d8Pseudo_UPD  : VLDQQWBPseudo<IIC_VLD4>;
def VLD4d16Pseudo_UPD : VLDQQWBPseudo<IIC_VLD4>;
def VLD4d32Pseudo_UPD : VLDQQWBPseudo<IIC_VLD4>;

// ...with double-spaced registers (non-updating versions for disassembly only):
def VLD4q8      : VLD4D<0b0001, {0,0,?,?}, "8">;
def VLD4q16     : VLD4D<0b0001, {0,1,?,?}, "16">;
def VLD4q32     : VLD4D<0b0001, {1,0,?,?}, "32">;
def VLD4q8_UPD  : VLD4DWB<0b0001, {0,0,?,?}, "8">;
def VLD4q16_UPD : VLD4DWB<0b0001, {0,1,?,?}, "16">;
def VLD4q32_UPD : VLD4DWB<0b0001, {1,0,?,?}, "32">;

def VLD4q8Pseudo_UPD  : VLDQQQQWBPseudo<IIC_VLD4>;
def VLD4q16Pseudo_UPD : VLDQQQQWBPseudo<IIC_VLD4>;
def VLD4q32Pseudo_UPD : VLDQQQQWBPseudo<IIC_VLD4>;

// ...alternate versions to be allocated odd register numbers:
def VLD4q8oddPseudo_UPD  : VLDQQQQWBPseudo<IIC_VLD4>;
def VLD4q16oddPseudo_UPD : VLDQQQQWBPseudo<IIC_VLD4>;
def VLD4q32oddPseudo_UPD : VLDQQQQWBPseudo<IIC_VLD4>;

} // mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1

// Classes for VLD*LN pseudo-instructions with multi-register operands.
// These are expanded to real instructions after register allocation.
class VLDQLNPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QPR:$dst),
                (ins addrmode6:$addr, QPR:$src, nohash_imm:$lane),
                itin, "$src = $dst">;
class VLDQLNWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QPR:$dst, GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset, QPR:$src,
                 nohash_imm:$lane), itin, "$addr.addr = $wb, $src = $dst">;
class VLDQQLNPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QQPR:$dst),
                (ins addrmode6:$addr, QQPR:$src, nohash_imm:$lane),
                itin, "$src = $dst">;
class VLDQQLNWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QQPR:$dst, GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset, QQPR:$src,
                 nohash_imm:$lane), itin, "$addr.addr = $wb, $src = $dst">;
class VLDQQQQLNPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QQQQPR:$dst),
                (ins addrmode6:$addr, QQQQPR:$src, nohash_imm:$lane),
                itin, "$src = $dst">;
class VLDQQQQLNWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs QQQQPR:$dst, GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset, QQQQPR:$src,
                 nohash_imm:$lane), itin, "$addr.addr = $wb, $src = $dst">;

//   VLD1LN   : Vector Load (single element to one lane)
class VLD1LN<bits<4> op11_8, bits<4> op7_4, string Dt, ValueType Ty,
             PatFrag LoadOp>
  : NLdStLn<1, 0b10, op11_8, op7_4, (outs DPR:$Vd),
          (ins addrmode6:$Rn, DPR:$src, nohash_imm:$lane),
          IIC_VLD1ln, "vld1", Dt, "\\{$Vd[$lane]\\}, $Rn",
          "$src = $Vd",
          [(set DPR:$Vd, (vector_insert (Ty DPR:$src),
                                         (i32 (LoadOp addrmode6:$Rn)),
                                         imm:$lane))]> {
  let Rm = 0b1111;
}
class VLD1QLNPseudo<ValueType Ty, PatFrag LoadOp> : VLDQLNPseudo<IIC_VLD1ln> {
  let Pattern = [(set QPR:$dst, (vector_insert (Ty QPR:$src),
                                               (i32 (LoadOp addrmode6:$addr)),
                                               imm:$lane))];
}

def VLD1LNd8  : VLD1LN<0b0000, {?,?,?,0}, "8", v8i8, extloadi8> {
  let Inst{7-5} = lane{2-0};
}
def VLD1LNd16 : VLD1LN<0b0100, {?,?,0,?}, "16", v4i16, extloadi16> {
  let Inst{7-6} = lane{1-0};
  let Inst{4}   = Rn{4};
}
def VLD1LNd32 : VLD1LN<0b1000, {?,0,?,?}, "32", v2i32, load> {
  let Inst{7} = lane{0};
  let Inst{5} = Rn{4};
  let Inst{4} = Rn{4};
}

def VLD1LNq8Pseudo  : VLD1QLNPseudo<v16i8, extloadi8>;
def VLD1LNq16Pseudo : VLD1QLNPseudo<v8i16, extloadi16>;
def VLD1LNq32Pseudo : VLD1QLNPseudo<v4i32, load>;

let mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1 in {

// ...with address register writeback:
class VLD1LNWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b10, op11_8, op7_4, (outs DPR:$Vd, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$src, nohash_imm:$lane), IIC_VLD1lnu, "vld1", Dt,
          "\\{$Vd[$lane]\\}, $Rn$Rm",
          "$src = $Vd, $Rn.addr = $wb", []>;

def VLD1LNd8_UPD  : VLD1LNWB<0b0000, {?,?,?,0}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VLD1LNd16_UPD : VLD1LNWB<0b0100, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
  let Inst{4}   = Rn{4};
}
def VLD1LNd32_UPD : VLD1LNWB<0b1000, {?,0,?,?}, "32"> {
  let Inst{7} = lane{0};
  let Inst{5} = Rn{4};
  let Inst{4} = Rn{4};
}

def VLD1LNq8Pseudo_UPD  : VLDQLNWBPseudo<IIC_VLD1lnu>;
def VLD1LNq16Pseudo_UPD : VLDQLNWBPseudo<IIC_VLD1lnu>;
def VLD1LNq32Pseudo_UPD : VLDQLNWBPseudo<IIC_VLD1lnu>;

//   VLD2LN   : Vector Load (single 2-element structure to one lane)
class VLD2LN<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b10, op11_8, op7_4, (outs DPR:$Vd, DPR:$dst2),
          (ins addrmode6:$Rn, DPR:$src1, DPR:$src2, nohash_imm:$lane),
          IIC_VLD2ln, "vld2", Dt, "\\{$Vd[$lane], $dst2[$lane]\\}, $Rn",
          "$src1 = $Vd, $src2 = $dst2", []> {
  let Rm = 0b1111;
  let Inst{4}   = Rn{4};
}

def VLD2LNd8  : VLD2LN<0b0001, {?,?,?,?}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VLD2LNd16 : VLD2LN<0b0101, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD2LNd32 : VLD2LN<0b1001, {?,0,0,?}, "32"> {
  let Inst{7} = lane{0};
}

def VLD2LNd8Pseudo  : VLDQLNPseudo<IIC_VLD2ln>;
def VLD2LNd16Pseudo : VLDQLNPseudo<IIC_VLD2ln>;
def VLD2LNd32Pseudo : VLDQLNPseudo<IIC_VLD2ln>;

// ...with double-spaced registers:
def VLD2LNq16 : VLD2LN<0b0101, {?,?,1,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD2LNq32 : VLD2LN<0b1001, {?,1,0,?}, "32"> {
  let Inst{7} = lane{0};
}

def VLD2LNq16Pseudo : VLDQQLNPseudo<IIC_VLD2ln>;
def VLD2LNq32Pseudo : VLDQQLNPseudo<IIC_VLD2ln>;

// ...with address register writeback:
class VLD2LNWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b10, op11_8, op7_4, (outs DPR:$Vd, DPR:$dst2, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$src1, DPR:$src2, nohash_imm:$lane), IIC_VLD2lnu, "vld2", Dt,
          "\\{$Vd[$lane], $dst2[$lane]\\}, $Rn$Rm",
          "$src1 = $Vd, $src2 = $dst2, $Rn.addr = $wb", []> {
  let Inst{4}   = Rn{4};
}

def VLD2LNd8_UPD  : VLD2LNWB<0b0001, {?,?,?,?}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VLD2LNd16_UPD : VLD2LNWB<0b0101, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD2LNd32_UPD : VLD2LNWB<0b1001, {?,0,0,?}, "32"> {
  let Inst{7} = lane{0};
}

def VLD2LNd8Pseudo_UPD  : VLDQLNWBPseudo<IIC_VLD2lnu>;
def VLD2LNd16Pseudo_UPD : VLDQLNWBPseudo<IIC_VLD2lnu>;
def VLD2LNd32Pseudo_UPD : VLDQLNWBPseudo<IIC_VLD2lnu>;

def VLD2LNq16_UPD : VLD2LNWB<0b0101, {?,?,1,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD2LNq32_UPD : VLD2LNWB<0b1001, {?,1,0,?}, "32"> {
  let Inst{7} = lane{0};
}

def VLD2LNq16Pseudo_UPD : VLDQQLNWBPseudo<IIC_VLD2lnu>;
def VLD2LNq32Pseudo_UPD : VLDQQLNWBPseudo<IIC_VLD2lnu>;

//   VLD3LN   : Vector Load (single 3-element structure to one lane)
class VLD3LN<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b10, op11_8, op7_4, (outs DPR:$Vd, DPR:$dst2, DPR:$dst3),
          (ins addrmode6:$Rn, DPR:$src1, DPR:$src2, DPR:$src3,
          nohash_imm:$lane), IIC_VLD3ln, "vld3", Dt,
          "\\{$Vd[$lane], $dst2[$lane], $dst3[$lane]\\}, $Rn",
          "$src1 = $Vd, $src2 = $dst2, $src3 = $dst3", []> {
  let Rm = 0b1111;
}

def VLD3LNd8  : VLD3LN<0b0010, {?,?,?,0}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VLD3LNd16 : VLD3LN<0b0110, {?,?,0,0}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD3LNd32 : VLD3LN<0b1010, {?,0,0,0}, "32"> {
  let Inst{7}   = lane{0};
}

def VLD3LNd8Pseudo  : VLDQQLNPseudo<IIC_VLD3ln>;
def VLD3LNd16Pseudo : VLDQQLNPseudo<IIC_VLD3ln>;
def VLD3LNd32Pseudo : VLDQQLNPseudo<IIC_VLD3ln>;

// ...with double-spaced registers:
def VLD3LNq16 : VLD3LN<0b0110, {?,?,1,0}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD3LNq32 : VLD3LN<0b1010, {?,1,0,0}, "32"> {
  let Inst{7}   = lane{0};
}

def VLD3LNq16Pseudo : VLDQQQQLNPseudo<IIC_VLD3ln>;
def VLD3LNq32Pseudo : VLDQQQQLNPseudo<IIC_VLD3ln>;

// ...with address register writeback:
class VLD3LNWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b10, op11_8, op7_4,
          (outs DPR:$Vd, DPR:$dst2, DPR:$dst3, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$src1, DPR:$src2, DPR:$src3, nohash_imm:$lane),
          IIC_VLD3lnu, "vld3", Dt,
          "\\{$Vd[$lane], $dst2[$lane], $dst3[$lane]\\}, $Rn$Rm",
          "$src1 = $Vd, $src2 = $dst2, $src3 = $dst3, $Rn.addr = $wb",
          []>;

def VLD3LNd8_UPD  : VLD3LNWB<0b0010, {?,?,?,0}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VLD3LNd16_UPD : VLD3LNWB<0b0110, {?,?,0,0}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD3LNd32_UPD : VLD3LNWB<0b1010, {?,0,0,0}, "32"> {
  let Inst{7}   = lane{0};
}

def VLD3LNd8Pseudo_UPD  : VLDQQLNWBPseudo<IIC_VLD3lnu>;
def VLD3LNd16Pseudo_UPD : VLDQQLNWBPseudo<IIC_VLD3lnu>;
def VLD3LNd32Pseudo_UPD : VLDQQLNWBPseudo<IIC_VLD3lnu>;

def VLD3LNq16_UPD : VLD3LNWB<0b0110, {?,?,1,0}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD3LNq32_UPD : VLD3LNWB<0b1010, {?,1,0,0}, "32"> {
  let Inst{7}   = lane{0};
}

def VLD3LNq16Pseudo_UPD : VLDQQQQLNWBPseudo<IIC_VLD3lnu>;
def VLD3LNq32Pseudo_UPD : VLDQQQQLNWBPseudo<IIC_VLD3lnu>;

//   VLD4LN   : Vector Load (single 4-element structure to one lane)
class VLD4LN<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b10, op11_8, op7_4,
          (outs DPR:$Vd, DPR:$dst2, DPR:$dst3, DPR:$dst4),
          (ins addrmode6:$Rn, DPR:$src1, DPR:$src2, DPR:$src3, DPR:$src4,
          nohash_imm:$lane), IIC_VLD4ln, "vld4", Dt,
          "\\{$Vd[$lane], $dst2[$lane], $dst3[$lane], $dst4[$lane]\\}, $Rn",
          "$src1 = $Vd, $src2 = $dst2, $src3 = $dst3, $src4 = $dst4", []> {
  let Rm = 0b1111;
  let Inst{4}   = Rn{4};
}

def VLD4LNd8  : VLD4LN<0b0011, {?,?,?,?}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VLD4LNd16 : VLD4LN<0b0111, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD4LNd32 : VLD4LN<0b1011, {?,0,?,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{5} = Rn{5};
}

def VLD4LNd8Pseudo  : VLDQQLNPseudo<IIC_VLD4ln>;
def VLD4LNd16Pseudo : VLDQQLNPseudo<IIC_VLD4ln>;
def VLD4LNd32Pseudo : VLDQQLNPseudo<IIC_VLD4ln>;

// ...with double-spaced registers:
def VLD4LNq16 : VLD4LN<0b0111, {?,?,1,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD4LNq32 : VLD4LN<0b1011, {?,1,?,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{5} = Rn{5};
}

def VLD4LNq16Pseudo : VLDQQQQLNPseudo<IIC_VLD4ln>;
def VLD4LNq32Pseudo : VLDQQQQLNPseudo<IIC_VLD4ln>;

// ...with address register writeback:
class VLD4LNWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b10, op11_8, op7_4,
          (outs DPR:$Vd, DPR:$dst2, DPR:$dst3, DPR:$dst4, GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$src1, DPR:$src2, DPR:$src3, DPR:$src4, nohash_imm:$lane),
          IIC_VLD4ln, "vld4", Dt,
"\\{$Vd[$lane], $dst2[$lane], $dst3[$lane], $dst4[$lane]\\}, $Rn$Rm",
"$src1 = $Vd, $src2 = $dst2, $src3 = $dst3, $src4 = $dst4, $Rn.addr = $wb",
          []> {
  let Inst{4}   = Rn{4};
}

def VLD4LNd8_UPD  : VLD4LNWB<0b0011, {?,?,?,?}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VLD4LNd16_UPD : VLD4LNWB<0b0111, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD4LNd32_UPD : VLD4LNWB<0b1011, {?,0,?,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{5} = Rn{5};
}

def VLD4LNd8Pseudo_UPD  : VLDQQLNWBPseudo<IIC_VLD4lnu>;
def VLD4LNd16Pseudo_UPD : VLDQQLNWBPseudo<IIC_VLD4lnu>;
def VLD4LNd32Pseudo_UPD : VLDQQLNWBPseudo<IIC_VLD4lnu>;

def VLD4LNq16_UPD : VLD4LNWB<0b0111, {?,?,1,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VLD4LNq32_UPD : VLD4LNWB<0b1011, {?,1,?,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{5} = Rn{5};
}

def VLD4LNq16Pseudo_UPD : VLDQQQQLNWBPseudo<IIC_VLD4lnu>;
def VLD4LNq32Pseudo_UPD : VLDQQQQLNWBPseudo<IIC_VLD4lnu>;

//   VLD1DUP  : Vector Load (single element to all lanes)
//   VLD2DUP  : Vector Load (single 2-element structure to all lanes)
//   VLD3DUP  : Vector Load (single 3-element structure to all lanes)
//   VLD4DUP  : Vector Load (single 4-element structure to all lanes)
//   FIXME: Not yet implemented.
} // mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1

let mayStore = 1, neverHasSideEffects = 1, hasExtraSrcRegAllocReq = 1 in {

// Classes for VST* pseudo-instructions with multi-register operands.
// These are expanded to real instructions after register allocation.
class VSTQPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs), (ins addrmode6:$addr, QPR:$src), itin, "">;
class VSTQWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset, QPR:$src), itin,
                "$addr.addr = $wb">;
class VSTQQPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs), (ins addrmode6:$addr, QQPR:$src), itin, "">;
class VSTQQWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset, QQPR:$src), itin,
                "$addr.addr = $wb">;
class VSTQQQQWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset, QQQQPR:$src), itin,
                "$addr.addr = $wb">;

//   VST1     : Vector Store (multiple single elements)
class VST1D<bits<4> op7_4, string Dt>
  : NLdSt<0,0b00,0b0111,op7_4, (outs), (ins addrmode6:$Rn, DPR:$Vd),
          IIC_VST1, "vst1", Dt, "\\{$Vd\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{4} = Rn{4};
}
class VST1Q<bits<4> op7_4, string Dt>
  : NLdSt<0,0b00,0b1010,op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, DPR:$src2), IIC_VST1x2,
          "vst1", Dt, "\\{$Vd, $src2\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{5-4} = Rn{5-4};
}

def  VST1d8   : VST1D<{0,0,0,?}, "8">;
def  VST1d16  : VST1D<{0,1,0,?}, "16">;
def  VST1d32  : VST1D<{1,0,0,?}, "32">;
def  VST1d64  : VST1D<{1,1,0,?}, "64">;

def  VST1q8   : VST1Q<{0,0,?,?}, "8">;
def  VST1q16  : VST1Q<{0,1,?,?}, "16">;
def  VST1q32  : VST1Q<{1,0,?,?}, "32">;
def  VST1q64  : VST1Q<{1,1,?,?}, "64">;

def  VST1q8Pseudo  : VSTQPseudo<IIC_VST1x2>;
def  VST1q16Pseudo : VSTQPseudo<IIC_VST1x2>;
def  VST1q32Pseudo : VSTQPseudo<IIC_VST1x2>;
def  VST1q64Pseudo : VSTQPseudo<IIC_VST1x2>;

// ...with address register writeback:
class VST1DWB<bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, 0b0111, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm, DPR:$Vd), IIC_VST1u,
          "vst1", Dt, "\\{$Vd\\}, $Rn$Rm", "$Rn.addr = $wb", []> {
  let Inst{4} = Rn{4};
}
class VST1QWB<bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, 0b1010, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm, DPR:$Vd, DPR:$src2),
          IIC_VST1x2u, "vst1", Dt, "\\{$Vd, $src2\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{5-4} = Rn{5-4};
}

def VST1d8_UPD  : VST1DWB<{0,0,0,?}, "8">;
def VST1d16_UPD : VST1DWB<{0,1,0,?}, "16">;
def VST1d32_UPD : VST1DWB<{1,0,0,?}, "32">;
def VST1d64_UPD : VST1DWB<{1,1,0,?}, "64">;

def VST1q8_UPD  : VST1QWB<{0,0,?,?}, "8">;
def VST1q16_UPD : VST1QWB<{0,1,?,?}, "16">;
def VST1q32_UPD : VST1QWB<{1,0,?,?}, "32">;
def VST1q64_UPD : VST1QWB<{1,1,?,?}, "64">;

def VST1q8Pseudo_UPD  : VSTQWBPseudo<IIC_VST1x2u>;
def VST1q16Pseudo_UPD : VSTQWBPseudo<IIC_VST1x2u>;
def VST1q32Pseudo_UPD : VSTQWBPseudo<IIC_VST1x2u>;
def VST1q64Pseudo_UPD : VSTQWBPseudo<IIC_VST1x2u>;

// ...with 3 registers (some of these are only for the disassembler):
class VST1D3<bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, 0b0110, op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, DPR:$src2, DPR:$src3),
          IIC_VST1x3, "vst1", Dt, "\\{$Vd, $src2, $src3\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{4} = Rn{4};
}
class VST1D3WB<bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, 0b0110, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$Vd, DPR:$src2, DPR:$src3),
          IIC_VST1x3u, "vst1", Dt, "\\{$Vd, $src2, $src3\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{4} = Rn{4};
}

def VST1d8T      : VST1D3<{0,0,0,?}, "8">;
def VST1d16T     : VST1D3<{0,1,0,?}, "16">;
def VST1d32T     : VST1D3<{1,0,0,?}, "32">;
def VST1d64T     : VST1D3<{1,1,0,?}, "64">;

def VST1d8T_UPD  : VST1D3WB<{0,0,0,?}, "8">;
def VST1d16T_UPD : VST1D3WB<{0,1,0,?}, "16">;
def VST1d32T_UPD : VST1D3WB<{1,0,0,?}, "32">;
def VST1d64T_UPD : VST1D3WB<{1,1,0,?}, "64">;

def VST1d64TPseudo     : VSTQQPseudo<IIC_VST1x3>;
def VST1d64TPseudo_UPD : VSTQQWBPseudo<IIC_VST1x3u>;

// ...with 4 registers (some of these are only for the disassembler):
class VST1D4<bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, 0b0010, op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, DPR:$src2, DPR:$src3, DPR:$src4),
          IIC_VST1x4, "vst1", Dt, "\\{$Vd, $src2, $src3, $src4\\}, $Rn", "",
          []> {
  let Rm = 0b1111;
  let Inst{5-4} = Rn{5-4};
}
class VST1D4WB<bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, 0b0010, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$Vd, DPR:$src2, DPR:$src3, DPR:$src4), IIC_VST1x4u,
          "vst1", Dt, "\\{$Vd, $src2, $src3, $src4\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{5-4} = Rn{5-4};
}

def VST1d8Q      : VST1D4<{0,0,?,?}, "8">;
def VST1d16Q     : VST1D4<{0,1,?,?}, "16">;
def VST1d32Q     : VST1D4<{1,0,?,?}, "32">;
def VST1d64Q     : VST1D4<{1,1,?,?}, "64">;

def VST1d8Q_UPD  : VST1D4WB<{0,0,?,?}, "8">;
def VST1d16Q_UPD : VST1D4WB<{0,1,?,?}, "16">;
def VST1d32Q_UPD : VST1D4WB<{1,0,?,?}, "32">;
def VST1d64Q_UPD : VST1D4WB<{1,1,?,?}, "64">;

def VST1d64QPseudo     : VSTQQPseudo<IIC_VST1x4>;
def VST1d64QPseudo_UPD : VSTQQWBPseudo<IIC_VST1x4u>;

//   VST2     : Vector Store (multiple 2-element structures)
class VST2D<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, op11_8, op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, DPR:$src2),
          IIC_VST2, "vst2", Dt, "\\{$Vd, $src2\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{5-4} = Rn{5-4};
}
class VST2Q<bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, 0b0011, op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, DPR:$src2, DPR:$src3, DPR:$src4),
          IIC_VST2x2, "vst2", Dt, "\\{$Vd, $src2, $src3, $src4\\}, $Rn",
          "", []> {
  let Rm = 0b1111;
  let Inst{5-4} = Rn{5-4};
}

def  VST2d8   : VST2D<0b1000, {0,0,?,?}, "8">;
def  VST2d16  : VST2D<0b1000, {0,1,?,?}, "16">;
def  VST2d32  : VST2D<0b1000, {1,0,?,?}, "32">;

def  VST2q8   : VST2Q<{0,0,?,?}, "8">;
def  VST2q16  : VST2Q<{0,1,?,?}, "16">;
def  VST2q32  : VST2Q<{1,0,?,?}, "32">;

def  VST2d8Pseudo  : VSTQPseudo<IIC_VST2>;
def  VST2d16Pseudo : VSTQPseudo<IIC_VST2>;
def  VST2d32Pseudo : VSTQPseudo<IIC_VST2>;

def  VST2q8Pseudo  : VSTQQPseudo<IIC_VST2x2>;
def  VST2q16Pseudo : VSTQQPseudo<IIC_VST2x2>;
def  VST2q32Pseudo : VSTQQPseudo<IIC_VST2x2>;

// ...with address register writeback:
class VST2DWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, op11_8, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm, DPR:$Vd, DPR:$src2),
          IIC_VST2u, "vst2", Dt, "\\{$Vd, $src2\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{5-4} = Rn{5-4};
}
class VST2QWB<bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, 0b0011, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$Vd, DPR:$src2, DPR:$src3, DPR:$src4), IIC_VST2x2u,
          "vst2", Dt, "\\{$Vd, $src2, $src3, $src4\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{5-4} = Rn{5-4};
}

def VST2d8_UPD  : VST2DWB<0b1000, {0,0,?,?}, "8">;
def VST2d16_UPD : VST2DWB<0b1000, {0,1,?,?}, "16">;
def VST2d32_UPD : VST2DWB<0b1000, {1,0,?,?}, "32">;

def VST2q8_UPD  : VST2QWB<{0,0,?,?}, "8">;
def VST2q16_UPD : VST2QWB<{0,1,?,?}, "16">;
def VST2q32_UPD : VST2QWB<{1,0,?,?}, "32">;

def VST2d8Pseudo_UPD  : VSTQWBPseudo<IIC_VST2u>;
def VST2d16Pseudo_UPD : VSTQWBPseudo<IIC_VST2u>;
def VST2d32Pseudo_UPD : VSTQWBPseudo<IIC_VST2u>;

def VST2q8Pseudo_UPD  : VSTQQWBPseudo<IIC_VST2x2u>;
def VST2q16Pseudo_UPD : VSTQQWBPseudo<IIC_VST2x2u>;
def VST2q32Pseudo_UPD : VSTQQWBPseudo<IIC_VST2x2u>;

// ...with double-spaced registers (for disassembly only):
def VST2b8      : VST2D<0b1001, {0,0,?,?}, "8">;
def VST2b16     : VST2D<0b1001, {0,1,?,?}, "16">;
def VST2b32     : VST2D<0b1001, {1,0,?,?}, "32">;
def VST2b8_UPD  : VST2DWB<0b1001, {0,0,?,?}, "8">;
def VST2b16_UPD : VST2DWB<0b1001, {0,1,?,?}, "16">;
def VST2b32_UPD : VST2DWB<0b1001, {1,0,?,?}, "32">;

//   VST3     : Vector Store (multiple 3-element structures)
class VST3D<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, op11_8, op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, DPR:$src2, DPR:$src3), IIC_VST3,
          "vst3", Dt, "\\{$Vd, $src2, $src3\\}, $Rn", "", []> {
  let Rm = 0b1111;
  let Inst{4} = Rn{4};
}

def  VST3d8   : VST3D<0b0100, {0,0,0,?}, "8">;
def  VST3d16  : VST3D<0b0100, {0,1,0,?}, "16">;
def  VST3d32  : VST3D<0b0100, {1,0,0,?}, "32">;

def  VST3d8Pseudo  : VSTQQPseudo<IIC_VST3>;
def  VST3d16Pseudo : VSTQQPseudo<IIC_VST3>;
def  VST3d32Pseudo : VSTQQPseudo<IIC_VST3>;

// ...with address register writeback:
class VST3DWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, op11_8, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$Vd, DPR:$src2, DPR:$src3), IIC_VST3u,
          "vst3", Dt, "\\{$Vd, $src2, $src3\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{4} = Rn{4};
}

def VST3d8_UPD  : VST3DWB<0b0100, {0,0,0,?}, "8">;
def VST3d16_UPD : VST3DWB<0b0100, {0,1,0,?}, "16">;
def VST3d32_UPD : VST3DWB<0b0100, {1,0,0,?}, "32">;

def VST3d8Pseudo_UPD  : VSTQQWBPseudo<IIC_VST3u>;
def VST3d16Pseudo_UPD : VSTQQWBPseudo<IIC_VST3u>;
def VST3d32Pseudo_UPD : VSTQQWBPseudo<IIC_VST3u>;

// ...with double-spaced registers (non-updating versions for disassembly only):
def VST3q8      : VST3D<0b0101, {0,0,0,?}, "8">;
def VST3q16     : VST3D<0b0101, {0,1,0,?}, "16">;
def VST3q32     : VST3D<0b0101, {1,0,0,?}, "32">;
def VST3q8_UPD  : VST3DWB<0b0101, {0,0,0,?}, "8">;
def VST3q16_UPD : VST3DWB<0b0101, {0,1,0,?}, "16">;
def VST3q32_UPD : VST3DWB<0b0101, {1,0,0,?}, "32">;

def VST3q8Pseudo_UPD  : VSTQQQQWBPseudo<IIC_VST3u>;
def VST3q16Pseudo_UPD : VSTQQQQWBPseudo<IIC_VST3u>;
def VST3q32Pseudo_UPD : VSTQQQQWBPseudo<IIC_VST3u>;

// ...alternate versions to be allocated odd register numbers:
def VST3q8oddPseudo_UPD  : VSTQQQQWBPseudo<IIC_VST3u>;
def VST3q16oddPseudo_UPD : VSTQQQQWBPseudo<IIC_VST3u>;
def VST3q32oddPseudo_UPD : VSTQQQQWBPseudo<IIC_VST3u>;

//   VST4     : Vector Store (multiple 4-element structures)
class VST4D<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, op11_8, op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, DPR:$src2, DPR:$src3, DPR:$src4),
          IIC_VST4, "vst4", Dt, "\\{$Vd, $src2, $src3, $src4\\}, $Rn",
          "", []> {
  let Rm = 0b1111;
  let Inst{5-4} = Rn{5-4};
}

def  VST4d8   : VST4D<0b0000, {0,0,?,?}, "8">;
def  VST4d16  : VST4D<0b0000, {0,1,?,?}, "16">;
def  VST4d32  : VST4D<0b0000, {1,0,?,?}, "32">;

def  VST4d8Pseudo  : VSTQQPseudo<IIC_VST4>;
def  VST4d16Pseudo : VSTQQPseudo<IIC_VST4>;
def  VST4d32Pseudo : VSTQQPseudo<IIC_VST4>;

// ...with address register writeback:
class VST4DWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdSt<0, 0b00, op11_8, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$Vd, DPR:$src2, DPR:$src3, DPR:$src4), IIC_VST4u,
           "vst4", Dt, "\\{$Vd, $src2, $src3, $src4\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{5-4} = Rn{5-4};
}

def VST4d8_UPD  : VST4DWB<0b0000, {0,0,?,?}, "8">;
def VST4d16_UPD : VST4DWB<0b0000, {0,1,?,?}, "16">;
def VST4d32_UPD : VST4DWB<0b0000, {1,0,?,?}, "32">;

def VST4d8Pseudo_UPD  : VSTQQWBPseudo<IIC_VST4u>;
def VST4d16Pseudo_UPD : VSTQQWBPseudo<IIC_VST4u>;
def VST4d32Pseudo_UPD : VSTQQWBPseudo<IIC_VST4u>;

// ...with double-spaced registers (non-updating versions for disassembly only):
def VST4q8      : VST4D<0b0001, {0,0,?,?}, "8">;
def VST4q16     : VST4D<0b0001, {0,1,?,?}, "16">;
def VST4q32     : VST4D<0b0001, {1,0,?,?}, "32">;
def VST4q8_UPD  : VST4DWB<0b0001, {0,0,?,?}, "8">;
def VST4q16_UPD : VST4DWB<0b0001, {0,1,?,?}, "16">;
def VST4q32_UPD : VST4DWB<0b0001, {1,0,?,?}, "32">;

def VST4q8Pseudo_UPD  : VSTQQQQWBPseudo<IIC_VST4u>;
def VST4q16Pseudo_UPD : VSTQQQQWBPseudo<IIC_VST4u>;
def VST4q32Pseudo_UPD : VSTQQQQWBPseudo<IIC_VST4u>;

// ...alternate versions to be allocated odd register numbers:
def VST4q8oddPseudo_UPD  : VSTQQQQWBPseudo<IIC_VST4u>;
def VST4q16oddPseudo_UPD : VSTQQQQWBPseudo<IIC_VST4u>;
def VST4q32oddPseudo_UPD : VSTQQQQWBPseudo<IIC_VST4u>;

} // mayStore = 1, neverHasSideEffects = 1, hasExtraSrcRegAllocReq = 1

// Classes for VST*LN pseudo-instructions with multi-register operands.
// These are expanded to real instructions after register allocation.
class VSTQLNPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs), (ins addrmode6:$addr, QPR:$src, nohash_imm:$lane),
                itin, "">;
class VSTQLNWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset, QPR:$src,
                 nohash_imm:$lane), itin, "$addr.addr = $wb">;
class VSTQQLNPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs), (ins addrmode6:$addr, QQPR:$src, nohash_imm:$lane),
                itin, "">;
class VSTQQLNWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset, QQPR:$src,
                 nohash_imm:$lane), itin, "$addr.addr = $wb">;
class VSTQQQQLNPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs), (ins addrmode6:$addr, QQQQPR:$src, nohash_imm:$lane),
                itin, "">;
class VSTQQQQLNWBPseudo<InstrItinClass itin>
  : PseudoNLdSt<(outs GPR:$wb),
                (ins addrmode6:$addr, am6offset:$offset, QQQQPR:$src,
                 nohash_imm:$lane), itin, "$addr.addr = $wb">;

//   VST1LN   : Vector Store (single element from one lane)
class VST1LN<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b00, op11_8, op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, nohash_imm:$lane),
          IIC_VST1ln, "vst1", Dt, "\\{$Vd[$lane]\\}, $Rn", "", []> {
  let Rm = 0b1111;
}

def VST1LNd8  : VST1LN<0b0000, {?,?,?,0}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VST1LNd16 : VST1LN<0b0100, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
  let Inst{4}   = Rn{5};
}
def VST1LNd32 : VST1LN<0b1000, {?,0,?,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{5-4} = Rn{5-4};
}

def VST1LNq8Pseudo  : VSTQLNPseudo<IIC_VST1ln>;
def VST1LNq16Pseudo : VSTQLNPseudo<IIC_VST1ln>;
def VST1LNq32Pseudo : VSTQLNPseudo<IIC_VST1ln>;

let mayStore = 1, neverHasSideEffects = 1, hasExtraSrcRegAllocReq = 1 in {

// ...with address register writeback:
class VST1LNWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b00, op11_8, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$Vd, nohash_imm:$lane), IIC_VST1lnu, "vst1", Dt,
          "\\{$Vd[$lane]\\}, $Rn$Rm",
          "$Rn.addr = $wb", []>;

def VST1LNd8_UPD  : VST1LNWB<0b0000, {?,?,?,0}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VST1LNd16_UPD : VST1LNWB<0b0100, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
  let Inst{4}   = Rn{5};
}
def VST1LNd32_UPD : VST1LNWB<0b1000, {?,0,?,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{5-4} = Rn{5-4};
}

def VST1LNq8Pseudo_UPD  : VSTQLNWBPseudo<IIC_VST1lnu>;
def VST1LNq16Pseudo_UPD : VSTQLNWBPseudo<IIC_VST1lnu>;
def VST1LNq32Pseudo_UPD : VSTQLNWBPseudo<IIC_VST1lnu>;

//   VST2LN   : Vector Store (single 2-element structure from one lane)
class VST2LN<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b00, op11_8, op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, DPR:$src2, nohash_imm:$lane),
          IIC_VST2ln, "vst2", Dt, "\\{$Vd[$lane], $src2[$lane]\\}, $Rn",
          "", []> {
  let Rm = 0b1111;
  let Inst{4}   = Rn{4};
}

def VST2LNd8  : VST2LN<0b0001, {?,?,?,?}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VST2LNd16 : VST2LN<0b0101, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST2LNd32 : VST2LN<0b1001, {?,0,0,?}, "32"> {
  let Inst{7}   = lane{0};
}

def VST2LNd8Pseudo  : VSTQLNPseudo<IIC_VST2ln>;
def VST2LNd16Pseudo : VSTQLNPseudo<IIC_VST2ln>;
def VST2LNd32Pseudo : VSTQLNPseudo<IIC_VST2ln>;

// ...with double-spaced registers:
def VST2LNq16 : VST2LN<0b0101, {?,?,1,?}, "16"> {
  let Inst{7-6} = lane{1-0};
  let Inst{4}   = Rn{4};
}
def VST2LNq32 : VST2LN<0b1001, {?,1,0,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{4}   = Rn{4};
}

def VST2LNq16Pseudo : VSTQQLNPseudo<IIC_VST2ln>;
def VST2LNq32Pseudo : VSTQQLNPseudo<IIC_VST2ln>;

// ...with address register writeback:
class VST2LNWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b00, op11_8, op7_4, (outs GPR:$wb),
          (ins addrmode6:$addr, am6offset:$offset,
           DPR:$src1, DPR:$src2, nohash_imm:$lane), IIC_VST2lnu, "vst2", Dt,
          "\\{$src1[$lane], $src2[$lane]\\}, $addr$offset",
          "$addr.addr = $wb", []> {
  let Inst{4}   = Rn{4};
}

def VST2LNd8_UPD  : VST2LNWB<0b0001, {?,?,?,?}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VST2LNd16_UPD : VST2LNWB<0b0101, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST2LNd32_UPD : VST2LNWB<0b1001, {?,0,0,?}, "32"> {
  let Inst{7}   = lane{0};
}

def VST2LNd8Pseudo_UPD  : VSTQLNWBPseudo<IIC_VST2lnu>;
def VST2LNd16Pseudo_UPD : VSTQLNWBPseudo<IIC_VST2lnu>;
def VST2LNd32Pseudo_UPD : VSTQLNWBPseudo<IIC_VST2lnu>;

def VST2LNq16_UPD : VST2LNWB<0b0101, {?,?,1,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST2LNq32_UPD : VST2LNWB<0b1001, {?,1,0,?}, "32"> {
  let Inst{7}   = lane{0};
}

def VST2LNq16Pseudo_UPD : VSTQQLNWBPseudo<IIC_VST2lnu>;
def VST2LNq32Pseudo_UPD : VSTQQLNWBPseudo<IIC_VST2lnu>;

//   VST3LN   : Vector Store (single 3-element structure from one lane)
class VST3LN<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b00, op11_8, op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, DPR:$src2, DPR:$src3,
           nohash_imm:$lane), IIC_VST3ln, "vst3", Dt,
          "\\{$Vd[$lane], $src2[$lane], $src3[$lane]\\}, $Rn", "", []> {
  let Rm = 0b1111;
}

def VST3LNd8  : VST3LN<0b0010, {?,?,?,0}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VST3LNd16 : VST3LN<0b0110, {?,?,0,0}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST3LNd32 : VST3LN<0b1010, {?,0,0,0}, "32"> {
  let Inst{7}   = lane{0};
}

def VST3LNd8Pseudo  : VSTQQLNPseudo<IIC_VST3ln>;
def VST3LNd16Pseudo : VSTQQLNPseudo<IIC_VST3ln>;
def VST3LNd32Pseudo : VSTQQLNPseudo<IIC_VST3ln>;

// ...with double-spaced registers:
def VST3LNq16 : VST3LN<0b0110, {?,?,1,0}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST3LNq32 : VST3LN<0b1010, {?,1,0,0}, "32"> {
  let Inst{7}   = lane{0};
}

def VST3LNq16Pseudo : VSTQQQQLNPseudo<IIC_VST3ln>;
def VST3LNq32Pseudo : VSTQQQQLNPseudo<IIC_VST3ln>;

// ...with address register writeback:
class VST3LNWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b00, op11_8, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$Vd, DPR:$src2, DPR:$src3, nohash_imm:$lane),
          IIC_VST3lnu, "vst3", Dt,
          "\\{$Vd[$lane], $src2[$lane], $src3[$lane]\\}, $Rn$Rm",
          "$Rn.addr = $wb", []>;

def VST3LNd8_UPD  : VST3LNWB<0b0010, {?,?,?,0}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VST3LNd16_UPD : VST3LNWB<0b0110, {?,?,0,0}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST3LNd32_UPD : VST3LNWB<0b1010, {?,0,0,0}, "32"> {
  let Inst{7}   = lane{0};
}

def VST3LNd8Pseudo_UPD  : VSTQQLNWBPseudo<IIC_VST3lnu>;
def VST3LNd16Pseudo_UPD : VSTQQLNWBPseudo<IIC_VST3lnu>;
def VST3LNd32Pseudo_UPD : VSTQQLNWBPseudo<IIC_VST3lnu>;

def VST3LNq16_UPD : VST3LNWB<0b0110, {?,?,1,0}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST3LNq32_UPD : VST3LNWB<0b1010, {?,1,0,0}, "32"> {
  let Inst{7}   = lane{0};
}

def VST3LNq16Pseudo_UPD : VSTQQQQLNWBPseudo<IIC_VST3lnu>;
def VST3LNq32Pseudo_UPD : VSTQQQQLNWBPseudo<IIC_VST3lnu>;

//   VST4LN   : Vector Store (single 4-element structure from one lane)
class VST4LN<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b00, op11_8, op7_4, (outs),
          (ins addrmode6:$Rn, DPR:$Vd, DPR:$src2, DPR:$src3, DPR:$src4,
           nohash_imm:$lane), IIC_VST4ln, "vst4", Dt,
          "\\{$Vd[$lane], $src2[$lane], $src3[$lane], $src4[$lane]\\}, $Rn",
          "", []> {
  let Rm = 0b1111;
  let Inst{4} = Rn{4};
}

def VST4LNd8  : VST4LN<0b0011, {?,?,?,?}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VST4LNd16 : VST4LN<0b0111, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST4LNd32 : VST4LN<0b1011, {?,0,?,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{5} = Rn{5};
}

def VST4LNd8Pseudo  : VSTQQLNPseudo<IIC_VST4ln>;
def VST4LNd16Pseudo : VSTQQLNPseudo<IIC_VST4ln>;
def VST4LNd32Pseudo : VSTQQLNPseudo<IIC_VST4ln>;

// ...with double-spaced registers:
def VST4LNq16 : VST4LN<0b0111, {?,?,1,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST4LNq32 : VST4LN<0b1011, {?,1,?,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{5} = Rn{5};
}

def VST4LNq16Pseudo : VSTQQQQLNPseudo<IIC_VST4ln>;
def VST4LNq32Pseudo : VSTQQQQLNPseudo<IIC_VST4ln>;

// ...with address register writeback:
class VST4LNWB<bits<4> op11_8, bits<4> op7_4, string Dt>
  : NLdStLn<1, 0b00, op11_8, op7_4, (outs GPR:$wb),
          (ins addrmode6:$Rn, am6offset:$Rm,
           DPR:$Vd, DPR:$src2, DPR:$src3, DPR:$src4, nohash_imm:$lane),
          IIC_VST4lnu, "vst4", Dt,
  "\\{$Vd[$lane], $src2[$lane], $src3[$lane], $src4[$lane]\\}, $Rn$Rm",
          "$Rn.addr = $wb", []> {
  let Inst{4} = Rn{4};
}

def VST4LNd8_UPD  : VST4LNWB<0b0011, {?,?,?,?}, "8"> {
  let Inst{7-5} = lane{2-0};
}
def VST4LNd16_UPD : VST4LNWB<0b0111, {?,?,0,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST4LNd32_UPD : VST4LNWB<0b1011, {?,0,?,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{5} = Rn{5};
}

def VST4LNd8Pseudo_UPD  : VSTQQLNWBPseudo<IIC_VST4lnu>;
def VST4LNd16Pseudo_UPD : VSTQQLNWBPseudo<IIC_VST4lnu>;
def VST4LNd32Pseudo_UPD : VSTQQLNWBPseudo<IIC_VST4lnu>;

def VST4LNq16_UPD : VST4LNWB<0b0111, {?,?,1,?}, "16"> {
  let Inst{7-6} = lane{1-0};
}
def VST4LNq32_UPD : VST4LNWB<0b1011, {?,1,?,?}, "32"> {
  let Inst{7}   = lane{0};
  let Inst{5} = Rn{5};
}

def VST4LNq16Pseudo_UPD : VSTQQQQLNWBPseudo<IIC_VST4lnu>;
def VST4LNq32Pseudo_UPD : VSTQQQQLNWBPseudo<IIC_VST4lnu>;

} // mayStore = 1, neverHasSideEffects = 1, hasExtraSrcRegAllocReq = 1


//===----------------------------------------------------------------------===//
// NEON pattern fragments
//===----------------------------------------------------------------------===//

// Extract D sub-registers of Q registers.
def DSubReg_i8_reg  : SDNodeXForm<imm, [{
  assert(ARM::dsub_7 == ARM::dsub_0+7 && "Unexpected subreg numbering");
  return CurDAG->getTargetConstant(ARM::dsub_0 + N->getZExtValue()/8, MVT::i32);
}]>;
def DSubReg_i16_reg : SDNodeXForm<imm, [{
  assert(ARM::dsub_7 == ARM::dsub_0+7 && "Unexpected subreg numbering");
  return CurDAG->getTargetConstant(ARM::dsub_0 + N->getZExtValue()/4, MVT::i32);
}]>;
def DSubReg_i32_reg : SDNodeXForm<imm, [{
  assert(ARM::dsub_7 == ARM::dsub_0+7 && "Unexpected subreg numbering");
  return CurDAG->getTargetConstant(ARM::dsub_0 + N->getZExtValue()/2, MVT::i32);
}]>;
def DSubReg_f64_reg : SDNodeXForm<imm, [{
  assert(ARM::dsub_7 == ARM::dsub_0+7 && "Unexpected subreg numbering");
  return CurDAG->getTargetConstant(ARM::dsub_0 + N->getZExtValue(), MVT::i32);
}]>;

// Extract S sub-registers of Q/D registers.
def SSubReg_f32_reg : SDNodeXForm<imm, [{
  assert(ARM::ssub_3 == ARM::ssub_0+3 && "Unexpected subreg numbering");
  return CurDAG->getTargetConstant(ARM::ssub_0 + N->getZExtValue(), MVT::i32);
}]>;

// Translate lane numbers from Q registers to D subregs.
def SubReg_i8_lane  : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 7, MVT::i32);
}]>;
def SubReg_i16_lane : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 3, MVT::i32);
}]>;
def SubReg_i32_lane : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 1, MVT::i32);
}]>;

//===----------------------------------------------------------------------===//
// Instruction Classes
//===----------------------------------------------------------------------===//

// Basic 2-register operations: single-, double- and quad-register.
class N2VS<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
           bits<2> op17_16, bits<5> op11_7, bit op4, string OpcodeStr,
           string Dt, ValueType ResTy, ValueType OpTy, SDNode OpNode>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4,
        (outs DPR_VFP2:$dst), (ins DPR_VFP2:$src),
        IIC_VUNAD, OpcodeStr, Dt, "$dst, $src", "", []>;
class N2VD<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
           bits<2> op17_16, bits<5> op11_7, bit op4, string OpcodeStr,
           string Dt, ValueType ResTy, ValueType OpTy, SDNode OpNode>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4, (outs DPR:$dst),
        (ins DPR:$src), IIC_VUNAD, OpcodeStr, Dt,"$dst, $src", "",
        [(set DPR:$dst, (ResTy (OpNode (OpTy DPR:$src))))]>;
class N2VQ<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
           bits<2> op17_16, bits<5> op11_7, bit op4, string OpcodeStr,
           string Dt, ValueType ResTy, ValueType OpTy, SDNode OpNode>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 1, op4, (outs QPR:$dst),
        (ins QPR:$src), IIC_VUNAQ, OpcodeStr, Dt,"$dst, $src", "",
        [(set QPR:$dst, (ResTy (OpNode (OpTy QPR:$src))))]>;

// Basic 2-register intrinsics, both double- and quad-register.
class N2VDInt<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
              bits<2> op17_16, bits<5> op11_7, bit op4,
              InstrItinClass itin, string OpcodeStr, string Dt,
              ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4, (outs DPR:$dst),
        (ins DPR:$src), itin, OpcodeStr, Dt, "$dst, $src", "",
        [(set DPR:$dst, (ResTy (IntOp (OpTy DPR:$src))))]>;
class N2VQInt<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
              bits<2> op17_16, bits<5> op11_7, bit op4,
              InstrItinClass itin, string OpcodeStr, string Dt,
              ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 1, op4, (outs QPR:$dst),
        (ins QPR:$src), itin, OpcodeStr, Dt, "$dst, $src", "",
        [(set QPR:$dst, (ResTy (IntOp (OpTy QPR:$src))))]>;

// Narrow 2-register operations.
class N2VN<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
           bits<2> op17_16, bits<5> op11_7, bit op6, bit op4,
           InstrItinClass itin, string OpcodeStr, string Dt,
           ValueType TyD, ValueType TyQ, SDNode OpNode>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, op6, op4, (outs DPR:$dst),
        (ins QPR:$src), itin, OpcodeStr, Dt, "$dst, $src", "",
        [(set DPR:$dst, (TyD (OpNode (TyQ QPR:$src))))]>;

// Narrow 2-register intrinsics.
class N2VNInt<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
              bits<2> op17_16, bits<5> op11_7, bit op6, bit op4,
              InstrItinClass itin, string OpcodeStr, string Dt,
              ValueType TyD, ValueType TyQ, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, op6, op4, (outs DPR:$dst),
        (ins QPR:$src), itin, OpcodeStr, Dt, "$dst, $src", "",
        [(set DPR:$dst, (TyD (IntOp (TyQ QPR:$src))))]>;

// Long 2-register operations (currently only used for VMOVL).
class N2VL<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
           bits<2> op17_16, bits<5> op11_7, bit op6, bit op4,
           InstrItinClass itin, string OpcodeStr, string Dt,
           ValueType TyQ, ValueType TyD, SDNode OpNode>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, op6, op4, (outs QPR:$dst),
        (ins DPR:$src), itin, OpcodeStr, Dt, "$dst, $src", "",
        [(set QPR:$dst, (TyQ (OpNode (TyD DPR:$src))))]>;

// 2-register shuffles (VTRN/VZIP/VUZP), both double- and quad-register.
class N2VDShuffle<bits<2> op19_18, bits<5> op11_7, string OpcodeStr, string Dt>
  : N2V<0b11, 0b11, op19_18, 0b10, op11_7, 0, 0, (outs DPR:$dst1, DPR:$dst2),
        (ins DPR:$src1, DPR:$src2), IIC_VPERMD, 
        OpcodeStr, Dt, "$dst1, $dst2",
        "$src1 = $dst1, $src2 = $dst2", []>;
class N2VQShuffle<bits<2> op19_18, bits<5> op11_7,
                  InstrItinClass itin, string OpcodeStr, string Dt>
  : N2V<0b11, 0b11, op19_18, 0b10, op11_7, 1, 0, (outs QPR:$dst1, QPR:$dst2),
        (ins QPR:$src1, QPR:$src2), itin, OpcodeStr, Dt, "$dst1, $dst2",
        "$src1 = $dst1, $src2 = $dst2", []>;

// Basic 3-register operations: single-, double- and quad-register.
class N3VS<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
           string OpcodeStr, string Dt, ValueType ResTy, ValueType OpTy,
           SDNode OpNode, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR_VFP2:$dst), (ins DPR_VFP2:$src1, DPR_VFP2:$src2), N3RegFrm,
        IIC_VBIND, OpcodeStr, Dt, "$dst, $src1, $src2", "", []> {
  let isCommutable = Commutable;
}

class N3VD<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
           InstrItinClass itin, string OpcodeStr, string Dt,
           ValueType ResTy, ValueType OpTy, SDNode OpNode, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$Vd), (ins DPR:$Vn, DPR:$Vm), N3RegFrm, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm", "",
        [(set DPR:$Vd, (ResTy (OpNode (OpTy DPR:$Vn), (OpTy DPR:$Vm))))]> {
  let isCommutable = Commutable;
}
// Same as N3VD but no data type.
class N3VDX<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
           InstrItinClass itin, string OpcodeStr,
           ValueType ResTy, ValueType OpTy,
           SDNode OpNode, bit Commutable>
  : N3VX<op24, op23, op21_20, op11_8, 0, op4,
         (outs DPR:$dst), (ins DPR:$src1, DPR:$src2), N3RegFrm, itin, 
         OpcodeStr, "$dst, $src1, $src2", "",
         [(set DPR:$dst, (ResTy (OpNode (OpTy DPR:$src1), (OpTy DPR:$src2))))]>{
  let isCommutable = Commutable;
}

class N3VDSL<bits<2> op21_20, bits<4> op11_8, 
             InstrItinClass itin, string OpcodeStr, string Dt,
             ValueType Ty, SDNode ShOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst), (ins DPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane),
        NVMulSLFrm, itin, OpcodeStr, Dt, "$dst, $src1, $src2[$lane]", "",
        [(set (Ty DPR:$dst),
              (Ty (ShOp (Ty DPR:$src1),
                        (Ty (NEONvduplane (Ty DPR_VFP2:$src2),imm:$lane)))))]> {
  let isCommutable = 0;
}
class N3VDSL16<bits<2> op21_20, bits<4> op11_8, 
               string OpcodeStr, string Dt, ValueType Ty, SDNode ShOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst), (ins DPR:$src1, DPR_8:$src2, nohash_imm:$lane),
        NVMulSLFrm, IIC_VMULi16D, OpcodeStr, Dt,"$dst, $src1, $src2[$lane]","",
        [(set (Ty DPR:$dst),
              (Ty (ShOp (Ty DPR:$src1),
                        (Ty (NEONvduplane (Ty DPR_8:$src2), imm:$lane)))))]> {
  let isCommutable = 0;
}

class N3VQ<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
           InstrItinClass itin, string OpcodeStr, string Dt,
           ValueType ResTy, ValueType OpTy, SDNode OpNode, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 1, op4,
        (outs QPR:$Qd), (ins QPR:$Qn, QPR:$Qm), N3RegFrm, itin, 
        OpcodeStr, Dt, "$Qd, $Qn, $Qm", "",
        [(set QPR:$Qd, (ResTy (OpNode (OpTy QPR:$Qn), (OpTy QPR:$Qm))))]> {
  let isCommutable = Commutable;
}
class N3VQX<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
           InstrItinClass itin, string OpcodeStr,
           ValueType ResTy, ValueType OpTy, SDNode OpNode, bit Commutable>
  : N3VX<op24, op23, op21_20, op11_8, 1, op4,
         (outs QPR:$dst), (ins QPR:$src1, QPR:$src2), N3RegFrm, itin, 
         OpcodeStr, "$dst, $src1, $src2", "",
         [(set QPR:$dst, (ResTy (OpNode (OpTy QPR:$src1), (OpTy QPR:$src2))))]>{
  let isCommutable = Commutable;
}
class N3VQSL<bits<2> op21_20, bits<4> op11_8, 
             InstrItinClass itin, string OpcodeStr, string Dt,
             ValueType ResTy, ValueType OpTy, SDNode ShOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins QPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane),
        NVMulSLFrm, itin, OpcodeStr, Dt, "$dst, $src1, $src2[$lane]", "",
        [(set (ResTy QPR:$dst),
              (ResTy (ShOp (ResTy QPR:$src1),
                           (ResTy (NEONvduplane (OpTy DPR_VFP2:$src2),
                                                imm:$lane)))))]> {
  let isCommutable = 0;
}
class N3VQSL16<bits<2> op21_20, bits<4> op11_8, string OpcodeStr, string Dt,
               ValueType ResTy, ValueType OpTy, SDNode ShOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins QPR:$src1, DPR_8:$src2, nohash_imm:$lane),
        NVMulSLFrm, IIC_VMULi16Q, OpcodeStr, Dt,"$dst, $src1, $src2[$lane]","",
        [(set (ResTy QPR:$dst),
              (ResTy (ShOp (ResTy QPR:$src1),
                           (ResTy (NEONvduplane (OpTy DPR_8:$src2),
                                                imm:$lane)))))]> {
  let isCommutable = 0;
}

// Basic 3-register intrinsics, both double- and quad-register.
class N3VDInt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              Format f, InstrItinClass itin, string OpcodeStr, string Dt,
              ValueType ResTy, ValueType OpTy, Intrinsic IntOp, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$Vd), (ins DPR:$Vn, DPR:$Vm), f, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm", "",
        [(set DPR:$Vd, (ResTy (IntOp (OpTy DPR:$Vn), (OpTy DPR:$Vm))))]> {
  let isCommutable = Commutable;
}
class N3VDIntSL<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin, 
                string OpcodeStr, string Dt, ValueType Ty, Intrinsic IntOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst), (ins DPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane),
        NVMulSLFrm, itin, OpcodeStr, Dt, "$dst, $src1, $src2[$lane]", "",
        [(set (Ty DPR:$dst),
              (Ty (IntOp (Ty DPR:$src1),
                         (Ty (NEONvduplane (Ty DPR_VFP2:$src2),
                                           imm:$lane)))))]> {
  let isCommutable = 0;
}
class N3VDIntSL16<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                  string OpcodeStr, string Dt, ValueType Ty, Intrinsic IntOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst), (ins DPR:$src1, DPR_8:$src2, nohash_imm:$lane),
        NVMulSLFrm, itin, OpcodeStr, Dt, "$dst, $src1, $src2[$lane]", "",
        [(set (Ty DPR:$dst),
              (Ty (IntOp (Ty DPR:$src1),
                         (Ty (NEONvduplane (Ty DPR_8:$src2), imm:$lane)))))]> {
  let isCommutable = 0;
}
class N3VDIntSh<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              Format f, InstrItinClass itin, string OpcodeStr, string Dt,
              ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$Vd), (ins DPR:$Vm, DPR:$Vn), f, itin,
        OpcodeStr, Dt, "$Vd, $Vm, $Vn", "",
        [(set DPR:$Vd, (ResTy (IntOp (OpTy DPR:$Vm), (OpTy DPR:$Vn))))]> {
  let isCommutable = 0;
}

class N3VQInt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              Format f, InstrItinClass itin, string OpcodeStr, string Dt,
              ValueType ResTy, ValueType OpTy, Intrinsic IntOp, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 1, op4,
        (outs QPR:$Vd), (ins QPR:$Vn, QPR:$Vm), f, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm", "",
        [(set QPR:$Vd, (ResTy (IntOp (OpTy QPR:$Vn), (OpTy QPR:$Vm))))]> {
  let isCommutable = Commutable;
}
class N3VQIntSL<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin, 
                string OpcodeStr, string Dt,
                ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins QPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane),
        NVMulSLFrm, itin, OpcodeStr, Dt, "$dst, $src1, $src2[$lane]", "",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (ResTy QPR:$src1),
                            (ResTy (NEONvduplane (OpTy DPR_VFP2:$src2),
                                                 imm:$lane)))))]> {
  let isCommutable = 0;
}
class N3VQIntSL16<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                  string OpcodeStr, string Dt,
                  ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins QPR:$src1, DPR_8:$src2, nohash_imm:$lane),
        NVMulSLFrm, itin, OpcodeStr, Dt, "$dst, $src1, $src2[$lane]", "",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (ResTy QPR:$src1),
                            (ResTy (NEONvduplane (OpTy DPR_8:$src2),
                                                 imm:$lane)))))]> {
  let isCommutable = 0;
}
class N3VQIntSh<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              Format f, InstrItinClass itin, string OpcodeStr, string Dt,
              ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, op23, op21_20, op11_8, 1, op4,
        (outs QPR:$Vd), (ins QPR:$Vm, QPR:$Vn), f, itin,
        OpcodeStr, Dt, "$Vd, $Vm, $Vn", "",
        [(set QPR:$Vd, (ResTy (IntOp (OpTy QPR:$Vm), (OpTy QPR:$Vn))))]> {
  let isCommutable = 0;
}

// Multiply-Add/Sub operations: single-, double- and quad-register.
class N3VSMulOp<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                InstrItinClass itin, string OpcodeStr, string Dt,
                ValueType Ty, SDNode MulOp, SDNode OpNode>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR_VFP2:$dst),
        (ins DPR_VFP2:$src1, DPR_VFP2:$src2, DPR_VFP2:$src3), N3RegFrm, itin,
        OpcodeStr, Dt, "$dst, $src2, $src3", "$src1 = $dst", []>;

class N3VDMulOp<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                InstrItinClass itin, string OpcodeStr, string Dt,
                ValueType Ty, SDNode MulOp, SDNode OpNode>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$Vd), (ins DPR:$src1, DPR:$Vn, DPR:$Vm), N3RegFrm, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm", "$src1 = $Vd",
        [(set DPR:$Vd, (Ty (OpNode DPR:$src1,
                             (Ty (MulOp DPR:$Vn, DPR:$Vm)))))]>;

class N3VDMulOpSL<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                  string OpcodeStr, string Dt,
                  ValueType Ty, SDNode MulOp, SDNode ShOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst),
        (ins DPR:$src1, DPR:$src2, DPR_VFP2:$src3, nohash_imm:$lane),
        NVMulSLFrm, itin,
        OpcodeStr, Dt, "$dst, $src2, $src3[$lane]", "$src1 = $dst",
        [(set (Ty DPR:$dst),
              (Ty (ShOp (Ty DPR:$src1),
                        (Ty (MulOp DPR:$src2,
                                   (Ty (NEONvduplane (Ty DPR_VFP2:$src3),
                                                     imm:$lane)))))))]>;
class N3VDMulOpSL16<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                    string OpcodeStr, string Dt,
                    ValueType Ty, SDNode MulOp, SDNode ShOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$Vd),
        (ins DPR:$src1, DPR:$Vn, DPR_8:$Vm, nohash_imm:$lane),
        NVMulSLFrm, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm[$lane]", "$src1 = $Vd",
        [(set (Ty DPR:$Vd),
              (Ty (ShOp (Ty DPR:$src1),
                        (Ty (MulOp DPR:$Vn,
                                   (Ty (NEONvduplane (Ty DPR_8:$Vm),
                                                     imm:$lane)))))))]>;

class N3VQMulOp<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                InstrItinClass itin, string OpcodeStr, string Dt, ValueType Ty,
                SDNode MulOp, SDNode OpNode>
  : N3V<op24, op23, op21_20, op11_8, 1, op4,
        (outs QPR:$Vd), (ins QPR:$src1, QPR:$Vn, QPR:$Vm), N3RegFrm, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm", "$src1 = $Vd",
        [(set QPR:$Vd, (Ty (OpNode QPR:$src1,
                             (Ty (MulOp QPR:$Vn, QPR:$Vm)))))]>;
class N3VQMulOpSL<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                  string OpcodeStr, string Dt, ValueType ResTy, ValueType OpTy,
                  SDNode MulOp, SDNode ShOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst),
        (ins QPR:$src1, QPR:$src2, DPR_VFP2:$src3, nohash_imm:$lane),
        NVMulSLFrm, itin,
        OpcodeStr, Dt, "$dst, $src2, $src3[$lane]", "$src1 = $dst",
        [(set (ResTy QPR:$dst),
              (ResTy (ShOp (ResTy QPR:$src1),
                           (ResTy (MulOp QPR:$src2,
                                   (ResTy (NEONvduplane (OpTy DPR_VFP2:$src3),
                                                        imm:$lane)))))))]>;
class N3VQMulOpSL16<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                    string OpcodeStr, string Dt,
                    ValueType ResTy, ValueType OpTy,
                    SDNode MulOp, SDNode ShOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst),
        (ins QPR:$src1, QPR:$src2, DPR_8:$src3, nohash_imm:$lane),
        NVMulSLFrm, itin,
        OpcodeStr, Dt, "$dst, $src2, $src3[$lane]", "$src1 = $dst",
        [(set (ResTy QPR:$dst),
              (ResTy (ShOp (ResTy QPR:$src1),
                           (ResTy (MulOp QPR:$src2,
                                   (ResTy (NEONvduplane (OpTy DPR_8:$src3),
                                                        imm:$lane)))))))]>;

// Neon Intrinsic-Op instructions (VABA): double- and quad-register.
class N3VDIntOp<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                InstrItinClass itin, string OpcodeStr, string Dt,
                ValueType Ty, Intrinsic IntOp, SDNode OpNode>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$Vd), (ins DPR:$src1, DPR:$Vn, DPR:$Vm), N3RegFrm, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm", "$src1 = $Vd",
        [(set DPR:$Vd, (Ty (OpNode DPR:$src1,
                             (Ty (IntOp (Ty DPR:$Vn), (Ty DPR:$Vm))))))]>;
class N3VQIntOp<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                InstrItinClass itin, string OpcodeStr, string Dt,
                ValueType Ty, Intrinsic IntOp, SDNode OpNode>
  : N3V<op24, op23, op21_20, op11_8, 1, op4,
        (outs QPR:$Vd), (ins QPR:$src1, QPR:$Vn, QPR:$Vm), N3RegFrm, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm", "$src1 = $Vd",
        [(set QPR:$Vd, (Ty (OpNode QPR:$src1,
                             (Ty (IntOp (Ty QPR:$Vn), (Ty QPR:$Vm))))))]>;

// Neon 3-argument intrinsics, both double- and quad-register.
// The destination register is also used as the first source operand register.
class N3VDInt3<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
               InstrItinClass itin, string OpcodeStr, string Dt,
               ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$dst), (ins DPR:$src1, DPR:$src2, DPR:$src3), N3RegFrm, itin,
        OpcodeStr, Dt, "$dst, $src2, $src3", "$src1 = $dst",
        [(set DPR:$dst, (ResTy (IntOp (OpTy DPR:$src1),
                                      (OpTy DPR:$src2), (OpTy DPR:$src3))))]>;
class N3VQInt3<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
               InstrItinClass itin, string OpcodeStr, string Dt,
               ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, op23, op21_20, op11_8, 1, op4,
        (outs QPR:$dst), (ins QPR:$src1, QPR:$src2, QPR:$src3), N3RegFrm, itin,
        OpcodeStr, Dt, "$dst, $src2, $src3", "$src1 = $dst",
        [(set QPR:$dst, (ResTy (IntOp (OpTy QPR:$src1),
                                      (OpTy QPR:$src2), (OpTy QPR:$src3))))]>;

// Long Multiply-Add/Sub operations.
class N3VLMulOp<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                InstrItinClass itin, string OpcodeStr, string Dt,
                ValueType TyQ, ValueType TyD, SDNode MulOp, SDNode OpNode>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$Vd), (ins QPR:$src1, DPR:$Vn, DPR:$Vm), N3RegFrm, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm", "$src1 = $Vd",
        [(set QPR:$Vd, (OpNode (TyQ QPR:$src1),
                                (TyQ (MulOp (TyD DPR:$Vn),
                                            (TyD DPR:$Vm)))))]>;
class N3VLMulOpSL<bit op24, bits<2> op21_20, bits<4> op11_8,
                  InstrItinClass itin, string OpcodeStr, string Dt,
                  ValueType TyQ, ValueType TyD, SDNode MulOp, SDNode OpNode>
  : N3V<op24, 1, op21_20, op11_8, 1, 0, (outs QPR:$dst),
        (ins QPR:$src1, DPR:$src2, DPR_VFP2:$src3, nohash_imm:$lane),
        NVMulSLFrm, itin,
        OpcodeStr, Dt, "$dst, $src2, $src3[$lane]", "$src1 = $dst",
        [(set QPR:$dst,
          (OpNode (TyQ QPR:$src1),
                  (TyQ (MulOp (TyD DPR:$src2),
                              (TyD (NEONvduplane (TyD DPR_VFP2:$src3),
                                                 imm:$lane))))))]>;
class N3VLMulOpSL16<bit op24, bits<2> op21_20, bits<4> op11_8,
                    InstrItinClass itin, string OpcodeStr, string Dt,
                    ValueType TyQ, ValueType TyD, SDNode MulOp, SDNode OpNode>
  : N3V<op24, 1, op21_20, op11_8, 1, 0, (outs QPR:$dst),
        (ins QPR:$src1, DPR:$src2, DPR_8:$src3, nohash_imm:$lane),
        NVMulSLFrm, itin,
        OpcodeStr, Dt, "$dst, $src2, $src3[$lane]", "$src1 = $dst",
        [(set QPR:$dst,
          (OpNode (TyQ QPR:$src1),
                  (TyQ (MulOp (TyD DPR:$src2),
                              (TyD (NEONvduplane (TyD DPR_8:$src3),
                                                 imm:$lane))))))]>;

// Long Intrinsic-Op vector operations with explicit extend (VABAL).
class N3VLIntExtOp<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                   InstrItinClass itin, string OpcodeStr, string Dt,
                   ValueType TyQ, ValueType TyD, Intrinsic IntOp, SDNode ExtOp,
                   SDNode OpNode>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$Vd), (ins QPR:$src1, DPR:$Vn, DPR:$Vm), N3RegFrm, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm", "$src1 = $Vd",
        [(set QPR:$Vd, (OpNode (TyQ QPR:$src1),
                                (TyQ (ExtOp (TyD (IntOp (TyD DPR:$Vn),
                                                        (TyD DPR:$Vm)))))))]>;

// Neon Long 3-argument intrinsic.  The destination register is
// a quad-register and is also used as the first source operand register.
class N3VLInt3<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
               InstrItinClass itin, string OpcodeStr, string Dt,
               ValueType TyQ, ValueType TyD, Intrinsic IntOp>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$Vd), (ins QPR:$src1, DPR:$Vn, DPR:$Vm), N3RegFrm, itin,
        OpcodeStr, Dt, "$Vd, $Vn, $Vm", "$src1 = $Vd",
        [(set QPR:$Vd,
          (TyQ (IntOp (TyQ QPR:$src1), (TyD DPR:$Vn), (TyD DPR:$Vm))))]>;
class N3VLInt3SL<bit op24, bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                 string OpcodeStr, string Dt,
                 ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst),
        (ins QPR:$src1, DPR:$src2, DPR_VFP2:$src3, nohash_imm:$lane),
        NVMulSLFrm, itin,
        OpcodeStr, Dt, "$dst, $src2, $src3[$lane]", "$src1 = $dst",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (ResTy QPR:$src1),
                            (OpTy DPR:$src2),
                            (OpTy (NEONvduplane (OpTy DPR_VFP2:$src3),
                                                imm:$lane)))))]>;
class N3VLInt3SL16<bit op24, bits<2> op21_20, bits<4> op11_8,
                   InstrItinClass itin, string OpcodeStr, string Dt,
                   ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst),
        (ins QPR:$src1, DPR:$src2, DPR_8:$src3, nohash_imm:$lane),
        NVMulSLFrm, itin,
        OpcodeStr, Dt, "$dst, $src2, $src3[$lane]", "$src1 = $dst",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (ResTy QPR:$src1),
                            (OpTy DPR:$src2),
                            (OpTy (NEONvduplane (OpTy DPR_8:$src3),
                                                imm:$lane)))))]>;

// Narrowing 3-register intrinsics.
class N3VNInt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              string OpcodeStr, string Dt, ValueType TyD, ValueType TyQ,
              Intrinsic IntOp, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$dst), (ins QPR:$src1, QPR:$src2), N3RegFrm, IIC_VBINi4D,
        OpcodeStr, Dt, "$dst, $src1, $src2", "",
        [(set DPR:$dst, (TyD (IntOp (TyQ QPR:$src1), (TyQ QPR:$src2))))]> {
  let isCommutable = Commutable;
}

// Long 3-register operations.
class N3VL<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
           InstrItinClass itin, string OpcodeStr, string Dt,
           ValueType TyQ, ValueType TyD, SDNode OpNode, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$dst), (ins DPR:$src1, DPR:$src2), N3RegFrm, itin,
        OpcodeStr, Dt, "$dst, $src1, $src2", "",
        [(set QPR:$dst, (TyQ (OpNode (TyD DPR:$src1), (TyD DPR:$src2))))]> {
  let isCommutable = Commutable;
}
class N3VLSL<bit op24, bits<2> op21_20, bits<4> op11_8,
             InstrItinClass itin, string OpcodeStr, string Dt,
             ValueType TyQ, ValueType TyD, SDNode OpNode>
  : N3V<op24, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins DPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane),
        NVMulSLFrm, itin, OpcodeStr, Dt, "$dst, $src1, $src2[$lane]", "",
        [(set QPR:$dst,
          (TyQ (OpNode (TyD DPR:$src1),
                       (TyD (NEONvduplane (TyD DPR_VFP2:$src2),imm:$lane)))))]>;
class N3VLSL16<bit op24, bits<2> op21_20, bits<4> op11_8,
               InstrItinClass itin, string OpcodeStr, string Dt,
               ValueType TyQ, ValueType TyD, SDNode OpNode>
  : N3V<op24, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins DPR:$src1, DPR_8:$src2, nohash_imm:$lane), 
        NVMulSLFrm, itin, OpcodeStr, Dt, "$dst, $src1, $src2[$lane]", "",
        [(set QPR:$dst,
          (TyQ (OpNode (TyD DPR:$src1),
                       (TyD (NEONvduplane (TyD DPR_8:$src2), imm:$lane)))))]>;

// Long 3-register operations with explicitly extended operands.
class N3VLExt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              InstrItinClass itin, string OpcodeStr, string Dt,
              ValueType TyQ, ValueType TyD, SDNode OpNode, SDNode ExtOp,
              bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$Qd), (ins DPR:$Dn, DPR:$Dm), N3RegFrm, itin,
        OpcodeStr, Dt, "$Qd, $Dn, $Dm", "",
        [(set QPR:$Qd, (OpNode (TyQ (ExtOp (TyD DPR:$Dn))),
                                (TyQ (ExtOp (TyD DPR:$Dm)))))]> {
  let isCommutable = Commutable;
}

// Long 3-register intrinsics with explicit extend (VABDL).
class N3VLIntExt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                 InstrItinClass itin, string OpcodeStr, string Dt,
                 ValueType TyQ, ValueType TyD, Intrinsic IntOp, SDNode ExtOp,
                 bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$dst), (ins DPR:$src1, DPR:$src2), N3RegFrm, itin,
        OpcodeStr, Dt, "$dst, $src1, $src2", "",
        [(set QPR:$dst, (TyQ (ExtOp (TyD (IntOp (TyD DPR:$src1),
                                                (TyD DPR:$src2))))))]> {
  let isCommutable = Commutable;
}

// Long 3-register intrinsics.
class N3VLInt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              InstrItinClass itin, string OpcodeStr, string Dt,
              ValueType TyQ, ValueType TyD, Intrinsic IntOp, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$dst), (ins DPR:$src1, DPR:$src2), N3RegFrm, itin,
        OpcodeStr, Dt, "$dst, $src1, $src2", "",
        [(set QPR:$dst, (TyQ (IntOp (TyD DPR:$src1), (TyD DPR:$src2))))]> {
  let isCommutable = Commutable;
}
class N3VLIntSL<bit op24, bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                string OpcodeStr, string Dt,
                ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins DPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane),
        NVMulSLFrm, itin, OpcodeStr, Dt, "$dst, $src1, $src2[$lane]", "",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (OpTy DPR:$src1),
                            (OpTy (NEONvduplane (OpTy DPR_VFP2:$src2),
                                                imm:$lane)))))]>;
class N3VLIntSL16<bit op24, bits<2> op21_20, bits<4> op11_8,
                  InstrItinClass itin, string OpcodeStr, string Dt,
                  ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins DPR:$src1, DPR_8:$src2, nohash_imm:$lane), 
        NVMulSLFrm, itin, OpcodeStr, Dt, "$dst, $src1, $src2[$lane]", "",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (OpTy DPR:$src1),
                            (OpTy (NEONvduplane (OpTy DPR_8:$src2),
                                                imm:$lane)))))]>;

// Wide 3-register operations.
class N3VW<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
           string OpcodeStr, string Dt, ValueType TyQ, ValueType TyD,
           SDNode OpNode, SDNode ExtOp, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$Qd), (ins QPR:$Qn, DPR:$Dm), N3RegFrm, IIC_VSUBiD,
        OpcodeStr, Dt, "$Qd, $Qn, $Dm", "",
        [(set QPR:$Qd, (OpNode (TyQ QPR:$Qn),
                                (TyQ (ExtOp (TyD DPR:$Dm)))))]> {
  let isCommutable = Commutable;
}

// Pairwise long 2-register intrinsics, both double- and quad-register.
class N2VDPLInt<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
                bits<2> op17_16, bits<5> op11_7, bit op4,
                string OpcodeStr, string Dt,
                ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4, (outs DPR:$dst),
        (ins DPR:$src), IIC_VSHLiD, OpcodeStr, Dt, "$dst, $src", "",
        [(set DPR:$dst, (ResTy (IntOp (OpTy DPR:$src))))]>;
class N2VQPLInt<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
                bits<2> op17_16, bits<5> op11_7, bit op4,
                string OpcodeStr, string Dt,
                ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 1, op4, (outs QPR:$dst),
        (ins QPR:$src), IIC_VSHLiD, OpcodeStr, Dt, "$dst, $src", "",
        [(set QPR:$dst, (ResTy (IntOp (OpTy QPR:$src))))]>;

// Pairwise long 2-register accumulate intrinsics,
// both double- and quad-register.
// The destination register is also used as the first source operand register.
class N2VDPLInt2<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
                 bits<2> op17_16, bits<5> op11_7, bit op4,
                 string OpcodeStr, string Dt,
                 ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4,
        (outs DPR:$Vd), (ins DPR:$src1, DPR:$Vm), IIC_VPALiD,
        OpcodeStr, Dt, "$Vd, $Vm", "$src1 = $Vd",
        [(set DPR:$Vd, (ResTy (IntOp (ResTy DPR:$src1), (OpTy DPR:$Vm))))]>;
class N2VQPLInt2<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
                 bits<2> op17_16, bits<5> op11_7, bit op4,
                 string OpcodeStr, string Dt,
                 ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 1, op4,
        (outs QPR:$Vd), (ins QPR:$src1, QPR:$Vm), IIC_VPALiQ,
        OpcodeStr, Dt, "$Vd, $Vm", "$src1 = $Vd",
        [(set QPR:$Vd, (ResTy (IntOp (ResTy QPR:$src1), (OpTy QPR:$Vm))))]>;

// Shift by immediate,
// both double- and quad-register.
class N2VDSh<bit op24, bit op23, bits<4> op11_8, bit op7, bit op4,
             Format f, InstrItinClass itin, string OpcodeStr, string Dt,
             ValueType Ty, SDNode OpNode>
  : N2VImm<op24, op23, op11_8, op7, 0, op4,
           (outs DPR:$dst), (ins DPR:$src, i32imm:$SIMM), f, itin,
           OpcodeStr, Dt, "$dst, $src, $SIMM", "",
           [(set DPR:$dst, (Ty (OpNode (Ty DPR:$src), (i32 imm:$SIMM))))]>;
class N2VQSh<bit op24, bit op23, bits<4> op11_8, bit op7, bit op4,
             Format f, InstrItinClass itin, string OpcodeStr, string Dt,
             ValueType Ty, SDNode OpNode>
  : N2VImm<op24, op23, op11_8, op7, 1, op4,
           (outs QPR:$dst), (ins QPR:$src, i32imm:$SIMM), f, itin,
           OpcodeStr, Dt, "$dst, $src, $SIMM", "",
           [(set QPR:$dst, (Ty (OpNode (Ty QPR:$src), (i32 imm:$SIMM))))]>;

// Long shift by immediate.
class N2VLSh<bit op24, bit op23, bits<4> op11_8, bit op7, bit op6, bit op4,
             string OpcodeStr, string Dt,
             ValueType ResTy, ValueType OpTy, SDNode OpNode>
  : N2VImm<op24, op23, op11_8, op7, op6, op4,
           (outs QPR:$dst), (ins DPR:$src, i32imm:$SIMM), N2RegVShLFrm,
           IIC_VSHLiD, OpcodeStr, Dt, "$dst, $src, $SIMM", "",
           [(set QPR:$dst, (ResTy (OpNode (OpTy DPR:$src),
                                          (i32 imm:$SIMM))))]>;

// Narrow shift by immediate.
class N2VNSh<bit op24, bit op23, bits<4> op11_8, bit op7, bit op6, bit op4,
             InstrItinClass itin, string OpcodeStr, string Dt,
             ValueType ResTy, ValueType OpTy, SDNode OpNode>
  : N2VImm<op24, op23, op11_8, op7, op6, op4,
           (outs DPR:$dst), (ins QPR:$src, i32imm:$SIMM), N2RegVShRFrm, itin,
           OpcodeStr, Dt, "$dst, $src, $SIMM", "",
           [(set DPR:$dst, (ResTy (OpNode (OpTy QPR:$src),
                                          (i32 imm:$SIMM))))]>;

// Shift right by immediate and accumulate,
// both double- and quad-register.
class N2VDShAdd<bit op24, bit op23, bits<4> op11_8, bit op7, bit op4,
                string OpcodeStr, string Dt, ValueType Ty, SDNode ShOp>
  : N2VImm<op24, op23, op11_8, op7, 0, op4, (outs DPR:$Vd),
           (ins DPR:$src1, DPR:$Vm, i32imm:$SIMM), N2RegVShRFrm, IIC_VPALiD,
           OpcodeStr, Dt, "$Vd, $Vm, $SIMM", "$src1 = $Vd",
           [(set DPR:$Vd, (Ty (add DPR:$src1,
                                (Ty (ShOp DPR:$Vm, (i32 imm:$SIMM))))))]>;
class N2VQShAdd<bit op24, bit op23, bits<4> op11_8, bit op7, bit op4,
                string OpcodeStr, string Dt, ValueType Ty, SDNode ShOp>
  : N2VImm<op24, op23, op11_8, op7, 1, op4, (outs QPR:$Vd),
           (ins QPR:$src1, QPR:$Vm, i32imm:$SIMM), N2RegVShRFrm, IIC_VPALiD,
           OpcodeStr, Dt, "$Vd, $Vm, $SIMM", "$src1 = $Vd",
           [(set QPR:$Vd, (Ty (add QPR:$src1,
                                (Ty (ShOp QPR:$Vm, (i32 imm:$SIMM))))))]>;

// Shift by immediate and insert,
// both double- and quad-register.
class N2VDShIns<bit op24, bit op23, bits<4> op11_8, bit op7, bit op4,
                Format f, string OpcodeStr, string Dt, ValueType Ty,SDNode ShOp>
  : N2VImm<op24, op23, op11_8, op7, 0, op4, (outs DPR:$Vd),
           (ins DPR:$src1, DPR:$Vm, i32imm:$SIMM), f, IIC_VSHLiD,
           OpcodeStr, Dt, "$Vd, $Vm, $SIMM", "$src1 = $Vd",
           [(set DPR:$Vd, (Ty (ShOp DPR:$src1, DPR:$Vm, (i32 imm:$SIMM))))]>;
class N2VQShIns<bit op24, bit op23, bits<4> op11_8, bit op7, bit op4,
                Format f, string OpcodeStr, string Dt, ValueType Ty,SDNode ShOp>
  : N2VImm<op24, op23, op11_8, op7, 1, op4, (outs QPR:$Vd),
           (ins QPR:$src1, QPR:$Vm, i32imm:$SIMM), f, IIC_VSHLiQ,
           OpcodeStr, Dt, "$Vd, $Vm, $SIMM", "$src1 = $Vd",
           [(set QPR:$Vd, (Ty (ShOp QPR:$src1, QPR:$Vm, (i32 imm:$SIMM))))]>;

// Convert, with fractional bits immediate,
// both double- and quad-register.
class N2VCvtD<bit op24, bit op23, bits<4> op11_8, bit op7, bit op4,
              string OpcodeStr, string Dt, ValueType ResTy, ValueType OpTy,
              Intrinsic IntOp>
  : N2VImm<op24, op23, op11_8, op7, 0, op4,
           (outs DPR:$Vd), (ins DPR:$Vm, neon_vcvt_imm32:$SIMM), NVCVTFrm,
           IIC_VUNAD, OpcodeStr, Dt, "$Vd, $Vm, $SIMM", "",
           [(set DPR:$Vd, (ResTy (IntOp (OpTy DPR:$Vm), (i32 imm:$SIMM))))]>;
class N2VCvtQ<bit op24, bit op23, bits<4> op11_8, bit op7, bit op4,
              string OpcodeStr, string Dt, ValueType ResTy, ValueType OpTy,
              Intrinsic IntOp>
  : N2VImm<op24, op23, op11_8, op7, 1, op4,
           (outs QPR:$Vd), (ins QPR:$Vm, neon_vcvt_imm32:$SIMM), NVCVTFrm,
           IIC_VUNAQ, OpcodeStr, Dt, "$Vd, $Vm, $SIMM", "",
           [(set QPR:$Vd, (ResTy (IntOp (OpTy QPR:$Vm), (i32 imm:$SIMM))))]>;

//===----------------------------------------------------------------------===//
// Multiclasses
//===----------------------------------------------------------------------===//

// Abbreviations used in multiclass suffixes:
//   Q = quarter int (8 bit) elements
//   H = half int (16 bit) elements
//   S = single int (32 bit) elements
//   D = double int (64 bit) elements

// Neon 2-register vector operations -- for disassembly only.

// First with only element sizes of 8, 16 and 32 bits:
multiclass N2V_QHS_cmp<bits<2> op24_23, bits<2> op21_20, bits<2> op17_16,
                       bits<5> op11_7, bit op4, string opc, string Dt,
                       string asm> {
  // 64-bit vector types.
  def v8i8  : N2V<op24_23, op21_20, 0b00, op17_16, op11_7, 0, op4,
                  (outs DPR:$dst), (ins DPR:$src), NoItinerary,
                  opc, !strconcat(Dt, "8"), asm, "", []>;
  def v4i16 : N2V<op24_23, op21_20, 0b01, op17_16, op11_7, 0, op4,
                  (outs DPR:$dst), (ins DPR:$src), NoItinerary,
                  opc, !strconcat(Dt, "16"), asm, "", []>;
  def v2i32 : N2V<op24_23, op21_20, 0b10, op17_16, op11_7, 0, op4,
                  (outs DPR:$dst), (ins DPR:$src), NoItinerary,
                  opc, !strconcat(Dt, "32"), asm, "", []>;
  def v2f32 : N2V<op24_23, op21_20, 0b10, op17_16, op11_7, 0, op4,
                  (outs DPR:$dst), (ins DPR:$src), NoItinerary,
                  opc, "f32", asm, "", []> {
    let Inst{10} = 1; // overwrite F = 1
  }

  // 128-bit vector types.
  def v16i8 : N2V<op24_23, op21_20, 0b00, op17_16, op11_7, 1, op4,
                  (outs QPR:$dst), (ins QPR:$src), NoItinerary,
                  opc, !strconcat(Dt, "8"), asm, "", []>;
  def v8i16 : N2V<op24_23, op21_20, 0b01, op17_16, op11_7, 1, op4,
                  (outs QPR:$dst), (ins QPR:$src), NoItinerary,
                  opc, !strconcat(Dt, "16"), asm, "", []>;
  def v4i32 : N2V<op24_23, op21_20, 0b10, op17_16, op11_7, 1, op4,
                  (outs QPR:$dst), (ins QPR:$src), NoItinerary,
                  opc, !strconcat(Dt, "32"), asm, "", []>;
  def v4f32 : N2V<op24_23, op21_20, 0b10, op17_16, op11_7, 1, op4,
                  (outs QPR:$dst), (ins QPR:$src), NoItinerary,
                  opc, "f32", asm, "", []> {
    let Inst{10} = 1; // overwrite F = 1
  }
}

// Neon 3-register vector operations.

// First with only element sizes of 8, 16 and 32 bits:
multiclass N3V_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                   InstrItinClass itinD16, InstrItinClass itinD32,
                   InstrItinClass itinQ16, InstrItinClass itinQ32,
                   string OpcodeStr, string Dt,
                   SDNode OpNode, bit Commutable = 0> {
  // 64-bit vector types.
  def v8i8  : N3VD<op24, op23, 0b00, op11_8, op4, itinD16, 
                   OpcodeStr, !strconcat(Dt, "8"),
                   v8i8, v8i8, OpNode, Commutable>;
  def v4i16 : N3VD<op24, op23, 0b01, op11_8, op4, itinD16,
                   OpcodeStr, !strconcat(Dt, "16"),
                   v4i16, v4i16, OpNode, Commutable>;
  def v2i32 : N3VD<op24, op23, 0b10, op11_8, op4, itinD32,
                   OpcodeStr, !strconcat(Dt, "32"),
                   v2i32, v2i32, OpNode, Commutable>;

  // 128-bit vector types.
  def v16i8 : N3VQ<op24, op23, 0b00, op11_8, op4, itinQ16,
                   OpcodeStr, !strconcat(Dt, "8"),
                   v16i8, v16i8, OpNode, Commutable>;
  def v8i16 : N3VQ<op24, op23, 0b01, op11_8, op4, itinQ16,
                   OpcodeStr, !strconcat(Dt, "16"),
                   v8i16, v8i16, OpNode, Commutable>;
  def v4i32 : N3VQ<op24, op23, 0b10, op11_8, op4, itinQ32,
                   OpcodeStr, !strconcat(Dt, "32"),
                   v4i32, v4i32, OpNode, Commutable>;
}

multiclass N3VSL_HS<bits<4> op11_8, string OpcodeStr, string Dt, SDNode ShOp> {
  def v4i16 : N3VDSL16<0b01, op11_8, OpcodeStr, !strconcat(Dt, "16"),
                       v4i16, ShOp>;
  def v2i32 : N3VDSL<0b10, op11_8, IIC_VMULi32D, OpcodeStr, !strconcat(Dt,"32"),
                     v2i32, ShOp>;
  def v8i16 : N3VQSL16<0b01, op11_8, OpcodeStr, !strconcat(Dt, "16"),
                       v8i16, v4i16, ShOp>;
  def v4i32 : N3VQSL<0b10, op11_8, IIC_VMULi32Q, OpcodeStr, !strconcat(Dt,"32"),
                     v4i32, v2i32, ShOp>;
}

// ....then also with element size 64 bits:
multiclass N3V_QHSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                    InstrItinClass itinD, InstrItinClass itinQ,
                    string OpcodeStr, string Dt,
                    SDNode OpNode, bit Commutable = 0>
  : N3V_QHS<op24, op23, op11_8, op4, itinD, itinD, itinQ, itinQ,
            OpcodeStr, Dt, OpNode, Commutable> {
  def v1i64 : N3VD<op24, op23, 0b11, op11_8, op4, itinD,
                   OpcodeStr, !strconcat(Dt, "64"),
                   v1i64, v1i64, OpNode, Commutable>;
  def v2i64 : N3VQ<op24, op23, 0b11, op11_8, op4, itinQ,
                   OpcodeStr, !strconcat(Dt, "64"),
                   v2i64, v2i64, OpNode, Commutable>;
}


// Neon Narrowing 2-register vector operations,
//   source operand element sizes of 16, 32 and 64 bits:
multiclass N2VN_HSD<bits<2> op24_23, bits<2> op21_20, bits<2> op17_16,
                    bits<5> op11_7, bit op6, bit op4, 
                    InstrItinClass itin, string OpcodeStr, string Dt,
                    SDNode OpNode> {
  def v8i8  : N2VN<op24_23, op21_20, 0b00, op17_16, op11_7, op6, op4,
                   itin, OpcodeStr, !strconcat(Dt, "16"),
                   v8i8, v8i16, OpNode>;
  def v4i16 : N2VN<op24_23, op21_20, 0b01, op17_16, op11_7, op6, op4,
                   itin, OpcodeStr, !strconcat(Dt, "32"),
                   v4i16, v4i32, OpNode>;
  def v2i32 : N2VN<op24_23, op21_20, 0b10, op17_16, op11_7, op6, op4,
                   itin, OpcodeStr, !strconcat(Dt, "64"),
                   v2i32, v2i64, OpNode>;
}

// Neon Narrowing 2-register vector intrinsics,
//   source operand element sizes of 16, 32 and 64 bits:
multiclass N2VNInt_HSD<bits<2> op24_23, bits<2> op21_20, bits<2> op17_16,
                       bits<5> op11_7, bit op6, bit op4, 
                       InstrItinClass itin, string OpcodeStr, string Dt,
                       Intrinsic IntOp> {
  def v8i8  : N2VNInt<op24_23, op21_20, 0b00, op17_16, op11_7, op6, op4,
                      itin, OpcodeStr, !strconcat(Dt, "16"),
                      v8i8, v8i16, IntOp>;
  def v4i16 : N2VNInt<op24_23, op21_20, 0b01, op17_16, op11_7, op6, op4,
                      itin, OpcodeStr, !strconcat(Dt, "32"),
                      v4i16, v4i32, IntOp>;
  def v2i32 : N2VNInt<op24_23, op21_20, 0b10, op17_16, op11_7, op6, op4,
                      itin, OpcodeStr, !strconcat(Dt, "64"),
                      v2i32, v2i64, IntOp>;
}


// Neon Lengthening 2-register vector intrinsic (currently specific to VMOVL).
//   source operand element sizes of 16, 32 and 64 bits:
multiclass N2VL_QHS<bits<2> op24_23, bits<5> op11_7, bit op6, bit op4,
                    string OpcodeStr, string Dt, SDNode OpNode> {
  def v8i16 : N2VL<op24_23, 0b00, 0b10, 0b00, op11_7, op6, op4, IIC_VQUNAiD,
                   OpcodeStr, !strconcat(Dt, "8"), v8i16, v8i8, OpNode>;
  def v4i32 : N2VL<op24_23, 0b01, 0b00, 0b00, op11_7, op6, op4, IIC_VQUNAiD,
                   OpcodeStr, !strconcat(Dt, "16"), v4i32, v4i16, OpNode>;
  def v2i64 : N2VL<op24_23, 0b10, 0b00, 0b00, op11_7, op6, op4, IIC_VQUNAiD,
                   OpcodeStr, !strconcat(Dt, "32"), v2i64, v2i32, OpNode>;
}


// Neon 3-register vector intrinsics.

// First with only element sizes of 16 and 32 bits:
multiclass N3VInt_HS<bit op24, bit op23, bits<4> op11_8, bit op4, Format f,
                     InstrItinClass itinD16, InstrItinClass itinD32,
                     InstrItinClass itinQ16, InstrItinClass itinQ32,
                     string OpcodeStr, string Dt,
                     Intrinsic IntOp, bit Commutable = 0> {
  // 64-bit vector types.
  def v4i16 : N3VDInt<op24, op23, 0b01, op11_8, op4, f, itinD16,
                      OpcodeStr, !strconcat(Dt, "16"),
                      v4i16, v4i16, IntOp, Commutable>;
  def v2i32 : N3VDInt<op24, op23, 0b10, op11_8, op4, f, itinD32,
                      OpcodeStr, !strconcat(Dt, "32"),
                      v2i32, v2i32, IntOp, Commutable>;

  // 128-bit vector types.
  def v8i16 : N3VQInt<op24, op23, 0b01, op11_8, op4, f, itinQ16,
                      OpcodeStr, !strconcat(Dt, "16"),
                      v8i16, v8i16, IntOp, Commutable>;
  def v4i32 : N3VQInt<op24, op23, 0b10, op11_8, op4, f, itinQ32,
                      OpcodeStr, !strconcat(Dt, "32"),
                      v4i32, v4i32, IntOp, Commutable>;
}
multiclass N3VInt_HSSh<bit op24, bit op23, bits<4> op11_8, bit op4, Format f,
                     InstrItinClass itinD16, InstrItinClass itinD32,
                     InstrItinClass itinQ16, InstrItinClass itinQ32,
                     string OpcodeStr, string Dt,
                     Intrinsic IntOp> {
  // 64-bit vector types.
  def v4i16 : N3VDIntSh<op24, op23, 0b01, op11_8, op4, f, itinD16,
                      OpcodeStr, !strconcat(Dt, "16"),
                      v4i16, v4i16, IntOp>;
  def v2i32 : N3VDIntSh<op24, op23, 0b10, op11_8, op4, f, itinD32,
                      OpcodeStr, !strconcat(Dt, "32"),
                      v2i32, v2i32, IntOp>;

  // 128-bit vector types.
  def v8i16 : N3VQIntSh<op24, op23, 0b01, op11_8, op4, f, itinQ16,
                      OpcodeStr, !strconcat(Dt, "16"),
                      v8i16, v8i16, IntOp>;
  def v4i32 : N3VQIntSh<op24, op23, 0b10, op11_8, op4, f, itinQ32,
                      OpcodeStr, !strconcat(Dt, "32"),
                      v4i32, v4i32, IntOp>;
}

multiclass N3VIntSL_HS<bits<4> op11_8, 
                       InstrItinClass itinD16, InstrItinClass itinD32,
                       InstrItinClass itinQ16, InstrItinClass itinQ32,
                       string OpcodeStr, string Dt, Intrinsic IntOp> {
  def v4i16 : N3VDIntSL16<0b01, op11_8, itinD16,
                          OpcodeStr, !strconcat(Dt, "16"), v4i16, IntOp>;
  def v2i32 : N3VDIntSL<0b10, op11_8, itinD32,
                        OpcodeStr, !strconcat(Dt, "32"), v2i32, IntOp>;
  def v8i16 : N3VQIntSL16<0b01, op11_8, itinQ16,
                          OpcodeStr, !strconcat(Dt, "16"), v8i16, v4i16, IntOp>;
  def v4i32 : N3VQIntSL<0b10, op11_8, itinQ32,
                        OpcodeStr, !strconcat(Dt, "32"), v4i32, v2i32, IntOp>;
}

// ....then also with element size of 8 bits:
multiclass N3VInt_QHS<bit op24, bit op23, bits<4> op11_8, bit op4, Format f,
                      InstrItinClass itinD16, InstrItinClass itinD32,
                      InstrItinClass itinQ16, InstrItinClass itinQ32,
                      string OpcodeStr, string Dt,
                      Intrinsic IntOp, bit Commutable = 0>
  : N3VInt_HS<op24, op23, op11_8, op4, f, itinD16, itinD32, itinQ16, itinQ32,
              OpcodeStr, Dt, IntOp, Commutable> {
  def v8i8  : N3VDInt<op24, op23, 0b00, op11_8, op4, f, itinD16,
                      OpcodeStr, !strconcat(Dt, "8"),
                      v8i8, v8i8, IntOp, Commutable>;
  def v16i8 : N3VQInt<op24, op23, 0b00, op11_8, op4, f, itinQ16,
                      OpcodeStr, !strconcat(Dt, "8"),
                      v16i8, v16i8, IntOp, Commutable>;
}
multiclass N3VInt_QHSSh<bit op24, bit op23, bits<4> op11_8, bit op4, Format f,
                      InstrItinClass itinD16, InstrItinClass itinD32,
                      InstrItinClass itinQ16, InstrItinClass itinQ32,
                      string OpcodeStr, string Dt,
                      Intrinsic IntOp>
  : N3VInt_HSSh<op24, op23, op11_8, op4, f, itinD16, itinD32, itinQ16, itinQ32,
              OpcodeStr, Dt, IntOp> {
  def v8i8  : N3VDIntSh<op24, op23, 0b00, op11_8, op4, f, itinD16,
                      OpcodeStr, !strconcat(Dt, "8"),
                      v8i8, v8i8, IntOp>;
  def v16i8 : N3VQIntSh<op24, op23, 0b00, op11_8, op4, f, itinQ16,
                      OpcodeStr, !strconcat(Dt, "8"),
                      v16i8, v16i8, IntOp>;
}


// ....then also with element size of 64 bits:
multiclass N3VInt_QHSD<bit op24, bit op23, bits<4> op11_8, bit op4, Format f,
                       InstrItinClass itinD16, InstrItinClass itinD32,
                       InstrItinClass itinQ16, InstrItinClass itinQ32,
                       string OpcodeStr, string Dt,
                       Intrinsic IntOp, bit Commutable = 0>
  : N3VInt_QHS<op24, op23, op11_8, op4, f, itinD16, itinD32, itinQ16, itinQ32,
               OpcodeStr, Dt, IntOp, Commutable> {
  def v1i64 : N3VDInt<op24, op23, 0b11, op11_8, op4, f, itinD32,
                      OpcodeStr, !strconcat(Dt, "64"),
                      v1i64, v1i64, IntOp, Commutable>;
  def v2i64 : N3VQInt<op24, op23, 0b11, op11_8, op4, f, itinQ32,
                      OpcodeStr, !strconcat(Dt, "64"),
                      v2i64, v2i64, IntOp, Commutable>;
}
multiclass N3VInt_QHSDSh<bit op24, bit op23, bits<4> op11_8, bit op4, Format f,
                       InstrItinClass itinD16, InstrItinClass itinD32,
                       InstrItinClass itinQ16, InstrItinClass itinQ32,
                       string OpcodeStr, string Dt,
                       Intrinsic IntOp>
  : N3VInt_QHSSh<op24, op23, op11_8, op4, f, itinD16, itinD32, itinQ16, itinQ32,
               OpcodeStr, Dt, IntOp> {
  def v1i64 : N3VDIntSh<op24, op23, 0b11, op11_8, op4, f, itinD32,
                      OpcodeStr, !strconcat(Dt, "64"),
                      v1i64, v1i64, IntOp>;
  def v2i64 : N3VQIntSh<op24, op23, 0b11, op11_8, op4, f, itinQ32,
                      OpcodeStr, !strconcat(Dt, "64"),
                      v2i64, v2i64, IntOp>;
}

// Neon Narrowing 3-register vector intrinsics,
//   source operand element sizes of 16, 32 and 64 bits:
multiclass N3VNInt_HSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                       string OpcodeStr, string Dt,
                       Intrinsic IntOp, bit Commutable = 0> {
  def v8i8  : N3VNInt<op24, op23, 0b00, op11_8, op4,
                      OpcodeStr, !strconcat(Dt, "16"),
                      v8i8, v8i16, IntOp, Commutable>;
  def v4i16 : N3VNInt<op24, op23, 0b01, op11_8, op4,
                      OpcodeStr, !strconcat(Dt, "32"),
                      v4i16, v4i32, IntOp, Commutable>;
  def v2i32 : N3VNInt<op24, op23, 0b10, op11_8, op4,
                      OpcodeStr, !strconcat(Dt, "64"),
                      v2i32, v2i64, IntOp, Commutable>;
}


// Neon Long 3-register vector operations.

multiclass N3VL_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                    InstrItinClass itin16, InstrItinClass itin32,
                    string OpcodeStr, string Dt,
                    SDNode OpNode, bit Commutable = 0> {
  def v8i16 : N3VL<op24, op23, 0b00, op11_8, op4, itin16,
                   OpcodeStr, !strconcat(Dt, "8"),
                   v8i16, v8i8, OpNode, Commutable>;
  def v4i32 : N3VL<op24, op23, 0b01, op11_8, op4, itin16, 
                   OpcodeStr, !strconcat(Dt, "16"),
                   v4i32, v4i16, OpNode, Commutable>;
  def v2i64 : N3VL<op24, op23, 0b10, op11_8, op4, itin32,
                   OpcodeStr, !strconcat(Dt, "32"),
                   v2i64, v2i32, OpNode, Commutable>;
}

multiclass N3VLSL_HS<bit op24, bits<4> op11_8,
                     InstrItinClass itin, string OpcodeStr, string Dt,
                     SDNode OpNode> {
  def v4i16 : N3VLSL16<op24, 0b01, op11_8, itin, OpcodeStr,
                       !strconcat(Dt, "16"), v4i32, v4i16, OpNode>;
  def v2i32 : N3VLSL<op24, 0b10, op11_8, itin, OpcodeStr,
                     !strconcat(Dt, "32"), v2i64, v2i32, OpNode>;
}

multiclass N3VLExt_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                       InstrItinClass itin16, InstrItinClass itin32,
                       string OpcodeStr, string Dt,
                       SDNode OpNode, SDNode ExtOp, bit Commutable = 0> {
  def v8i16 : N3VLExt<op24, op23, 0b00, op11_8, op4, itin16,
                      OpcodeStr, !strconcat(Dt, "8"),
                      v8i16, v8i8, OpNode, ExtOp, Commutable>;
  def v4i32 : N3VLExt<op24, op23, 0b01, op11_8, op4, itin16, 
                      OpcodeStr, !strconcat(Dt, "16"),
                      v4i32, v4i16, OpNode, ExtOp, Commutable>;
  def v2i64 : N3VLExt<op24, op23, 0b10, op11_8, op4, itin32,
                      OpcodeStr, !strconcat(Dt, "32"),
                      v2i64, v2i32, OpNode, ExtOp, Commutable>;
}

// Neon Long 3-register vector intrinsics.

// First with only element sizes of 16 and 32 bits:
multiclass N3VLInt_HS<bit op24, bit op23, bits<4> op11_8, bit op4,
                      InstrItinClass itin16, InstrItinClass itin32,
                      string OpcodeStr, string Dt,
                      Intrinsic IntOp, bit Commutable = 0> {
  def v4i32 : N3VLInt<op24, op23, 0b01, op11_8, op4, itin16, 
                      OpcodeStr, !strconcat(Dt, "16"),
                      v4i32, v4i16, IntOp, Commutable>;
  def v2i64 : N3VLInt<op24, op23, 0b10, op11_8, op4, itin32,
                      OpcodeStr, !strconcat(Dt, "32"),
                      v2i64, v2i32, IntOp, Commutable>;
}

multiclass N3VLIntSL_HS<bit op24, bits<4> op11_8,
                        InstrItinClass itin, string OpcodeStr, string Dt,
                        Intrinsic IntOp> {
  def v4i16 : N3VLIntSL16<op24, 0b01, op11_8, itin, 
                          OpcodeStr, !strconcat(Dt, "16"), v4i32, v4i16, IntOp>;
  def v2i32 : N3VLIntSL<op24, 0b10, op11_8, itin,
                        OpcodeStr, !strconcat(Dt, "32"), v2i64, v2i32, IntOp>;
}

// ....then also with element size of 8 bits:
multiclass N3VLInt_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                       InstrItinClass itin16, InstrItinClass itin32,
                       string OpcodeStr, string Dt,
                       Intrinsic IntOp, bit Commutable = 0>
  : N3VLInt_HS<op24, op23, op11_8, op4, itin16, itin32, OpcodeStr, Dt,
               IntOp, Commutable> {
  def v8i16 : N3VLInt<op24, op23, 0b00, op11_8, op4, itin16,
                      OpcodeStr, !strconcat(Dt, "8"),
                      v8i16, v8i8, IntOp, Commutable>;
}

// ....with explicit extend (VABDL).
multiclass N3VLIntExt_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                       InstrItinClass itin, string OpcodeStr, string Dt,
                       Intrinsic IntOp, SDNode ExtOp, bit Commutable = 0> {
  def v8i16 : N3VLIntExt<op24, op23, 0b00, op11_8, op4, itin,
                         OpcodeStr, !strconcat(Dt, "8"),
                         v8i16, v8i8, IntOp, ExtOp, Commutable>;
  def v4i32 : N3VLIntExt<op24, op23, 0b01, op11_8, op4, itin, 
                         OpcodeStr, !strconcat(Dt, "16"),
                         v4i32, v4i16, IntOp, ExtOp, Commutable>;
  def v2i64 : N3VLIntExt<op24, op23, 0b10, op11_8, op4, itin,
                         OpcodeStr, !strconcat(Dt, "32"),
                         v2i64, v2i32, IntOp, ExtOp, Commutable>;
}


// Neon Wide 3-register vector intrinsics,
//   source operand element sizes of 8, 16 and 32 bits:
multiclass N3VW_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                    string OpcodeStr, string Dt,
                    SDNode OpNode, SDNode ExtOp, bit Commutable = 0> {
  def v8i16 : N3VW<op24, op23, 0b00, op11_8, op4,
                   OpcodeStr, !strconcat(Dt, "8"),
                   v8i16, v8i8, OpNode, ExtOp, Commutable>;
  def v4i32 : N3VW<op24, op23, 0b01, op11_8, op4,
                   OpcodeStr, !strconcat(Dt, "16"),
                   v4i32, v4i16, OpNode, ExtOp, Commutable>;
  def v2i64 : N3VW<op24, op23, 0b10, op11_8, op4,
                   OpcodeStr, !strconcat(Dt, "32"),
                   v2i64, v2i32, OpNode, ExtOp, Commutable>;
}


// Neon Multiply-Op vector operations,
//   element sizes of 8, 16 and 32 bits:
multiclass N3VMulOp_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                        InstrItinClass itinD16, InstrItinClass itinD32,
                        InstrItinClass itinQ16, InstrItinClass itinQ32,
                        string OpcodeStr, string Dt, SDNode OpNode> {
  // 64-bit vector types.
  def v8i8  : N3VDMulOp<op24, op23, 0b00, op11_8, op4, itinD16,
                        OpcodeStr, !strconcat(Dt, "8"), v8i8, mul, OpNode>;
  def v4i16 : N3VDMulOp<op24, op23, 0b01, op11_8, op4, itinD16,
                        OpcodeStr, !strconcat(Dt, "16"), v4i16, mul, OpNode>;
  def v2i32 : N3VDMulOp<op24, op23, 0b10, op11_8, op4, itinD32,
                        OpcodeStr, !strconcat(Dt, "32"), v2i32, mul, OpNode>;

  // 128-bit vector types.
  def v16i8 : N3VQMulOp<op24, op23, 0b00, op11_8, op4, itinQ16,
                        OpcodeStr, !strconcat(Dt, "8"), v16i8, mul, OpNode>;
  def v8i16 : N3VQMulOp<op24, op23, 0b01, op11_8, op4, itinQ16,
                        OpcodeStr, !strconcat(Dt, "16"), v8i16, mul, OpNode>;
  def v4i32 : N3VQMulOp<op24, op23, 0b10, op11_8, op4, itinQ32,
                        OpcodeStr, !strconcat(Dt, "32"), v4i32, mul, OpNode>;
}

multiclass N3VMulOpSL_HS<bits<4> op11_8, 
                         InstrItinClass itinD16, InstrItinClass itinD32,
                         InstrItinClass itinQ16, InstrItinClass itinQ32,
                         string OpcodeStr, string Dt, SDNode ShOp> {
  def v4i16 : N3VDMulOpSL16<0b01, op11_8, itinD16,
                            OpcodeStr, !strconcat(Dt, "16"), v4i16, mul, ShOp>;
  def v2i32 : N3VDMulOpSL<0b10, op11_8, itinD32,
                          OpcodeStr, !strconcat(Dt, "32"), v2i32, mul, ShOp>;
  def v8i16 : N3VQMulOpSL16<0b01, op11_8, itinQ16,
                            OpcodeStr, !strconcat(Dt, "16"), v8i16, v4i16,
                            mul, ShOp>;
  def v4i32 : N3VQMulOpSL<0b10, op11_8, itinQ32,
                          OpcodeStr, !strconcat(Dt, "32"), v4i32, v2i32,
                          mul, ShOp>;
}

// Neon Intrinsic-Op vector operations,
//   element sizes of 8, 16 and 32 bits:
multiclass N3VIntOp_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                        InstrItinClass itinD, InstrItinClass itinQ,
                        string OpcodeStr, string Dt, Intrinsic IntOp,
                        SDNode OpNode> {
  // 64-bit vector types.
  def v8i8  : N3VDIntOp<op24, op23, 0b00, op11_8, op4, itinD,
                        OpcodeStr, !strconcat(Dt, "8"), v8i8, IntOp, OpNode>;
  def v4i16 : N3VDIntOp<op24, op23, 0b01, op11_8, op4, itinD,
                        OpcodeStr, !strconcat(Dt, "16"), v4i16, IntOp, OpNode>;
  def v2i32 : N3VDIntOp<op24, op23, 0b10, op11_8, op4, itinD,
                        OpcodeStr, !strconcat(Dt, "32"), v2i32, IntOp, OpNode>;

  // 128-bit vector types.
  def v16i8 : N3VQIntOp<op24, op23, 0b00, op11_8, op4, itinQ,
                        OpcodeStr, !strconcat(Dt, "8"), v16i8, IntOp, OpNode>;
  def v8i16 : N3VQIntOp<op24, op23, 0b01, op11_8, op4, itinQ,
                        OpcodeStr, !strconcat(Dt, "16"), v8i16, IntOp, OpNode>;
  def v4i32 : N3VQIntOp<op24, op23, 0b10, op11_8, op4, itinQ,
                        OpcodeStr, !strconcat(Dt, "32"), v4i32, IntOp, OpNode>;
}

// Neon 3-argument intrinsics,
//   element sizes of 8, 16 and 32 bits:
multiclass N3VInt3_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                       InstrItinClass itinD, InstrItinClass itinQ,
                       string OpcodeStr, string Dt, Intrinsic IntOp> {
  // 64-bit vector types.
  def v8i8  : N3VDInt3<op24, op23, 0b00, op11_8, op4, itinD,
                       OpcodeStr, !strconcat(Dt, "8"), v8i8, v8i8, IntOp>;
  def v4i16 : N3VDInt3<op24, op23, 0b01, op11_8, op4, itinD,
                       OpcodeStr, !strconcat(Dt, "16"), v4i16, v4i16, IntOp>;
  def v2i32 : N3VDInt3<op24, op23, 0b10, op11_8, op4, itinD,
                       OpcodeStr, !strconcat(Dt, "32"), v2i32, v2i32, IntOp>;

  // 128-bit vector types.
  def v16i8 : N3VQInt3<op24, op23, 0b00, op11_8, op4, itinQ,
                       OpcodeStr, !strconcat(Dt, "8"), v16i8, v16i8, IntOp>;
  def v8i16 : N3VQInt3<op24, op23, 0b01, op11_8, op4, itinQ,
                       OpcodeStr, !strconcat(Dt, "16"), v8i16, v8i16, IntOp>;
  def v4i32 : N3VQInt3<op24, op23, 0b10, op11_8, op4, itinQ,
                       OpcodeStr, !strconcat(Dt, "32"), v4i32, v4i32, IntOp>;
}


// Neon Long Multiply-Op vector operations,
//   element sizes of 8, 16 and 32 bits:
multiclass N3VLMulOp_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                         InstrItinClass itin16, InstrItinClass itin32,
                         string OpcodeStr, string Dt, SDNode MulOp,
                         SDNode OpNode> {
  def v8i16 : N3VLMulOp<op24, op23, 0b00, op11_8, op4, itin16, OpcodeStr,
                        !strconcat(Dt, "8"), v8i16, v8i8, MulOp, OpNode>;
  def v4i32 : N3VLMulOp<op24, op23, 0b01, op11_8, op4, itin16, OpcodeStr,
                        !strconcat(Dt, "16"), v4i32, v4i16, MulOp, OpNode>;
  def v2i64 : N3VLMulOp<op24, op23, 0b10, op11_8, op4, itin32, OpcodeStr,
                        !strconcat(Dt, "32"), v2i64, v2i32, MulOp, OpNode>;
}

multiclass N3VLMulOpSL_HS<bit op24, bits<4> op11_8, string OpcodeStr,
                          string Dt, SDNode MulOp, SDNode OpNode> {
  def v4i16 : N3VLMulOpSL16<op24, 0b01, op11_8, IIC_VMACi16D, OpcodeStr,
                            !strconcat(Dt,"16"), v4i32, v4i16, MulOp, OpNode>;
  def v2i32 : N3VLMulOpSL<op24, 0b10, op11_8, IIC_VMACi32D, OpcodeStr,
                          !strconcat(Dt, "32"), v2i64, v2i32, MulOp, OpNode>;
}


// Neon Long 3-argument intrinsics.

// First with only element sizes of 16 and 32 bits:
multiclass N3VLInt3_HS<bit op24, bit op23, bits<4> op11_8, bit op4,
                       InstrItinClass itin16, InstrItinClass itin32,
                       string OpcodeStr, string Dt, Intrinsic IntOp> {
  def v4i32 : N3VLInt3<op24, op23, 0b01, op11_8, op4, itin16,
                       OpcodeStr, !strconcat(Dt, "16"), v4i32, v4i16, IntOp>;
  def v2i64 : N3VLInt3<op24, op23, 0b10, op11_8, op4, itin32,
                       OpcodeStr, !strconcat(Dt, "32"), v2i64, v2i32, IntOp>;
}

multiclass N3VLInt3SL_HS<bit op24, bits<4> op11_8,
                         string OpcodeStr, string Dt, Intrinsic IntOp> {
  def v4i16 : N3VLInt3SL16<op24, 0b01, op11_8, IIC_VMACi16D,
                           OpcodeStr, !strconcat(Dt,"16"), v4i32, v4i16, IntOp>;
  def v2i32 : N3VLInt3SL<op24, 0b10, op11_8, IIC_VMACi32D,
                         OpcodeStr, !strconcat(Dt, "32"), v2i64, v2i32, IntOp>;
}

// ....then also with element size of 8 bits:
multiclass N3VLInt3_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                        InstrItinClass itin16, InstrItinClass itin32,
                        string OpcodeStr, string Dt, Intrinsic IntOp>
  : N3VLInt3_HS<op24, op23, op11_8, op4, itin16, itin32, OpcodeStr, Dt, IntOp> {
  def v8i16 : N3VLInt3<op24, op23, 0b00, op11_8, op4, itin16,
                       OpcodeStr, !strconcat(Dt, "8"), v8i16, v8i8, IntOp>;
}

// ....with explicit extend (VABAL).
multiclass N3VLIntExtOp_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                            InstrItinClass itin, string OpcodeStr, string Dt,
                            Intrinsic IntOp, SDNode ExtOp, SDNode OpNode> {
  def v8i16 : N3VLIntExtOp<op24, op23, 0b00, op11_8, op4, itin,
                           OpcodeStr, !strconcat(Dt, "8"), v8i16, v8i8,
                           IntOp, ExtOp, OpNode>;
  def v4i32 : N3VLIntExtOp<op24, op23, 0b01, op11_8, op4, itin,
                           OpcodeStr, !strconcat(Dt, "16"), v4i32, v4i16,
                           IntOp, ExtOp, OpNode>;
  def v2i64 : N3VLIntExtOp<op24, op23, 0b10, op11_8, op4, itin,
                           OpcodeStr, !strconcat(Dt, "32"), v2i64, v2i32,
                           IntOp, ExtOp, OpNode>;
}


// Neon 2-register vector intrinsics,
//   element sizes of 8, 16 and 32 bits:
multiclass N2VInt_QHS<bits<2> op24_23, bits<2> op21_20, bits<2> op17_16,
                      bits<5> op11_7, bit op4,
                      InstrItinClass itinD, InstrItinClass itinQ,
                      string OpcodeStr, string Dt, Intrinsic IntOp> {
  // 64-bit vector types.
  def v8i8  : N2VDInt<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                      itinD, OpcodeStr, !strconcat(Dt, "8"), v8i8, v8i8, IntOp>;
  def v4i16 : N2VDInt<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                      itinD, OpcodeStr, !strconcat(Dt, "16"),v4i16,v4i16,IntOp>;
  def v2i32 : N2VDInt<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                      itinD, OpcodeStr, !strconcat(Dt, "32"),v2i32,v2i32,IntOp>;

  // 128-bit vector types.
  def v16i8 : N2VQInt<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                      itinQ, OpcodeStr, !strconcat(Dt, "8"), v16i8,v16i8,IntOp>;
  def v8i16 : N2VQInt<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                      itinQ, OpcodeStr, !strconcat(Dt, "16"),v8i16,v8i16,IntOp>;
  def v4i32 : N2VQInt<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                      itinQ, OpcodeStr, !strconcat(Dt, "32"),v4i32,v4i32,IntOp>;
}


// Neon Pairwise long 2-register intrinsics,
//   element sizes of 8, 16 and 32 bits:
multiclass N2VPLInt_QHS<bits<2> op24_23, bits<2> op21_20, bits<2> op17_16,
                        bits<5> op11_7, bit op4,
                        string OpcodeStr, string Dt, Intrinsic IntOp> {
  // 64-bit vector types.
  def v8i8  : N2VDPLInt<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                        OpcodeStr, !strconcat(Dt, "8"), v4i16, v8i8, IntOp>;
  def v4i16 : N2VDPLInt<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                        OpcodeStr, !strconcat(Dt, "16"), v2i32, v4i16, IntOp>;
  def v2i32 : N2VDPLInt<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                        OpcodeStr, !strconcat(Dt, "32"), v1i64, v2i32, IntOp>;

  // 128-bit vector types.
  def v16i8 : N2VQPLInt<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                        OpcodeStr, !strconcat(Dt, "8"), v8i16, v16i8, IntOp>;
  def v8i16 : N2VQPLInt<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                        OpcodeStr, !strconcat(Dt, "16"), v4i32, v8i16, IntOp>;
  def v4i32 : N2VQPLInt<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                        OpcodeStr, !strconcat(Dt, "32"), v2i64, v4i32, IntOp>;
}


// Neon Pairwise long 2-register accumulate intrinsics,
//   element sizes of 8, 16 and 32 bits:
multiclass N2VPLInt2_QHS<bits<2> op24_23, bits<2> op21_20, bits<2> op17_16,
                         bits<5> op11_7, bit op4,
                         string OpcodeStr, string Dt, Intrinsic IntOp> {
  // 64-bit vector types.
  def v8i8  : N2VDPLInt2<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                         OpcodeStr, !strconcat(Dt, "8"), v4i16, v8i8, IntOp>;
  def v4i16 : N2VDPLInt2<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                         OpcodeStr, !strconcat(Dt, "16"), v2i32, v4i16, IntOp>;
  def v2i32 : N2VDPLInt2<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                         OpcodeStr, !strconcat(Dt, "32"), v1i64, v2i32, IntOp>;

  // 128-bit vector types.
  def v16i8 : N2VQPLInt2<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                         OpcodeStr, !strconcat(Dt, "8"), v8i16, v16i8, IntOp>;
  def v8i16 : N2VQPLInt2<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                         OpcodeStr, !strconcat(Dt, "16"), v4i32, v8i16, IntOp>;
  def v4i32 : N2VQPLInt2<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                         OpcodeStr, !strconcat(Dt, "32"), v2i64, v4i32, IntOp>;
}


// Neon 2-register vector shift by immediate,
//   with f of either N2RegVShLFrm or N2RegVShRFrm
//   element sizes of 8, 16, 32 and 64 bits:
multiclass N2VSh_QHSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                     InstrItinClass itin, string OpcodeStr, string Dt,
                     SDNode OpNode, Format f> {
  // 64-bit vector types.
  def v8i8  : N2VDSh<op24, op23, op11_8, 0, op4, f, itin,
                     OpcodeStr, !strconcat(Dt, "8"), v8i8, OpNode> {
    let Inst{21-19} = 0b001; // imm6 = 001xxx
  }
  def v4i16 : N2VDSh<op24, op23, op11_8, 0, op4, f, itin,
                     OpcodeStr, !strconcat(Dt, "16"), v4i16, OpNode> {
    let Inst{21-20} = 0b01;  // imm6 = 01xxxx
  }
  def v2i32 : N2VDSh<op24, op23, op11_8, 0, op4, f, itin,
                     OpcodeStr, !strconcat(Dt, "32"), v2i32, OpNode> {
    let Inst{21} = 0b1;      // imm6 = 1xxxxx
  }
  def v1i64 : N2VDSh<op24, op23, op11_8, 1, op4, f, itin,
                     OpcodeStr, !strconcat(Dt, "64"), v1i64, OpNode>;
                             // imm6 = xxxxxx

  // 128-bit vector types.
  def v16i8 : N2VQSh<op24, op23, op11_8, 0, op4, f, itin,
                     OpcodeStr, !strconcat(Dt, "8"), v16i8, OpNode> {
    let Inst{21-19} = 0b001; // imm6 = 001xxx
  }
  def v8i16 : N2VQSh<op24, op23, op11_8, 0, op4, f, itin,
                     OpcodeStr, !strconcat(Dt, "16"), v8i16, OpNode> {
    let Inst{21-20} = 0b01;  // imm6 = 01xxxx
  }
  def v4i32 : N2VQSh<op24, op23, op11_8, 0, op4, f, itin,
                     OpcodeStr, !strconcat(Dt, "32"), v4i32, OpNode> {
    let Inst{21} = 0b1;      // imm6 = 1xxxxx
  }
  def v2i64 : N2VQSh<op24, op23, op11_8, 1, op4, f, itin,
                     OpcodeStr, !strconcat(Dt, "64"), v2i64, OpNode>;
                             // imm6 = xxxxxx
}

// Neon Shift-Accumulate vector operations,
//   element sizes of 8, 16, 32 and 64 bits:
multiclass N2VShAdd_QHSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                         string OpcodeStr, string Dt, SDNode ShOp> {
  // 64-bit vector types.
  def v8i8  : N2VDShAdd<op24, op23, op11_8, 0, op4,
                        OpcodeStr, !strconcat(Dt, "8"), v8i8, ShOp> {
    let Inst{21-19} = 0b001; // imm6 = 001xxx
  }
  def v4i16 : N2VDShAdd<op24, op23, op11_8, 0, op4,
                        OpcodeStr, !strconcat(Dt, "16"), v4i16, ShOp> {
    let Inst{21-20} = 0b01;  // imm6 = 01xxxx
  }
  def v2i32 : N2VDShAdd<op24, op23, op11_8, 0, op4,
                        OpcodeStr, !strconcat(Dt, "32"), v2i32, ShOp> {
    let Inst{21} = 0b1;      // imm6 = 1xxxxx
  }
  def v1i64 : N2VDShAdd<op24, op23, op11_8, 1, op4,
                        OpcodeStr, !strconcat(Dt, "64"), v1i64, ShOp>;
                             // imm6 = xxxxxx

  // 128-bit vector types.
  def v16i8 : N2VQShAdd<op24, op23, op11_8, 0, op4,
                        OpcodeStr, !strconcat(Dt, "8"), v16i8, ShOp> {
    let Inst{21-19} = 0b001; // imm6 = 001xxx
  }
  def v8i16 : N2VQShAdd<op24, op23, op11_8, 0, op4,
                        OpcodeStr, !strconcat(Dt, "16"), v8i16, ShOp> {
    let Inst{21-20} = 0b01;  // imm6 = 01xxxx
  }
  def v4i32 : N2VQShAdd<op24, op23, op11_8, 0, op4,
                        OpcodeStr, !strconcat(Dt, "32"), v4i32, ShOp> {
    let Inst{21} = 0b1;      // imm6 = 1xxxxx
  }
  def v2i64 : N2VQShAdd<op24, op23, op11_8, 1, op4,
                        OpcodeStr, !strconcat(Dt, "64"), v2i64, ShOp>;
                             // imm6 = xxxxxx
}


// Neon Shift-Insert vector operations,
//   with f of either N2RegVShLFrm or N2RegVShRFrm
//   element sizes of 8, 16, 32 and 64 bits:
multiclass N2VShIns_QHSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                         string OpcodeStr, SDNode ShOp,
                         Format f> {
  // 64-bit vector types.
  def v8i8  : N2VDShIns<op24, op23, op11_8, 0, op4,
                        f, OpcodeStr, "8", v8i8, ShOp> {
    let Inst{21-19} = 0b001; // imm6 = 001xxx
  }
  def v4i16 : N2VDShIns<op24, op23, op11_8, 0, op4,
                        f, OpcodeStr, "16", v4i16, ShOp> {
    let Inst{21-20} = 0b01;  // imm6 = 01xxxx
  }
  def v2i32 : N2VDShIns<op24, op23, op11_8, 0, op4,
                        f, OpcodeStr, "32", v2i32, ShOp> {
    let Inst{21} = 0b1;      // imm6 = 1xxxxx
  }
  def v1i64 : N2VDShIns<op24, op23, op11_8, 1, op4,
                        f, OpcodeStr, "64", v1i64, ShOp>;
                             // imm6 = xxxxxx

  // 128-bit vector types.
  def v16i8 : N2VQShIns<op24, op23, op11_8, 0, op4,
                        f, OpcodeStr, "8", v16i8, ShOp> {
    let Inst{21-19} = 0b001; // imm6 = 001xxx
  }
  def v8i16 : N2VQShIns<op24, op23, op11_8, 0, op4,
                        f, OpcodeStr, "16", v8i16, ShOp> {
    let Inst{21-20} = 0b01;  // imm6 = 01xxxx
  }
  def v4i32 : N2VQShIns<op24, op23, op11_8, 0, op4,
                        f, OpcodeStr, "32", v4i32, ShOp> {
    let Inst{21} = 0b1;      // imm6 = 1xxxxx
  }
  def v2i64 : N2VQShIns<op24, op23, op11_8, 1, op4,
                        f, OpcodeStr, "64", v2i64, ShOp>;
                             // imm6 = xxxxxx
}

// Neon Shift Long operations,
//   element sizes of 8, 16, 32 bits:
multiclass N2VLSh_QHS<bit op24, bit op23, bits<4> op11_8, bit op7, bit op6,
                      bit op4, string OpcodeStr, string Dt, SDNode OpNode> {
  def v8i16 : N2VLSh<op24, op23, op11_8, op7, op6, op4,
                 OpcodeStr, !strconcat(Dt, "8"), v8i16, v8i8, OpNode> {
    let Inst{21-19} = 0b001; // imm6 = 001xxx
  }
  def v4i32 : N2VLSh<op24, op23, op11_8, op7, op6, op4,
                  OpcodeStr, !strconcat(Dt, "16"), v4i32, v4i16, OpNode> {
    let Inst{21-20} = 0b01;  // imm6 = 01xxxx
  }
  def v2i64 : N2VLSh<op24, op23, op11_8, op7, op6, op4,
                  OpcodeStr, !strconcat(Dt, "32"), v2i64, v2i32, OpNode> {
    let Inst{21} = 0b1;      // imm6 = 1xxxxx
  }
}

// Neon Shift Narrow operations,
//   element sizes of 16, 32, 64 bits:
multiclass N2VNSh_HSD<bit op24, bit op23, bits<4> op11_8, bit op7, bit op6,
                      bit op4, InstrItinClass itin, string OpcodeStr, string Dt,
                      SDNode OpNode> {
  def v8i8 : N2VNSh<op24, op23, op11_8, op7, op6, op4, itin,
                    OpcodeStr, !strconcat(Dt, "16"), v8i8, v8i16, OpNode> {
    let Inst{21-19} = 0b001; // imm6 = 001xxx
  }
  def v4i16 : N2VNSh<op24, op23, op11_8, op7, op6, op4, itin,
                     OpcodeStr, !strconcat(Dt, "32"), v4i16, v4i32, OpNode> {
    let Inst{21-20} = 0b01;  // imm6 = 01xxxx
  }
  def v2i32 : N2VNSh<op24, op23, op11_8, op7, op6, op4, itin,
                     OpcodeStr, !strconcat(Dt, "64"), v2i32, v2i64, OpNode> {
    let Inst{21} = 0b1;      // imm6 = 1xxxxx
  }
}

//===----------------------------------------------------------------------===//
// Instruction Definitions.
//===----------------------------------------------------------------------===//

// Vector Add Operations.

//   VADD     : Vector Add (integer and floating-point)
defm VADD     : N3V_QHSD<0, 0, 0b1000, 0, IIC_VBINiD, IIC_VBINiQ, "vadd", "i",
                         add, 1>;
def  VADDfd   : N3VD<0, 0, 0b00, 0b1101, 0, IIC_VBIND, "vadd", "f32",
                     v2f32, v2f32, fadd, 1>;
def  VADDfq   : N3VQ<0, 0, 0b00, 0b1101, 0, IIC_VBINQ, "vadd", "f32",
                     v4f32, v4f32, fadd, 1>;
//   VADDL    : Vector Add Long (Q = D + D)
defm VADDLs   : N3VLExt_QHS<0,1,0b0000,0, IIC_VSHLiD, IIC_VSHLiD,
                            "vaddl", "s", add, sext, 1>;
defm VADDLu   : N3VLExt_QHS<1,1,0b0000,0, IIC_VSHLiD, IIC_VSHLiD,
                            "vaddl", "u", add, zext, 1>;
//   VADDW    : Vector Add Wide (Q = Q + D)
defm VADDWs   : N3VW_QHS<0,1,0b0001,0, "vaddw", "s", add, sext, 0>;
defm VADDWu   : N3VW_QHS<1,1,0b0001,0, "vaddw", "u", add, zext, 0>;
//   VHADD    : Vector Halving Add
defm VHADDs   : N3VInt_QHS<0, 0, 0b0000, 0, N3RegFrm,
                           IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, IIC_VBINi4Q,
                           "vhadd", "s", int_arm_neon_vhadds, 1>;
defm VHADDu   : N3VInt_QHS<1, 0, 0b0000, 0, N3RegFrm,
                           IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, IIC_VBINi4Q,
                           "vhadd", "u", int_arm_neon_vhaddu, 1>;
//   VRHADD   : Vector Rounding Halving Add
defm VRHADDs  : N3VInt_QHS<0, 0, 0b0001, 0, N3RegFrm,
                           IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, IIC_VBINi4Q,
                           "vrhadd", "s", int_arm_neon_vrhadds, 1>;
defm VRHADDu  : N3VInt_QHS<1, 0, 0b0001, 0, N3RegFrm,
                           IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, IIC_VBINi4Q,
                           "vrhadd", "u", int_arm_neon_vrhaddu, 1>;
//   VQADD    : Vector Saturating Add
defm VQADDs   : N3VInt_QHSD<0, 0, 0b0000, 1, N3RegFrm,
                            IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, IIC_VBINi4Q,
                            "vqadd", "s", int_arm_neon_vqadds, 1>;
defm VQADDu   : N3VInt_QHSD<1, 0, 0b0000, 1, N3RegFrm,
                            IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, IIC_VBINi4Q,
                            "vqadd", "u", int_arm_neon_vqaddu, 1>;
//   VADDHN   : Vector Add and Narrow Returning High Half (D = Q + Q)
defm VADDHN   : N3VNInt_HSD<0,1,0b0100,0, "vaddhn", "i",
                            int_arm_neon_vaddhn, 1>;
//   VRADDHN  : Vector Rounding Add and Narrow Returning High Half (D = Q + Q)
defm VRADDHN  : N3VNInt_HSD<1,1,0b0100,0, "vraddhn", "i",
                            int_arm_neon_vraddhn, 1>;

// Vector Multiply Operations.

//   VMUL     : Vector Multiply (integer, polynomial and floating-point)
defm VMUL     : N3V_QHS<0, 0, 0b1001, 1, IIC_VMULi16D, IIC_VMULi32D,
                        IIC_VMULi16Q, IIC_VMULi32Q, "vmul", "i", mul, 1>;
def  VMULpd   : N3VDInt<1, 0, 0b00, 0b1001, 1, N3RegFrm, IIC_VMULi16D, "vmul",
                        "p8", v8i8, v8i8, int_arm_neon_vmulp, 1>;
def  VMULpq   : N3VQInt<1, 0, 0b00, 0b1001, 1, N3RegFrm, IIC_VMULi16Q, "vmul",
                        "p8", v16i8, v16i8, int_arm_neon_vmulp, 1>;
def  VMULfd   : N3VD<1, 0, 0b00, 0b1101, 1, IIC_VFMULD, "vmul", "f32",
                     v2f32, v2f32, fmul, 1>;
def  VMULfq   : N3VQ<1, 0, 0b00, 0b1101, 1, IIC_VFMULQ, "vmul", "f32",
                     v4f32, v4f32, fmul, 1>;
defm VMULsl   : N3VSL_HS<0b1000, "vmul", "i", mul>;
def  VMULslfd : N3VDSL<0b10, 0b1001, IIC_VBIND, "vmul", "f32", v2f32, fmul>;
def  VMULslfq : N3VQSL<0b10, 0b1001, IIC_VBINQ, "vmul", "f32", v4f32,
                       v2f32, fmul>;

def : Pat<(v8i16 (mul (v8i16 QPR:$src1),
                      (v8i16 (NEONvduplane (v8i16 QPR:$src2), imm:$lane)))),
          (v8i16 (VMULslv8i16 (v8i16 QPR:$src1),
                              (v4i16 (EXTRACT_SUBREG QPR:$src2,
                                      (DSubReg_i16_reg imm:$lane))),
                              (SubReg_i16_lane imm:$lane)))>;
def : Pat<(v4i32 (mul (v4i32 QPR:$src1),
                      (v4i32 (NEONvduplane (v4i32 QPR:$src2), imm:$lane)))),
          (v4i32 (VMULslv4i32 (v4i32 QPR:$src1),
                              (v2i32 (EXTRACT_SUBREG QPR:$src2,
                                      (DSubReg_i32_reg imm:$lane))),
                              (SubReg_i32_lane imm:$lane)))>;
def : Pat<(v4f32 (fmul (v4f32 QPR:$src1),
                       (v4f32 (NEONvduplane (v4f32 QPR:$src2), imm:$lane)))),
          (v4f32 (VMULslfq (v4f32 QPR:$src1),
                           (v2f32 (EXTRACT_SUBREG QPR:$src2,
                                   (DSubReg_i32_reg imm:$lane))),
                           (SubReg_i32_lane imm:$lane)))>;

//   VQDMULH  : Vector Saturating Doubling Multiply Returning High Half
defm VQDMULH  : N3VInt_HS<0, 0, 0b1011, 0, N3RegFrm, IIC_VMULi16D, IIC_VMULi32D,
                          IIC_VMULi16Q, IIC_VMULi32Q, 
                          "vqdmulh", "s", int_arm_neon_vqdmulh, 1>;
defm VQDMULHsl: N3VIntSL_HS<0b1100, IIC_VMULi16D, IIC_VMULi32D,
                            IIC_VMULi16Q, IIC_VMULi32Q,
                            "vqdmulh", "s",  int_arm_neon_vqdmulh>;
def : Pat<(v8i16 (int_arm_neon_vqdmulh (v8i16 QPR:$src1),
                                       (v8i16 (NEONvduplane (v8i16 QPR:$src2),
                                                            imm:$lane)))),
          (v8i16 (VQDMULHslv8i16 (v8i16 QPR:$src1),
                                 (v4i16 (EXTRACT_SUBREG QPR:$src2,
                                         (DSubReg_i16_reg imm:$lane))),
                                 (SubReg_i16_lane imm:$lane)))>;
def : Pat<(v4i32 (int_arm_neon_vqdmulh (v4i32 QPR:$src1),
                                       (v4i32 (NEONvduplane (v4i32 QPR:$src2),
                                                            imm:$lane)))),
          (v4i32 (VQDMULHslv4i32 (v4i32 QPR:$src1),
                                 (v2i32 (EXTRACT_SUBREG QPR:$src2,
                                         (DSubReg_i32_reg imm:$lane))),
                                 (SubReg_i32_lane imm:$lane)))>;

//   VQRDMULH : Vector Rounding Saturating Doubling Multiply Returning High Half
defm VQRDMULH   : N3VInt_HS<1, 0, 0b1011, 0, N3RegFrm,
                            IIC_VMULi16D,IIC_VMULi32D,IIC_VMULi16Q,IIC_VMULi32Q,
                            "vqrdmulh", "s", int_arm_neon_vqrdmulh, 1>;
defm VQRDMULHsl : N3VIntSL_HS<0b1101, IIC_VMULi16D, IIC_VMULi32D,
                              IIC_VMULi16Q, IIC_VMULi32Q,
                              "vqrdmulh", "s",  int_arm_neon_vqrdmulh>;
def : Pat<(v8i16 (int_arm_neon_vqrdmulh (v8i16 QPR:$src1),
                                        (v8i16 (NEONvduplane (v8i16 QPR:$src2),
                                                             imm:$lane)))),
          (v8i16 (VQRDMULHslv8i16 (v8i16 QPR:$src1),
                                  (v4i16 (EXTRACT_SUBREG QPR:$src2,
                                          (DSubReg_i16_reg imm:$lane))),
                                  (SubReg_i16_lane imm:$lane)))>;
def : Pat<(v4i32 (int_arm_neon_vqrdmulh (v4i32 QPR:$src1),
                                        (v4i32 (NEONvduplane (v4i32 QPR:$src2),
                                                             imm:$lane)))),
          (v4i32 (VQRDMULHslv4i32 (v4i32 QPR:$src1),
                                  (v2i32 (EXTRACT_SUBREG QPR:$src2,
                                          (DSubReg_i32_reg imm:$lane))),
                                  (SubReg_i32_lane imm:$lane)))>;

//   VMULL    : Vector Multiply Long (integer and polynomial) (Q = D * D)
defm VMULLs   : N3VL_QHS<0,1,0b1100,0, IIC_VMULi16D, IIC_VMULi32D,
                         "vmull", "s", NEONvmulls, 1>;
defm VMULLu   : N3VL_QHS<1,1,0b1100,0, IIC_VMULi16D, IIC_VMULi32D,
                         "vmull", "u", NEONvmullu, 1>;
def  VMULLp   : N3VLInt<0, 1, 0b00, 0b1110, 0, IIC_VMULi16D, "vmull", "p8",
                        v8i16, v8i8, int_arm_neon_vmullp, 1>;
defm VMULLsls : N3VLSL_HS<0, 0b1010, IIC_VMULi16D, "vmull", "s", NEONvmulls>;
defm VMULLslu : N3VLSL_HS<1, 0b1010, IIC_VMULi16D, "vmull", "u", NEONvmullu>;

//   VQDMULL  : Vector Saturating Doubling Multiply Long (Q = D * D)
defm VQDMULL  : N3VLInt_HS<0,1,0b1101,0, IIC_VMULi16D, IIC_VMULi32D,
                           "vqdmull", "s", int_arm_neon_vqdmull, 1>;
defm VQDMULLsl: N3VLIntSL_HS<0, 0b1011, IIC_VMULi16D,
                             "vqdmull", "s", int_arm_neon_vqdmull>;

// Vector Multiply-Accumulate and Multiply-Subtract Operations.

//   VMLA     : Vector Multiply Accumulate (integer and floating-point)
defm VMLA     : N3VMulOp_QHS<0, 0, 0b1001, 0, IIC_VMACi16D, IIC_VMACi32D,
                             IIC_VMACi16Q, IIC_VMACi32Q, "vmla", "i", add>;
def  VMLAfd   : N3VDMulOp<0, 0, 0b00, 0b1101, 1, IIC_VMACD, "vmla", "f32",
                          v2f32, fmul, fadd>;
def  VMLAfq   : N3VQMulOp<0, 0, 0b00, 0b1101, 1, IIC_VMACQ, "vmla", "f32",
                          v4f32, fmul, fadd>;
defm VMLAsl   : N3VMulOpSL_HS<0b0000, IIC_VMACi16D, IIC_VMACi32D,
                              IIC_VMACi16Q, IIC_VMACi32Q, "vmla", "i", add>;
def  VMLAslfd : N3VDMulOpSL<0b10, 0b0001, IIC_VMACD, "vmla", "f32",
                            v2f32, fmul, fadd>;
def  VMLAslfq : N3VQMulOpSL<0b10, 0b0001, IIC_VMACQ, "vmla", "f32",
                            v4f32, v2f32, fmul, fadd>;

def : Pat<(v8i16 (add (v8i16 QPR:$src1),
                  (mul (v8i16 QPR:$src2),
                       (v8i16 (NEONvduplane (v8i16 QPR:$src3), imm:$lane))))),
          (v8i16 (VMLAslv8i16 (v8i16 QPR:$src1), (v8i16 QPR:$src2),
                              (v4i16 (EXTRACT_SUBREG QPR:$src3,
                                      (DSubReg_i16_reg imm:$lane))),
                              (SubReg_i16_lane imm:$lane)))>;

def : Pat<(v4i32 (add (v4i32 QPR:$src1),
                  (mul (v4i32 QPR:$src2),
                       (v4i32 (NEONvduplane (v4i32 QPR:$src3), imm:$lane))))),
          (v4i32 (VMLAslv4i32 (v4i32 QPR:$src1), (v4i32 QPR:$src2),
                              (v2i32 (EXTRACT_SUBREG QPR:$src3,
                                      (DSubReg_i32_reg imm:$lane))),
                              (SubReg_i32_lane imm:$lane)))>;

def : Pat<(v4f32 (fadd (v4f32 QPR:$src1),
                  (fmul (v4f32 QPR:$src2),
                        (v4f32 (NEONvduplane (v4f32 QPR:$src3), imm:$lane))))),
          (v4f32 (VMLAslfq (v4f32 QPR:$src1),
                           (v4f32 QPR:$src2),
                           (v2f32 (EXTRACT_SUBREG QPR:$src3,
                                   (DSubReg_i32_reg imm:$lane))),
                           (SubReg_i32_lane imm:$lane)))>;

//   VMLAL    : Vector Multiply Accumulate Long (Q += D * D)
defm VMLALs   : N3VLMulOp_QHS<0,1,0b1000,0, IIC_VMACi16D, IIC_VMACi32D,
                              "vmlal", "s", NEONvmulls, add>;
defm VMLALu   : N3VLMulOp_QHS<1,1,0b1000,0, IIC_VMACi16D, IIC_VMACi32D,
                              "vmlal", "u", NEONvmullu, add>;

defm VMLALsls : N3VLMulOpSL_HS<0, 0b0010, "vmlal", "s", NEONvmulls, add>;
defm VMLALslu : N3VLMulOpSL_HS<1, 0b0010, "vmlal", "u", NEONvmullu, add>;

//   VQDMLAL  : Vector Saturating Doubling Multiply Accumulate Long (Q += D * D)
defm VQDMLAL  : N3VLInt3_HS<0, 1, 0b1001, 0, IIC_VMACi16D, IIC_VMACi32D,
                            "vqdmlal", "s", int_arm_neon_vqdmlal>;
defm VQDMLALsl: N3VLInt3SL_HS<0, 0b0011, "vqdmlal", "s", int_arm_neon_vqdmlal>;

//   VMLS     : Vector Multiply Subtract (integer and floating-point)
defm VMLS     : N3VMulOp_QHS<1, 0, 0b1001, 0, IIC_VMACi16D, IIC_VMACi32D,
                             IIC_VMACi16Q, IIC_VMACi32Q, "vmls", "i", sub>;
def  VMLSfd   : N3VDMulOp<0, 0, 0b10, 0b1101, 1, IIC_VMACD, "vmls", "f32",
                          v2f32, fmul, fsub>;
def  VMLSfq   : N3VQMulOp<0, 0, 0b10, 0b1101, 1, IIC_VMACQ, "vmls", "f32",
                          v4f32, fmul, fsub>;
defm VMLSsl   : N3VMulOpSL_HS<0b0100, IIC_VMACi16D, IIC_VMACi32D,
                              IIC_VMACi16Q, IIC_VMACi32Q, "vmls", "i", sub>;
def  VMLSslfd : N3VDMulOpSL<0b10, 0b0101, IIC_VMACD, "vmls", "f32",
                            v2f32, fmul, fsub>;
def  VMLSslfq : N3VQMulOpSL<0b10, 0b0101, IIC_VMACQ, "vmls", "f32",
                            v4f32, v2f32, fmul, fsub>;

def : Pat<(v8i16 (sub (v8i16 QPR:$src1),
                  (mul (v8i16 QPR:$src2),
                       (v8i16 (NEONvduplane (v8i16 QPR:$src3), imm:$lane))))),
          (v8i16 (VMLSslv8i16 (v8i16 QPR:$src1), (v8i16 QPR:$src2),
                              (v4i16 (EXTRACT_SUBREG QPR:$src3,
                                      (DSubReg_i16_reg imm:$lane))),
                              (SubReg_i16_lane imm:$lane)))>;

def : Pat<(v4i32 (sub (v4i32 QPR:$src1),
                  (mul (v4i32 QPR:$src2),
                     (v4i32 (NEONvduplane (v4i32 QPR:$src3), imm:$lane))))),
          (v4i32 (VMLSslv4i32 (v4i32 QPR:$src1), (v4i32 QPR:$src2),
                              (v2i32 (EXTRACT_SUBREG QPR:$src3,
                                      (DSubReg_i32_reg imm:$lane))),
                              (SubReg_i32_lane imm:$lane)))>;

def : Pat<(v4f32 (fsub (v4f32 QPR:$src1),
                  (fmul (v4f32 QPR:$src2),
                        (v4f32 (NEONvduplane (v4f32 QPR:$src3), imm:$lane))))),
          (v4f32 (VMLSslfq (v4f32 QPR:$src1), (v4f32 QPR:$src2),
                           (v2f32 (EXTRACT_SUBREG QPR:$src3,
                                   (DSubReg_i32_reg imm:$lane))),
                           (SubReg_i32_lane imm:$lane)))>;

//   VMLSL    : Vector Multiply Subtract Long (Q -= D * D)
defm VMLSLs   : N3VLMulOp_QHS<0,1,0b1010,0, IIC_VMACi16D, IIC_VMACi32D,
                              "vmlsl", "s", NEONvmulls, sub>;
defm VMLSLu   : N3VLMulOp_QHS<1,1,0b1010,0, IIC_VMACi16D, IIC_VMACi32D,
                              "vmlsl", "u", NEONvmullu, sub>;

defm VMLSLsls : N3VLMulOpSL_HS<0, 0b0110, "vmlsl", "s", NEONvmulls, sub>;
defm VMLSLslu : N3VLMulOpSL_HS<1, 0b0110, "vmlsl", "u", NEONvmullu, sub>;

//   VQDMLSL  : Vector Saturating Doubling Multiply Subtract Long (Q -= D * D)
defm VQDMLSL  : N3VLInt3_HS<0, 1, 0b1011, 0, IIC_VMACi16D, IIC_VMACi32D,
                            "vqdmlsl", "s", int_arm_neon_vqdmlsl>;
defm VQDMLSLsl: N3VLInt3SL_HS<0, 0b111, "vqdmlsl", "s", int_arm_neon_vqdmlsl>;

// Vector Subtract Operations.

//   VSUB     : Vector Subtract (integer and floating-point)
defm VSUB     : N3V_QHSD<1, 0, 0b1000, 0, IIC_VSUBiD, IIC_VSUBiQ,
                         "vsub", "i", sub, 0>;
def  VSUBfd   : N3VD<0, 0, 0b10, 0b1101, 0, IIC_VBIND, "vsub", "f32",
                     v2f32, v2f32, fsub, 0>;
def  VSUBfq   : N3VQ<0, 0, 0b10, 0b1101, 0, IIC_VBINQ, "vsub", "f32",
                     v4f32, v4f32, fsub, 0>;
//   VSUBL    : Vector Subtract Long (Q = D - D)
defm VSUBLs   : N3VLExt_QHS<0,1,0b0010,0, IIC_VSHLiD, IIC_VSHLiD,
                            "vsubl", "s", sub, sext, 0>;
defm VSUBLu   : N3VLExt_QHS<1,1,0b0010,0, IIC_VSHLiD, IIC_VSHLiD,
                            "vsubl", "u", sub, zext, 0>;
//   VSUBW    : Vector Subtract Wide (Q = Q - D)
defm VSUBWs   : N3VW_QHS<0,1,0b0011,0, "vsubw", "s", sub, sext, 0>;
defm VSUBWu   : N3VW_QHS<1,1,0b0011,0, "vsubw", "u", sub, zext, 0>;
//   VHSUB    : Vector Halving Subtract
defm VHSUBs   : N3VInt_QHS<0, 0, 0b0010, 0, N3RegFrm,
                           IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, IIC_VSUBi4Q,
                           "vhsub", "s", int_arm_neon_vhsubs, 0>;
defm VHSUBu   : N3VInt_QHS<1, 0, 0b0010, 0, N3RegFrm,
                           IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, IIC_VSUBi4Q,
                           "vhsub", "u", int_arm_neon_vhsubu, 0>;
//   VQSUB    : Vector Saturing Subtract
defm VQSUBs   : N3VInt_QHSD<0, 0, 0b0010, 1, N3RegFrm,
                            IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, IIC_VSUBi4Q,
                            "vqsub", "s", int_arm_neon_vqsubs, 0>;
defm VQSUBu   : N3VInt_QHSD<1, 0, 0b0010, 1, N3RegFrm,
                            IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, IIC_VSUBi4Q,
                            "vqsub", "u", int_arm_neon_vqsubu, 0>;
//   VSUBHN   : Vector Subtract and Narrow Returning High Half (D = Q - Q)
defm VSUBHN   : N3VNInt_HSD<0,1,0b0110,0, "vsubhn", "i",
                            int_arm_neon_vsubhn, 0>;
//   VRSUBHN  : Vector Rounding Subtract and Narrow Returning High Half (D=Q-Q)
defm VRSUBHN  : N3VNInt_HSD<1,1,0b0110,0, "vrsubhn", "i",
                            int_arm_neon_vrsubhn, 0>;

// Vector Comparisons.

//   VCEQ     : Vector Compare Equal
defm VCEQ     : N3V_QHS<1, 0, 0b1000, 1, IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q,
                        IIC_VSUBi4Q, "vceq", "i", NEONvceq, 1>;
def  VCEQfd   : N3VD<0,0,0b00,0b1110,0, IIC_VBIND, "vceq", "f32", v2i32, v2f32,
                     NEONvceq, 1>;
def  VCEQfq   : N3VQ<0,0,0b00,0b1110,0, IIC_VBINQ, "vceq", "f32", v4i32, v4f32,
                     NEONvceq, 1>;
// For disassembly only.
defm VCEQz    : N2V_QHS_cmp<0b11, 0b11, 0b01, 0b00010, 0, "vceq", "i",
                            "$dst, $src, #0">;

//   VCGE     : Vector Compare Greater Than or Equal
defm VCGEs    : N3V_QHS<0, 0, 0b0011, 1, IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q,
                        IIC_VSUBi4Q, "vcge", "s", NEONvcge, 0>;
defm VCGEu    : N3V_QHS<1, 0, 0b0011, 1, IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, 
                        IIC_VSUBi4Q, "vcge", "u", NEONvcgeu, 0>;
def  VCGEfd   : N3VD<1,0,0b00,0b1110,0, IIC_VBIND, "vcge", "f32", v2i32, v2f32,
                     NEONvcge, 0>;
def  VCGEfq   : N3VQ<1,0,0b00,0b1110,0, IIC_VBINQ, "vcge", "f32", v4i32, v4f32,
                     NEONvcge, 0>;
// For disassembly only.
// FIXME: This instruction's encoding MAY NOT BE correct.
defm VCGEz    : N2V_QHS_cmp<0b11, 0b11, 0b01, 0b00001, 0, "vcge", "s",
                            "$dst, $src, #0">;
// For disassembly only.
// FIXME: This instruction's encoding MAY NOT BE correct.
defm VCLEz    : N2V_QHS_cmp<0b11, 0b11, 0b01, 0b00011, 0, "vcle", "s",
                            "$dst, $src, #0">;

//   VCGT     : Vector Compare Greater Than
defm VCGTs    : N3V_QHS<0, 0, 0b0011, 0, IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q,
                        IIC_VSUBi4Q, "vcgt", "s", NEONvcgt, 0>;
defm VCGTu    : N3V_QHS<1, 0, 0b0011, 0, IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q,
                        IIC_VSUBi4Q, "vcgt", "u", NEONvcgtu, 0>;
def  VCGTfd   : N3VD<1,0,0b10,0b1110,0, IIC_VBIND, "vcgt", "f32", v2i32, v2f32,
                     NEONvcgt, 0>;
def  VCGTfq   : N3VQ<1,0,0b10,0b1110,0, IIC_VBINQ, "vcgt", "f32", v4i32, v4f32,
                     NEONvcgt, 0>;
// For disassembly only.
// FIXME: This instruction's encoding MAY NOT BE correct.
defm VCGTz    : N2V_QHS_cmp<0b11, 0b11, 0b01, 0b00000, 0, "vcgt", "s",
                            "$dst, $src, #0">;
// For disassembly only.
// FIXME: This instruction's encoding MAY NOT BE correct.
defm VCLTz    : N2V_QHS_cmp<0b11, 0b11, 0b01, 0b00100, 0, "vclt", "s",
                            "$dst, $src, #0">;

//   VACGE    : Vector Absolute Compare Greater Than or Equal (aka VCAGE)
def  VACGEd   : N3VDInt<1, 0, 0b00, 0b1110, 1, N3RegFrm, IIC_VBIND, "vacge",
                        "f32", v2i32, v2f32, int_arm_neon_vacged, 0>;
def  VACGEq   : N3VQInt<1, 0, 0b00, 0b1110, 1, N3RegFrm, IIC_VBINQ, "vacge",
                        "f32", v4i32, v4f32, int_arm_neon_vacgeq, 0>;
//   VACGT    : Vector Absolute Compare Greater Than (aka VCAGT)
def  VACGTd   : N3VDInt<1, 0, 0b10, 0b1110, 1, N3RegFrm, IIC_VBIND, "vacgt",
                        "f32", v2i32, v2f32, int_arm_neon_vacgtd, 0>;
def  VACGTq   : N3VQInt<1, 0, 0b10, 0b1110, 1, N3RegFrm, IIC_VBINQ, "vacgt",
                        "f32", v4i32, v4f32, int_arm_neon_vacgtq, 0>;
//   VTST     : Vector Test Bits
defm VTST     : N3V_QHS<0, 0, 0b1000, 1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, 
                        IIC_VBINi4Q, "vtst", "", NEONvtst, 1>;

// Vector Bitwise Operations.

def vnotd : PatFrag<(ops node:$in),
                    (xor node:$in, (bitconvert (v8i8 NEONimmAllOnesV)))>;
def vnotq : PatFrag<(ops node:$in),
                    (xor node:$in, (bitconvert (v16i8 NEONimmAllOnesV)))>;


//   VAND     : Vector Bitwise AND
def  VANDd    : N3VDX<0, 0, 0b00, 0b0001, 1, IIC_VBINiD, "vand",
                      v2i32, v2i32, and, 1>;
def  VANDq    : N3VQX<0, 0, 0b00, 0b0001, 1, IIC_VBINiQ, "vand",
                      v4i32, v4i32, and, 1>;

//   VEOR     : Vector Bitwise Exclusive OR
def  VEORd    : N3VDX<1, 0, 0b00, 0b0001, 1, IIC_VBINiD, "veor",
                      v2i32, v2i32, xor, 1>;
def  VEORq    : N3VQX<1, 0, 0b00, 0b0001, 1, IIC_VBINiQ, "veor",
                      v4i32, v4i32, xor, 1>;

//   VORR     : Vector Bitwise OR
def  VORRd    : N3VDX<0, 0, 0b10, 0b0001, 1, IIC_VBINiD, "vorr",
                      v2i32, v2i32, or, 1>;
def  VORRq    : N3VQX<0, 0, 0b10, 0b0001, 1, IIC_VBINiQ, "vorr",
                      v4i32, v4i32, or, 1>;

//   VBIC     : Vector Bitwise Bit Clear (AND NOT)
def  VBICd    : N3VX<0, 0, 0b01, 0b0001, 0, 1, (outs DPR:$dst),
                     (ins DPR:$src1, DPR:$src2), N3RegFrm, IIC_VBINiD,
                     "vbic", "$dst, $src1, $src2", "",
                     [(set DPR:$dst, (v2i32 (and DPR:$src1,
                                                 (vnotd DPR:$src2))))]>;
def  VBICq    : N3VX<0, 0, 0b01, 0b0001, 1, 1, (outs QPR:$dst),
                     (ins QPR:$src1, QPR:$src2), N3RegFrm, IIC_VBINiQ,
                     "vbic", "$dst, $src1, $src2", "",
                     [(set QPR:$dst, (v4i32 (and QPR:$src1,
                                                 (vnotq QPR:$src2))))]>;

//   VORN     : Vector Bitwise OR NOT
def  VORNd    : N3VX<0, 0, 0b11, 0b0001, 0, 1, (outs DPR:$dst),
                     (ins DPR:$src1, DPR:$src2), N3RegFrm, IIC_VBINiD,
                     "vorn", "$dst, $src1, $src2", "",
                     [(set DPR:$dst, (v2i32 (or DPR:$src1,
                                                (vnotd DPR:$src2))))]>;
def  VORNq    : N3VX<0, 0, 0b11, 0b0001, 1, 1, (outs QPR:$dst),
                     (ins QPR:$src1, QPR:$src2), N3RegFrm, IIC_VBINiQ,
                     "vorn", "$dst, $src1, $src2", "",
                     [(set QPR:$dst, (v4i32 (or QPR:$src1,
                                                (vnotq QPR:$src2))))]>;

//   VMVN     : Vector Bitwise NOT (Immediate)

let isReMaterializable = 1 in {

def VMVNv4i16 : N1ModImm<1, 0b000, {1,0,?,0}, 0, 0, 1, 1, (outs DPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmvn", "i16", "$dst, $SIMM", "",
                         [(set DPR:$dst, (v4i16 (NEONvmvnImm timm:$SIMM)))]> {
  let Inst{9} = SIMM{9};
}

def VMVNv8i16 : N1ModImm<1, 0b000, {1,0,?,0}, 0, 1, 1, 1, (outs QPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmvn", "i16", "$dst, $SIMM", "",
                         [(set QPR:$dst, (v8i16 (NEONvmvnImm timm:$SIMM)))]> {
  let Inst{9} = SIMM{9};
}

def VMVNv2i32 : N1ModImm<1, 0b000, {?,?,?,?}, 0, 0, 1, 1, (outs DPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmvn", "i32", "$dst, $SIMM", "",
                         [(set DPR:$dst, (v2i32 (NEONvmvnImm timm:$SIMM)))]> {
  let Inst{11-8} = SIMM{11-8};
}

def VMVNv4i32 : N1ModImm<1, 0b000, {?,?,?,?}, 0, 1, 1, 1, (outs QPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmvn", "i32", "$dst, $SIMM", "",
                         [(set QPR:$dst, (v4i32 (NEONvmvnImm timm:$SIMM)))]> {
  let Inst{11-8} = SIMM{11-8};
}
}

//   VMVN     : Vector Bitwise NOT
def  VMVNd    : N2VX<0b11, 0b11, 0b00, 0b00, 0b01011, 0, 0,
                     (outs DPR:$dst), (ins DPR:$src), IIC_VSUBiD,
                     "vmvn", "$dst, $src", "",
                     [(set DPR:$dst, (v2i32 (vnotd DPR:$src)))]>;
def  VMVNq    : N2VX<0b11, 0b11, 0b00, 0b00, 0b01011, 1, 0,
                     (outs QPR:$dst), (ins QPR:$src), IIC_VSUBiD,
                     "vmvn", "$dst, $src", "",
                     [(set QPR:$dst, (v4i32 (vnotq QPR:$src)))]>;
def : Pat<(v2i32 (vnotd DPR:$src)), (VMVNd DPR:$src)>;
def : Pat<(v4i32 (vnotq QPR:$src)), (VMVNq QPR:$src)>;

//   VBSL     : Vector Bitwise Select
def  VBSLd    : N3VX<1, 0, 0b01, 0b0001, 0, 1, (outs DPR:$Vd),
                     (ins DPR:$src1, DPR:$Vn, DPR:$Vm),
                     N3RegFrm, IIC_VCNTiD,
                     "vbsl", "$Vd, $Vn, $Vm", "$src1 = $Vd",
                     [(set DPR:$Vd,
                       (v2i32 (or (and DPR:$Vn, DPR:$src1),
                                  (and DPR:$Vm, (vnotd DPR:$src1)))))]>;
def  VBSLq    : N3VX<1, 0, 0b01, 0b0001, 1, 1, (outs QPR:$Vd),
                     (ins QPR:$src1, QPR:$Vn, QPR:$Vm),
                     N3RegFrm, IIC_VCNTiQ,
                     "vbsl", "$Vd, $Vn, $Vm", "$src1 = $Vd",
                     [(set QPR:$Vd,
                       (v4i32 (or (and QPR:$Vn, QPR:$src1),
                                  (and QPR:$Vm, (vnotq QPR:$src1)))))]>;

//   VBIF     : Vector Bitwise Insert if False
//              like VBSL but with: "vbif $dst, $src3, $src1", "$src2 = $dst",
// FIXME: This instruction's encoding MAY NOT BE correct.
def  VBIFd    : N3VX<1, 0, 0b11, 0b0001, 0, 1,
                     (outs DPR:$Vd), (ins DPR:$src1, DPR:$Vn, DPR:$Vm),
                     N3RegFrm, IIC_VBINiD,
                     "vbif", "$Vd, $Vn, $Vm", "$src1 = $Vd",
                     [/* For disassembly only; pattern left blank */]>;
def  VBIFq    : N3VX<1, 0, 0b11, 0b0001, 1, 1,
                     (outs QPR:$Vd), (ins QPR:$src1, QPR:$Vn, QPR:$Vm),
                     N3RegFrm, IIC_VBINiQ,
                     "vbif", "$Vd, $Vn, $Vm", "$src1 = $Vd",
                     [/* For disassembly only; pattern left blank */]>;

//   VBIT     : Vector Bitwise Insert if True
//              like VBSL but with: "vbit $dst, $src2, $src1", "$src3 = $dst",
// FIXME: This instruction's encoding MAY NOT BE correct.
def  VBITd    : N3VX<1, 0, 0b10, 0b0001, 0, 1,
                     (outs DPR:$Vd), (ins DPR:$src1, DPR:$Vn, DPR:$Vm),
                     N3RegFrm, IIC_VBINiD,
                     "vbit", "$Vd, $Vn, $Vm", "$src1 = $Vd",
                     [/* For disassembly only; pattern left blank */]>;
def  VBITq    : N3VX<1, 0, 0b10, 0b0001, 1, 1,
                     (outs QPR:$Vd), (ins QPR:$src1, QPR:$Vn, QPR:$Vm),
                     N3RegFrm, IIC_VBINiQ,
                     "vbit", "$Vd, $Vn, $Vm", "$src1 = $Vd",
                     [/* For disassembly only; pattern left blank */]>;

// VBIT/VBIF are not yet implemented.  The TwoAddress pass will not go looking
// for equivalent operations with different register constraints; it just
// inserts copies.

// Vector Absolute Differences.

//   VABD     : Vector Absolute Difference
defm VABDs    : N3VInt_QHS<0, 0, 0b0111, 0, N3RegFrm,
                           IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, IIC_VSUBi4Q,
                           "vabd", "s", int_arm_neon_vabds, 1>;
defm VABDu    : N3VInt_QHS<1, 0, 0b0111, 0, N3RegFrm,
                           IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, IIC_VSUBi4Q,
                           "vabd", "u", int_arm_neon_vabdu, 1>;
def  VABDfd   : N3VDInt<1, 0, 0b10, 0b1101, 0, N3RegFrm, IIC_VBIND,
                        "vabd", "f32", v2f32, v2f32, int_arm_neon_vabds, 1>;
def  VABDfq   : N3VQInt<1, 0, 0b10, 0b1101, 0, N3RegFrm, IIC_VBINQ,
                        "vabd", "f32", v4f32, v4f32, int_arm_neon_vabds, 1>;

//   VABDL    : Vector Absolute Difference Long (Q = | D - D |)
defm VABDLs   : N3VLIntExt_QHS<0,1,0b0111,0, IIC_VSUBi4Q,
                               "vabdl", "s", int_arm_neon_vabds, zext, 1>;
defm VABDLu   : N3VLIntExt_QHS<1,1,0b0111,0, IIC_VSUBi4Q,
                               "vabdl", "u", int_arm_neon_vabdu, zext, 1>;

//   VABA     : Vector Absolute Difference and Accumulate
defm VABAs    : N3VIntOp_QHS<0,0,0b0111,1, IIC_VABAD, IIC_VABAQ,
                             "vaba", "s", int_arm_neon_vabds, add>;
defm VABAu    : N3VIntOp_QHS<1,0,0b0111,1, IIC_VABAD, IIC_VABAQ,
                             "vaba", "u", int_arm_neon_vabdu, add>;

//   VABAL    : Vector Absolute Difference and Accumulate Long (Q += | D - D |)
defm VABALs   : N3VLIntExtOp_QHS<0,1,0b0101,0, IIC_VABAD,
                                 "vabal", "s", int_arm_neon_vabds, zext, add>;
defm VABALu   : N3VLIntExtOp_QHS<1,1,0b0101,0, IIC_VABAD,
                                 "vabal", "u", int_arm_neon_vabdu, zext, add>;

// Vector Maximum and Minimum.

//   VMAX     : Vector Maximum
defm VMAXs    : N3VInt_QHS<0, 0, 0b0110, 0, N3RegFrm,
                           IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, IIC_VSUBi4Q,
                           "vmax", "s", int_arm_neon_vmaxs, 1>;
defm VMAXu    : N3VInt_QHS<1, 0, 0b0110, 0, N3RegFrm,
                           IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, IIC_VSUBi4Q,
                           "vmax", "u", int_arm_neon_vmaxu, 1>;
def  VMAXfd   : N3VDInt<0, 0, 0b00, 0b1111, 0, N3RegFrm, IIC_VBIND,
                        "vmax", "f32",
                        v2f32, v2f32, int_arm_neon_vmaxs, 1>;
def  VMAXfq   : N3VQInt<0, 0, 0b00, 0b1111, 0, N3RegFrm, IIC_VBINQ,
                        "vmax", "f32",
                        v4f32, v4f32, int_arm_neon_vmaxs, 1>;

//   VMIN     : Vector Minimum
defm VMINs    : N3VInt_QHS<0, 0, 0b0110, 1, N3RegFrm,
                           IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, IIC_VSUBi4Q,
                           "vmin", "s", int_arm_neon_vmins, 1>;
defm VMINu    : N3VInt_QHS<1, 0, 0b0110, 1, N3RegFrm,
                           IIC_VSUBi4D, IIC_VSUBi4D, IIC_VSUBi4Q, IIC_VSUBi4Q,
                           "vmin", "u", int_arm_neon_vminu, 1>;
def  VMINfd   : N3VDInt<0, 0, 0b10, 0b1111, 0, N3RegFrm, IIC_VBIND,
                        "vmin", "f32",
                        v2f32, v2f32, int_arm_neon_vmins, 1>;
def  VMINfq   : N3VQInt<0, 0, 0b10, 0b1111, 0, N3RegFrm, IIC_VBINQ,
                        "vmin", "f32",
                        v4f32, v4f32, int_arm_neon_vmins, 1>;

// Vector Pairwise Operations.

//   VPADD    : Vector Pairwise Add
def  VPADDi8  : N3VDInt<0, 0, 0b00, 0b1011, 1, N3RegFrm, IIC_VSHLiD,
                        "vpadd", "i8",
                        v8i8, v8i8, int_arm_neon_vpadd, 0>;
def  VPADDi16 : N3VDInt<0, 0, 0b01, 0b1011, 1, N3RegFrm, IIC_VSHLiD,
                        "vpadd", "i16",
                        v4i16, v4i16, int_arm_neon_vpadd, 0>;
def  VPADDi32 : N3VDInt<0, 0, 0b10, 0b1011, 1, N3RegFrm, IIC_VSHLiD,
                        "vpadd", "i32",
                        v2i32, v2i32, int_arm_neon_vpadd, 0>;
def  VPADDf   : N3VDInt<1, 0, 0b00, 0b1101, 0, N3RegFrm, 
                        IIC_VPBIND, "vpadd", "f32",
                        v2f32, v2f32, int_arm_neon_vpadd, 0>;

//   VPADDL   : Vector Pairwise Add Long
defm VPADDLs  : N2VPLInt_QHS<0b11, 0b11, 0b00, 0b00100, 0, "vpaddl", "s",
                             int_arm_neon_vpaddls>;
defm VPADDLu  : N2VPLInt_QHS<0b11, 0b11, 0b00, 0b00101, 0, "vpaddl", "u",
                             int_arm_neon_vpaddlu>;

//   VPADAL   : Vector Pairwise Add and Accumulate Long
defm VPADALs  : N2VPLInt2_QHS<0b11, 0b11, 0b00, 0b01100, 0, "vpadal", "s",
                              int_arm_neon_vpadals>;
defm VPADALu  : N2VPLInt2_QHS<0b11, 0b11, 0b00, 0b01101, 0, "vpadal", "u",
                              int_arm_neon_vpadalu>;

//   VPMAX    : Vector Pairwise Maximum
def  VPMAXs8  : N3VDInt<0, 0, 0b00, 0b1010, 0, N3RegFrm, IIC_VSUBi4D, "vpmax",
                        "s8", v8i8, v8i8, int_arm_neon_vpmaxs, 0>;
def  VPMAXs16 : N3VDInt<0, 0, 0b01, 0b1010, 0, N3RegFrm, IIC_VSUBi4D, "vpmax",
                        "s16", v4i16, v4i16, int_arm_neon_vpmaxs, 0>;
def  VPMAXs32 : N3VDInt<0, 0, 0b10, 0b1010, 0, N3RegFrm, IIC_VSUBi4D, "vpmax",
                        "s32", v2i32, v2i32, int_arm_neon_vpmaxs, 0>;
def  VPMAXu8  : N3VDInt<1, 0, 0b00, 0b1010, 0, N3RegFrm, IIC_VSUBi4D, "vpmax",
                        "u8", v8i8, v8i8, int_arm_neon_vpmaxu, 0>;
def  VPMAXu16 : N3VDInt<1, 0, 0b01, 0b1010, 0, N3RegFrm, IIC_VSUBi4D, "vpmax",
                        "u16", v4i16, v4i16, int_arm_neon_vpmaxu, 0>;
def  VPMAXu32 : N3VDInt<1, 0, 0b10, 0b1010, 0, N3RegFrm, IIC_VSUBi4D, "vpmax",
                        "u32", v2i32, v2i32, int_arm_neon_vpmaxu, 0>;
def  VPMAXf   : N3VDInt<1, 0, 0b00, 0b1111, 0, N3RegFrm, IIC_VPBIND, "vpmax",
                        "f32", v2f32, v2f32, int_arm_neon_vpmaxs, 0>;

//   VPMIN    : Vector Pairwise Minimum
def  VPMINs8  : N3VDInt<0, 0, 0b00, 0b1010, 1, N3RegFrm, IIC_VSUBi4D, "vpmin",
                        "s8", v8i8, v8i8, int_arm_neon_vpmins, 0>;
def  VPMINs16 : N3VDInt<0, 0, 0b01, 0b1010, 1, N3RegFrm, IIC_VSUBi4D, "vpmin",
                        "s16", v4i16, v4i16, int_arm_neon_vpmins, 0>;
def  VPMINs32 : N3VDInt<0, 0, 0b10, 0b1010, 1, N3RegFrm, IIC_VSUBi4D, "vpmin",
                        "s32", v2i32, v2i32, int_arm_neon_vpmins, 0>;
def  VPMINu8  : N3VDInt<1, 0, 0b00, 0b1010, 1, N3RegFrm, IIC_VSUBi4D, "vpmin",
                        "u8", v8i8, v8i8, int_arm_neon_vpminu, 0>;
def  VPMINu16 : N3VDInt<1, 0, 0b01, 0b1010, 1, N3RegFrm, IIC_VSUBi4D, "vpmin",
                        "u16", v4i16, v4i16, int_arm_neon_vpminu, 0>;
def  VPMINu32 : N3VDInt<1, 0, 0b10, 0b1010, 1, N3RegFrm, IIC_VSUBi4D, "vpmin",
                        "u32", v2i32, v2i32, int_arm_neon_vpminu, 0>;
def  VPMINf   : N3VDInt<1, 0, 0b10, 0b1111, 0, N3RegFrm, IIC_VPBIND, "vpmin",
                        "f32", v2f32, v2f32, int_arm_neon_vpmins, 0>;

// Vector Reciprocal and Reciprocal Square Root Estimate and Step.

//   VRECPE   : Vector Reciprocal Estimate
def  VRECPEd  : N2VDInt<0b11, 0b11, 0b10, 0b11, 0b01000, 0, 
                        IIC_VUNAD, "vrecpe", "u32",
                        v2i32, v2i32, int_arm_neon_vrecpe>;
def  VRECPEq  : N2VQInt<0b11, 0b11, 0b10, 0b11, 0b01000, 0, 
                        IIC_VUNAQ, "vrecpe", "u32",
                        v4i32, v4i32, int_arm_neon_vrecpe>;
def  VRECPEfd : N2VDInt<0b11, 0b11, 0b10, 0b11, 0b01010, 0,
                        IIC_VUNAD, "vrecpe", "f32",
                        v2f32, v2f32, int_arm_neon_vrecpe>;
def  VRECPEfq : N2VQInt<0b11, 0b11, 0b10, 0b11, 0b01010, 0,
                        IIC_VUNAQ, "vrecpe", "f32",
                        v4f32, v4f32, int_arm_neon_vrecpe>;

//   VRECPS   : Vector Reciprocal Step
def  VRECPSfd : N3VDInt<0, 0, 0b00, 0b1111, 1, N3RegFrm,
                        IIC_VRECSD, "vrecps", "f32",
                        v2f32, v2f32, int_arm_neon_vrecps, 1>;
def  VRECPSfq : N3VQInt<0, 0, 0b00, 0b1111, 1, N3RegFrm,
                        IIC_VRECSQ, "vrecps", "f32",
                        v4f32, v4f32, int_arm_neon_vrecps, 1>;

//   VRSQRTE  : Vector Reciprocal Square Root Estimate
def  VRSQRTEd  : N2VDInt<0b11, 0b11, 0b10, 0b11, 0b01001, 0,
                         IIC_VUNAD, "vrsqrte", "u32",
                         v2i32, v2i32, int_arm_neon_vrsqrte>;
def  VRSQRTEq  : N2VQInt<0b11, 0b11, 0b10, 0b11, 0b01001, 0,
                         IIC_VUNAQ, "vrsqrte", "u32",
                         v4i32, v4i32, int_arm_neon_vrsqrte>;
def  VRSQRTEfd : N2VDInt<0b11, 0b11, 0b10, 0b11, 0b01011, 0,
                         IIC_VUNAD, "vrsqrte", "f32",
                         v2f32, v2f32, int_arm_neon_vrsqrte>;
def  VRSQRTEfq : N2VQInt<0b11, 0b11, 0b10, 0b11, 0b01011, 0, 
                         IIC_VUNAQ, "vrsqrte", "f32",
                         v4f32, v4f32, int_arm_neon_vrsqrte>;

//   VRSQRTS  : Vector Reciprocal Square Root Step
def VRSQRTSfd : N3VDInt<0, 0, 0b10, 0b1111, 1, N3RegFrm,
                        IIC_VRECSD, "vrsqrts", "f32",
                        v2f32, v2f32, int_arm_neon_vrsqrts, 1>;
def VRSQRTSfq : N3VQInt<0, 0, 0b10, 0b1111, 1, N3RegFrm,
                        IIC_VRECSQ, "vrsqrts", "f32",
                        v4f32, v4f32, int_arm_neon_vrsqrts, 1>;

// Vector Shifts.

//   VSHL     : Vector Shift
defm VSHLs    : N3VInt_QHSDSh<0, 0, 0b0100, 0, N3RegVShFrm,
                            IIC_VSHLiD, IIC_VSHLiD, IIC_VSHLiQ, IIC_VSHLiQ,
                            "vshl", "s", int_arm_neon_vshifts>;
defm VSHLu    : N3VInt_QHSDSh<1, 0, 0b0100, 0, N3RegVShFrm,
                            IIC_VSHLiD, IIC_VSHLiD, IIC_VSHLiQ, IIC_VSHLiQ,
                            "vshl", "u", int_arm_neon_vshiftu>;
//   VSHL     : Vector Shift Left (Immediate)
defm VSHLi    : N2VSh_QHSD<0, 1, 0b0101, 1, IIC_VSHLiD, "vshl", "i", NEONvshl,
                           N2RegVShLFrm>;
//   VSHR     : Vector Shift Right (Immediate)
defm VSHRs    : N2VSh_QHSD<0, 1, 0b0000, 1, IIC_VSHLiD, "vshr", "s", NEONvshrs,
                           N2RegVShRFrm>;
defm VSHRu    : N2VSh_QHSD<1, 1, 0b0000, 1, IIC_VSHLiD, "vshr", "u", NEONvshru,
                           N2RegVShRFrm>;

//   VSHLL    : Vector Shift Left Long
defm VSHLLs   : N2VLSh_QHS<0, 1, 0b1010, 0, 0, 1, "vshll", "s", NEONvshlls>;
defm VSHLLu   : N2VLSh_QHS<1, 1, 0b1010, 0, 0, 1, "vshll", "u", NEONvshllu>;

//   VSHLL    : Vector Shift Left Long (with maximum shift count)
class N2VLShMax<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
                bit op6, bit op4, string OpcodeStr, string Dt, ValueType ResTy,
                ValueType OpTy, SDNode OpNode>
  : N2VLSh<op24, op23, op11_8, op7, op6, op4, OpcodeStr, Dt,
           ResTy, OpTy, OpNode> {
  let Inst{21-16} = op21_16;
}
def  VSHLLi8  : N2VLShMax<1, 1, 0b110010, 0b0011, 0, 0, 0, "vshll", "i8",
                          v8i16, v8i8, NEONvshlli>;
def  VSHLLi16 : N2VLShMax<1, 1, 0b110110, 0b0011, 0, 0, 0, "vshll", "i16",
                          v4i32, v4i16, NEONvshlli>;
def  VSHLLi32 : N2VLShMax<1, 1, 0b111010, 0b0011, 0, 0, 0, "vshll", "i32",
                          v2i64, v2i32, NEONvshlli>;

//   VSHRN    : Vector Shift Right and Narrow
defm VSHRN    : N2VNSh_HSD<0,1,0b1000,0,0,1, IIC_VSHLiD, "vshrn", "i",
                           NEONvshrn>;

//   VRSHL    : Vector Rounding Shift
defm VRSHLs   : N3VInt_QHSDSh<0, 0, 0b0101, 0, N3RegVShFrm,
                            IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q, IIC_VSHLi4Q,
                            "vrshl", "s", int_arm_neon_vrshifts>;
defm VRSHLu   : N3VInt_QHSDSh<1, 0, 0b0101, 0, N3RegVShFrm,
                            IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q, IIC_VSHLi4Q,
                            "vrshl", "u", int_arm_neon_vrshiftu>;
//   VRSHR    : Vector Rounding Shift Right
defm VRSHRs   : N2VSh_QHSD<0,1,0b0010,1, IIC_VSHLi4D, "vrshr", "s", NEONvrshrs,
                           N2RegVShRFrm>;
defm VRSHRu   : N2VSh_QHSD<1,1,0b0010,1, IIC_VSHLi4D, "vrshr", "u", NEONvrshru,
                           N2RegVShRFrm>;

//   VRSHRN   : Vector Rounding Shift Right and Narrow
defm VRSHRN   : N2VNSh_HSD<0, 1, 0b1000, 0, 1, 1, IIC_VSHLi4D, "vrshrn", "i",
                           NEONvrshrn>;

//   VQSHL    : Vector Saturating Shift
defm VQSHLs   : N3VInt_QHSDSh<0, 0, 0b0100, 1, N3RegVShFrm,
                            IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q, IIC_VSHLi4Q,
                            "vqshl", "s", int_arm_neon_vqshifts>;
defm VQSHLu   : N3VInt_QHSDSh<1, 0, 0b0100, 1, N3RegVShFrm,
                            IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q, IIC_VSHLi4Q,
                            "vqshl", "u", int_arm_neon_vqshiftu>;
//   VQSHL    : Vector Saturating Shift Left (Immediate)
defm VQSHLsi  : N2VSh_QHSD<0,1,0b0111,1, IIC_VSHLi4D, "vqshl", "s",NEONvqshls,
                           N2RegVShLFrm>;
defm VQSHLui  : N2VSh_QHSD<1,1,0b0111,1, IIC_VSHLi4D, "vqshl", "u",NEONvqshlu,
                           N2RegVShLFrm>;
//   VQSHLU   : Vector Saturating Shift Left (Immediate, Unsigned)
defm VQSHLsu  : N2VSh_QHSD<1,1,0b0110,1, IIC_VSHLi4D,"vqshlu","s",NEONvqshlsu,
                           N2RegVShLFrm>;

//   VQSHRN   : Vector Saturating Shift Right and Narrow
defm VQSHRNs  : N2VNSh_HSD<0, 1, 0b1001, 0, 0, 1, IIC_VSHLi4D, "vqshrn", "s",
                           NEONvqshrns>;
defm VQSHRNu  : N2VNSh_HSD<1, 1, 0b1001, 0, 0, 1, IIC_VSHLi4D, "vqshrn", "u",
                           NEONvqshrnu>;

//   VQSHRUN  : Vector Saturating Shift Right and Narrow (Unsigned)
defm VQSHRUN  : N2VNSh_HSD<1, 1, 0b1000, 0, 0, 1, IIC_VSHLi4D, "vqshrun", "s",
                           NEONvqshrnsu>;

//   VQRSHL   : Vector Saturating Rounding Shift
defm VQRSHLs  : N3VInt_QHSDSh<0, 0, 0b0101, 1, N3RegVShFrm,
                            IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q, IIC_VSHLi4Q,
                            "vqrshl", "s", int_arm_neon_vqrshifts>;
defm VQRSHLu  : N3VInt_QHSDSh<1, 0, 0b0101, 1, N3RegVShFrm,
                            IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q, IIC_VSHLi4Q,
                            "vqrshl", "u", int_arm_neon_vqrshiftu>;

//   VQRSHRN  : Vector Saturating Rounding Shift Right and Narrow
defm VQRSHRNs : N2VNSh_HSD<0, 1, 0b1001, 0, 1, 1, IIC_VSHLi4D, "vqrshrn", "s",
                           NEONvqrshrns>;
defm VQRSHRNu : N2VNSh_HSD<1, 1, 0b1001, 0, 1, 1, IIC_VSHLi4D, "vqrshrn", "u",
                           NEONvqrshrnu>;

//   VQRSHRUN : Vector Saturating Rounding Shift Right and Narrow (Unsigned)
defm VQRSHRUN : N2VNSh_HSD<1, 1, 0b1000, 0, 1, 1, IIC_VSHLi4D, "vqrshrun", "s",
                           NEONvqrshrnsu>;

//   VSRA     : Vector Shift Right and Accumulate
defm VSRAs    : N2VShAdd_QHSD<0, 1, 0b0001, 1, "vsra", "s", NEONvshrs>;
defm VSRAu    : N2VShAdd_QHSD<1, 1, 0b0001, 1, "vsra", "u", NEONvshru>;
//   VRSRA    : Vector Rounding Shift Right and Accumulate
defm VRSRAs   : N2VShAdd_QHSD<0, 1, 0b0011, 1, "vrsra", "s", NEONvrshrs>;
defm VRSRAu   : N2VShAdd_QHSD<1, 1, 0b0011, 1, "vrsra", "u", NEONvrshru>;

//   VSLI     : Vector Shift Left and Insert
defm VSLI     : N2VShIns_QHSD<1, 1, 0b0101, 1, "vsli", NEONvsli, N2RegVShLFrm>;
//   VSRI     : Vector Shift Right and Insert
defm VSRI     : N2VShIns_QHSD<1, 1, 0b0100, 1, "vsri", NEONvsri, N2RegVShRFrm>;

// Vector Absolute and Saturating Absolute.

//   VABS     : Vector Absolute Value
defm VABS     : N2VInt_QHS<0b11, 0b11, 0b01, 0b00110, 0, 
                           IIC_VUNAiD, IIC_VUNAiQ, "vabs", "s",
                           int_arm_neon_vabs>;
def  VABSfd   : N2VDInt<0b11, 0b11, 0b10, 0b01, 0b01110, 0,
                        IIC_VUNAD, "vabs", "f32",
                        v2f32, v2f32, int_arm_neon_vabs>;
def  VABSfq   : N2VQInt<0b11, 0b11, 0b10, 0b01, 0b01110, 0,
                        IIC_VUNAQ, "vabs", "f32",
                        v4f32, v4f32, int_arm_neon_vabs>;

//   VQABS    : Vector Saturating Absolute Value
defm VQABS    : N2VInt_QHS<0b11, 0b11, 0b00, 0b01110, 0, 
                           IIC_VQUNAiD, IIC_VQUNAiQ, "vqabs", "s",
                           int_arm_neon_vqabs>;

// Vector Negate.

def vnegd  : PatFrag<(ops node:$in),
                     (sub (bitconvert (v2i32 NEONimmAllZerosV)), node:$in)>;
def vnegq  : PatFrag<(ops node:$in),
                     (sub (bitconvert (v4i32 NEONimmAllZerosV)), node:$in)>;

class VNEGD<bits<2> size, string OpcodeStr, string Dt, ValueType Ty>
  : N2V<0b11, 0b11, size, 0b01, 0b00111, 0, 0, (outs DPR:$dst), (ins DPR:$src),
        IIC_VSHLiD, OpcodeStr, Dt, "$dst, $src", "",
        [(set DPR:$dst, (Ty (vnegd DPR:$src)))]>;
class VNEGQ<bits<2> size, string OpcodeStr, string Dt, ValueType Ty>
  : N2V<0b11, 0b11, size, 0b01, 0b00111, 1, 0, (outs QPR:$dst), (ins QPR:$src),
        IIC_VSHLiQ, OpcodeStr, Dt, "$dst, $src", "",
        [(set QPR:$dst, (Ty (vnegq QPR:$src)))]>;

//   VNEG     : Vector Negate (integer)
def  VNEGs8d  : VNEGD<0b00, "vneg", "s8", v8i8>;
def  VNEGs16d : VNEGD<0b01, "vneg", "s16", v4i16>;
def  VNEGs32d : VNEGD<0b10, "vneg", "s32", v2i32>;
def  VNEGs8q  : VNEGQ<0b00, "vneg", "s8", v16i8>;
def  VNEGs16q : VNEGQ<0b01, "vneg", "s16", v8i16>;
def  VNEGs32q : VNEGQ<0b10, "vneg", "s32", v4i32>;

//   VNEG     : Vector Negate (floating-point)
def  VNEGfd   : N2V<0b11, 0b11, 0b10, 0b01, 0b01111, 0, 0,
                    (outs DPR:$dst), (ins DPR:$src), IIC_VUNAD,
                    "vneg", "f32", "$dst, $src", "",
                    [(set DPR:$dst, (v2f32 (fneg DPR:$src)))]>;
def  VNEGf32q : N2V<0b11, 0b11, 0b10, 0b01, 0b01111, 1, 0,
                    (outs QPR:$dst), (ins QPR:$src), IIC_VUNAQ,
                    "vneg", "f32", "$dst, $src", "",
                    [(set QPR:$dst, (v4f32 (fneg QPR:$src)))]>;

def : Pat<(v8i8  (vnegd  DPR:$src)), (VNEGs8d DPR:$src)>;
def : Pat<(v4i16 (vnegd  DPR:$src)), (VNEGs16d DPR:$src)>;
def : Pat<(v2i32 (vnegd  DPR:$src)), (VNEGs32d DPR:$src)>;
def : Pat<(v16i8 (vnegq QPR:$src)), (VNEGs8q QPR:$src)>;
def : Pat<(v8i16 (vnegq QPR:$src)), (VNEGs16q QPR:$src)>;
def : Pat<(v4i32 (vnegq QPR:$src)), (VNEGs32q QPR:$src)>;

//   VQNEG    : Vector Saturating Negate
defm VQNEG    : N2VInt_QHS<0b11, 0b11, 0b00, 0b01111, 0, 
                           IIC_VQUNAiD, IIC_VQUNAiQ, "vqneg", "s",
                           int_arm_neon_vqneg>;

// Vector Bit Counting Operations.

//   VCLS     : Vector Count Leading Sign Bits
defm VCLS     : N2VInt_QHS<0b11, 0b11, 0b00, 0b01000, 0, 
                           IIC_VCNTiD, IIC_VCNTiQ, "vcls", "s",
                           int_arm_neon_vcls>;
//   VCLZ     : Vector Count Leading Zeros
defm VCLZ     : N2VInt_QHS<0b11, 0b11, 0b00, 0b01001, 0, 
                           IIC_VCNTiD, IIC_VCNTiQ, "vclz", "i",
                           int_arm_neon_vclz>;
//   VCNT     : Vector Count One Bits
def  VCNTd    : N2VDInt<0b11, 0b11, 0b00, 0b00, 0b01010, 0, 
                        IIC_VCNTiD, "vcnt", "8",
                        v8i8, v8i8, int_arm_neon_vcnt>;
def  VCNTq    : N2VQInt<0b11, 0b11, 0b00, 0b00, 0b01010, 0,
                        IIC_VCNTiQ, "vcnt", "8",
                        v16i8, v16i8, int_arm_neon_vcnt>;

// Vector Swap -- for disassembly only.
def  VSWPd    : N2VX<0b11, 0b11, 0b00, 0b10, 0b00000, 0, 0,
                     (outs DPR:$dst), (ins DPR:$src), NoItinerary,
                     "vswp", "$dst, $src", "", []>;
def  VSWPq    : N2VX<0b11, 0b11, 0b00, 0b10, 0b00000, 1, 0,
                     (outs QPR:$dst), (ins QPR:$src), NoItinerary,
                     "vswp", "$dst, $src", "", []>;

// Vector Move Operations.

//   VMOV     : Vector Move (Register)

let neverHasSideEffects = 1 in {
def  VMOVDneon: N3VX<0, 0, 0b10, 0b0001, 0, 1, (outs DPR:$dst), (ins DPR:$src),
                     N3RegFrm, IIC_VMOV, "vmov", "$dst, $src", "", []>;
def  VMOVQ    : N3VX<0, 0, 0b10, 0b0001, 1, 1, (outs QPR:$dst), (ins QPR:$src),
                     N3RegFrm, IIC_VMOV, "vmov", "$dst, $src", "", []>;

// Pseudo vector move instructions for QQ and QQQQ registers. This should
// be expanded after register allocation is completed.
def  VMOVQQ   : PseudoInst<(outs QQPR:$dst), (ins QQPR:$src),
                NoItinerary, "", []>;

def  VMOVQQQQ : PseudoInst<(outs QQQQPR:$dst), (ins QQQQPR:$src),
                NoItinerary, "", []>;
} // neverHasSideEffects

//   VMOV     : Vector Move (Immediate)

let isReMaterializable = 1 in {
def VMOVv8i8  : N1ModImm<1, 0b000, 0b1110, 0, 0, 0, 1, (outs DPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmov", "i8", "$dst, $SIMM", "",
                         [(set DPR:$dst, (v8i8 (NEONvmovImm timm:$SIMM)))]>;
def VMOVv16i8 : N1ModImm<1, 0b000, 0b1110, 0, 1, 0, 1, (outs QPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmov", "i8", "$dst, $SIMM", "",
                         [(set QPR:$dst, (v16i8 (NEONvmovImm timm:$SIMM)))]>;

def VMOVv4i16 : N1ModImm<1, 0b000, {1,0,?,0}, 0, 0, 0, 1, (outs DPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmov", "i16", "$dst, $SIMM", "",
                         [(set DPR:$dst, (v4i16 (NEONvmovImm timm:$SIMM)))]> {
  let Inst{9} = SIMM{9};                         
}

def VMOVv8i16 : N1ModImm<1, 0b000, {1,0,?,0}, 0, 1, 0, 1, (outs QPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmov", "i16", "$dst, $SIMM", "",
                         [(set QPR:$dst, (v8i16 (NEONvmovImm timm:$SIMM)))]> {
 let Inst{9} = SIMM{9};
}

def VMOVv2i32 : N1ModImm<1, 0b000, {?,?,?,?}, 0, 0, 0, 1, (outs DPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmov", "i32", "$dst, $SIMM", "",
                         [(set DPR:$dst, (v2i32 (NEONvmovImm timm:$SIMM)))]> {
  let Inst{11-8} = SIMM{11-8};
}

def VMOVv4i32 : N1ModImm<1, 0b000, {?,?,?,?}, 0, 1, 0, 1, (outs QPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmov", "i32", "$dst, $SIMM", "",
                         [(set QPR:$dst, (v4i32 (NEONvmovImm timm:$SIMM)))]> {
  let Inst{11-8} = SIMM{11-8};
}

def VMOVv1i64 : N1ModImm<1, 0b000, 0b1110, 0, 0, 1, 1, (outs DPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmov", "i64", "$dst, $SIMM", "",
                         [(set DPR:$dst, (v1i64 (NEONvmovImm timm:$SIMM)))]>;
def VMOVv2i64 : N1ModImm<1, 0b000, 0b1110, 0, 1, 1, 1, (outs QPR:$dst),
                         (ins nModImm:$SIMM), IIC_VMOVImm,
                         "vmov", "i64", "$dst, $SIMM", "",
                         [(set QPR:$dst, (v2i64 (NEONvmovImm timm:$SIMM)))]>;
} // isReMaterializable

//   VMOV     : Vector Get Lane (move scalar to ARM core register)

def VGETLNs8  : NVGetLane<{1,1,1,0,0,1,?,1}, 0b1011, {?,?},
                          (outs GPR:$R), (ins DPR:$V, nohash_imm:$lane),
                          IIC_VMOVSI, "vmov", "s8", "$R, $V[$lane]",
                          [(set GPR:$R, (NEONvgetlanes (v8i8 DPR:$V),
                                           imm:$lane))]> {
  let Inst{21}  = lane{2};
  let Inst{6-5} = lane{1-0};
}
def VGETLNs16 : NVGetLane<{1,1,1,0,0,0,?,1}, 0b1011, {?,1},
                          (outs GPR:$R), (ins DPR:$V, nohash_imm:$lane),
                          IIC_VMOVSI, "vmov", "s16", "$R, $V[$lane]",
                          [(set GPR:$R, (NEONvgetlanes (v4i16 DPR:$V),
                                           imm:$lane))]> {
  let Inst{21} = lane{1};
  let Inst{6}  = lane{0};
}
def VGETLNu8  : NVGetLane<{1,1,1,0,1,1,?,1}, 0b1011, {?,?},
                          (outs GPR:$R), (ins DPR:$V, nohash_imm:$lane),
                          IIC_VMOVSI, "vmov", "u8", "$R, $V[$lane]",
                          [(set GPR:$R, (NEONvgetlaneu (v8i8 DPR:$V),
                                           imm:$lane))]> {
  let Inst{21}  = lane{2};
  let Inst{6-5} = lane{1-0};
}
def VGETLNu16 : NVGetLane<{1,1,1,0,1,0,?,1}, 0b1011, {?,1},
                          (outs GPR:$R), (ins DPR:$V, nohash_imm:$lane),
                          IIC_VMOVSI, "vmov", "u16", "$R, $V[$lane]",
                          [(set GPR:$R, (NEONvgetlaneu (v4i16 DPR:$V),
                                           imm:$lane))]> {
  let Inst{21} = lane{1};
  let Inst{6}  = lane{0};
}
def VGETLNi32 : NVGetLane<{1,1,1,0,0,0,?,1}, 0b1011, 0b00,
                          (outs GPR:$R), (ins DPR:$V, nohash_imm:$lane),
                          IIC_VMOVSI, "vmov", "32", "$R, $V[$lane]",
                          [(set GPR:$R, (extractelt (v2i32 DPR:$V),
                                           imm:$lane))]> {
  let Inst{21} = lane{0};
}
// def VGETLNf32: see FMRDH and FMRDL in ARMInstrVFP.td
def : Pat<(NEONvgetlanes (v16i8 QPR:$src), imm:$lane),
          (VGETLNs8 (v8i8 (EXTRACT_SUBREG QPR:$src,
                           (DSubReg_i8_reg imm:$lane))),
                     (SubReg_i8_lane imm:$lane))>;
def : Pat<(NEONvgetlanes (v8i16 QPR:$src), imm:$lane),
          (VGETLNs16 (v4i16 (EXTRACT_SUBREG QPR:$src,
                             (DSubReg_i16_reg imm:$lane))),
                     (SubReg_i16_lane imm:$lane))>;
def : Pat<(NEONvgetlaneu (v16i8 QPR:$src), imm:$lane),
          (VGETLNu8 (v8i8 (EXTRACT_SUBREG QPR:$src,
                           (DSubReg_i8_reg imm:$lane))),
                     (SubReg_i8_lane imm:$lane))>;
def : Pat<(NEONvgetlaneu (v8i16 QPR:$src), imm:$lane),
          (VGETLNu16 (v4i16 (EXTRACT_SUBREG QPR:$src,
                             (DSubReg_i16_reg imm:$lane))),
                     (SubReg_i16_lane imm:$lane))>;
def : Pat<(extractelt (v4i32 QPR:$src), imm:$lane),
          (VGETLNi32 (v2i32 (EXTRACT_SUBREG QPR:$src,
                             (DSubReg_i32_reg imm:$lane))),
                     (SubReg_i32_lane imm:$lane))>;
def : Pat<(extractelt (v2f32 DPR:$src1), imm:$src2),
          (EXTRACT_SUBREG (v2f32 (COPY_TO_REGCLASS (v2f32 DPR:$src1),DPR_VFP2)),
                          (SSubReg_f32_reg imm:$src2))>;
def : Pat<(extractelt (v4f32 QPR:$src1), imm:$src2),
          (EXTRACT_SUBREG (v4f32 (COPY_TO_REGCLASS (v4f32 QPR:$src1),QPR_VFP2)),
                          (SSubReg_f32_reg imm:$src2))>;
//def : Pat<(extractelt (v2i64 QPR:$src1), imm:$src2),
//          (EXTRACT_SUBREG QPR:$src1, (DSubReg_f64_reg imm:$src2))>;
def : Pat<(extractelt (v2f64 QPR:$src1), imm:$src2),
          (EXTRACT_SUBREG QPR:$src1, (DSubReg_f64_reg imm:$src2))>;


//   VMOV     : Vector Set Lane (move ARM core register to scalar)

let Constraints = "$src1 = $V" in {
def VSETLNi8  : NVSetLane<{1,1,1,0,0,1,?,0}, 0b1011, {?,?}, (outs DPR:$V),
                          (ins DPR:$src1, GPR:$R, nohash_imm:$lane),
                          IIC_VMOVISL, "vmov", "8", "$V[$lane], $R",
                          [(set DPR:$V, (vector_insert (v8i8 DPR:$src1),
                                           GPR:$R, imm:$lane))]> {
  let Inst{21}  = lane{2};
  let Inst{6-5} = lane{1-0};
}
def VSETLNi16 : NVSetLane<{1,1,1,0,0,0,?,0}, 0b1011, {?,1}, (outs DPR:$V),
                          (ins DPR:$src1, GPR:$R, nohash_imm:$lane),
                          IIC_VMOVISL, "vmov", "16", "$V[$lane], $R",
                          [(set DPR:$V, (vector_insert (v4i16 DPR:$src1),
                                           GPR:$R, imm:$lane))]> {
  let Inst{21} = lane{1};
  let Inst{6}  = lane{0};
}
def VSETLNi32 : NVSetLane<{1,1,1,0,0,0,?,0}, 0b1011, 0b00, (outs DPR:$V),
                          (ins DPR:$src1, GPR:$R, nohash_imm:$lane),
                          IIC_VMOVISL, "vmov", "32", "$V[$lane], $R",
                          [(set DPR:$V, (insertelt (v2i32 DPR:$src1),
                                           GPR:$R, imm:$lane))]> {
  let Inst{21} = lane{0};
}
}
def : Pat<(vector_insert (v16i8 QPR:$src1), GPR:$src2, imm:$lane),
          (v16i8 (INSERT_SUBREG QPR:$src1, 
                  (v8i8 (VSETLNi8 (v8i8 (EXTRACT_SUBREG QPR:$src1,
                                   (DSubReg_i8_reg imm:$lane))),
                            GPR:$src2, (SubReg_i8_lane imm:$lane))),
                  (DSubReg_i8_reg imm:$lane)))>;
def : Pat<(vector_insert (v8i16 QPR:$src1), GPR:$src2, imm:$lane),
          (v8i16 (INSERT_SUBREG QPR:$src1, 
                  (v4i16 (VSETLNi16 (v4i16 (EXTRACT_SUBREG QPR:$src1,
                                     (DSubReg_i16_reg imm:$lane))),
                             GPR:$src2, (SubReg_i16_lane imm:$lane))),
                  (DSubReg_i16_reg imm:$lane)))>;
def : Pat<(insertelt (v4i32 QPR:$src1), GPR:$src2, imm:$lane),
          (v4i32 (INSERT_SUBREG QPR:$src1, 
                  (v2i32 (VSETLNi32 (v2i32 (EXTRACT_SUBREG QPR:$src1,
                                     (DSubReg_i32_reg imm:$lane))),
                             GPR:$src2, (SubReg_i32_lane imm:$lane))),
                  (DSubReg_i32_reg imm:$lane)))>;

def : Pat<(v2f32 (insertelt DPR:$src1, SPR:$src2, imm:$src3)),
          (INSERT_SUBREG (v2f32 (COPY_TO_REGCLASS DPR:$src1, DPR_VFP2)),
                                SPR:$src2, (SSubReg_f32_reg imm:$src3))>;
def : Pat<(v4f32 (insertelt QPR:$src1, SPR:$src2, imm:$src3)),
          (INSERT_SUBREG (v4f32 (COPY_TO_REGCLASS QPR:$src1, QPR_VFP2)),
                                SPR:$src2, (SSubReg_f32_reg imm:$src3))>;

//def : Pat<(v2i64 (insertelt QPR:$src1, DPR:$src2, imm:$src3)),
//          (INSERT_SUBREG QPR:$src1, DPR:$src2, (DSubReg_f64_reg imm:$src3))>;
def : Pat<(v2f64 (insertelt QPR:$src1, DPR:$src2, imm:$src3)),
          (INSERT_SUBREG QPR:$src1, DPR:$src2, (DSubReg_f64_reg imm:$src3))>;

def : Pat<(v2f32 (scalar_to_vector SPR:$src)),
          (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)), SPR:$src, ssub_0)>;
def : Pat<(v2f64 (scalar_to_vector (f64 DPR:$src))),
          (INSERT_SUBREG (v2f64 (IMPLICIT_DEF)), DPR:$src, dsub_0)>;
def : Pat<(v4f32 (scalar_to_vector SPR:$src)),
          (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)), SPR:$src, ssub_0)>;

def : Pat<(v8i8 (scalar_to_vector GPR:$src)),
          (VSETLNi8  (v8i8  (IMPLICIT_DEF)), GPR:$src, (i32 0))>;
def : Pat<(v4i16 (scalar_to_vector GPR:$src)),
          (VSETLNi16 (v4i16 (IMPLICIT_DEF)), GPR:$src, (i32 0))>;
def : Pat<(v2i32 (scalar_to_vector GPR:$src)),
          (VSETLNi32 (v2i32 (IMPLICIT_DEF)), GPR:$src, (i32 0))>;

def : Pat<(v16i8 (scalar_to_vector GPR:$src)),
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
                         (VSETLNi8 (v8i8 (IMPLICIT_DEF)), GPR:$src, (i32 0)),
                         dsub_0)>;
def : Pat<(v8i16 (scalar_to_vector GPR:$src)),
          (INSERT_SUBREG (v8i16 (IMPLICIT_DEF)),
                         (VSETLNi16 (v4i16 (IMPLICIT_DEF)), GPR:$src, (i32 0)),
                         dsub_0)>;
def : Pat<(v4i32 (scalar_to_vector GPR:$src)),
          (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)),
                         (VSETLNi32 (v2i32 (IMPLICIT_DEF)), GPR:$src, (i32 0)),
                         dsub_0)>;

//   VDUP     : Vector Duplicate (from ARM core register to all elements)

class VDUPD<bits<8> opcod1, bits<2> opcod3, string Dt, ValueType Ty>
  : NVDup<opcod1, 0b1011, opcod3, (outs DPR:$dst), (ins GPR:$src),
          IIC_VMOVIS, "vdup", Dt, "$dst, $src",
          [(set DPR:$dst, (Ty (NEONvdup (i32 GPR:$src))))]>;
class VDUPQ<bits<8> opcod1, bits<2> opcod3, string Dt, ValueType Ty>
  : NVDup<opcod1, 0b1011, opcod3, (outs QPR:$dst), (ins GPR:$src),
          IIC_VMOVIS, "vdup", Dt, "$dst, $src",
          [(set QPR:$dst, (Ty (NEONvdup (i32 GPR:$src))))]>;

def  VDUP8d   : VDUPD<0b11101100, 0b00, "8", v8i8>;
def  VDUP16d  : VDUPD<0b11101000, 0b01, "16", v4i16>;
def  VDUP32d  : VDUPD<0b11101000, 0b00, "32", v2i32>;
def  VDUP8q   : VDUPQ<0b11101110, 0b00, "8", v16i8>;
def  VDUP16q  : VDUPQ<0b11101010, 0b01, "16", v8i16>;
def  VDUP32q  : VDUPQ<0b11101010, 0b00, "32", v4i32>;

def  VDUPfd   : NVDup<0b11101000, 0b1011, 0b00, (outs DPR:$dst), (ins GPR:$src),
                      IIC_VMOVIS, "vdup", "32", "$dst, $src",
                      [(set DPR:$dst, (v2f32 (NEONvdup
                                              (f32 (bitconvert GPR:$src)))))]>;
def  VDUPfq   : NVDup<0b11101010, 0b1011, 0b00, (outs QPR:$dst), (ins GPR:$src),
                      IIC_VMOVIS, "vdup", "32", "$dst, $src",
                      [(set QPR:$dst, (v4f32 (NEONvdup
                                              (f32 (bitconvert GPR:$src)))))]>;

//   VDUP     : Vector Duplicate Lane (from scalar to all elements)

class VDUPLND<bits<4> op19_16, string OpcodeStr, string Dt,
              ValueType Ty>
  : NVDupLane<op19_16, 0, (outs DPR:$dst), (ins DPR:$src, nohash_imm:$lane),
              IIC_VMOVD, OpcodeStr, Dt, "$dst, $src[$lane]",
              [(set DPR:$dst, (Ty (NEONvduplane (Ty DPR:$src), imm:$lane)))]>;

class VDUPLNQ<bits<4> op19_16, string OpcodeStr, string Dt,
              ValueType ResTy, ValueType OpTy>
  : NVDupLane<op19_16, 1, (outs QPR:$dst), (ins DPR:$src, nohash_imm:$lane),
              IIC_VMOVQ, OpcodeStr, Dt, "$dst, $src[$lane]",
              [(set QPR:$dst, (ResTy (NEONvduplane (OpTy DPR:$src),
                                      imm:$lane)))]>;

// Inst{19-16} is partially specified depending on the element size.

def VDUPLN8d  : VDUPLND<{?,?,?,1}, "vdup", "8", v8i8> {
  let Inst{19-17} = lane{2-0};
}
def VDUPLN16d : VDUPLND<{?,?,1,0}, "vdup", "16", v4i16> {
  let Inst{19-18} = lane{1-0};
}
def VDUPLN32d : VDUPLND<{?,1,0,0}, "vdup", "32", v2i32> {
  let Inst{19} = lane{0};
}
def VDUPLNfd  : VDUPLND<{?,1,0,0}, "vdup", "32", v2f32> {
  let Inst{19} = lane{0};
}
def VDUPLN8q  : VDUPLNQ<{?,?,?,1}, "vdup", "8", v16i8, v8i8> {
  let Inst{19-17} = lane{2-0};
}
def VDUPLN16q : VDUPLNQ<{?,?,1,0}, "vdup", "16", v8i16, v4i16> {
  let Inst{19-18} = lane{1-0};
}
def VDUPLN32q : VDUPLNQ<{?,1,0,0}, "vdup", "32", v4i32, v2i32> {
  let Inst{19} = lane{0};
}
def VDUPLNfq  : VDUPLNQ<{?,1,0,0}, "vdup", "32", v4f32, v2f32> {
  let Inst{19} = lane{0};
}

def : Pat<(v16i8 (NEONvduplane (v16i8 QPR:$src), imm:$lane)),
          (v16i8 (VDUPLN8q (v8i8 (EXTRACT_SUBREG QPR:$src,
                                  (DSubReg_i8_reg imm:$lane))),
                           (SubReg_i8_lane imm:$lane)))>;
def : Pat<(v8i16 (NEONvduplane (v8i16 QPR:$src), imm:$lane)),
          (v8i16 (VDUPLN16q (v4i16 (EXTRACT_SUBREG QPR:$src,
                                    (DSubReg_i16_reg imm:$lane))),
                            (SubReg_i16_lane imm:$lane)))>;
def : Pat<(v4i32 (NEONvduplane (v4i32 QPR:$src), imm:$lane)),
          (v4i32 (VDUPLN32q (v2i32 (EXTRACT_SUBREG QPR:$src,
                                    (DSubReg_i32_reg imm:$lane))),
                            (SubReg_i32_lane imm:$lane)))>;
def : Pat<(v4f32 (NEONvduplane (v4f32 QPR:$src), imm:$lane)),
          (v4f32 (VDUPLNfq (v2f32 (EXTRACT_SUBREG QPR:$src,
                                   (DSubReg_i32_reg imm:$lane))),
                           (SubReg_i32_lane imm:$lane)))>;

def  VDUPfdf : PseudoNeonI<(outs DPR:$dst), (ins SPR:$src), IIC_VMOVD, "",
                    [(set DPR:$dst, (v2f32 (NEONvdup (f32 SPR:$src))))]>;
def  VDUPfqf : PseudoNeonI<(outs QPR:$dst), (ins SPR:$src), IIC_VMOVD, "",
                    [(set QPR:$dst, (v4f32 (NEONvdup (f32 SPR:$src))))]>;

//   VMOVN    : Vector Narrowing Move
defm VMOVN    : N2VN_HSD<0b11,0b11,0b10,0b00100,0,0, IIC_VMOVN,
                         "vmovn", "i", trunc>;
//   VQMOVN   : Vector Saturating Narrowing Move
defm VQMOVNs  : N2VNInt_HSD<0b11,0b11,0b10,0b00101,0,0, IIC_VQUNAiD,
                            "vqmovn", "s", int_arm_neon_vqmovns>;
defm VQMOVNu  : N2VNInt_HSD<0b11,0b11,0b10,0b00101,1,0, IIC_VQUNAiD,
                            "vqmovn", "u", int_arm_neon_vqmovnu>;
defm VQMOVNsu : N2VNInt_HSD<0b11,0b11,0b10,0b00100,1,0, IIC_VQUNAiD,
                            "vqmovun", "s", int_arm_neon_vqmovnsu>;
//   VMOVL    : Vector Lengthening Move
defm VMOVLs   : N2VL_QHS<0b01,0b10100,0,1, "vmovl", "s", sext>;
defm VMOVLu   : N2VL_QHS<0b11,0b10100,0,1, "vmovl", "u", zext>;

// Vector Conversions.

//   VCVT     : Vector Convert Between Floating-Point and Integers
def  VCVTf2sd : N2VD<0b11, 0b11, 0b10, 0b11, 0b01110, 0, "vcvt", "s32.f32",
                     v2i32, v2f32, fp_to_sint>;
def  VCVTf2ud : N2VD<0b11, 0b11, 0b10, 0b11, 0b01111, 0, "vcvt", "u32.f32",
                     v2i32, v2f32, fp_to_uint>;
def  VCVTs2fd : N2VD<0b11, 0b11, 0b10, 0b11, 0b01100, 0, "vcvt", "f32.s32",
                     v2f32, v2i32, sint_to_fp>;
def  VCVTu2fd : N2VD<0b11, 0b11, 0b10, 0b11, 0b01101, 0, "vcvt", "f32.u32",
                     v2f32, v2i32, uint_to_fp>;

def  VCVTf2sq : N2VQ<0b11, 0b11, 0b10, 0b11, 0b01110, 0, "vcvt", "s32.f32",
                     v4i32, v4f32, fp_to_sint>;
def  VCVTf2uq : N2VQ<0b11, 0b11, 0b10, 0b11, 0b01111, 0, "vcvt", "u32.f32",
                     v4i32, v4f32, fp_to_uint>;
def  VCVTs2fq : N2VQ<0b11, 0b11, 0b10, 0b11, 0b01100, 0, "vcvt", "f32.s32",
                     v4f32, v4i32, sint_to_fp>;
def  VCVTu2fq : N2VQ<0b11, 0b11, 0b10, 0b11, 0b01101, 0, "vcvt", "f32.u32",
                     v4f32, v4i32, uint_to_fp>;

//   VCVT     : Vector Convert Between Floating-Point and Fixed-Point.
def VCVTf2xsd : N2VCvtD<0, 1, 0b1111, 0, 1, "vcvt", "s32.f32",
                        v2i32, v2f32, int_arm_neon_vcvtfp2fxs>;
def VCVTf2xud : N2VCvtD<1, 1, 0b1111, 0, 1, "vcvt", "u32.f32",
                        v2i32, v2f32, int_arm_neon_vcvtfp2fxu>;
def VCVTxs2fd : N2VCvtD<0, 1, 0b1110, 0, 1, "vcvt", "f32.s32",
                        v2f32, v2i32, int_arm_neon_vcvtfxs2fp>;
def VCVTxu2fd : N2VCvtD<1, 1, 0b1110, 0, 1, "vcvt", "f32.u32",
                        v2f32, v2i32, int_arm_neon_vcvtfxu2fp>;

def VCVTf2xsq : N2VCvtQ<0, 1, 0b1111, 0, 1, "vcvt", "s32.f32",
                        v4i32, v4f32, int_arm_neon_vcvtfp2fxs>;
def VCVTf2xuq : N2VCvtQ<1, 1, 0b1111, 0, 1, "vcvt", "u32.f32",
                        v4i32, v4f32, int_arm_neon_vcvtfp2fxu>;
def VCVTxs2fq : N2VCvtQ<0, 1, 0b1110, 0, 1, "vcvt", "f32.s32",
                        v4f32, v4i32, int_arm_neon_vcvtfxs2fp>;
def VCVTxu2fq : N2VCvtQ<1, 1, 0b1110, 0, 1, "vcvt", "f32.u32",
                        v4f32, v4i32, int_arm_neon_vcvtfxu2fp>;

// Vector Reverse.

//   VREV64   : Vector Reverse elements within 64-bit doublewords

class VREV64D<bits<2> op19_18, string OpcodeStr, string Dt, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00000, 0, 0, (outs DPR:$dst),
        (ins DPR:$src), IIC_VMOVD, 
        OpcodeStr, Dt, "$dst, $src", "",
        [(set DPR:$dst, (Ty (NEONvrev64 (Ty DPR:$src))))]>;
class VREV64Q<bits<2> op19_18, string OpcodeStr, string Dt, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00000, 1, 0, (outs QPR:$dst),
        (ins QPR:$src), IIC_VMOVQ, 
        OpcodeStr, Dt, "$dst, $src", "",
        [(set QPR:$dst, (Ty (NEONvrev64 (Ty QPR:$src))))]>;

def VREV64d8  : VREV64D<0b00, "vrev64", "8", v8i8>;
def VREV64d16 : VREV64D<0b01, "vrev64", "16", v4i16>;
def VREV64d32 : VREV64D<0b10, "vrev64", "32", v2i32>;
def VREV64df  : VREV64D<0b10, "vrev64", "32", v2f32>;

def VREV64q8  : VREV64Q<0b00, "vrev64", "8", v16i8>;
def VREV64q16 : VREV64Q<0b01, "vrev64", "16", v8i16>;
def VREV64q32 : VREV64Q<0b10, "vrev64", "32", v4i32>;
def VREV64qf  : VREV64Q<0b10, "vrev64", "32", v4f32>;

//   VREV32   : Vector Reverse elements within 32-bit words

class VREV32D<bits<2> op19_18, string OpcodeStr, string Dt, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00001, 0, 0, (outs DPR:$dst),
        (ins DPR:$src), IIC_VMOVD, 
        OpcodeStr, Dt, "$dst, $src", "",
        [(set DPR:$dst, (Ty (NEONvrev32 (Ty DPR:$src))))]>;
class VREV32Q<bits<2> op19_18, string OpcodeStr, string Dt, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00001, 1, 0, (outs QPR:$dst),
        (ins QPR:$src), IIC_VMOVQ, 
        OpcodeStr, Dt, "$dst, $src", "",
        [(set QPR:$dst, (Ty (NEONvrev32 (Ty QPR:$src))))]>;

def VREV32d8  : VREV32D<0b00, "vrev32", "8", v8i8>;
def VREV32d16 : VREV32D<0b01, "vrev32", "16", v4i16>;

def VREV32q8  : VREV32Q<0b00, "vrev32", "8", v16i8>;
def VREV32q16 : VREV32Q<0b01, "vrev32", "16", v8i16>;

//   VREV16   : Vector Reverse elements within 16-bit halfwords

class VREV16D<bits<2> op19_18, string OpcodeStr, string Dt, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00010, 0, 0, (outs DPR:$dst),
        (ins DPR:$src), IIC_VMOVD, 
        OpcodeStr, Dt, "$dst, $src", "",
        [(set DPR:$dst, (Ty (NEONvrev16 (Ty DPR:$src))))]>;
class VREV16Q<bits<2> op19_18, string OpcodeStr, string Dt, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00010, 1, 0, (outs QPR:$dst),
        (ins QPR:$src), IIC_VMOVQ, 
        OpcodeStr, Dt, "$dst, $src", "",
        [(set QPR:$dst, (Ty (NEONvrev16 (Ty QPR:$src))))]>;

def VREV16d8  : VREV16D<0b00, "vrev16", "8", v8i8>;
def VREV16q8  : VREV16Q<0b00, "vrev16", "8", v16i8>;

// Other Vector Shuffles.

//   VEXT     : Vector Extract

class VEXTd<string OpcodeStr, string Dt, ValueType Ty>
  : N3V<0,1,0b11,{?,?,?,?},0,0, (outs DPR:$dst),
        (ins DPR:$lhs, DPR:$rhs, i32imm:$index), NVExtFrm,
        IIC_VEXTD, OpcodeStr, Dt, "$dst, $lhs, $rhs, $index", "",
        [(set DPR:$dst, (Ty (NEONvext (Ty DPR:$lhs),
                                      (Ty DPR:$rhs), imm:$index)))]> {
  bits<4> index;
  let Inst{11-8} = index{3-0};
}

class VEXTq<string OpcodeStr, string Dt, ValueType Ty>
  : N3V<0,1,0b11,{?,?,?,?},1,0, (outs QPR:$dst),
        (ins QPR:$lhs, QPR:$rhs, i32imm:$index), NVExtFrm,
        IIC_VEXTQ, OpcodeStr, Dt, "$dst, $lhs, $rhs, $index", "",
        [(set QPR:$dst, (Ty (NEONvext (Ty QPR:$lhs),
                                      (Ty QPR:$rhs), imm:$index)))]> {
  bits<4> index;
  let Inst{11-8} = index{3-0};
}

def VEXTd8  : VEXTd<"vext", "8",  v8i8>;
def VEXTd16 : VEXTd<"vext", "16", v4i16>;
def VEXTd32 : VEXTd<"vext", "32", v2i32>;
def VEXTdf  : VEXTd<"vext", "32", v2f32>;

def VEXTq8  : VEXTq<"vext", "8",  v16i8>;
def VEXTq16 : VEXTq<"vext", "16", v8i16>;
def VEXTq32 : VEXTq<"vext", "32", v4i32>;
def VEXTqf  : VEXTq<"vext", "32", v4f32>;

//   VTRN     : Vector Transpose

def  VTRNd8   : N2VDShuffle<0b00, 0b00001, "vtrn", "8">;
def  VTRNd16  : N2VDShuffle<0b01, 0b00001, "vtrn", "16">;
def  VTRNd32  : N2VDShuffle<0b10, 0b00001, "vtrn", "32">;

def  VTRNq8   : N2VQShuffle<0b00, 0b00001, IIC_VPERMQ, "vtrn", "8">;
def  VTRNq16  : N2VQShuffle<0b01, 0b00001, IIC_VPERMQ, "vtrn", "16">;
def  VTRNq32  : N2VQShuffle<0b10, 0b00001, IIC_VPERMQ, "vtrn", "32">;

//   VUZP     : Vector Unzip (Deinterleave)

def  VUZPd8   : N2VDShuffle<0b00, 0b00010, "vuzp", "8">;
def  VUZPd16  : N2VDShuffle<0b01, 0b00010, "vuzp", "16">;
def  VUZPd32  : N2VDShuffle<0b10, 0b00010, "vuzp", "32">;

def  VUZPq8   : N2VQShuffle<0b00, 0b00010, IIC_VPERMQ3, "vuzp", "8">;
def  VUZPq16  : N2VQShuffle<0b01, 0b00010, IIC_VPERMQ3, "vuzp", "16">;
def  VUZPq32  : N2VQShuffle<0b10, 0b00010, IIC_VPERMQ3, "vuzp", "32">;

//   VZIP     : Vector Zip (Interleave)

def  VZIPd8   : N2VDShuffle<0b00, 0b00011, "vzip", "8">;
def  VZIPd16  : N2VDShuffle<0b01, 0b00011, "vzip", "16">;
def  VZIPd32  : N2VDShuffle<0b10, 0b00011, "vzip", "32">;

def  VZIPq8   : N2VQShuffle<0b00, 0b00011, IIC_VPERMQ3, "vzip", "8">;
def  VZIPq16  : N2VQShuffle<0b01, 0b00011, IIC_VPERMQ3, "vzip", "16">;
def  VZIPq32  : N2VQShuffle<0b10, 0b00011, IIC_VPERMQ3, "vzip", "32">;

// Vector Table Lookup and Table Extension.

//   VTBL     : Vector Table Lookup
def  VTBL1
  : N3V<1,1,0b11,0b1000,0,0, (outs DPR:$Vd),
        (ins DPR:$Vn, DPR:$Vm), NVTBLFrm, IIC_VTB1,
        "vtbl", "8", "$Vd, \\{$Vn\\}, $Vm", "",
        [(set DPR:$Vd, (v8i8 (int_arm_neon_vtbl1 DPR:$Vn, DPR:$Vm)))]>;
let hasExtraSrcRegAllocReq = 1 in {
def  VTBL2
  : N3V<1,1,0b11,0b1001,0,0, (outs DPR:$Vd),
        (ins DPR:$Vn, DPR:$tbl2, DPR:$Vm), NVTBLFrm, IIC_VTB2,
        "vtbl", "8", "$Vd, \\{$Vn, $tbl2\\}, $Vm", "", []>;
def  VTBL3
  : N3V<1,1,0b11,0b1010,0,0, (outs DPR:$Vd),
        (ins DPR:$Vn, DPR:$tbl2, DPR:$tbl3, DPR:$Vm), NVTBLFrm, IIC_VTB3,
        "vtbl", "8", "$Vd, \\{$Vn, $tbl2, $tbl3\\}, $Vm", "", []>;
def  VTBL4
  : N3V<1,1,0b11,0b1011,0,0, (outs DPR:$Vd),
        (ins DPR:$Vn, DPR:$tbl2, DPR:$tbl3, DPR:$tbl4, DPR:$Vm),
        NVTBLFrm, IIC_VTB4,
        "vtbl", "8", "$Vd, \\{$Vn, $tbl2, $tbl3, $tbl4\\}, $Vm", "", []>;
} // hasExtraSrcRegAllocReq = 1

def  VTBL2Pseudo
  : PseudoNeonI<(outs DPR:$dst), (ins QPR:$tbl, DPR:$src), IIC_VTB2, "", []>;
def  VTBL3Pseudo
  : PseudoNeonI<(outs DPR:$dst), (ins QQPR:$tbl, DPR:$src), IIC_VTB3, "", []>;
def  VTBL4Pseudo
  : PseudoNeonI<(outs DPR:$dst), (ins QQPR:$tbl, DPR:$src), IIC_VTB4, "", []>;

//   VTBX     : Vector Table Extension
def  VTBX1
  : N3V<1,1,0b11,0b1000,1,0, (outs DPR:$Vd),
        (ins DPR:$orig, DPR:$Vn, DPR:$Vm), NVTBLFrm, IIC_VTBX1,
        "vtbx", "8", "$Vd, \\{$Vn\\}, $Vm", "$orig = $Vd",
        [(set DPR:$Vd, (v8i8 (int_arm_neon_vtbx1
                               DPR:$orig, DPR:$Vn, DPR:$Vm)))]>;
let hasExtraSrcRegAllocReq = 1 in {
def  VTBX2
  : N3V<1,1,0b11,0b1001,1,0, (outs DPR:$Vd),
        (ins DPR:$orig, DPR:$Vn, DPR:$tbl2, DPR:$Vm), NVTBLFrm, IIC_VTBX2,
        "vtbx", "8", "$Vd, \\{$Vn, $tbl2\\}, $Vm", "$orig = $Vd", []>;
def  VTBX3
  : N3V<1,1,0b11,0b1010,1,0, (outs DPR:$Vd),
        (ins DPR:$orig, DPR:$Vn, DPR:$tbl2, DPR:$tbl3, DPR:$Vm),
        NVTBLFrm, IIC_VTBX3,
        "vtbx", "8", "$Vd, \\{$Vn, $tbl2, $tbl3\\}, $Vm",
        "$orig = $Vd", []>;
def  VTBX4
  : N3V<1,1,0b11,0b1011,1,0, (outs DPR:$Vd), (ins DPR:$orig, DPR:$Vn,
        DPR:$tbl2, DPR:$tbl3, DPR:$tbl4, DPR:$Vm), NVTBLFrm, IIC_VTBX4,
        "vtbx", "8", "$Vd, \\{$Vn, $tbl2, $tbl3, $tbl4\\}, $Vm",
        "$orig = $Vd", []>;
} // hasExtraSrcRegAllocReq = 1

def  VTBX2Pseudo
  : PseudoNeonI<(outs DPR:$dst), (ins DPR:$orig, QPR:$tbl, DPR:$src),
                IIC_VTBX2, "$orig = $dst", []>;
def  VTBX3Pseudo
  : PseudoNeonI<(outs DPR:$dst), (ins DPR:$orig, QQPR:$tbl, DPR:$src),
                IIC_VTBX3, "$orig = $dst", []>;
def  VTBX4Pseudo
  : PseudoNeonI<(outs DPR:$dst), (ins DPR:$orig, QQPR:$tbl, DPR:$src),
                IIC_VTBX4, "$orig = $dst", []>;

//===----------------------------------------------------------------------===//
// NEON instructions for single-precision FP math
//===----------------------------------------------------------------------===//

class N2VSPat<SDNode OpNode, ValueType ResTy, ValueType OpTy, NeonI Inst>
  : NEONFPPat<(ResTy (OpNode SPR:$a)),
              (EXTRACT_SUBREG (OpTy (Inst (INSERT_SUBREG (OpTy (IMPLICIT_DEF)),
                                                       SPR:$a, ssub_0))),
                              ssub_0)>;

class N3VSPat<SDNode OpNode, NeonI Inst>
  : NEONFPPat<(f32 (OpNode SPR:$a, SPR:$b)),
              (EXTRACT_SUBREG (v2f32
                                 (Inst (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)),
                                                      SPR:$a, ssub_0),
                                       (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)),
                                                      SPR:$b, ssub_0))),
                              ssub_0)>;

class N3VSMulOpPat<SDNode MulNode, SDNode OpNode, NeonI Inst>
  : NEONFPPat<(f32 (OpNode SPR:$acc, (f32 (MulNode SPR:$a, SPR:$b)))),
              (EXTRACT_SUBREG (Inst (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)),
                                                   SPR:$acc, ssub_0),
                                    (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)),
                                                   SPR:$a, ssub_0),
                                    (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)),
                                                   SPR:$b, ssub_0)),
                              ssub_0)>;

// These need separate instructions because they must use DPR_VFP2 register
// class which have SPR sub-registers.

// Vector Add Operations used for single-precision FP
let neverHasSideEffects = 1 in
def VADDfd_sfp : N3VS<0,0,0b00,0b1101,0, "vadd", "f32", v2f32, v2f32, fadd, 1>;
def : N3VSPat<fadd, VADDfd_sfp>;

// Vector Sub Operations used for single-precision FP
let neverHasSideEffects = 1 in
def VSUBfd_sfp : N3VS<0,0,0b10,0b1101,0, "vsub", "f32", v2f32, v2f32, fsub, 0>;
def : N3VSPat<fsub, VSUBfd_sfp>;

// Vector Multiply Operations used for single-precision FP
let neverHasSideEffects = 1 in
def VMULfd_sfp : N3VS<1,0,0b00,0b1101,1, "vmul", "f32", v2f32, v2f32, fmul, 1>;
def : N3VSPat<fmul, VMULfd_sfp>;

// Vector Multiply-Accumulate/Subtract used for single-precision FP
// vml[as].f32 can cause 4-8 cycle stalls in following ASIMD instructions, so
// we want to avoid them for now. e.g., alternating vmla/vadd instructions.

//let neverHasSideEffects = 1 in
//def VMLAfd_sfp : N3VSMulOp<0,0,0b00,0b1101,1, IIC_VMACD, "vmla", "f32",
//                           v2f32, fmul, fadd>;
//def : N3VSMulOpPat<fmul, fadd, VMLAfd_sfp>;

//let neverHasSideEffects = 1 in
//def VMLSfd_sfp : N3VSMulOp<0,0,0b10,0b1101,1, IIC_VMACD, "vmls", "f32",
//                           v2f32, fmul, fsub>;
//def : N3VSMulOpPat<fmul, fsub, VMLSfd_sfp>;

// Vector Absolute used for single-precision FP
let neverHasSideEffects = 1 in
def  VABSfd_sfp : N2V<0b11, 0b11, 0b10, 0b01, 0b01110, 0, 0,
                      (outs DPR_VFP2:$dst), (ins DPR_VFP2:$src), IIC_VUNAD,
                      "vabs", "f32", "$dst, $src", "", []>;
def : N2VSPat<fabs, f32, v2f32, VABSfd_sfp>;

// Vector Negate used for single-precision FP
let neverHasSideEffects = 1 in
def  VNEGfd_sfp : N2V<0b11, 0b11, 0b10, 0b01, 0b01111, 0, 0,
                      (outs DPR_VFP2:$dst), (ins DPR_VFP2:$src), IIC_VUNAD,
                      "vneg", "f32", "$dst, $src", "", []>;
def : N2VSPat<fneg, f32, v2f32, VNEGfd_sfp>;

// Vector Maximum used for single-precision FP
let neverHasSideEffects = 1 in
def VMAXfd_sfp : N3V<0, 0, 0b00, 0b1111, 0, 0, (outs DPR_VFP2:$dst),
                     (ins DPR_VFP2:$src1, DPR_VFP2:$src2), N3RegFrm, IIC_VBIND,
                     "vmax", "f32", "$dst, $src1, $src2", "", []>;
def : N3VSPat<NEONfmax, VMAXfd_sfp>;

// Vector Minimum used for single-precision FP
let neverHasSideEffects = 1 in
def VMINfd_sfp : N3V<0, 0, 0b00, 0b1111, 0, 0, (outs DPR_VFP2:$dst),
                     (ins DPR_VFP2:$src1, DPR_VFP2:$src2), N3RegFrm, IIC_VBIND,
                     "vmin", "f32", "$dst, $src1, $src2", "", []>;
def : N3VSPat<NEONfmin, VMINfd_sfp>;

// Vector Convert between single-precision FP and integer
let neverHasSideEffects = 1 in
def  VCVTf2sd_sfp : N2VS<0b11, 0b11, 0b10, 0b11, 0b01110, 0, "vcvt", "s32.f32",
                         v2i32, v2f32, fp_to_sint>;
def : N2VSPat<arm_ftosi, f32, v2f32, VCVTf2sd_sfp>;

let neverHasSideEffects = 1 in
def  VCVTf2ud_sfp : N2VS<0b11, 0b11, 0b10, 0b11, 0b01111, 0, "vcvt", "u32.f32",
                         v2i32, v2f32, fp_to_uint>;
def : N2VSPat<arm_ftoui, f32, v2f32, VCVTf2ud_sfp>;

let neverHasSideEffects = 1 in
def  VCVTs2fd_sfp : N2VS<0b11, 0b11, 0b10, 0b11, 0b01100, 0, "vcvt", "f32.s32",
                         v2f32, v2i32, sint_to_fp>;
def : N2VSPat<arm_sitof, f32, v2i32, VCVTs2fd_sfp>;

let neverHasSideEffects = 1 in
def  VCVTu2fd_sfp : N2VS<0b11, 0b11, 0b10, 0b11, 0b01101, 0, "vcvt", "f32.u32",
                         v2f32, v2i32, uint_to_fp>;
def : N2VSPat<arm_uitof, f32, v2i32, VCVTu2fd_sfp>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// bit_convert
def : Pat<(v1i64 (bitconvert (v2i32 DPR:$src))), (v1i64 DPR:$src)>;
def : Pat<(v1i64 (bitconvert (v4i16 DPR:$src))), (v1i64 DPR:$src)>;
def : Pat<(v1i64 (bitconvert (v8i8  DPR:$src))), (v1i64 DPR:$src)>;
def : Pat<(v1i64 (bitconvert (f64   DPR:$src))), (v1i64 DPR:$src)>;
def : Pat<(v1i64 (bitconvert (v2f32 DPR:$src))), (v1i64 DPR:$src)>;
def : Pat<(v2i32 (bitconvert (v1i64 DPR:$src))), (v2i32 DPR:$src)>;
def : Pat<(v2i32 (bitconvert (v4i16 DPR:$src))), (v2i32 DPR:$src)>;
def : Pat<(v2i32 (bitconvert (v8i8  DPR:$src))), (v2i32 DPR:$src)>;
def : Pat<(v2i32 (bitconvert (f64   DPR:$src))), (v2i32 DPR:$src)>;
def : Pat<(v2i32 (bitconvert (v2f32 DPR:$src))), (v2i32 DPR:$src)>;
def : Pat<(v4i16 (bitconvert (v1i64 DPR:$src))), (v4i16 DPR:$src)>;
def : Pat<(v4i16 (bitconvert (v2i32 DPR:$src))), (v4i16 DPR:$src)>;
def : Pat<(v4i16 (bitconvert (v8i8  DPR:$src))), (v4i16 DPR:$src)>;
def : Pat<(v4i16 (bitconvert (f64   DPR:$src))), (v4i16 DPR:$src)>;
def : Pat<(v4i16 (bitconvert (v2f32 DPR:$src))), (v4i16 DPR:$src)>;
def : Pat<(v8i8  (bitconvert (v1i64 DPR:$src))), (v8i8  DPR:$src)>;
def : Pat<(v8i8  (bitconvert (v2i32 DPR:$src))), (v8i8  DPR:$src)>;
def : Pat<(v8i8  (bitconvert (v4i16 DPR:$src))), (v8i8  DPR:$src)>;
def : Pat<(v8i8  (bitconvert (f64   DPR:$src))), (v8i8  DPR:$src)>;
def : Pat<(v8i8  (bitconvert (v2f32 DPR:$src))), (v8i8  DPR:$src)>;
def : Pat<(f64   (bitconvert (v1i64 DPR:$src))), (f64   DPR:$src)>;
def : Pat<(f64   (bitconvert (v2i32 DPR:$src))), (f64   DPR:$src)>;
def : Pat<(f64   (bitconvert (v4i16 DPR:$src))), (f64   DPR:$src)>;
def : Pat<(f64   (bitconvert (v8i8  DPR:$src))), (f64   DPR:$src)>;
def : Pat<(f64   (bitconvert (v2f32 DPR:$src))), (f64   DPR:$src)>;
def : Pat<(v2f32 (bitconvert (f64   DPR:$src))), (v2f32 DPR:$src)>;
def : Pat<(v2f32 (bitconvert (v1i64 DPR:$src))), (v2f32 DPR:$src)>;
def : Pat<(v2f32 (bitconvert (v2i32 DPR:$src))), (v2f32 DPR:$src)>;
def : Pat<(v2f32 (bitconvert (v4i16 DPR:$src))), (v2f32 DPR:$src)>;
def : Pat<(v2f32 (bitconvert (v8i8  DPR:$src))), (v2f32 DPR:$src)>;

def : Pat<(v2i64 (bitconvert (v4i32 QPR:$src))), (v2i64 QPR:$src)>;
def : Pat<(v2i64 (bitconvert (v8i16 QPR:$src))), (v2i64 QPR:$src)>;
def : Pat<(v2i64 (bitconvert (v16i8 QPR:$src))), (v2i64 QPR:$src)>;
def : Pat<(v2i64 (bitconvert (v2f64 QPR:$src))), (v2i64 QPR:$src)>;
def : Pat<(v2i64 (bitconvert (v4f32 QPR:$src))), (v2i64 QPR:$src)>;
def : Pat<(v4i32 (bitconvert (v2i64 QPR:$src))), (v4i32 QPR:$src)>;
def : Pat<(v4i32 (bitconvert (v8i16 QPR:$src))), (v4i32 QPR:$src)>;
def : Pat<(v4i32 (bitconvert (v16i8 QPR:$src))), (v4i32 QPR:$src)>;
def : Pat<(v4i32 (bitconvert (v2f64 QPR:$src))), (v4i32 QPR:$src)>;
def : Pat<(v4i32 (bitconvert (v4f32 QPR:$src))), (v4i32 QPR:$src)>;
def : Pat<(v8i16 (bitconvert (v2i64 QPR:$src))), (v8i16 QPR:$src)>;
def : Pat<(v8i16 (bitconvert (v4i32 QPR:$src))), (v8i16 QPR:$src)>;
def : Pat<(v8i16 (bitconvert (v16i8 QPR:$src))), (v8i16 QPR:$src)>;
def : Pat<(v8i16 (bitconvert (v2f64 QPR:$src))), (v8i16 QPR:$src)>;
def : Pat<(v8i16 (bitconvert (v4f32 QPR:$src))), (v8i16 QPR:$src)>;
def : Pat<(v16i8 (bitconvert (v2i64 QPR:$src))), (v16i8 QPR:$src)>;
def : Pat<(v16i8 (bitconvert (v4i32 QPR:$src))), (v16i8 QPR:$src)>;
def : Pat<(v16i8 (bitconvert (v8i16 QPR:$src))), (v16i8 QPR:$src)>;
def : Pat<(v16i8 (bitconvert (v2f64 QPR:$src))), (v16i8 QPR:$src)>;
def : Pat<(v16i8 (bitconvert (v4f32 QPR:$src))), (v16i8 QPR:$src)>;
def : Pat<(v4f32 (bitconvert (v2i64 QPR:$src))), (v4f32 QPR:$src)>;
def : Pat<(v4f32 (bitconvert (v4i32 QPR:$src))), (v4f32 QPR:$src)>;
def : Pat<(v4f32 (bitconvert (v8i16 QPR:$src))), (v4f32 QPR:$src)>;
def : Pat<(v4f32 (bitconvert (v16i8 QPR:$src))), (v4f32 QPR:$src)>;
def : Pat<(v4f32 (bitconvert (v2f64 QPR:$src))), (v4f32 QPR:$src)>;
def : Pat<(v2f64 (bitconvert (v2i64 QPR:$src))), (v2f64 QPR:$src)>;
def : Pat<(v2f64 (bitconvert (v4i32 QPR:$src))), (v2f64 QPR:$src)>;
def : Pat<(v2f64 (bitconvert (v8i16 QPR:$src))), (v2f64 QPR:$src)>;
def : Pat<(v2f64 (bitconvert (v16i8 QPR:$src))), (v2f64 QPR:$src)>;
def : Pat<(v2f64 (bitconvert (v4f32 QPR:$src))), (v2f64 QPR:$src)>;
