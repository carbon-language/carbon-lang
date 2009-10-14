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
                                         SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>]>;
def NEONzip       : SDNode<"ARMISD::VZIP", SDTARMVSHUF2>;
def NEONuzp       : SDNode<"ARMISD::VUZP", SDTARMVSHUF2>;
def NEONtrn       : SDNode<"ARMISD::VTRN", SDTARMVSHUF2>;

//===----------------------------------------------------------------------===//
// NEON operand definitions
//===----------------------------------------------------------------------===//

// addrmode_neonldstm := reg
//
/* TODO: Take advantage of vldm.
def addrmode_neonldstm : Operand<i32>,
                ComplexPattern<i32, 2, "SelectAddrModeNeonLdStM", []> {
  let PrintMethod = "printAddrNeonLdStMOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}
*/

//===----------------------------------------------------------------------===//
// NEON load / store instructions
//===----------------------------------------------------------------------===//

/* TODO: Take advantage of vldm.
let mayLoad = 1, hasExtraDefRegAllocReq = 1 in {
def VLDMD : NI<(outs),
               (ins addrmode_neonldstm:$addr, reglist:$dst1, variable_ops),
               IIC_fpLoadm,
               "vldm${addr:submode} ${addr:base}, $dst1",
               []> {
  let Inst{27-25} = 0b110;
  let Inst{20}    = 1;
  let Inst{11-9}  = 0b101;
}

def VLDMS : NI<(outs),
               (ins addrmode_neonldstm:$addr, reglist:$dst1, variable_ops),
               IIC_fpLoadm,
               "vldm${addr:submode} ${addr:base}, $dst1",
               []> {
  let Inst{27-25} = 0b110;
  let Inst{20}    = 1;
  let Inst{11-9}  = 0b101;
}
}
*/

// Use vldmia to load a Q register as a D register pair.
def VLDRQ : NI4<(outs QPR:$dst), (ins addrmode4:$addr),
               IIC_fpLoadm,
               "vldmia $addr, ${dst:dregpair}",
               [(set QPR:$dst, (v2f64 (load addrmode4:$addr)))]> {
  let Inst{27-25} = 0b110;
  let Inst{24}    = 0; // P bit
  let Inst{23}    = 1; // U bit
  let Inst{20}    = 1;
  let Inst{11-9}  = 0b101;
}

// Use vstmia to store a Q register as a D register pair.
def VSTRQ : NI4<(outs), (ins QPR:$src, addrmode4:$addr),
               IIC_fpStorem,
               "vstmia $addr, ${src:dregpair}",
               [(store (v2f64 QPR:$src), addrmode4:$addr)]> {
  let Inst{27-25} = 0b110;
  let Inst{24}    = 0; // P bit
  let Inst{23}    = 1; // U bit
  let Inst{20}    = 0;
  let Inst{11-9}  = 0b101;
}

//   VLD1     : Vector Load (multiple single elements)
class VLD1D<bits<4> op7_4, string OpcodeStr, ValueType Ty, Intrinsic IntOp>
  : NLdSt<0,0b10,0b0111,op7_4, (outs DPR:$dst), (ins addrmode6:$addr), IIC_VLD1,
          !strconcat(OpcodeStr, "\t\\{$dst\\}, $addr"), "",
          [(set DPR:$dst, (Ty (IntOp addrmode6:$addr)))]>;
class VLD1Q<bits<4> op7_4, string OpcodeStr, ValueType Ty, Intrinsic IntOp>
  : NLdSt<0,0b10,0b1010,op7_4, (outs QPR:$dst), (ins addrmode6:$addr), IIC_VLD1,
          !strconcat(OpcodeStr, "\t${dst:dregpair}, $addr"), "",
          [(set QPR:$dst, (Ty (IntOp addrmode6:$addr)))]>;

def  VLD1d8   : VLD1D<0b0000, "vld1.8",  v8i8,  int_arm_neon_vld1>;
def  VLD1d16  : VLD1D<0b0100, "vld1.16", v4i16, int_arm_neon_vld1>;
def  VLD1d32  : VLD1D<0b1000, "vld1.32", v2i32, int_arm_neon_vld1>;
def  VLD1df   : VLD1D<0b1000, "vld1.32", v2f32, int_arm_neon_vld1>;
def  VLD1d64  : VLD1D<0b1100, "vld1.64", v1i64, int_arm_neon_vld1>;

def  VLD1q8   : VLD1Q<0b0000, "vld1.8",  v16i8, int_arm_neon_vld1>;
def  VLD1q16  : VLD1Q<0b0100, "vld1.16", v8i16, int_arm_neon_vld1>;
def  VLD1q32  : VLD1Q<0b1000, "vld1.32", v4i32, int_arm_neon_vld1>;
def  VLD1qf   : VLD1Q<0b1000, "vld1.32", v4f32, int_arm_neon_vld1>;
def  VLD1q64  : VLD1Q<0b1100, "vld1.64", v2i64, int_arm_neon_vld1>;

let mayLoad = 1, hasExtraDefRegAllocReq = 1 in {

//   VLD2     : Vector Load (multiple 2-element structures)
class VLD2D<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b10,0b1000,op7_4, (outs DPR:$dst1, DPR:$dst2),
          (ins addrmode6:$addr), IIC_VLD2,
          !strconcat(OpcodeStr, "\t\\{$dst1,$dst2\\}, $addr"), "", []>;
class VLD2Q<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b10,0b0011,op7_4,
          (outs DPR:$dst1, DPR:$dst2, DPR:$dst3, DPR:$dst4),
          (ins addrmode6:$addr), IIC_VLD2,
          !strconcat(OpcodeStr, "\t\\{$dst1,$dst2,$dst3,$dst4\\}, $addr"),
          "", []>;

def  VLD2d8   : VLD2D<0b0000, "vld2.8">;
def  VLD2d16  : VLD2D<0b0100, "vld2.16">;
def  VLD2d32  : VLD2D<0b1000, "vld2.32">;
def  VLD2d64  : NLdSt<0,0b10,0b1010,0b1100, (outs DPR:$dst1, DPR:$dst2),
                      (ins addrmode6:$addr), IIC_VLD1,
                      "vld1.64\t\\{$dst1,$dst2\\}, $addr", "", []>;

def  VLD2q8   : VLD2Q<0b0000, "vld2.8">;
def  VLD2q16  : VLD2Q<0b0100, "vld2.16">;
def  VLD2q32  : VLD2Q<0b1000, "vld2.32">;

//   VLD3     : Vector Load (multiple 3-element structures)
class VLD3D<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b10,0b0100,op7_4, (outs DPR:$dst1, DPR:$dst2, DPR:$dst3),
          (ins addrmode6:$addr), IIC_VLD3,
          !strconcat(OpcodeStr, "\t\\{$dst1,$dst2,$dst3\\}, $addr"), "", []>;
class VLD3WB<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b10,0b0101,op7_4, (outs DPR:$dst1, DPR:$dst2, DPR:$dst3, GPR:$wb),
          (ins addrmode6:$addr), IIC_VLD3,
          !strconcat(OpcodeStr, "\t\\{$dst1,$dst2,$dst3\\}, $addr"),
          "$addr.addr = $wb", []>;

def  VLD3d8   : VLD3D<0b0000, "vld3.8">;
def  VLD3d16  : VLD3D<0b0100, "vld3.16">;
def  VLD3d32  : VLD3D<0b1000, "vld3.32">;
def  VLD3d64  : NLdSt<0,0b10,0b0110,0b1100,
                      (outs DPR:$dst1, DPR:$dst2, DPR:$dst3),
                      (ins addrmode6:$addr), IIC_VLD1,
                      "vld1.64\t\\{$dst1,$dst2,$dst3\\}, $addr", "", []>;

// vld3 to double-spaced even registers.
def  VLD3q8a  : VLD3WB<0b0000, "vld3.8">;
def  VLD3q16a : VLD3WB<0b0100, "vld3.16">;
def  VLD3q32a : VLD3WB<0b1000, "vld3.32">;

// vld3 to double-spaced odd registers.
def  VLD3q8b  : VLD3WB<0b0000, "vld3.8">;
def  VLD3q16b : VLD3WB<0b0100, "vld3.16">;
def  VLD3q32b : VLD3WB<0b1000, "vld3.32">;

//   VLD4     : Vector Load (multiple 4-element structures)
class VLD4D<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b10,0b0000,op7_4,
          (outs DPR:$dst1, DPR:$dst2, DPR:$dst3, DPR:$dst4),
          (ins addrmode6:$addr), IIC_VLD4,
          !strconcat(OpcodeStr, "\t\\{$dst1,$dst2,$dst3,$dst4\\}, $addr"),
          "", []>;
class VLD4WB<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b10,0b0001,op7_4,
          (outs DPR:$dst1, DPR:$dst2, DPR:$dst3, DPR:$dst4, GPR:$wb),
          (ins addrmode6:$addr), IIC_VLD4,
          !strconcat(OpcodeStr, "\t\\{$dst1,$dst2,$dst3,$dst4\\}, $addr"),
          "$addr.addr = $wb", []>;

def  VLD4d8   : VLD4D<0b0000, "vld4.8">;
def  VLD4d16  : VLD4D<0b0100, "vld4.16">;
def  VLD4d32  : VLD4D<0b1000, "vld4.32">;
def  VLD4d64  : NLdSt<0,0b10,0b0010,0b1100,
                      (outs DPR:$dst1, DPR:$dst2, DPR:$dst3, DPR:$dst4),
                      (ins addrmode6:$addr), IIC_VLD1,
                      "vld1.64\t\\{$dst1,$dst2,$dst3,$dst4\\}, $addr", "", []>;

// vld4 to double-spaced even registers.
def  VLD4q8a  : VLD4WB<0b0000, "vld4.8">;
def  VLD4q16a : VLD4WB<0b0100, "vld4.16">;
def  VLD4q32a : VLD4WB<0b1000, "vld4.32">;

// vld4 to double-spaced odd registers.
def  VLD4q8b  : VLD4WB<0b0000, "vld4.8">;
def  VLD4q16b : VLD4WB<0b0100, "vld4.16">;
def  VLD4q32b : VLD4WB<0b1000, "vld4.32">;

//   VLD1LN   : Vector Load (single element to one lane)
//   FIXME: Not yet implemented.

//   VLD2LN   : Vector Load (single 2-element structure to one lane)
class VLD2LN<bits<4> op11_8, string OpcodeStr>
  : NLdSt<1,0b10,op11_8,0b0000, (outs DPR:$dst1, DPR:$dst2),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, nohash_imm:$lane),
          IIC_VLD2,
          !strconcat(OpcodeStr, "\t\\{$dst1[$lane],$dst2[$lane]\\}, $addr"),
          "$src1 = $dst1, $src2 = $dst2", []>;

def VLD2LNd8  : VLD2LN<0b0001, "vld2.8">;
def VLD2LNd16 : VLD2LN<0b0101, "vld2.16">;
def VLD2LNd32 : VLD2LN<0b1001, "vld2.32">;

// vld2 to double-spaced even registers.
def VLD2LNq16a: VLD2LN<0b0101, "vld2.16">;
def VLD2LNq32a: VLD2LN<0b1001, "vld2.32">;

// vld2 to double-spaced odd registers.
def VLD2LNq16b: VLD2LN<0b0101, "vld2.16">;
def VLD2LNq32b: VLD2LN<0b1001, "vld2.32">;

//   VLD3LN   : Vector Load (single 3-element structure to one lane)
class VLD3LN<bits<4> op11_8, string OpcodeStr>
  : NLdSt<1,0b10,op11_8,0b0000, (outs DPR:$dst1, DPR:$dst2, DPR:$dst3),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3,
          nohash_imm:$lane), IIC_VLD3,
          !strconcat(OpcodeStr,
          "\t\\{$dst1[$lane],$dst2[$lane],$dst3[$lane]\\}, $addr"),
          "$src1 = $dst1, $src2 = $dst2, $src3 = $dst3", []>;

def VLD3LNd8  : VLD3LN<0b0010, "vld3.8">;
def VLD3LNd16 : VLD3LN<0b0110, "vld3.16">;
def VLD3LNd32 : VLD3LN<0b1010, "vld3.32">;

// vld3 to double-spaced even registers.
def VLD3LNq16a: VLD3LN<0b0110, "vld3.16">;
def VLD3LNq32a: VLD3LN<0b1010, "vld3.32">;

// vld3 to double-spaced odd registers.
def VLD3LNq16b: VLD3LN<0b0110, "vld3.16">;
def VLD3LNq32b: VLD3LN<0b1010, "vld3.32">;

//   VLD4LN   : Vector Load (single 4-element structure to one lane)
class VLD4LN<bits<4> op11_8, string OpcodeStr>
  : NLdSt<1,0b10,op11_8,0b0000,
          (outs DPR:$dst1, DPR:$dst2, DPR:$dst3, DPR:$dst4),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3, DPR:$src4,
          nohash_imm:$lane), IIC_VLD4,
          !strconcat(OpcodeStr,
          "\t\\{$dst1[$lane],$dst2[$lane],$dst3[$lane],$dst4[$lane]\\}, $addr"),
          "$src1 = $dst1, $src2 = $dst2, $src3 = $dst3, $src4 = $dst4", []>;

def VLD4LNd8  : VLD4LN<0b0011, "vld4.8">;
def VLD4LNd16 : VLD4LN<0b0111, "vld4.16">;
def VLD4LNd32 : VLD4LN<0b1011, "vld4.32">;

// vld4 to double-spaced even registers.
def VLD4LNq16a: VLD4LN<0b0111, "vld4.16">;
def VLD4LNq32a: VLD4LN<0b1011, "vld4.32">;

// vld4 to double-spaced odd registers.
def VLD4LNq16b: VLD4LN<0b0111, "vld4.16">;
def VLD4LNq32b: VLD4LN<0b1011, "vld4.32">;

//   VLD1DUP  : Vector Load (single element to all lanes)
//   VLD2DUP  : Vector Load (single 2-element structure to all lanes)
//   VLD3DUP  : Vector Load (single 3-element structure to all lanes)
//   VLD4DUP  : Vector Load (single 4-element structure to all lanes)
//   FIXME: Not yet implemented.
} // mayLoad = 1, hasExtraDefRegAllocReq = 1

//   VST1     : Vector Store (multiple single elements)
class VST1D<bits<4> op7_4, string OpcodeStr, ValueType Ty, Intrinsic IntOp>
  : NLdSt<0,0b00,0b0111,op7_4, (outs), (ins addrmode6:$addr, DPR:$src), IIC_VST,
          !strconcat(OpcodeStr, "\t\\{$src\\}, $addr"), "",
          [(IntOp addrmode6:$addr, (Ty DPR:$src))]>;
class VST1Q<bits<4> op7_4, string OpcodeStr, ValueType Ty, Intrinsic IntOp>
  : NLdSt<0,0b00,0b1010,op7_4, (outs), (ins addrmode6:$addr, QPR:$src), IIC_VST,
          !strconcat(OpcodeStr, "\t${src:dregpair}, $addr"), "",
          [(IntOp addrmode6:$addr, (Ty QPR:$src))]>;

let hasExtraSrcRegAllocReq = 1 in {
def  VST1d8   : VST1D<0b0000, "vst1.8",  v8i8,  int_arm_neon_vst1>;
def  VST1d16  : VST1D<0b0100, "vst1.16", v4i16, int_arm_neon_vst1>;
def  VST1d32  : VST1D<0b1000, "vst1.32", v2i32, int_arm_neon_vst1>;
def  VST1df   : VST1D<0b1000, "vst1.32", v2f32, int_arm_neon_vst1>;
def  VST1d64  : VST1D<0b1100, "vst1.64", v1i64, int_arm_neon_vst1>;

def  VST1q8   : VST1Q<0b0000, "vst1.8",  v16i8, int_arm_neon_vst1>;
def  VST1q16  : VST1Q<0b0100, "vst1.16", v8i16, int_arm_neon_vst1>;
def  VST1q32  : VST1Q<0b1000, "vst1.32", v4i32, int_arm_neon_vst1>;
def  VST1qf   : VST1Q<0b1000, "vst1.32", v4f32, int_arm_neon_vst1>;
def  VST1q64  : VST1Q<0b1100, "vst1.64", v2i64, int_arm_neon_vst1>;
} // hasExtraSrcRegAllocReq

let mayStore = 1, hasExtraSrcRegAllocReq = 1 in {

//   VST2     : Vector Store (multiple 2-element structures)
class VST2D<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b00,0b1000,op7_4, (outs),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2), IIC_VST,
          !strconcat(OpcodeStr, "\t\\{$src1,$src2\\}, $addr"), "", []>;
class VST2Q<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b00,0b0011,op7_4, (outs),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3, DPR:$src4),
          IIC_VST,
          !strconcat(OpcodeStr, "\t\\{$src1,$src2,$src3,$src4\\}, $addr"),
          "", []>;

def  VST2d8   : VST2D<0b0000, "vst2.8">;
def  VST2d16  : VST2D<0b0100, "vst2.16">;
def  VST2d32  : VST2D<0b1000, "vst2.32">;
def  VST2d64  : NLdSt<0,0b00,0b1010,0b1100, (outs),
                      (ins addrmode6:$addr, DPR:$src1, DPR:$src2), IIC_VST,
                      "vst1.64\t\\{$src1,$src2\\}, $addr", "", []>;

def  VST2q8   : VST2Q<0b0000, "vst2.8">;
def  VST2q16  : VST2Q<0b0100, "vst2.16">;
def  VST2q32  : VST2Q<0b1000, "vst2.32">;

//   VST3     : Vector Store (multiple 3-element structures)
class VST3D<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b00,0b0100,op7_4, (outs),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3), IIC_VST,
          !strconcat(OpcodeStr, "\t\\{$src1,$src2,$src3\\}, $addr"), "", []>;
class VST3WB<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b00,0b0101,op7_4, (outs GPR:$wb),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3), IIC_VST,
          !strconcat(OpcodeStr, "\t\\{$src1,$src2,$src3\\}, $addr"),
          "$addr.addr = $wb", []>;

def  VST3d8   : VST3D<0b0000, "vst3.8">;
def  VST3d16  : VST3D<0b0100, "vst3.16">;
def  VST3d32  : VST3D<0b1000, "vst3.32">;
def  VST3d64  : NLdSt<0,0b00,0b0110,0b1100, (outs),
                      (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3),
                      IIC_VST,
                      "vst1.64\t\\{$src1,$src2,$src3\\}, $addr", "", []>;

// vst3 to double-spaced even registers.
def  VST3q8a  : VST3WB<0b0000, "vst3.8">;
def  VST3q16a : VST3WB<0b0100, "vst3.16">;
def  VST3q32a : VST3WB<0b1000, "vst3.32">;

// vst3 to double-spaced odd registers.
def  VST3q8b  : VST3WB<0b0000, "vst3.8">;
def  VST3q16b : VST3WB<0b0100, "vst3.16">;
def  VST3q32b : VST3WB<0b1000, "vst3.32">;

//   VST4     : Vector Store (multiple 4-element structures)
class VST4D<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b00,0b0000,op7_4, (outs),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3, DPR:$src4),
          IIC_VST,
          !strconcat(OpcodeStr, "\t\\{$src1,$src2,$src3,$src4\\}, $addr"),
          "", []>;
class VST4WB<bits<4> op7_4, string OpcodeStr>
  : NLdSt<0,0b00,0b0001,op7_4, (outs GPR:$wb),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3, DPR:$src4),
          IIC_VST,
          !strconcat(OpcodeStr, "\t\\{$src1,$src2,$src3,$src4\\}, $addr"),
          "$addr.addr = $wb", []>;

def  VST4d8   : VST4D<0b0000, "vst4.8">;
def  VST4d16  : VST4D<0b0100, "vst4.16">;
def  VST4d32  : VST4D<0b1000, "vst4.32">;
def  VST4d64  : NLdSt<0,0b00,0b0010,0b1100, (outs),
                      (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3,
                       DPR:$src4), IIC_VST,
                      "vst1.64\t\\{$src1,$src2,$src3,$src4\\}, $addr", "", []>;

// vst4 to double-spaced even registers.
def  VST4q8a  : VST4WB<0b0000, "vst4.8">;
def  VST4q16a : VST4WB<0b0100, "vst4.16">;
def  VST4q32a : VST4WB<0b1000, "vst4.32">;

// vst4 to double-spaced odd registers.
def  VST4q8b  : VST4WB<0b0000, "vst4.8">;
def  VST4q16b : VST4WB<0b0100, "vst4.16">;
def  VST4q32b : VST4WB<0b1000, "vst4.32">;

//   VST1LN   : Vector Store (single element from one lane)
//   FIXME: Not yet implemented.

//   VST2LN   : Vector Store (single 2-element structure from one lane)
class VST2LN<bits<4> op11_8, string OpcodeStr>
  : NLdSt<1,0b00,op11_8,0b0000, (outs),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, nohash_imm:$lane),
          IIC_VST,
          !strconcat(OpcodeStr, "\t\\{$src1[$lane],$src2[$lane]\\}, $addr"),
          "", []>;

def VST2LNd8  : VST2LN<0b0000, "vst2.8">;
def VST2LNd16 : VST2LN<0b0100, "vst2.16">;
def VST2LNd32 : VST2LN<0b1000, "vst2.32">;

// vst2 to double-spaced even registers.
def VST2LNq16a: VST2LN<0b0100, "vst2.16">;
def VST2LNq32a: VST2LN<0b1000, "vst2.32">;

// vst2 to double-spaced odd registers.
def VST2LNq16b: VST2LN<0b0100, "vst2.16">;
def VST2LNq32b: VST2LN<0b1000, "vst2.32">;

//   VST3LN   : Vector Store (single 3-element structure from one lane)
class VST3LN<bits<4> op11_8, string OpcodeStr>
  : NLdSt<1,0b00,op11_8,0b0000, (outs),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3,
           nohash_imm:$lane), IIC_VST,
          !strconcat(OpcodeStr,
          "\t\\{$src1[$lane],$src2[$lane],$src3[$lane]\\}, $addr"), "", []>;

def VST3LNd8  : VST3LN<0b0010, "vst3.8">;
def VST3LNd16 : VST3LN<0b0110, "vst3.16">;
def VST3LNd32 : VST3LN<0b1010, "vst3.32">;

// vst3 to double-spaced even registers.
def VST3LNq16a: VST3LN<0b0110, "vst3.16">;
def VST3LNq32a: VST3LN<0b1010, "vst3.32">;

// vst3 to double-spaced odd registers.
def VST3LNq16b: VST3LN<0b0110, "vst3.16">;
def VST3LNq32b: VST3LN<0b1010, "vst3.32">;

//   VST4LN   : Vector Store (single 4-element structure from one lane)
class VST4LN<bits<4> op11_8, string OpcodeStr>
  : NLdSt<1,0b00,op11_8,0b0000, (outs),
          (ins addrmode6:$addr, DPR:$src1, DPR:$src2, DPR:$src3, DPR:$src4,
           nohash_imm:$lane), IIC_VST,
          !strconcat(OpcodeStr,
          "\t\\{$src1[$lane],$src2[$lane],$src3[$lane],$src4[$lane]\\}, $addr"),
          "", []>;

def VST4LNd8  : VST4LN<0b0011, "vst4.8">;
def VST4LNd16 : VST4LN<0b0111, "vst4.16">;
def VST4LNd32 : VST4LN<0b1011, "vst4.32">;

// vst4 to double-spaced even registers.
def VST4LNq16a: VST4LN<0b0111, "vst4.16">;
def VST4LNq32a: VST4LN<0b1011, "vst4.32">;

// vst4 to double-spaced odd registers.
def VST4LNq16b: VST4LN<0b0111, "vst4.16">;
def VST4LNq32b: VST4LN<0b1011, "vst4.32">;

} // mayStore = 1, hasExtraSrcRegAllocReq = 1


//===----------------------------------------------------------------------===//
// NEON pattern fragments
//===----------------------------------------------------------------------===//

// Extract D sub-registers of Q registers.
// (arm_dsubreg_0 is 5; arm_dsubreg_1 is 6)
def DSubReg_i8_reg  : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(5 + N->getZExtValue() / 8, MVT::i32);
}]>;
def DSubReg_i16_reg : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(5 + N->getZExtValue() / 4, MVT::i32);
}]>;
def DSubReg_i32_reg : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(5 + N->getZExtValue() / 2, MVT::i32);
}]>;
def DSubReg_f64_reg : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(5 + N->getZExtValue(), MVT::i32);
}]>;
def DSubReg_f64_other_reg : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(5 + (1 - N->getZExtValue()), MVT::i32);
}]>;

// Extract S sub-registers of Q/D registers.
// (arm_ssubreg_0 is 1; arm_ssubreg_1 is 2; etc.)
def SSubReg_f32_reg : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(1 + N->getZExtValue(), MVT::i32);
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

// Basic 2-register operations, both double- and quad-register.
class N2VD<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
           bits<2> op17_16, bits<5> op11_7, bit op4, string OpcodeStr,
           ValueType ResTy, ValueType OpTy, SDNode OpNode>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4, (outs DPR:$dst),
        (ins DPR:$src), IIC_VUNAD, !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set DPR:$dst, (ResTy (OpNode (OpTy DPR:$src))))]>;
class N2VQ<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
           bits<2> op17_16, bits<5> op11_7, bit op4, string OpcodeStr,
           ValueType ResTy, ValueType OpTy, SDNode OpNode>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 1, op4, (outs QPR:$dst),
        (ins QPR:$src), IIC_VUNAQ, !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set QPR:$dst, (ResTy (OpNode (OpTy QPR:$src))))]>;

// Basic 2-register operations, scalar single-precision.
class N2VDs<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
            bits<2> op17_16, bits<5> op11_7, bit op4, string OpcodeStr,
            ValueType ResTy, ValueType OpTy, SDNode OpNode>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4,
        (outs DPR_VFP2:$dst), (ins DPR_VFP2:$src),
        IIC_VUNAD, !strconcat(OpcodeStr, "\t$dst, $src"), "", []>;

class N2VDsPat<SDNode OpNode, ValueType ResTy, ValueType OpTy, NeonI Inst>
  : NEONFPPat<(ResTy (OpNode SPR:$a)),
       (EXTRACT_SUBREG
           (Inst (INSERT_SUBREG (OpTy (IMPLICIT_DEF)), SPR:$a, arm_ssubreg_0)),
        arm_ssubreg_0)>;

// Basic 2-register intrinsics, both double- and quad-register.
class N2VDInt<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
              bits<2> op17_16, bits<5> op11_7, bit op4, 
              InstrItinClass itin, string OpcodeStr,
              ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4, (outs DPR:$dst),
        (ins DPR:$src), itin, !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set DPR:$dst, (ResTy (IntOp (OpTy DPR:$src))))]>;
class N2VQInt<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
              bits<2> op17_16, bits<5> op11_7, bit op4,
              InstrItinClass itin, string OpcodeStr,
              ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 1, op4, (outs QPR:$dst),
        (ins QPR:$src), itin, !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set QPR:$dst, (ResTy (IntOp (OpTy QPR:$src))))]>;

// Basic 2-register intrinsics, scalar single-precision
class N2VDInts<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
              bits<2> op17_16, bits<5> op11_7, bit op4, 
              InstrItinClass itin, string OpcodeStr,
              ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4,
        (outs DPR_VFP2:$dst), (ins DPR_VFP2:$src), itin,
        !strconcat(OpcodeStr, "\t$dst, $src"), "", []>;

class N2VDIntsPat<SDNode OpNode, NeonI Inst>
  : NEONFPPat<(f32 (OpNode SPR:$a)),
       (EXTRACT_SUBREG
           (Inst (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)), SPR:$a, arm_ssubreg_0)),
        arm_ssubreg_0)>;

// Narrow 2-register intrinsics.
class N2VNInt<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
              bits<2> op17_16, bits<5> op11_7, bit op6, bit op4,
              InstrItinClass itin, string OpcodeStr,
              ValueType TyD, ValueType TyQ, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, op6, op4, (outs DPR:$dst),
        (ins QPR:$src), itin, !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set DPR:$dst, (TyD (IntOp (TyQ QPR:$src))))]>;

// Long 2-register intrinsics.  (This is currently only used for VMOVL and is
// derived from N2VImm instead of N2V because of the way the size is encoded.)
class N2VLInt<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
              bit op6, bit op4, InstrItinClass itin, string OpcodeStr,
              ValueType TyQ, ValueType TyD, Intrinsic IntOp>
  : N2VImm<op24, op23, op21_16, op11_8, op7, op6, op4, (outs QPR:$dst),
        (ins DPR:$src), itin, !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set QPR:$dst, (TyQ (IntOp (TyD DPR:$src))))]>;

// 2-register shuffles (VTRN/VZIP/VUZP), both double- and quad-register.
class N2VDShuffle<bits<2> op19_18, bits<5> op11_7, string OpcodeStr>
  : N2V<0b11, 0b11, op19_18, 0b10, op11_7, 0, 0, (outs DPR:$dst1, DPR:$dst2),
        (ins DPR:$src1, DPR:$src2), IIC_VPERMD, 
        !strconcat(OpcodeStr, "\t$dst1, $dst2"),
        "$src1 = $dst1, $src2 = $dst2", []>;
class N2VQShuffle<bits<2> op19_18, bits<5> op11_7,
                  InstrItinClass itin, string OpcodeStr>
  : N2V<0b11, 0b11, op19_18, 0b10, op11_7, 1, 0, (outs QPR:$dst1, QPR:$dst2),
        (ins QPR:$src1, QPR:$src2), itin, 
        !strconcat(OpcodeStr, "\t$dst1, $dst2"),
        "$src1 = $dst1, $src2 = $dst2", []>;

// Basic 3-register operations, both double- and quad-register.
class N3VD<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
           InstrItinClass itin, string OpcodeStr, ValueType ResTy, ValueType OpTy,
           SDNode OpNode, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$dst), (ins DPR:$src1, DPR:$src2), itin, 
        !strconcat(OpcodeStr, "\t$dst, $src1, $src2"), "",
        [(set DPR:$dst, (ResTy (OpNode (OpTy DPR:$src1), (OpTy DPR:$src2))))]> {
  let isCommutable = Commutable;
}
class N3VDSL<bits<2> op21_20, bits<4> op11_8, 
             InstrItinClass itin, string OpcodeStr, ValueType Ty, SDNode ShOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst), (ins DPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane),
        itin, !strconcat(OpcodeStr, "\t$dst, $src1, $src2[$lane]"), "",
        [(set (Ty DPR:$dst),
              (Ty (ShOp (Ty DPR:$src1),
                        (Ty (NEONvduplane (Ty DPR_VFP2:$src2),
                                          imm:$lane)))))]> {
  let isCommutable = 0;
}
class N3VDSL16<bits<2> op21_20, bits<4> op11_8, 
               string OpcodeStr, ValueType Ty, SDNode ShOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst), (ins DPR:$src1, DPR_8:$src2, nohash_imm:$lane),
        IIC_VMULi16D,
        !strconcat(OpcodeStr, "\t$dst, $src1, $src2[$lane]"), "",
        [(set (Ty DPR:$dst),
              (Ty (ShOp (Ty DPR:$src1),
                        (Ty (NEONvduplane (Ty DPR_8:$src2),
                                          imm:$lane)))))]> {
  let isCommutable = 0;
}

class N3VQ<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
           InstrItinClass itin, string OpcodeStr, ValueType ResTy, ValueType OpTy,
           SDNode OpNode, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 1, op4,
        (outs QPR:$dst), (ins QPR:$src1, QPR:$src2), itin, 
        !strconcat(OpcodeStr, "\t$dst, $src1, $src2"), "",
        [(set QPR:$dst, (ResTy (OpNode (OpTy QPR:$src1), (OpTy QPR:$src2))))]> {
  let isCommutable = Commutable;
}
class N3VQSL<bits<2> op21_20, bits<4> op11_8, 
             InstrItinClass itin, string OpcodeStr, 
             ValueType ResTy, ValueType OpTy, SDNode ShOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins QPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane),
        itin, !strconcat(OpcodeStr, "\t$dst, $src1, $src2[$lane]"), "",
        [(set (ResTy QPR:$dst),
              (ResTy (ShOp (ResTy QPR:$src1),
                           (ResTy (NEONvduplane (OpTy DPR_VFP2:$src2),
                                                imm:$lane)))))]> {
  let isCommutable = 0;
}
class N3VQSL16<bits<2> op21_20, bits<4> op11_8, 
               string OpcodeStr, ValueType ResTy, ValueType OpTy, SDNode ShOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins QPR:$src1, DPR_8:$src2, nohash_imm:$lane),
        IIC_VMULi16Q,
        !strconcat(OpcodeStr, "\t$dst, $src1, $src2[$lane]"), "",
        [(set (ResTy QPR:$dst),
              (ResTy (ShOp (ResTy QPR:$src1),
                           (ResTy (NEONvduplane (OpTy DPR_8:$src2),
                                                imm:$lane)))))]> {
  let isCommutable = 0;
}

// Basic 3-register operations, scalar single-precision
class N3VDs<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
           string OpcodeStr, ValueType ResTy, ValueType OpTy,
           SDNode OpNode, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR_VFP2:$dst), (ins DPR_VFP2:$src1, DPR_VFP2:$src2), IIC_VBIND,
        !strconcat(OpcodeStr, "\t$dst, $src1, $src2"), "", []> {
  let isCommutable = Commutable;
}
class N3VDsPat<SDNode OpNode, NeonI Inst>
  : NEONFPPat<(f32 (OpNode SPR:$a, SPR:$b)),
       (EXTRACT_SUBREG
           (Inst (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)), SPR:$a, arm_ssubreg_0),
                 (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)), SPR:$b, arm_ssubreg_0)),
        arm_ssubreg_0)>;

// Basic 3-register intrinsics, both double- and quad-register.
class N3VDInt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              InstrItinClass itin, string OpcodeStr, ValueType ResTy, ValueType OpTy,
              Intrinsic IntOp, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$dst), (ins DPR:$src1, DPR:$src2), itin, 
        !strconcat(OpcodeStr, "\t$dst, $src1, $src2"), "",
        [(set DPR:$dst, (ResTy (IntOp (OpTy DPR:$src1), (OpTy DPR:$src2))))]> {
  let isCommutable = Commutable;
}
class N3VDIntSL<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin, 
                string OpcodeStr, ValueType Ty, Intrinsic IntOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst), (ins DPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane),
        itin, !strconcat(OpcodeStr, "\t$dst, $src1, $src2[$lane]"), "",
        [(set (Ty DPR:$dst),
              (Ty (IntOp (Ty DPR:$src1),
                         (Ty (NEONvduplane (Ty DPR_VFP2:$src2),
                                           imm:$lane)))))]> {
  let isCommutable = 0;
}
class N3VDIntSL16<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                  string OpcodeStr, ValueType Ty, Intrinsic IntOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst), (ins DPR:$src1, DPR_8:$src2, nohash_imm:$lane),
        itin, !strconcat(OpcodeStr, "\t$dst, $src1, $src2[$lane]"), "",
        [(set (Ty DPR:$dst),
              (Ty (IntOp (Ty DPR:$src1),
                         (Ty (NEONvduplane (Ty DPR_8:$src2),
                                           imm:$lane)))))]> {
  let isCommutable = 0;
}

class N3VQInt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              InstrItinClass itin, string OpcodeStr, ValueType ResTy, ValueType OpTy,
              Intrinsic IntOp, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 1, op4,
        (outs QPR:$dst), (ins QPR:$src1, QPR:$src2), itin, 
        !strconcat(OpcodeStr, "\t$dst, $src1, $src2"), "",
        [(set QPR:$dst, (ResTy (IntOp (OpTy QPR:$src1), (OpTy QPR:$src2))))]> {
  let isCommutable = Commutable;
}
class N3VQIntSL<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin, 
                string OpcodeStr, ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins QPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane),
        itin, !strconcat(OpcodeStr, "\t$dst, $src1, $src2[$lane]"), "",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (ResTy QPR:$src1),
                            (ResTy (NEONvduplane (OpTy DPR_VFP2:$src2),
                                                 imm:$lane)))))]> {
  let isCommutable = 0;
}
class N3VQIntSL16<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                  string OpcodeStr, ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins QPR:$src1, DPR_8:$src2, nohash_imm:$lane),
        itin, !strconcat(OpcodeStr, "\t$dst, $src1, $src2[$lane]"), "",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (ResTy QPR:$src1),
                            (ResTy (NEONvduplane (OpTy DPR_8:$src2),
                                                 imm:$lane)))))]> {
  let isCommutable = 0;
}

// Multiply-Add/Sub operations, both double- and quad-register.
class N3VDMulOp<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                InstrItinClass itin, string OpcodeStr, 
                ValueType Ty, SDNode MulOp, SDNode OpNode>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$dst), (ins DPR:$src1, DPR:$src2, DPR:$src3), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3"), "$src1 = $dst",
        [(set DPR:$dst, (Ty (OpNode DPR:$src1,
                             (Ty (MulOp DPR:$src2, DPR:$src3)))))]>;
class N3VDMulOpSL<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                  string OpcodeStr, ValueType Ty, SDNode MulOp, SDNode ShOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst),
        (ins DPR:$src1, DPR:$src2, DPR_VFP2:$src3, nohash_imm:$lane), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3[$lane]"), "$src1 = $dst",
        [(set (Ty DPR:$dst),
              (Ty (ShOp (Ty DPR:$src1),
                        (Ty (MulOp DPR:$src2,
                                   (Ty (NEONvduplane (Ty DPR_VFP2:$src3),
                                                     imm:$lane)))))))]>;
class N3VDMulOpSL16<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                    string OpcodeStr, ValueType Ty, SDNode MulOp, SDNode ShOp>
  : N3V<0, 1, op21_20, op11_8, 1, 0,
        (outs DPR:$dst),
        (ins DPR:$src1, DPR:$src2, DPR_8:$src3, nohash_imm:$lane), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3[$lane]"), "$src1 = $dst",
        [(set (Ty DPR:$dst),
              (Ty (ShOp (Ty DPR:$src1),
                        (Ty (MulOp DPR:$src2,
                                   (Ty (NEONvduplane (Ty DPR_8:$src3),
                                                     imm:$lane)))))))]>;

class N3VQMulOp<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                InstrItinClass itin, string OpcodeStr, ValueType Ty,
                SDNode MulOp, SDNode OpNode>
  : N3V<op24, op23, op21_20, op11_8, 1, op4,
        (outs QPR:$dst), (ins QPR:$src1, QPR:$src2, QPR:$src3), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3"), "$src1 = $dst",
        [(set QPR:$dst, (Ty (OpNode QPR:$src1,
                             (Ty (MulOp QPR:$src2, QPR:$src3)))))]>;
class N3VQMulOpSL<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                  string OpcodeStr, ValueType ResTy, ValueType OpTy,
                  SDNode MulOp, SDNode ShOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst),
        (ins QPR:$src1, QPR:$src2, DPR_VFP2:$src3, nohash_imm:$lane), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3[$lane]"), "$src1 = $dst",
        [(set (ResTy QPR:$dst),
              (ResTy (ShOp (ResTy QPR:$src1),
                           (ResTy (MulOp QPR:$src2,
                                         (ResTy (NEONvduplane (OpTy DPR_VFP2:$src3),
                                                              imm:$lane)))))))]>;
class N3VQMulOpSL16<bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                    string OpcodeStr, ValueType ResTy, ValueType OpTy,
                    SDNode MulOp, SDNode ShOp>
  : N3V<1, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst),
        (ins QPR:$src1, QPR:$src2, DPR_8:$src3, nohash_imm:$lane), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3[$lane]"), "$src1 = $dst",
        [(set (ResTy QPR:$dst),
              (ResTy (ShOp (ResTy QPR:$src1),
                           (ResTy (MulOp QPR:$src2,
                                         (ResTy (NEONvduplane (OpTy DPR_8:$src3),
                                                              imm:$lane)))))))]>;

// Multiply-Add/Sub operations, scalar single-precision
class N3VDMulOps<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
                 InstrItinClass itin, string OpcodeStr,
                 ValueType Ty, SDNode MulOp, SDNode OpNode>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR_VFP2:$dst),
        (ins DPR_VFP2:$src1, DPR_VFP2:$src2, DPR_VFP2:$src3), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3"), "$src1 = $dst", []>;

class N3VDMulOpsPat<SDNode MulNode, SDNode OpNode, NeonI Inst>
  : NEONFPPat<(f32 (OpNode SPR:$acc, (f32 (MulNode SPR:$a, SPR:$b)))),
      (EXTRACT_SUBREG
          (Inst (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)), SPR:$acc, arm_ssubreg_0),
                (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)), SPR:$a,   arm_ssubreg_0),
                (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)), SPR:$b,   arm_ssubreg_0)),
       arm_ssubreg_0)>;

// Neon 3-argument intrinsics, both double- and quad-register.
// The destination register is also used as the first source operand register.
class N3VDInt3<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
               InstrItinClass itin, string OpcodeStr,
               ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$dst), (ins DPR:$src1, DPR:$src2, DPR:$src3), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3"), "$src1 = $dst",
        [(set DPR:$dst, (ResTy (IntOp (OpTy DPR:$src1),
                                      (OpTy DPR:$src2), (OpTy DPR:$src3))))]>;
class N3VQInt3<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
               InstrItinClass itin, string OpcodeStr,
               ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, op23, op21_20, op11_8, 1, op4,
        (outs QPR:$dst), (ins QPR:$src1, QPR:$src2, QPR:$src3), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3"), "$src1 = $dst",
        [(set QPR:$dst, (ResTy (IntOp (OpTy QPR:$src1),
                                      (OpTy QPR:$src2), (OpTy QPR:$src3))))]>;

// Neon Long 3-argument intrinsic.  The destination register is
// a quad-register and is also used as the first source operand register.
class N3VLInt3<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
               InstrItinClass itin, string OpcodeStr,
               ValueType TyQ, ValueType TyD, Intrinsic IntOp>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$dst), (ins QPR:$src1, DPR:$src2, DPR:$src3), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3"), "$src1 = $dst",
        [(set QPR:$dst,
          (TyQ (IntOp (TyQ QPR:$src1), (TyD DPR:$src2), (TyD DPR:$src3))))]>;
class N3VLInt3SL<bit op24, bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                 string OpcodeStr, ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst),
        (ins QPR:$src1, DPR:$src2, DPR_VFP2:$src3, nohash_imm:$lane), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3[$lane]"), "$src1 = $dst",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (ResTy QPR:$src1),
                            (OpTy DPR:$src2),
                            (OpTy (NEONvduplane (OpTy DPR_VFP2:$src3),
                                                imm:$lane)))))]>;
class N3VLInt3SL16<bit op24, bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                   string OpcodeStr, ValueType ResTy, ValueType OpTy,
                   Intrinsic IntOp>
  : N3V<op24, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst),
        (ins QPR:$src1, DPR:$src2, DPR_8:$src3, nohash_imm:$lane), itin,
        !strconcat(OpcodeStr, "\t$dst, $src2, $src3[$lane]"), "$src1 = $dst",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (ResTy QPR:$src1),
                            (OpTy DPR:$src2),
                            (OpTy (NEONvduplane (OpTy DPR_8:$src3),
                                                imm:$lane)))))]>;


// Narrowing 3-register intrinsics.
class N3VNInt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              string OpcodeStr, ValueType TyD, ValueType TyQ,
              Intrinsic IntOp, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs DPR:$dst), (ins QPR:$src1, QPR:$src2), IIC_VBINi4D,
        !strconcat(OpcodeStr, "\t$dst, $src1, $src2"), "",
        [(set DPR:$dst, (TyD (IntOp (TyQ QPR:$src1), (TyQ QPR:$src2))))]> {
  let isCommutable = Commutable;
}

// Long 3-register intrinsics.
class N3VLInt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              InstrItinClass itin, string OpcodeStr, ValueType TyQ, ValueType TyD,
              Intrinsic IntOp, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$dst), (ins DPR:$src1, DPR:$src2), itin,
        !strconcat(OpcodeStr, "\t$dst, $src1, $src2"), "",
        [(set QPR:$dst, (TyQ (IntOp (TyD DPR:$src1), (TyD DPR:$src2))))]> {
  let isCommutable = Commutable;
}
class N3VLIntSL<bit op24, bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                string OpcodeStr, ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N3V<op24, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins DPR:$src1, DPR_VFP2:$src2, nohash_imm:$lane), 
        itin, !strconcat(OpcodeStr, "\t$dst, $src1, $src2[$lane]"), "",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (OpTy DPR:$src1),
                            (OpTy (NEONvduplane (OpTy DPR_VFP2:$src2),
                                                imm:$lane)))))]>;
class N3VLIntSL16<bit op24, bits<2> op21_20, bits<4> op11_8, InstrItinClass itin,
                  string OpcodeStr, ValueType ResTy, ValueType OpTy, 
                  Intrinsic IntOp>
  : N3V<op24, 1, op21_20, op11_8, 1, 0,
        (outs QPR:$dst), (ins DPR:$src1, DPR_8:$src2, nohash_imm:$lane), 
        itin, !strconcat(OpcodeStr, "\t$dst, $src1, $src2[$lane]"), "",
        [(set (ResTy QPR:$dst),
              (ResTy (IntOp (OpTy DPR:$src1),
                            (OpTy (NEONvduplane (OpTy DPR_8:$src2),
                                                imm:$lane)))))]>;

// Wide 3-register intrinsics.
class N3VWInt<bit op24, bit op23, bits<2> op21_20, bits<4> op11_8, bit op4,
              string OpcodeStr, ValueType TyQ, ValueType TyD,
              Intrinsic IntOp, bit Commutable>
  : N3V<op24, op23, op21_20, op11_8, 0, op4,
        (outs QPR:$dst), (ins QPR:$src1, DPR:$src2), IIC_VSUBiD,
        !strconcat(OpcodeStr, "\t$dst, $src1, $src2"), "",
        [(set QPR:$dst, (TyQ (IntOp (TyQ QPR:$src1), (TyD DPR:$src2))))]> {
  let isCommutable = Commutable;
}

// Pairwise long 2-register intrinsics, both double- and quad-register.
class N2VDPLInt<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
                bits<2> op17_16, bits<5> op11_7, bit op4, string OpcodeStr,
                ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4, (outs DPR:$dst),
        (ins DPR:$src), IIC_VSHLiD, !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set DPR:$dst, (ResTy (IntOp (OpTy DPR:$src))))]>;
class N2VQPLInt<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
                bits<2> op17_16, bits<5> op11_7, bit op4, string OpcodeStr,
                ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 1, op4, (outs QPR:$dst),
        (ins QPR:$src), IIC_VSHLiD, !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set QPR:$dst, (ResTy (IntOp (OpTy QPR:$src))))]>;

// Pairwise long 2-register accumulate intrinsics,
// both double- and quad-register.
// The destination register is also used as the first source operand register.
class N2VDPLInt2<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
                 bits<2> op17_16, bits<5> op11_7, bit op4, string OpcodeStr,
                 ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 0, op4,
        (outs DPR:$dst), (ins DPR:$src1, DPR:$src2), IIC_VPALiD,
        !strconcat(OpcodeStr, "\t$dst, $src2"), "$src1 = $dst",
        [(set DPR:$dst, (ResTy (IntOp (ResTy DPR:$src1), (OpTy DPR:$src2))))]>;
class N2VQPLInt2<bits<2> op24_23, bits<2> op21_20, bits<2> op19_18,
                 bits<2> op17_16, bits<5> op11_7, bit op4, string OpcodeStr,
                 ValueType ResTy, ValueType OpTy, Intrinsic IntOp>
  : N2V<op24_23, op21_20, op19_18, op17_16, op11_7, 1, op4,
        (outs QPR:$dst), (ins QPR:$src1, QPR:$src2), IIC_VPALiQ,
        !strconcat(OpcodeStr, "\t$dst, $src2"), "$src1 = $dst",
        [(set QPR:$dst, (ResTy (IntOp (ResTy QPR:$src1), (OpTy QPR:$src2))))]>;

// Shift by immediate,
// both double- and quad-register.
class N2VDSh<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
             bit op4, InstrItinClass itin, string OpcodeStr,
             ValueType Ty, SDNode OpNode>
  : N2VImm<op24, op23, op21_16, op11_8, op7, 0, op4,
           (outs DPR:$dst), (ins DPR:$src, i32imm:$SIMM), itin,
           !strconcat(OpcodeStr, "\t$dst, $src, $SIMM"), "",
           [(set DPR:$dst, (Ty (OpNode (Ty DPR:$src), (i32 imm:$SIMM))))]>;
class N2VQSh<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
             bit op4, InstrItinClass itin, string OpcodeStr,
             ValueType Ty, SDNode OpNode>
  : N2VImm<op24, op23, op21_16, op11_8, op7, 1, op4,
           (outs QPR:$dst), (ins QPR:$src, i32imm:$SIMM), itin,
           !strconcat(OpcodeStr, "\t$dst, $src, $SIMM"), "",
           [(set QPR:$dst, (Ty (OpNode (Ty QPR:$src), (i32 imm:$SIMM))))]>;

// Long shift by immediate.
class N2VLSh<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
             bit op6, bit op4, string OpcodeStr, ValueType ResTy,
             ValueType OpTy, SDNode OpNode>
  : N2VImm<op24, op23, op21_16, op11_8, op7, op6, op4,
           (outs QPR:$dst), (ins DPR:$src, i32imm:$SIMM), IIC_VSHLiD,
           !strconcat(OpcodeStr, "\t$dst, $src, $SIMM"), "",
           [(set QPR:$dst, (ResTy (OpNode (OpTy DPR:$src),
                                          (i32 imm:$SIMM))))]>;

// Narrow shift by immediate.
class N2VNSh<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
             bit op6, bit op4, InstrItinClass itin, string OpcodeStr,
             ValueType ResTy, ValueType OpTy, SDNode OpNode>
  : N2VImm<op24, op23, op21_16, op11_8, op7, op6, op4,
           (outs DPR:$dst), (ins QPR:$src, i32imm:$SIMM), itin,
           !strconcat(OpcodeStr, "\t$dst, $src, $SIMM"), "",
           [(set DPR:$dst, (ResTy (OpNode (OpTy QPR:$src),
                                          (i32 imm:$SIMM))))]>;

// Shift right by immediate and accumulate,
// both double- and quad-register.
class N2VDShAdd<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
                bit op4, string OpcodeStr, ValueType Ty, SDNode ShOp>
  : N2VImm<op24, op23, op21_16, op11_8, op7, 0, op4,
           (outs DPR:$dst), (ins DPR:$src1, DPR:$src2, i32imm:$SIMM),
           IIC_VPALiD, 
           !strconcat(OpcodeStr, "\t$dst, $src2, $SIMM"), "$src1 = $dst",
           [(set DPR:$dst, (Ty (add DPR:$src1,
                                (Ty (ShOp DPR:$src2, (i32 imm:$SIMM))))))]>;
class N2VQShAdd<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
                bit op4, string OpcodeStr, ValueType Ty, SDNode ShOp>
  : N2VImm<op24, op23, op21_16, op11_8, op7, 1, op4,
           (outs QPR:$dst), (ins QPR:$src1, QPR:$src2, i32imm:$SIMM),
           IIC_VPALiD, 
           !strconcat(OpcodeStr, "\t$dst, $src2, $SIMM"), "$src1 = $dst",
           [(set QPR:$dst, (Ty (add QPR:$src1,
                                (Ty (ShOp QPR:$src2, (i32 imm:$SIMM))))))]>;

// Shift by immediate and insert,
// both double- and quad-register.
class N2VDShIns<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
                bit op4, string OpcodeStr, ValueType Ty, SDNode ShOp>
  : N2VImm<op24, op23, op21_16, op11_8, op7, 0, op4,
           (outs DPR:$dst), (ins DPR:$src1, DPR:$src2, i32imm:$SIMM),
           IIC_VSHLiD, 
           !strconcat(OpcodeStr, "\t$dst, $src2, $SIMM"), "$src1 = $dst",
           [(set DPR:$dst, (Ty (ShOp DPR:$src1, DPR:$src2, (i32 imm:$SIMM))))]>;
class N2VQShIns<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
                bit op4, string OpcodeStr, ValueType Ty, SDNode ShOp>
  : N2VImm<op24, op23, op21_16, op11_8, op7, 1, op4,
           (outs QPR:$dst), (ins QPR:$src1, QPR:$src2, i32imm:$SIMM),
           IIC_VSHLiQ, 
           !strconcat(OpcodeStr, "\t$dst, $src2, $SIMM"), "$src1 = $dst",
           [(set QPR:$dst, (Ty (ShOp QPR:$src1, QPR:$src2, (i32 imm:$SIMM))))]>;

// Convert, with fractional bits immediate,
// both double- and quad-register.
class N2VCvtD<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
              bit op4, string OpcodeStr, ValueType ResTy, ValueType OpTy,
              Intrinsic IntOp>
  : N2VImm<op24, op23, op21_16, op11_8, op7, 0, op4,
           (outs DPR:$dst), (ins DPR:$src, i32imm:$SIMM), IIC_VUNAD, 
           !strconcat(OpcodeStr, "\t$dst, $src, $SIMM"), "",
           [(set DPR:$dst, (ResTy (IntOp (OpTy DPR:$src), (i32 imm:$SIMM))))]>;
class N2VCvtQ<bit op24, bit op23, bits<6> op21_16, bits<4> op11_8, bit op7,
              bit op4, string OpcodeStr, ValueType ResTy, ValueType OpTy,
              Intrinsic IntOp>
  : N2VImm<op24, op23, op21_16, op11_8, op7, 1, op4,
           (outs QPR:$dst), (ins QPR:$src, i32imm:$SIMM), IIC_VUNAQ, 
           !strconcat(OpcodeStr, "\t$dst, $src, $SIMM"), "",
           [(set QPR:$dst, (ResTy (IntOp (OpTy QPR:$src), (i32 imm:$SIMM))))]>;

//===----------------------------------------------------------------------===//
// Multiclasses
//===----------------------------------------------------------------------===//

// Abbreviations used in multiclass suffixes:
//   Q = quarter int (8 bit) elements
//   H = half int (16 bit) elements
//   S = single int (32 bit) elements
//   D = double int (64 bit) elements

// Neon 3-register vector operations.

// First with only element sizes of 8, 16 and 32 bits:
multiclass N3V_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                   InstrItinClass itinD16, InstrItinClass itinD32,
                   InstrItinClass itinQ16, InstrItinClass itinQ32,
                   string OpcodeStr, SDNode OpNode, bit Commutable = 0> {
  // 64-bit vector types.
  def v8i8  : N3VD<op24, op23, 0b00, op11_8, op4, itinD16, 
                   !strconcat(OpcodeStr, "8"), v8i8, v8i8, OpNode, Commutable>;
  def v4i16 : N3VD<op24, op23, 0b01, op11_8, op4, itinD16,
                   !strconcat(OpcodeStr, "16"), v4i16, v4i16, OpNode, Commutable>;
  def v2i32 : N3VD<op24, op23, 0b10, op11_8, op4, itinD32,
                   !strconcat(OpcodeStr, "32"), v2i32, v2i32, OpNode, Commutable>;

  // 128-bit vector types.
  def v16i8 : N3VQ<op24, op23, 0b00, op11_8, op4, itinQ16,
                   !strconcat(OpcodeStr, "8"), v16i8, v16i8, OpNode, Commutable>;
  def v8i16 : N3VQ<op24, op23, 0b01, op11_8, op4, itinQ16,
                   !strconcat(OpcodeStr, "16"), v8i16, v8i16, OpNode, Commutable>;
  def v4i32 : N3VQ<op24, op23, 0b10, op11_8, op4, itinQ32,
                   !strconcat(OpcodeStr, "32"), v4i32, v4i32, OpNode, Commutable>;
}

multiclass N3VSL_HS<bits<4> op11_8, string OpcodeStr, SDNode ShOp> {
  def v4i16 : N3VDSL16<0b01, op11_8, !strconcat(OpcodeStr, "16"), v4i16, ShOp>;
  def v2i32 : N3VDSL<0b10, op11_8, IIC_VMULi32D, !strconcat(OpcodeStr, "32"), v2i32, ShOp>;
  def v8i16 : N3VQSL16<0b01, op11_8, !strconcat(OpcodeStr, "16"), v8i16, v4i16, ShOp>;
  def v4i32 : N3VQSL<0b10, op11_8, IIC_VMULi32Q, !strconcat(OpcodeStr, "32"), v4i32, v2i32, ShOp>;
}

// ....then also with element size 64 bits:
multiclass N3V_QHSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                    InstrItinClass itinD, InstrItinClass itinQ,
                    string OpcodeStr, SDNode OpNode, bit Commutable = 0>
  : N3V_QHS<op24, op23, op11_8, op4, itinD, itinD, itinQ, itinQ,
            OpcodeStr, OpNode, Commutable> {
  def v1i64 : N3VD<op24, op23, 0b11, op11_8, op4, itinD,
                   !strconcat(OpcodeStr, "64"), v1i64, v1i64, OpNode, Commutable>;
  def v2i64 : N3VQ<op24, op23, 0b11, op11_8, op4, itinQ,
                   !strconcat(OpcodeStr, "64"), v2i64, v2i64, OpNode, Commutable>;
}


// Neon Narrowing 2-register vector intrinsics,
//   source operand element sizes of 16, 32 and 64 bits:
multiclass N2VNInt_HSD<bits<2> op24_23, bits<2> op21_20, bits<2> op17_16,
                       bits<5> op11_7, bit op6, bit op4, 
                       InstrItinClass itin, string OpcodeStr,
                       Intrinsic IntOp> {
  def v8i8  : N2VNInt<op24_23, op21_20, 0b00, op17_16, op11_7, op6, op4,
                      itin, !strconcat(OpcodeStr, "16"), v8i8, v8i16, IntOp>;
  def v4i16 : N2VNInt<op24_23, op21_20, 0b01, op17_16, op11_7, op6, op4,
                      itin, !strconcat(OpcodeStr, "32"), v4i16, v4i32, IntOp>;
  def v2i32 : N2VNInt<op24_23, op21_20, 0b10, op17_16, op11_7, op6, op4,
                      itin, !strconcat(OpcodeStr, "64"), v2i32, v2i64, IntOp>;
}


// Neon Lengthening 2-register vector intrinsic (currently specific to VMOVL).
//   source operand element sizes of 16, 32 and 64 bits:
multiclass N2VLInt_QHS<bit op24, bit op23, bits<4> op11_8, bit op7, bit op6,
                       bit op4, string OpcodeStr, Intrinsic IntOp> {
  def v8i16 : N2VLInt<op24, op23, 0b001000, op11_8, op7, op6, op4,
                      IIC_VQUNAiD, !strconcat(OpcodeStr, "8"), v8i16, v8i8, IntOp>;
  def v4i32 : N2VLInt<op24, op23, 0b010000, op11_8, op7, op6, op4,
                      IIC_VQUNAiD, !strconcat(OpcodeStr, "16"), v4i32, v4i16, IntOp>;
  def v2i64 : N2VLInt<op24, op23, 0b100000, op11_8, op7, op6, op4,
                      IIC_VQUNAiD, !strconcat(OpcodeStr, "32"), v2i64, v2i32, IntOp>;
}


// Neon 3-register vector intrinsics.

// First with only element sizes of 16 and 32 bits:
multiclass N3VInt_HS<bit op24, bit op23, bits<4> op11_8, bit op4,
                     InstrItinClass itinD16, InstrItinClass itinD32,
                     InstrItinClass itinQ16, InstrItinClass itinQ32,
                     string OpcodeStr, Intrinsic IntOp, bit Commutable = 0> {
  // 64-bit vector types.
  def v4i16 : N3VDInt<op24, op23, 0b01, op11_8, op4, itinD16, !strconcat(OpcodeStr,"16"),
                      v4i16, v4i16, IntOp, Commutable>;
  def v2i32 : N3VDInt<op24, op23, 0b10, op11_8, op4, itinD32, !strconcat(OpcodeStr,"32"),
                      v2i32, v2i32, IntOp, Commutable>;

  // 128-bit vector types.
  def v8i16 : N3VQInt<op24, op23, 0b01, op11_8, op4, itinQ16, !strconcat(OpcodeStr,"16"),
                      v8i16, v8i16, IntOp, Commutable>;
  def v4i32 : N3VQInt<op24, op23, 0b10, op11_8, op4, itinQ32, !strconcat(OpcodeStr,"32"),
                      v4i32, v4i32, IntOp, Commutable>;
}

multiclass N3VIntSL_HS<bits<4> op11_8, 
                       InstrItinClass itinD16, InstrItinClass itinD32,
                       InstrItinClass itinQ16, InstrItinClass itinQ32,
                       string OpcodeStr, Intrinsic IntOp> {
  def v4i16 : N3VDIntSL16<0b01, op11_8, itinD16, !strconcat(OpcodeStr, "16"), v4i16, IntOp>;
  def v2i32 : N3VDIntSL<0b10, op11_8, itinD32, !strconcat(OpcodeStr, "32"), v2i32, IntOp>;
  def v8i16 : N3VQIntSL16<0b01, op11_8, itinQ16, !strconcat(OpcodeStr, "16"), v8i16, v4i16, IntOp>;
  def v4i32 : N3VQIntSL<0b10, op11_8, itinQ32, !strconcat(OpcodeStr, "32"), v4i32, v2i32, IntOp>;
}

// ....then also with element size of 8 bits:
multiclass N3VInt_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                      InstrItinClass itinD16, InstrItinClass itinD32,
                      InstrItinClass itinQ16, InstrItinClass itinQ32,
                      string OpcodeStr, Intrinsic IntOp, bit Commutable = 0>
  : N3VInt_HS<op24, op23, op11_8, op4, itinD16, itinD32, itinQ16, itinQ32,
              OpcodeStr, IntOp, Commutable> {
  def v8i8  : N3VDInt<op24, op23, 0b00, op11_8, op4, itinD16,
                      !strconcat(OpcodeStr, "8"), v8i8, v8i8, IntOp, Commutable>;
  def v16i8 : N3VQInt<op24, op23, 0b00, op11_8, op4, itinQ16,
                      !strconcat(OpcodeStr, "8"), v16i8, v16i8, IntOp, Commutable>;
}

// ....then also with element size of 64 bits:
multiclass N3VInt_QHSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                       InstrItinClass itinD16, InstrItinClass itinD32,
                       InstrItinClass itinQ16, InstrItinClass itinQ32,
                       string OpcodeStr, Intrinsic IntOp, bit Commutable = 0>
  : N3VInt_QHS<op24, op23, op11_8, op4, itinD16, itinD32, itinQ16, itinQ32,
               OpcodeStr, IntOp, Commutable> {
  def v1i64 : N3VDInt<op24, op23, 0b11, op11_8, op4, itinD32,
                      !strconcat(OpcodeStr,"64"), v1i64, v1i64, IntOp, Commutable>;
  def v2i64 : N3VQInt<op24, op23, 0b11, op11_8, op4, itinQ32,
                      !strconcat(OpcodeStr,"64"), v2i64, v2i64, IntOp, Commutable>;
}


// Neon Narrowing 3-register vector intrinsics,
//   source operand element sizes of 16, 32 and 64 bits:
multiclass N3VNInt_HSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                       string OpcodeStr, Intrinsic IntOp, bit Commutable = 0> {
  def v8i8  : N3VNInt<op24, op23, 0b00, op11_8, op4, !strconcat(OpcodeStr,"16"),
                      v8i8, v8i16, IntOp, Commutable>;
  def v4i16 : N3VNInt<op24, op23, 0b01, op11_8, op4, !strconcat(OpcodeStr,"32"),
                      v4i16, v4i32, IntOp, Commutable>;
  def v2i32 : N3VNInt<op24, op23, 0b10, op11_8, op4, !strconcat(OpcodeStr,"64"),
                      v2i32, v2i64, IntOp, Commutable>;
}


// Neon Long 3-register vector intrinsics.

// First with only element sizes of 16 and 32 bits:
multiclass N3VLInt_HS<bit op24, bit op23, bits<4> op11_8, bit op4,
                      InstrItinClass itin, string OpcodeStr,
                      Intrinsic IntOp, bit Commutable = 0> {
  def v4i32 : N3VLInt<op24, op23, 0b01, op11_8, op4, itin, 
                      !strconcat(OpcodeStr,"16"), v4i32, v4i16, IntOp, Commutable>;
  def v2i64 : N3VLInt<op24, op23, 0b10, op11_8, op4, itin,
                      !strconcat(OpcodeStr,"32"), v2i64, v2i32, IntOp, Commutable>;
}

multiclass N3VLIntSL_HS<bit op24, bits<4> op11_8,
                        InstrItinClass itin, string OpcodeStr, Intrinsic IntOp> {
  def v4i16 : N3VLIntSL16<op24, 0b01, op11_8, itin, 
                          !strconcat(OpcodeStr, "16"), v4i32, v4i16, IntOp>;
  def v2i32 : N3VLIntSL<op24, 0b10, op11_8, itin,
                        !strconcat(OpcodeStr, "32"), v2i64, v2i32, IntOp>;
}

// ....then also with element size of 8 bits:
multiclass N3VLInt_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                       InstrItinClass itin, string OpcodeStr,
                       Intrinsic IntOp, bit Commutable = 0>
  : N3VLInt_HS<op24, op23, op11_8, op4, itin, OpcodeStr, IntOp, Commutable> {
  def v8i16 : N3VLInt<op24, op23, 0b00, op11_8, op4, itin, 
                      !strconcat(OpcodeStr, "8"), v8i16, v8i8, IntOp, Commutable>;
}


// Neon Wide 3-register vector intrinsics,
//   source operand element sizes of 8, 16 and 32 bits:
multiclass N3VWInt_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                       string OpcodeStr, Intrinsic IntOp, bit Commutable = 0> {
  def v8i16 : N3VWInt<op24, op23, 0b00, op11_8, op4, !strconcat(OpcodeStr, "8"),
                      v8i16, v8i8, IntOp, Commutable>;
  def v4i32 : N3VWInt<op24, op23, 0b01, op11_8, op4, !strconcat(OpcodeStr,"16"),
                      v4i32, v4i16, IntOp, Commutable>;
  def v2i64 : N3VWInt<op24, op23, 0b10, op11_8, op4, !strconcat(OpcodeStr,"32"),
                      v2i64, v2i32, IntOp, Commutable>;
}


// Neon Multiply-Op vector operations,
//   element sizes of 8, 16 and 32 bits:
multiclass N3VMulOp_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                        InstrItinClass itinD16, InstrItinClass itinD32,
                        InstrItinClass itinQ16, InstrItinClass itinQ32,
                        string OpcodeStr, SDNode OpNode> {
  // 64-bit vector types.
  def v8i8  : N3VDMulOp<op24, op23, 0b00, op11_8, op4, itinD16,
                        !strconcat(OpcodeStr, "8"), v8i8, mul, OpNode>;
  def v4i16 : N3VDMulOp<op24, op23, 0b01, op11_8, op4, itinD16,
                        !strconcat(OpcodeStr, "16"), v4i16, mul, OpNode>;
  def v2i32 : N3VDMulOp<op24, op23, 0b10, op11_8, op4, itinD32,
                        !strconcat(OpcodeStr, "32"), v2i32, mul, OpNode>;

  // 128-bit vector types.
  def v16i8 : N3VQMulOp<op24, op23, 0b00, op11_8, op4, itinQ16,
                        !strconcat(OpcodeStr, "8"), v16i8, mul, OpNode>;
  def v8i16 : N3VQMulOp<op24, op23, 0b01, op11_8, op4, itinQ16,
                        !strconcat(OpcodeStr, "16"), v8i16, mul, OpNode>;
  def v4i32 : N3VQMulOp<op24, op23, 0b10, op11_8, op4, itinQ32,
                        !strconcat(OpcodeStr, "32"), v4i32, mul, OpNode>;
}

multiclass N3VMulOpSL_HS<bits<4> op11_8, 
                         InstrItinClass itinD16, InstrItinClass itinD32,
                         InstrItinClass itinQ16, InstrItinClass itinQ32,
                         string OpcodeStr, SDNode ShOp> {
  def v4i16 : N3VDMulOpSL16<0b01, op11_8, itinD16,
                            !strconcat(OpcodeStr, "16"), v4i16, mul, ShOp>;
  def v2i32 : N3VDMulOpSL<0b10, op11_8, itinD32,
                          !strconcat(OpcodeStr, "32"), v2i32, mul, ShOp>;
  def v8i16 : N3VQMulOpSL16<0b01, op11_8, itinQ16,
                            !strconcat(OpcodeStr, "16"), v8i16, v4i16, mul, ShOp>;
  def v4i32 : N3VQMulOpSL<0b10, op11_8, itinQ32,
                          !strconcat(OpcodeStr, "32"), v4i32, v2i32, mul, ShOp>;
}

// Neon 3-argument intrinsics,
//   element sizes of 8, 16 and 32 bits:
multiclass N3VInt3_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                       string OpcodeStr, Intrinsic IntOp> {
  // 64-bit vector types.
  def v8i8  : N3VDInt3<op24, op23, 0b00, op11_8, op4, IIC_VMACi16D,
                        !strconcat(OpcodeStr, "8"), v8i8, v8i8, IntOp>;
  def v4i16 : N3VDInt3<op24, op23, 0b01, op11_8, op4, IIC_VMACi16D,
                        !strconcat(OpcodeStr, "16"), v4i16, v4i16, IntOp>;
  def v2i32 : N3VDInt3<op24, op23, 0b10, op11_8, op4, IIC_VMACi32D,
                        !strconcat(OpcodeStr, "32"), v2i32, v2i32, IntOp>;

  // 128-bit vector types.
  def v16i8 : N3VQInt3<op24, op23, 0b00, op11_8, op4, IIC_VMACi16Q,
                        !strconcat(OpcodeStr, "8"), v16i8, v16i8, IntOp>;
  def v8i16 : N3VQInt3<op24, op23, 0b01, op11_8, op4, IIC_VMACi16Q,
                        !strconcat(OpcodeStr, "16"), v8i16, v8i16, IntOp>;
  def v4i32 : N3VQInt3<op24, op23, 0b10, op11_8, op4, IIC_VMACi32Q,
                        !strconcat(OpcodeStr, "32"), v4i32, v4i32, IntOp>;
}


// Neon Long 3-argument intrinsics.

// First with only element sizes of 16 and 32 bits:
multiclass N3VLInt3_HS<bit op24, bit op23, bits<4> op11_8, bit op4,
                       string OpcodeStr, Intrinsic IntOp> {
  def v4i32 : N3VLInt3<op24, op23, 0b01, op11_8, op4, IIC_VMACi16D,
                       !strconcat(OpcodeStr, "16"), v4i32, v4i16, IntOp>;
  def v2i64 : N3VLInt3<op24, op23, 0b10, op11_8, op4, IIC_VMACi16D,
                       !strconcat(OpcodeStr, "32"), v2i64, v2i32, IntOp>;
}

multiclass N3VLInt3SL_HS<bit op24, bits<4> op11_8,
                         string OpcodeStr, Intrinsic IntOp> {
  def v4i16 : N3VLInt3SL16<op24, 0b01, op11_8, IIC_VMACi16D,
                           !strconcat(OpcodeStr, "16"), v4i32, v4i16, IntOp>;
  def v2i32 : N3VLInt3SL<op24, 0b10, op11_8, IIC_VMACi32D,
                         !strconcat(OpcodeStr, "32"), v2i64, v2i32, IntOp>;
}

// ....then also with element size of 8 bits:
multiclass N3VLInt3_QHS<bit op24, bit op23, bits<4> op11_8, bit op4,
                        string OpcodeStr, Intrinsic IntOp>
  : N3VLInt3_HS<op24, op23, op11_8, op4, OpcodeStr, IntOp> {
  def v8i16 : N3VLInt3<op24, op23, 0b01, op11_8, op4, IIC_VMACi16D,
                       !strconcat(OpcodeStr, "8"), v8i16, v8i8, IntOp>;
}


// Neon 2-register vector intrinsics,
//   element sizes of 8, 16 and 32 bits:
multiclass N2VInt_QHS<bits<2> op24_23, bits<2> op21_20, bits<2> op17_16,
                      bits<5> op11_7, bit op4,
                      InstrItinClass itinD, InstrItinClass itinQ,
                      string OpcodeStr, Intrinsic IntOp> {
  // 64-bit vector types.
  def v8i8  : N2VDInt<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                      itinD, !strconcat(OpcodeStr, "8"), v8i8, v8i8, IntOp>;
  def v4i16 : N2VDInt<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                      itinD, !strconcat(OpcodeStr, "16"), v4i16, v4i16, IntOp>;
  def v2i32 : N2VDInt<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                      itinD, !strconcat(OpcodeStr, "32"), v2i32, v2i32, IntOp>;

  // 128-bit vector types.
  def v16i8 : N2VQInt<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                      itinQ, !strconcat(OpcodeStr, "8"), v16i8, v16i8, IntOp>;
  def v8i16 : N2VQInt<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                      itinQ, !strconcat(OpcodeStr, "16"), v8i16, v8i16, IntOp>;
  def v4i32 : N2VQInt<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                      itinQ, !strconcat(OpcodeStr, "32"), v4i32, v4i32, IntOp>;
}


// Neon Pairwise long 2-register intrinsics,
//   element sizes of 8, 16 and 32 bits:
multiclass N2VPLInt_QHS<bits<2> op24_23, bits<2> op21_20, bits<2> op17_16,
                        bits<5> op11_7, bit op4,
                        string OpcodeStr, Intrinsic IntOp> {
  // 64-bit vector types.
  def v8i8  : N2VDPLInt<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                        !strconcat(OpcodeStr, "8"), v4i16, v8i8, IntOp>;
  def v4i16 : N2VDPLInt<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                        !strconcat(OpcodeStr, "16"), v2i32, v4i16, IntOp>;
  def v2i32 : N2VDPLInt<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                        !strconcat(OpcodeStr, "32"), v1i64, v2i32, IntOp>;

  // 128-bit vector types.
  def v16i8 : N2VQPLInt<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                        !strconcat(OpcodeStr, "8"), v8i16, v16i8, IntOp>;
  def v8i16 : N2VQPLInt<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                        !strconcat(OpcodeStr, "16"), v4i32, v8i16, IntOp>;
  def v4i32 : N2VQPLInt<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                        !strconcat(OpcodeStr, "32"), v2i64, v4i32, IntOp>;
}


// Neon Pairwise long 2-register accumulate intrinsics,
//   element sizes of 8, 16 and 32 bits:
multiclass N2VPLInt2_QHS<bits<2> op24_23, bits<2> op21_20, bits<2> op17_16,
                         bits<5> op11_7, bit op4,
                         string OpcodeStr, Intrinsic IntOp> {
  // 64-bit vector types.
  def v8i8  : N2VDPLInt2<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                         !strconcat(OpcodeStr, "8"), v4i16, v8i8, IntOp>;
  def v4i16 : N2VDPLInt2<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                         !strconcat(OpcodeStr, "16"), v2i32, v4i16, IntOp>;
  def v2i32 : N2VDPLInt2<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                         !strconcat(OpcodeStr, "32"), v1i64, v2i32, IntOp>;

  // 128-bit vector types.
  def v16i8 : N2VQPLInt2<op24_23, op21_20, 0b00, op17_16, op11_7, op4,
                         !strconcat(OpcodeStr, "8"), v8i16, v16i8, IntOp>;
  def v8i16 : N2VQPLInt2<op24_23, op21_20, 0b01, op17_16, op11_7, op4,
                         !strconcat(OpcodeStr, "16"), v4i32, v8i16, IntOp>;
  def v4i32 : N2VQPLInt2<op24_23, op21_20, 0b10, op17_16, op11_7, op4,
                         !strconcat(OpcodeStr, "32"), v2i64, v4i32, IntOp>;
}


// Neon 2-register vector shift by immediate,
//   element sizes of 8, 16, 32 and 64 bits:
multiclass N2VSh_QHSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                      InstrItinClass itin, string OpcodeStr, SDNode OpNode> {
  // 64-bit vector types.
  def v8i8  : N2VDSh<op24, op23, 0b001000, op11_8, 0, op4, itin,
                     !strconcat(OpcodeStr, "8"), v8i8, OpNode>;
  def v4i16 : N2VDSh<op24, op23, 0b010000, op11_8, 0, op4, itin,
                     !strconcat(OpcodeStr, "16"), v4i16, OpNode>;
  def v2i32 : N2VDSh<op24, op23, 0b100000, op11_8, 0, op4, itin,
                     !strconcat(OpcodeStr, "32"), v2i32, OpNode>;
  def v1i64 : N2VDSh<op24, op23, 0b000000, op11_8, 1, op4, itin,
                     !strconcat(OpcodeStr, "64"), v1i64, OpNode>;

  // 128-bit vector types.
  def v16i8 : N2VQSh<op24, op23, 0b001000, op11_8, 0, op4, itin,
                     !strconcat(OpcodeStr, "8"), v16i8, OpNode>;
  def v8i16 : N2VQSh<op24, op23, 0b010000, op11_8, 0, op4, itin,
                     !strconcat(OpcodeStr, "16"), v8i16, OpNode>;
  def v4i32 : N2VQSh<op24, op23, 0b100000, op11_8, 0, op4, itin,
                     !strconcat(OpcodeStr, "32"), v4i32, OpNode>;
  def v2i64 : N2VQSh<op24, op23, 0b000000, op11_8, 1, op4, itin,
                     !strconcat(OpcodeStr, "64"), v2i64, OpNode>;
}


// Neon Shift-Accumulate vector operations,
//   element sizes of 8, 16, 32 and 64 bits:
multiclass N2VShAdd_QHSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                         string OpcodeStr, SDNode ShOp> {
  // 64-bit vector types.
  def v8i8  : N2VDShAdd<op24, op23, 0b001000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "8"), v8i8, ShOp>;
  def v4i16 : N2VDShAdd<op24, op23, 0b010000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "16"), v4i16, ShOp>;
  def v2i32 : N2VDShAdd<op24, op23, 0b100000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "32"), v2i32, ShOp>;
  def v1i64 : N2VDShAdd<op24, op23, 0b000000, op11_8, 1, op4,
                        !strconcat(OpcodeStr, "64"), v1i64, ShOp>;

  // 128-bit vector types.
  def v16i8 : N2VQShAdd<op24, op23, 0b001000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "8"), v16i8, ShOp>;
  def v8i16 : N2VQShAdd<op24, op23, 0b010000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "16"), v8i16, ShOp>;
  def v4i32 : N2VQShAdd<op24, op23, 0b100000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "32"), v4i32, ShOp>;
  def v2i64 : N2VQShAdd<op24, op23, 0b000000, op11_8, 1, op4,
                        !strconcat(OpcodeStr, "64"), v2i64, ShOp>;
}


// Neon Shift-Insert vector operations,
//   element sizes of 8, 16, 32 and 64 bits:
multiclass N2VShIns_QHSD<bit op24, bit op23, bits<4> op11_8, bit op4,
                         string OpcodeStr, SDNode ShOp> {
  // 64-bit vector types.
  def v8i8  : N2VDShIns<op24, op23, 0b001000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "8"), v8i8, ShOp>;
  def v4i16 : N2VDShIns<op24, op23, 0b010000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "16"), v4i16, ShOp>;
  def v2i32 : N2VDShIns<op24, op23, 0b100000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "32"), v2i32, ShOp>;
  def v1i64 : N2VDShIns<op24, op23, 0b000000, op11_8, 1, op4,
                        !strconcat(OpcodeStr, "64"), v1i64, ShOp>;

  // 128-bit vector types.
  def v16i8 : N2VQShIns<op24, op23, 0b001000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "8"), v16i8, ShOp>;
  def v8i16 : N2VQShIns<op24, op23, 0b010000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "16"), v8i16, ShOp>;
  def v4i32 : N2VQShIns<op24, op23, 0b100000, op11_8, 0, op4,
                        !strconcat(OpcodeStr, "32"), v4i32, ShOp>;
  def v2i64 : N2VQShIns<op24, op23, 0b000000, op11_8, 1, op4,
                        !strconcat(OpcodeStr, "64"), v2i64, ShOp>;
}

//===----------------------------------------------------------------------===//
// Instruction Definitions.
//===----------------------------------------------------------------------===//

// Vector Add Operations.

//   VADD     : Vector Add (integer and floating-point)
defm VADD     : N3V_QHSD<0, 0, 0b1000, 0, IIC_VBINiD, IIC_VBINiQ, "vadd.i", add, 1>;
def  VADDfd   : N3VD<0, 0, 0b00, 0b1101, 0, IIC_VBIND, "vadd.f32", v2f32, v2f32, fadd, 1>;
def  VADDfq   : N3VQ<0, 0, 0b00, 0b1101, 0, IIC_VBINQ, "vadd.f32", v4f32, v4f32, fadd, 1>;
//   VADDL    : Vector Add Long (Q = D + D)
defm VADDLs   : N3VLInt_QHS<0,1,0b0000,0, IIC_VSHLiD, "vaddl.s", int_arm_neon_vaddls, 1>;
defm VADDLu   : N3VLInt_QHS<1,1,0b0000,0, IIC_VSHLiD, "vaddl.u", int_arm_neon_vaddlu, 1>;
//   VADDW    : Vector Add Wide (Q = Q + D)
defm VADDWs   : N3VWInt_QHS<0,1,0b0001,0, "vaddw.s", int_arm_neon_vaddws, 0>;
defm VADDWu   : N3VWInt_QHS<1,1,0b0001,0, "vaddw.u", int_arm_neon_vaddwu, 0>;
//   VHADD    : Vector Halving Add
defm VHADDs   : N3VInt_QHS<0,0,0b0000,0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vhadd.s", int_arm_neon_vhadds, 1>;
defm VHADDu   : N3VInt_QHS<1,0,0b0000,0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vhadd.u", int_arm_neon_vhaddu, 1>;
//   VRHADD   : Vector Rounding Halving Add
defm VRHADDs  : N3VInt_QHS<0,0,0b0001,0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vrhadd.s", int_arm_neon_vrhadds, 1>;
defm VRHADDu  : N3VInt_QHS<1,0,0b0001,0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vrhadd.u", int_arm_neon_vrhaddu, 1>;
//   VQADD    : Vector Saturating Add
defm VQADDs   : N3VInt_QHSD<0,0,0b0000,1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                            IIC_VBINi4Q, "vqadd.s", int_arm_neon_vqadds, 1>;
defm VQADDu   : N3VInt_QHSD<1,0,0b0000,1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                            IIC_VBINi4Q, "vqadd.u", int_arm_neon_vqaddu, 1>;
//   VADDHN   : Vector Add and Narrow Returning High Half (D = Q + Q)
defm VADDHN   : N3VNInt_HSD<0,1,0b0100,0, "vaddhn.i", int_arm_neon_vaddhn, 1>;
//   VRADDHN  : Vector Rounding Add and Narrow Returning High Half (D = Q + Q)
defm VRADDHN  : N3VNInt_HSD<1,1,0b0100,0, "vraddhn.i", int_arm_neon_vraddhn, 1>;

// Vector Multiply Operations.

//   VMUL     : Vector Multiply (integer, polynomial and floating-point)
defm VMUL     : N3V_QHS<0, 0, 0b1001, 1, IIC_VMULi16D, IIC_VMULi32D, IIC_VMULi16Q,
                        IIC_VMULi32Q, "vmul.i", mul, 1>;
def  VMULpd   : N3VDInt<1, 0, 0b00, 0b1001, 1, IIC_VMULi16D, "vmul.p8", v8i8, v8i8,
                        int_arm_neon_vmulp, 1>;
def  VMULpq   : N3VQInt<1, 0, 0b00, 0b1001, 1, IIC_VMULi16Q, "vmul.p8", v16i8, v16i8,
                        int_arm_neon_vmulp, 1>;
def  VMULfd   : N3VD<1, 0, 0b00, 0b1101, 1, IIC_VBIND, "vmul.f32", v2f32, v2f32, fmul, 1>;
def  VMULfq   : N3VQ<1, 0, 0b00, 0b1101, 1, IIC_VBINQ, "vmul.f32", v4f32, v4f32, fmul, 1>;
defm VMULsl  : N3VSL_HS<0b1000, "vmul.i", mul>;
def VMULslfd : N3VDSL<0b10, 0b1001, IIC_VBIND, "vmul.f32", v2f32, fmul>;
def VMULslfq : N3VQSL<0b10, 0b1001, IIC_VBINQ, "vmul.f32", v4f32, v2f32, fmul>;
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
defm VQDMULH  : N3VInt_HS<0, 0, 0b1011, 0, IIC_VMULi16D, IIC_VMULi32D,
                          IIC_VMULi16Q, IIC_VMULi32Q, 
                          "vqdmulh.s", int_arm_neon_vqdmulh, 1>;
defm VQDMULHsl: N3VIntSL_HS<0b1100, IIC_VMULi16D, IIC_VMULi32D,
                            IIC_VMULi16Q, IIC_VMULi32Q,
                            "vqdmulh.s",  int_arm_neon_vqdmulh>;
def : Pat<(v8i16 (int_arm_neon_vqdmulh (v8i16 QPR:$src1),
                                       (v8i16 (NEONvduplane (v8i16 QPR:$src2), imm:$lane)))),
          (v8i16 (VQDMULHslv8i16 (v8i16 QPR:$src1),
                                 (v4i16 (EXTRACT_SUBREG QPR:$src2,
                                                        (DSubReg_i16_reg imm:$lane))),
                                 (SubReg_i16_lane imm:$lane)))>;
def : Pat<(v4i32 (int_arm_neon_vqdmulh (v4i32 QPR:$src1),
                                       (v4i32 (NEONvduplane (v4i32 QPR:$src2), imm:$lane)))),
          (v4i32 (VQDMULHslv4i32 (v4i32 QPR:$src1),
                                 (v2i32 (EXTRACT_SUBREG QPR:$src2,
                                                        (DSubReg_i32_reg imm:$lane))),
                                 (SubReg_i32_lane imm:$lane)))>;

//   VQRDMULH : Vector Rounding Saturating Doubling Multiply Returning High Half
defm VQRDMULH   : N3VInt_HS<1, 0, 0b1011, 0, IIC_VMULi16D, IIC_VMULi32D,
                            IIC_VMULi16Q, IIC_VMULi32Q,
                            "vqrdmulh.s", int_arm_neon_vqrdmulh, 1>;
defm VQRDMULHsl : N3VIntSL_HS<0b1101, IIC_VMULi16D, IIC_VMULi32D,
                              IIC_VMULi16Q, IIC_VMULi32Q,
                              "vqrdmulh.s",  int_arm_neon_vqrdmulh>;
def : Pat<(v8i16 (int_arm_neon_vqrdmulh (v8i16 QPR:$src1),
                                        (v8i16 (NEONvduplane (v8i16 QPR:$src2), imm:$lane)))),
          (v8i16 (VQRDMULHslv8i16 (v8i16 QPR:$src1),
                                  (v4i16 (EXTRACT_SUBREG QPR:$src2,
                                                         (DSubReg_i16_reg imm:$lane))),
                                  (SubReg_i16_lane imm:$lane)))>;
def : Pat<(v4i32 (int_arm_neon_vqrdmulh (v4i32 QPR:$src1),
                                        (v4i32 (NEONvduplane (v4i32 QPR:$src2), imm:$lane)))),
          (v4i32 (VQRDMULHslv4i32 (v4i32 QPR:$src1),
                                  (v2i32 (EXTRACT_SUBREG QPR:$src2,
                                                         (DSubReg_i32_reg imm:$lane))),
                                  (SubReg_i32_lane imm:$lane)))>;

//   VMULL    : Vector Multiply Long (integer and polynomial) (Q = D * D)
defm VMULLs   : N3VLInt_QHS<0,1,0b1100,0, IIC_VMULi16D, "vmull.s", int_arm_neon_vmulls, 1>;
defm VMULLu   : N3VLInt_QHS<1,1,0b1100,0, IIC_VMULi16D, "vmull.u", int_arm_neon_vmullu, 1>;
def  VMULLp   : N3VLInt<0, 1, 0b00, 0b1110, 0, IIC_VMULi16D, "vmull.p8", v8i16, v8i8,
                        int_arm_neon_vmullp, 1>;
defm VMULLsls : N3VLIntSL_HS<0, 0b1010, IIC_VMULi16D, "vmull.s", int_arm_neon_vmulls>;
defm VMULLslu : N3VLIntSL_HS<1, 0b1010, IIC_VMULi16D, "vmull.u", int_arm_neon_vmullu>;

//   VQDMULL  : Vector Saturating Doubling Multiply Long (Q = D * D)
defm VQDMULL  : N3VLInt_HS<0,1,0b1101,0, IIC_VMULi16D, "vqdmull.s", int_arm_neon_vqdmull, 1>;
defm VQDMULLsl: N3VLIntSL_HS<0, 0b1011, IIC_VMULi16D, "vqdmull.s", int_arm_neon_vqdmull>;

// Vector Multiply-Accumulate and Multiply-Subtract Operations.

//   VMLA     : Vector Multiply Accumulate (integer and floating-point)
defm VMLA     : N3VMulOp_QHS<0, 0, 0b1001, 0, IIC_VMACi16D, IIC_VMACi32D,
                             IIC_VMACi16Q, IIC_VMACi32Q, "vmla.i", add>;
def  VMLAfd   : N3VDMulOp<0, 0, 0b00, 0b1101, 1, IIC_VMACD, "vmla.f32", v2f32, fmul, fadd>;
def  VMLAfq   : N3VQMulOp<0, 0, 0b00, 0b1101, 1, IIC_VMACQ, "vmla.f32", v4f32, fmul, fadd>;
defm VMLAsl   : N3VMulOpSL_HS<0b0000, IIC_VMACi16D, IIC_VMACi32D,
                              IIC_VMACi16Q, IIC_VMACi32Q, "vmla.i", add>;
def  VMLAslfd : N3VDMulOpSL<0b10, 0b0001, IIC_VMACD, "vmla.f32", v2f32, fmul, fadd>;
def  VMLAslfq : N3VQMulOpSL<0b10, 0b0001, IIC_VMACQ, "vmla.f32", v4f32, v2f32, fmul, fadd>;

def : Pat<(v8i16 (add (v8i16 QPR:$src1),
                      (mul (v8i16 QPR:$src2),
                           (v8i16 (NEONvduplane (v8i16 QPR:$src3), imm:$lane))))),
          (v8i16 (VMLAslv8i16 (v8i16 QPR:$src1),
                              (v8i16 QPR:$src2),
                              (v4i16 (EXTRACT_SUBREG QPR:$src3,
                                                     (DSubReg_i16_reg imm:$lane))),
                              (SubReg_i16_lane imm:$lane)))>;

def : Pat<(v4i32 (add (v4i32 QPR:$src1),
                      (mul (v4i32 QPR:$src2),
                           (v4i32 (NEONvduplane (v4i32 QPR:$src3), imm:$lane))))),
          (v4i32 (VMLAslv4i32 (v4i32 QPR:$src1),
                              (v4i32 QPR:$src2),
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
defm VMLALs   : N3VLInt3_QHS<0,1,0b1000,0, "vmlal.s", int_arm_neon_vmlals>;
defm VMLALu   : N3VLInt3_QHS<1,1,0b1000,0, "vmlal.u", int_arm_neon_vmlalu>;

defm VMLALsls : N3VLInt3SL_HS<0, 0b0010, "vmlal.s", int_arm_neon_vmlals>;
defm VMLALslu : N3VLInt3SL_HS<1, 0b0010, "vmlal.u", int_arm_neon_vmlalu>;

//   VQDMLAL  : Vector Saturating Doubling Multiply Accumulate Long (Q += D * D)
defm VQDMLAL  : N3VLInt3_HS<0, 1, 0b1001, 0, "vqdmlal.s", int_arm_neon_vqdmlal>;
defm VQDMLALsl: N3VLInt3SL_HS<0, 0b0011, "vqdmlal.s", int_arm_neon_vqdmlal>;

//   VMLS     : Vector Multiply Subtract (integer and floating-point)
defm VMLS     : N3VMulOp_QHS<1, 0, 0b1001, 0, IIC_VMACi16D, IIC_VMACi32D,
                             IIC_VMACi16Q, IIC_VMACi32Q, "vmls.i", sub>;
def  VMLSfd   : N3VDMulOp<0, 0, 0b10, 0b1101, 1, IIC_VMACD, "vmls.f32", v2f32, fmul, fsub>;
def  VMLSfq   : N3VQMulOp<0, 0, 0b10, 0b1101, 1, IIC_VMACQ, "vmls.f32", v4f32, fmul, fsub>;
defm VMLSsl   : N3VMulOpSL_HS<0b0100, IIC_VMACi16D, IIC_VMACi32D,
                              IIC_VMACi16Q, IIC_VMACi32Q, "vmls.i", sub>;
def  VMLSslfd : N3VDMulOpSL<0b10, 0b0101, IIC_VMACD, "vmls.f32", v2f32, fmul, fsub>;
def  VMLSslfq : N3VQMulOpSL<0b10, 0b0101, IIC_VMACQ, "vmls.f32", v4f32, v2f32, fmul, fsub>;

def : Pat<(v8i16 (sub (v8i16 QPR:$src1),
                      (mul (v8i16 QPR:$src2),
                           (v8i16 (NEONvduplane (v8i16 QPR:$src3), imm:$lane))))),
          (v8i16 (VMLSslv8i16 (v8i16 QPR:$src1),
                              (v8i16 QPR:$src2),
                              (v4i16 (EXTRACT_SUBREG QPR:$src3,
                                                     (DSubReg_i16_reg imm:$lane))),
                              (SubReg_i16_lane imm:$lane)))>;

def : Pat<(v4i32 (sub (v4i32 QPR:$src1),
                      (mul (v4i32 QPR:$src2),
                           (v4i32 (NEONvduplane (v4i32 QPR:$src3), imm:$lane))))),
          (v4i32 (VMLSslv4i32 (v4i32 QPR:$src1),
                              (v4i32 QPR:$src2),
                              (v2i32 (EXTRACT_SUBREG QPR:$src3,
                                                     (DSubReg_i32_reg imm:$lane))),
                              (SubReg_i32_lane imm:$lane)))>;

def : Pat<(v4f32 (fsub (v4f32 QPR:$src1),
                       (fmul (v4f32 QPR:$src2),
                             (v4f32 (NEONvduplane (v4f32 QPR:$src3), imm:$lane))))),
          (v4f32 (VMLSslfq (v4f32 QPR:$src1),
                           (v4f32 QPR:$src2),
                           (v2f32 (EXTRACT_SUBREG QPR:$src3,
                                                  (DSubReg_i32_reg imm:$lane))),
                           (SubReg_i32_lane imm:$lane)))>;

//   VMLSL    : Vector Multiply Subtract Long (Q -= D * D)
defm VMLSLs   : N3VLInt3_QHS<0,1,0b1010,0, "vmlsl.s", int_arm_neon_vmlsls>;
defm VMLSLu   : N3VLInt3_QHS<1,1,0b1010,0, "vmlsl.u", int_arm_neon_vmlslu>;

defm VMLSLsls : N3VLInt3SL_HS<0, 0b0110, "vmlsl.s", int_arm_neon_vmlsls>;
defm VMLSLslu : N3VLInt3SL_HS<1, 0b0110, "vmlsl.u", int_arm_neon_vmlslu>;

//   VQDMLSL  : Vector Saturating Doubling Multiply Subtract Long (Q -= D * D)
defm VQDMLSL  : N3VLInt3_HS<0, 1, 0b1011, 0, "vqdmlsl.s", int_arm_neon_vqdmlsl>;
defm VQDMLSLsl: N3VLInt3SL_HS<0, 0b111, "vqdmlsl.s", int_arm_neon_vqdmlsl>;

// Vector Subtract Operations.

//   VSUB     : Vector Subtract (integer and floating-point)
defm VSUB     : N3V_QHSD<1, 0, 0b1000, 0, IIC_VSUBiD, IIC_VSUBiQ, "vsub.i", sub, 0>;
def  VSUBfd   : N3VD<0, 0, 0b10, 0b1101, 0, IIC_VBIND, "vsub.f32", v2f32, v2f32, fsub, 0>;
def  VSUBfq   : N3VQ<0, 0, 0b10, 0b1101, 0, IIC_VBINQ, "vsub.f32", v4f32, v4f32, fsub, 0>;
//   VSUBL    : Vector Subtract Long (Q = D - D)
defm VSUBLs   : N3VLInt_QHS<0,1,0b0010,0, IIC_VSHLiD, "vsubl.s", int_arm_neon_vsubls, 1>;
defm VSUBLu   : N3VLInt_QHS<1,1,0b0010,0, IIC_VSHLiD, "vsubl.u", int_arm_neon_vsublu, 1>;
//   VSUBW    : Vector Subtract Wide (Q = Q - D)
defm VSUBWs   : N3VWInt_QHS<0,1,0b0011,0, "vsubw.s", int_arm_neon_vsubws, 0>;
defm VSUBWu   : N3VWInt_QHS<1,1,0b0011,0, "vsubw.u", int_arm_neon_vsubwu, 0>;
//   VHSUB    : Vector Halving Subtract
defm VHSUBs   : N3VInt_QHS<0, 0, 0b0010, 0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vhsub.s", int_arm_neon_vhsubs, 0>;
defm VHSUBu   : N3VInt_QHS<1, 0, 0b0010, 0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vhsub.u", int_arm_neon_vhsubu, 0>;
//   VQSUB    : Vector Saturing Subtract
defm VQSUBs   : N3VInt_QHSD<0, 0, 0b0010, 1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                            IIC_VBINi4Q, "vqsub.s", int_arm_neon_vqsubs, 0>;
defm VQSUBu   : N3VInt_QHSD<1, 0, 0b0010, 1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                            IIC_VBINi4Q, "vqsub.u", int_arm_neon_vqsubu, 0>;
//   VSUBHN   : Vector Subtract and Narrow Returning High Half (D = Q - Q)
defm VSUBHN   : N3VNInt_HSD<0,1,0b0110,0, "vsubhn.i", int_arm_neon_vsubhn, 0>;
//   VRSUBHN  : Vector Rounding Subtract and Narrow Returning High Half (D=Q-Q)
defm VRSUBHN  : N3VNInt_HSD<1,1,0b0110,0, "vrsubhn.i", int_arm_neon_vrsubhn, 0>;

// Vector Comparisons.

//   VCEQ     : Vector Compare Equal
defm VCEQ     : N3V_QHS<1, 0, 0b1000, 1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                        IIC_VBINi4Q, "vceq.i", NEONvceq, 1>;
def  VCEQfd   : N3VD<0,0,0b00,0b1110,0, IIC_VBIND, "vceq.f32", v2i32, v2f32, NEONvceq, 1>;
def  VCEQfq   : N3VQ<0,0,0b00,0b1110,0, IIC_VBINQ, "vceq.f32", v4i32, v4f32, NEONvceq, 1>;
//   VCGE     : Vector Compare Greater Than or Equal
defm VCGEs    : N3V_QHS<0, 0, 0b0011, 1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                        IIC_VBINi4Q, "vcge.s", NEONvcge, 0>;
defm VCGEu    : N3V_QHS<1, 0, 0b0011, 1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, 
                        IIC_VBINi4Q, "vcge.u", NEONvcgeu, 0>;
def  VCGEfd   : N3VD<1,0,0b00,0b1110,0, IIC_VBIND, "vcge.f32", v2i32, v2f32, NEONvcge, 0>;
def  VCGEfq   : N3VQ<1,0,0b00,0b1110,0, IIC_VBINQ, "vcge.f32", v4i32, v4f32, NEONvcge, 0>;
//   VCGT     : Vector Compare Greater Than
defm VCGTs    : N3V_QHS<0, 0, 0b0011, 0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, 
                        IIC_VBINi4Q, "vcgt.s", NEONvcgt, 0>;
defm VCGTu    : N3V_QHS<1, 0, 0b0011, 0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, 
                        IIC_VBINi4Q, "vcgt.u", NEONvcgtu, 0>;
def  VCGTfd   : N3VD<1,0,0b10,0b1110,0, IIC_VBIND, "vcgt.f32", v2i32, v2f32, NEONvcgt, 0>;
def  VCGTfq   : N3VQ<1,0,0b10,0b1110,0, IIC_VBINQ, "vcgt.f32", v4i32, v4f32, NEONvcgt, 0>;
//   VACGE    : Vector Absolute Compare Greater Than or Equal (aka VCAGE)
def  VACGEd   : N3VDInt<1, 0, 0b00, 0b1110, 1, IIC_VBIND, "vacge.f32", v2i32, v2f32,
                        int_arm_neon_vacged, 0>;
def  VACGEq   : N3VQInt<1, 0, 0b00, 0b1110, 1, IIC_VBINQ, "vacge.f32", v4i32, v4f32,
                        int_arm_neon_vacgeq, 0>;
//   VACGT    : Vector Absolute Compare Greater Than (aka VCAGT)
def  VACGTd   : N3VDInt<1, 0, 0b10, 0b1110, 1, IIC_VBIND, "vacgt.f32", v2i32, v2f32,
                        int_arm_neon_vacgtd, 0>;
def  VACGTq   : N3VQInt<1, 0, 0b10, 0b1110, 1, IIC_VBINQ, "vacgt.f32", v4i32, v4f32,
                        int_arm_neon_vacgtq, 0>;
//   VTST     : Vector Test Bits
defm VTST     : N3V_QHS<0, 0, 0b1000, 1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q, 
                        IIC_VBINi4Q, "vtst.i", NEONvtst, 1>;

// Vector Bitwise Operations.

//   VAND     : Vector Bitwise AND
def  VANDd    : N3VD<0, 0, 0b00, 0b0001, 1, IIC_VBINiD, "vand", v2i32, v2i32, and, 1>;
def  VANDq    : N3VQ<0, 0, 0b00, 0b0001, 1, IIC_VBINiQ, "vand", v4i32, v4i32, and, 1>;

//   VEOR     : Vector Bitwise Exclusive OR
def  VEORd    : N3VD<1, 0, 0b00, 0b0001, 1, IIC_VBINiD, "veor", v2i32, v2i32, xor, 1>;
def  VEORq    : N3VQ<1, 0, 0b00, 0b0001, 1, IIC_VBINiQ, "veor", v4i32, v4i32, xor, 1>;

//   VORR     : Vector Bitwise OR
def  VORRd    : N3VD<0, 0, 0b10, 0b0001, 1, IIC_VBINiD, "vorr", v2i32, v2i32, or, 1>;
def  VORRq    : N3VQ<0, 0, 0b10, 0b0001, 1, IIC_VBINiQ, "vorr", v4i32, v4i32, or, 1>;

//   VBIC     : Vector Bitwise Bit Clear (AND NOT)
def  VBICd    : N3V<0, 0, 0b01, 0b0001, 0, 1, (outs DPR:$dst),
                    (ins DPR:$src1, DPR:$src2), IIC_VBINiD,
                    "vbic\t$dst, $src1, $src2", "",
                    [(set DPR:$dst, (v2i32 (and DPR:$src1,
                                                (vnot_conv DPR:$src2))))]>;
def  VBICq    : N3V<0, 0, 0b01, 0b0001, 1, 1, (outs QPR:$dst),
                    (ins QPR:$src1, QPR:$src2), IIC_VBINiQ,
                    "vbic\t$dst, $src1, $src2", "",
                    [(set QPR:$dst, (v4i32 (and QPR:$src1,
                                                (vnot_conv QPR:$src2))))]>;

//   VORN     : Vector Bitwise OR NOT
def  VORNd    : N3V<0, 0, 0b11, 0b0001, 0, 1, (outs DPR:$dst),
                    (ins DPR:$src1, DPR:$src2), IIC_VBINiD,
                    "vorn\t$dst, $src1, $src2", "",
                    [(set DPR:$dst, (v2i32 (or DPR:$src1,
                                               (vnot_conv DPR:$src2))))]>;
def  VORNq    : N3V<0, 0, 0b11, 0b0001, 1, 1, (outs QPR:$dst),
                    (ins QPR:$src1, QPR:$src2), IIC_VBINiQ,
                    "vorn\t$dst, $src1, $src2", "",
                    [(set QPR:$dst, (v4i32 (or QPR:$src1,
                                               (vnot_conv QPR:$src2))))]>;

//   VMVN     : Vector Bitwise NOT
def  VMVNd    : N2V<0b11, 0b11, 0b00, 0b00, 0b01011, 0, 0,
                    (outs DPR:$dst), (ins DPR:$src), IIC_VSHLiD,
                    "vmvn\t$dst, $src", "",
                    [(set DPR:$dst, (v2i32 (vnot DPR:$src)))]>;
def  VMVNq    : N2V<0b11, 0b11, 0b00, 0b00, 0b01011, 1, 0,
                    (outs QPR:$dst), (ins QPR:$src), IIC_VSHLiD,
                    "vmvn\t$dst, $src", "",
                    [(set QPR:$dst, (v4i32 (vnot QPR:$src)))]>;
def : Pat<(v2i32 (vnot_conv DPR:$src)), (VMVNd DPR:$src)>;
def : Pat<(v4i32 (vnot_conv QPR:$src)), (VMVNq QPR:$src)>;

//   VBSL     : Vector Bitwise Select
def  VBSLd    : N3V<1, 0, 0b01, 0b0001, 0, 1, (outs DPR:$dst),
                    (ins DPR:$src1, DPR:$src2, DPR:$src3), IIC_VCNTiD,
                    "vbsl\t$dst, $src2, $src3", "$src1 = $dst",
                    [(set DPR:$dst,
                      (v2i32 (or (and DPR:$src2, DPR:$src1),
                                 (and DPR:$src3, (vnot_conv DPR:$src1)))))]>;
def  VBSLq    : N3V<1, 0, 0b01, 0b0001, 1, 1, (outs QPR:$dst),
                    (ins QPR:$src1, QPR:$src2, QPR:$src3), IIC_VCNTiQ,
                    "vbsl\t$dst, $src2, $src3", "$src1 = $dst",
                    [(set QPR:$dst,
                      (v4i32 (or (and QPR:$src2, QPR:$src1),
                                 (and QPR:$src3, (vnot_conv QPR:$src1)))))]>;

//   VBIF     : Vector Bitwise Insert if False
//              like VBSL but with: "vbif\t$dst, $src3, $src1", "$src2 = $dst",
//   VBIT     : Vector Bitwise Insert if True
//              like VBSL but with: "vbit\t$dst, $src2, $src1", "$src3 = $dst",
// These are not yet implemented.  The TwoAddress pass will not go looking
// for equivalent operations with different register constraints; it just
// inserts copies.

// Vector Absolute Differences.

//   VABD     : Vector Absolute Difference
defm VABDs    : N3VInt_QHS<0, 0, 0b0111, 0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vabd.s", int_arm_neon_vabds, 0>;
defm VABDu    : N3VInt_QHS<1, 0, 0b0111, 0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vabd.u", int_arm_neon_vabdu, 0>;
def  VABDfd   : N3VDInt<1, 0, 0b10, 0b1101, 0, IIC_VBIND, "vabd.f32", v2f32, v2f32,
                        int_arm_neon_vabds, 0>;
def  VABDfq   : N3VQInt<1, 0, 0b10, 0b1101, 0, IIC_VBINQ, "vabd.f32", v4f32, v4f32,
                        int_arm_neon_vabds, 0>;

//   VABDL    : Vector Absolute Difference Long (Q = | D - D |)
defm VABDLs   : N3VLInt_QHS<0,1,0b0111,0, IIC_VBINi4Q, "vabdl.s", int_arm_neon_vabdls, 0>;
defm VABDLu   : N3VLInt_QHS<1,1,0b0111,0, IIC_VBINi4Q, "vabdl.u", int_arm_neon_vabdlu, 0>;

//   VABA     : Vector Absolute Difference and Accumulate
defm VABAs    : N3VInt3_QHS<0,1,0b0101,0, "vaba.s", int_arm_neon_vabas>;
defm VABAu    : N3VInt3_QHS<1,1,0b0101,0, "vaba.u", int_arm_neon_vabau>;

//   VABAL    : Vector Absolute Difference and Accumulate Long (Q += | D - D |)
defm VABALs   : N3VLInt3_QHS<0,1,0b0101,0, "vabal.s", int_arm_neon_vabals>;
defm VABALu   : N3VLInt3_QHS<1,1,0b0101,0, "vabal.u", int_arm_neon_vabalu>;

// Vector Maximum and Minimum.

//   VMAX     : Vector Maximum
defm VMAXs    : N3VInt_QHS<0, 0, 0b0110, 0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vmax.s", int_arm_neon_vmaxs, 1>;
defm VMAXu    : N3VInt_QHS<1, 0, 0b0110, 0, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vmax.u", int_arm_neon_vmaxu, 1>;
def  VMAXfd   : N3VDInt<0, 0, 0b00, 0b1111, 0, IIC_VBIND, "vmax.f32", v2f32, v2f32,
                        int_arm_neon_vmaxs, 1>;
def  VMAXfq   : N3VQInt<0, 0, 0b00, 0b1111, 0, IIC_VBINQ, "vmax.f32", v4f32, v4f32,
                        int_arm_neon_vmaxs, 1>;

//   VMIN     : Vector Minimum
defm VMINs    : N3VInt_QHS<0, 0, 0b0110, 1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vmin.s", int_arm_neon_vmins, 1>;
defm VMINu    : N3VInt_QHS<1, 0, 0b0110, 1, IIC_VBINi4D, IIC_VBINi4D, IIC_VBINi4Q,
                           IIC_VBINi4Q, "vmin.u", int_arm_neon_vminu, 1>;
def  VMINfd   : N3VDInt<0, 0, 0b10, 0b1111, 0, IIC_VBIND, "vmin.f32", v2f32, v2f32,
                        int_arm_neon_vmins, 1>;
def  VMINfq   : N3VQInt<0, 0, 0b10, 0b1111, 0, IIC_VBINQ, "vmin.f32", v4f32, v4f32,
                        int_arm_neon_vmins, 1>;

// Vector Pairwise Operations.

//   VPADD    : Vector Pairwise Add
def  VPADDi8  : N3VDInt<0, 0, 0b00, 0b1011, 1, IIC_VBINiD, "vpadd.i8", v8i8, v8i8,
                        int_arm_neon_vpadd, 0>;
def  VPADDi16 : N3VDInt<0, 0, 0b01, 0b1011, 1, IIC_VBINiD, "vpadd.i16", v4i16, v4i16,
                        int_arm_neon_vpadd, 0>;
def  VPADDi32 : N3VDInt<0, 0, 0b10, 0b1011, 1, IIC_VBINiD, "vpadd.i32", v2i32, v2i32,
                        int_arm_neon_vpadd, 0>;
def  VPADDf   : N3VDInt<1, 0, 0b00, 0b1101, 0, IIC_VBIND, "vpadd.f32", v2f32, v2f32,
                        int_arm_neon_vpadd, 0>;

//   VPADDL   : Vector Pairwise Add Long
defm VPADDLs  : N2VPLInt_QHS<0b11, 0b11, 0b00, 0b00100, 0, "vpaddl.s",
                             int_arm_neon_vpaddls>;
defm VPADDLu  : N2VPLInt_QHS<0b11, 0b11, 0b00, 0b00101, 0, "vpaddl.u",
                             int_arm_neon_vpaddlu>;

//   VPADAL   : Vector Pairwise Add and Accumulate Long
defm VPADALs  : N2VPLInt2_QHS<0b11, 0b11, 0b00, 0b01100, 0, "vpadal.s",
                              int_arm_neon_vpadals>;
defm VPADALu  : N2VPLInt2_QHS<0b11, 0b11, 0b00, 0b01101, 0, "vpadal.u",
                              int_arm_neon_vpadalu>;

//   VPMAX    : Vector Pairwise Maximum
def  VPMAXs8  : N3VDInt<0, 0, 0b00, 0b1010, 0, IIC_VBINi4D, "vpmax.s8", v8i8, v8i8,
                        int_arm_neon_vpmaxs, 0>;
def  VPMAXs16 : N3VDInt<0, 0, 0b01, 0b1010, 0, IIC_VBINi4D, "vpmax.s16", v4i16, v4i16,
                        int_arm_neon_vpmaxs, 0>;
def  VPMAXs32 : N3VDInt<0, 0, 0b10, 0b1010, 0, IIC_VBINi4D, "vpmax.s32", v2i32, v2i32,
                        int_arm_neon_vpmaxs, 0>;
def  VPMAXu8  : N3VDInt<1, 0, 0b00, 0b1010, 0, IIC_VBINi4D, "vpmax.u8", v8i8, v8i8,
                        int_arm_neon_vpmaxu, 0>;
def  VPMAXu16 : N3VDInt<1, 0, 0b01, 0b1010, 0, IIC_VBINi4D, "vpmax.u16", v4i16, v4i16,
                        int_arm_neon_vpmaxu, 0>;
def  VPMAXu32 : N3VDInt<1, 0, 0b10, 0b1010, 0, IIC_VBINi4D, "vpmax.u32", v2i32, v2i32,
                        int_arm_neon_vpmaxu, 0>;
def  VPMAXf   : N3VDInt<1, 0, 0b00, 0b1111, 0, IIC_VBINi4D, "vpmax.f32", v2f32, v2f32,
                        int_arm_neon_vpmaxs, 0>;

//   VPMIN    : Vector Pairwise Minimum
def  VPMINs8  : N3VDInt<0, 0, 0b00, 0b1010, 1, IIC_VBINi4D, "vpmin.s8", v8i8, v8i8,
                        int_arm_neon_vpmins, 0>;
def  VPMINs16 : N3VDInt<0, 0, 0b01, 0b1010, 1, IIC_VBINi4D, "vpmin.s16", v4i16, v4i16,
                        int_arm_neon_vpmins, 0>;
def  VPMINs32 : N3VDInt<0, 0, 0b10, 0b1010, 1, IIC_VBINi4D, "vpmin.s32", v2i32, v2i32,
                        int_arm_neon_vpmins, 0>;
def  VPMINu8  : N3VDInt<1, 0, 0b00, 0b1010, 1, IIC_VBINi4D, "vpmin.u8", v8i8, v8i8,
                        int_arm_neon_vpminu, 0>;
def  VPMINu16 : N3VDInt<1, 0, 0b01, 0b1010, 1, IIC_VBINi4D, "vpmin.u16", v4i16, v4i16,
                        int_arm_neon_vpminu, 0>;
def  VPMINu32 : N3VDInt<1, 0, 0b10, 0b1010, 1, IIC_VBINi4D, "vpmin.u32", v2i32, v2i32,
                        int_arm_neon_vpminu, 0>;
def  VPMINf   : N3VDInt<1, 0, 0b10, 0b1111, 0, IIC_VBINi4D, "vpmin.f32", v2f32, v2f32,
                        int_arm_neon_vpmins, 0>;

// Vector Reciprocal and Reciprocal Square Root Estimate and Step.

//   VRECPE   : Vector Reciprocal Estimate
def  VRECPEd  : N2VDInt<0b11, 0b11, 0b10, 0b11, 0b01000, 0, 
                        IIC_VUNAD, "vrecpe.u32",
                        v2i32, v2i32, int_arm_neon_vrecpe>;
def  VRECPEq  : N2VQInt<0b11, 0b11, 0b10, 0b11, 0b01000, 0, 
                        IIC_VUNAQ, "vrecpe.u32",
                        v4i32, v4i32, int_arm_neon_vrecpe>;
def  VRECPEfd : N2VDInt<0b11, 0b11, 0b10, 0b11, 0b01010, 0,
                        IIC_VUNAD, "vrecpe.f32",
                        v2f32, v2f32, int_arm_neon_vrecpe>;
def  VRECPEfq : N2VQInt<0b11, 0b11, 0b10, 0b11, 0b01010, 0,
                        IIC_VUNAQ, "vrecpe.f32",
                        v4f32, v4f32, int_arm_neon_vrecpe>;

//   VRECPS   : Vector Reciprocal Step
def  VRECPSfd : N3VDInt<0, 0, 0b00, 0b1111, 1, IIC_VRECSD, "vrecps.f32", v2f32, v2f32,
                        int_arm_neon_vrecps, 1>;
def  VRECPSfq : N3VQInt<0, 0, 0b00, 0b1111, 1, IIC_VRECSQ, "vrecps.f32", v4f32, v4f32,
                        int_arm_neon_vrecps, 1>;

//   VRSQRTE  : Vector Reciprocal Square Root Estimate
def  VRSQRTEd  : N2VDInt<0b11, 0b11, 0b10, 0b11, 0b01001, 0,
                         IIC_VUNAD, "vrsqrte.u32",
                         v2i32, v2i32, int_arm_neon_vrsqrte>;
def  VRSQRTEq  : N2VQInt<0b11, 0b11, 0b10, 0b11, 0b01001, 0,
                         IIC_VUNAQ, "vrsqrte.u32",
                         v4i32, v4i32, int_arm_neon_vrsqrte>;
def  VRSQRTEfd : N2VDInt<0b11, 0b11, 0b10, 0b11, 0b01011, 0,
                         IIC_VUNAD, "vrsqrte.f32",
                         v2f32, v2f32, int_arm_neon_vrsqrte>;
def  VRSQRTEfq : N2VQInt<0b11, 0b11, 0b10, 0b11, 0b01011, 0, 
                         IIC_VUNAQ, "vrsqrte.f32",
                         v4f32, v4f32, int_arm_neon_vrsqrte>;

//   VRSQRTS  : Vector Reciprocal Square Root Step
def VRSQRTSfd : N3VDInt<0, 0, 0b10, 0b1111, 1, IIC_VRECSD, "vrsqrts.f32", v2f32, v2f32,
                        int_arm_neon_vrsqrts, 1>;
def VRSQRTSfq : N3VQInt<0, 0, 0b10, 0b1111, 1, IIC_VRECSQ, "vrsqrts.f32", v4f32, v4f32,
                        int_arm_neon_vrsqrts, 1>;

// Vector Shifts.

//   VSHL     : Vector Shift
defm VSHLs    : N3VInt_QHSD<0, 0, 0b0100, 0, IIC_VSHLiD, IIC_VSHLiD, IIC_VSHLiQ,
                            IIC_VSHLiQ, "vshl.s", int_arm_neon_vshifts, 0>;
defm VSHLu    : N3VInt_QHSD<1, 0, 0b0100, 0, IIC_VSHLiD, IIC_VSHLiD, IIC_VSHLiQ,
                            IIC_VSHLiQ, "vshl.u", int_arm_neon_vshiftu, 0>;
//   VSHL     : Vector Shift Left (Immediate)
defm VSHLi    : N2VSh_QHSD<0, 1, 0b0101, 1, IIC_VSHLiD, "vshl.i", NEONvshl>;
//   VSHR     : Vector Shift Right (Immediate)
defm VSHRs    : N2VSh_QHSD<0, 1, 0b0000, 1, IIC_VSHLiD, "vshr.s", NEONvshrs>;
defm VSHRu    : N2VSh_QHSD<1, 1, 0b0000, 1, IIC_VSHLiD, "vshr.u", NEONvshru>;

//   VSHLL    : Vector Shift Left Long
def  VSHLLs8  : N2VLSh<0, 1, 0b001000, 0b1010, 0, 0, 1, "vshll.s8",
                       v8i16, v8i8, NEONvshlls>;
def  VSHLLs16 : N2VLSh<0, 1, 0b010000, 0b1010, 0, 0, 1, "vshll.s16",
                       v4i32, v4i16, NEONvshlls>;
def  VSHLLs32 : N2VLSh<0, 1, 0b100000, 0b1010, 0, 0, 1, "vshll.s32",
                       v2i64, v2i32, NEONvshlls>;
def  VSHLLu8  : N2VLSh<1, 1, 0b001000, 0b1010, 0, 0, 1, "vshll.u8",
                       v8i16, v8i8, NEONvshllu>;
def  VSHLLu16 : N2VLSh<1, 1, 0b010000, 0b1010, 0, 0, 1, "vshll.u16",
                       v4i32, v4i16, NEONvshllu>;
def  VSHLLu32 : N2VLSh<1, 1, 0b100000, 0b1010, 0, 0, 1, "vshll.u32",
                       v2i64, v2i32, NEONvshllu>;

//   VSHLL    : Vector Shift Left Long (with maximum shift count)
def  VSHLLi8  : N2VLSh<1, 1, 0b110010, 0b0011, 0, 0, 0, "vshll.i8",
                       v8i16, v8i8, NEONvshlli>;
def  VSHLLi16 : N2VLSh<1, 1, 0b110110, 0b0011, 0, 0, 0, "vshll.i16",
                       v4i32, v4i16, NEONvshlli>;
def  VSHLLi32 : N2VLSh<1, 1, 0b111010, 0b0011, 0, 0, 0, "vshll.i32",
                       v2i64, v2i32, NEONvshlli>;

//   VSHRN    : Vector Shift Right and Narrow
def  VSHRN16  : N2VNSh<0, 1, 0b001000, 0b1000, 0, 0, 1, 
                       IIC_VSHLiD, "vshrn.i16", v8i8, v8i16, NEONvshrn>;
def  VSHRN32  : N2VNSh<0, 1, 0b010000, 0b1000, 0, 0, 1,
                       IIC_VSHLiD, "vshrn.i32", v4i16, v4i32, NEONvshrn>;
def  VSHRN64  : N2VNSh<0, 1, 0b100000, 0b1000, 0, 0, 1,
                       IIC_VSHLiD, "vshrn.i64", v2i32, v2i64, NEONvshrn>;

//   VRSHL    : Vector Rounding Shift
defm VRSHLs   : N3VInt_QHSD<0,0,0b0101,0, IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q,
                            IIC_VSHLi4Q, "vrshl.s", int_arm_neon_vrshifts, 0>;
defm VRSHLu   : N3VInt_QHSD<1,0,0b0101,0, IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q,
                            IIC_VSHLi4Q, "vrshl.u", int_arm_neon_vrshiftu, 0>;
//   VRSHR    : Vector Rounding Shift Right
defm VRSHRs   : N2VSh_QHSD<0, 1, 0b0010, 1, IIC_VSHLi4D, "vrshr.s", NEONvrshrs>;
defm VRSHRu   : N2VSh_QHSD<1, 1, 0b0010, 1, IIC_VSHLi4D, "vrshr.u", NEONvrshru>;

//   VRSHRN   : Vector Rounding Shift Right and Narrow
def  VRSHRN16 : N2VNSh<0, 1, 0b001000, 0b1000, 0, 1, 1,
                       IIC_VSHLi4D, "vrshrn.i16", v8i8, v8i16, NEONvrshrn>;
def  VRSHRN32 : N2VNSh<0, 1, 0b010000, 0b1000, 0, 1, 1, 
                       IIC_VSHLi4D, "vrshrn.i32", v4i16, v4i32, NEONvrshrn>;
def  VRSHRN64 : N2VNSh<0, 1, 0b100000, 0b1000, 0, 1, 1,
                       IIC_VSHLi4D, "vrshrn.i64", v2i32, v2i64, NEONvrshrn>;

//   VQSHL    : Vector Saturating Shift
defm VQSHLs   : N3VInt_QHSD<0,0,0b0100,1, IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q,
                            IIC_VSHLi4Q, "vqshl.s", int_arm_neon_vqshifts, 0>;
defm VQSHLu   : N3VInt_QHSD<1,0,0b0100,1, IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q,
                            IIC_VSHLi4Q, "vqshl.u", int_arm_neon_vqshiftu, 0>;
//   VQSHL    : Vector Saturating Shift Left (Immediate)
defm VQSHLsi  : N2VSh_QHSD<0, 1, 0b0111, 1, IIC_VSHLi4D, "vqshl.s", NEONvqshls>;
defm VQSHLui  : N2VSh_QHSD<1, 1, 0b0111, 1, IIC_VSHLi4D, "vqshl.u", NEONvqshlu>;
//   VQSHLU   : Vector Saturating Shift Left (Immediate, Unsigned)
defm VQSHLsu  : N2VSh_QHSD<1, 1, 0b0110, 1, IIC_VSHLi4D, "vqshlu.s", NEONvqshlsu>;

//   VQSHRN   : Vector Saturating Shift Right and Narrow
def VQSHRNs16 : N2VNSh<0, 1, 0b001000, 0b1001, 0, 0, 1, 
                       IIC_VSHLi4D, "vqshrn.s16", v8i8, v8i16, NEONvqshrns>;
def VQSHRNs32 : N2VNSh<0, 1, 0b010000, 0b1001, 0, 0, 1,
                       IIC_VSHLi4D, "vqshrn.s32", v4i16, v4i32, NEONvqshrns>;
def VQSHRNs64 : N2VNSh<0, 1, 0b100000, 0b1001, 0, 0, 1, 
                       IIC_VSHLi4D, "vqshrn.s64", v2i32, v2i64, NEONvqshrns>;
def VQSHRNu16 : N2VNSh<1, 1, 0b001000, 0b1001, 0, 0, 1,
                       IIC_VSHLi4D, "vqshrn.u16", v8i8, v8i16, NEONvqshrnu>;
def VQSHRNu32 : N2VNSh<1, 1, 0b010000, 0b1001, 0, 0, 1,
                       IIC_VSHLi4D, "vqshrn.u32", v4i16, v4i32, NEONvqshrnu>;
def VQSHRNu64 : N2VNSh<1, 1, 0b100000, 0b1001, 0, 0, 1,
                       IIC_VSHLi4D, "vqshrn.u64", v2i32, v2i64, NEONvqshrnu>;

//   VQSHRUN  : Vector Saturating Shift Right and Narrow (Unsigned)
def VQSHRUN16 : N2VNSh<1, 1, 0b001000, 0b1000, 0, 0, 1,
                       IIC_VSHLi4D, "vqshrun.s16", v8i8, v8i16, NEONvqshrnsu>;
def VQSHRUN32 : N2VNSh<1, 1, 0b010000, 0b1000, 0, 0, 1,
                       IIC_VSHLi4D, "vqshrun.s32", v4i16, v4i32, NEONvqshrnsu>;
def VQSHRUN64 : N2VNSh<1, 1, 0b100000, 0b1000, 0, 0, 1,
                       IIC_VSHLi4D, "vqshrun.s64", v2i32, v2i64, NEONvqshrnsu>;

//   VQRSHL   : Vector Saturating Rounding Shift
defm VQRSHLs  : N3VInt_QHSD<0, 0, 0b0101, 1, IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q,
                            IIC_VSHLi4Q, "vqrshl.s", int_arm_neon_vqrshifts, 0>;
defm VQRSHLu  : N3VInt_QHSD<1, 0, 0b0101, 1, IIC_VSHLi4D, IIC_VSHLi4D, IIC_VSHLi4Q,
                            IIC_VSHLi4Q, "vqrshl.u", int_arm_neon_vqrshiftu, 0>;

//   VQRSHRN  : Vector Saturating Rounding Shift Right and Narrow
def VQRSHRNs16: N2VNSh<0, 1, 0b001000, 0b1001, 0, 1, 1,
                       IIC_VSHLi4D, "vqrshrn.s16", v8i8, v8i16, NEONvqrshrns>;
def VQRSHRNs32: N2VNSh<0, 1, 0b010000, 0b1001, 0, 1, 1,
                       IIC_VSHLi4D, "vqrshrn.s32", v4i16, v4i32, NEONvqrshrns>;
def VQRSHRNs64: N2VNSh<0, 1, 0b100000, 0b1001, 0, 1, 1,
                       IIC_VSHLi4D, "vqrshrn.s64", v2i32, v2i64, NEONvqrshrns>;
def VQRSHRNu16: N2VNSh<1, 1, 0b001000, 0b1001, 0, 1, 1,
                       IIC_VSHLi4D, "vqrshrn.u16", v8i8, v8i16, NEONvqrshrnu>;
def VQRSHRNu32: N2VNSh<1, 1, 0b010000, 0b1001, 0, 1, 1,
                       IIC_VSHLi4D, "vqrshrn.u32", v4i16, v4i32, NEONvqrshrnu>;
def VQRSHRNu64: N2VNSh<1, 1, 0b100000, 0b1001, 0, 1, 1, 
                       IIC_VSHLi4D, "vqrshrn.u64", v2i32, v2i64, NEONvqrshrnu>;

//   VQRSHRUN : Vector Saturating Rounding Shift Right and Narrow (Unsigned)
def VQRSHRUN16: N2VNSh<1, 1, 0b001000, 0b1000, 0, 1, 1,
                       IIC_VSHLi4D, "vqrshrun.s16", v8i8, v8i16, NEONvqrshrnsu>;
def VQRSHRUN32: N2VNSh<1, 1, 0b010000, 0b1000, 0, 1, 1, 
                       IIC_VSHLi4D, "vqrshrun.s32", v4i16, v4i32, NEONvqrshrnsu>;
def VQRSHRUN64: N2VNSh<1, 1, 0b100000, 0b1000, 0, 1, 1,
                       IIC_VSHLi4D, "vqrshrun.s64", v2i32, v2i64, NEONvqrshrnsu>;

//   VSRA     : Vector Shift Right and Accumulate
defm VSRAs    : N2VShAdd_QHSD<0, 1, 0b0001, 1, "vsra.s", NEONvshrs>;
defm VSRAu    : N2VShAdd_QHSD<1, 1, 0b0001, 1, "vsra.u", NEONvshru>;
//   VRSRA    : Vector Rounding Shift Right and Accumulate
defm VRSRAs   : N2VShAdd_QHSD<0, 1, 0b0011, 1, "vrsra.s", NEONvrshrs>;
defm VRSRAu   : N2VShAdd_QHSD<1, 1, 0b0011, 1, "vrsra.u", NEONvrshru>;

//   VSLI     : Vector Shift Left and Insert
defm VSLI     : N2VShIns_QHSD<1, 1, 0b0101, 1, "vsli.", NEONvsli>;
//   VSRI     : Vector Shift Right and Insert
defm VSRI     : N2VShIns_QHSD<1, 1, 0b0100, 1, "vsri.", NEONvsri>;

// Vector Absolute and Saturating Absolute.

//   VABS     : Vector Absolute Value
defm VABS     : N2VInt_QHS<0b11, 0b11, 0b01, 0b00110, 0, 
                           IIC_VUNAiD, IIC_VUNAiQ, "vabs.s",
                           int_arm_neon_vabs>;
def  VABSfd   : N2VDInt<0b11, 0b11, 0b10, 0b01, 0b01110, 0,
                        IIC_VUNAD, "vabs.f32",
                        v2f32, v2f32, int_arm_neon_vabs>;
def  VABSfq   : N2VQInt<0b11, 0b11, 0b10, 0b01, 0b01110, 0,
                        IIC_VUNAQ, "vabs.f32",
                        v4f32, v4f32, int_arm_neon_vabs>;

//   VQABS    : Vector Saturating Absolute Value
defm VQABS    : N2VInt_QHS<0b11, 0b11, 0b00, 0b01110, 0, 
                           IIC_VQUNAiD, IIC_VQUNAiQ, "vqabs.s",
                           int_arm_neon_vqabs>;

// Vector Negate.

def vneg      : PatFrag<(ops node:$in), (sub immAllZerosV, node:$in)>;
def vneg_conv : PatFrag<(ops node:$in), (sub immAllZerosV_bc, node:$in)>;

class VNEGD<bits<2> size, string OpcodeStr, ValueType Ty>
  : N2V<0b11, 0b11, size, 0b01, 0b00111, 0, 0, (outs DPR:$dst), (ins DPR:$src),
        IIC_VSHLiD, !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set DPR:$dst, (Ty (vneg DPR:$src)))]>;
class VNEGQ<bits<2> size, string OpcodeStr, ValueType Ty>
  : N2V<0b11, 0b11, size, 0b01, 0b00111, 1, 0, (outs QPR:$dst), (ins QPR:$src),
        IIC_VSHLiD, !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set QPR:$dst, (Ty (vneg QPR:$src)))]>;

//   VNEG     : Vector Negate
def  VNEGs8d  : VNEGD<0b00, "vneg.s8", v8i8>;
def  VNEGs16d : VNEGD<0b01, "vneg.s16", v4i16>;
def  VNEGs32d : VNEGD<0b10, "vneg.s32", v2i32>;
def  VNEGs8q  : VNEGQ<0b00, "vneg.s8", v16i8>;
def  VNEGs16q : VNEGQ<0b01, "vneg.s16", v8i16>;
def  VNEGs32q : VNEGQ<0b10, "vneg.s32", v4i32>;

//   VNEG     : Vector Negate (floating-point)
def  VNEGf32d : N2V<0b11, 0b11, 0b10, 0b01, 0b01111, 0, 0,
                    (outs DPR:$dst), (ins DPR:$src), IIC_VUNAD,
                    "vneg.f32\t$dst, $src", "",
                    [(set DPR:$dst, (v2f32 (fneg DPR:$src)))]>;
def  VNEGf32q : N2V<0b11, 0b11, 0b10, 0b01, 0b01111, 1, 0,
                    (outs QPR:$dst), (ins QPR:$src), IIC_VUNAQ,
                    "vneg.f32\t$dst, $src", "",
                    [(set QPR:$dst, (v4f32 (fneg QPR:$src)))]>;

def : Pat<(v8i8 (vneg_conv DPR:$src)), (VNEGs8d DPR:$src)>;
def : Pat<(v4i16 (vneg_conv DPR:$src)), (VNEGs16d DPR:$src)>;
def : Pat<(v2i32 (vneg_conv DPR:$src)), (VNEGs32d DPR:$src)>;
def : Pat<(v16i8 (vneg_conv QPR:$src)), (VNEGs8q QPR:$src)>;
def : Pat<(v8i16 (vneg_conv QPR:$src)), (VNEGs16q QPR:$src)>;
def : Pat<(v4i32 (vneg_conv QPR:$src)), (VNEGs32q QPR:$src)>;

//   VQNEG    : Vector Saturating Negate
defm VQNEG    : N2VInt_QHS<0b11, 0b11, 0b00, 0b01111, 0, 
                           IIC_VQUNAiD, IIC_VQUNAiQ, "vqneg.s",
                           int_arm_neon_vqneg>;

// Vector Bit Counting Operations.

//   VCLS     : Vector Count Leading Sign Bits
defm VCLS     : N2VInt_QHS<0b11, 0b11, 0b00, 0b01000, 0, 
                           IIC_VCNTiD, IIC_VCNTiQ, "vcls.s",
                           int_arm_neon_vcls>;
//   VCLZ     : Vector Count Leading Zeros
defm VCLZ     : N2VInt_QHS<0b11, 0b11, 0b00, 0b01001, 0, 
                           IIC_VCNTiD, IIC_VCNTiQ, "vclz.i",
                           int_arm_neon_vclz>;
//   VCNT     : Vector Count One Bits
def  VCNTd    : N2VDInt<0b11, 0b11, 0b00, 0b00, 0b01010, 0, 
                        IIC_VCNTiD, "vcnt.8",
                        v8i8, v8i8, int_arm_neon_vcnt>;
def  VCNTq    : N2VQInt<0b11, 0b11, 0b00, 0b00, 0b01010, 0,
                        IIC_VCNTiQ, "vcnt.8",
                        v16i8, v16i8, int_arm_neon_vcnt>;

// Vector Move Operations.

//   VMOV     : Vector Move (Register)

def  VMOVD    : N3V<0, 0, 0b10, 0b0001, 0, 1, (outs DPR:$dst), (ins DPR:$src),
                    IIC_VMOVD, "vmov\t$dst, $src", "", []>;
def  VMOVQ    : N3V<0, 0, 0b10, 0b0001, 1, 1, (outs QPR:$dst), (ins QPR:$src),
                    IIC_VMOVD, "vmov\t$dst, $src", "", []>;

//   VMOV     : Vector Move (Immediate)

// VMOV_get_imm8 xform function: convert build_vector to VMOV.i8 imm.
def VMOV_get_imm8 : SDNodeXForm<build_vector, [{
  return ARM::getVMOVImm(N, 1, *CurDAG);
}]>;
def vmovImm8 : PatLeaf<(build_vector), [{
  return ARM::getVMOVImm(N, 1, *CurDAG).getNode() != 0;
}], VMOV_get_imm8>;

// VMOV_get_imm16 xform function: convert build_vector to VMOV.i16 imm.
def VMOV_get_imm16 : SDNodeXForm<build_vector, [{
  return ARM::getVMOVImm(N, 2, *CurDAG);
}]>;
def vmovImm16 : PatLeaf<(build_vector), [{
  return ARM::getVMOVImm(N, 2, *CurDAG).getNode() != 0;
}], VMOV_get_imm16>;

// VMOV_get_imm32 xform function: convert build_vector to VMOV.i32 imm.
def VMOV_get_imm32 : SDNodeXForm<build_vector, [{
  return ARM::getVMOVImm(N, 4, *CurDAG);
}]>;
def vmovImm32 : PatLeaf<(build_vector), [{
  return ARM::getVMOVImm(N, 4, *CurDAG).getNode() != 0;
}], VMOV_get_imm32>;

// VMOV_get_imm64 xform function: convert build_vector to VMOV.i64 imm.
def VMOV_get_imm64 : SDNodeXForm<build_vector, [{
  return ARM::getVMOVImm(N, 8, *CurDAG);
}]>;
def vmovImm64 : PatLeaf<(build_vector), [{
  return ARM::getVMOVImm(N, 8, *CurDAG).getNode() != 0;
}], VMOV_get_imm64>;

// Note: Some of the cmode bits in the following VMOV instructions need to
// be encoded based on the immed values.

def VMOVv8i8  : N1ModImm<1, 0b000, 0b1110, 0, 0, 0, 1, (outs DPR:$dst),
                         (ins i8imm:$SIMM), IIC_VMOVImm,
                         "vmov.i8\t$dst, $SIMM", "",
                         [(set DPR:$dst, (v8i8 vmovImm8:$SIMM))]>;
def VMOVv16i8 : N1ModImm<1, 0b000, 0b1110, 0, 1, 0, 1, (outs QPR:$dst),
                         (ins i8imm:$SIMM), IIC_VMOVImm,
                         "vmov.i8\t$dst, $SIMM", "",
                         [(set QPR:$dst, (v16i8 vmovImm8:$SIMM))]>;

def VMOVv4i16 : N1ModImm<1, 0b000, 0b1000, 0, 0, 0, 1, (outs DPR:$dst),
                         (ins i16imm:$SIMM), IIC_VMOVImm,
                         "vmov.i16\t$dst, $SIMM", "",
                         [(set DPR:$dst, (v4i16 vmovImm16:$SIMM))]>;
def VMOVv8i16 : N1ModImm<1, 0b000, 0b1000, 0, 1, 0, 1, (outs QPR:$dst),
                         (ins i16imm:$SIMM), IIC_VMOVImm,
                         "vmov.i16\t$dst, $SIMM", "",
                         [(set QPR:$dst, (v8i16 vmovImm16:$SIMM))]>;

def VMOVv2i32 : N1ModImm<1, 0b000, 0b0000, 0, 0, 0, 1, (outs DPR:$dst),
                         (ins i32imm:$SIMM), IIC_VMOVImm,
                         "vmov.i32\t$dst, $SIMM", "",
                         [(set DPR:$dst, (v2i32 vmovImm32:$SIMM))]>;
def VMOVv4i32 : N1ModImm<1, 0b000, 0b0000, 0, 1, 0, 1, (outs QPR:$dst),
                         (ins i32imm:$SIMM), IIC_VMOVImm,
                         "vmov.i32\t$dst, $SIMM", "",
                         [(set QPR:$dst, (v4i32 vmovImm32:$SIMM))]>;

def VMOVv1i64 : N1ModImm<1, 0b000, 0b1110, 0, 0, 1, 1, (outs DPR:$dst),
                         (ins i64imm:$SIMM), IIC_VMOVImm,
                         "vmov.i64\t$dst, $SIMM", "",
                         [(set DPR:$dst, (v1i64 vmovImm64:$SIMM))]>;
def VMOVv2i64 : N1ModImm<1, 0b000, 0b1110, 0, 1, 1, 1, (outs QPR:$dst),
                         (ins i64imm:$SIMM), IIC_VMOVImm,
                         "vmov.i64\t$dst, $SIMM", "",
                         [(set QPR:$dst, (v2i64 vmovImm64:$SIMM))]>;

//   VMOV     : Vector Get Lane (move scalar to ARM core register)

def VGETLNs8  : NVGetLane<0b11100101, 0b1011, 0b00,
                          (outs GPR:$dst), (ins DPR:$src, nohash_imm:$lane),
                          IIC_VMOVSI, "vmov", ".s8\t$dst, $src[$lane]",
                          [(set GPR:$dst, (NEONvgetlanes (v8i8 DPR:$src),
                                           imm:$lane))]>;
def VGETLNs16 : NVGetLane<0b11100001, 0b1011, 0b01,
                          (outs GPR:$dst), (ins DPR:$src, nohash_imm:$lane),
                          IIC_VMOVSI, "vmov", ".s16\t$dst, $src[$lane]",
                          [(set GPR:$dst, (NEONvgetlanes (v4i16 DPR:$src),
                                           imm:$lane))]>;
def VGETLNu8  : NVGetLane<0b11101101, 0b1011, 0b00,
                          (outs GPR:$dst), (ins DPR:$src, nohash_imm:$lane),
                          IIC_VMOVSI, "vmov", ".u8\t$dst, $src[$lane]",
                          [(set GPR:$dst, (NEONvgetlaneu (v8i8 DPR:$src),
                                           imm:$lane))]>;
def VGETLNu16 : NVGetLane<0b11101001, 0b1011, 0b01,
                          (outs GPR:$dst), (ins DPR:$src, nohash_imm:$lane),
                          IIC_VMOVSI, "vmov", ".u16\t$dst, $src[$lane]",
                          [(set GPR:$dst, (NEONvgetlaneu (v4i16 DPR:$src),
                                           imm:$lane))]>;
def VGETLNi32 : NVGetLane<0b11100001, 0b1011, 0b00,
                          (outs GPR:$dst), (ins DPR:$src, nohash_imm:$lane),
                          IIC_VMOVSI, "vmov", ".32\t$dst, $src[$lane]",
                          [(set GPR:$dst, (extractelt (v2i32 DPR:$src),
                                           imm:$lane))]>;
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
          (EXTRACT_SUBREG (COPY_TO_REGCLASS DPR:$src1, DPR_VFP2),
                          (SSubReg_f32_reg imm:$src2))>;
def : Pat<(extractelt (v4f32 QPR:$src1), imm:$src2),
          (EXTRACT_SUBREG (COPY_TO_REGCLASS QPR:$src1, QPR_VFP2),
                          (SSubReg_f32_reg imm:$src2))>;
//def : Pat<(extractelt (v2i64 QPR:$src1), imm:$src2),
//          (EXTRACT_SUBREG QPR:$src1, (DSubReg_f64_reg imm:$src2))>;
def : Pat<(extractelt (v2f64 QPR:$src1), imm:$src2),
          (EXTRACT_SUBREG QPR:$src1, (DSubReg_f64_reg imm:$src2))>;


//   VMOV     : Vector Set Lane (move ARM core register to scalar)

let Constraints = "$src1 = $dst" in {
def VSETLNi8  : NVSetLane<0b11100100, 0b1011, 0b00, (outs DPR:$dst),
                          (ins DPR:$src1, GPR:$src2, nohash_imm:$lane),
                          IIC_VMOVISL, "vmov", ".8\t$dst[$lane], $src2",
                          [(set DPR:$dst, (vector_insert (v8i8 DPR:$src1),
                                           GPR:$src2, imm:$lane))]>;
def VSETLNi16 : NVSetLane<0b11100000, 0b1011, 0b01, (outs DPR:$dst),
                          (ins DPR:$src1, GPR:$src2, nohash_imm:$lane),
                          IIC_VMOVISL, "vmov", ".16\t$dst[$lane], $src2",
                          [(set DPR:$dst, (vector_insert (v4i16 DPR:$src1),
                                           GPR:$src2, imm:$lane))]>;
def VSETLNi32 : NVSetLane<0b11100000, 0b1011, 0b00, (outs DPR:$dst),
                          (ins DPR:$src1, GPR:$src2, nohash_imm:$lane),
                          IIC_VMOVISL, "vmov", ".32\t$dst[$lane], $src2",
                          [(set DPR:$dst, (insertelt (v2i32 DPR:$src1),
                                           GPR:$src2, imm:$lane))]>;
}
def : Pat<(vector_insert (v16i8 QPR:$src1), GPR:$src2, imm:$lane),
          (v16i8 (INSERT_SUBREG QPR:$src1, 
                  (VSETLNi8 (v8i8 (EXTRACT_SUBREG QPR:$src1,
                                   (DSubReg_i8_reg imm:$lane))),
                            GPR:$src2, (SubReg_i8_lane imm:$lane)),
                  (DSubReg_i8_reg imm:$lane)))>;
def : Pat<(vector_insert (v8i16 QPR:$src1), GPR:$src2, imm:$lane),
          (v8i16 (INSERT_SUBREG QPR:$src1, 
                  (VSETLNi16 (v4i16 (EXTRACT_SUBREG QPR:$src1,
                                     (DSubReg_i16_reg imm:$lane))),
                             GPR:$src2, (SubReg_i16_lane imm:$lane)),
                  (DSubReg_i16_reg imm:$lane)))>;
def : Pat<(insertelt (v4i32 QPR:$src1), GPR:$src2, imm:$lane),
          (v4i32 (INSERT_SUBREG QPR:$src1, 
                  (VSETLNi32 (v2i32 (EXTRACT_SUBREG QPR:$src1,
                                     (DSubReg_i32_reg imm:$lane))),
                             GPR:$src2, (SubReg_i32_lane imm:$lane)),
                  (DSubReg_i32_reg imm:$lane)))>;

def : Pat<(v2f32 (insertelt DPR:$src1, SPR:$src2, imm:$src3)),
          (INSERT_SUBREG (COPY_TO_REGCLASS DPR:$src1, DPR_VFP2),
                         SPR:$src2, (SSubReg_f32_reg imm:$src3))>;
def : Pat<(v4f32 (insertelt QPR:$src1, SPR:$src2, imm:$src3)),
          (INSERT_SUBREG (COPY_TO_REGCLASS QPR:$src1, QPR_VFP2),
                         SPR:$src2, (SSubReg_f32_reg imm:$src3))>;

//def : Pat<(v2i64 (insertelt QPR:$src1, DPR:$src2, imm:$src3)),
//          (INSERT_SUBREG QPR:$src1, DPR:$src2, (DSubReg_f64_reg imm:$src3))>;
def : Pat<(v2f64 (insertelt QPR:$src1, DPR:$src2, imm:$src3)),
          (INSERT_SUBREG QPR:$src1, DPR:$src2, (DSubReg_f64_reg imm:$src3))>;

def : Pat<(v2f32 (scalar_to_vector SPR:$src)),
          (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)), SPR:$src, arm_ssubreg_0)>;
def : Pat<(v2f64 (scalar_to_vector DPR:$src)),
          (INSERT_SUBREG (v2f64 (IMPLICIT_DEF)), DPR:$src, arm_dsubreg_0)>;
def : Pat<(v4f32 (scalar_to_vector SPR:$src)),
          (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)), SPR:$src, arm_ssubreg_0)>;

def : Pat<(v8i8 (scalar_to_vector GPR:$src)),
          (VSETLNi8  (v8i8  (IMPLICIT_DEF)), GPR:$src, (i32 0))>;
def : Pat<(v4i16 (scalar_to_vector GPR:$src)),
          (VSETLNi16 (v4i16 (IMPLICIT_DEF)), GPR:$src, (i32 0))>;
def : Pat<(v2i32 (scalar_to_vector GPR:$src)),
          (VSETLNi32 (v2i32 (IMPLICIT_DEF)), GPR:$src, (i32 0))>;

def : Pat<(v16i8 (scalar_to_vector GPR:$src)),
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
                         (VSETLNi8 (v8i8 (IMPLICIT_DEF)), GPR:$src, (i32 0)),
                         arm_dsubreg_0)>;
def : Pat<(v8i16 (scalar_to_vector GPR:$src)),
          (INSERT_SUBREG (v8i16 (IMPLICIT_DEF)),
                         (VSETLNi16 (v4i16 (IMPLICIT_DEF)), GPR:$src, (i32 0)),
                         arm_dsubreg_0)>;
def : Pat<(v4i32 (scalar_to_vector GPR:$src)),
          (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)),
                         (VSETLNi32 (v2i32 (IMPLICIT_DEF)), GPR:$src, (i32 0)),
                         arm_dsubreg_0)>;

//   VDUP     : Vector Duplicate (from ARM core register to all elements)

class VDUPD<bits<8> opcod1, bits<2> opcod3, string asmSize, ValueType Ty>
  : NVDup<opcod1, 0b1011, opcod3, (outs DPR:$dst), (ins GPR:$src),
          IIC_VMOVIS, "vdup", !strconcat(asmSize, "\t$dst, $src"),
          [(set DPR:$dst, (Ty (NEONvdup (i32 GPR:$src))))]>;
class VDUPQ<bits<8> opcod1, bits<2> opcod3, string asmSize, ValueType Ty>
  : NVDup<opcod1, 0b1011, opcod3, (outs QPR:$dst), (ins GPR:$src),
          IIC_VMOVIS, "vdup", !strconcat(asmSize, "\t$dst, $src"),
          [(set QPR:$dst, (Ty (NEONvdup (i32 GPR:$src))))]>;

def  VDUP8d   : VDUPD<0b11101100, 0b00, ".8", v8i8>;
def  VDUP16d  : VDUPD<0b11101000, 0b01, ".16", v4i16>;
def  VDUP32d  : VDUPD<0b11101000, 0b00, ".32", v2i32>;
def  VDUP8q   : VDUPQ<0b11101110, 0b00, ".8", v16i8>;
def  VDUP16q  : VDUPQ<0b11101010, 0b01, ".16", v8i16>;
def  VDUP32q  : VDUPQ<0b11101010, 0b00, ".32", v4i32>;

def  VDUPfd   : NVDup<0b11101000, 0b1011, 0b00, (outs DPR:$dst), (ins GPR:$src),
                      IIC_VMOVIS, "vdup", ".32\t$dst, $src",
                      [(set DPR:$dst, (v2f32 (NEONvdup
                                              (f32 (bitconvert GPR:$src)))))]>;
def  VDUPfq   : NVDup<0b11101010, 0b1011, 0b00, (outs QPR:$dst), (ins GPR:$src),
                      IIC_VMOVIS, "vdup", ".32\t$dst, $src",
                      [(set QPR:$dst, (v4f32 (NEONvdup
                                              (f32 (bitconvert GPR:$src)))))]>;

//   VDUP     : Vector Duplicate Lane (from scalar to all elements)

class VDUPLND<bits<2> op19_18, bits<2> op17_16, string OpcodeStr, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, op17_16, 0b11000, 0, 0,
        (outs DPR:$dst), (ins DPR:$src, nohash_imm:$lane), IIC_VMOVD,
        !strconcat(OpcodeStr, "\t$dst, $src[$lane]"), "",
        [(set DPR:$dst, (Ty (NEONvduplane (Ty DPR:$src), imm:$lane)))]>;

class VDUPLNQ<bits<2> op19_18, bits<2> op17_16, string OpcodeStr,
              ValueType ResTy, ValueType OpTy>
  : N2V<0b11, 0b11, op19_18, op17_16, 0b11000, 1, 0,
        (outs QPR:$dst), (ins DPR:$src, nohash_imm:$lane), IIC_VMOVD,
        !strconcat(OpcodeStr, "\t$dst, $src[$lane]"), "",
        [(set QPR:$dst, (ResTy (NEONvduplane (OpTy DPR:$src), imm:$lane)))]>;

def VDUPLN8d  : VDUPLND<0b00, 0b01, "vdup.8", v8i8>;
def VDUPLN16d : VDUPLND<0b00, 0b10, "vdup.16", v4i16>;
def VDUPLN32d : VDUPLND<0b01, 0b00, "vdup.32", v2i32>;
def VDUPLNfd  : VDUPLND<0b01, 0b00, "vdup.32", v2f32>;
def VDUPLN8q  : VDUPLNQ<0b00, 0b01, "vdup.8", v16i8, v8i8>;
def VDUPLN16q : VDUPLNQ<0b00, 0b10, "vdup.16", v8i16, v4i16>;
def VDUPLN32q : VDUPLNQ<0b01, 0b00, "vdup.32", v4i32, v2i32>;
def VDUPLNfq  : VDUPLNQ<0b01, 0b00, "vdup.32", v4f32, v2f32>;

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

def VDUPfdf   : N2V<0b11, 0b11, 0b01, 0b00, 0b11000, 0, 0,
                    (outs DPR:$dst), (ins SPR:$src),
                    IIC_VMOVD, "vdup.32\t$dst, ${src:lane}", "",
                    [(set DPR:$dst, (v2f32 (NEONvdup (f32 SPR:$src))))]>;

def VDUPfqf   : N2V<0b11, 0b11, 0b01, 0b00, 0b11000, 1, 0,
                    (outs QPR:$dst), (ins SPR:$src),
                    IIC_VMOVD, "vdup.32\t$dst, ${src:lane}", "",
                    [(set QPR:$dst, (v4f32 (NEONvdup (f32 SPR:$src))))]>;

def : Pat<(v2i64 (NEONvduplane (v2i64 QPR:$src), imm:$lane)),
          (INSERT_SUBREG QPR:$src, 
                         (i64 (EXTRACT_SUBREG QPR:$src, (DSubReg_f64_reg imm:$lane))),
                         (DSubReg_f64_other_reg imm:$lane))>;
def : Pat<(v2f64 (NEONvduplane (v2f64 QPR:$src), imm:$lane)),
          (INSERT_SUBREG QPR:$src, 
                         (f64 (EXTRACT_SUBREG QPR:$src, (DSubReg_f64_reg imm:$lane))),
                         (DSubReg_f64_other_reg imm:$lane))>;

//   VMOVN    : Vector Narrowing Move
defm VMOVN    : N2VNInt_HSD<0b11,0b11,0b10,0b00100,0,0, IIC_VMOVD, "vmovn.i",
                            int_arm_neon_vmovn>;
//   VQMOVN   : Vector Saturating Narrowing Move
defm VQMOVNs  : N2VNInt_HSD<0b11,0b11,0b10,0b00101,0,0, IIC_VQUNAiD, "vqmovn.s",
                            int_arm_neon_vqmovns>;
defm VQMOVNu  : N2VNInt_HSD<0b11,0b11,0b10,0b00101,1,0, IIC_VQUNAiD, "vqmovn.u",
                            int_arm_neon_vqmovnu>;
defm VQMOVNsu : N2VNInt_HSD<0b11,0b11,0b10,0b00100,1,0, IIC_VQUNAiD, "vqmovun.s",
                            int_arm_neon_vqmovnsu>;
//   VMOVL    : Vector Lengthening Move
defm VMOVLs   : N2VLInt_QHS<0,1,0b1010,0,0,1, "vmovl.s", int_arm_neon_vmovls>;
defm VMOVLu   : N2VLInt_QHS<1,1,0b1010,0,0,1, "vmovl.u", int_arm_neon_vmovlu>;

// Vector Conversions.

//   VCVT     : Vector Convert Between Floating-Point and Integers
def  VCVTf2sd : N2VD<0b11, 0b11, 0b10, 0b11, 0b01110, 0, "vcvt.s32.f32",
                     v2i32, v2f32, fp_to_sint>;
def  VCVTf2ud : N2VD<0b11, 0b11, 0b10, 0b11, 0b01111, 0, "vcvt.u32.f32",
                     v2i32, v2f32, fp_to_uint>;
def  VCVTs2fd : N2VD<0b11, 0b11, 0b10, 0b11, 0b01100, 0, "vcvt.f32.s32",
                     v2f32, v2i32, sint_to_fp>;
def  VCVTu2fd : N2VD<0b11, 0b11, 0b10, 0b11, 0b01101, 0, "vcvt.f32.u32",
                     v2f32, v2i32, uint_to_fp>;

def  VCVTf2sq : N2VQ<0b11, 0b11, 0b10, 0b11, 0b01110, 0, "vcvt.s32.f32",
                     v4i32, v4f32, fp_to_sint>;
def  VCVTf2uq : N2VQ<0b11, 0b11, 0b10, 0b11, 0b01111, 0, "vcvt.u32.f32",
                     v4i32, v4f32, fp_to_uint>;
def  VCVTs2fq : N2VQ<0b11, 0b11, 0b10, 0b11, 0b01100, 0, "vcvt.f32.s32",
                     v4f32, v4i32, sint_to_fp>;
def  VCVTu2fq : N2VQ<0b11, 0b11, 0b10, 0b11, 0b01101, 0, "vcvt.f32.u32",
                     v4f32, v4i32, uint_to_fp>;

//   VCVT     : Vector Convert Between Floating-Point and Fixed-Point.
// Note: Some of the opcode bits in the following VCVT instructions need to
// be encoded based on the immed values.
def VCVTf2xsd : N2VCvtD<0, 1, 0b000000, 0b1111, 0, 1, "vcvt.s32.f32",
                        v2i32, v2f32, int_arm_neon_vcvtfp2fxs>;
def VCVTf2xud : N2VCvtD<1, 1, 0b000000, 0b1111, 0, 1, "vcvt.u32.f32",
                        v2i32, v2f32, int_arm_neon_vcvtfp2fxu>;
def VCVTxs2fd : N2VCvtD<0, 1, 0b000000, 0b1110, 0, 1, "vcvt.f32.s32",
                        v2f32, v2i32, int_arm_neon_vcvtfxs2fp>;
def VCVTxu2fd : N2VCvtD<1, 1, 0b000000, 0b1110, 0, 1, "vcvt.f32.u32",
                        v2f32, v2i32, int_arm_neon_vcvtfxu2fp>;

def VCVTf2xsq : N2VCvtQ<0, 1, 0b000000, 0b1111, 0, 1, "vcvt.s32.f32",
                        v4i32, v4f32, int_arm_neon_vcvtfp2fxs>;
def VCVTf2xuq : N2VCvtQ<1, 1, 0b000000, 0b1111, 0, 1, "vcvt.u32.f32",
                        v4i32, v4f32, int_arm_neon_vcvtfp2fxu>;
def VCVTxs2fq : N2VCvtQ<0, 1, 0b000000, 0b1110, 0, 1, "vcvt.f32.s32",
                        v4f32, v4i32, int_arm_neon_vcvtfxs2fp>;
def VCVTxu2fq : N2VCvtQ<1, 1, 0b000000, 0b1110, 0, 1, "vcvt.f32.u32",
                        v4f32, v4i32, int_arm_neon_vcvtfxu2fp>;

// Vector Reverse.

//   VREV64   : Vector Reverse elements within 64-bit doublewords

class VREV64D<bits<2> op19_18, string OpcodeStr, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00000, 0, 0, (outs DPR:$dst),
        (ins DPR:$src), IIC_VMOVD, 
        !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set DPR:$dst, (Ty (NEONvrev64 (Ty DPR:$src))))]>;
class VREV64Q<bits<2> op19_18, string OpcodeStr, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00000, 1, 0, (outs QPR:$dst),
        (ins QPR:$src), IIC_VMOVD, 
        !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set QPR:$dst, (Ty (NEONvrev64 (Ty QPR:$src))))]>;

def VREV64d8  : VREV64D<0b00, "vrev64.8", v8i8>;
def VREV64d16 : VREV64D<0b01, "vrev64.16", v4i16>;
def VREV64d32 : VREV64D<0b10, "vrev64.32", v2i32>;
def VREV64df  : VREV64D<0b10, "vrev64.32", v2f32>;

def VREV64q8  : VREV64Q<0b00, "vrev64.8", v16i8>;
def VREV64q16 : VREV64Q<0b01, "vrev64.16", v8i16>;
def VREV64q32 : VREV64Q<0b10, "vrev64.32", v4i32>;
def VREV64qf  : VREV64Q<0b10, "vrev64.32", v4f32>;

//   VREV32   : Vector Reverse elements within 32-bit words

class VREV32D<bits<2> op19_18, string OpcodeStr, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00001, 0, 0, (outs DPR:$dst),
        (ins DPR:$src), IIC_VMOVD, 
        !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set DPR:$dst, (Ty (NEONvrev32 (Ty DPR:$src))))]>;
class VREV32Q<bits<2> op19_18, string OpcodeStr, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00001, 1, 0, (outs QPR:$dst),
        (ins QPR:$src), IIC_VMOVD, 
        !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set QPR:$dst, (Ty (NEONvrev32 (Ty QPR:$src))))]>;

def VREV32d8  : VREV32D<0b00, "vrev32.8", v8i8>;
def VREV32d16 : VREV32D<0b01, "vrev32.16", v4i16>;

def VREV32q8  : VREV32Q<0b00, "vrev32.8", v16i8>;
def VREV32q16 : VREV32Q<0b01, "vrev32.16", v8i16>;

//   VREV16   : Vector Reverse elements within 16-bit halfwords

class VREV16D<bits<2> op19_18, string OpcodeStr, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00010, 0, 0, (outs DPR:$dst),
        (ins DPR:$src), IIC_VMOVD, 
        !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set DPR:$dst, (Ty (NEONvrev16 (Ty DPR:$src))))]>;
class VREV16Q<bits<2> op19_18, string OpcodeStr, ValueType Ty>
  : N2V<0b11, 0b11, op19_18, 0b00, 0b00010, 1, 0, (outs QPR:$dst),
        (ins QPR:$src), IIC_VMOVD, 
        !strconcat(OpcodeStr, "\t$dst, $src"), "",
        [(set QPR:$dst, (Ty (NEONvrev16 (Ty QPR:$src))))]>;

def VREV16d8  : VREV16D<0b00, "vrev16.8", v8i8>;
def VREV16q8  : VREV16Q<0b00, "vrev16.8", v16i8>;

// Other Vector Shuffles.

//   VEXT     : Vector Extract

class VEXTd<string OpcodeStr, ValueType Ty>
  : N3V<0,1,0b11,0b0000,0,0, (outs DPR:$dst),
        (ins DPR:$lhs, DPR:$rhs, i32imm:$index), IIC_VEXTD,
        !strconcat(OpcodeStr, "\t$dst, $lhs, $rhs, $index"), "",
        [(set DPR:$dst, (Ty (NEONvext (Ty DPR:$lhs),
                                      (Ty DPR:$rhs), imm:$index)))]>;

class VEXTq<string OpcodeStr, ValueType Ty>
  : N3V<0,1,0b11,0b0000,1,0, (outs QPR:$dst),
        (ins QPR:$lhs, QPR:$rhs, i32imm:$index), IIC_VEXTQ,
        !strconcat(OpcodeStr, "\t$dst, $lhs, $rhs, $index"), "",
        [(set QPR:$dst, (Ty (NEONvext (Ty QPR:$lhs),
                                      (Ty QPR:$rhs), imm:$index)))]>;

def VEXTd8  : VEXTd<"vext.8",  v8i8>;
def VEXTd16 : VEXTd<"vext.16", v4i16>;
def VEXTd32 : VEXTd<"vext.32", v2i32>;
def VEXTdf  : VEXTd<"vext.32", v2f32>;

def VEXTq8  : VEXTq<"vext.8",  v16i8>;
def VEXTq16 : VEXTq<"vext.16", v8i16>;
def VEXTq32 : VEXTq<"vext.32", v4i32>;
def VEXTqf  : VEXTq<"vext.32", v4f32>;

//   VTRN     : Vector Transpose

def  VTRNd8   : N2VDShuffle<0b00, 0b00001, "vtrn.8">;
def  VTRNd16  : N2VDShuffle<0b01, 0b00001, "vtrn.16">;
def  VTRNd32  : N2VDShuffle<0b10, 0b00001, "vtrn.32">;

def  VTRNq8   : N2VQShuffle<0b00, 0b00001, IIC_VPERMQ, "vtrn.8">;
def  VTRNq16  : N2VQShuffle<0b01, 0b00001, IIC_VPERMQ, "vtrn.16">;
def  VTRNq32  : N2VQShuffle<0b10, 0b00001, IIC_VPERMQ, "vtrn.32">;

//   VUZP     : Vector Unzip (Deinterleave)

def  VUZPd8   : N2VDShuffle<0b00, 0b00010, "vuzp.8">;
def  VUZPd16  : N2VDShuffle<0b01, 0b00010, "vuzp.16">;
def  VUZPd32  : N2VDShuffle<0b10, 0b00010, "vuzp.32">;

def  VUZPq8   : N2VQShuffle<0b00, 0b00010, IIC_VPERMQ3, "vuzp.8">;
def  VUZPq16  : N2VQShuffle<0b01, 0b00010, IIC_VPERMQ3, "vuzp.16">;
def  VUZPq32  : N2VQShuffle<0b10, 0b00010, IIC_VPERMQ3, "vuzp.32">;

//   VZIP     : Vector Zip (Interleave)

def  VZIPd8   : N2VDShuffle<0b00, 0b00011, "vzip.8">;
def  VZIPd16  : N2VDShuffle<0b01, 0b00011, "vzip.16">;
def  VZIPd32  : N2VDShuffle<0b10, 0b00011, "vzip.32">;

def  VZIPq8   : N2VQShuffle<0b00, 0b00011, IIC_VPERMQ3, "vzip.8">;
def  VZIPq16  : N2VQShuffle<0b01, 0b00011, IIC_VPERMQ3, "vzip.16">;
def  VZIPq32  : N2VQShuffle<0b10, 0b00011, IIC_VPERMQ3, "vzip.32">;

// Vector Table Lookup and Table Extension.

//   VTBL     : Vector Table Lookup
def  VTBL1
  : N3V<1,1,0b11,0b1000,0,0, (outs DPR:$dst),
        (ins DPR:$tbl1, DPR:$src), IIC_VTB1,
        "vtbl.8\t$dst, \\{$tbl1\\}, $src", "",
        [(set DPR:$dst, (v8i8 (int_arm_neon_vtbl1 DPR:$tbl1, DPR:$src)))]>;
let hasExtraSrcRegAllocReq = 1 in {
def  VTBL2
  : N3V<1,1,0b11,0b1001,0,0, (outs DPR:$dst),
        (ins DPR:$tbl1, DPR:$tbl2, DPR:$src), IIC_VTB2,
        "vtbl.8\t$dst, \\{$tbl1,$tbl2\\}, $src", "",
        [(set DPR:$dst, (v8i8 (int_arm_neon_vtbl2
                               DPR:$tbl1, DPR:$tbl2, DPR:$src)))]>;
def  VTBL3
  : N3V<1,1,0b11,0b1010,0,0, (outs DPR:$dst),
        (ins DPR:$tbl1, DPR:$tbl2, DPR:$tbl3, DPR:$src), IIC_VTB3,
        "vtbl.8\t$dst, \\{$tbl1,$tbl2,$tbl3\\}, $src", "",
        [(set DPR:$dst, (v8i8 (int_arm_neon_vtbl3
                               DPR:$tbl1, DPR:$tbl2, DPR:$tbl3, DPR:$src)))]>;
def  VTBL4
  : N3V<1,1,0b11,0b1011,0,0, (outs DPR:$dst),
        (ins DPR:$tbl1, DPR:$tbl2, DPR:$tbl3, DPR:$tbl4, DPR:$src), IIC_VTB4,
        "vtbl.8\t$dst, \\{$tbl1,$tbl2,$tbl3,$tbl4\\}, $src", "",
        [(set DPR:$dst, (v8i8 (int_arm_neon_vtbl4 DPR:$tbl1, DPR:$tbl2,
                               DPR:$tbl3, DPR:$tbl4, DPR:$src)))]>;
} // hasExtraSrcRegAllocReq = 1

//   VTBX     : Vector Table Extension
def  VTBX1
  : N3V<1,1,0b11,0b1000,1,0, (outs DPR:$dst),
        (ins DPR:$orig, DPR:$tbl1, DPR:$src), IIC_VTBX1,
        "vtbx.8\t$dst, \\{$tbl1\\}, $src", "$orig = $dst",
        [(set DPR:$dst, (v8i8 (int_arm_neon_vtbx1
                               DPR:$orig, DPR:$tbl1, DPR:$src)))]>;
let hasExtraSrcRegAllocReq = 1 in {
def  VTBX2
  : N3V<1,1,0b11,0b1001,1,0, (outs DPR:$dst),
        (ins DPR:$orig, DPR:$tbl1, DPR:$tbl2, DPR:$src), IIC_VTBX2,
        "vtbx.8\t$dst, \\{$tbl1,$tbl2\\}, $src", "$orig = $dst",
        [(set DPR:$dst, (v8i8 (int_arm_neon_vtbx2
                               DPR:$orig, DPR:$tbl1, DPR:$tbl2, DPR:$src)))]>;
def  VTBX3
  : N3V<1,1,0b11,0b1010,1,0, (outs DPR:$dst),
        (ins DPR:$orig, DPR:$tbl1, DPR:$tbl2, DPR:$tbl3, DPR:$src), IIC_VTBX3,
        "vtbx.8\t$dst, \\{$tbl1,$tbl2,$tbl3\\}, $src", "$orig = $dst",
        [(set DPR:$dst, (v8i8 (int_arm_neon_vtbx3 DPR:$orig, DPR:$tbl1,
                               DPR:$tbl2, DPR:$tbl3, DPR:$src)))]>;
def  VTBX4
  : N3V<1,1,0b11,0b1011,1,0, (outs DPR:$dst), (ins DPR:$orig, DPR:$tbl1,
        DPR:$tbl2, DPR:$tbl3, DPR:$tbl4, DPR:$src), IIC_VTBX4,
        "vtbx.8\t$dst, \\{$tbl1,$tbl2,$tbl3,$tbl4\\}, $src", "$orig = $dst",
        [(set DPR:$dst, (v8i8 (int_arm_neon_vtbx4 DPR:$orig, DPR:$tbl1,
                               DPR:$tbl2, DPR:$tbl3, DPR:$tbl4, DPR:$src)))]>;
} // hasExtraSrcRegAllocReq = 1

//===----------------------------------------------------------------------===//
// NEON instructions for single-precision FP math
//===----------------------------------------------------------------------===//

// These need separate instructions because they must use DPR_VFP2 register
// class which have SPR sub-registers.

// Vector Add Operations used for single-precision FP
let neverHasSideEffects = 1 in
def VADDfd_sfp : N3VDs<0, 0, 0b00, 0b1101, 0, "vadd.f32", v2f32, v2f32, fadd,1>;
def : N3VDsPat<fadd, VADDfd_sfp>;

// Vector Sub Operations used for single-precision FP
let neverHasSideEffects = 1 in
def VSUBfd_sfp : N3VDs<0, 0, 0b10, 0b1101, 0, "vsub.f32", v2f32, v2f32, fsub,0>;
def : N3VDsPat<fsub, VSUBfd_sfp>;

// Vector Multiply Operations used for single-precision FP
let neverHasSideEffects = 1 in
def VMULfd_sfp : N3VDs<1, 0, 0b00, 0b1101, 1, "vmul.f32", v2f32, v2f32, fmul,1>;
def : N3VDsPat<fmul, VMULfd_sfp>;

// Vector Multiply-Accumulate/Subtract used for single-precision FP
let neverHasSideEffects = 1 in
def VMLAfd_sfp : N3VDMulOps<0, 0, 0b00, 0b1101, 1, IIC_VMACD, "vmla.f32", v2f32,fmul,fadd>;
def : N3VDMulOpsPat<fmul, fadd, VMLAfd_sfp>;

let neverHasSideEffects = 1 in
def VMLSfd_sfp : N3VDMulOps<0, 0, 0b10, 0b1101, 1, IIC_VMACD, "vmls.f32", v2f32,fmul,fsub>;
def : N3VDMulOpsPat<fmul, fsub, VMLSfd_sfp>;

// Vector Absolute used for single-precision FP
let neverHasSideEffects = 1 in
def  VABSfd_sfp : N2VDInts<0b11, 0b11, 0b10, 0b01, 0b01110, 0,
                           IIC_VUNAD, "vabs.f32",
                           v2f32, v2f32, int_arm_neon_vabs>;
def : N2VDIntsPat<fabs, VABSfd_sfp>;

// Vector Negate used for single-precision FP
let neverHasSideEffects = 1 in
def  VNEGf32d_sfp : N2V<0b11, 0b11, 0b10, 0b01, 0b01111, 0, 0,
                        (outs DPR_VFP2:$dst), (ins DPR_VFP2:$src), IIC_VUNAD,
                        "vneg.f32\t$dst, $src", "", []>;
def : N2VDIntsPat<fneg, VNEGf32d_sfp>;

// Vector Convert between single-precision FP and integer
let neverHasSideEffects = 1 in
def  VCVTf2sd_sfp : N2VDs<0b11, 0b11, 0b10, 0b11, 0b01110, 0, "vcvt.s32.f32",
                          v2i32, v2f32, fp_to_sint>;
def : N2VDsPat<arm_ftosi, f32, v2f32, VCVTf2sd_sfp>;

let neverHasSideEffects = 1 in
def  VCVTf2ud_sfp : N2VDs<0b11, 0b11, 0b10, 0b11, 0b01111, 0, "vcvt.u32.f32",
                          v2i32, v2f32, fp_to_uint>;
def : N2VDsPat<arm_ftoui, f32, v2f32, VCVTf2ud_sfp>;

let neverHasSideEffects = 1 in
def  VCVTs2fd_sfp : N2VDs<0b11, 0b11, 0b10, 0b11, 0b01100, 0, "vcvt.f32.s32",
                          v2f32, v2i32, sint_to_fp>;
def : N2VDsPat<arm_sitof, f32, v2i32, VCVTs2fd_sfp>;

let neverHasSideEffects = 1 in
def  VCVTu2fd_sfp : N2VDs<0b11, 0b11, 0b10, 0b11, 0b01101, 0, "vcvt.f32.u32",
                          v2f32, v2i32, uint_to_fp>;
def : N2VDsPat<arm_uitof, f32, v2i32, VCVTu2fd_sfp>;

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
