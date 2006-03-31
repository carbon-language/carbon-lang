//====- X86InstrSSE.td - Describe the X86 Instruction Set -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 SSE instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SSE specific DAG Nodes.
//===----------------------------------------------------------------------===//

def X86loadp   : SDNode<"X86ISD::LOAD_PACK", SDTLoad, 
                        [SDNPHasChain]>;
def X86fand    : SDNode<"X86ISD::FAND",     SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86fxor    : SDNode<"X86ISD::FXOR",     SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86s2vec   : SDNode<"X86ISD::S2VEC",
                        SDTypeProfile<1, 1, []>, []>;
def X86zexts2vec : SDNode<"X86ISD::ZEXT_S2VEC",
                          SDTypeProfile<1, 1, []>, []>;

def X86pextrw  : SDNode<"X86ISD::PEXTRW",
                        SDTypeProfile<1, 2, []>, []>;

//===----------------------------------------------------------------------===//
// SSE pattern fragments
//===----------------------------------------------------------------------===//

def X86loadpf32  : PatFrag<(ops node:$ptr), (f32   (X86loadp node:$ptr))>;
def X86loadpf64  : PatFrag<(ops node:$ptr), (f64   (X86loadp node:$ptr))>;

def loadv4f32    : PatFrag<(ops node:$ptr), (v4f32 (load node:$ptr))>;
def loadv2f64    : PatFrag<(ops node:$ptr), (v2f64 (load node:$ptr))>;
def loadv16i8    : PatFrag<(ops node:$ptr), (v16i8 (load node:$ptr))>;
def loadv8i16    : PatFrag<(ops node:$ptr), (v8i16 (load node:$ptr))>;
def loadv4i32    : PatFrag<(ops node:$ptr), (v4i32 (load node:$ptr))>;
def loadv2i64    : PatFrag<(ops node:$ptr), (v2i64 (load node:$ptr))>;

def bc_v4f32 : PatFrag<(ops node:$in), (v4f32 (bitconvert node:$in))>;
def bc_v2f64 : PatFrag<(ops node:$in), (v2f64 (bitconvert node:$in))>;
def bc_v16i8 : PatFrag<(ops node:$in), (v16i8 (bitconvert node:$in))>;
def bc_v8i16 : PatFrag<(ops node:$in), (v8i16 (bitconvert node:$in))>;
def bc_v4i32 : PatFrag<(ops node:$in), (v4i32 (bitconvert node:$in))>;
def bc_v2i64 : PatFrag<(ops node:$in), (v2i64 (bitconvert node:$in))>;

def fp32imm0 : PatLeaf<(f32 fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

// SHUFFLE_get_shuf_imm xform function: convert vector_shuffle mask to PSHUF*,
// SHUFP* etc. imm.
def SHUFFLE_get_shuf_imm : SDNodeXForm<build_vector, [{
  return getI8Imm(X86::getShuffleSHUFImmediate(N));
}]>;

// SHUFFLE_get_pshufhw_imm xform function: convert vector_shuffle mask to 
// PSHUFHW imm.
def SHUFFLE_get_pshufhw_imm : SDNodeXForm<build_vector, [{
  return getI8Imm(X86::getShufflePSHUFHWImmediate(N));
}]>;

// SHUFFLE_get_pshuflw_imm xform function: convert vector_shuffle mask to 
// PSHUFLW imm.
def SHUFFLE_get_pshuflw_imm : SDNodeXForm<build_vector, [{
  return getI8Imm(X86::getShufflePSHUFLWImmediate(N));
}]>;

def SSE_splat_mask : PatLeaf<(build_vector), [{
  return X86::isSplatMask(N);
}], SHUFFLE_get_shuf_imm>;

def MOVLHPS_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVLHPSMask(N);
}]>;

def MOVHLPS_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVHLPSMask(N);
}]>;

def UNPCKL_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isUNPCKLMask(N);
}]>;

def UNPCKH_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isUNPCKHMask(N);
}]>;

def PSHUFD_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isPSHUFDMask(N);
}], SHUFFLE_get_shuf_imm>;

def PSHUFHW_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isPSHUFHWMask(N);
}], SHUFFLE_get_pshufhw_imm>;

def PSHUFLW_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isPSHUFLWMask(N);
}], SHUFFLE_get_pshuflw_imm>;

// Only use PSHUF* for v4f32 if SHUFP does not match.
def PSHUFD_fp_shuffle_mask : PatLeaf<(build_vector), [{
  return !X86::isSHUFPMask(N) &&
          X86::isPSHUFDMask(N);
}], SHUFFLE_get_shuf_imm>;

def PSHUFHW_fp_shuffle_mask : PatLeaf<(build_vector), [{
  return !X86::isSHUFPMask(N) &&
          X86::isPSHUFHWMask(N);
}], SHUFFLE_get_pshufhw_imm>;

def PSHUFLW_fp_shuffle_mask : PatLeaf<(build_vector), [{
  return !X86::isSHUFPMask(N) &&
          X86::isPSHUFLWMask(N);
}], SHUFFLE_get_pshuflw_imm>;

def SHUFP_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isSHUFPMask(N);
}], SHUFFLE_get_shuf_imm>;

// Only use SHUFP for v4i32 if PSHUF* do not match.
def SHUFP_int_shuffle_mask : PatLeaf<(build_vector), [{
  return !X86::isPSHUFDMask(N) &&
         !X86::isPSHUFHWMask(N) &&
         !X86::isPSHUFLWMask(N) &&
          X86::isSHUFPMask(N);
}], SHUFFLE_get_shuf_imm>;

//===----------------------------------------------------------------------===//
// SSE scalar FP Instructions
//===----------------------------------------------------------------------===//

// Instruction templates
// SSI - SSE1 instructions with XS prefix.
// SDI - SSE2 instructions with XD prefix.
// PSI - SSE1 instructions with TB prefix.
// PDI - SSE2 instructions with TB and OpSize prefixes.
// PSIi8 - SSE1 instructions with ImmT == Imm8 and TB prefix.
// PDIi8 - SSE2 instructions with ImmT == Imm8 and TB and OpSize prefixes.
// S3SI - SSE3 instructions with XD prefix.
// S3DI - SSE3 instructions with TB and OpSize prefixes.
class SSI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, XS, Requires<[HasSSE1]>;
class SDI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, XD, Requires<[HasSSE2]>;
class PSI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, TB, Requires<[HasSSE1]>;
class PDI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, TB, OpSize, Requires<[HasSSE2]>;
class PSIi8<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : X86Inst<o, F, Imm8, ops, asm>, TB, Requires<[HasSSE1]> {
  let Pattern = pattern;
}
class PDIi8<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : X86Inst<o, F, Imm8, ops, asm>, TB, OpSize, Requires<[HasSSE2]> {
  let Pattern = pattern;
}
class S3SI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, XD, Requires<[HasSSE3]>;
class S3DI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, TB, OpSize, Requires<[HasSSE3]>;

//===----------------------------------------------------------------------===//
// Helpers for defining instructions that directly correspond to intrinsics.
class S3S_Intrr<bits<8> o, string asm, Intrinsic IntId>
  : S3SI<o, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2), asm,
         [(set VR128:$dst, (v4f32 (IntId VR128:$src1, VR128:$src2)))]>;
class S3S_Intrm<bits<8> o, string asm, Intrinsic IntId>
  : S3SI<o, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2), asm,
         [(set VR128:$dst, (v4f32 (IntId VR128:$src1,
                                   (loadv4f32 addr:$src2))))]>;
class S3D_Intrr<bits<8> o, string asm, Intrinsic IntId>
  : S3DI<o, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2), asm,
         [(set VR128:$dst, (v2f64 (IntId VR128:$src1, VR128:$src2)))]>;
class S3D_Intrm<bits<8> o, string asm, Intrinsic IntId>
  : S3DI<o, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2), asm,
         [(set VR128:$dst, (v2f64 (IntId VR128:$src1,
                                   (loadv2f64 addr:$src2))))]>;

// Some 'special' instructions
def IMPLICIT_DEF_FR32 : I<0, Pseudo, (ops FR32:$dst),
                         "#IMPLICIT_DEF $dst",
                         [(set FR32:$dst, (undef))]>, Requires<[HasSSE2]>;
def IMPLICIT_DEF_FR64 : I<0, Pseudo, (ops FR64:$dst),
                         "#IMPLICIT_DEF $dst",
                         [(set FR64:$dst, (undef))]>, Requires<[HasSSE2]>;

// CMOV* - Used to implement the SSE SELECT DAG operation.  Expanded by the
// scheduler into a branch sequence.
let usesCustomDAGSchedInserter = 1 in {  // Expanded by the scheduler.
  def CMOV_FR32 : I<0, Pseudo,
                    (ops FR32:$dst, FR32:$t, FR32:$f, i8imm:$cond),
                    "#CMOV_FR32 PSEUDO!",
                    [(set FR32:$dst, (X86cmov FR32:$t, FR32:$f, imm:$cond))]>;
  def CMOV_FR64 : I<0, Pseudo,
                    (ops FR64:$dst, FR64:$t, FR64:$f, i8imm:$cond),
                    "#CMOV_FR64 PSEUDO!",
                    [(set FR64:$dst, (X86cmov FR64:$t, FR64:$f, imm:$cond))]>;
}

// Move Instructions
def MOVSSrr : SSI<0x10, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                "movss {$src, $dst|$dst, $src}", []>;
def MOVSSrm : SSI<0x10, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                "movss {$src, $dst|$dst, $src}",
                [(set FR32:$dst, (loadf32 addr:$src))]>;
def MOVSDrr : SDI<0x10, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                "movsd {$src, $dst|$dst, $src}", []>;
def MOVSDrm : SDI<0x10, MRMSrcMem, (ops FR64:$dst, f64mem:$src),
                "movsd {$src, $dst|$dst, $src}",
                [(set FR64:$dst, (loadf64 addr:$src))]>;

def MOVSSmr : SSI<0x11, MRMDestMem, (ops f32mem:$dst, FR32:$src),
                "movss {$src, $dst|$dst, $src}",
                [(store FR32:$src, addr:$dst)]>;
def MOVSDmr : SDI<0x11, MRMDestMem, (ops f64mem:$dst, FR64:$src),
                "movsd {$src, $dst|$dst, $src}",
                [(store FR64:$src, addr:$dst)]>;

// FR32 / FR64 to 128-bit vector conversion.
def MOVSS128rr : SSI<0x10, MRMSrcReg, (ops VR128:$dst, FR32:$src),
                      "movss {$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4f32 (scalar_to_vector FR32:$src)))]>;
def MOVSS128rm : SSI<0x10, MRMSrcMem, (ops VR128:$dst, f32mem:$src),
                     "movss {$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (v4f32 (scalar_to_vector (loadf32 addr:$src))))]>;
def MOVSD128rr : SDI<0x10, MRMSrcReg, (ops VR128:$dst, FR64:$src),
                      "movsd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v2f64 (scalar_to_vector FR64:$src)))]>;
def MOVSD128rm : SDI<0x10, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
                     "movsd {$src, $dst|$dst, $src}",
                     [(set VR128:$dst, 
                       (v2f64 (scalar_to_vector (loadf64 addr:$src))))]>;

// Arithmetic instructions
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def ADDSSrr : SSI<0x58, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "addss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fadd FR32:$src1, FR32:$src2))]>;
def ADDSDrr : SDI<0x58, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "addsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fadd FR64:$src1, FR64:$src2))]>;
def MULSSrr : SSI<0x59, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "mulss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fmul FR32:$src1, FR32:$src2))]>;
def MULSDrr : SDI<0x59, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "mulsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fmul FR64:$src1, FR64:$src2))]>;
}

def ADDSSrm : SSI<0x58, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "addss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fadd FR32:$src1, (loadf32 addr:$src2)))]>;
def ADDSDrm : SDI<0x58, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "addsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fadd FR64:$src1, (loadf64 addr:$src2)))]>;
def MULSSrm : SSI<0x59, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "mulss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fmul FR32:$src1, (loadf32 addr:$src2)))]>;
def MULSDrm : SDI<0x59, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "mulsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fmul FR64:$src1, (loadf64 addr:$src2)))]>;

def DIVSSrr : SSI<0x5E, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "divss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fdiv FR32:$src1, FR32:$src2))]>;
def DIVSSrm : SSI<0x5E, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "divss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fdiv FR32:$src1, (loadf32 addr:$src2)))]>;
def DIVSDrr : SDI<0x5E, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "divsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fdiv FR64:$src1, FR64:$src2))]>;
def DIVSDrm : SDI<0x5E, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "divsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fdiv FR64:$src1, (loadf64 addr:$src2)))]>;

def SUBSSrr : SSI<0x5C, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "subss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fsub FR32:$src1, FR32:$src2))]>;
def SUBSSrm : SSI<0x5C, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "subss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fsub FR32:$src1, (loadf32 addr:$src2)))]>;
def SUBSDrr : SDI<0x5C, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "subsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fsub FR64:$src1, FR64:$src2))]>;
def SUBSDrm : SDI<0x5C, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "subsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fsub FR64:$src1, (loadf64 addr:$src2)))]>;
}

def SQRTSSrr : SSI<0x51, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                 "sqrtss {$src, $dst|$dst, $src}",
                 [(set FR32:$dst, (fsqrt FR32:$src))]>;
def SQRTSSrm : SSI<0x51, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                 "sqrtss {$src, $dst|$dst, $src}",
                 [(set FR32:$dst, (fsqrt (loadf32 addr:$src)))]>;
def SQRTSDrr : SDI<0x51, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                 "sqrtsd {$src, $dst|$dst, $src}",
                 [(set FR64:$dst, (fsqrt FR64:$src))]>;
def SQRTSDrm : SDI<0x51, MRMSrcMem, (ops FR64:$dst, f64mem:$src),
                 "sqrtsd {$src, $dst|$dst, $src}",
                 [(set FR64:$dst, (fsqrt (loadf64 addr:$src)))]>;

def RSQRTSSrr : SSI<0x52, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                   "rsqrtss {$src, $dst|$dst, $src}", []>;
def RSQRTSSrm : SSI<0x52, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                   "rsqrtss {$src, $dst|$dst, $src}", []>;
def RCPSSrr : SSI<0x53, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                  "rcpss {$src, $dst|$dst, $src}", []>;
def RCPSSrm : SSI<0x53, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                  "rcpss {$src, $dst|$dst, $src}", []>;

def MAXSSrr : SSI<0x5F, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                  "maxss {$src, $dst|$dst, $src}", []>;
def MAXSSrm : SSI<0x5F, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                  "maxss {$src, $dst|$dst, $src}", []>;
def MAXSDrr : SDI<0x5F, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                  "maxsd {$src, $dst|$dst, $src}", []>;
def MAXSDrm : SDI<0x5F, MRMSrcMem, (ops FR64:$dst, f64mem:$src),
                  "maxsd {$src, $dst|$dst, $src}", []>;
def MINSSrr : SSI<0x5D, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                  "minss {$src, $dst|$dst, $src}", []>;
def MINSSrm : SSI<0x5D, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                  "minss {$src, $dst|$dst, $src}", []>;
def MINSDrr : SDI<0x5D, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                  "minsd {$src, $dst|$dst, $src}", []>;
def MINSDrm : SDI<0x5D, MRMSrcMem, (ops FR64:$dst, f64mem:$src),
                  "minsd {$src, $dst|$dst, $src}", []>;


// Aliases to match intrinsics which expect XMM operand(s).
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def Int_ADDSSrr : SSI<0x58, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                        VR128:$src2),
                "addss {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse_add_ss VR128:$src1,
                                   VR128:$src2))]>;
def Int_ADDSDrr : SDI<0x58, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                        VR128:$src2),
                "addsd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse2_add_sd VR128:$src1,
                                   VR128:$src2))]>;
def Int_MULSSrr : SSI<0x59, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                        VR128:$src2),
                "mulss {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse_mul_ss VR128:$src1,
                                   VR128:$src2))]>;
def Int_MULSDrr : SDI<0x59, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                        VR128:$src2),
                "mulsd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse2_mul_sd VR128:$src1,
                                   VR128:$src2))]>;
}

def Int_ADDSSrm : SSI<0x58, MRMSrcMem, (ops VR128:$dst, VR128:$src1,
                                        f32mem:$src2),
                "addss {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse_add_ss VR128:$src1,
                                   (load addr:$src2)))]>;
def Int_ADDSDrm : SDI<0x58, MRMSrcMem, (ops VR128:$dst, VR128:$src1,
                                        f64mem:$src2),
                "addsd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse2_add_sd VR128:$src1,
                                   (load addr:$src2)))]>;
def Int_MULSSrm : SSI<0x59, MRMSrcMem, (ops VR128:$dst, VR128:$src1,
                                        f32mem:$src2),
                "mulss {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse_mul_ss VR128:$src1,
                                   (load addr:$src2)))]>;
def Int_MULSDrm : SDI<0x59, MRMSrcMem, (ops VR128:$dst, VR128:$src1,
                                        f64mem:$src2),
                "mulsd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse2_mul_sd VR128:$src1,
                                   (load addr:$src2)))]>;

def Int_DIVSSrr : SSI<0x5E, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                "divss {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse_div_ss VR128:$src1,
                                   VR128:$src2))]>;
def Int_DIVSSrm : SSI<0x5E, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f32mem:$src2),
                "divss {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse_div_ss VR128:$src1,
                                   (load addr:$src2)))]>;
def Int_DIVSDrr : SDI<0x5E, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                "divsd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse2_div_sd VR128:$src1,
                                   VR128:$src2))]>;
def Int_DIVSDrm : SDI<0x5E, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
                "divsd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse2_div_sd VR128:$src1,
                                   (load addr:$src2)))]>;

def Int_SUBSSrr : SSI<0x5C, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                "subss {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse_sub_ss VR128:$src1,
                                   VR128:$src2))]>;
def Int_SUBSSrm : SSI<0x5C, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f32mem:$src2),
                "subss {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse_sub_ss VR128:$src1,
                                   (load addr:$src2)))]>;
def Int_SUBSDrr : SDI<0x5C, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                "subsd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse2_sub_sd VR128:$src1,
                                   VR128:$src2))]>;
def Int_SUBSDrm : SDI<0x5C, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
                "subsd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst, (int_x86_sse2_sub_sd VR128:$src1,
                                   (load addr:$src2)))]>;
}

def Int_SQRTSSrr : SSI<0x51, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                 "sqrtss {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse_sqrt_ss VR128:$src))]>;
def Int_SQRTSSrm : SSI<0x51, MRMSrcMem, (ops VR128:$dst, f32mem:$src),
                 "sqrtss {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse_sqrt_ss
                                          (load addr:$src)))]>;
def Int_SQRTSDrr : SDI<0x51, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                      "sqrtsd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse2_sqrt_sd VR128:$src))]>;
def Int_SQRTSDrm : SDI<0x51, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
                      "sqrtsd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse2_sqrt_sd
                                         (load addr:$src)))]>;

def Int_RSQRTSSrr : SSI<0x52, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                       "rsqrtss {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse_rsqrt_ss VR128:$src))]>;
def Int_RSQRTSSrm : SSI<0x52, MRMSrcMem, (ops VR128:$dst, f32mem:$src),
                       "rsqrtss {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse_rsqrt_ss
                                          (load addr:$src)))]>;
def Int_RCPSSrr : SSI<0x53, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                     "rcpss {$src, $dst|$dst, $src}",
                     [(set VR128:$dst, (int_x86_sse_rcp_ss VR128:$src))]>;
def Int_RCPSSrm : SSI<0x53, MRMSrcMem, (ops VR128:$dst, f32mem:$src),
                     "rcpss {$src, $dst|$dst, $src}",
                     [(set VR128:$dst, (int_x86_sse_rcp_ss
                                        (load addr:$src)))]>;

let isTwoAddress = 1 in {
def Int_MAXSSrr : SSI<0x5F, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                        VR128:$src2),
                      "maxss {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse_max_ss VR128:$src1,
                                         VR128:$src2))]>;
def Int_MAXSSrm : SSI<0x5F, MRMSrcMem, (ops VR128:$dst, VR128:$src1,
                                        f32mem:$src2),
                      "maxss {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse_max_ss VR128:$src1,
                                         (load addr:$src2)))]>;
def Int_MAXSDrr : SDI<0x5F, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                        VR128:$src2),
                      "maxsd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse2_max_sd VR128:$src1,
                                         VR128:$src2))]>;
def Int_MAXSDrm : SDI<0x5F, MRMSrcMem, (ops VR128:$dst, VR128:$src1,
                                        f64mem:$src2),
                      "maxsd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse2_max_sd VR128:$src1,
                                         (load addr:$src2)))]>;
def Int_MINSSrr : SSI<0x5D, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                        VR128:$src2),
                      "minss {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse_min_ss VR128:$src1,
                                         VR128:$src2))]>;
def Int_MINSSrm : SSI<0x5D, MRMSrcMem, (ops VR128:$dst, VR128:$src1,
                                        f32mem:$src2),
                      "minss {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse_min_ss VR128:$src1,
                                         (load addr:$src2)))]>;
def Int_MINSDrr : SDI<0x5D, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                        VR128:$src2),
                      "minsd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse2_min_sd VR128:$src1,
                                         VR128:$src2))]>;
def Int_MINSDrm : SDI<0x5D, MRMSrcMem, (ops VR128:$dst, VR128:$src1,
                                        f64mem:$src2),
                      "minsd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse2_min_sd VR128:$src1,
                                         (load addr:$src2)))]>;
}

// Conversion instructions
def CVTSS2SIrr: SSI<0x2D, MRMSrcReg, (ops R32:$dst, FR32:$src),
                   "cvtss2si {$src, $dst|$dst, $src}", []>;
def CVTSS2SIrm: SSI<0x2D, MRMSrcMem, (ops R32:$dst, f32mem:$src),
                   "cvtss2si {$src, $dst|$dst, $src}", []>;

def CVTTSS2SIrr: SSI<0x2C, MRMSrcReg, (ops R32:$dst, FR32:$src),
                   "cvttss2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint FR32:$src))]>;
def CVTTSS2SIrm: SSI<0x2C, MRMSrcMem, (ops R32:$dst, f32mem:$src),
                   "cvttss2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint (loadf32 addr:$src)))]>;
def CVTTSD2SIrr: SDI<0x2C, MRMSrcReg, (ops R32:$dst, FR64:$src),
                   "cvttsd2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint FR64:$src))]>;
def CVTTSD2SIrm: SDI<0x2C, MRMSrcMem, (ops R32:$dst, f64mem:$src),
                   "cvttsd2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint (loadf64 addr:$src)))]>;
def CVTSD2SSrr: SDI<0x5A, MRMSrcReg, (ops FR32:$dst, FR64:$src),
                  "cvtsd2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (fround FR64:$src))]>;
def CVTSD2SSrm: SDI<0x5A, MRMSrcMem, (ops FR32:$dst, f64mem:$src), 
                  "cvtsd2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (fround (loadf64 addr:$src)))]>;
def CVTSI2SSrr: SSI<0x2A, MRMSrcReg, (ops FR32:$dst, R32:$src),
                  "cvtsi2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (sint_to_fp R32:$src))]>;
def CVTSI2SSrm: SSI<0x2A, MRMSrcMem, (ops FR32:$dst, i32mem:$src),
                  "cvtsi2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (sint_to_fp (loadi32 addr:$src)))]>;
def CVTSI2SDrr: SDI<0x2A, MRMSrcReg, (ops FR64:$dst, R32:$src),
                  "cvtsi2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (sint_to_fp R32:$src))]>;
def CVTSI2SDrm: SDI<0x2A, MRMSrcMem, (ops FR64:$dst, i32mem:$src),
                  "cvtsi2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (sint_to_fp (loadi32 addr:$src)))]>;
// SSE2 instructions with XS prefix
def CVTSS2SDrr: I<0x5A, MRMSrcReg, (ops FR64:$dst, FR32:$src),
                  "cvtss2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (fextend FR32:$src))]>, XS,
                Requires<[HasSSE2]>;
def CVTSS2SDrm: I<0x5A, MRMSrcMem, (ops FR64:$dst, f32mem:$src),
                  "cvtss2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (fextend (loadf32 addr:$src)))]>, XS,
                Requires<[HasSSE2]>;

// Comparison instructions
let isTwoAddress = 1 in {
def CMPSSrr : SSI<0xC2, MRMSrcReg, 
                (ops FR32:$dst, FR32:$src1, FR32:$src, SSECC:$cc),
                "cmp${cc}ss {$src, $dst|$dst, $src}",
                 []>;
def CMPSSrm : SSI<0xC2, MRMSrcMem, 
                (ops FR32:$dst, FR32:$src1, f32mem:$src, SSECC:$cc),
                "cmp${cc}ss {$src, $dst|$dst, $src}", []>;
def CMPSDrr : SDI<0xC2, MRMSrcReg, 
                (ops FR64:$dst, FR64:$src1, FR64:$src, SSECC:$cc),
                "cmp${cc}sd {$src, $dst|$dst, $src}", []>;
def CMPSDrm : SDI<0xC2, MRMSrcMem, 
                (ops FR64:$dst, FR64:$src1, f64mem:$src, SSECC:$cc),
                "cmp${cc}sd {$src, $dst|$dst, $src}", []>;
}

def UCOMISSrr: PSI<0x2E, MRMSrcReg, (ops FR32:$src1, FR32:$src2),
                 "ucomiss {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR32:$src1, FR32:$src2)]>;
def UCOMISSrm: PSI<0x2E, MRMSrcMem, (ops FR32:$src1, f32mem:$src2),
                 "ucomiss {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR32:$src1, (loadf32 addr:$src2))]>;
def UCOMISDrr: PDI<0x2E, MRMSrcReg, (ops FR64:$src1, FR64:$src2),
                 "ucomisd {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR64:$src1, FR64:$src2)]>;
def UCOMISDrm: PDI<0x2E, MRMSrcMem, (ops FR64:$src1, f64mem:$src2),
                 "ucomisd {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR64:$src1, (loadf64 addr:$src2))]>;

// Aliases to match intrinsics which expect XMM operand(s).
let isTwoAddress = 1 in {
def Int_CMPSSrr : SSI<0xC2, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src, SSECC:$cc),
                      "cmp${cc}ss {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse_cmp_ss VR128:$src1,
                                         VR128:$src, imm:$cc))]>;
def Int_CMPSSrm : SSI<0xC2, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, f32mem:$src, SSECC:$cc),
                      "cmp${cc}ss {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse_cmp_ss VR128:$src1,
                                         (load addr:$src), imm:$cc))]>;
def Int_CMPSDrr : SDI<0xC2, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src, SSECC:$cc),
                      "cmp${cc}sd {$src, $dst|$dst, $src}", []>;
def Int_CMPSDrm : SDI<0xC2, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, f64mem:$src, SSECC:$cc),
                      "cmp${cc}sd {$src, $dst|$dst, $src}", []>;
}


// Aliases of packed instructions for scalar use. These all have names that
// start with 'Fs'.

// Alias instructions that map fld0 to pxor for sse.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
def FsFLD0SS : I<0xEF, MRMInitReg, (ops FR32:$dst),
                 "pxor $dst, $dst", [(set FR32:$dst, fp32imm0)]>,
               Requires<[HasSSE1]>, TB, OpSize;
def FsFLD0SD : I<0xEF, MRMInitReg, (ops FR64:$dst),
                 "pxor $dst, $dst", [(set FR64:$dst, fp64imm0)]>,
               Requires<[HasSSE2]>, TB, OpSize;

// Alias instructions to do FR32 / FR64 reg-to-reg copy using movaps / movapd.
// Upper bits are disregarded.
def FsMOVAPSrr : PSI<0x28, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                   "movaps {$src, $dst|$dst, $src}", []>;
def FsMOVAPDrr : PDI<0x28, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                   "movapd {$src, $dst|$dst, $src}", []>;

// Alias instructions to load FR32 / FR64 from f128mem using movaps / movapd.
// Upper bits are disregarded.
def FsMOVAPSrm : PSI<0x28, MRMSrcMem, (ops FR32:$dst, f128mem:$src),
                   "movaps {$src, $dst|$dst, $src}",
                   [(set FR32:$dst, (X86loadpf32 addr:$src))]>;
def FsMOVAPDrm : PDI<0x28, MRMSrcMem, (ops FR64:$dst, f128mem:$src),
                  "movapd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (X86loadpf64 addr:$src))]>;

// Alias bitwise logical operations using SSE logical ops on packed FP values.
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def FsANDPSrr : PSI<0x54, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fand FR32:$src1, FR32:$src2))]>;
def FsANDPDrr : PDI<0x54, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                  "andpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fand FR64:$src1, FR64:$src2))]>;
def FsORPSrr  : PSI<0x56, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                  "orps {$src2, $dst|$dst, $src2}", []>;
def FsORPDrr  : PDI<0x56, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                  "orpd {$src2, $dst|$dst, $src2}", []>;
def FsXORPSrr : PSI<0x57, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fxor FR32:$src1, FR32:$src2))]>;
def FsXORPDrr : PDI<0x57, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fxor FR64:$src1, FR64:$src2))]>;
}
def FsANDPSrm : PSI<0x54, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fand FR32:$src1,
                                    (X86loadpf32 addr:$src2)))]>;
def FsANDPDrm : PDI<0x54, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                  "andpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fand FR64:$src1,
                                    (X86loadpf64 addr:$src2)))]>;
def FsORPSrm  : PSI<0x56, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                  "orps {$src2, $dst|$dst, $src2}", []>;
def FsORPDrm  : PDI<0x56, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                  "orpd {$src2, $dst|$dst, $src2}", []>;
def FsXORPSrm : PSI<0x57, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fxor FR32:$src1,
                                    (X86loadpf32 addr:$src2)))]>;
def FsXORPDrm : PDI<0x57, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fxor FR64:$src1,
                                    (X86loadpf64 addr:$src2)))]>;

def FsANDNPSrr : PSI<0x55, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                   "andnps {$src2, $dst|$dst, $src2}", []>;
def FsANDNPSrm : PSI<0x55, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                   "andnps {$src2, $dst|$dst, $src2}", []>;
def FsANDNPDrr : PDI<0x55, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                   "andnpd {$src2, $dst|$dst, $src2}", []>;
def FsANDNPDrm : PDI<0x55, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                   "andnpd {$src2, $dst|$dst, $src2}", []>;
}

//===----------------------------------------------------------------------===//
// SSE packed FP Instructions
//===----------------------------------------------------------------------===//

// Some 'special' instructions
def IMPLICIT_DEF_VR128 : I<0, Pseudo, (ops VR128:$dst),
                           "#IMPLICIT_DEF $dst",
                           [(set VR128:$dst, (v4f32 (undef)))]>,
                         Requires<[HasSSE1]>;

// Move Instructions
def MOVAPSrr : PSI<0x28, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "movaps {$src, $dst|$dst, $src}", []>;
def MOVAPSrm : PSI<0x28, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                   "movaps {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (loadv4f32 addr:$src))]>;
def MOVAPDrr : PDI<0x28, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "movapd {$src, $dst|$dst, $src}", []>;
def MOVAPDrm : PDI<0x28, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                   "movapd {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (loadv2f64 addr:$src))]>;

def MOVAPSmr : PSI<0x29, MRMDestMem, (ops f128mem:$dst, VR128:$src),
                   "movaps {$src, $dst|$dst, $src}",
                   [(store (v4f32 VR128:$src), addr:$dst)]>;
def MOVAPDmr : PDI<0x29, MRMDestMem, (ops f128mem:$dst, VR128:$src),
                   "movapd {$src, $dst|$dst, $src}",
                   [(store (v2f64 VR128:$src), addr:$dst)]>;

def MOVUPSrr : PSI<0x10, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "movups {$src, $dst|$dst, $src}", []>;
def MOVUPSrm : PSI<0x10, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                   "movups {$src, $dst|$dst, $src}", []>;
def MOVUPSmr : PSI<0x11, MRMDestMem, (ops f128mem:$dst, VR128:$src),
                   "movups {$src, $dst|$dst, $src}", []>;
def MOVUPDrr : PDI<0x10, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "movupd {$src, $dst|$dst, $src}", []>;
def MOVUPDrm : PDI<0x10, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                   "movupd {$src, $dst|$dst, $src}", []>;
def MOVUPDmr : PDI<0x11, MRMDestMem, (ops f128mem:$dst, VR128:$src),
                   "movupd {$src, $dst|$dst, $src}", []>;

let isTwoAddress = 1 in {
def MOVLPSrm : PSI<0x12, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
                   "movlps {$src2, $dst|$dst, $src2}", []>;
def MOVLPDrm : PDI<0x12, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
                   "movlpd {$src2, $dst|$dst, $src2}", []>;
def MOVHPSrm : PSI<0x16, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
                   "movhps {$src2, $dst|$dst, $src2}", []>;
def MOVHPDrm : PDI<0x16, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
                   "movhpd {$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst, 
                     (v2f64 (vector_shuffle VR128:$src1,
                             (scalar_to_vector (loadf64 addr:$src2)),
                             UNPCKL_shuffle_mask)))]>;
}

def MOVLPSmr : PSI<0x13, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movlps {$src, $dst|$dst, $src}", []>;
def MOVLPDmr : PDI<0x13, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movlpd {$src, $dst|$dst, $src}", []>;

def MOVHPSmr : PSI<0x17, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movhps {$src, $dst|$dst, $src}", []>;
def MOVHPDmr : PDI<0x17, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movhpd {$src, $dst|$dst, $src}", []>;

let isTwoAddress = 1 in {
def MOVLHPSrr : PSI<0x16, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "movlhps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (v4f32 (vector_shuffle VR128:$src1, VR128:$src2,
                              MOVLHPS_shuffle_mask)))]>;

def MOVHLPSrr : PSI<0x12, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "movhlps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (v4f32 (vector_shuffle VR128:$src1, VR128:$src2,
                              MOVHLPS_shuffle_mask)))]>;
}

// Conversion instructions
def CVTPI2PSrr : PSI<0x2A, MRMSrcReg, (ops VR128:$dst, VR64:$src),
                     "cvtpi2ps {$src, $dst|$dst, $src}", []>;
def CVTPI2PSrm : PSI<0x2A, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                     "cvtpi2ps {$src, $dst|$dst, $src}", []>;
def CVTPI2PDrr : PDI<0x2A, MRMSrcReg, (ops VR128:$dst, VR64:$src),
                     "cvtpi2pd {$src, $dst|$dst, $src}", []>;
def CVTPI2PDrm : PDI<0x2A, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                     "cvtpi2pd {$src, $dst|$dst, $src}", []>;

// SSE2 instructions without OpSize prefix
def CVTDQ2PSrr : I<0x5B, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "cvtdq2ps {$src, $dst|$dst, $src}", []>, TB,
                 Requires<[HasSSE2]>;
def CVTDQ2PSrm : I<0x5B, MRMSrcMem, (ops VR128:$dst, i128mem:$src),
                   "cvtdq2ps {$src, $dst|$dst, $src}", []>, TB,
                 Requires<[HasSSE2]>;

// SSE2 instructions with XS prefix
def CVTDQ2PDrr : I<0xE6, MRMSrcReg, (ops VR128:$dst, VR64:$src),
                   "cvtdq2pd {$src, $dst|$dst, $src}", []>,
                 XS, Requires<[HasSSE2]>;
def CVTDQ2PDrm : I<0xE6, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                   "cvtdq2pd {$src, $dst|$dst, $src}", []>,
                 XS, Requires<[HasSSE2]>;

def CVTPS2PIrr : PSI<0x2D, MRMSrcReg, (ops VR64:$dst, VR128:$src),
                    "cvtps2pi {$src, $dst|$dst, $src}", []>;
def CVTPS2PIrm : PSI<0x2D, MRMSrcMem, (ops VR64:$dst, f64mem:$src),
                    "cvtps2pi {$src, $dst|$dst, $src}", []>;
def CVTPD2PIrr : PDI<0x2D, MRMSrcReg, (ops VR64:$dst, VR128:$src),
                    "cvtpd2pi {$src, $dst|$dst, $src}", []>;
def CVTPD2PIrm : PDI<0x2D, MRMSrcMem, (ops VR64:$dst, f128mem:$src),
                    "cvtpd2pi {$src, $dst|$dst, $src}", []>;

def CVTPS2DQrr : PDI<0x5B, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                     "cvtps2dq {$src, $dst|$dst, $src}", []>;
def CVTPS2DQrm : PDI<0x5B, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                     "cvtps2dq {$src, $dst|$dst, $src}", []>;
// SSE2 packed instructions with XD prefix
def CVTPD2DQrr : SDI<0xE6, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                     "cvtpd2dq {$src, $dst|$dst, $src}", []>;
def CVTPD2DQrm : SDI<0xE6, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                     "cvtpd2dq {$src, $dst|$dst, $src}", []>;

// SSE2 instructions without OpSize prefix
def CVTPS2PDrr : I<0x5A, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "cvtps2pd {$src, $dst|$dst, $src}", []>, TB,
                 Requires<[HasSSE2]>;
def CVTPS2PDrm : I<0x5A, MRMSrcReg, (ops VR128:$dst, f64mem:$src),
                   "cvtps2pd {$src, $dst|$dst, $src}", []>, TB,
                 Requires<[HasSSE2]>;

def CVTPD2PSrr : PDI<0x5A, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                     "cvtpd2ps {$src, $dst|$dst, $src}", []>;
def CVTPD2PSrm : PDI<0x5A, MRMSrcReg, (ops VR128:$dst, f128mem:$src),
                     "cvtpd2ps {$src, $dst|$dst, $src}", []>;

// Arithmetic
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def ADDPSrr : PSI<0x58, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "addps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4f32 (fadd VR128:$src1, VR128:$src2)))]>;
def ADDPDrr : PDI<0x58, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "addpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2f64 (fadd VR128:$src1, VR128:$src2)))]>;
def MULPSrr : PSI<0x59, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "mulps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4f32 (fmul VR128:$src1, VR128:$src2)))]>;
def MULPDrr : PDI<0x59, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "mulpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2f64 (fmul VR128:$src1, VR128:$src2)))]>;
}

def ADDPSrm : PSI<0x58, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "addps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4f32 (fadd VR128:$src1,
                                            (load addr:$src2))))]>;
def ADDPDrm : PDI<0x58, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "addpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2f64 (fadd VR128:$src1,
                                            (load addr:$src2))))]>;
def MULPSrm : PSI<0x59, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "mulps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4f32 (fmul VR128:$src1,
                                            (load addr:$src2))))]>;
def MULPDrm : PDI<0x59, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "mulpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2f64 (fmul VR128:$src1,
                                            (load addr:$src2))))]>;

def DIVPSrr : PSI<0x5E, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "divps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4f32 (fdiv VR128:$src1, VR128:$src2)))]>;
def DIVPSrm : PSI<0x5E, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "divps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4f32 (fdiv VR128:$src1,
                                            (load addr:$src2))))]>;
def DIVPDrr : PDI<0x5E, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                "divpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2f64 (fdiv VR128:$src1, VR128:$src2)))]>;
def DIVPDrm : PDI<0x5E, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                "divpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2f64 (fdiv VR128:$src1,
                                            (load addr:$src2))))]>;

def SUBPSrr : PSI<0x5C, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "subps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4f32 (fsub VR128:$src1, VR128:$src2)))]>;
def SUBPSrm : PSI<0x5C, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "subps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4f32 (fsub VR128:$src1,
                                            (load addr:$src2))))]>;
def SUBPDrr : PDI<0x5C, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "subpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2f64 (fsub VR128:$src1, VR128:$src2)))]>;
def SUBPDrm : PDI<0x5C, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "subpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2f64 (fsub VR128:$src1,
                                            (load addr:$src2))))]>;
}

def SQRTPSrr : PSI<0x51, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "sqrtps {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (v4f32 (fsqrt VR128:$src)))]>;
def SQRTPSrm : PSI<0x51, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                   "sqrtps {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (v4f32 (fsqrt (load addr:$src))))]>;
def SQRTPDrr : PDI<0x51, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "sqrtpd {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (v2f64 (fsqrt VR128:$src)))]>;
def SQRTPDrm : PDI<0x51, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                   "sqrtpd {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (v2f64 (fsqrt (load addr:$src))))]>;

def RSQRTPSrr : PSI<0x52, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                    "rsqrtps {$src, $dst|$dst, $src}", []>;
def RSQRTPSrm : PSI<0x52, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                    "rsqrtps {$src, $dst|$dst, $src}", []>;
def RCPPSrr : PSI<0x53, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                  "rcpps {$src, $dst|$dst, $src}", []>;
def RCPPSrm : PSI<0x53, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                  "rcpps {$src, $dst|$dst, $src}", []>;

def MAXPSrr : PSI<0x5F, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                  "maxps {$src, $dst|$dst, $src}", []>;
def MAXPSrm : PSI<0x5F, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                  "maxps {$src, $dst|$dst, $src}", []>;
def MAXPDrr : PDI<0x5F, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                  "maxpd {$src, $dst|$dst, $src}", []>;
def MAXPDrm : PDI<0x5F, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                  "maxpd {$src, $dst|$dst, $src}", []>;
def MINPSrr : PSI<0x5D, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                  "minps {$src, $dst|$dst, $src}", []>;
def MINPSrm : PSI<0x5D, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                  "minps {$src, $dst|$dst, $src}", []>;
def MINPDrr : PDI<0x5D, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                  "minpd {$src, $dst|$dst, $src}", []>;
def MINPDrm : PDI<0x5D, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                  "minpd {$src, $dst|$dst, $src}", []>;

// Logical
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def ANDPSrr : PSI<0x54, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (and (bc_v4i32 (v4f32 VR128:$src1)),
                     (bc_v4i32 (v4f32 VR128:$src2))))]>;
def ANDPDrr : PDI<0x54, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                "andpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (and (bc_v2i64 (v2f64 VR128:$src1)),
                     (bc_v2i64 (v2f64 VR128:$src2))))]>;
def ORPSrr  : PSI<0x56, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "orps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (or (bc_v4i32 (v4f32 VR128:$src1)),
                     (bc_v4i32 (v4f32 VR128:$src2))))]>;
def ORPDrr  : PDI<0x56, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "orpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (or (bc_v2i64 (v2f64 VR128:$src1)),
                     (bc_v2i64 (v2f64 VR128:$src2))))]>;
def XORPSrr : PSI<0x57, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (xor (bc_v4i32 (v4f32 VR128:$src1)),
                     (bc_v4i32 (v4f32 VR128:$src2))))]>;
def XORPDrr : PDI<0x57, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (xor (bc_v2i64 (v2f64 VR128:$src1)),
                     (bc_v2i64 (v2f64 VR128:$src2))))]>;
}
def ANDPSrm : PSI<0x54, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst,
                  (and (bc_v4i32 (v4f32 VR128:$src1)),
                   (bc_v4i32 (loadv4f32 addr:$src2))))]>;
def ANDPDrm : PDI<0x54, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "andpd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst,
                  (and (bc_v2i64 (v2f64 VR128:$src1)),
                   (bc_v2i64 (loadv2f64 addr:$src2))))]>;
def ORPSrm  : PSI<0x56, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "orps {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst,
                   (or (bc_v4i32 (v4f32 VR128:$src1)),
                    (bc_v4i32 (loadv4f32 addr:$src2))))]>;
def ORPDrm  : PDI<0x56, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                "orpd {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst,
                   (or (bc_v2i64 (v2f64 VR128:$src1)),
                    (bc_v2i64 (loadv2f64 addr:$src2))))]>;
def XORPSrm : PSI<0x57, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst,
                  (xor (bc_v4i32 (v4f32 VR128:$src1)),
                   (bc_v4i32 (loadv4f32 addr:$src2))))]>;
def XORPDrm : PDI<0x57, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst,
                  (xor (bc_v2i64 (v2f64 VR128:$src1)),
                   (bc_v2i64 (loadv2f64 addr:$src2))))]>;
def ANDNPSrr : PSI<0x55, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "andnps {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst,
                  (and (vnot (bc_v4i32 (v4f32 VR128:$src1))),
                   (bc_v4i32 (v4f32 VR128:$src2))))]>;
def ANDNPSrm : PSI<0x55, MRMSrcMem, (ops VR128:$dst, VR128:$src1,f128mem:$src2),
                  "andnps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (and (vnot (bc_v4i32 (v4f32 VR128:$src1))),
                     (bc_v4i32 (loadv4f32 addr:$src2))))]>;
def ANDNPDrr : PDI<0x55, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "andnpd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst,
                  (and (vnot (bc_v2i64 (v2f64 VR128:$src1))),
                   (bc_v2i64 (v2f64 VR128:$src2))))]>;
def ANDNPDrm : PDI<0x55, MRMSrcMem, (ops VR128:$dst, VR128:$src1,f128mem:$src2),
                  "andnpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (and (vnot (bc_v2i64 (v2f64 VR128:$src1))),
                     (bc_v2i64 (loadv2f64 addr:$src2))))]>;
}

let isTwoAddress = 1 in {
def CMPPSrr : PSI<0xC2, MRMSrcReg, 
                  (ops VR128:$dst, VR128:$src1, VR128:$src, SSECC:$cc),
                  "cmp${cc}ps {$src, $dst|$dst, $src}",
                  [(set VR128:$dst, (int_x86_sse_cmp_ps VR128:$src1,
                                     VR128:$src, imm:$cc))]>;
def CMPPSrm : PSI<0xC2, MRMSrcMem, 
                  (ops VR128:$dst, VR128:$src1, f128mem:$src, SSECC:$cc),
                  "cmp${cc}ps {$src, $dst|$dst, $src}",
                  [(set VR128:$dst, (int_x86_sse_cmp_ps VR128:$src1,
                                     (load addr:$src), imm:$cc))]>;
def CMPPDrr : PDI<0xC2, MRMSrcReg, 
                  (ops VR128:$dst, VR128:$src1, VR128:$src, SSECC:$cc),
                  "cmp${cc}pd {$src, $dst|$dst, $src}", []>;
def CMPPDrm : PDI<0xC2, MRMSrcMem, 
                  (ops VR128:$dst, VR128:$src1, f128mem:$src, SSECC:$cc),
                  "cmp${cc}pd {$src, $dst|$dst, $src}", []>;
}

// Shuffle and unpack instructions
let isTwoAddress = 1 in {
def SHUFPSrr : PSIi8<0xC6, MRMSrcReg, 
                     (ops VR128:$dst, VR128:$src1, VR128:$src2, i32i8imm:$src3),
                     "shufps {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst, (v4f32 (vector_shuffle
                                               VR128:$src1, VR128:$src2,
                                               SHUFP_shuffle_mask:$src3)))]>;
def SHUFPSrm : PSIi8<0xC6, MRMSrcMem, 
                   (ops VR128:$dst, VR128:$src1, f128mem:$src2, i32i8imm:$src3),
                     "shufps {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst, (v4f32 (vector_shuffle
                                               VR128:$src1, (load addr:$src2),
                                               SHUFP_shuffle_mask:$src3)))]>;
def SHUFPDrr : PDIi8<0xC6, MRMSrcReg, 
                     (ops VR128:$dst, VR128:$src1, VR128:$src2, i8imm:$src3),
                     "shufpd {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst, (v2f64 (vector_shuffle
                                               VR128:$src1, VR128:$src2,
                                               SHUFP_shuffle_mask:$src3)))]>;
def SHUFPDrm : PDIi8<0xC6, MRMSrcMem, 
                     (ops VR128:$dst, VR128:$src1, f128mem:$src2, i8imm:$src3),
                     "shufpd {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst, (v2f64 (vector_shuffle
                                               VR128:$src1, (load addr:$src2),
                                               SHUFP_shuffle_mask:$src3)))]>;

def UNPCKHPSrr : PSI<0x15, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpckhps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v4f32 (vector_shuffle
                                              VR128:$src1, VR128:$src2,
                                              UNPCKH_shuffle_mask)))]>;
def UNPCKHPSrm : PSI<0x15, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpckhps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v4f32 (vector_shuffle
                                              VR128:$src1, (load addr:$src2),
                                              UNPCKH_shuffle_mask)))]>;
def UNPCKHPDrr : PDI<0x15, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpckhpd {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2f64 (vector_shuffle
                                              VR128:$src1, VR128:$src2,
                                              UNPCKH_shuffle_mask)))]>;
def UNPCKHPDrm : PDI<0x15, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpckhpd {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2f64 (vector_shuffle
                                              VR128:$src1, (load addr:$src2),
                                              UNPCKH_shuffle_mask)))]>;

def UNPCKLPSrr : PSI<0x14, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpcklps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v4f32 (vector_shuffle
                                              VR128:$src1, VR128:$src2,
                                              UNPCKL_shuffle_mask)))]>;
def UNPCKLPSrm : PSI<0x14, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpcklps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v4f32 (vector_shuffle
                                              VR128:$src1, (load addr:$src2),
                                              UNPCKL_shuffle_mask)))]>;
def UNPCKLPDrr : PDI<0x14, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpcklpd {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2f64 (vector_shuffle
                                              VR128:$src1, VR128:$src2,
                                              UNPCKL_shuffle_mask)))]>;
def UNPCKLPDrm : PDI<0x14, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpcklpd {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2f64 (vector_shuffle
                                              VR128:$src1, (load addr:$src2),
                                              UNPCKL_shuffle_mask)))]>;
}

// Horizontal ops
let isTwoAddress = 1 in {
def HADDPSrr : S3S_Intrr<0x7C, "haddps {$src2, $dst|$dst, $src2}",
                          int_x86_sse3_hadd_ps>;
def HADDPSrm : S3S_Intrm<0x7C, "haddps {$src2, $dst|$dst, $src2}",
                          int_x86_sse3_hadd_ps>;
def HADDPDrr : S3D_Intrr<0x7C, "haddpd {$src2, $dst|$dst, $src2}",
                          int_x86_sse3_hadd_pd>;
def HADDPDrm : S3D_Intrm<0x7C, "haddpd {$src2, $dst|$dst, $src2}",
                          int_x86_sse3_hadd_pd>;
def HSUBPSrr : S3S_Intrr<0x7C, "hsubps {$src2, $dst|$dst, $src2}",
                          int_x86_sse3_hsub_ps>;
def HSUBPSrm : S3S_Intrm<0x7C, "hsubps {$src2, $dst|$dst, $src2}",
                          int_x86_sse3_hsub_ps>;
def HSUBPDrr : S3D_Intrr<0x7C, "hsubpd {$src2, $dst|$dst, $src2}",
                          int_x86_sse3_hsub_pd>;
def HSUBPDrm : S3D_Intrm<0x7C, "hsubpd {$src2, $dst|$dst, $src2}",
                          int_x86_sse3_hsub_pd>;
}

//===----------------------------------------------------------------------===//
// SSE integer instructions
//===----------------------------------------------------------------------===//

// Move Instructions
def MOVD128rr : PDI<0x6E, MRMSrcReg, (ops VR128:$dst, R32:$src),
                    "movd {$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v4i32 (scalar_to_vector R32:$src)))]>;
def MOVD128rm : PDI<0x6E, MRMSrcMem, (ops VR128:$dst, i32mem:$src),
                  "movd {$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v4i32 (scalar_to_vector (loadi32 addr:$src))))]>;

def MOVD128mr : PDI<0x7E, MRMDestMem, (ops i32mem:$dst, VR128:$src),
                  "movd {$src, $dst|$dst, $src}", []>;

def MOVDQArr : PDI<0x6F, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "movdqa {$src, $dst|$dst, $src}", []>;
def MOVDQArm : PDI<0x6F, MRMSrcMem, (ops VR128:$dst, i128mem:$src),
                   "movdqa {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (loadv4i32 addr:$src))]>;
def MOVDQAmr : PDI<0x7F, MRMDestMem, (ops i128mem:$dst, VR128:$src),
                   "movdqa {$src, $dst|$dst, $src}",
                   [(store (v4i32 VR128:$src), addr:$dst)]>;

// SSE2 instructions with XS prefix
def MOVQ128rr : I<0x7E, MRMSrcReg, (ops VR128:$dst, VR64:$src),
                  "movq {$src, $dst|$dst, $src}",
                  [(set VR128:$dst,
                      (v2i64 (scalar_to_vector VR64:$src)))]>, XS,
                Requires<[HasSSE2]>;
def MOVQ128rm : I<0x7E, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                  "movq {$src, $dst|$dst, $src}", []>, XS,
                Requires<[HasSSE2]>;
def MOVQ128mr : PDI<0xD6, MRMSrcMem, (ops i64mem:$dst, VR128:$src),
                  "movq {$src, $dst|$dst, $src}", []>;

// 128-bit Integer Arithmetic
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def PADDBrr : PDI<0xFC, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "paddb {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v16i8 (add VR128:$src1, VR128:$src2)))]>;
def PADDWrr : PDI<0xFD, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "paddw {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v8i16 (add VR128:$src1, VR128:$src2)))]>;
def PADDDrr : PDI<0xFE, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "paddd {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v4i32 (add VR128:$src1, VR128:$src2)))]>;

def PADDQrr : PDI<0xD4, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "paddq {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v2i64 (add VR128:$src1, VR128:$src2)))]>;
}
def PADDBrm : PDI<0xFC, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "paddb {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v16i8 (add VR128:$src1,
                                        (load addr:$src2))))]>;
def PADDWrm : PDI<0xFD, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "paddw {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v8i16 (add VR128:$src1,
                                        (load addr:$src2))))]>;
def PADDDrm : PDI<0xFE, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "paddd {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v4i32 (add VR128:$src1,
                                        (load addr:$src2))))]>;
def PADDQrm : PDI<0xD4, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "paddd {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v2i64 (add VR128:$src1,
                                        (load addr:$src2))))]>;

def PSUBBrr : PDI<0xF8, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "psubb {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v16i8 (sub VR128:$src1, VR128:$src2)))]>;
def PSUBWrr : PDI<0xF9, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "psubw {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v8i16 (sub VR128:$src1, VR128:$src2)))]>;
def PSUBDrr : PDI<0xFA, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "psubd {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v4i32 (sub VR128:$src1, VR128:$src2)))]>;
def PSUBQrr : PDI<0xFB, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "psubq {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v2i64 (sub VR128:$src1, VR128:$src2)))]>;

def PSUBBrm : PDI<0xF8, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "psubb {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v16i8 (sub VR128:$src1,
                                        (load addr:$src2))))]>;
def PSUBWrm : PDI<0xF9, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "psubw {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v8i16 (sub VR128:$src1,
                                        (load addr:$src2))))]>;
def PSUBDrm : PDI<0xFA, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "psubd {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v4i32 (sub VR128:$src1,
                                        (load addr:$src2))))]>;
def PSUBQrm : PDI<0xFB, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "psubd {$src2, $dst|$dst, $src2}",
              [(set VR128:$dst, (v2i64 (sub VR128:$src1,
                                        (load addr:$src2))))]>;
}

// Logical
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def PANDrr : PDI<0xDB, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                 "pand {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v2i64 (and VR128:$src1, VR128:$src2)))]>;

def PANDrm : PDI<0xDB, MRMSrcMem, (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                 "pand {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v2i64 (and VR128:$src1,
                                           (load addr:$src2))))]>;
def PORrr  : PDI<0xDB, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                 "por {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v2i64 (or VR128:$src1, VR128:$src2)))]>;

def PORrm  : PDI<0xDB, MRMSrcMem, (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                 "por {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v2i64 (or VR128:$src1,
                                           (load addr:$src2))))]>;
def PXORrr : PDI<0xEF, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                 "pxor {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v2i64 (xor VR128:$src1, VR128:$src2)))]>;

def PXORrm : PDI<0xEF, MRMSrcMem, (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                 "pxor {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v2i64 (xor VR128:$src1,
                                           (load addr:$src2))))]>;
}

def PANDNrr : PDI<0xDF, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "pandn {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (and (vnot VR128:$src1),
                                            VR128:$src2)))]>;

def PANDNrm : PDI<0xDF, MRMSrcMem, (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                 "pandn {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v2i64 (and (vnot VR128:$src1),
                                           (load addr:$src2))))]>;
}

// Pack instructions
let isTwoAddress = 1 in {
def PACKSSWBrr : PDI<0x63, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                       VR128:$src2),
                 "packsswb {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v8i16 (int_x86_sse2_packsswb_128
                                           VR128:$src1,
                                           VR128:$src2)))]>;
def PACKSSWBrm : PDI<0x63, MRMSrcMem, (ops VR128:$dst, VR128:$src1,
                                       i128mem:$src2),
                 "packsswb {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v8i16 (int_x86_sse2_packsswb_128
                                           VR128:$src1,
                                         (bc_v8i16 (loadv2f64 addr:$src2)))))]>;
def PACKSSDWrr : PDI<0x6B, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                       VR128:$src2),
                 "packssdw {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v4i32 (int_x86_sse2_packssdw_128
                                           VR128:$src1,
                                           VR128:$src2)))]>;
def PACKSSDWrm : PDI<0x6B, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                       i128mem:$src2),
                 "packssdw {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v4i32 (int_x86_sse2_packssdw_128
                                           VR128:$src1,
                                         (bc_v4i32 (loadv2i64 addr:$src2)))))]>;
def PACKUSWBrr : PDI<0x67, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                       VR128:$src2),
                     "packuswb {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v8i16 (int_x86_sse2_packuswb_128
                                           VR128:$src1,
                                           VR128:$src2)))]>;
def PACKUSWBrm : PDI<0x67, MRMSrcReg, (ops VR128:$dst, VR128:$src1,
                                       i128mem:$src2),
                     "packuswb {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v8i16 (int_x86_sse2_packuswb_128
                                           VR128:$src1,
                                         (bc_v8i16 (loadv2i64 addr:$src2)))))]>;
}

// Shuffle and unpack instructions
def PSHUFWrr : PSIi8<0x70, MRMDestReg,
                     (ops VR64:$dst, VR64:$src1, i8imm:$src2),
                     "pshufw {$src2, $src1, $dst|$dst, $src1, $src2}", []>;
def PSHUFWrm : PSIi8<0x70, MRMSrcMem,
                     (ops VR64:$dst, i64mem:$src1, i8imm:$src2),
                     "pshufw {$src2, $src1, $dst|$dst, $src1, $src2}", []>;

def PSHUFDrr : PDIi8<0x70, MRMDestReg,
                     (ops VR128:$dst, VR128:$src1, i8imm:$src2),
                     "pshufd {$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set VR128:$dst, (v4i32 (vector_shuffle
                                               VR128:$src1, (undef),
                                               PSHUFD_shuffle_mask:$src2)))]>;
def PSHUFDrm : PDIi8<0x70, MRMSrcMem,
                     (ops VR128:$dst, i128mem:$src1, i8imm:$src2),
                     "pshufd {$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set VR128:$dst, (v4i32 (vector_shuffle
                                               (load addr:$src1), (undef),
                                               PSHUFD_shuffle_mask:$src2)))]>;

// SSE2 with ImmT == Imm8 and XS prefix.
def PSHUFHWrr : Ii8<0x70, MRMDestReg,
                    (ops VR128:$dst, VR128:$src1, i8imm:$src2),
                    "pshufhw {$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                              VR128:$src1, (undef),
                                              PSHUFHW_shuffle_mask:$src2)))]>,
                XS, Requires<[HasSSE2]>;
def PSHUFHWrm : Ii8<0x70, MRMDestMem,
                    (ops VR128:$dst, i128mem:$src1, i8imm:$src2),
                    "pshufhw {$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                     (bc_v8i16 (loadv2i64 addr:$src1)), (undef),
                                              PSHUFHW_shuffle_mask:$src2)))]>,
                XS, Requires<[HasSSE2]>;

// SSE2 with ImmT == Imm8 and XD prefix.
def PSHUFLWrr : Ii8<0x70, MRMDestReg,
                    (ops VR128:$dst, VR128:$src1, i32i8imm:$src2),
                    "pshuflw {$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                              VR128:$src1, (undef),
                                              PSHUFLW_shuffle_mask:$src2)))]>,
                XD, Requires<[HasSSE2]>;
def PSHUFLWrm : Ii8<0x70, MRMDestMem,
                    (ops VR128:$dst, i128mem:$src1, i32i8imm:$src2),
                    "pshuflw {$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                     (bc_v8i16 (loadv2i64 addr:$src1)), (undef),
                                              PSHUFLW_shuffle_mask:$src2)))]>,
                XD, Requires<[HasSSE2]>;

let isTwoAddress = 1 in {
def PUNPCKLBWrr : PDI<0x60, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpcklbw {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v16i8 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLBWrm : PDI<0x60, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpcklbw {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v16i8 (vector_shuffle VR128:$src1, (load addr:$src2),
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLWDrr : PDI<0x61, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpcklwd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v8i16 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLWDrm : PDI<0x61, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpcklwd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v8i16 (vector_shuffle VR128:$src1, (load addr:$src2),
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLDQrr : PDI<0x62, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpckldq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLDQrm : PDI<0x62, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpckldq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4i32 (vector_shuffle VR128:$src1, (load addr:$src2),
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLQDQrr : PDI<0x6C, MRMSrcReg, 
                       (ops VR128:$dst, VR128:$src1, VR128:$src2),
                       "punpcklqdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v2i64 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLQDQrm : PDI<0x6C, MRMSrcMem, 
                       (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                       "punpcklqdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v2i64 (vector_shuffle VR128:$src1, (load addr:$src2),
                                UNPCKL_shuffle_mask)))]>;

def PUNPCKHBWrr : PDI<0x68, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpckhbw {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v16i8 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHBWrm : PDI<0x68, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpckhbw {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v16i8 (vector_shuffle VR128:$src1, (load addr:$src2),
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHWDrr : PDI<0x69, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpckhwd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v8i16 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHWDrm : PDI<0x69, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpckhwd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v8i16 (vector_shuffle VR128:$src1, (load addr:$src2),
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHDQrr : PDI<0x6A, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpckhdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHDQrm : PDI<0x6A, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpckhdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4i32 (vector_shuffle VR128:$src1, (load addr:$src2),
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHQDQrr : PDI<0x6D, MRMSrcReg, 
                       (ops VR128:$dst, VR128:$src1, VR128:$src2),
                       "punpckhdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v2i64 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHQDQrm : PDI<0x6D, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpckhqdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v2i64 (vector_shuffle VR128:$src1, (load addr:$src2),
                                UNPCKH_shuffle_mask)))]>;
}

// Extract / Insert
def PEXTRWrr : PDIi8<0xC5, MRMSrcReg,
                     (ops R32:$dst, VR128:$src1, i32i8imm:$src2),
                     "pextrw {$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set R32:$dst, (X86pextrw (v8i16 VR128:$src1),
                                      (i32 imm:$src2)))]>;
def PEXTRWrm : PDIi8<0xC5, MRMSrcMem,
                     (ops R32:$dst, i128mem:$src1, i32i8imm:$src2),
                     "pextrw {$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set R32:$dst, (X86pextrw (loadv8i16 addr:$src1),
                                      (i32 imm:$src2)))]>;

let isTwoAddress = 1 in {
def PINSRWrr : PDIi8<0xC4, MRMSrcReg,
                     (ops VR128:$dst, VR128:$src1, R32:$src2, i32i8imm:$src3),
                     "pinsrw {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst, (v8i16 (vector_insert (v8i16 VR128:$src1),
                               R32:$src2, (i32 imm:$src3))))]>;
def PINSRWrm : PDIi8<0xC4, MRMSrcMem,
                     (ops VR128:$dst, VR128:$src1, i16mem:$src2, i32i8imm:$src3),
                     "pinsrw {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst,
                       (v8i16 (vector_insert (v8i16 VR128:$src1),
                               (i32 (anyext (loadi16 addr:$src2))),
                               (i32 imm:$src3))))]>;
}

//===----------------------------------------------------------------------===//
// Miscellaneous Instructions
//===----------------------------------------------------------------------===//

// Mask creation
def MOVMSKPSrr : PSI<0x50, MRMSrcReg, (ops R32:$dst, VR128:$src),
                     "movmskps {$src, $dst|$dst, $src}",
                     [(set R32:$dst, (int_x86_sse_movmsk_ps VR128:$src))]>;
def MOVMSKPDrr : PSI<0x50, MRMSrcReg, (ops R32:$dst, VR128:$src),
                     "movmskpd {$src, $dst|$dst, $src}",
                     [(set R32:$dst, (int_x86_sse2_movmskpd VR128:$src))]>;

def PMOVMSKBrr : PDI<0xD7, MRMSrcReg, (ops R32:$dst, VR128:$src),
                     "pmovmskb {$src, $dst|$dst, $src}",
                     [(set R32:$dst, (int_x86_sse2_pmovmskb_128 VR128:$src))]>;

// Prefetching loads
def PREFETCHT0   : I<0x18, MRM1m, (ops i8mem:$src),
                     "prefetcht0 $src", []>, TB,
                   Requires<[HasSSE1]>;
def PREFETCHT1   : I<0x18, MRM2m, (ops i8mem:$src),
                     "prefetcht0 $src", []>, TB,
                   Requires<[HasSSE1]>;
def PREFETCHT2   : I<0x18, MRM3m, (ops i8mem:$src),
                     "prefetcht0 $src", []>, TB,
                   Requires<[HasSSE1]>;
def PREFETCHTNTA : I<0x18, MRM0m, (ops i8mem:$src),
                     "prefetcht0 $src", []>, TB,
                   Requires<[HasSSE1]>;

// Non-temporal stores
def MOVNTQ   : I<0xE7, MRMDestMem, (ops i64mem:$dst, VR64:$src),
                 "movntq {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasSSE1]>;
def MOVNTPS  : I<0x2B, MRMDestMem, (ops i128mem:$dst, VR128:$src),
                "movntps {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasSSE1]>;
def MASKMOVQ : I<0xF7, MRMDestMem, (ops i64mem:$dst, VR64:$src),
                 "maskmovq {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasSSE1]>;

// Store fence
def SFENCE : I<0xAE, MRM7m, (ops),
               "sfence", []>, TB, Requires<[HasSSE1]>;

// Load MXCSR register
def LDMXCSR : I<0xAE, MRM2m, (ops i32mem:$src),
                "ldmxcsr {$src|$src}", []>, TB, Requires<[HasSSE1]>;

//===----------------------------------------------------------------------===//
// Alias Instructions
//===----------------------------------------------------------------------===//

// Alias instructions that map zero vector to pxor / xorp* for sse.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
def V_SET0_PI : PDI<0xEF, MRMInitReg, (ops VR128:$dst),
                    "pxor $dst, $dst",
                    [(set VR128:$dst, (v2i64 immAllZerosV))]>;
def V_SET0_PS : PSI<0x57, MRMInitReg, (ops VR128:$dst),
                    "xorps $dst, $dst",
                    [(set VR128:$dst, (v4f32 immAllZerosV))]>;
def V_SET0_PD : PDI<0x57, MRMInitReg, (ops VR128:$dst),
                    "xorpd $dst, $dst",
                    [(set VR128:$dst, (v2f64 immAllZerosV))]>;

def V_SETALLONES : PDI<0x76, MRMInitReg, (ops VR128:$dst),
                       "pcmpeqd $dst, $dst",
                       [(set VR128:$dst, (v2f64 immAllOnesV))]>;

// Scalar to 128-bit vector with zero extension.
// Three operand (but two address) aliases.
let isTwoAddress = 1 in {
def MOVZSS128rr : SSI<0x10, MRMSrcReg, (ops VR128:$dst, VR128:$src1, FR32:$src2),
                      "movss {$src2, $dst|$dst, $src2}", []>;
def MOVZSD128rr : SDI<0x10, MRMSrcReg, (ops VR128:$dst, VR128:$src1, FR64:$src2),
                      "movsd {$src2, $dst|$dst, $src2}", []>;
def MOVZD128rr  : PDI<0x6E, MRMSrcReg, (ops VR128:$dst, VR128:$src1, R32:$src2),
                      "movd {$src2, $dst|$dst, $src2}", []>;
def MOVZQ128rr  : I<0x7E, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR64:$src2),
                    "movq {$src2, $dst|$dst, $src2}", []>;
}

// Loading from memory automatically zeroing upper bits.
def MOVZSS128rm : SSI<0x10, MRMSrcMem, (ops VR128:$dst, f32mem:$src),
                      "movss {$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4f32 (X86zexts2vec (loadf32 addr:$src))))]>;
def MOVZSD128rm : SDI<0x10, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
                      "movsd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v2f64 (X86zexts2vec (loadf64 addr:$src))))]>;
def MOVZD128rm : PDI<0x6E, MRMSrcMem, (ops VR128:$dst, i32mem:$src),
                     "movd {$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (v4i32 (X86zexts2vec (loadi32 addr:$src))))]>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// 128-bit vector undef's.
def : Pat<(v2f64 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v16i8 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v8i16 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;

// 128-bit vector all zero's.
def : Pat<(v16i8 immAllZerosV), (v16i8 (V_SET0_PI))>, Requires<[HasSSE2]>;
def : Pat<(v8i16 immAllZerosV), (v8i16 (V_SET0_PI))>, Requires<[HasSSE2]>;
def : Pat<(v4i32 immAllZerosV), (v4i32 (V_SET0_PI))>, Requires<[HasSSE2]>;

// 128-bit vector all one's.
def : Pat<(v16i8 immAllOnesV), (v16i8 (V_SETALLONES))>, Requires<[HasSSE2]>;
def : Pat<(v8i16 immAllOnesV), (v8i16 (V_SETALLONES))>, Requires<[HasSSE2]>;
def : Pat<(v4i32 immAllOnesV), (v4i32 (V_SETALLONES))>, Requires<[HasSSE2]>;
def : Pat<(v2i64 immAllOnesV), (v2i64 (V_SETALLONES))>, Requires<[HasSSE2]>;
def : Pat<(v4f32 immAllOnesV), (v4f32 (V_SETALLONES))>, Requires<[HasSSE1]>;

// Load 128-bit integer vector values.
def : Pat<(v16i8 (load addr:$src)), (MOVDQArm addr:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v8i16 (load addr:$src)), (MOVDQArm addr:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v4i32 (load addr:$src)), (MOVDQArm addr:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v2i64 (load addr:$src)), (MOVDQArm addr:$src)>,
      Requires<[HasSSE2]>;

// Store 128-bit integer vector values.
def : Pat<(store (v16i8 VR128:$src), addr:$dst),
          (MOVDQAmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(store (v8i16 VR128:$src), addr:$dst),
          (MOVDQAmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(store (v4i32 VR128:$src), addr:$dst),
          (MOVDQAmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(store (v2i64 VR128:$src), addr:$dst),
          (MOVDQAmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;

// Scalar to v8i16 / v16i8. The source may be a R32, but only the lower 8 or
// 16-bits matter.
def : Pat<(v8i16 (X86s2vec R32:$src)), (MOVD128rr R32:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v16i8 (X86s2vec R32:$src)), (MOVD128rr R32:$src)>,
      Requires<[HasSSE2]>;

// bit_convert
def : Pat<(v2i64 (bitconvert (v4i32 VR128:$src))), (v2i64 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v2i64 (bitconvert (v8i16 VR128:$src))), (v2i64 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v2i64 (bitconvert (v16i8 VR128:$src))), (v2i64 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v4i32 (bitconvert (v2i64 VR128:$src))), (v4i32 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v4i32 (bitconvert (v8i16 VR128:$src))), (v4i32 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v4i32 (bitconvert (v16i8 VR128:$src))), (v4i32 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v8i16 (bitconvert (v2i64 VR128:$src))), (v4i32 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v8i16 (bitconvert (v4i32 VR128:$src))), (v4i32 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v8i16 (bitconvert (v16i8 VR128:$src))), (v4i32 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v16i8 (bitconvert (v2i64 VR128:$src))), (v4i32 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v16i8 (bitconvert (v4i32 VR128:$src))), (v4i32 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v16i8 (bitconvert (v8i16 VR128:$src))), (v4i32 VR128:$src)>,
      Requires<[HasSSE2]>;

def : Pat<(v4i32 (bitconvert (v4f32 VR128:$src))), (v4i32 VR128:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v4f32 (bitconvert (v4i32 VR128:$src))), (v4f32 VR128:$src)>,
      Requires<[HasSSE2]>;

// Zeroing a VR128 then do a MOVS* to the lower bits.
def : Pat<(v2f64 (X86zexts2vec FR64:$src)),
          (MOVZSD128rr (V_SET0_PD), FR64:$src)>, Requires<[HasSSE2]>;
def : Pat<(v4f32 (X86zexts2vec FR32:$src)),
          (MOVZSS128rr (V_SET0_PS), FR32:$src)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (X86zexts2vec VR64:$src)),
          (MOVZQ128rr (V_SET0_PI), VR64:$src)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (X86zexts2vec R32:$src)),
          (MOVZD128rr (V_SET0_PI), R32:$src)>, Requires<[HasSSE2]>;
def : Pat<(v8i16 (X86zexts2vec R16:$src)),
         (MOVZD128rr (V_SET0_PI), (MOVZX32rr16 R16:$src))>, Requires<[HasSSE2]>;
def : Pat<(v16i8 (X86zexts2vec R8:$src)),
          (MOVZD128rr (V_SET0_PI), (MOVZX32rr8 R8:$src))>, Requires<[HasSSE2]>;

// Splat v2f64 / v2i64
def : Pat<(vector_shuffle (v2f64 VR128:$src), (undef), SSE_splat_mask:$sm),
          (v2f64 (UNPCKLPDrr VR128:$src, VR128:$src))>,   Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (v2i64 VR128:$src), (undef), SSE_splat_mask:$sm),
          (v2i64 (PUNPCKLQDQrr VR128:$src, VR128:$src))>, Requires<[HasSSE2]>;

// Splat v4f32
def : Pat<(vector_shuffle (v4f32 VR128:$src), (undef), SSE_splat_mask:$sm),
          (v4f32 (SHUFPSrr VR128:$src, VR128:$src, SSE_splat_mask:$sm))>,
      Requires<[HasSSE1]>;

// Shuffle v4i32 with SHUFP* if others do not match.
def : Pat<(vector_shuffle (v4i32 VR128:$src1), (v4i32 VR128:$src2),
           SHUFP_int_shuffle_mask:$sm),
          (v4i32 (SHUFPSrr VR128:$src1, VR128:$src2,
                  SHUFP_int_shuffle_mask:$sm))>, Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (v4i32 VR128:$src1), (load addr:$src2),
           SHUFP_int_shuffle_mask:$sm),
          (v4i32 (SHUFPSrm VR128:$src1, addr:$src2,
                  SHUFP_int_shuffle_mask:$sm))>, Requires<[HasSSE2]>;

// Shuffle v4f32 with PSHUF* if others do not match.
def : Pat<(vector_shuffle (v4f32 VR128:$src1), (undef),
           PSHUFD_fp_shuffle_mask:$sm),
          (v4f32 (PSHUFDrr VR128:$src1, PSHUFD_fp_shuffle_mask:$sm))>,
      Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (loadv4f32 addr:$src1), (undef),
           PSHUFD_fp_shuffle_mask:$sm),
          (v4f32 (PSHUFDrm addr:$src1, PSHUFD_fp_shuffle_mask:$sm))>,
      Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (v4f32 VR128:$src1), (undef),
           PSHUFHW_fp_shuffle_mask:$sm),
          (v4f32 (PSHUFHWrr VR128:$src1, PSHUFHW_fp_shuffle_mask:$sm))>,
      Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (loadv4f32 addr:$src1), (undef),
           PSHUFHW_fp_shuffle_mask:$sm),
          (v4f32 (PSHUFHWrm addr:$src1, PSHUFHW_fp_shuffle_mask:$sm))>,
      Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (v4f32 VR128:$src1), (undef),
           PSHUFLW_fp_shuffle_mask:$sm),
          (v4f32 (PSHUFLWrr VR128:$src1, PSHUFLW_fp_shuffle_mask:$sm))>,
      Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (loadv4f32 addr:$src1), (undef),
           PSHUFLW_fp_shuffle_mask:$sm),
          (v4f32 (PSHUFLWrm addr:$src1, PSHUFLW_fp_shuffle_mask:$sm))>,
      Requires<[HasSSE2]>;

// Logical ops
def : Pat<(and (bc_v4i32 (v4f32 VR128:$src1)), (loadv4i32 addr:$src2)),
          (ANDPSrm VR128:$src1, addr:$src2)>;
def : Pat<(and (bc_v2i64 (v2f64 VR128:$src1)), (loadv2i64 addr:$src2)),
          (ANDPDrm VR128:$src1, addr:$src2)>;
def : Pat<(or  (bc_v4i32 (v4f32 VR128:$src1)), (loadv4i32 addr:$src2)),
          (ORPSrm VR128:$src1, addr:$src2)>;
def : Pat<(or  (bc_v2i64 (v2f64 VR128:$src1)), (loadv2i64 addr:$src2)),
          (ORPDrm VR128:$src1, addr:$src2)>;
def : Pat<(xor (bc_v4i32 (v4f32 VR128:$src1)), (loadv4i32 addr:$src2)),
          (XORPSrm VR128:$src1, addr:$src2)>;
def : Pat<(xor (bc_v2i64 (v2f64 VR128:$src1)), (loadv2i64 addr:$src2)),
          (XORPDrm VR128:$src1, addr:$src2)>;
def : Pat<(and (vnot (bc_v4i32 (v4f32 VR128:$src1))), (loadv4i32 addr:$src2)),
          (ANDNPSrm VR128:$src1, addr:$src2)>;
def : Pat<(and (vnot (bc_v2i64 (v2f64 VR128:$src1))), (loadv2i64 addr:$src2)),
          (ANDNPDrm VR128:$src1, addr:$src2)>;

def : Pat<(bc_v4f32 (v4i32 (and VR128:$src1, VR128:$src2))),
          (ANDPSrr VR128:$src1, VR128:$src2)>;
def : Pat<(bc_v4f32 (v4i32 (or VR128:$src1, VR128:$src2))),
          (ORPSrr VR128:$src1, VR128:$src2)>;
def : Pat<(bc_v4f32 (v4i32 (xor VR128:$src1, VR128:$src2))),
          (XORPSrr VR128:$src1, VR128:$src2)>;
def : Pat<(bc_v4f32 (v4i32 (and (vnot VR128:$src1), VR128:$src2))),
          (ANDNPSrr VR128:$src1, VR128:$src2)>;

def : Pat<(bc_v4f32 (v4i32 (and VR128:$src1, (load addr:$src2)))),
          (ANDPSrm (v4i32 VR128:$src1), addr:$src2)>;
def : Pat<(bc_v4f32 (v4i32 (or VR128:$src1, (load addr:$src2)))),
          (ORPSrm VR128:$src1, addr:$src2)>;
def : Pat<(bc_v4f32 (v4i32 (xor VR128:$src1, (load addr:$src2)))),
          (XORPSrm VR128:$src1, addr:$src2)>;
def : Pat<(bc_v4f32 (v4i32 (and (vnot VR128:$src1), (load addr:$src2)))),
          (ANDNPSrm VR128:$src1, addr:$src2)>;

def : Pat<(bc_v2f64 (v2i64 (and VR128:$src1, VR128:$src2))),
          (ANDPDrr VR128:$src1, VR128:$src2)>;
def : Pat<(bc_v2f64 (v2i64 (or VR128:$src1, VR128:$src2))),
          (ORPDrr VR128:$src1, VR128:$src2)>;
def : Pat<(bc_v2f64 (v2i64 (xor VR128:$src1, VR128:$src2))),
          (XORPDrr VR128:$src1, VR128:$src2)>;
def : Pat<(bc_v2f64 (v2i64 (and (vnot VR128:$src1), VR128:$src2))),
          (ANDNPDrr VR128:$src1, VR128:$src2)>;

def : Pat<(bc_v2f64 (v2i64 (and VR128:$src1, (load addr:$src2)))),
          (ANDPSrm (v2i64 VR128:$src1), addr:$src2)>;
def : Pat<(bc_v2f64 (v2i64 (or VR128:$src1, (load addr:$src2)))),
          (ORPSrm VR128:$src1, addr:$src2)>;
def : Pat<(bc_v2f64 (v2i64 (xor VR128:$src1, (load addr:$src2)))),
          (XORPSrm VR128:$src1, addr:$src2)>;
def : Pat<(bc_v2f64 (v2i64 (and (vnot VR128:$src1), (load addr:$src2)))),
          (ANDNPSrm VR128:$src1, addr:$src2)>;

def : Pat<(v4i32 (and VR128:$src1, VR128:$src2)),
          (PANDrr VR128:$src1, VR128:$src2)>;
def : Pat<(v8i16 (and VR128:$src1, VR128:$src2)),
          (PANDrr VR128:$src1, VR128:$src2)>;
def : Pat<(v16i8 (and VR128:$src1, VR128:$src2)),
          (PANDrr VR128:$src1, VR128:$src2)>;
def : Pat<(v4i32 (or VR128:$src1, VR128:$src2)),
          (PORrr VR128:$src1, VR128:$src2)>;
def : Pat<(v8i16 (or VR128:$src1, VR128:$src2)),
          (PORrr VR128:$src1, VR128:$src2)>;
def : Pat<(v16i8 (or VR128:$src1, VR128:$src2)),
          (PORrr VR128:$src1, VR128:$src2)>;
def : Pat<(v4i32 (xor VR128:$src1, VR128:$src2)),
          (PXORrr VR128:$src1, VR128:$src2)>;
def : Pat<(v8i16 (xor VR128:$src1, VR128:$src2)),
          (PXORrr VR128:$src1, VR128:$src2)>;
def : Pat<(v16i8 (xor VR128:$src1, VR128:$src2)),
          (PXORrr VR128:$src1, VR128:$src2)>;
def : Pat<(v4i32 (and (vnot VR128:$src1), VR128:$src2)),
          (PANDNrr VR128:$src1, VR128:$src2)>;
def : Pat<(v8i16 (and (vnot VR128:$src1), VR128:$src2)),
          (PANDNrr VR128:$src1, VR128:$src2)>;
def : Pat<(v16i8 (and (vnot VR128:$src1), VR128:$src2)),
          (PANDNrr VR128:$src1, VR128:$src2)>;

def : Pat<(v4i32 (and VR128:$src1, (load addr:$src2))),
          (PANDrm VR128:$src1, addr:$src2)>;
def : Pat<(v8i16 (and VR128:$src1, (load addr:$src2))),
          (PANDrm VR128:$src1, addr:$src2)>;
def : Pat<(v16i8 (and VR128:$src1, (load addr:$src2))),
          (PANDrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (or VR128:$src1, (load addr:$src2))),
          (PORrm VR128:$src1, addr:$src2)>;
def : Pat<(v8i16 (or VR128:$src1, (load addr:$src2))),
          (PORrm VR128:$src1, addr:$src2)>;
def : Pat<(v16i8 (or VR128:$src1, (load addr:$src2))),
          (PORrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (xor VR128:$src1, (load addr:$src2))),
          (PXORrm VR128:$src1, addr:$src2)>;
def : Pat<(v8i16 (xor VR128:$src1, (load addr:$src2))),
          (PXORrm VR128:$src1, addr:$src2)>;
def : Pat<(v16i8 (xor VR128:$src1, (load addr:$src2))),
          (PXORrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (and (vnot VR128:$src1), (load addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>;
def : Pat<(v8i16 (and (vnot VR128:$src1), (load addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>;
def : Pat<(v16i8 (and (vnot VR128:$src1), (load addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>;
