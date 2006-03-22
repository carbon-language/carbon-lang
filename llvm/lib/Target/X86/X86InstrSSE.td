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

def SDTX86Unpcklp : SDTypeProfile<1, 2,
                                  [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>]>;

def X86loadp   : SDNode<"X86ISD::LOAD_PACK", SDTLoad, 
                        [SDNPHasChain]>;
def X86fand    : SDNode<"X86ISD::FAND",     SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86fxor    : SDNode<"X86ISD::FXOR",     SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86s2vec   : SDNode<"X86ISD::SCALAR_TO_VECTOR",
                        SDTypeProfile<1, 1, []>, []>;
def X86unpcklp : SDNode<"X86ISD::UNPCKLP",
                        SDTX86Unpcklp, []>;

//===----------------------------------------------------------------------===//
// SSE pattern fragments
//===----------------------------------------------------------------------===//

def X86loadpf32  : PatFrag<(ops node:$ptr), (f32   (X86loadp node:$ptr))>;
def X86loadpf64  : PatFrag<(ops node:$ptr), (f64   (X86loadp node:$ptr))>;

def loadv4f32    : PatFrag<(ops node:$ptr), (v4f32 (load node:$ptr))>;
def loadv2f64    : PatFrag<(ops node:$ptr), (v2f64 (load node:$ptr))>;

// SHUFFLE_get_shuf_imm xform function: convert vector_shuffle mask to PSHUF*,
// SHUFP* etc. imm.
def SHUFFLE_get_shuf_imm : SDNodeXForm<build_vector, [{
  return getI8Imm(X86::getShuffleSHUFImmediate(N));
}]>;

def SHUFFLE_get_pshufd_imm : SDNodeXForm<build_vector, [{
  return getI8Imm(X86::getShufflePSHUFDImmediate(N));
}]>;

def SHUFP_splat_mask : PatLeaf<(build_vector), [{
  return X86::isSplatMask(N);
}], SHUFFLE_get_shuf_imm>;

def PSHUFD_shuffle_mask : PatLeaf<(build_vector), [{}], SHUFFLE_get_pshufd_imm>;

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

// Conversion instructions
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

// Comparison instructions
let isTwoAddress = 1 in {
def CMPSSrr : SSI<0xC2, MRMSrcReg, 
                (ops FR32:$dst, FR32:$src1, FR32:$src, SSECC:$cc),
                "cmp${cc}ss {$src, $dst|$dst, $src}", []>;
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

def MOVLPSrm : PSI<0x12, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
                   "movlps {$src, $dst|$dst, $src}", []>;
def MOVLPSmr : PSI<0x13, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movlps {$src, $dst|$dst, $src}", []>;
def MOVLPDrm : PDI<0x12, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
                   "movlpd {$src, $dst|$dst, $src}", []>;
def MOVLPDmr : PDI<0x13, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movlpd {$src, $dst|$dst, $src}", []>;

def MOVHPSrm : PSI<0x16, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
                   "movhps {$src, $dst|$dst, $src}", []>;
def MOVHPSmr : PSI<0x17, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movhps {$src, $dst|$dst, $src}", []>;
def MOVHPDrm : PDI<0x16, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
                   "movhpd {$src, $dst|$dst, $src}", []>;
def MOVHPDmr : PDI<0x17, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movhpd {$src, $dst|$dst, $src}", []>;

def MOVLHPSrr : PSI<0x16, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                    "movlhps {$src, $dst|$dst, $src}", []>;
def MOVHLPSrr : PSI<0x12, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                    "movlhps {$src, $dst|$dst, $src}", []>;

def MOVMSKPSrr : PSI<0x50, MRMSrcReg, (ops R32:$dst, VR128:$src),
                     "movmskps {$src, $dst|$dst, $src}", []>;
def MOVMSKPDrr : PSI<0x50, MRMSrcReg, (ops R32:$dst, VR128:$src),
                     "movmskpd {$src, $dst|$dst, $src}", []>;

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
                  [(set VR128:$dst, (fsub VR128:$src1, VR128:$src2))]>;
def SUBPDrm : PDI<0x5C, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "subpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (fsub VR128:$src1,
                                     (load addr:$src2)))]>;
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
                  [(set VR128:$dst, (v4i32 (and VR128:$src1, VR128:$src2)))]>;
def ANDPDrr : PDI<0x54, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                "andpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (and VR128:$src1, VR128:$src2)))]>;
def ORPSrr  : PSI<0x56, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "orps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4i32 (or VR128:$src1, VR128:$src2)))]>;
def ORPDrr  : PDI<0x56, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "orpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (or VR128:$src1, VR128:$src2)))]>;
def XORPSrr : PSI<0x57, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4i32 (xor VR128:$src1, VR128:$src2)))]>;
def XORPDrr : PDI<0x57, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (xor VR128:$src1, VR128:$src2)))]>;
}
def ANDPSrm : PSI<0x54, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4i32 (and VR128:$src1,
                                            (load addr:$src2))))]>;
def ANDPDrm : PDI<0x54, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "andpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (and VR128:$src1,
                                            (load addr:$src2))))]>;
def ORPSrm  : PSI<0x56, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "orps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4i32 (or VR128:$src1,
                                            (load addr:$src2))))]>;
def ORPDrm  : PDI<0x56, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                "orpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (or VR128:$src1,
                                            (load addr:$src2))))]>;
def XORPSrm : PSI<0x57, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4i32 (xor VR128:$src1,
                                            (load addr:$src2))))]>;
def XORPDrm : PDI<0x57, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (xor VR128:$src1,
                                            (load addr:$src2))))]>;
def ANDNPSrr : PSI<0x55, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "andnps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4i32 (and (not VR128:$src1),
                                            VR128:$src2)))]>;
def ANDNPSrm : PSI<0x55, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "andnps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v4i32 (and (not VR128:$src1),
                                            (load addr:$src2))))]>;
def ANDNPDrr : PDI<0x55, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "andnpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (and (not VR128:$src1),
                                            VR128:$src2)))]>;

def ANDNPDrm : PDI<0x55, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "andnpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (and VR128:$src1,
                                            (load addr:$src2))))]>;
}

let isTwoAddress = 1 in {
def CMPPSrr : PSI<0xC2, MRMSrcReg, 
                (ops VR128:$dst, VR128:$src1, VR128:$src, SSECC:$cc),
                "cmp${cc}ps {$src, $dst|$dst, $src}", []>;
def CMPPSrm : PSI<0xC2, MRMSrcMem, 
                (ops VR128:$dst, VR128:$src1, f128mem:$src, SSECC:$cc),
                "cmp${cc}ps {$src, $dst|$dst, $src}", []>;
def CMPPDrr : PDI<0xC2, MRMSrcReg, 
                (ops VR128:$dst, VR128:$src1, VR128:$src, SSECC:$cc),
                "cmp${cc}pd {$src, $dst|$dst, $src}", []>;
def CMPPDrm : PDI<0xC2, MRMSrcMem, 
                (ops VR128:$dst, VR128:$src1, f128mem:$src, SSECC:$cc),
                "cmp${cc}pd {$src, $dst|$dst, $src}", []>;
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
                 [(set VR128:$dst, (vector_shuffle (v4f32 VR128:$src1), (undef),
                                    PSHUFD_shuffle_mask:$src2))]>;
def PSHUFDrm : PDIi8<0x70, MRMSrcMem,
                     (ops VR128:$dst, i128mem:$src1, i8imm:$src2),
                     "pshufd {$src2, $src1, $dst|$dst, $src1, $src2}", []>;

def SHUFPSrr : PSIi8<0xC6, MRMSrcReg, 
                     (ops VR128:$dst, VR128:$src1, VR128:$src2, i8imm:$src3),
                     "shufps {$src3, $src2, $dst|$dst, $src2, $src3}", []>;
def SHUFPSrm : PSIi8<0xC6, MRMSrcMem, 
                     (ops VR128:$dst, VR128:$src1, f128mem:$src2, i8imm:$src3),
                     "shufps {$src3, $src2, $dst|$dst, $src2, $src3}", []>;
def SHUFPDrr : PDIi8<0xC6, MRMSrcReg, 
                     (ops VR128:$dst, VR128:$src1, VR128:$src2, i8imm:$src3),
                     "shufpd {$src3, $src2, $dst|$dst, $src2, $src3}", []>;
def SHUFPDrm : PDIi8<0xC6, MRMSrcMem, 
                     (ops VR128:$dst, VR128:$src1, f128mem:$src2, i8imm:$src3),
                     "shufpd {$src3, $src2, $dst|$dst, $src2, $src3}", []>;

def UNPCKHPSrr : PSI<0x15, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpckhps {$src2, $dst|$dst, $src2}", []>;
def UNPCKHPSrm : PSI<0x15, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpckhps {$src2, $dst|$dst, $src2}", []>;
def UNPCKHPDrr : PDI<0x15, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpckhpd {$src2, $dst|$dst, $src2}", []>;
def UNPCKHPDrm : PDI<0x15, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpckhpd {$src2, $dst|$dst, $src2}", []>;
def UNPCKLPSrr : PSI<0x14, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpcklps {$src2, $dst|$dst, $src2}", []>;
def UNPCKLPSrm : PSI<0x14, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpcklps {$src2, $dst|$dst, $src2}", []>;
def UNPCKLPDrr : PDI<0x14, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpcklpd {$src2, $dst|$dst, $src2}", []>;
def UNPCKLPDrm : PDI<0x14, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpcklpd {$src2, $dst|$dst, $src2}", []>;

//===----------------------------------------------------------------------===//
// SSE integer instructions
//===----------------------------------------------------------------------===//

// Move Instructions
def MOVD128rr : PDI<0x6E, MRMSrcReg, (ops VR128:$dst, R32:$src),
                  "movd {$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v4i32 (scalar_to_vector R32:$src)))]>;
def MOVD128rm : PDI<0x6E, MRMSrcMem, (ops VR128:$dst, i32mem:$src),
                  "movd {$src, $dst|$dst, $src}", []>;
def MOVD128mr : PDI<0x7E, MRMDestMem, (ops i32mem:$dst, VR128:$src),
                  "movd {$src, $dst|$dst, $src}", []>;

// SSE2 instructions with XS prefix
def MOVQ128rr : I<0x7E, MRMSrcReg, (ops VR128:$dst, VR64:$src),
                  "movq {$src, $dst|$dst, $src}",
                  [(set VR128:$dst,
                      (v2i64 (scalar_to_vector VR64:$src)))]>, XS,
                Requires<[HasSSE2]>;
def MOVQ128rm : I<0x7E, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                  "movq {$src, $dst|$dst, $src}", []>, XS;

def MOVQ128mr : PDI<0xD6, MRMSrcMem, (ops i64mem:$dst, VR128:$src),
                  "movq {$src, $dst|$dst, $src}", []>;


//===----------------------------------------------------------------------===//
// Alias Instructions
//===----------------------------------------------------------------------===//

def FR32ToV4F32 : PSI<0x28, MRMSrcReg, (ops VR128:$dst, FR32:$src),
                      "movaps {$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4f32 (scalar_to_vector FR32:$src)))]>;

def FR64ToV2F64 : PDI<0x28, MRMSrcReg, (ops VR128:$dst, FR64:$src),
                      "movapd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v2f64 (scalar_to_vector FR64:$src)))]>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// 128-bit vector undef's.
def : Pat<(v2f64 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v16i8 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v8i16 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;

// Store 128-bit integer vector values.
def : Pat<(store (v16i8 VR128:$src), addr:$dst), (MOVAPSmr addr:$dst, VR128:$src)>;
def : Pat<(store (v8i16 VR128:$src), addr:$dst), (MOVAPSmr addr:$dst, VR128:$src)>;
def : Pat<(store (v4i32 VR128:$src), addr:$dst), (MOVAPSmr addr:$dst, VR128:$src)>;
def : Pat<(store (v2i64 VR128:$src), addr:$dst), (MOVAPSmr addr:$dst, VR128:$src)>;

// Scalar to v8i16 / v16i8. The source may be a R32, but only the lower 8 or
// 16-bits matter.
def : Pat<(v8i16 (X86s2vec R32:$src)), (MOVD128rr R32:$src)>,
  Requires<[HasSSE2]>;
def : Pat<(v16i8 (X86s2vec R32:$src)), (MOVD128rr R32:$src)>,
  Requires<[HasSSE2]>;


// Splat v4f32 / v4i32
def : Pat<(vector_shuffle (v4f32 VR128:$src), (undef), SHUFP_splat_mask:$sm),
          (v4f32 (SHUFPSrr VR128:$src, VR128:$src, SHUFP_splat_mask:$sm))>;
def : Pat<(vector_shuffle (v4i32 VR128:$src), (undef), SHUFP_splat_mask:$sm),
          (v4i32 (SHUFPSrr VR128:$src, VR128:$src, SHUFP_splat_mask:$sm))>;

// Splat v2f64 / v2i64
def : Pat<(X86unpcklp (v2f64 VR128:$src1), VR128:$src2),
          (v2f64 (UNPCKLPDrr VR128:$src1, VR128:$src2))>;
def : Pat<(X86unpcklp (v2i64 VR128:$src1), VR128:$src2),
          (v2i64 (UNPCKLPDrr VR128:$src1, VR128:$src2))>;

// Shuffle v4i32, undef
def : Pat<(vector_shuffle (v4i32 VR128:$src1), (undef),
           PSHUFD_shuffle_mask:$src2),
          (v4i32 (PSHUFDrr VR128:$src1, PSHUFD_shuffle_mask:$src2))>;
