//===-- MipsInstrFPU.td - Mips FPU Instruction Information -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Mips FPU instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Floating Point Instructions
// ------------------------
// * 64bit fp:
//    - 32 64-bit registers (default mode)
//    - 16 even 32-bit registers (32-bit compatible mode) for
//      single and double access.
// * 32bit fp:
//    - 16 even 32-bit registers - single and double (aliased)
//    - 32 32-bit registers (within single-only mode)
//===----------------------------------------------------------------------===//

// Floating Point Compare and Branch
def SDT_MipsFPBrcond : SDTypeProfile<0, 2, [SDTCisInt<0>,
                                            SDTCisVT<1, OtherVT>]>;
def SDT_MipsFPCmp : SDTypeProfile<0, 3, [SDTCisSameAs<0, 1>, SDTCisFP<1>,
                                         SDTCisVT<2, i32>]>;
def SDT_MipsCMovFP : SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>,
                                          SDTCisSameAs<1, 2>]>;
def SDT_MipsTruncIntFP : SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisFP<1>]>;
def SDT_MipsBuildPairF64 : SDTypeProfile<1, 2, [SDTCisVT<0, f64>,
                                                SDTCisVT<1, i32>,
                                                SDTCisSameAs<1, 2>]>;
def SDT_MipsExtractElementF64 : SDTypeProfile<1, 2, [SDTCisVT<0, i32>,
                                                     SDTCisVT<1, f64>,
                                                     SDTCisVT<2, i32>]>;

def MipsFPCmp : SDNode<"MipsISD::FPCmp", SDT_MipsFPCmp, [SDNPOutGlue]>;
def MipsCMovFP_T : SDNode<"MipsISD::CMovFP_T", SDT_MipsCMovFP, [SDNPInGlue]>;
def MipsCMovFP_F : SDNode<"MipsISD::CMovFP_F", SDT_MipsCMovFP, [SDNPInGlue]>;
def MipsFPBrcond : SDNode<"MipsISD::FPBrcond", SDT_MipsFPBrcond,
                          [SDNPHasChain, SDNPOptInGlue]>;
def MipsTruncIntFP : SDNode<"MipsISD::TruncIntFP", SDT_MipsTruncIntFP>;
def MipsBuildPairF64 : SDNode<"MipsISD::BuildPairF64", SDT_MipsBuildPairF64>;
def MipsExtractElementF64 : SDNode<"MipsISD::ExtractElementF64",
                                   SDT_MipsExtractElementF64>;

// Operand for printing out a condition code.
let PrintMethod = "printFCCOperand", DecoderMethod = "DecodeCondCode" in
  def condcode : Operand<i32>;

//===----------------------------------------------------------------------===//
// Feature predicates.
//===----------------------------------------------------------------------===//

def IsFP64bit        : Predicate<"Subtarget.isFP64bit()">,
                       AssemblerPredicate<"FeatureFP64Bit">;
def NotFP64bit       : Predicate<"!Subtarget.isFP64bit()">,
                       AssemblerPredicate<"!FeatureFP64Bit">;
def IsSingleFloat    : Predicate<"Subtarget.isSingleFloat()">,
                       AssemblerPredicate<"FeatureSingleFloat">;
def IsNotSingleFloat : Predicate<"!Subtarget.isSingleFloat()">,
                       AssemblerPredicate<"!FeatureSingleFloat">;

// FP immediate patterns.
def fpimm0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fpimm0neg : PatLeaf<(fpimm), [{
  return N->isExactlyValue(-0.0);
}]>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//
// A set of multiclasses is used to address the register usage.
//
// S32 - single precision in 16 32bit even fp registers
//       single precision in 32 32bit fp registers in SingleOnly mode
// S64 - single precision in 32 64bit fp registers (In64BitMode)
// D32 - double precision in 16 32bit even fp registers
// D64 - double precision in 32 64bit fp registers (In64BitMode)
//
// Only S32 and D32 are supported right now.
//===----------------------------------------------------------------------===//

class ADDS_FT<string opstr, RegisterOperand RC, InstrItinClass Itin, bit IsComm,
              SDPatternOperator OpNode= null_frag> :
  InstSE<(outs RC:$fd), (ins RC:$fs, RC:$ft),
         !strconcat(opstr, "\t$fd, $fs, $ft"),
         [(set RC:$fd, (OpNode RC:$fs, RC:$ft))], Itin, FrmFR> {
  let isCommutable = IsComm;
}

multiclass ADDS_M<string opstr, InstrItinClass Itin, bit IsComm,
                  SDPatternOperator OpNode = null_frag> {
  def _D32 : ADDS_FT<opstr, AFGR64RegsOpnd, Itin, IsComm, OpNode>,
             Requires<[NotFP64bit, HasStdEnc]>;
  def _D64 : ADDS_FT<opstr, FGR64RegsOpnd, Itin, IsComm, OpNode>,
             Requires<[IsFP64bit, HasStdEnc]> {
    string DecoderNamespace = "Mips64";
  }
}

class ABSS_FT<string opstr, RegisterOperand DstRC, RegisterOperand SrcRC,
              InstrItinClass Itin, SDPatternOperator OpNode= null_frag> :
  InstSE<(outs DstRC:$fd), (ins SrcRC:$fs), !strconcat(opstr, "\t$fd, $fs"),
         [(set DstRC:$fd, (OpNode SrcRC:$fs))], Itin, FrmFR>,
  NeverHasSideEffects;

multiclass ABSS_M<string opstr, InstrItinClass Itin,
                  SDPatternOperator OpNode= null_frag> {
  def _D32 : ABSS_FT<opstr, AFGR64RegsOpnd, AFGR64RegsOpnd, Itin, OpNode>,
             Requires<[NotFP64bit, HasStdEnc]>;
  def _D64 : ABSS_FT<opstr, FGR64RegsOpnd, FGR64RegsOpnd, Itin, OpNode>,
             Requires<[IsFP64bit, HasStdEnc]> {
    string DecoderNamespace = "Mips64";
  }
}

multiclass ROUND_M<string opstr, InstrItinClass Itin> {
  def _D32 : ABSS_FT<opstr, FGR32RegsOpnd, AFGR64RegsOpnd, Itin>,
             Requires<[NotFP64bit, HasStdEnc]>;
  def _D64 : ABSS_FT<opstr, FGR32RegsOpnd, FGR64RegsOpnd, Itin>,
             Requires<[IsFP64bit, HasStdEnc]> {
    let DecoderNamespace = "Mips64";
  }
}

class MFC1_FT<string opstr, RegisterOperand DstRC, RegisterOperand SrcRC,
              InstrItinClass Itin, SDPatternOperator OpNode= null_frag> :
  InstSE<(outs DstRC:$rt), (ins SrcRC:$fs), !strconcat(opstr, "\t$rt, $fs"),
         [(set DstRC:$rt, (OpNode SrcRC:$fs))], Itin, FrmFR>;

class MTC1_FT<string opstr, RegisterOperand DstRC, RegisterOperand SrcRC,
              InstrItinClass Itin, SDPatternOperator OpNode= null_frag> :
  InstSE<(outs DstRC:$fs), (ins SrcRC:$rt), !strconcat(opstr, "\t$rt, $fs"),
         [(set DstRC:$fs, (OpNode SrcRC:$rt))], Itin, FrmFR>;

class MFC1_FT_CCR<string opstr, RegisterOperand DstRC, RegisterOperand SrcRC,
              InstrItinClass Itin, SDPatternOperator OpNode= null_frag> :
  InstSE<(outs DstRC:$rt), (ins SrcRC:$fs), !strconcat(opstr, "\t$rt, $fs"),
         [(set DstRC:$rt, (OpNode SrcRC:$fs))], Itin, FrmFR>;

class MTC1_FT_CCR<string opstr, RegisterOperand DstRC, RegisterOperand SrcRC,
              InstrItinClass Itin, SDPatternOperator OpNode= null_frag> :
  InstSE<(outs DstRC:$fs), (ins SrcRC:$rt), !strconcat(opstr, "\t$rt, $fs"),
         [(set DstRC:$fs, (OpNode SrcRC:$rt))], Itin, FrmFR>;

class LW_FT<string opstr, RegisterOperand RC, InstrItinClass Itin,
            Operand MemOpnd, SDPatternOperator OpNode= null_frag> :
  InstSE<(outs RC:$rt), (ins MemOpnd:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(set RC:$rt, (OpNode addrDefault:$addr))], Itin, FrmFI> {
  let DecoderMethod = "DecodeFMem";
  let mayLoad = 1;
}

class SW_FT<string opstr, RegisterOperand RC, InstrItinClass Itin,
            Operand MemOpnd, SDPatternOperator OpNode= null_frag> :
  InstSE<(outs), (ins RC:$rt, MemOpnd:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(OpNode RC:$rt, addrDefault:$addr)], Itin, FrmFI> {
  let DecoderMethod = "DecodeFMem";
  let mayStore = 1;
}

class MADDS_FT<string opstr, RegisterOperand RC, InstrItinClass Itin,
               SDPatternOperator OpNode = null_frag> :
  InstSE<(outs RC:$fd), (ins RC:$fr, RC:$fs, RC:$ft),
         !strconcat(opstr, "\t$fd, $fr, $fs, $ft"),
         [(set RC:$fd, (OpNode (fmul RC:$fs, RC:$ft), RC:$fr))], Itin, FrmFR>;

class NMADDS_FT<string opstr, RegisterOperand RC, InstrItinClass Itin,
                SDPatternOperator OpNode = null_frag> :
  InstSE<(outs RC:$fd), (ins RC:$fr, RC:$fs, RC:$ft),
         !strconcat(opstr, "\t$fd, $fr, $fs, $ft"),
         [(set RC:$fd, (fsub fpimm0, (OpNode (fmul RC:$fs, RC:$ft), RC:$fr)))],
         Itin, FrmFR>;

class LWXC1_FT<string opstr, RegisterOperand DRC, RegisterOperand PRC,
               InstrItinClass Itin, SDPatternOperator OpNode = null_frag> :
  InstSE<(outs DRC:$fd), (ins PRC:$base, PRC:$index),
         !strconcat(opstr, "\t$fd, ${index}(${base})"),
         [(set DRC:$fd, (OpNode (add PRC:$base, PRC:$index)))], Itin, FrmFI> {
  let AddedComplexity = 20;
}

class SWXC1_FT<string opstr, RegisterOperand DRC, RegisterOperand PRC,
               InstrItinClass Itin, SDPatternOperator OpNode = null_frag> :
  InstSE<(outs), (ins DRC:$fs, PRC:$base, PRC:$index),
         !strconcat(opstr, "\t$fs, ${index}(${base})"),
         [(OpNode DRC:$fs, (add PRC:$base, PRC:$index))], Itin, FrmFI> {
  let AddedComplexity = 20;
}

class BC1F_FT<string opstr, InstrItinClass Itin,
              SDPatternOperator Op = null_frag>  :
  InstSE<(outs), (ins brtarget:$offset), !strconcat(opstr, "\t$offset"),
         [(MipsFPBrcond Op, bb:$offset)], Itin, FrmFI> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
  let Uses = [FCR31];
}

class CEQS_FT<string typestr, RegisterClass RC, InstrItinClass Itin,
              SDPatternOperator OpNode = null_frag>  :
  InstSE<(outs), (ins RC:$fs, RC:$ft, condcode:$cond),
         !strconcat("c.$cond.", typestr, "\t$fs, $ft"),
         [(OpNode RC:$fs, RC:$ft, imm:$cond)], Itin, FrmFR> {
  let Defs = [FCR31];
  let isCodeGenOnly = 1;
}

class C_COND_FT<string CondStr, string Typestr, RegisterOperand RC>  :
   InstSE<(outs), (ins RC:$fs, RC:$ft),
          !strconcat("c.", CondStr, ".", Typestr, "\t$fs, $ft"), [], IIFcmp,
          FrmFR>;

multiclass C_COND_M<string TypeStr, RegisterOperand RC, bits<5> fmt> {
  def C_F_#NAME : C_COND_FT<"f", TypeStr, RC>, C_COND_FM<fmt, 0>;
  def C_UN_#NAME : C_COND_FT<"un", TypeStr, RC>, C_COND_FM<fmt, 1>;
  def C_EQ_#NAME : C_COND_FT<"eq", TypeStr, RC>, C_COND_FM<fmt, 2>;
  def C_UEQ_#NAME : C_COND_FT<"ueq", TypeStr, RC>, C_COND_FM<fmt, 3>;
  def C_OLT_#NAME : C_COND_FT<"olt", TypeStr, RC>, C_COND_FM<fmt, 4>;
  def C_ULT_#NAME : C_COND_FT<"ult", TypeStr, RC>, C_COND_FM<fmt, 5>;
  def C_OLE_#NAME : C_COND_FT<"ole", TypeStr, RC>, C_COND_FM<fmt, 6>;
  def C_ULE_#NAME : C_COND_FT<"ule", TypeStr, RC>, C_COND_FM<fmt, 7>;
  def C_SF_#NAME : C_COND_FT<"sf", TypeStr, RC>, C_COND_FM<fmt, 8>;
  def C_NGLE_#NAME : C_COND_FT<"ngle", TypeStr, RC>, C_COND_FM<fmt, 9>;
  def C_SEQ_#NAME : C_COND_FT<"seq", TypeStr, RC>, C_COND_FM<fmt, 10>;
  def C_NGL_#NAME : C_COND_FT<"ngl", TypeStr, RC>, C_COND_FM<fmt, 11>;
  def C_LT_#NAME : C_COND_FT<"lt", TypeStr, RC>, C_COND_FM<fmt, 12>;
  def C_NGE_#NAME : C_COND_FT<"nge", TypeStr, RC>, C_COND_FM<fmt, 13>;
  def C_LE_#NAME : C_COND_FT<"le", TypeStr, RC>, C_COND_FM<fmt, 14>;
  def C_NGT_#NAME : C_COND_FT<"ngt", TypeStr, RC>, C_COND_FM<fmt, 15>;
}

defm S : C_COND_M<"s", FGR32RegsOpnd, 16>;
defm D32 : C_COND_M<"d", AFGR64RegsOpnd, 17>,
                    Requires<[NotFP64bit, HasStdEnc]>;
let DecoderNamespace = "Mips64" in
defm D64 : C_COND_M<"d", FGR64RegsOpnd, 17>, Requires<[IsFP64bit, HasStdEnc]>;

//===----------------------------------------------------------------------===//
// Floating Point Instructions
//===----------------------------------------------------------------------===//
def ROUND_W_S  : ABSS_FT<"round.w.s", FGR32RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                 ABSS_FM<0xc, 16>;
def TRUNC_W_S  : ABSS_FT<"trunc.w.s", FGR32RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                 ABSS_FM<0xd, 16>;
def CEIL_W_S   : ABSS_FT<"ceil.w.s", FGR32RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                 ABSS_FM<0xe, 16>;
def FLOOR_W_S  : ABSS_FT<"floor.w.s", FGR32RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                 ABSS_FM<0xf, 16>;
def CVT_W_S    : ABSS_FT<"cvt.w.s", FGR32RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                 ABSS_FM<0x24, 16>;

defm ROUND_W : ROUND_M<"round.w.d", IIFcvt>, ABSS_FM<0xc, 17>;
defm TRUNC_W : ROUND_M<"trunc.w.d", IIFcvt>, ABSS_FM<0xd, 17>;
defm CEIL_W  : ROUND_M<"ceil.w.d", IIFcvt>, ABSS_FM<0xe, 17>;
defm FLOOR_W : ROUND_M<"floor.w.d", IIFcvt>, ABSS_FM<0xf, 17>;
defm CVT_W   : ROUND_M<"cvt.w.d", IIFcvt>, ABSS_FM<0x24, 17>;

let Predicates = [IsFP64bit, HasStdEnc], DecoderNamespace = "Mips64" in {
  def ROUND_L_S : ABSS_FT<"round.l.s", FGR64RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                  ABSS_FM<0x8, 16>;
  def ROUND_L_D64 : ABSS_FT<"round.l.d", FGR64RegsOpnd, FGR64RegsOpnd, IIFcvt>,
                    ABSS_FM<0x8, 17>;
  def TRUNC_L_S : ABSS_FT<"trunc.l.s", FGR64RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                  ABSS_FM<0x9, 16>;
  def TRUNC_L_D64 : ABSS_FT<"trunc.l.d", FGR64RegsOpnd, FGR64RegsOpnd, IIFcvt>,
                    ABSS_FM<0x9, 17>;
  def CEIL_L_S  : ABSS_FT<"ceil.l.s", FGR64RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                  ABSS_FM<0xa, 16>;
  def CEIL_L_D64 : ABSS_FT<"ceil.l.d", FGR64RegsOpnd, FGR64RegsOpnd, IIFcvt>,
                   ABSS_FM<0xa, 17>;
  def FLOOR_L_S : ABSS_FT<"floor.l.s", FGR64RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                  ABSS_FM<0xb, 16>;
  def FLOOR_L_D64 : ABSS_FT<"floor.l.d", FGR64RegsOpnd, FGR64RegsOpnd, IIFcvt>,
                    ABSS_FM<0xb, 17>;
}

def CVT_S_W : ABSS_FT<"cvt.s.w", FGR32RegsOpnd, FGR32RegsOpnd, IIFcvt>,
              ABSS_FM<0x20, 20>;
def CVT_L_S : ABSS_FT<"cvt.l.s", FGR64RegsOpnd, FGR32RegsOpnd, IIFcvt>,
              ABSS_FM<0x25, 16>;
def CVT_L_D64: ABSS_FT<"cvt.l.d", FGR64RegsOpnd, FGR64RegsOpnd, IIFcvt>,
               ABSS_FM<0x25, 17>;

let Predicates = [NotFP64bit, HasStdEnc] in {
  def CVT_S_D32 : ABSS_FT<"cvt.s.d", FGR32RegsOpnd, AFGR64RegsOpnd, IIFcvt>,
                  ABSS_FM<0x20, 17>;
  def CVT_D32_W : ABSS_FT<"cvt.d.w", AFGR64RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                  ABSS_FM<0x21, 20>;
  def CVT_D32_S : ABSS_FT<"cvt.d.s", AFGR64RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                  ABSS_FM<0x21, 16>;
}

let Predicates = [IsFP64bit, HasStdEnc], DecoderNamespace = "Mips64" in {
  def CVT_S_D64 : ABSS_FT<"cvt.s.d", FGR32RegsOpnd, FGR64RegsOpnd, IIFcvt>,
                  ABSS_FM<0x20, 17>;
  def CVT_S_L   : ABSS_FT<"cvt.s.l", FGR32RegsOpnd, FGR64RegsOpnd, IIFcvt>,
                  ABSS_FM<0x20, 21>;
  def CVT_D64_W : ABSS_FT<"cvt.d.w", FGR64RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                  ABSS_FM<0x21, 20>;
  def CVT_D64_S : ABSS_FT<"cvt.d.s", FGR64RegsOpnd, FGR32RegsOpnd, IIFcvt>,
                  ABSS_FM<0x21, 16>;
  def CVT_D64_L : ABSS_FT<"cvt.d.l", FGR64RegsOpnd, FGR64RegsOpnd, IIFcvt>,
                  ABSS_FM<0x21, 21>;
}

let isPseudo = 1, isCodeGenOnly = 1 in {
  def PseudoCVT_S_W : ABSS_FT<"", FGR32RegsOpnd, CPURegsOpnd, IIFcvt>;
  def PseudoCVT_D32_W : ABSS_FT<"", AFGR64RegsOpnd, CPURegsOpnd, IIFcvt>;
  def PseudoCVT_S_L : ABSS_FT<"", FGR64RegsOpnd, CPU64RegsOpnd, IIFcvt>;
  def PseudoCVT_D64_W : ABSS_FT<"", FGR64RegsOpnd, CPURegsOpnd, IIFcvt>;
  def PseudoCVT_D64_L : ABSS_FT<"", FGR64RegsOpnd, CPU64RegsOpnd, IIFcvt>;
}

let Predicates = [NoNaNsFPMath, HasStdEnc] in {
  def FABS_S : ABSS_FT<"abs.s", FGR32RegsOpnd, FGR32RegsOpnd, IIFcvt, fabs>,
               ABSS_FM<0x5, 16>;
  def FNEG_S : ABSS_FT<"neg.s", FGR32RegsOpnd, FGR32RegsOpnd, IIFcvt, fneg>,
               ABSS_FM<0x7, 16>;
  defm FABS : ABSS_M<"abs.d", IIFcvt, fabs>, ABSS_FM<0x5, 17>;
  defm FNEG : ABSS_M<"neg.d", IIFcvt, fneg>, ABSS_FM<0x7, 17>;
}

def  FSQRT_S : ABSS_FT<"sqrt.s", FGR32RegsOpnd, FGR32RegsOpnd, IIFsqrtSingle,
               fsqrt>, ABSS_FM<0x4, 16>;
defm FSQRT : ABSS_M<"sqrt.d", IIFsqrtDouble, fsqrt>, ABSS_FM<0x4, 17>;

// The odd-numbered registers are only referenced when doing loads,
// stores, and moves between floating-point and integer registers.
// When defining instructions, we reference all 32-bit registers,
// regardless of register aliasing.

/// Move Control Registers From/To CPU Registers
def CFC1 : MFC1_FT_CCR<"cfc1", CPURegsOpnd, CCROpnd, IIFmove>, MFC1_FM<2>;
def CTC1 : MTC1_FT_CCR<"ctc1", CCROpnd, CPURegsOpnd, IIFmove>, MFC1_FM<6>;
def MFC1 : MFC1_FT<"mfc1", CPURegsOpnd, FGR32RegsOpnd, IIFmoveC1, bitconvert>,
           MFC1_FM<0>;
def MTC1 : MTC1_FT<"mtc1", FGR32RegsOpnd, CPURegsOpnd, IIFmoveC1, bitconvert>,
           MFC1_FM<4>;
def DMFC1 : MFC1_FT<"dmfc1", CPU64RegsOpnd, FGR64RegsOpnd, IIFmoveC1,
            bitconvert>, MFC1_FM<1>;
def DMTC1 : MTC1_FT<"dmtc1", FGR64RegsOpnd, CPU64RegsOpnd, IIFmoveC1,
            bitconvert>, MFC1_FM<5>;

def FMOV_S   : ABSS_FT<"mov.s", FGR32RegsOpnd, FGR32RegsOpnd, IIFmove>,
               ABSS_FM<0x6, 16>;
def FMOV_D32 : ABSS_FT<"mov.d", AFGR64RegsOpnd, AFGR64RegsOpnd, IIFmove>,
               ABSS_FM<0x6, 17>, Requires<[NotFP64bit, HasStdEnc]>;
def FMOV_D64 : ABSS_FT<"mov.d", FGR64RegsOpnd, FGR64RegsOpnd, IIFmove>,
               ABSS_FM<0x6, 17>, Requires<[IsFP64bit, HasStdEnc]> {
                 let DecoderNamespace = "Mips64";
}

/// Floating Point Memory Instructions
let Predicates = [IsN64, HasStdEnc], DecoderNamespace = "Mips64" in {
  def LWC1_P8 : LW_FT<"lwc1", FGR32RegsOpnd, IIFLoad, mem64, load>,
                LW_FM<0x31>;
  def SWC1_P8 : SW_FT<"swc1", FGR32RegsOpnd, IIFStore, mem64, store>,
                LW_FM<0x39>;
  def LDC164_P8 : LW_FT<"ldc1", FGR64RegsOpnd, IIFLoad, mem64, load>,
                  LW_FM<0x35> {
    let isCodeGenOnly =1;
  }
  def SDC164_P8 : SW_FT<"sdc1", FGR64RegsOpnd, IIFStore, mem64, store>,
                  LW_FM<0x3d> {
    let isCodeGenOnly =1;
  }
}

let Predicates = [NotN64, HasStdEnc] in {
  def LWC1 : LW_FT<"lwc1", FGR32RegsOpnd, IIFLoad, mem, load>, LW_FM<0x31>;
  def SWC1 : SW_FT<"swc1", FGR32RegsOpnd, IIFStore, mem, store>, LW_FM<0x39>;
}

let Predicates = [NotN64, HasMips64, HasStdEnc],
  DecoderNamespace = "Mips64" in {
  def LDC164 : LW_FT<"ldc1", FGR64RegsOpnd, IIFLoad, mem, load>, LW_FM<0x35>;
  def SDC164 : SW_FT<"sdc1", FGR64RegsOpnd, IIFStore, mem, store>, LW_FM<0x3d>;
}

let Predicates = [NotN64, NotMips64, HasStdEnc] in {
  let isPseudo = 1, isCodeGenOnly = 1 in {
    def PseudoLDC1 : LW_FT<"", AFGR64RegsOpnd, IIFLoad, mem, load>;
    def PseudoSDC1 : SW_FT<"", AFGR64RegsOpnd, IIFStore, mem, store>;
  }
  def LDC1 : LW_FT<"ldc1", AFGR64RegsOpnd, IIFLoad, mem>, LW_FM<0x35>;
  def SDC1 : SW_FT<"sdc1", AFGR64RegsOpnd, IIFStore, mem>, LW_FM<0x3d>;
}

// Indexed loads and stores.
let Predicates = [HasFPIdx, HasStdEnc] in {
  def LWXC1 : LWXC1_FT<"lwxc1", FGR32RegsOpnd, CPURegsOpnd, IIFLoad, load>,
              LWXC1_FM<0>;
  def SWXC1 : SWXC1_FT<"swxc1", FGR32RegsOpnd, CPURegsOpnd, IIFStore, store>,
              SWXC1_FM<8>;
}

let Predicates = [HasMips32r2, NotMips64, HasStdEnc] in {
  def LDXC1 : LWXC1_FT<"ldxc1", AFGR64RegsOpnd, CPURegsOpnd, IIFLoad, load>,
              LWXC1_FM<1>;
  def SDXC1 : SWXC1_FT<"sdxc1", AFGR64RegsOpnd, CPURegsOpnd, IIFStore, store>,
              SWXC1_FM<9>;
}

let Predicates = [HasMips64, NotN64, HasStdEnc], DecoderNamespace="Mips64" in {
  def LDXC164 : LWXC1_FT<"ldxc1", FGR64RegsOpnd, CPURegsOpnd, IIFLoad, load>,
                LWXC1_FM<1>;
  def SDXC164 : SWXC1_FT<"sdxc1", FGR64RegsOpnd, CPURegsOpnd, IIFStore, store>,
                SWXC1_FM<9>;
}

// n64
let Predicates = [IsN64, HasStdEnc], isCodeGenOnly=1 in {
  def LWXC1_P8 : LWXC1_FT<"lwxc1", FGR32RegsOpnd, CPU64RegsOpnd, IIFLoad, load>,
                 LWXC1_FM<0>;
  def LDXC164_P8 : LWXC1_FT<"ldxc1", FGR64RegsOpnd, CPU64RegsOpnd, IIFLoad,
                             load>, LWXC1_FM<1>;
  def SWXC1_P8 : SWXC1_FT<"swxc1", FGR32RegsOpnd, CPU64RegsOpnd, IIFStore,
                          store>, SWXC1_FM<8>;
  def SDXC164_P8 : SWXC1_FT<"sdxc1", FGR64RegsOpnd, CPU64RegsOpnd, IIFStore,
                            store>, SWXC1_FM<9>;
}

// Load/store doubleword indexed unaligned.
let Predicates = [NotMips64, HasStdEnc] in {
  def LUXC1 : LWXC1_FT<"luxc1", AFGR64RegsOpnd, CPURegsOpnd, IIFLoad>,
              LWXC1_FM<0x5>;
  def SUXC1 : SWXC1_FT<"suxc1", AFGR64RegsOpnd, CPURegsOpnd, IIFStore>,
              SWXC1_FM<0xd>;
}

let Predicates = [HasMips64, HasStdEnc],
  DecoderNamespace="Mips64" in {
  def LUXC164 : LWXC1_FT<"luxc1", FGR64RegsOpnd, CPURegsOpnd, IIFLoad>,
                LWXC1_FM<0x5>;
  def SUXC164 : SWXC1_FT<"suxc1", FGR64RegsOpnd, CPURegsOpnd, IIFStore>,
                SWXC1_FM<0xd>;
}

/// Floating-point Aritmetic
def FADD_S : ADDS_FT<"add.s", FGR32RegsOpnd, IIFadd, 1, fadd>,
             ADDS_FM<0x00, 16>;
defm FADD :  ADDS_M<"add.d", IIFadd, 1, fadd>, ADDS_FM<0x00, 17>;
def FDIV_S : ADDS_FT<"div.s", FGR32RegsOpnd, IIFdivSingle, 0, fdiv>,
             ADDS_FM<0x03, 16>;
defm FDIV :  ADDS_M<"div.d", IIFdivDouble, 0, fdiv>, ADDS_FM<0x03, 17>;
def FMUL_S : ADDS_FT<"mul.s", FGR32RegsOpnd, IIFmulSingle, 1, fmul>,
             ADDS_FM<0x02, 16>;
defm FMUL :  ADDS_M<"mul.d", IIFmulDouble, 1, fmul>, ADDS_FM<0x02, 17>;
def FSUB_S : ADDS_FT<"sub.s", FGR32RegsOpnd, IIFadd, 0, fsub>,
             ADDS_FM<0x01, 16>;
defm FSUB :  ADDS_M<"sub.d", IIFadd, 0, fsub>, ADDS_FM<0x01, 17>;

let Predicates = [HasMips32r2, HasStdEnc] in {
  def MADD_S : MADDS_FT<"madd.s", FGR32RegsOpnd, IIFmulSingle, fadd>,
               MADDS_FM<4, 0>;
  def MSUB_S : MADDS_FT<"msub.s", FGR32RegsOpnd, IIFmulSingle, fsub>,
               MADDS_FM<5, 0>;
}

let Predicates = [HasMips32r2, NoNaNsFPMath, HasStdEnc] in {
  def NMADD_S : NMADDS_FT<"nmadd.s", FGR32RegsOpnd, IIFmulSingle, fadd>,
                MADDS_FM<6, 0>;
  def NMSUB_S : NMADDS_FT<"nmsub.s", FGR32RegsOpnd, IIFmulSingle, fsub>,
                MADDS_FM<7, 0>;
}

let Predicates = [HasMips32r2, NotFP64bit, HasStdEnc] in {
  def MADD_D32 : MADDS_FT<"madd.d", AFGR64RegsOpnd, IIFmulDouble, fadd>,
                 MADDS_FM<4, 1>;
  def MSUB_D32 : MADDS_FT<"msub.d", AFGR64RegsOpnd, IIFmulDouble, fsub>,
                 MADDS_FM<5, 1>;
}

let Predicates = [HasMips32r2, NotFP64bit, NoNaNsFPMath, HasStdEnc] in {
  def NMADD_D32 : NMADDS_FT<"nmadd.d", AFGR64RegsOpnd, IIFmulDouble, fadd>,
                  MADDS_FM<6, 1>;
  def NMSUB_D32 : NMADDS_FT<"nmsub.d", AFGR64RegsOpnd, IIFmulDouble, fsub>,
                  MADDS_FM<7, 1>;
}

let Predicates = [HasMips32r2, IsFP64bit, HasStdEnc], isCodeGenOnly=1 in {
  def MADD_D64 : MADDS_FT<"madd.d", FGR64RegsOpnd, IIFmulDouble, fadd>,
                 MADDS_FM<4, 1>;
  def MSUB_D64 : MADDS_FT<"msub.d", FGR64RegsOpnd, IIFmulDouble, fsub>,
                 MADDS_FM<5, 1>;
}

let Predicates = [HasMips32r2, IsFP64bit, NoNaNsFPMath, HasStdEnc],
    isCodeGenOnly=1 in {
  def NMADD_D64 : NMADDS_FT<"nmadd.d", FGR64RegsOpnd, IIFmulDouble, fadd>,
                  MADDS_FM<6, 1>;
  def NMSUB_D64 : NMADDS_FT<"nmsub.d", FGR64RegsOpnd, IIFmulDouble, fsub>,
                  MADDS_FM<7, 1>;
}

//===----------------------------------------------------------------------===//
// Floating Point Branch Codes
//===----------------------------------------------------------------------===//
// Mips branch codes. These correspond to condcode in MipsInstrInfo.h.
// They must be kept in synch.
def MIPS_BRANCH_F  : PatLeaf<(i32 0)>;
def MIPS_BRANCH_T  : PatLeaf<(i32 1)>;

let DecoderMethod = "DecodeBC1" in {
def BC1F : BC1F_FT<"bc1f", IIBranch, MIPS_BRANCH_F>, BC1F_FM<0, 0>;
def BC1T : BC1F_FT<"bc1t", IIBranch, MIPS_BRANCH_T>, BC1F_FM<0, 1>;
}
//===----------------------------------------------------------------------===//
// Floating Point Flag Conditions
//===----------------------------------------------------------------------===//
// Mips condition codes. They must correspond to condcode in MipsInstrInfo.h.
// They must be kept in synch.
def MIPS_FCOND_F    : PatLeaf<(i32 0)>;
def MIPS_FCOND_UN   : PatLeaf<(i32 1)>;
def MIPS_FCOND_OEQ  : PatLeaf<(i32 2)>;
def MIPS_FCOND_UEQ  : PatLeaf<(i32 3)>;
def MIPS_FCOND_OLT  : PatLeaf<(i32 4)>;
def MIPS_FCOND_ULT  : PatLeaf<(i32 5)>;
def MIPS_FCOND_OLE  : PatLeaf<(i32 6)>;
def MIPS_FCOND_ULE  : PatLeaf<(i32 7)>;
def MIPS_FCOND_SF   : PatLeaf<(i32 8)>;
def MIPS_FCOND_NGLE : PatLeaf<(i32 9)>;
def MIPS_FCOND_SEQ  : PatLeaf<(i32 10)>;
def MIPS_FCOND_NGL  : PatLeaf<(i32 11)>;
def MIPS_FCOND_LT   : PatLeaf<(i32 12)>;
def MIPS_FCOND_NGE  : PatLeaf<(i32 13)>;
def MIPS_FCOND_LE   : PatLeaf<(i32 14)>;
def MIPS_FCOND_NGT  : PatLeaf<(i32 15)>;

/// Floating Point Compare
def FCMP_S32 : CEQS_FT<"s", FGR32, IIFcmp, MipsFPCmp>, CEQS_FM<16>;
def FCMP_D32 : CEQS_FT<"d", AFGR64, IIFcmp, MipsFPCmp>, CEQS_FM<17>,
               Requires<[NotFP64bit, HasStdEnc]>;
let DecoderNamespace = "Mips64" in
def FCMP_D64 : CEQS_FT<"d", FGR64, IIFcmp, MipsFPCmp>, CEQS_FM<17>,
               Requires<[IsFP64bit, HasStdEnc]>;

//===----------------------------------------------------------------------===//
// Floating Point Pseudo-Instructions
//===----------------------------------------------------------------------===//
def MOVCCRToCCR : PseudoSE<(outs CCR:$dst), (ins CCROpnd:$src), []>;

// This pseudo instr gets expanded into 2 mtc1 instrs after register
// allocation.
def BuildPairF64 :
  PseudoSE<(outs AFGR64RegsOpnd:$dst),
           (ins CPURegsOpnd:$lo, CPURegsOpnd:$hi),
           [(set AFGR64RegsOpnd:$dst,
            (MipsBuildPairF64 CPURegsOpnd:$lo, CPURegsOpnd:$hi))]>;

// This pseudo instr gets expanded into 2 mfc1 instrs after register
// allocation.
// if n is 0, lower part of src is extracted.
// if n is 1, higher part of src is extracted.
def ExtractElementF64 :
  PseudoSE<(outs CPURegsOpnd:$dst), (ins AFGR64RegsOpnd:$src, i32imm:$n),
           [(set CPURegsOpnd:$dst,
            (MipsExtractElementF64 AFGR64RegsOpnd:$src, imm:$n))]>;

//===----------------------------------------------------------------------===//
// Floating Point Patterns
//===----------------------------------------------------------------------===//
def : MipsPat<(f32 fpimm0), (MTC1 ZERO)>;
def : MipsPat<(f32 fpimm0neg), (FNEG_S (MTC1 ZERO))>;

def : MipsPat<(f32 (sint_to_fp CPURegsOpnd:$src)),
              (PseudoCVT_S_W CPURegsOpnd:$src)>;
def : MipsPat<(MipsTruncIntFP FGR32RegsOpnd:$src),
              (TRUNC_W_S FGR32RegsOpnd:$src)>;

let Predicates = [NotFP64bit, HasStdEnc] in {
  def : MipsPat<(f64 (sint_to_fp CPURegsOpnd:$src)),
                (PseudoCVT_D32_W CPURegsOpnd:$src)>;
  def : MipsPat<(MipsTruncIntFP AFGR64RegsOpnd:$src),
                (TRUNC_W_D32 AFGR64RegsOpnd:$src)>;
  def : MipsPat<(f32 (fround AFGR64RegsOpnd:$src)),
                (CVT_S_D32 AFGR64RegsOpnd:$src)>;
  def : MipsPat<(f64 (fextend FGR32RegsOpnd:$src)),
                (CVT_D32_S FGR32RegsOpnd:$src)>;
}

let Predicates = [IsFP64bit, HasStdEnc] in {
  def : MipsPat<(f64 fpimm0), (DMTC1 ZERO_64)>;
  def : MipsPat<(f64 fpimm0neg), (FNEG_D64 (DMTC1 ZERO_64))>;

  def : MipsPat<(f64 (sint_to_fp CPURegsOpnd:$src)),
                (PseudoCVT_D64_W CPURegsOpnd:$src)>;
  def : MipsPat<(f32 (sint_to_fp CPU64RegsOpnd:$src)),
                (EXTRACT_SUBREG (PseudoCVT_S_L CPU64RegsOpnd:$src), sub_32)>;
  def : MipsPat<(f64 (sint_to_fp CPU64RegsOpnd:$src)),
                (PseudoCVT_D64_L CPU64RegsOpnd:$src)>;

  def : MipsPat<(MipsTruncIntFP FGR64RegsOpnd:$src),
                (TRUNC_W_D64 FGR64RegsOpnd:$src)>;
  def : MipsPat<(MipsTruncIntFP FGR32RegsOpnd:$src),
                (TRUNC_L_S FGR32RegsOpnd:$src)>;
  def : MipsPat<(MipsTruncIntFP FGR64RegsOpnd:$src),
                (TRUNC_L_D64 FGR64RegsOpnd:$src)>;

  def : MipsPat<(f32 (fround FGR64RegsOpnd:$src)),
                (CVT_S_D64 FGR64RegsOpnd:$src)>;
  def : MipsPat<(f64 (fextend FGR32RegsOpnd:$src)),
                (CVT_D64_S FGR32RegsOpnd:$src)>;
}

// Patterns for loads/stores with a reg+imm operand.
let AddedComplexity = 40 in {
  let Predicates = [IsN64, HasStdEnc] in {
    def : LoadRegImmPat<LWC1_P8, f32, load>;
    def : StoreRegImmPat<SWC1_P8, f32>;
    def : LoadRegImmPat<LDC164_P8, f64, load>;
    def : StoreRegImmPat<SDC164_P8, f64>;
  }

  let Predicates = [NotN64, HasStdEnc] in {
    def : LoadRegImmPat<LWC1, f32, load>;
    def : StoreRegImmPat<SWC1, f32>;
  }

  let Predicates = [NotN64, HasMips64, HasStdEnc] in {
    def : LoadRegImmPat<LDC164, f64, load>;
    def : StoreRegImmPat<SDC164, f64>;
  }

  let Predicates = [NotN64, NotMips64, HasStdEnc] in {
    def : LoadRegImmPat<PseudoLDC1, f64, load>;
    def : StoreRegImmPat<PseudoSDC1, f64>;
  }
}
