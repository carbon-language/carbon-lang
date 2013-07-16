//===-- MipsCondMov.td - Describe Mips Conditional Moves --*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the Conditional Moves implementation.
//
//===----------------------------------------------------------------------===//

// Conditional moves:
// These instructions are expanded in
// MipsISelLowering::EmitInstrWithCustomInserter if target does not have
// conditional move instructions.
// cond:int, data:int
class CMov_I_I_FT<string opstr, RegisterOperand CRC, RegisterOperand DRC,
                  InstrItinClass Itin> :
  InstSE<(outs DRC:$rd), (ins DRC:$rs, CRC:$rt, DRC:$F),
         !strconcat(opstr, "\t$rd, $rs, $rt"), [], Itin, FrmFR> {
  let Constraints = "$F = $rd";
}

// cond:int, data:float
class CMov_I_F_FT<string opstr, RegisterOperand CRC, RegisterOperand DRC,
                  InstrItinClass Itin> :
  InstSE<(outs DRC:$fd), (ins DRC:$fs, CRC:$rt, DRC:$F),
         !strconcat(opstr, "\t$fd, $fs, $rt"), [], Itin, FrmFR> {
  let Constraints = "$F = $fd";
}

// cond:float, data:int
class CMov_F_I_FT<string opstr, RegisterOperand RC, InstrItinClass Itin,
                  SDPatternOperator OpNode = null_frag> :
  InstSE<(outs RC:$rd), (ins RC:$rs, RC:$F),
         !strconcat(opstr, "\t$rd, $rs, $$fcc0"),
         [(set RC:$rd, (OpNode RC:$rs, RC:$F))], Itin, FrmFR> {
  let Uses = [FCR31];
  let Constraints = "$F = $rd";
}

// cond:float, data:float
class CMov_F_F_FT<string opstr, RegisterClass RC, InstrItinClass Itin,
                  SDPatternOperator OpNode = null_frag> :
  InstSE<(outs RC:$fd), (ins RC:$fs, RC:$F),
         !strconcat(opstr, "\t$fd, $fs, $$fcc0"),
         [(set RC:$fd, (OpNode RC:$fs, RC:$F))], Itin, FrmFR> {
  let Uses = [FCR31];
  let Constraints = "$F = $fd";
}

// select patterns
multiclass MovzPats0<RegisterClass CRC, RegisterClass DRC,
                     Instruction MOVZInst, Instruction SLTOp,
                     Instruction SLTuOp, Instruction SLTiOp,
                     Instruction SLTiuOp> {
  def : MipsPat<(select (i32 (setge CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (SLTOp CRC:$lhs, CRC:$rhs), DRC:$F)>;
  def : MipsPat<(select (i32 (setuge CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (SLTuOp CRC:$lhs, CRC:$rhs), DRC:$F)>;
  def : MipsPat<(select (i32 (setge CRC:$lhs, immSExt16:$rhs)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (SLTiOp CRC:$lhs, immSExt16:$rhs), DRC:$F)>;
  def : MipsPat<(select (i32 (setuge CRC:$lh, immSExt16:$rh)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (SLTiuOp CRC:$lh, immSExt16:$rh), DRC:$F)>;
  def : MipsPat<(select (i32 (setle CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (SLTOp CRC:$rhs, CRC:$lhs), DRC:$F)>;
  def : MipsPat<(select (i32 (setule CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (SLTuOp CRC:$rhs, CRC:$lhs), DRC:$F)>;
  def : MipsPat<(select (i32 (setgt CRC:$lhs, immSExt16Plus1:$rhs)),
                        DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (SLTiOp CRC:$lhs, (Plus1 imm:$rhs)), DRC:$F)>;
  def : MipsPat<(select (i32 (setugt CRC:$lhs, immSExt16Plus1:$rhs)),
                        DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (SLTiuOp CRC:$lhs, (Plus1 imm:$rhs)),
                          DRC:$F)>;
}

multiclass MovzPats1<RegisterClass CRC, RegisterClass DRC,
                     Instruction MOVZInst, Instruction XOROp> {
  def : MipsPat<(select (i32 (seteq CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (XOROp CRC:$lhs, CRC:$rhs), DRC:$F)>;
  def : MipsPat<(select (i32 (seteq CRC:$lhs, 0)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, CRC:$lhs, DRC:$F)>;
}

multiclass MovzPats2<RegisterClass CRC, RegisterClass DRC,
                     Instruction MOVZInst, Instruction XORiOp> {
  def : MipsPat<
            (select (i32 (seteq CRC:$lhs, immZExt16:$uimm16)), DRC:$T, DRC:$F),
            (MOVZInst DRC:$T, (XORiOp CRC:$lhs, immZExt16:$uimm16), DRC:$F)>;
}

multiclass MovnPats<RegisterClass CRC, RegisterClass DRC, Instruction MOVNInst,
                    Instruction XOROp> {
  def : MipsPat<(select (i32 (setne CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
                (MOVNInst DRC:$T, (XOROp CRC:$lhs, CRC:$rhs), DRC:$F)>;
  def : MipsPat<(select CRC:$cond, DRC:$T, DRC:$F),
                (MOVNInst DRC:$T, CRC:$cond, DRC:$F)>;
  def : MipsPat<(select (i32 (setne CRC:$lhs, 0)),DRC:$T, DRC:$F),
                (MOVNInst DRC:$T, CRC:$lhs, DRC:$F)>;
}

// Instantiation of instructions.
def MOVZ_I_I : CMov_I_I_FT<"movz", CPURegsOpnd, CPURegsOpnd, NoItinerary>,
               ADD_FM<0, 0xa>;
let Predicates = [HasStdEnc],
                  DecoderNamespace = "Mips64" in {
  def MOVZ_I_I64   : CMov_I_I_FT<"movz", CPURegsOpnd, CPU64RegsOpnd,
                                  NoItinerary>, ADD_FM<0, 0xa>;
  def MOVZ_I64_I   : CMov_I_I_FT<"movz", CPU64RegsOpnd, CPURegsOpnd,
                                  NoItinerary>, ADD_FM<0, 0xa> {
    let isCodeGenOnly = 1;
  }
  def MOVZ_I64_I64 : CMov_I_I_FT<"movz", CPU64RegsOpnd, CPU64RegsOpnd,
                                  NoItinerary>, ADD_FM<0, 0xa> {
    let isCodeGenOnly = 1;
  }
}

def MOVN_I_I       : CMov_I_I_FT<"movn", CPURegsOpnd, CPURegsOpnd,
                                  NoItinerary>, ADD_FM<0, 0xb>;
let Predicates = [HasStdEnc],
                  DecoderNamespace = "Mips64" in {
  def MOVN_I_I64   : CMov_I_I_FT<"movn", CPURegsOpnd, CPU64RegsOpnd,
                                  NoItinerary>, ADD_FM<0, 0xb>;
  def MOVN_I64_I   : CMov_I_I_FT<"movn", CPU64RegsOpnd, CPURegsOpnd,
                                  NoItinerary>, ADD_FM<0, 0xb> {
    let isCodeGenOnly = 1;
  }
  def MOVN_I64_I64 : CMov_I_I_FT<"movn", CPU64RegsOpnd, CPU64RegsOpnd,
                                  NoItinerary>, ADD_FM<0, 0xb> {
    let isCodeGenOnly = 1;
  }
}

def MOVZ_I_S : CMov_I_F_FT<"movz.s", CPURegsOpnd, FGR32RegsOpnd, IIFmove>,
               CMov_I_F_FM<18, 16>;
def MOVZ_I64_S : CMov_I_F_FT<"movz.s", CPU64RegsOpnd, FGR32RegsOpnd, IIFmove>,
                 CMov_I_F_FM<18, 16>, Requires<[HasMips64, HasStdEnc]> {
  let DecoderNamespace = "Mips64";
}

def MOVN_I_S : CMov_I_F_FT<"movn.s", CPURegsOpnd, FGR32RegsOpnd, IIFmove>,
               CMov_I_F_FM<19, 16>;
def MOVN_I64_S : CMov_I_F_FT<"movn.s", CPU64RegsOpnd, FGR32RegsOpnd, IIFmove>,
                 CMov_I_F_FM<19, 16>, Requires<[HasMips64, HasStdEnc]> {
  let DecoderNamespace = "Mips64";
}

let Predicates = [NotFP64bit, HasStdEnc] in {
  def MOVZ_I_D32 : CMov_I_F_FT<"movz.d", CPURegsOpnd, AFGR64RegsOpnd, IIFmove>,
                   CMov_I_F_FM<18, 17>;
  def MOVN_I_D32 : CMov_I_F_FT<"movn.d", CPURegsOpnd, AFGR64RegsOpnd, IIFmove>,
                   CMov_I_F_FM<19, 17>;
}
let Predicates = [IsFP64bit, HasStdEnc],
                  DecoderNamespace = "Mips64" in {
  def MOVZ_I_D64 : CMov_I_F_FT<"movz.d", CPURegsOpnd, FGR64RegsOpnd, IIFmove>,
                   CMov_I_F_FM<18, 17>;
  def MOVZ_I64_D64 : CMov_I_F_FT<"movz.d", CPU64RegsOpnd, FGR64RegsOpnd,
                                  IIFmove>, CMov_I_F_FM<18, 17> {
    let isCodeGenOnly = 1;
  }
  def MOVN_I_D64 : CMov_I_F_FT<"movn.d", CPURegsOpnd, FGR64RegsOpnd, IIFmove>,
                   CMov_I_F_FM<19, 17>;
  def MOVN_I64_D64 : CMov_I_F_FT<"movn.d", CPU64RegsOpnd, FGR64RegsOpnd,
                                  IIFmove>, CMov_I_F_FM<19, 17> {
    let isCodeGenOnly = 1;
  }
}

def MOVT_I : CMov_F_I_FT<"movt", CPURegsOpnd, IIAlu, MipsCMovFP_T>,
             CMov_F_I_FM<1>;
def MOVT_I64 : CMov_F_I_FT<"movt", CPU64RegsOpnd, IIAlu, MipsCMovFP_T>,
               CMov_F_I_FM<1>, Requires<[HasMips64, HasStdEnc]> {
  let DecoderNamespace = "Mips64";
}

def MOVF_I : CMov_F_I_FT<"movf", CPURegsOpnd, IIAlu, MipsCMovFP_F>,
             CMov_F_I_FM<0>;
def MOVF_I64 : CMov_F_I_FT<"movf", CPU64RegsOpnd, IIAlu, MipsCMovFP_F>,
               CMov_F_I_FM<0>, Requires<[HasMips64, HasStdEnc]> {
  let DecoderNamespace = "Mips64";
}

def MOVT_S : CMov_F_F_FT<"movt.s", FGR32, IIFmove, MipsCMovFP_T>,
             CMov_F_F_FM<16, 1>;
def MOVF_S : CMov_F_F_FT<"movf.s", FGR32, IIFmove, MipsCMovFP_F>,
             CMov_F_F_FM<16, 0>;

let Predicates = [NotFP64bit, HasStdEnc] in {
  def MOVT_D32 : CMov_F_F_FT<"movt.d", AFGR64, IIFmove, MipsCMovFP_T>,
                 CMov_F_F_FM<17, 1>;
  def MOVF_D32 : CMov_F_F_FT<"movf.d", AFGR64, IIFmove, MipsCMovFP_F>,
                 CMov_F_F_FM<17, 0>;
}
let Predicates = [IsFP64bit, HasStdEnc],
    DecoderNamespace = "Mips64" in {
  def MOVT_D64 : CMov_F_F_FT<"movt.d", FGR64, IIFmove, MipsCMovFP_T>,
                 CMov_F_F_FM<17, 1>;
  def MOVF_D64 : CMov_F_F_FT<"movf.d", FGR64, IIFmove, MipsCMovFP_F>,
                 CMov_F_F_FM<17, 0>;
}

// Instantiation of conditional move patterns.
defm : MovzPats0<CPURegs, CPURegs, MOVZ_I_I, SLT, SLTu, SLTi, SLTiu>;
defm : MovzPats1<CPURegs, CPURegs, MOVZ_I_I, XOR>;
defm : MovzPats2<CPURegs, CPURegs, MOVZ_I_I, XORi>;
let Predicates = [HasMips64, HasStdEnc] in {
  defm : MovzPats0<CPURegs, CPU64Regs, MOVZ_I_I64, SLT, SLTu, SLTi, SLTiu>;
  defm : MovzPats0<CPU64Regs, CPURegs, MOVZ_I_I, SLT64, SLTu64, SLTi64,
                   SLTiu64>;
  defm : MovzPats0<CPU64Regs, CPU64Regs, MOVZ_I_I64, SLT64, SLTu64, SLTi64,
                   SLTiu64>;
  defm : MovzPats1<CPURegs, CPU64Regs, MOVZ_I_I64, XOR>;
  defm : MovzPats1<CPU64Regs, CPURegs, MOVZ_I64_I, XOR64>;
  defm : MovzPats1<CPU64Regs, CPU64Regs, MOVZ_I64_I64, XOR64>;
  defm : MovzPats2<CPURegs, CPU64Regs, MOVZ_I_I64, XORi>;
  defm : MovzPats2<CPU64Regs, CPURegs, MOVZ_I64_I, XORi64>;
  defm : MovzPats2<CPU64Regs, CPU64Regs, MOVZ_I64_I64, XORi64>;
}

defm : MovnPats<CPURegs, CPURegs, MOVN_I_I, XOR>;
let Predicates = [HasMips64, HasStdEnc] in {
  defm : MovnPats<CPURegs, CPU64Regs, MOVN_I_I64, XOR>;
  defm : MovnPats<CPU64Regs, CPURegs, MOVN_I64_I, XOR64>;
  defm : MovnPats<CPU64Regs, CPU64Regs, MOVN_I64_I64, XOR64>;
}

defm : MovzPats0<CPURegs, FGR32, MOVZ_I_S, SLT, SLTu, SLTi, SLTiu>;
defm : MovzPats1<CPURegs, FGR32, MOVZ_I_S, XOR>;
defm : MovnPats<CPURegs, FGR32, MOVN_I_S, XOR>;
let Predicates = [HasMips64, HasStdEnc] in {
  defm : MovzPats0<CPU64Regs, FGR32, MOVZ_I_S, SLT64, SLTu64, SLTi64,
                   SLTiu64>;
  defm : MovzPats1<CPU64Regs, FGR32, MOVZ_I64_S, XOR64>;
  defm : MovnPats<CPU64Regs, FGR32, MOVN_I64_S, XOR64>;
}

let Predicates = [NotFP64bit, HasStdEnc] in {
  defm : MovzPats0<CPURegs, AFGR64, MOVZ_I_D32, SLT, SLTu, SLTi, SLTiu>;
  defm : MovzPats1<CPURegs, AFGR64, MOVZ_I_D32, XOR>;
  defm : MovnPats<CPURegs, AFGR64, MOVN_I_D32, XOR>;
}
let Predicates = [IsFP64bit, HasStdEnc] in {
  defm : MovzPats0<CPURegs, FGR64, MOVZ_I_D64, SLT, SLTu, SLTi, SLTiu>;
  defm : MovzPats0<CPU64Regs, FGR64, MOVZ_I_D64, SLT64, SLTu64, SLTi64,
                   SLTiu64>;
  defm : MovzPats1<CPURegs, FGR64, MOVZ_I_D64, XOR>;
  defm : MovzPats1<CPU64Regs, FGR64, MOVZ_I64_D64, XOR64>;
  defm : MovnPats<CPURegs, FGR64, MOVN_I_D64, XOR>;
  defm : MovnPats<CPU64Regs, FGR64, MOVN_I64_D64, XOR64>;
}
