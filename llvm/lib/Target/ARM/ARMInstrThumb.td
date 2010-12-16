//===- ARMInstrThumb.td - Thumb support for ARM ------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Thumb instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Thumb specific DAG Nodes.
//

def ARMtcall : SDNode<"ARMISD::tCALL", SDT_ARMcall,
                      [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag,
                       SDNPVariadic]>;

def imm_neg_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(-(int)N->getZExtValue(), MVT::i32);
}]>;
def imm_comp_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(~((uint32_t)N->getZExtValue()), MVT::i32);
}]>;

/// imm0_7 predicate - True if the 32-bit immediate is in the range [0,7].
def imm0_7 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() < 8;
}]>;
def imm0_7_neg : PatLeaf<(i32 imm), [{
  return (uint32_t)-N->getZExtValue() < 8;
}], imm_neg_XFORM>;

def imm0_255 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() < 256;
}]>;
def imm0_255_comp : PatLeaf<(i32 imm), [{
  return ~((uint32_t)N->getZExtValue()) < 256;
}]>;

def imm8_255 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() >= 8 && (uint32_t)N->getZExtValue() < 256;
}]>;
def imm8_255_neg : PatLeaf<(i32 imm), [{
  unsigned Val = -N->getZExtValue();
  return Val >= 8 && Val < 256;
}], imm_neg_XFORM>;

// Break imm's up into two pieces: an immediate + a left shift. This uses
// thumb_immshifted to match and thumb_immshifted_val and thumb_immshifted_shamt
// to get the val/shift pieces.
def thumb_immshifted : PatLeaf<(imm), [{
  return ARM_AM::isThumbImmShiftedVal((unsigned)N->getZExtValue());
}]>;

def thumb_immshifted_val : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getThumbImmNonShiftedVal((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

def thumb_immshifted_shamt : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getThumbImmValShift((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

// ADR instruction labels.
def t_adrlabel : Operand<i32> {
  let EncoderMethod = "getThumbAdrLabelOpValue";
}

// Scaled 4 immediate.
def t_imm_s4 : Operand<i32> {
  let PrintMethod = "printThumbS4ImmOperand";
}

// Define Thumb specific addressing modes.

def t_brtarget : Operand<OtherVT> {
  let EncoderMethod = "getThumbBRTargetOpValue";
}

def t_bcctarget : Operand<i32> {
  let EncoderMethod = "getThumbBCCTargetOpValue";
}

def t_cbtarget : Operand<i32> {
  let EncoderMethod = "getThumbCBTargetOpValue";
}

def t_bltarget : Operand<i32> {
  let EncoderMethod = "getThumbBLTargetOpValue";
}

def t_blxtarget : Operand<i32> {
  let EncoderMethod = "getThumbBLXTargetOpValue";
}

def MemModeRegThumbAsmOperand : AsmOperandClass {
  let Name = "MemModeRegThumb";
  let SuperClasses = [];
}

def MemModeImmThumbAsmOperand : AsmOperandClass {
  let Name = "MemModeImmThumb";
  let SuperClasses = [];
}

// t_addrmode_rr := reg + reg
//
def t_addrmode_rr : Operand<i32>,
                    ComplexPattern<i32, 2, "SelectThumbAddrModeRR", []> {
  let EncoderMethod = "getThumbAddrModeRegRegOpValue";
  let PrintMethod = "printThumbAddrModeRROperand";
  let MIOperandInfo = (ops tGPR:$base, tGPR:$offsreg);
}

// t_addrmode_rrs := reg + reg
//
def t_addrmode_rrs1 : Operand<i32>,
                      ComplexPattern<i32, 2, "SelectThumbAddrModeRI5S1", []> {
  let EncoderMethod = "getThumbAddrModeRegRegOpValue";
  let PrintMethod = "printThumbAddrModeRROperand";
  let MIOperandInfo = (ops tGPR:$base, tGPR:$offsreg);
  let ParserMatchClass = MemModeRegThumbAsmOperand;
}
def t_addrmode_rrs2 : Operand<i32>,
                      ComplexPattern<i32, 2, "SelectThumbAddrModeRI5S2", []> {
  let EncoderMethod = "getThumbAddrModeRegRegOpValue";
  let PrintMethod = "printThumbAddrModeRROperand";
  let MIOperandInfo = (ops tGPR:$base, tGPR:$offsreg);
  let ParserMatchClass = MemModeRegThumbAsmOperand;
}
def t_addrmode_rrs4 : Operand<i32>,
                      ComplexPattern<i32, 2, "SelectThumbAddrModeRI5S4", []> {
  let EncoderMethod = "getThumbAddrModeRegRegOpValue";
  let PrintMethod = "printThumbAddrModeRROperand";
  let MIOperandInfo = (ops tGPR:$base, tGPR:$offsreg);
  let ParserMatchClass = MemModeRegThumbAsmOperand;
}

// t_addrmode_is4 := reg + imm5 * 4
//
def t_addrmode_is4 : Operand<i32>,
                     ComplexPattern<i32, 2, "SelectThumbAddrModeImm5S4", []> {
  let EncoderMethod = "getAddrModeISOpValue";
  let PrintMethod = "printThumbAddrModeImm5S4Operand";
  let MIOperandInfo = (ops tGPR:$base, i32imm:$offsimm);
  let ParserMatchClass = MemModeImmThumbAsmOperand;
}

// t_addrmode_is2 := reg + imm5 * 2
//
def t_addrmode_is2 : Operand<i32>,
                     ComplexPattern<i32, 2, "SelectThumbAddrModeImm5S2", []> {
  let EncoderMethod = "getAddrModeISOpValue";
  let PrintMethod = "printThumbAddrModeImm5S2Operand";
  let MIOperandInfo = (ops tGPR:$base, i32imm:$offsimm);
  let ParserMatchClass = MemModeImmThumbAsmOperand;
}

// t_addrmode_is1 := reg + imm5
//
def t_addrmode_is1 : Operand<i32>,
                     ComplexPattern<i32, 2, "SelectThumbAddrModeImm5S1", []> {
  let EncoderMethod = "getAddrModeISOpValue";
  let PrintMethod = "printThumbAddrModeImm5S1Operand";
  let MIOperandInfo = (ops tGPR:$base, i32imm:$offsimm);
  let ParserMatchClass = MemModeImmThumbAsmOperand;
}

// t_addrmode_sp := sp + imm8 * 4
//
def t_addrmode_sp : Operand<i32>,
                    ComplexPattern<i32, 2, "SelectThumbAddrModeSP", []> {
  let EncoderMethod = "getAddrModeThumbSPOpValue";
  let PrintMethod = "printThumbAddrModeSPOperand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
  let ParserMatchClass = MemModeImmThumbAsmOperand;
}

// t_addrmode_pc := <label> => pc + imm8 * 4
//
def t_addrmode_pc : Operand<i32> {
  let EncoderMethod = "getAddrModePCOpValue";
  let ParserMatchClass = MemModeImmThumbAsmOperand;
}

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

// FIXME: Marking these as hasSideEffects is necessary to prevent machine DCE
// from removing one half of the matched pairs. That breaks PEI, which assumes
// these will always be in pairs, and asserts if it finds otherwise. Better way?
let Defs = [SP], Uses = [SP], hasSideEffects = 1 in {
def tADJCALLSTACKUP :
  PseudoInst<(outs), (ins i32imm:$amt1, i32imm:$amt2), NoItinerary,
             [(ARMcallseq_end imm:$amt1, imm:$amt2)]>,
            Requires<[IsThumb, IsThumb1Only]>;

def tADJCALLSTACKDOWN :
  PseudoInst<(outs), (ins i32imm:$amt), NoItinerary,
             [(ARMcallseq_start imm:$amt)]>,
            Requires<[IsThumb, IsThumb1Only]>;
}

// T1Disassembly - A simple class to make encoding some disassembly patterns
// easier and less verbose.
class T1Disassembly<bits<2> op1, bits<8> op2>
  : T1Encoding<0b101111> {
  let Inst{9-8} = op1;
  let Inst{7-0} = op2;
}

def tNOP : T1pI<(outs), (ins), NoItinerary, "nop", "",
                [/* For disassembly only; pattern left blank */]>,
           T1Disassembly<0b11, 0x00>; // A8.6.110

def tYIELD : T1pI<(outs), (ins), NoItinerary, "yield", "",
                  [/* For disassembly only; pattern left blank */]>,
           T1Disassembly<0b11, 0x10>; // A8.6.410

def tWFE : T1pI<(outs), (ins), NoItinerary, "wfe", "",
                [/* For disassembly only; pattern left blank */]>,
           T1Disassembly<0b11, 0x20>; // A8.6.408

def tWFI : T1pI<(outs), (ins), NoItinerary, "wfi", "",
                [/* For disassembly only; pattern left blank */]>,
           T1Disassembly<0b11, 0x30>; // A8.6.409

def tSEV : T1pI<(outs), (ins), NoItinerary, "sev", "",
                [/* For disassembly only; pattern left blank */]>,
           T1Disassembly<0b11, 0x40>; // A8.6.157

// The i32imm operand $val can be used by a debugger to store more information
// about the breakpoint.
def tBKPT : T1I<(outs), (ins i32imm:$val), NoItinerary, "bkpt\t$val",
                [/* For disassembly only; pattern left blank */]>,
           T1Disassembly<0b10, {?,?,?,?,?,?,?,?}> {
  // A8.6.22
  bits<8> val;
  let Inst{7-0} = val;
}

def tSETENDBE : T1I<(outs), (ins), NoItinerary, "setend\tbe",
                    [/* For disassembly only; pattern left blank */]>,
                T1Encoding<0b101101> {
  // A8.6.156
  let Inst{9-5} = 0b10010;
  let Inst{4}   = 1;
  let Inst{3}   = 1;            // Big-Endian
  let Inst{2-0} = 0b000;
}

def tSETENDLE : T1I<(outs), (ins), NoItinerary, "setend\tle",
                    [/* For disassembly only; pattern left blank */]>,
                T1Encoding<0b101101> {
  // A8.6.156
  let Inst{9-5} = 0b10010;
  let Inst{4}   = 1;
  let Inst{3}   = 0;            // Little-Endian
  let Inst{2-0} = 0b000;
}

// Change Processor State is a system instruction -- for disassembly only.
// The singleton $opt operand contains the following information:
// 
//   opt{4-0} = mode ==> don't care
//   opt{5} = changemode ==> 0 (false for 16-bit Thumb instr)
//   opt{8-6} = AIF from Inst{2-0}
//   opt{10-9} = 1:imod from Inst{4} with 0b10 as enable and 0b11 as disable
//
// The opt{4-0} and opt{5} sub-fields are to accommodate 32-bit Thumb and ARM
// CPS which has more options.
def tCPS : T1I<(outs), (ins cps_opt:$opt), NoItinerary, "cps$opt",
              [/* For disassembly only; pattern left blank */]>,
           T1Misc<0b0110011> {
  // A8.6.38 & B6.1.1
  let Inst{3} = 0;
  // FIXME: Finish encoding.
}

// For both thumb1 and thumb2.
let isNotDuplicable = 1, isCodeGenOnly = 1 in
def tPICADD : TIt<(outs GPR:$dst), (ins GPR:$lhs, pclabel:$cp), IIC_iALUr, "",
                  [(set GPR:$dst, (ARMpic_add GPR:$lhs, imm:$cp))]>,
              T1Special<{0,0,?,?}> {
  // A8.6.6
  bits<3> dst;
  let Inst{6-3} = 0b1111; // Rm = pc
  let Inst{2-0} = dst;
}

// PC relative add (ADR).
def tADDrPCi : T1I<(outs tGPR:$dst), (ins t_imm_s4:$rhs), IIC_iALUi,
                   "add\t$dst, pc, $rhs", []>,
               T1Encoding<{1,0,1,0,0,?}> {
  // A6.2 & A8.6.10
  bits<3> dst;
  bits<8> rhs;
  let Inst{10-8} = dst;
  let Inst{7-0}  = rhs;
}

// ADD <Rd>, sp, #<imm8>
// This is rematerializable, which is particularly useful for taking the
// address of locals.
let isReMaterializable = 1 in
def tADDrSPi : T1I<(outs tGPR:$dst), (ins GPR:$sp, t_imm_s4:$rhs), IIC_iALUi,
                   "add\t$dst, $sp, $rhs", []>,
               T1Encoding<{1,0,1,0,1,?}> {
  // A6.2 & A8.6.8
  bits<3> dst;
  bits<8> rhs;
  let Inst{10-8} = dst;
  let Inst{7-0}  = rhs;
}

// ADD sp, sp, #<imm7>
def tADDspi : TIt<(outs GPR:$dst), (ins GPR:$lhs, t_imm_s4:$rhs), IIC_iALUi,
                  "add\t$dst, $rhs", []>,
              T1Misc<{0,0,0,0,0,?,?}> {
  // A6.2.5 & A8.6.8
  bits<7> rhs;
  let Inst{6-0} = rhs;
}

// SUB sp, sp, #<imm7>
// FIXME: The encoding and the ASM string don't match up.
def tSUBspi : TIt<(outs GPR:$dst), (ins GPR:$lhs, t_imm_s4:$rhs), IIC_iALUi,
                  "sub\t$dst, $rhs", []>,
              T1Misc<{0,0,0,0,1,?,?}> {
  // A6.2.5 & A8.6.214
  bits<7> rhs;
  let Inst{6-0} = rhs;
}

// ADD <Rm>, sp
def tADDrSP : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALUr,
                  "add\t$dst, $rhs", []>,
              T1Special<{0,0,?,?}> {
  // A8.6.9 Encoding T1
  bits<4> dst;
  let Inst{7}   = dst{3};
  let Inst{6-3} = 0b1101;
  let Inst{2-0} = dst{2-0};
}

// ADD sp, <Rm>
def tADDspr : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALUr,
                  "add\t$dst, $rhs", []>,
              T1Special<{0,0,?,?}> {
  // A8.6.9 Encoding T2
  bits<4> dst;
  let Inst{7} = 1;
  let Inst{6-3} = dst;
  let Inst{2-0} = 0b101;
}

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
  def tBX_RET : TI<(outs), (ins), IIC_Br, "bx\tlr",
                   [(ARMretflag)]>,
                T1Special<{1,1,0,?}> {
    // A6.2.3 & A8.6.25
    let Inst{6-3} = 0b1110; // Rm = lr
    let Inst{2-0} = 0b000;
  }

  // Alternative return instruction used by vararg functions.
  def tBX_RET_vararg : TI<(outs), (ins tGPR:$Rm),
                          IIC_Br, "bx\t$Rm",
                          []>,
                       T1Special<{1,1,0,?}> {
    // A6.2.3 & A8.6.25
    bits<4> Rm;
    let Inst{6-3} = Rm;
    let Inst{2-0} = 0b000;
  }
}

// Indirect branches
let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1 in {
  def tBRIND : TI<(outs), (ins GPR:$Rm),
                  IIC_Br,
                  "mov\tpc, $Rm",
                  [(brind GPR:$Rm)]>,
               T1Special<{1,0,?,?}> {
    // A8.6.97
    bits<4> Rm;
    let Inst{7}   = 1;          // <Rd> = Inst{7:2-0} = pc
    let Inst{6-3} = Rm;
    let Inst{2-0} = 0b111;
  }
}

// FIXME: remove when we have a way to marking a MI with these properties.
let isReturn = 1, isTerminator = 1, isBarrier = 1, mayLoad = 1,
    hasExtraDefRegAllocReq = 1 in
def tPOP_RET : T1I<(outs), (ins pred:$p, reglist:$regs, variable_ops),
                   IIC_iPop_Br,
                   "pop${p}\t$regs", []>,
               T1Misc<{1,1,0,?,?,?,?}> {
  // A8.6.121
  bits<16> regs;
  let Inst{8}   = regs{15};     // registers = P:'0000000':register_list
  let Inst{7-0} = regs{7-0};
}

// All calls clobber the non-callee saved registers. SP is marked as a use to
// prevent stack-pointer assignments that appear immediately before calls from
// potentially appearing dead.
let isCall = 1,
  // On non-Darwin platforms R9 is callee-saved.
  Defs = [R0,  R1,  R2,  R3,  R12, LR,
          D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
          D16, D17, D18, D19, D20, D21, D22, D23,
          D24, D25, D26, D27, D28, D29, D30, D31, CPSR, FPSCR],
  Uses = [SP] in {
  // Also used for Thumb2
  def tBL  : TIx2<0b11110, 0b11, 1,
                  (outs), (ins t_bltarget:$func, variable_ops), IIC_Br,
                  "bl\t$func",
                  [(ARMtcall tglobaladdr:$func)]>,
             Requires<[IsThumb, IsNotDarwin]> {
    bits<21> func;
    let Inst{25-16} = func{20-11};
    let Inst{13} = 1;
    let Inst{11} = 1;
    let Inst{10-0} = func{10-0};
  }

  // ARMv5T and above, also used for Thumb2
  def tBLXi : TIx2<0b11110, 0b11, 0,
                   (outs), (ins t_blxtarget:$func, variable_ops), IIC_Br,
                   "blx\t$func",
                   [(ARMcall tglobaladdr:$func)]>,
              Requires<[IsThumb, HasV5T, IsNotDarwin]> {
    bits<21> func;
    let Inst{25-16} = func{20-11};
    let Inst{13} = 1;
    let Inst{11} = 1;
    let Inst{10-1} = func{10-1};
    let Inst{0} = 0; // func{0} is assumed zero
  }

  // Also used for Thumb2
  def tBLXr : TI<(outs), (ins GPR:$func, variable_ops), IIC_Br,
                  "blx\t$func",
                  [(ARMtcall GPR:$func)]>,
              Requires<[IsThumb, HasV5T, IsNotDarwin]>,
              T1Special<{1,1,1,?}>; // A6.2.3 & A8.6.24;

  // ARMv4T
  // FIXME: Should be a pseudo.
  let isCodeGenOnly = 1 in
  def tBX : TIx2<{?,?,?,?,?}, {?,?}, ?,
                  (outs), (ins tGPR:$func, variable_ops), IIC_Br,
                  "mov\tlr, pc\n\tbx\t$func",
                  [(ARMcall_nolink tGPR:$func)]>,
            Requires<[IsThumb, IsThumb1Only, IsNotDarwin]>;
}

let isCall = 1,
  // On Darwin R9 is call-clobbered.
  // R7 is marked as a use to prevent frame-pointer assignments from being
  // moved above / below calls.
  Defs = [R0,  R1,  R2,  R3,  R9,  R12, LR,
          D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
          D16, D17, D18, D19, D20, D21, D22, D23,
          D24, D25, D26, D27, D28, D29, D30, D31, CPSR, FPSCR],
  Uses = [R7, SP] in {
  // Also used for Thumb2
  def tBLr9 : TIx2<0b11110, 0b11, 1,
                   (outs), (ins pred:$p, t_bltarget:$func, variable_ops),
                   IIC_Br, "bl${p}\t$func",
                   [(ARMtcall tglobaladdr:$func)]>,
              Requires<[IsThumb, IsDarwin]> {
    bits<21> func;
    let Inst{25-16} = func{20-11};
    let Inst{13} = 1;
    let Inst{11} = 1;
    let Inst{10-0} = func{10-0};
  }

  // ARMv5T and above, also used for Thumb2
  def tBLXi_r9 : TIx2<0b11110, 0b11, 0,
                      (outs), (ins pred:$p, t_blxtarget:$func, variable_ops),
                      IIC_Br, "blx${p}\t$func",
                      [(ARMcall tglobaladdr:$func)]>,
                 Requires<[IsThumb, HasV5T, IsDarwin]> {
    bits<21> func;
    let Inst{25-16} = func{20-11};
    let Inst{13} = 1;
    let Inst{11} = 1;
    let Inst{10-1} = func{10-1};
    let Inst{0} = 0; // func{0} is assumed zero
  }

  // Also used for Thumb2
  def tBLXr_r9 : TI<(outs), (ins pred:$p, GPR:$func, variable_ops), IIC_Br,
                    "blx${p}\t$func",
                    [(ARMtcall GPR:$func)]>,
                 Requires<[IsThumb, HasV5T, IsDarwin]>,
                 T1Special<{1,1,1,?}> {
    // A6.2.3 & A8.6.24
    bits<4> func;
    let Inst{6-3} = func;
    let Inst{2-0} = 0b000;
  }

  // ARMv4T
  let isCodeGenOnly = 1 in
  // FIXME: Should be a pseudo.
  def tBXr9 : TIx2<{?,?,?,?,?}, {?,?}, ?,
                   (outs), (ins tGPR:$func, variable_ops), IIC_Br,
                   "mov\tlr, pc\n\tbx\t$func",
                   [(ARMcall_nolink tGPR:$func)]>,
              Requires<[IsThumb, IsThumb1Only, IsDarwin]>;
}

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
  let isPredicable = 1 in
  def tB   : T1I<(outs), (ins t_brtarget:$target), IIC_Br,
                 "b\t$target", [(br bb:$target)]>,
             T1Encoding<{1,1,1,0,0,?}> {
    bits<11> target;
    let Inst{10-0} = target;
  }

  // Far jump
  // FIXME: Encoding. This should probably be a pseudo for tBL
  let Defs = [LR] in
  def tBfar : TIx2<0b11110, 0b11, 1, (outs), (ins brtarget:$target), IIC_Br,
                    "bl\t$target",[]>;

  def tBR_JTr : tPseudoInst<(outs),
                      (ins tGPR:$target, i32imm:$jt, i32imm:$id),
                      Size2Bytes, IIC_Br,
                      [(ARMbrjt tGPR:$target, tjumptable:$jt, imm:$id)]> {
    list<Predicate> Predicates = [IsThumb, IsThumb1Only];
  }
}

// FIXME: should be able to write a pattern for ARMBrcond, but can't use
// a two-value operand where a dag node expects two operands. :(
let isBranch = 1, isTerminator = 1 in
  def tBcc : T1I<(outs), (ins t_bcctarget:$target, pred:$p), IIC_Br,
                 "b${p}\t$target",
                 [/*(ARMbrcond bb:$target, imm:$cc)*/]>,
             T1Encoding<{1,1,0,1,?,?}> {
  bits<4> p;
  bits<8> target;
  let Inst{11-8} = p;
  let Inst{7-0} = target;
}

// Compare and branch on zero / non-zero
let isBranch = 1, isTerminator = 1 in {
  def tCBZ  : T1I<(outs), (ins tGPR:$Rn, t_cbtarget:$target), IIC_Br,
                  "cbz\t$Rn, $target", []>,
              T1Misc<{0,0,?,1,?,?,?}> {
    // A8.6.27
    bits<6> target;
    bits<3> Rn;
    let Inst{9}   = target{5};
    let Inst{7-3} = target{4-0};
    let Inst{2-0} = Rn;
  }

  def tCBNZ : T1I<(outs), (ins tGPR:$cmp, t_cbtarget:$target), IIC_Br,
                  "cbnz\t$cmp, $target", []>,
              T1Misc<{1,0,?,1,?,?,?}> {
    // A8.6.27
    bits<6> target;
    bits<3> Rn;
    let Inst{9}   = target{5};
    let Inst{7-3} = target{4-0};
    let Inst{2-0} = Rn;
  }
}

// A8.6.218 Supervisor Call (Software Interrupt) -- for disassembly only
// A8.6.16 B: Encoding T1
// If Inst{11-8} == 0b1111 then SEE SVC
let isCall = 1, Uses = [SP] in
def tSVC : T1pI<(outs), (ins i32imm:$imm), IIC_Br,
                "svc", "\t$imm", []>, Encoding16 {
  bits<8> imm;
  let Inst{15-12} = 0b1101;
  let Inst{11-8}  = 0b1111;
  let Inst{7-0}   = imm;
}

// The assembler uses 0xDEFE for a trap instruction.
let isBarrier = 1, isTerminator = 1 in
def tTRAP : TI<(outs), (ins), IIC_Br, 
               "trap", [(trap)]>, Encoding16 {
  let Inst = 0xdefe;
}

//===----------------------------------------------------------------------===//
//  Load Store Instructions.
//

// Loads: reg/reg and reg/imm5
let canFoldAsLoad = 1, isReMaterializable = 1 in
multiclass thumb_ld_rr_ri_enc<bits<3> reg_opc, bits<4> imm_opc,
                              Operand AddrMode_r, Operand AddrMode_i,
                              AddrMode am, InstrItinClass itin_r,
                              InstrItinClass itin_i, string asm,
                              PatFrag opnode> {
  def r : // reg/reg
    T1pILdStEncode<reg_opc,
                   (outs tGPR:$Rt), (ins AddrMode_r:$addr),
                   am, itin_r, asm, "\t$Rt, $addr",
                   [(set tGPR:$Rt, (opnode AddrMode_r:$addr))]>;
  def i : // reg/imm5
    T1pILdStEncodeImm<imm_opc, 1 /* Load */,
                      (outs tGPR:$Rt), (ins AddrMode_i:$addr),
                      am, itin_i, asm, "\t$Rt, $addr",
                      [(set tGPR:$Rt, (opnode AddrMode_i:$addr))]>;
}
// Stores: reg/reg and reg/imm5
multiclass thumb_st_rr_ri_enc<bits<3> reg_opc, bits<4> imm_opc,
                              Operand AddrMode_r, Operand AddrMode_i,
                              AddrMode am, InstrItinClass itin_r,
                              InstrItinClass itin_i, string asm,
                              PatFrag opnode> {
  def r : // reg/reg
    T1pILdStEncode<reg_opc,
                   (outs), (ins tGPR:$Rt, AddrMode_r:$addr),
                   am, itin_r, asm, "\t$Rt, $addr",
                   [(opnode tGPR:$Rt, AddrMode_r:$addr)]>;
  def i : // reg/imm5
    T1pILdStEncodeImm<imm_opc, 0 /* Store */,
                      (outs), (ins tGPR:$Rt, AddrMode_i:$addr),
                      am, itin_i, asm, "\t$Rt, $addr",
                      [(opnode tGPR:$Rt, AddrMode_i:$addr)]>;
}

// A8.6.57 & A8.6.60
defm tLDR  : thumb_ld_rr_ri_enc<0b100, 0b0110, t_addrmode_rrs4,
                                t_addrmode_is4, AddrModeT1_4,
                                IIC_iLoad_r, IIC_iLoad_i, "ldr",
                                UnOpFrag<(load node:$Src)>>;

// A8.6.64 & A8.6.61
defm tLDRB : thumb_ld_rr_ri_enc<0b110, 0b0111, t_addrmode_rrs1,
                                t_addrmode_is1, AddrModeT1_1,
                                IIC_iLoad_bh_r, IIC_iLoad_bh_i, "ldrb",
                                UnOpFrag<(zextloadi8 node:$Src)>>;

// A8.6.76 & A8.6.73
defm tLDRH : thumb_ld_rr_ri_enc<0b101, 0b1000, t_addrmode_rrs2,
                                t_addrmode_is2, AddrModeT1_2,
                                IIC_iLoad_bh_r, IIC_iLoad_bh_i, "ldrh",
                                UnOpFrag<(zextloadi16 node:$Src)>>;

let AddedComplexity = 10 in
def tLDRSB :                    // A8.6.80
  T1pILdStEncode<0b011, (outs tGPR:$dst), (ins t_addrmode_rr:$addr),
                 AddrModeT1_1, IIC_iLoad_bh_r,
                 "ldrsb", "\t$dst, $addr",
                 [(set tGPR:$dst, (sextloadi8 t_addrmode_rr:$addr))]>;

let AddedComplexity = 10 in
def tLDRSH :                    // A8.6.84
  T1pILdStEncode<0b111, (outs tGPR:$dst), (ins t_addrmode_rr:$addr),
                 AddrModeT1_2, IIC_iLoad_bh_r,
                 "ldrsh", "\t$dst, $addr",
                 [(set tGPR:$dst, (sextloadi16 t_addrmode_rr:$addr))]>;

let canFoldAsLoad = 1 in
def tLDRspi : T1pIs<(outs tGPR:$Rt), (ins t_addrmode_sp:$addr), IIC_iLoad_i,
                    "ldr", "\t$Rt, $addr",
                    [(set tGPR:$Rt, (load t_addrmode_sp:$addr))]>,
              T1LdStSP<{1,?,?}> {
  bits<3> Rt;
  bits<8> addr;
  let Inst{10-8} = Rt;
  let Inst{7-0} = addr;
}

// Special instruction for restore. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
let canFoldAsLoad = 1, mayLoad = 1, neverHasSideEffects = 1 in
// FIXME: Pseudo for tLDRspi
def tRestore : T1pIs<(outs tGPR:$dst), (ins t_addrmode_sp:$addr), IIC_iLoad_i,
                     "ldr", "\t$dst, $addr", []>,
               T1LdStSP<{1,?,?}> {
  bits<3> Rt;
  bits<8> addr;
  let Inst{10-8} = Rt;
  let Inst{7-0} = addr;
}

// Load tconstpool
// FIXME: Use ldr.n to work around a Darwin assembler bug.
let canFoldAsLoad = 1, isReMaterializable = 1 in
def tLDRpci : T1pIs<(outs tGPR:$Rt), (ins t_addrmode_pc:$addr), IIC_iLoad_i,
                  "ldr", ".n\t$Rt, $addr",
                  [(set tGPR:$Rt, (load (ARMWrapper tconstpool:$addr)))]>,
              T1Encoding<{0,1,0,0,1,?}> {
  // A6.2 & A8.6.59
  bits<3> Rt;
  bits<8> addr;
  let Inst{10-8} = Rt;
  let Inst{7-0}  = addr;
}

// A8.6.194 & A8.6.192
defm tSTR  : thumb_st_rr_ri_enc<0b000, 0b0110, t_addrmode_rrs4,
                                t_addrmode_is4, AddrModeT1_4,
                                IIC_iStore_r, IIC_iStore_i, "str",
                                BinOpFrag<(store node:$LHS, node:$RHS)>>;

// A8.6.197 & A8.6.195
defm tSTRB : thumb_st_rr_ri_enc<0b010, 0b0111, t_addrmode_rrs1,
                                t_addrmode_is1, AddrModeT1_1,
                                IIC_iStore_bh_r, IIC_iStore_bh_i, "strb",
                                BinOpFrag<(truncstorei8 node:$LHS, node:$RHS)>>;

// A8.6.207 & A8.6.205
defm tSTRH : thumb_st_rr_ri_enc<0b001, 0b1000, t_addrmode_rrs2,
                                t_addrmode_is2, AddrModeT1_2,
                                IIC_iStore_bh_r, IIC_iStore_bh_i, "strh",
                                BinOpFrag<(truncstorei16 node:$LHS, node:$RHS)>>;


def tSTRspi : T1pIs<(outs), (ins tGPR:$Rt, t_addrmode_sp:$addr), IIC_iStore_i,
                    "str", "\t$Rt, $addr",
                    [(store tGPR:$Rt, t_addrmode_sp:$addr)]>,
              T1LdStSP<{0,?,?}> {
  bits<3> Rt;
  bits<8> addr;
  let Inst{10-8} = Rt;
  let Inst{7-0} = addr;
}

let mayStore = 1, neverHasSideEffects = 1 in
// Special instruction for spill. It cannot clobber condition register when it's
// expanded by eliminateCallFramePseudoInstr().
// FIXME: Pseudo for tSTRspi
def tSpill : T1pIs<(outs), (ins tGPR:$src, t_addrmode_sp:$addr), IIC_iStore_i,
                  "str", "\t$src, $addr", []>,
             T1LdStSP<{0,?,?}> {
  bits<3> Rt;
  bits<8> addr;
  let Inst{10-8} = Rt;
  let Inst{7-0} = addr;
}

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

multiclass thumb_ldst_mult<string asm, InstrItinClass itin,
                           InstrItinClass itin_upd, bits<6> T1Enc,
                           bit L_bit> {
  def IA :
    T1I<(outs), (ins GPR:$Rn, pred:$p, reglist:$regs, variable_ops),
        itin, !strconcat(asm, "ia${p}\t$Rn, $regs"), []>,
       T1Encoding<T1Enc> {
    bits<3> Rn;
    bits<8> regs;
    let Inst{10-8} = Rn;
    let Inst{7-0}  = regs;
  }
  def IA_UPD :
    T1It<(outs GPR:$wb), (ins GPR:$Rn, pred:$p, reglist:$regs, variable_ops),
         itin_upd, !strconcat(asm, "ia${p}\t$Rn!, $regs"), "$Rn = $wb", []>,
        T1Encoding<T1Enc> {
    bits<3> Rn;
    bits<8> regs;
    let Inst{10-8} = Rn;
    let Inst{7-0}  = regs;
  }
}

// These require base address to be written back or one of the loaded regs.
let neverHasSideEffects = 1 in {

let mayLoad = 1, hasExtraDefRegAllocReq = 1 in
defm tLDM : thumb_ldst_mult<"ldm", IIC_iLoad_m, IIC_iLoad_mu,
                            {1,1,0,0,1,?}, 1>;

let mayStore = 1, hasExtraSrcRegAllocReq = 1 in
defm tSTM : thumb_ldst_mult<"stm", IIC_iStore_m, IIC_iStore_mu,
                            {1,1,0,0,0,?}, 0>;
 
} // neverHasSideEffects

let mayLoad = 1, Uses = [SP], Defs = [SP], hasExtraDefRegAllocReq = 1 in
def tPOP : T1I<(outs), (ins pred:$p, reglist:$regs, variable_ops),
               IIC_iPop,
               "pop${p}\t$regs", []>,
           T1Misc<{1,1,0,?,?,?,?}> {
  bits<16> regs;
  let Inst{8}   = regs{15};
  let Inst{7-0} = regs{7-0};
}

let mayStore = 1, Uses = [SP], Defs = [SP], hasExtraSrcRegAllocReq = 1 in
def tPUSH : T1I<(outs), (ins pred:$p, reglist:$regs, variable_ops),
                IIC_iStore_m,
                "push${p}\t$regs", []>,
            T1Misc<{0,1,0,?,?,?,?}> {
  bits<16> regs;
  let Inst{8}   = regs{14};
  let Inst{7-0} = regs{7-0};
}

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

// Helper classes for encoding T1pI patterns:
class T1pIDPEncode<bits<4> opA, dag oops, dag iops, InstrItinClass itin,
                   string opc, string asm, list<dag> pattern>
    : T1pI<oops, iops, itin, opc, asm, pattern>,
      T1DataProcessing<opA> {
  bits<3> Rm;
  bits<3> Rn;
  let Inst{5-3} = Rm;
  let Inst{2-0} = Rn;
}
class T1pIMiscEncode<bits<7> opA, dag oops, dag iops, InstrItinClass itin,
                     string opc, string asm, list<dag> pattern>
    : T1pI<oops, iops, itin, opc, asm, pattern>,
      T1Misc<opA> {
  bits<3> Rm;
  bits<3> Rd;
  let Inst{5-3} = Rm;
  let Inst{2-0} = Rd;
}

// Helper classes for encoding T1sI patterns:
class T1sIDPEncode<bits<4> opA, dag oops, dag iops, InstrItinClass itin,
                   string opc, string asm, list<dag> pattern>
    : T1sI<oops, iops, itin, opc, asm, pattern>,
      T1DataProcessing<opA> {
  bits<3> Rd;
  bits<3> Rn;
  let Inst{5-3} = Rn;
  let Inst{2-0} = Rd;
}
class T1sIGenEncode<bits<5> opA, dag oops, dag iops, InstrItinClass itin,
                    string opc, string asm, list<dag> pattern>
    : T1sI<oops, iops, itin, opc, asm, pattern>,
      T1General<opA> {
  bits<3> Rm;
  bits<3> Rn;
  bits<3> Rd;
  let Inst{8-6} = Rm;
  let Inst{5-3} = Rn;
  let Inst{2-0} = Rd;
}
class T1sIGenEncodeImm<bits<5> opA, dag oops, dag iops, InstrItinClass itin,
                       string opc, string asm, list<dag> pattern>
    : T1sI<oops, iops, itin, opc, asm, pattern>,
      T1General<opA> {
  bits<3> Rd;
  bits<3> Rm;
  let Inst{5-3} = Rm;
  let Inst{2-0} = Rd;
}

// Helper classes for encoding T1sIt patterns:
class T1sItDPEncode<bits<4> opA, dag oops, dag iops, InstrItinClass itin,
                    string opc, string asm, list<dag> pattern>
    : T1sIt<oops, iops, itin, opc, asm, pattern>,
      T1DataProcessing<opA> {
  bits<3> Rdn;
  bits<3> Rm;
  let Inst{5-3} = Rm;
  let Inst{2-0} = Rdn;
}
class T1sItGenEncodeImm<bits<5> opA, dag oops, dag iops, InstrItinClass itin,
                        string opc, string asm, list<dag> pattern>
    : T1sIt<oops, iops, itin, opc, asm, pattern>,
      T1General<opA> {
  bits<3> Rdn;
  bits<8> imm8;
  let Inst{10-8} = Rdn;
  let Inst{7-0}  = imm8;
}

// Add with carry register
let isCommutable = 1, Uses = [CPSR] in
def tADC :                      // A8.6.2
  T1sItDPEncode<0b0101, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm), IIC_iALUr,
                "adc", "\t$Rdn, $Rm",
                [(set tGPR:$Rdn, (adde tGPR:$Rn, tGPR:$Rm))]>;

// Add immediate
def tADDi3 :                    // A8.6.4 T1
  T1sIGenEncodeImm<0b01110, (outs tGPR:$Rd), (ins tGPR:$Rm, i32imm:$imm3), IIC_iALUi,
                   "add", "\t$Rd, $Rm, $imm3",
                   [(set tGPR:$Rd, (add tGPR:$Rm, imm0_7:$imm3))]> {
  bits<3> imm3;
  let Inst{8-6} = imm3;
}

def tADDi8 :                    // A8.6.4 T2
  T1sItGenEncodeImm<{1,1,0,?,?}, (outs tGPR:$Rdn), (ins tGPR:$Rn, i32imm:$imm8),
                    IIC_iALUi,
                    "add", "\t$Rdn, $imm8",
                    [(set tGPR:$Rdn, (add tGPR:$Rn, imm8_255:$imm8))]>;

// Add register
let isCommutable = 1 in
def tADDrr :                    // A8.6.6 T1
  T1sIGenEncode<0b01100, (outs tGPR:$Rd), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iALUr,
                "add", "\t$Rd, $Rn, $Rm",
                [(set tGPR:$Rd, (add tGPR:$Rn, tGPR:$Rm))]>;

let neverHasSideEffects = 1 in
def tADDhirr : T1pIt<(outs GPR:$Rdn), (ins GPR:$Rn, GPR:$Rm), IIC_iALUr,
                     "add", "\t$Rdn, $Rm", []>,
               T1Special<{0,0,?,?}> {
  // A8.6.6 T2
  bits<4> Rdn;
  bits<4> Rm;
  let Inst{7}   = Rdn{3};
  let Inst{6-3} = Rm;
  let Inst{2-0} = Rdn{2-0};
}

// AND register
let isCommutable = 1 in
def tAND :                      // A8.6.12
  T1sItDPEncode<0b0000, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iBITr,
                "and", "\t$Rdn, $Rm",
                [(set tGPR:$Rdn, (and tGPR:$Rn, tGPR:$Rm))]>;

// ASR immediate
def tASRri :                    // A8.6.14
  T1sIGenEncodeImm<{0,1,0,?,?}, (outs tGPR:$Rd), (ins tGPR:$Rm, i32imm:$imm5),
                   IIC_iMOVsi,
                   "asr", "\t$Rd, $Rm, $imm5",
                   [(set tGPR:$Rd, (sra tGPR:$Rm, (i32 imm:$imm5)))]> {
  bits<5> imm5;
  let Inst{10-6} = imm5;
}

// ASR register
def tASRrr :                    // A8.6.15
  T1sItDPEncode<0b0100, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iMOVsr,
                "asr", "\t$Rdn, $Rm",
                [(set tGPR:$Rdn, (sra tGPR:$Rn, tGPR:$Rm))]>;

// BIC register
def tBIC :                      // A8.6.20
  T1sItDPEncode<0b1110, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iBITr,
                "bic", "\t$Rdn, $Rm",
                [(set tGPR:$Rdn, (and tGPR:$Rn, (not tGPR:$Rm)))]>;

// CMN register
let isCompare = 1, Defs = [CPSR] in {
//FIXME: Disable CMN, as CCodes are backwards from compare expectations
//       Compare-to-zero still works out, just not the relationals
//def tCMN :                     // A8.6.33
//  T1pIDPEncode<0b1011, (outs), (ins tGPR:$lhs, tGPR:$rhs),
//               IIC_iCMPr,
//               "cmn", "\t$lhs, $rhs",
//               [(ARMcmp tGPR:$lhs, (ineg tGPR:$rhs))]>;

def tCMNz :                     // A8.6.33
  T1pIDPEncode<0b1011, (outs), (ins tGPR:$Rn, tGPR:$Rm),
               IIC_iCMPr,
               "cmn", "\t$Rn, $Rm",
               [(ARMcmpZ tGPR:$Rn, (ineg tGPR:$Rm))]>;

} // isCompare = 1, Defs = [CPSR]

// CMP immediate
let isCompare = 1, Defs = [CPSR] in {
def tCMPi8 : T1pI<(outs), (ins tGPR:$Rn, i32imm:$imm8), IIC_iCMPi,
                  "cmp", "\t$Rn, $imm8",
                  [(ARMcmp tGPR:$Rn, imm0_255:$imm8)]>,
             T1General<{1,0,1,?,?}> {
  // A8.6.35
  bits<3> Rn;
  bits<8> imm8;
  let Inst{10-8} = Rn;
  let Inst{7-0}  = imm8;
}

// CMP register
def tCMPr :                     // A8.6.36 T1
  T1pIDPEncode<0b1010, (outs), (ins tGPR:$Rn, tGPR:$Rm),
               IIC_iCMPr,
               "cmp", "\t$Rn, $Rm",
               [(ARMcmp tGPR:$Rn, tGPR:$Rm)]>;

def tCMPhir : T1pI<(outs), (ins GPR:$Rn, GPR:$Rm), IIC_iCMPr,
                   "cmp", "\t$Rn, $Rm", []>,
              T1Special<{0,1,?,?}> {
  // A8.6.36 T2
  bits<4> Rm;
  bits<4> Rn;
  let Inst{7}   = Rn{3};
  let Inst{6-3} = Rm;
  let Inst{2-0} = Rn{2-0};
}
} // isCompare = 1, Defs = [CPSR]


// XOR register
let isCommutable = 1 in
def tEOR :                      // A8.6.45
  T1sItDPEncode<0b0001, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iBITr,
                "eor", "\t$Rdn, $Rm",
                [(set tGPR:$Rdn, (xor tGPR:$Rn, tGPR:$Rm))]>;

// LSL immediate
def tLSLri :                    // A8.6.88
  T1sIGenEncodeImm<{0,0,0,?,?}, (outs tGPR:$Rd), (ins tGPR:$Rm, i32imm:$imm5),
                   IIC_iMOVsi,
                   "lsl", "\t$Rd, $Rm, $imm5",
                   [(set tGPR:$Rd, (shl tGPR:$Rm, (i32 imm:$imm5)))]> {
  bits<5> imm5;
  let Inst{10-6} = imm5;
}

// LSL register
def tLSLrr :                    // A8.6.89
  T1sItDPEncode<0b0010, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iMOVsr,
                "lsl", "\t$Rdn, $Rm",
                [(set tGPR:$Rdn, (shl tGPR:$Rn, tGPR:$Rm))]>;

// LSR immediate
def tLSRri :                    // A8.6.90
  T1sIGenEncodeImm<{0,0,1,?,?}, (outs tGPR:$Rd), (ins tGPR:$Rm, i32imm:$imm5),
                   IIC_iMOVsi,
                   "lsr", "\t$Rd, $Rm, $imm5",
                   [(set tGPR:$Rd, (srl tGPR:$Rm, (i32 imm:$imm5)))]> {
  bits<5> imm5;
  let Inst{10-6} = imm5;
}

// LSR register
def tLSRrr :                    // A8.6.91
  T1sItDPEncode<0b0011, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iMOVsr,
                "lsr", "\t$Rdn, $Rm",
                [(set tGPR:$Rdn, (srl tGPR:$Rn, tGPR:$Rm))]>;

// Move register
let isMoveImm = 1 in
def tMOVi8 : T1sI<(outs tGPR:$Rd), (ins i32imm:$imm8), IIC_iMOVi,
                  "mov", "\t$Rd, $imm8",
                  [(set tGPR:$Rd, imm0_255:$imm8)]>,
             T1General<{1,0,0,?,?}> {
  // A8.6.96
  bits<3> Rd;
  bits<8> imm8;
  let Inst{10-8} = Rd;
  let Inst{7-0}  = imm8;
}

// TODO: A7-73: MOV(2) - mov setting flag.

let neverHasSideEffects = 1 in {
// FIXME: Make this predicable.
def tMOVr       : T1I<(outs tGPR:$Rd), (ins tGPR:$Rm), IIC_iMOVr,
                      "mov\t$Rd, $Rm", []>,
                  T1Special<0b1000> {
  // A8.6.97
  bits<4> Rd;
  bits<4> Rm;
  // Bits {7-6} are encoded by the T1Special value.
  let Inst{5-3} = Rm{2-0};
  let Inst{2-0} = Rd{2-0};
}
let Defs = [CPSR] in
def tMOVSr      : T1I<(outs tGPR:$Rd), (ins tGPR:$Rm), IIC_iMOVr,
                      "movs\t$Rd, $Rm", []>, Encoding16 {
  // A8.6.97
  bits<3> Rd;
  bits<3> Rm;
  let Inst{15-6} = 0b0000000000;
  let Inst{5-3}  = Rm;
  let Inst{2-0}  = Rd;
}

// FIXME: Make these predicable.
def tMOVgpr2tgpr : T1I<(outs tGPR:$Rd), (ins GPR:$Rm), IIC_iMOVr,
                       "mov\t$Rd, $Rm", []>,
                   T1Special<{1,0,0,?}> {
  // A8.6.97
  bits<4> Rd;
  bits<4> Rm;
  // Bit {7} is encoded by the T1Special value.
  let Inst{6-3} = Rm;
  let Inst{2-0} = Rd{2-0};
}
def tMOVtgpr2gpr : T1I<(outs GPR:$Rd), (ins tGPR:$Rm), IIC_iMOVr,
                       "mov\t$Rd, $Rm", []>,
                   T1Special<{1,0,?,0}> {
  // A8.6.97
  bits<4> Rd;
  bits<4> Rm;
  // Bit {6} is encoded by the T1Special value.
  let Inst{7}   = Rd{3};
  let Inst{5-3} = Rm{2-0};
  let Inst{2-0} = Rd{2-0};
}
def tMOVgpr2gpr  : T1I<(outs GPR:$Rd), (ins GPR:$Rm), IIC_iMOVr,
                       "mov\t$Rd, $Rm", []>,
                   T1Special<{1,0,?,?}> {
  // A8.6.97
  bits<4> Rd;
  bits<4> Rm;
  let Inst{7}   = Rd{3};
  let Inst{6-3} = Rm;
  let Inst{2-0} = Rd{2-0};
}
} // neverHasSideEffects

// Multiply register
let isCommutable = 1 in
def tMUL :                      // A8.6.105 T1
  T1sItDPEncode<0b1101, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iMUL32,
                "mul", "\t$Rdn, $Rm, $Rdn",
                [(set tGPR:$Rdn, (mul tGPR:$Rn, tGPR:$Rm))]>;

// Move inverse register
def tMVN :                      // A8.6.107
  T1sIDPEncode<0b1111, (outs tGPR:$Rd), (ins tGPR:$Rn), IIC_iMVNr,
               "mvn", "\t$Rd, $Rn",
               [(set tGPR:$Rd, (not tGPR:$Rn))]>;

// Bitwise or register
let isCommutable = 1 in
def tORR :                      // A8.6.114
  T1sItDPEncode<0b1100, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iBITr,
                "orr", "\t$Rdn, $Rm",
                [(set tGPR:$Rdn, (or tGPR:$Rn, tGPR:$Rm))]>;

// Swaps
def tREV :                      // A8.6.134
  T1pIMiscEncode<{1,0,1,0,0,0,?}, (outs tGPR:$Rd), (ins tGPR:$Rm),
                 IIC_iUNAr,
                 "rev", "\t$Rd, $Rm",
                 [(set tGPR:$Rd, (bswap tGPR:$Rm))]>,
                 Requires<[IsThumb, IsThumb1Only, HasV6]>;

def tREV16 :                    // A8.6.135
  T1pIMiscEncode<{1,0,1,0,0,1,?}, (outs tGPR:$Rd), (ins tGPR:$Rm),
                 IIC_iUNAr,
                 "rev16", "\t$Rd, $Rm",
             [(set tGPR:$Rd,
                   (or (and (srl tGPR:$Rm, (i32 8)), 0xFF),
                       (or (and (shl tGPR:$Rm, (i32 8)), 0xFF00),
                           (or (and (srl tGPR:$Rm, (i32 8)), 0xFF0000),
                               (and (shl tGPR:$Rm, (i32 8)), 0xFF000000)))))]>,
                Requires<[IsThumb, IsThumb1Only, HasV6]>;

def tREVSH :                    // A8.6.136
  T1pIMiscEncode<{1,0,1,0,1,1,?}, (outs tGPR:$Rd), (ins tGPR:$Rm),
                 IIC_iUNAr,
                 "revsh", "\t$Rd, $Rm",
                 [(set tGPR:$Rd,
                       (sext_inreg
                         (or (srl (and tGPR:$Rm, 0xFF00), (i32 8)),
                             (shl tGPR:$Rm, (i32 8))), i16))]>,
                 Requires<[IsThumb, IsThumb1Only, HasV6]>;

// Rotate right register
def tROR :                      // A8.6.139
  T1sItDPEncode<0b0111, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iMOVsr,
                "ror", "\t$Rdn, $Rm",
                [(set tGPR:$Rdn, (rotr tGPR:$Rn, tGPR:$Rm))]>;

// Negate register
def tRSB :                      // A8.6.141
  T1sIDPEncode<0b1001, (outs tGPR:$Rd), (ins tGPR:$Rn),
               IIC_iALUi,
               "rsb", "\t$Rd, $Rn, #0",
               [(set tGPR:$Rd, (ineg tGPR:$Rn))]>;

// Subtract with carry register
let Uses = [CPSR] in
def tSBC :                      // A8.6.151
  T1sItDPEncode<0b0110, (outs tGPR:$Rdn), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iALUr,
                "sbc", "\t$Rdn, $Rm",
                [(set tGPR:$Rdn, (sube tGPR:$Rn, tGPR:$Rm))]>;

// Subtract immediate
def tSUBi3 :                    // A8.6.210 T1
  T1sIGenEncodeImm<0b01111, (outs tGPR:$Rd), (ins tGPR:$Rm, i32imm:$imm3),
                   IIC_iALUi,
                   "sub", "\t$Rd, $Rm, $imm3",
                   [(set tGPR:$Rd, (add tGPR:$Rm, imm0_7_neg:$imm3))]> {
  bits<3> imm3;
  let Inst{8-6} = imm3;
}

def tSUBi8 :                    // A8.6.210 T2
  T1sItGenEncodeImm<{1,1,1,?,?}, (outs tGPR:$Rdn), (ins tGPR:$Rn, i32imm:$imm8),
                    IIC_iALUi,
                    "sub", "\t$Rdn, $imm8",
                    [(set tGPR:$Rdn, (add tGPR:$Rn, imm8_255_neg:$imm8))]>;

// Subtract register
def tSUBrr :                    // A8.6.212
  T1sIGenEncode<0b01101, (outs tGPR:$Rd), (ins tGPR:$Rn, tGPR:$Rm),
                IIC_iALUr,
                "sub", "\t$Rd, $Rn, $Rm",
                [(set tGPR:$Rd, (sub tGPR:$Rn, tGPR:$Rm))]>;

// TODO: A7-96: STMIA - store multiple.

// Sign-extend byte
def tSXTB :                     // A8.6.222
  T1pIMiscEncode<{0,0,1,0,0,1,?}, (outs tGPR:$Rd), (ins tGPR:$Rm),
                 IIC_iUNAr,
                 "sxtb", "\t$Rd, $Rm",
                 [(set tGPR:$Rd, (sext_inreg tGPR:$Rm, i8))]>,
                 Requires<[IsThumb, IsThumb1Only, HasV6]>;

// Sign-extend short
def tSXTH :                     // A8.6.224
  T1pIMiscEncode<{0,0,1,0,0,0,?}, (outs tGPR:$Rd), (ins tGPR:$Rm),
                 IIC_iUNAr,
                 "sxth", "\t$Rd, $Rm",
                 [(set tGPR:$Rd, (sext_inreg tGPR:$Rm, i16))]>,
                 Requires<[IsThumb, IsThumb1Only, HasV6]>;

// Test
let isCompare = 1, isCommutable = 1, Defs = [CPSR] in
def tTST :                      // A8.6.230
  T1pIDPEncode<0b1000, (outs), (ins tGPR:$Rn, tGPR:$Rm), IIC_iTSTr,
               "tst", "\t$Rn, $Rm",
               [(ARMcmpZ (and_su tGPR:$Rn, tGPR:$Rm), 0)]>;

// Zero-extend byte
def tUXTB :                     // A8.6.262
  T1pIMiscEncode<{0,0,1,0,1,1,?}, (outs tGPR:$Rd), (ins tGPR:$Rm),
                 IIC_iUNAr,
                 "uxtb", "\t$Rd, $Rm",
                 [(set tGPR:$Rd, (and tGPR:$Rm, 0xFF))]>,
                 Requires<[IsThumb, IsThumb1Only, HasV6]>;

// Zero-extend short
def tUXTH :                     // A8.6.264
  T1pIMiscEncode<{0,0,1,0,1,0,?}, (outs tGPR:$Rd), (ins tGPR:$Rm),
                 IIC_iUNAr,
                 "uxth", "\t$Rd, $Rm",
                 [(set tGPR:$Rd, (and tGPR:$Rm, 0xFFFF))]>,
                 Requires<[IsThumb, IsThumb1Only, HasV6]>;

// Conditional move tMOVCCr - Used to implement the Thumb SELECT_CC operation.
// Expanded after instruction selection into a branch sequence.
let usesCustomInserter = 1 in  // Expanded after instruction selection.
  def tMOVCCr_pseudo :
  PseudoInst<(outs tGPR:$dst), (ins tGPR:$false, tGPR:$true, pred:$cc),
              NoItinerary,
             [/*(set tGPR:$dst, (ARMcmov tGPR:$false, tGPR:$true, imm:$cc))*/]>;


// 16-bit movcc in IT blocks for Thumb2.
let neverHasSideEffects = 1 in {
def tMOVCCr : T1pIt<(outs GPR:$Rdn), (ins GPR:$Rn, GPR:$Rm), IIC_iCMOVr,
                    "mov", "\t$Rdn, $Rm", []>,
              T1Special<{1,0,?,?}> {
  bits<4> Rdn;
  bits<4> Rm;
  let Inst{7}   = Rdn{3};
  let Inst{6-3} = Rm;
  let Inst{2-0} = Rdn{2-0};
}

let isMoveImm = 1 in
def tMOVCCi : T1pIt<(outs tGPR:$Rdn), (ins tGPR:$Rn, i32imm:$Rm), IIC_iCMOVi,
                    "mov", "\t$Rdn, $Rm", []>,
              T1General<{1,0,0,?,?}> {
  bits<3> Rdn;
  bits<8> Rm;
  let Inst{10-8} = Rdn;
  let Inst{7-0}  = Rm;
}

} // neverHasSideEffects

// tLEApcrel - Load a pc-relative address into a register without offending the
// assembler.

def tADR : T1I<(outs tGPR:$Rd), (ins t_adrlabel:$addr, pred:$p),
               IIC_iALUi, "adr{$p}\t$Rd, #$addr", []>,
               T1Encoding<{1,0,1,0,0,?}> {
  bits<3> Rd;
  bits<8> addr;
  let Inst{10-8} = Rd;
  let Inst{7-0} = addr;
}

let neverHasSideEffects = 1, isReMaterializable = 1 in
def tLEApcrel   : tPseudoInst<(outs tGPR:$Rd), (ins i32imm:$label, pred:$p),
                              Size2Bytes, IIC_iALUi, []>;

def tLEApcrelJT : tPseudoInst<(outs tGPR:$Rd),
                              (ins i32imm:$label, nohash_imm:$id, pred:$p),
                              Size2Bytes, IIC_iALUi, []>;

//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1, Defs = [R0, LR], Uses = [SP] in
def tTPsoft : TIx2<0b11110, 0b11, 1, (outs), (ins), IIC_Br,
                   "bl\t__aeabi_read_tp",
                   [(set R0, ARMthread_pointer)]> {
  // Encoding is 0xf7fffffe.
  let Inst = 0xf7fffffe;
}

//===----------------------------------------------------------------------===//
// SJLJ Exception handling intrinsics
// 

// eh_sjlj_setjmp() is an instruction sequence to store the return address and
// save #0 in R0 for the non-longjmp case.  Since by its nature we may be coming
// from some other function to get here, and we're using the stack frame for the
// containing function to save/restore registers, we can't keep anything live in
// regs across the eh_sjlj_setjmp(), else it will almost certainly have been
// tromped upon when we get here from a longjmp(). We force everthing out of
// registers except for our own input by listing the relevant registers in
// Defs. By doing so, we also cause the prologue/epilogue code to actively
// preserve all of the callee-saved resgisters, which is exactly what we want.
// $val is a scratch register for our use.
let Defs = [ R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7, R12 ],
    hasSideEffects = 1, isBarrier = 1, isCodeGenOnly = 1 in
def tInt_eh_sjlj_setjmp : ThumbXI<(outs),(ins tGPR:$src, tGPR:$val),
                                  AddrModeNone, SizeSpecial, NoItinerary, "","",
                          [(set R0, (ARMeh_sjlj_setjmp tGPR:$src, tGPR:$val))]>;

// FIXME: Non-Darwin version(s)
let isBarrier = 1, hasSideEffects = 1, isTerminator = 1, isCodeGenOnly = 1,
    Defs = [ R7, LR, SP ] in
def tInt_eh_sjlj_longjmp : XI<(outs), (ins GPR:$src, GPR:$scratch),
                              AddrModeNone, SizeSpecial, IndexModeNone,
                              Pseudo, NoItinerary, "", "",
                              [(ARMeh_sjlj_longjmp GPR:$src, GPR:$scratch)]>,
                             Requires<[IsThumb, IsDarwin]>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// Comparisons
def : T1Pat<(ARMcmpZ tGPR:$Rn, imm0_255:$imm8),
            (tCMPi8  tGPR:$Rn, imm0_255:$imm8)>;
def : T1Pat<(ARMcmpZ tGPR:$Rn, tGPR:$Rm),
            (tCMPr   tGPR:$Rn, tGPR:$Rm)>;

// Add with carry
def : T1Pat<(addc   tGPR:$lhs, imm0_7:$rhs),
            (tADDi3 tGPR:$lhs, imm0_7:$rhs)>;
def : T1Pat<(addc   tGPR:$lhs, imm8_255:$rhs),
            (tADDi8 tGPR:$lhs, imm8_255:$rhs)>;
def : T1Pat<(addc   tGPR:$lhs, tGPR:$rhs),
            (tADDrr tGPR:$lhs, tGPR:$rhs)>;

// Subtract with carry
def : T1Pat<(addc   tGPR:$lhs, imm0_7_neg:$rhs),
            (tSUBi3 tGPR:$lhs, imm0_7_neg:$rhs)>;
def : T1Pat<(addc   tGPR:$lhs, imm8_255_neg:$rhs),
            (tSUBi8 tGPR:$lhs, imm8_255_neg:$rhs)>;
def : T1Pat<(subc   tGPR:$lhs, tGPR:$rhs),
            (tSUBrr tGPR:$lhs, tGPR:$rhs)>;

// ConstantPool, GlobalAddress
def : T1Pat<(ARMWrapper  tglobaladdr :$dst), (tLEApcrel tglobaladdr :$dst)>;
def : T1Pat<(ARMWrapper  tconstpool  :$dst), (tLEApcrel tconstpool  :$dst)>;

// JumpTable
def : T1Pat<(ARMWrapperJT tjumptable:$dst, imm:$id),
            (tLEApcrelJT tjumptable:$dst, imm:$id)>;

// Direct calls
def : T1Pat<(ARMtcall texternalsym:$func), (tBL texternalsym:$func)>,
      Requires<[IsThumb, IsNotDarwin]>;
def : T1Pat<(ARMtcall texternalsym:$func), (tBLr9 texternalsym:$func)>,
      Requires<[IsThumb, IsDarwin]>;

def : Tv5Pat<(ARMcall texternalsym:$func), (tBLXi texternalsym:$func)>,
      Requires<[IsThumb, HasV5T, IsNotDarwin]>;
def : Tv5Pat<(ARMcall texternalsym:$func), (tBLXi_r9 texternalsym:$func)>,
      Requires<[IsThumb, HasV5T, IsDarwin]>;

// Indirect calls to ARM routines
def : Tv5Pat<(ARMcall GPR:$dst), (tBLXr GPR:$dst)>,
      Requires<[IsThumb, HasV5T, IsNotDarwin]>;
def : Tv5Pat<(ARMcall GPR:$dst), (tBLXr_r9 GPR:$dst)>,
      Requires<[IsThumb, HasV5T, IsDarwin]>;

// zextload i1 -> zextload i8
def : T1Pat<(zextloadi1 t_addrmode_rrs1:$addr),
            (tLDRBr t_addrmode_rrs1:$addr)>;
def : T1Pat<(zextloadi1 t_addrmode_is1:$addr),
            (tLDRBi t_addrmode_is1:$addr)>;

// extload -> zextload
def : T1Pat<(extloadi1  t_addrmode_rrs1:$addr), (tLDRBr t_addrmode_rrs1:$addr)>;
def : T1Pat<(extloadi1  t_addrmode_is1:$addr),  (tLDRBi t_addrmode_is1:$addr)>;
def : T1Pat<(extloadi8  t_addrmode_rrs1:$addr), (tLDRBr t_addrmode_rrs1:$addr)>;
def : T1Pat<(extloadi8  t_addrmode_is1:$addr),  (tLDRBi t_addrmode_is1:$addr)>;
def : T1Pat<(extloadi16 t_addrmode_rrs2:$addr), (tLDRHr t_addrmode_rrs2:$addr)>;
def : T1Pat<(extloadi16 t_addrmode_is2:$addr),  (tLDRHi t_addrmode_is2:$addr)>;

// If it's impossible to use [r,r] address mode for sextload, select to
// ldr{b|h} + sxt{b|h} instead.
def : T1Pat<(sextloadi8 t_addrmode_is1:$addr),
            (tSXTB (tLDRBi t_addrmode_is1:$addr))>,
      Requires<[IsThumb, IsThumb1Only, HasV6]>;
def : T1Pat<(sextloadi8 t_addrmode_rrs1:$addr),
            (tSXTB (tLDRBr t_addrmode_rrs1:$addr))>,
      Requires<[IsThumb, IsThumb1Only, HasV6]>;
def : T1Pat<(sextloadi16 t_addrmode_is2:$addr),
            (tSXTH (tLDRHi t_addrmode_is2:$addr))>,
      Requires<[IsThumb, IsThumb1Only, HasV6]>;
def : T1Pat<(sextloadi16 t_addrmode_rrs2:$addr),
            (tSXTH (tLDRHr t_addrmode_rrs2:$addr))>,
      Requires<[IsThumb, IsThumb1Only, HasV6]>;

def : T1Pat<(sextloadi8 t_addrmode_rrs1:$addr),
            (tASRri (tLSLri (tLDRBr t_addrmode_rrs1:$addr), 24), 24)>;
def : T1Pat<(sextloadi8 t_addrmode_is1:$addr),
            (tASRri (tLSLri (tLDRBi t_addrmode_is1:$addr), 24), 24)>;
def : T1Pat<(sextloadi16 t_addrmode_rrs2:$addr),
            (tASRri (tLSLri (tLDRHr t_addrmode_rrs2:$addr), 16), 16)>;
def : T1Pat<(sextloadi16 t_addrmode_is2:$addr),
            (tASRri (tLSLri (tLDRHi t_addrmode_is2:$addr), 16), 16)>;

// Large immediate handling.

// Two piece imms.
def : T1Pat<(i32 thumb_immshifted:$src),
            (tLSLri (tMOVi8 (thumb_immshifted_val imm:$src)),
                    (thumb_immshifted_shamt imm:$src))>;

def : T1Pat<(i32 imm0_255_comp:$src),
            (tMVN (tMOVi8 (imm_comp_XFORM imm:$src)))>;

// Pseudo instruction that combines ldr from constpool and add pc. This should
// be expanded into two instructions late to allow if-conversion and
// scheduling.
let isReMaterializable = 1 in
def tLDRpci_pic : PseudoInst<(outs GPR:$dst), (ins i32imm:$addr, pclabel:$cp),
                             NoItinerary,
               [(set GPR:$dst, (ARMpic_add (load (ARMWrapper tconstpool:$addr)),
                                           imm:$cp))]>,
               Requires<[IsThumb, IsThumb1Only]>;
