//===- ARMInstrThumb.td - Thumb support for ARM ---------------------------===//
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
                      [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

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

// Break imm's up into two pieces: an immediate + a left shift.
// This uses thumb_immshifted to match and thumb_immshifted_val and
// thumb_immshifted_shamt to get the val/shift pieces.
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

// Scaled 4 immediate.
def t_imm_s4 : Operand<i32> {
  let PrintMethod = "printThumbS4ImmOperand";
}

// Define Thumb specific addressing modes.

// t_addrmode_rr := reg + reg
//
def t_addrmode_rr : Operand<i32>,
                    ComplexPattern<i32, 2, "SelectThumbAddrModeRR", []> {
  let PrintMethod = "printThumbAddrModeRROperand";
  let MIOperandInfo = (ops tGPR:$base, tGPR:$offsreg);
}

// t_addrmode_s4 := reg + reg
//                  reg + imm5 * 4
//
def t_addrmode_s4 : Operand<i32>,
                    ComplexPattern<i32, 3, "SelectThumbAddrModeS4", []> {
  let PrintMethod = "printThumbAddrModeS4Operand";
  let MIOperandInfo = (ops tGPR:$base, i32imm:$offsimm, tGPR:$offsreg);
}

// t_addrmode_s2 := reg + reg
//                  reg + imm5 * 2
//
def t_addrmode_s2 : Operand<i32>,
                    ComplexPattern<i32, 3, "SelectThumbAddrModeS2", []> {
  let PrintMethod = "printThumbAddrModeS2Operand";
  let MIOperandInfo = (ops tGPR:$base, i32imm:$offsimm, tGPR:$offsreg);
}

// t_addrmode_s1 := reg + reg
//                  reg + imm5
//
def t_addrmode_s1 : Operand<i32>,
                    ComplexPattern<i32, 3, "SelectThumbAddrModeS1", []> {
  let PrintMethod = "printThumbAddrModeS1Operand";
  let MIOperandInfo = (ops tGPR:$base, i32imm:$offsimm, tGPR:$offsreg);
}

// t_addrmode_sp := sp + imm8 * 4
//
def t_addrmode_sp : Operand<i32>,
                    ComplexPattern<i32, 2, "SelectThumbAddrModeSP", []> {
  let PrintMethod = "printThumbAddrModeSPOperand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
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
           "@ tADJCALLSTACKUP $amt1",
           [(ARMcallseq_end imm:$amt1, imm:$amt2)]>, Requires<[IsThumb1Only]>;

def tADJCALLSTACKDOWN :
PseudoInst<(outs), (ins i32imm:$amt), NoItinerary,
           "@ tADJCALLSTACKDOWN $amt",
           [(ARMcallseq_start imm:$amt)]>, Requires<[IsThumb1Only]>;
}

def tNOP : T1pI<(outs), (ins), NoItinerary, "nop", "",
                [/* For disassembly only; pattern left blank */]>,
           T1Encoding<0b101111> {
  let Inst{9-8} = 0b11;
  let Inst{7-0} = 0b00000000;
} 

def tYIELD : T1pI<(outs), (ins), NoItinerary, "yield", "",
                  [/* For disassembly only; pattern left blank */]>,
             T1Encoding<0b101111> {
  let Inst{9-8} = 0b11;
  let Inst{7-0} = 0b00010000;
} 

def tWFE : T1pI<(outs), (ins), NoItinerary, "wfe", "",
                [/* For disassembly only; pattern left blank */]>,
           T1Encoding<0b101111> {
  let Inst{9-8} = 0b11;
  let Inst{7-0} = 0b00100000;
} 

def tWFI : T1pI<(outs), (ins), NoItinerary, "wfi", "",
                [/* For disassembly only; pattern left blank */]>,
           T1Encoding<0b101111> {
  let Inst{9-8} = 0b11;
  let Inst{7-0} = 0b00110000;
} 

def tSEV : T1pI<(outs), (ins), NoItinerary, "sev", "",
                [/* For disassembly only; pattern left blank */]>,
           T1Encoding<0b101111> {
  let Inst{9-8} = 0b11;
  let Inst{7-0} = 0b01000000;
} 

def tSETENDBE : T1I<(outs), (ins), NoItinerary, "setend\tbe",
                    [/* For disassembly only; pattern left blank */]>,
                T1Encoding<0b101101> {
  let Inst{9-5} = 0b10010;
  let Inst{3} = 1;
}

def tSETENDLE : T1I<(outs), (ins), NoItinerary, "setend\tle",
                    [/* For disassembly only; pattern left blank */]>,
                T1Encoding<0b101101> {
  let Inst{9-5} = 0b10010;
  let Inst{3} = 0;
}

// The i32imm operand $val can be used by a debugger to store more information
// about the breakpoint.
def tBKPT : T1I<(outs), (ins i32imm:$val), NoItinerary, "bkpt\t$val",
                [/* For disassembly only; pattern left blank */]>,
            T1Encoding<0b101111> {
  let Inst{9-8} = 0b10;
}

// Change Processor State is a system instruction -- for disassembly only.
// The singleton $opt operand contains the following information:
// opt{4-0} = mode ==> don't care
// opt{5} = changemode ==> 0 (false for 16-bit Thumb instr)
// opt{8-6} = AIF from Inst{2-0}
// opt{10-9} = 1:imod from Inst{4} with 0b10 as enable and 0b11 as disable
//
// The opt{4-0} and opt{5} sub-fields are to accommodate 32-bit Thumb and ARM
// CPS which has more options.
def tCPS : T1I<(outs), (ins i32imm:$opt), NoItinerary, "cps${opt:cps}",
              [/* For disassembly only; pattern left blank */]>,
           T1Misc<0b0110011>;

// For both thumb1 and thumb2.
let isNotDuplicable = 1 in
def tPICADD : TIt<(outs GPR:$dst), (ins GPR:$lhs, pclabel:$cp), IIC_iALUr,
                 "\n$cp:\n\tadd\t$dst, pc",
                 [(set GPR:$dst, (ARMpic_add GPR:$lhs, imm:$cp))]>,
              T1Special<{0,0,?,?}> {
  let Inst{6-3} = 0b1111; // A8.6.6 Rm = pc
}

// PC relative add.
def tADDrPCi : T1I<(outs tGPR:$dst), (ins t_imm_s4:$rhs), IIC_iALUi,
                  "add\t$dst, pc, $rhs", []>,
               T1Encoding<{1,0,1,0,0,?}>; // A6.2 & A8.6.10

// ADD rd, sp, #imm8
def tADDrSPi : T1I<(outs tGPR:$dst), (ins GPR:$sp, t_imm_s4:$rhs), IIC_iALUi,
                  "add\t$dst, $sp, $rhs", []>,
               T1Encoding<{1,0,1,0,1,?}>; // A6.2 & A8.6.8

// ADD sp, sp, #imm7
def tADDspi : TIt<(outs GPR:$dst), (ins GPR:$lhs, t_imm_s4:$rhs), IIC_iALUi,
                  "add\t$dst, $rhs", []>,
              T1Misc<{0,0,0,0,0,?,?}>; // A6.2.5 & A8.6.8

// SUB sp, sp, #imm7
def tSUBspi : TIt<(outs GPR:$dst), (ins GPR:$lhs, t_imm_s4:$rhs), IIC_iALUi,
                  "sub\t$dst, $rhs", []>,
              T1Misc<{0,0,0,0,1,?,?}>; // A6.2.5 & A8.6.215

// ADD rm, sp
def tADDrSP : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALUr,
                  "add\t$dst, $rhs", []>,
              T1Special<{0,0,?,?}> {
  let Inst{6-3} = 0b1101; // A8.6.9 Encoding T1
}

// ADD sp, rm
def tADDspr : TIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALUr,
                  "add\t$dst, $rhs", []>,
              T1Special<{0,0,?,?}> {
  // A8.6.9 Encoding T2
  let Inst{7} = 1;
  let Inst{2-0} = 0b101;
}

// Pseudo instruction that will expand into a tSUBspi + a copy.
let usesCustomInserter = 1 in { // Expanded after instruction selection.
def tSUBspi_ : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, t_imm_s4:$rhs),
               NoItinerary, "@ sub\t$dst, $rhs", []>;

def tADDspr_ : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
               NoItinerary, "@ add\t$dst, $rhs", []>;

let Defs = [CPSR] in
def tANDsp : PseudoInst<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
             NoItinerary, "@ and\t$dst, $rhs", []>;
} // usesCustomInserter

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
  def tBX_RET : TI<(outs), (ins), IIC_Br, "bx\tlr", [(ARMretflag)]>,
                T1Special<{1,1,0,?}> { // A6.2.3 & A8.6.25
    let Inst{6-3} = 0b1110; // Rm = lr
  }
  // Alternative return instruction used by vararg functions.
  def tBX_RET_vararg : TI<(outs), (ins tGPR:$target), IIC_Br, "bx\t$target",[]>,
                       T1Special<{1,1,0,?}>; // A6.2.3 & A8.6.25
}

// Indirect branches
let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1 in {
  def tBRIND : TI<(outs), (ins GPR:$dst), IIC_Br, "mov\tpc, $dst",
                  [(brind GPR:$dst)]>,
               T1Special<{1,0,1,?}> {
    // <Rd> = Inst{7:2-0} = pc
    let Inst{2-0} = 0b111;
  }
}

// FIXME: remove when we have a way to marking a MI with these properties.
let isReturn = 1, isTerminator = 1, isBarrier = 1, mayLoad = 1,
    hasExtraDefRegAllocReq = 1 in
def tPOP_RET : T1I<(outs), (ins pred:$p, reglist:$wb, variable_ops), IIC_Br,
                   "pop${p}\t$wb", []>,
               T1Misc<{1,1,0,?,?,?,?}>;

let isCall = 1,
  Defs = [R0,  R1,  R2,  R3,  R12, LR,
          D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
          D16, D17, D18, D19, D20, D21, D22, D23,
          D24, D25, D26, D27, D28, D29, D30, D31, CPSR, FPSCR] in {
  // Also used for Thumb2
  def tBL  : TIx2<0b11110, 0b11, 1,
                  (outs), (ins i32imm:$func, variable_ops), IIC_Br,
                  "bl\t${func:call}",
                  [(ARMtcall tglobaladdr:$func)]>,
             Requires<[IsThumb, IsNotDarwin]>;

  // ARMv5T and above, also used for Thumb2
  def tBLXi : TIx2<0b11110, 0b11, 0,
                   (outs), (ins i32imm:$func, variable_ops), IIC_Br,
                   "blx\t${func:call}",
                   [(ARMcall tglobaladdr:$func)]>,
              Requires<[IsThumb, HasV5T, IsNotDarwin]>;

  // Also used for Thumb2
  def tBLXr : TI<(outs), (ins GPR:$func, variable_ops), IIC_Br,
                  "blx\t$func",
                  [(ARMtcall GPR:$func)]>,
              Requires<[IsThumb, HasV5T, IsNotDarwin]>,
              T1Special<{1,1,1,?}>; // A6.2.3 & A8.6.24;

  // ARMv4T
  def tBX : TIx2<{?,?,?,?,?}, {?,?}, ?,
                  (outs), (ins tGPR:$func, variable_ops), IIC_Br,
                  "mov\tlr, pc\n\tbx\t$func",
                  [(ARMcall_nolink tGPR:$func)]>,
            Requires<[IsThumb1Only, IsNotDarwin]>;
}

// On Darwin R9 is call-clobbered.
let isCall = 1,
  Defs = [R0,  R1,  R2,  R3,  R9,  R12, LR,
          D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
          D16, D17, D18, D19, D20, D21, D22, D23,
          D24, D25, D26, D27, D28, D29, D30, D31, CPSR, FPSCR] in {
  // Also used for Thumb2
  def tBLr9 : TIx2<0b11110, 0b11, 1,
                   (outs), (ins i32imm:$func, variable_ops), IIC_Br,
                   "bl\t${func:call}",
                   [(ARMtcall tglobaladdr:$func)]>,
              Requires<[IsThumb, IsDarwin]>;

  // ARMv5T and above, also used for Thumb2
  def tBLXi_r9 : TIx2<0b11110, 0b11, 0,
                      (outs), (ins i32imm:$func, variable_ops), IIC_Br,
                      "blx\t${func:call}",
                      [(ARMcall tglobaladdr:$func)]>,
                 Requires<[IsThumb, HasV5T, IsDarwin]>;

  // Also used for Thumb2
  def tBLXr_r9 : TI<(outs), (ins GPR:$func, variable_ops), IIC_Br,
                    "blx\t$func",
                    [(ARMtcall GPR:$func)]>,
                 Requires<[IsThumb, HasV5T, IsDarwin]>,
                 T1Special<{1,1,1,?}>; // A6.2.3 & A8.6.24

  // ARMv4T
  def tBXr9 : TIx2<{?,?,?,?,?}, {?,?}, ?,
                   (outs), (ins tGPR:$func, variable_ops), IIC_Br,
                   "mov\tlr, pc\n\tbx\t$func",
                   [(ARMcall_nolink tGPR:$func)]>,
              Requires<[IsThumb1Only, IsDarwin]>;
}

let isBranch = 1, isTerminator = 1 in {
  let isBarrier = 1 in {
    let isPredicable = 1 in
    def tB   : T1I<(outs), (ins brtarget:$target), IIC_Br,
                   "b\t$target", [(br bb:$target)]>,
               T1Encoding<{1,1,1,0,0,?}>;

  // Far jump
  let Defs = [LR] in
  def tBfar : TIx2<0b11110, 0b11, 1, (outs), (ins brtarget:$target), IIC_Br,
                    "bl\t$target\t@ far jump",[]>;

  def tBR_JTr : T1JTI<(outs),
                      (ins tGPR:$target, jtblock_operand:$jt, i32imm:$id),
                      IIC_Br, "mov\tpc, $target\n\t.align\t2\n$jt",
                      [(ARMbrjt tGPR:$target, tjumptable:$jt, imm:$id)]>,
                Encoding16 {
    let Inst{15-7} = 0b010001101;
    let Inst{2-0} = 0b111;
  }
  }
}

// FIXME: should be able to write a pattern for ARMBrcond, but can't use
// a two-value operand where a dag node expects two operands. :(
let isBranch = 1, isTerminator = 1 in
  def tBcc : T1I<(outs), (ins brtarget:$target, pred:$cc), IIC_Br,
                 "b$cc\t$target",
                 [/*(ARMbrcond bb:$target, imm:$cc)*/]>,
             T1Encoding<{1,1,0,1,?,?}>;

// Compare and branch on zero / non-zero
let isBranch = 1, isTerminator = 1 in {
  def tCBZ  : T1I<(outs), (ins tGPR:$cmp, brtarget:$target), IIC_Br,
                  "cbz\t$cmp, $target", []>,
              T1Misc<{0,0,?,1,?,?,?}>;

  def tCBNZ : T1I<(outs), (ins tGPR:$cmp, brtarget:$target), IIC_Br,
                  "cbnz\t$cmp, $target", []>,
              T1Misc<{1,0,?,1,?,?,?}>;
}

// A8.6.218 Supervisor Call (Software Interrupt) -- for disassembly only
// A8.6.16 B: Encoding T1
// If Inst{11-8} == 0b1111 then SEE SVC
let isCall = 1 in {
def tSVC : T1pI<(outs), (ins i32imm:$svc), IIC_Br, "svc", "\t$svc", []>,
           Encoding16 {
  let Inst{15-12} = 0b1101;
  let Inst{11-8} = 0b1111;
}
}

// A8.6.16 B: Encoding T1 -- for disassembly only
// If Inst{11-8} == 0b1110 then UNDEFINED
def tTRAP : T1I<(outs), (ins), IIC_Br, "trap", []>, Encoding16 {
  let Inst{15-12} = 0b1101;
  let Inst{11-8} = 0b1110;
}

//===----------------------------------------------------------------------===//
//  Load Store Instructions.
//

let canFoldAsLoad = 1, isReMaterializable = 1 in
def tLDR : T1pI4<(outs tGPR:$dst), (ins t_addrmode_s4:$addr), IIC_iLoadr,
               "ldr", "\t$dst, $addr",
               [(set tGPR:$dst, (load t_addrmode_s4:$addr))]>,
           T1LdSt<0b100>;
def tLDRi: T1pI4<(outs tGPR:$dst), (ins t_addrmode_s4:$addr), IIC_iLoadr,
               "ldr", "\t$dst, $addr",
               []>,
           T1LdSt4Imm<{1,?,?}>;

def tLDRB : T1pI1<(outs tGPR:$dst), (ins t_addrmode_s1:$addr), IIC_iLoadr,
                "ldrb", "\t$dst, $addr",
                [(set tGPR:$dst, (zextloadi8 t_addrmode_s1:$addr))]>,
            T1LdSt<0b110>;
def tLDRBi: T1pI1<(outs tGPR:$dst), (ins t_addrmode_s1:$addr), IIC_iLoadr,
                "ldrb", "\t$dst, $addr",
                []>,
            T1LdSt1Imm<{1,?,?}>;

def tLDRH : T1pI2<(outs tGPR:$dst), (ins t_addrmode_s2:$addr), IIC_iLoadr,
                "ldrh", "\t$dst, $addr",
                [(set tGPR:$dst, (zextloadi16 t_addrmode_s2:$addr))]>,
            T1LdSt<0b101>;
def tLDRHi: T1pI2<(outs tGPR:$dst), (ins t_addrmode_s2:$addr), IIC_iLoadr,
                "ldrh", "\t$dst, $addr",
                []>,
            T1LdSt2Imm<{1,?,?}>;

let AddedComplexity = 10 in
def tLDRSB : T1pI1<(outs tGPR:$dst), (ins t_addrmode_rr:$addr), IIC_iLoadr,
                 "ldrsb", "\t$dst, $addr",
                 [(set tGPR:$dst, (sextloadi8 t_addrmode_rr:$addr))]>,
             T1LdSt<0b011>;

let AddedComplexity = 10 in
def tLDRSH : T1pI2<(outs tGPR:$dst), (ins t_addrmode_rr:$addr), IIC_iLoadr,
                 "ldrsh", "\t$dst, $addr",
                 [(set tGPR:$dst, (sextloadi16 t_addrmode_rr:$addr))]>,
             T1LdSt<0b111>;

let canFoldAsLoad = 1 in
def tLDRspi : T1pIs<(outs tGPR:$dst), (ins t_addrmode_sp:$addr), IIC_iLoadi,
                  "ldr", "\t$dst, $addr",
                  [(set tGPR:$dst, (load t_addrmode_sp:$addr))]>,
              T1LdStSP<{1,?,?}>;

// Special instruction for restore. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
let canFoldAsLoad = 1, mayLoad = 1 in
def tRestore : T1pIs<(outs tGPR:$dst), (ins t_addrmode_sp:$addr), IIC_iLoadi,
                    "ldr", "\t$dst, $addr", []>,
               T1LdStSP<{1,?,?}>;

// Load tconstpool
// FIXME: Use ldr.n to work around a Darwin assembler bug.
let canFoldAsLoad = 1, isReMaterializable = 1 in
def tLDRpci : T1pIs<(outs tGPR:$dst), (ins i32imm:$addr), IIC_iLoadi,
                  "ldr", ".n\t$dst, $addr",
                  [(set tGPR:$dst, (load (ARMWrapper tconstpool:$addr)))]>,
              T1Encoding<{0,1,0,0,1,?}>; // A6.2 & A8.6.59

// Special LDR for loads from non-pc-relative constpools.
let canFoldAsLoad = 1, mayLoad = 1, isReMaterializable = 1 in
def tLDRcp  : T1pIs<(outs tGPR:$dst), (ins i32imm:$addr), IIC_iLoadi,
                  "ldr", "\t$dst, $addr", []>,
              T1LdStSP<{1,?,?}>;

def tSTR : T1pI4<(outs), (ins tGPR:$src, t_addrmode_s4:$addr), IIC_iStorer,
               "str", "\t$src, $addr",
               [(store tGPR:$src, t_addrmode_s4:$addr)]>,
           T1LdSt<0b000>;
def tSTRi: T1pI4<(outs), (ins tGPR:$src, t_addrmode_s4:$addr), IIC_iStorer,
               "str", "\t$src, $addr",
               []>,
           T1LdSt4Imm<{0,?,?}>;

def tSTRB : T1pI1<(outs), (ins tGPR:$src, t_addrmode_s1:$addr), IIC_iStorer,
                 "strb", "\t$src, $addr",
                 [(truncstorei8 tGPR:$src, t_addrmode_s1:$addr)]>,
            T1LdSt<0b010>;
def tSTRBi: T1pI1<(outs), (ins tGPR:$src, t_addrmode_s1:$addr), IIC_iStorer,
                 "strb", "\t$src, $addr",
                 []>,
            T1LdSt1Imm<{0,?,?}>;

def tSTRH : T1pI2<(outs), (ins tGPR:$src, t_addrmode_s2:$addr), IIC_iStorer,
                 "strh", "\t$src, $addr",
                 [(truncstorei16 tGPR:$src, t_addrmode_s2:$addr)]>,
            T1LdSt<0b001>;
def tSTRHi: T1pI2<(outs), (ins tGPR:$src, t_addrmode_s2:$addr), IIC_iStorer,
                 "strh", "\t$src, $addr",
                 []>,
            T1LdSt2Imm<{0,?,?}>;

def tSTRspi : T1pIs<(outs), (ins tGPR:$src, t_addrmode_sp:$addr), IIC_iStorei,
                   "str", "\t$src, $addr",
                   [(store tGPR:$src, t_addrmode_sp:$addr)]>,
              T1LdStSP<{0,?,?}>;

let mayStore = 1 in {
// Special instruction for spill. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
def tSpill : T1pIs<(outs), (ins tGPR:$src, t_addrmode_sp:$addr), IIC_iStorei,
                  "str", "\t$src, $addr", []>,
             T1LdStSP<{0,?,?}>;
}

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

// These requires base address to be written back or one of the loaded regs.
let mayLoad = 1, hasExtraDefRegAllocReq = 1 in
def tLDM : T1I<(outs),
               (ins addrmode4:$addr, pred:$p, reglist:$wb, variable_ops),
               IIC_iLoadm,
               "ldm${addr:submode}${p}\t$addr, $wb", []>,
           T1Encoding<{1,1,0,0,1,?}>; // A6.2 & A8.6.53

let mayStore = 1, hasExtraSrcRegAllocReq = 1 in
def tSTM : T1I<(outs),
               (ins addrmode4:$addr, pred:$p, reglist:$wb, variable_ops),
               IIC_iStorem,
               "stm${addr:submode}${p}\t$addr, $wb", []>,
           T1Encoding<{1,1,0,0,0,?}>; // A6.2 & A8.6.189

let mayLoad = 1, Uses = [SP], Defs = [SP], hasExtraDefRegAllocReq = 1 in
def tPOP : T1I<(outs), (ins pred:$p, reglist:$wb, variable_ops), IIC_Br,
               "pop${p}\t$wb", []>,
           T1Misc<{1,1,0,?,?,?,?}>;

let mayStore = 1, Uses = [SP], Defs = [SP], hasExtraSrcRegAllocReq = 1 in
def tPUSH : T1I<(outs), (ins pred:$p, reglist:$wb, variable_ops), IIC_Br,
                "push${p}\t$wb", []>,
            T1Misc<{0,1,0,?,?,?,?}>;

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

// Add with carry register
let isCommutable = 1, Uses = [CPSR] in
def tADC : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iALUr,
                 "adc", "\t$dst, $rhs",
                 [(set tGPR:$dst, (adde tGPR:$lhs, tGPR:$rhs))]>,
           T1DataProcessing<0b0101>;

// Add immediate
def tADDi3 : T1sI<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs), IIC_iALUi,
                   "add", "\t$dst, $lhs, $rhs",
                   [(set tGPR:$dst, (add tGPR:$lhs, imm0_7:$rhs))]>,
             T1General<0b01110>;

def tADDi8 : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs), IIC_iALUi,
                   "add", "\t$dst, $rhs",
                   [(set tGPR:$dst, (add tGPR:$lhs, imm8_255:$rhs))]>,
             T1General<{1,1,0,?,?}>;

// Add register
let isCommutable = 1 in
def tADDrr : T1sI<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iALUr,
                   "add", "\t$dst, $lhs, $rhs",
                   [(set tGPR:$dst, (add tGPR:$lhs, tGPR:$rhs))]>,
             T1General<0b01100>;

let neverHasSideEffects = 1 in
def tADDhirr : T1pIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALUr,
                     "add", "\t$dst, $rhs", []>,
               T1Special<{0,0,?,?}>;

// And register
let isCommutable = 1 in
def tAND : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iALUr,
                 "and", "\t$dst, $rhs",
                 [(set tGPR:$dst, (and tGPR:$lhs, tGPR:$rhs))]>,
           T1DataProcessing<0b0000>;

// ASR immediate
def tASRri : T1sI<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs), IIC_iMOVsi,
                  "asr", "\t$dst, $lhs, $rhs",
                  [(set tGPR:$dst, (sra tGPR:$lhs, (i32 imm:$rhs)))]>,
             T1General<{0,1,0,?,?}>;

// ASR register
def tASRrr : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iMOVsr,
                   "asr", "\t$dst, $rhs",
                   [(set tGPR:$dst, (sra tGPR:$lhs, tGPR:$rhs))]>,
             T1DataProcessing<0b0100>;

// BIC register
def tBIC : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iALUr,
                 "bic", "\t$dst, $rhs",
                 [(set tGPR:$dst, (and tGPR:$lhs, (not tGPR:$rhs)))]>,
           T1DataProcessing<0b1110>;

// CMN register
let Defs = [CPSR] in {
//FIXME: Disable CMN, as CCodes are backwards from compare expectations
//       Compare-to-zero still works out, just not the relationals
//def tCMN : T1pI<(outs), (ins tGPR:$lhs, tGPR:$rhs), IIC_iCMPr,
//                "cmn", "\t$lhs, $rhs",
//                [(ARMcmp tGPR:$lhs, (ineg tGPR:$rhs))]>,
//           T1DataProcessing<0b1011>;
def tCMNz : T1pI<(outs), (ins tGPR:$lhs, tGPR:$rhs), IIC_iCMPr,
                 "cmn", "\t$lhs, $rhs",
                 [(ARMcmpZ tGPR:$lhs, (ineg tGPR:$rhs))]>,
            T1DataProcessing<0b1011>;
}

// CMP immediate
let Defs = [CPSR] in {
def tCMPi8 : T1pI<(outs), (ins tGPR:$lhs, i32imm:$rhs), IIC_iCMPi,
                  "cmp", "\t$lhs, $rhs",
                  [(ARMcmp tGPR:$lhs, imm0_255:$rhs)]>,
             T1General<{1,0,1,?,?}>;
def tCMPzi8 : T1pI<(outs), (ins tGPR:$lhs, i32imm:$rhs), IIC_iCMPi,
                  "cmp", "\t$lhs, $rhs",
                  [(ARMcmpZ tGPR:$lhs, imm0_255:$rhs)]>,
              T1General<{1,0,1,?,?}>;
}

// CMP register
let Defs = [CPSR] in {
def tCMPr : T1pI<(outs), (ins tGPR:$lhs, tGPR:$rhs), IIC_iCMPr,
                 "cmp", "\t$lhs, $rhs",
                 [(ARMcmp tGPR:$lhs, tGPR:$rhs)]>,
            T1DataProcessing<0b1010>;
def tCMPzr : T1pI<(outs), (ins tGPR:$lhs, tGPR:$rhs), IIC_iCMPr,
                  "cmp", "\t$lhs, $rhs",
                  [(ARMcmpZ tGPR:$lhs, tGPR:$rhs)]>,
             T1DataProcessing<0b1010>;

def tCMPhir : T1pI<(outs), (ins GPR:$lhs, GPR:$rhs), IIC_iCMPr,
                   "cmp", "\t$lhs, $rhs", []>,
              T1Special<{0,1,?,?}>;
def tCMPzhir : T1pI<(outs), (ins GPR:$lhs, GPR:$rhs), IIC_iCMPr,
                    "cmp", "\t$lhs, $rhs", []>,
               T1Special<{0,1,?,?}>;
}


// XOR register
let isCommutable = 1 in
def tEOR : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iALUr,
                 "eor", "\t$dst, $rhs",
                 [(set tGPR:$dst, (xor tGPR:$lhs, tGPR:$rhs))]>,
           T1DataProcessing<0b0001>;

// LSL immediate
def tLSLri : T1sI<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs), IIC_iMOVsi,
                  "lsl", "\t$dst, $lhs, $rhs",
                  [(set tGPR:$dst, (shl tGPR:$lhs, (i32 imm:$rhs)))]>,
             T1General<{0,0,0,?,?}>;

// LSL register
def tLSLrr : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iMOVsr,
                   "lsl", "\t$dst, $rhs",
                   [(set tGPR:$dst, (shl tGPR:$lhs, tGPR:$rhs))]>,
             T1DataProcessing<0b0010>;

// LSR immediate
def tLSRri : T1sI<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs), IIC_iMOVsi,
                  "lsr", "\t$dst, $lhs, $rhs",
                  [(set tGPR:$dst, (srl tGPR:$lhs, (i32 imm:$rhs)))]>,
             T1General<{0,0,1,?,?}>;

// LSR register
def tLSRrr : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iMOVsr,
                   "lsr", "\t$dst, $rhs",
                   [(set tGPR:$dst, (srl tGPR:$lhs, tGPR:$rhs))]>,
             T1DataProcessing<0b0011>;

// move register
def tMOVi8 : T1sI<(outs tGPR:$dst), (ins i32imm:$src), IIC_iMOVi,
                  "mov", "\t$dst, $src",
                  [(set tGPR:$dst, imm0_255:$src)]>,
             T1General<{1,0,0,?,?}>;

// TODO: A7-73: MOV(2) - mov setting flag.


let neverHasSideEffects = 1 in {
// FIXME: Make this predicable.
def tMOVr       : T1I<(outs tGPR:$dst), (ins tGPR:$src), IIC_iMOVr,
                      "mov\t$dst, $src", []>,
                  T1Special<0b1000>;
let Defs = [CPSR] in
def tMOVSr      : T1I<(outs tGPR:$dst), (ins tGPR:$src), IIC_iMOVr,
                       "movs\t$dst, $src", []>, Encoding16 {
  let Inst{15-6} = 0b0000000000;
}

// FIXME: Make these predicable.
def tMOVgpr2tgpr : T1I<(outs tGPR:$dst), (ins GPR:$src), IIC_iMOVr,
                       "mov\t$dst, $src", []>,
                   T1Special<{1,0,0,?}>;
def tMOVtgpr2gpr : T1I<(outs GPR:$dst), (ins tGPR:$src), IIC_iMOVr,
                       "mov\t$dst, $src", []>,
                   T1Special<{1,0,?,0}>;
def tMOVgpr2gpr  : T1I<(outs GPR:$dst), (ins GPR:$src), IIC_iMOVr,
                       "mov\t$dst, $src", []>,
                   T1Special<{1,0,?,?}>;
} // neverHasSideEffects

// multiply register
let isCommutable = 1 in
def tMUL : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iMUL32,
                 "mul", "\t$dst, $rhs",
                 [(set tGPR:$dst, (mul tGPR:$lhs, tGPR:$rhs))]>,
           T1DataProcessing<0b1101>;

// move inverse register
def tMVN : T1sI<(outs tGPR:$dst), (ins tGPR:$src), IIC_iMOVr,
                "mvn", "\t$dst, $src",
                [(set tGPR:$dst, (not tGPR:$src))]>,
           T1DataProcessing<0b1111>;

// bitwise or register
let isCommutable = 1 in
def tORR : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),  IIC_iALUr,
                 "orr", "\t$dst, $rhs",
                 [(set tGPR:$dst, (or tGPR:$lhs, tGPR:$rhs))]>,
           T1DataProcessing<0b1100>;

// swaps
def tREV : T1pI<(outs tGPR:$dst), (ins tGPR:$src), IIC_iUNAr,
                "rev", "\t$dst, $src",
                [(set tGPR:$dst, (bswap tGPR:$src))]>,
                Requires<[IsThumb1Only, HasV6]>,
           T1Misc<{1,0,1,0,0,0,?}>;

def tREV16 : T1pI<(outs tGPR:$dst), (ins tGPR:$src), IIC_iUNAr,
                  "rev16", "\t$dst, $src",
             [(set tGPR:$dst,
                   (or (and (srl tGPR:$src, (i32 8)), 0xFF),
                       (or (and (shl tGPR:$src, (i32 8)), 0xFF00),
                           (or (and (srl tGPR:$src, (i32 8)), 0xFF0000),
                               (and (shl tGPR:$src, (i32 8)), 0xFF000000)))))]>,
                Requires<[IsThumb1Only, HasV6]>,
             T1Misc<{1,0,1,0,0,1,?}>;

def tREVSH : T1pI<(outs tGPR:$dst), (ins tGPR:$src), IIC_iUNAr,
                  "revsh", "\t$dst, $src",
                  [(set tGPR:$dst,
                        (sext_inreg
                          (or (srl (and tGPR:$src, 0xFF00), (i32 8)),
                              (shl tGPR:$src, (i32 8))), i16))]>,
                  Requires<[IsThumb1Only, HasV6]>,
             T1Misc<{1,0,1,0,1,1,?}>;

// rotate right register
def tROR : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iMOVsr,
                 "ror", "\t$dst, $rhs",
                 [(set tGPR:$dst, (rotr tGPR:$lhs, tGPR:$rhs))]>,
           T1DataProcessing<0b0111>;

// negate register
def tRSB : T1sI<(outs tGPR:$dst), (ins tGPR:$src), IIC_iALUi,
                "rsb", "\t$dst, $src, #0",
                [(set tGPR:$dst, (ineg tGPR:$src))]>,
           T1DataProcessing<0b1001>;

// Subtract with carry register
let Uses = [CPSR] in
def tSBC : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iALUr,
                 "sbc", "\t$dst, $rhs",
                 [(set tGPR:$dst, (sube tGPR:$lhs, tGPR:$rhs))]>,
           T1DataProcessing<0b0110>;

// Subtract immediate
def tSUBi3 : T1sI<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs), IIC_iALUi,
                  "sub", "\t$dst, $lhs, $rhs",
                  [(set tGPR:$dst, (add tGPR:$lhs, imm0_7_neg:$rhs))]>,
             T1General<0b01111>;

def tSUBi8 : T1sIt<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs), IIC_iALUi,
                   "sub", "\t$dst, $rhs",
                   [(set tGPR:$dst, (add tGPR:$lhs, imm8_255_neg:$rhs))]>,
             T1General<{1,1,1,?,?}>;

// subtract register
def tSUBrr : T1sI<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs), IIC_iALUr,
                  "sub", "\t$dst, $lhs, $rhs",
                  [(set tGPR:$dst, (sub tGPR:$lhs, tGPR:$rhs))]>,
             T1General<0b01101>;

// TODO: A7-96: STMIA - store multiple.

// sign-extend byte
def tSXTB  : T1pI<(outs tGPR:$dst), (ins tGPR:$src), IIC_iUNAr,
                  "sxtb", "\t$dst, $src",
                  [(set tGPR:$dst, (sext_inreg tGPR:$src, i8))]>,
                  Requires<[IsThumb1Only, HasV6]>,
             T1Misc<{0,0,1,0,0,1,?}>;

// sign-extend short
def tSXTH  : T1pI<(outs tGPR:$dst), (ins tGPR:$src), IIC_iUNAr,
                  "sxth", "\t$dst, $src",
                  [(set tGPR:$dst, (sext_inreg tGPR:$src, i16))]>,
                  Requires<[IsThumb1Only, HasV6]>,
             T1Misc<{0,0,1,0,0,0,?}>;

// test
let isCommutable = 1, Defs = [CPSR] in
def tTST  : T1pI<(outs), (ins tGPR:$lhs, tGPR:$rhs), IIC_iCMPr,
                 "tst", "\t$lhs, $rhs",
                 [(ARMcmpZ (and tGPR:$lhs, tGPR:$rhs), 0)]>,
            T1DataProcessing<0b1000>;

// zero-extend byte
def tUXTB  : T1pI<(outs tGPR:$dst), (ins tGPR:$src), IIC_iUNAr,
                  "uxtb", "\t$dst, $src",
                  [(set tGPR:$dst, (and tGPR:$src, 0xFF))]>,
                  Requires<[IsThumb1Only, HasV6]>,
             T1Misc<{0,0,1,0,1,1,?}>;

// zero-extend short
def tUXTH  : T1pI<(outs tGPR:$dst), (ins tGPR:$src), IIC_iUNAr,
                  "uxth", "\t$dst, $src",
                  [(set tGPR:$dst, (and tGPR:$src, 0xFFFF))]>,
                  Requires<[IsThumb1Only, HasV6]>,
             T1Misc<{0,0,1,0,1,0,?}>;


// Conditional move tMOVCCr - Used to implement the Thumb SELECT_CC operation.
// Expanded after instruction selection into a branch sequence.
let usesCustomInserter = 1 in  // Expanded after instruction selection.
  def tMOVCCr_pseudo :
  PseudoInst<(outs tGPR:$dst), (ins tGPR:$false, tGPR:$true, pred:$cc),
              NoItinerary, "@ tMOVCCr $cc",
             [/*(set tGPR:$dst, (ARMcmov tGPR:$false, tGPR:$true, imm:$cc))*/]>;


// 16-bit movcc in IT blocks for Thumb2.
def tMOVCCr : T1pIt<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iCMOVr,
                    "mov", "\t$dst, $rhs", []>,
              T1Special<{1,0,?,?}>;

def tMOVCCi : T1pIt<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs), IIC_iCMOVi,
                    "mov", "\t$dst, $rhs", []>,
              T1General<{1,0,0,?,?}>;

// tLEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def tLEApcrel : T1I<(outs tGPR:$dst), (ins i32imm:$label, pred:$p), IIC_iALUi,
                    "adr$p\t$dst, #$label", []>,
                T1Encoding<{1,0,1,0,0,?}>; // A6.2 & A8.6.10

def tLEApcrelJT : T1I<(outs tGPR:$dst),
                      (ins i32imm:$label, nohash_imm:$id, pred:$p),
                      IIC_iALUi, "adr$p\t$dst, #${label}_${id}", []>,
                  T1Encoding<{1,0,1,0,0,?}>; // A6.2 & A8.6.10

//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1,
  Defs = [R0, LR] in {
  def tTPsoft : TIx2<0b11110, 0b11, 1, (outs), (ins), IIC_Br,
                     "bl\t__aeabi_read_tp",
                     [(set R0, ARMthread_pointer)]>;
}

// SJLJ Exception handling intrinsics
//   eh_sjlj_setjmp() is an instruction sequence to store the return
//   address and save #0 in R0 for the non-longjmp case.
//   Since by its nature we may be coming from some other function to get
//   here, and we're using the stack frame for the containing function to
//   save/restore registers, we can't keep anything live in regs across
//   the eh_sjlj_setjmp(), else it will almost certainly have been tromped upon
//   when we get here from a longjmp(). We force everthing out of registers
//   except for our own input by listing the relevant registers in Defs. By
//   doing so, we also cause the prologue/epilogue code to actively preserve
//   all of the callee-saved resgisters, which is exactly what we want.
//   The current SP is passed in $val, and we reuse the reg as a scratch.
let Defs =
  [ R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7, R12 ] in {
  def tInt_eh_sjlj_setjmp : ThumbXI<(outs),(ins tGPR:$src, tGPR:$val),
                              AddrModeNone, SizeSpecial, NoItinerary,
                              "str\t$val, [$src, #8]\t@ begin eh.setjmp\n"
                              "\tmov\t$val, pc\n"
                              "\tadds\t$val, #9\n"
                              "\tstr\t$val, [$src, #4]\n"
                              "\tmovs\tr0, #0\n"
                              "\tb\t1f\n"
                              "\tmovs\tr0, #1\t@ end eh.setjmp\n"
                              "1:", "",
                   [(set R0, (ARMeh_sjlj_setjmp tGPR:$src, tGPR:$val))]>;
}
//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

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
def : T1Pat<(zextloadi1 t_addrmode_s1:$addr),
            (tLDRB t_addrmode_s1:$addr)>;

// extload -> zextload
def : T1Pat<(extloadi1  t_addrmode_s1:$addr),  (tLDRB t_addrmode_s1:$addr)>;
def : T1Pat<(extloadi8  t_addrmode_s1:$addr),  (tLDRB t_addrmode_s1:$addr)>;
def : T1Pat<(extloadi16 t_addrmode_s2:$addr),  (tLDRH t_addrmode_s2:$addr)>;

// If it's impossible to use [r,r] address mode for sextload, select to
// ldr{b|h} + sxt{b|h} instead.
def : T1Pat<(sextloadi8 t_addrmode_s1:$addr),
            (tSXTB (tLDRB t_addrmode_s1:$addr))>,
      Requires<[IsThumb1Only, HasV6]>;
def : T1Pat<(sextloadi16 t_addrmode_s2:$addr),
            (tSXTH (tLDRH t_addrmode_s2:$addr))>,
      Requires<[IsThumb1Only, HasV6]>;

def : T1Pat<(sextloadi8 t_addrmode_s1:$addr),
            (tASRri (tLSLri (tLDRB t_addrmode_s1:$addr), 24), 24)>;
def : T1Pat<(sextloadi16 t_addrmode_s1:$addr),
            (tASRri (tLSLri (tLDRH t_addrmode_s1:$addr), 16), 16)>;

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
                   NoItinerary, "@ ldr.n\t$dst, $addr\n$cp:\n\tadd\t$dst, pc",
               [(set GPR:$dst, (ARMpic_add (load (ARMWrapper tconstpool:$addr)),
                                           imm:$cp))]>,
               Requires<[IsThumb1Only]>;
