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
  let MIOperandInfo = (ops tGPR:$base, i32imm:$offsimm);
}

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

let Defs = [SP], Uses = [SP] in {
def tADJCALLSTACKUP :
PseudoInst<(outs), (ins i32imm:$amt1, i32imm:$amt2),
           "@ tADJCALLSTACKUP $amt1",
           [(ARMcallseq_end imm:$amt1, imm:$amt2)]>, Requires<[IsThumb1Only]>;

def tADJCALLSTACKDOWN :
PseudoInst<(outs), (ins i32imm:$amt),
           "@ tADJCALLSTACKDOWN $amt",
           [(ARMcallseq_start imm:$amt)]>, Requires<[IsThumb1Only]>;
}

let isNotDuplicable = 1 in
def tPICADD : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, pclabel:$cp),
                  "$cp:\n\tadd $dst, pc",
                  [(set tGPR:$dst, (ARMpic_add tGPR:$lhs, imm:$cp))]>;

// PC relative add.
def tADDrPCi : T1I<(outs tGPR:$dst), (ins i32imm:$rhs),
                  "add $dst, pc, $rhs * 4", []>;

// ADD rd, sp, #imm8
// FIXME: hard code sp?
def tADDrSPi : T1I<(outs tGPR:$dst), (ins GPR:$sp, i32imm:$rhs),
                  "add $dst, $sp, $rhs * 4 @ addrspi", []>;

// ADD sp, sp, #imm7
// FIXME: hard code sp?
def tADDspi : T1It<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                  "add $dst, $rhs * 4", []>;

// FIXME: Make use of the following?
// ADD rm, sp, rm
// ADD sp, rm

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1 in {
  def tBX_RET : T1I<(outs), (ins), "bx lr", [(ARMretflag)]>;
  // Alternative return instruction used by vararg functions.
  def tBX_RET_vararg : T1I<(outs), (ins tGPR:$target), "bx $target", []>;
}

// FIXME: remove when we have a way to marking a MI with these properties.
let isReturn = 1, isTerminator = 1 in
def tPOP_RET : T1I<(outs reglist:$dst1, variable_ops), (ins),
                   "pop $dst1", []>;

let isCall = 1,
  Defs = [R0, R1, R2, R3, LR,
          D0, D1, D2, D3, D4, D5, D6, D7] in {
  def tBL  : T1Ix2<(outs), (ins i32imm:$func, variable_ops),
                   "bl ${func:call}",
                   [(ARMtcall tglobaladdr:$func)]>;
  // ARMv5T and above
  def tBLXi : T1Ix2<(outs), (ins i32imm:$func, variable_ops),
                    "blx ${func:call}",
                    [(ARMcall tglobaladdr:$func)]>, Requires<[HasV5T]>;
  def tBLXr : T1I<(outs), (ins tGPR:$func, variable_ops),
                  "blx $func",
                  [(ARMtcall tGPR:$func)]>, Requires<[HasV5T]>;
  // ARMv4T
  def tBX : T1Ix2<(outs), (ins tGPR:$func, variable_ops),
                  "cpy lr, pc\n\tbx $func",
                  [(ARMcall_nolink tGPR:$func)]>;
}

let isBranch = 1, isTerminator = 1 in {
  let isBarrier = 1 in {
    let isPredicable = 1 in
    def tB   : T1I<(outs), (ins brtarget:$target), "b $target",
                   [(br bb:$target)]>;

  // Far jump
  def tBfar : T1Ix2<(outs), (ins brtarget:$target), 
                    "bl $target\t@ far jump",[]>;

  def tBR_JTr : T1JTI<(outs),
                      (ins tGPR:$target, jtblock_operand:$jt, i32imm:$id),
                      "cpy pc, $target \n\t.align\t2\n$jt",
                      [(ARMbrjt tGPR:$target, tjumptable:$jt, imm:$id)]>;
  }
}

// FIXME: should be able to write a pattern for ARMBrcond, but can't use
// a two-value operand where a dag node expects two operands. :(
let isBranch = 1, isTerminator = 1 in
  def tBcc : T1I<(outs), (ins brtarget:$target, pred:$cc), "b$cc $target",
                 [/*(ARMbrcond bb:$target, imm:$cc)*/]>;

//===----------------------------------------------------------------------===//
//  Load Store Instructions.
//

let canFoldAsLoad = 1 in
def tLDR : T1I4<(outs tGPR:$dst), (ins t_addrmode_s4:$addr),
               "ldr $dst, $addr",
               [(set tGPR:$dst, (load t_addrmode_s4:$addr))]>;

def tLDRB : T1I1<(outs tGPR:$dst), (ins t_addrmode_s1:$addr),
                "ldrb $dst, $addr",
                [(set tGPR:$dst, (zextloadi8 t_addrmode_s1:$addr))]>;

def tLDRH : T1I2<(outs tGPR:$dst), (ins t_addrmode_s2:$addr),
                "ldrh $dst, $addr",
                [(set tGPR:$dst, (zextloadi16 t_addrmode_s2:$addr))]>;

def tLDRSB : T1I1<(outs tGPR:$dst), (ins t_addrmode_rr:$addr),
                 "ldrsb $dst, $addr",
                 [(set tGPR:$dst, (sextloadi8 t_addrmode_rr:$addr))]>;

def tLDRSH : T1I2<(outs tGPR:$dst), (ins t_addrmode_rr:$addr),
                 "ldrsh $dst, $addr",
                 [(set tGPR:$dst, (sextloadi16 t_addrmode_rr:$addr))]>;

let canFoldAsLoad = 1 in
def tLDRspi : T1Is<(outs tGPR:$dst), (ins t_addrmode_sp:$addr),
                  "ldr $dst, $addr",
                  [(set tGPR:$dst, (load t_addrmode_sp:$addr))]>;

// Special instruction for restore. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
let canFoldAsLoad = 1, mayLoad = 1 in
def tRestore : T1Is<(outs tGPR:$dst), (ins t_addrmode_sp:$addr),
                    "ldr $dst, $addr", []>;

// Load tconstpool
let canFoldAsLoad = 1 in
def tLDRpci : T1Is<(outs tGPR:$dst), (ins i32imm:$addr),
                  "ldr $dst, $addr",
                  [(set tGPR:$dst, (load (ARMWrapper tconstpool:$addr)))]>;

// Special LDR for loads from non-pc-relative constpools.
let canFoldAsLoad = 1, mayLoad = 1, isReMaterializable = 1 in
def tLDRcp  : T1Is<(outs tGPR:$dst), (ins i32imm:$addr),
                  "ldr $dst, $addr", []>;

def tSTR : T1I4<(outs), (ins tGPR:$src, t_addrmode_s4:$addr),
               "str $src, $addr",
               [(store tGPR:$src, t_addrmode_s4:$addr)]>;

def tSTRB : T1I1<(outs), (ins tGPR:$src, t_addrmode_s1:$addr),
                 "strb $src, $addr",
                 [(truncstorei8 tGPR:$src, t_addrmode_s1:$addr)]>;

def tSTRH : T1I2<(outs), (ins tGPR:$src, t_addrmode_s2:$addr),
                 "strh $src, $addr",
                 [(truncstorei16 tGPR:$src, t_addrmode_s2:$addr)]>;

def tSTRspi : T1Is<(outs), (ins tGPR:$src, t_addrmode_sp:$addr),
                   "str $src, $addr",
                   [(store tGPR:$src, t_addrmode_sp:$addr)]>;

let mayStore = 1 in {
// Special instruction for spill. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
def tSpill : T1Is<(outs), (ins tGPR:$src, t_addrmode_sp:$addr),
                  "str $src, $addr", []>;
}

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

// TODO: A7-44: LDMIA - load multiple

let mayLoad = 1 in
def tPOP : T1I<(outs reglist:$dst1, variable_ops), (ins),
               "pop $dst1", []>;

let mayStore = 1 in
def tPUSH : T1I<(outs), (ins reglist:$src1, variable_ops),
                "push $src1", []>;

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

// Add with carry register
let isCommutable = 1, Defs = [CPSR], Uses = [CPSR] in
def tADCS : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                 "adc $dst, $rhs",
                 [(set tGPR:$dst, (adde tGPR:$lhs, tGPR:$rhs))]>;

// Add immediate
let Defs = [CPSR] in
def tADDi3 : T1I<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                 "add $dst, $lhs, $rhs",
                 [(set tGPR:$dst, (add tGPR:$lhs, imm0_7:$rhs))]>;

let Defs = [CPSR] in
def tADDi8 : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                  "add $dst, $rhs",
                  [(set tGPR:$dst, (add tGPR:$lhs, imm8_255:$rhs))]>;

// Add register
let isCommutable = 1, Defs = [CPSR] in
def tADDrr : T1I<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                 "add $dst, $lhs, $rhs",
                 [(set tGPR:$dst, (add tGPR:$lhs, tGPR:$rhs))]>;

let neverHasSideEffects = 1 in
def tADDhirr : T1It<(outs tGPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                   "add $dst, $rhs @ addhirr", []>;

// And register
let isCommutable = 1, Defs = [CPSR] in
def tAND : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                "and $dst, $rhs",
                [(set tGPR:$dst, (and tGPR:$lhs, tGPR:$rhs))]>;

// ASR immediate
let Defs = [CPSR] in
def tASRri : T1I<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                "asr $dst, $lhs, $rhs",
                [(set tGPR:$dst, (sra tGPR:$lhs, (i32 imm:$rhs)))]>;

// ASR register
let Defs = [CPSR] in
def tASRrr : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                 "asr $dst, $rhs",
                 [(set tGPR:$dst, (sra tGPR:$lhs, tGPR:$rhs))]>;

// BIC register
let Defs = [CPSR] in
def tBIC : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
               "bic $dst, $rhs",
               [(set tGPR:$dst, (and tGPR:$lhs, (not tGPR:$rhs)))]>;

// CMN register
let Defs = [CPSR] in {
def tCMN : T1I<(outs), (ins tGPR:$lhs, tGPR:$rhs),
              "cmn $lhs, $rhs",
              [(ARMcmp tGPR:$lhs, (ineg tGPR:$rhs))]>;
def tCMNZ : T1I<(outs), (ins tGPR:$lhs, tGPR:$rhs),
               "cmn $lhs, $rhs",
               [(ARMcmpZ tGPR:$lhs, (ineg tGPR:$rhs))]>;
}

// CMP immediate
let Defs = [CPSR] in {
def tCMPi8 : T1I<(outs), (ins tGPR:$lhs, i32imm:$rhs),
               "cmp $lhs, $rhs",
               [(ARMcmp tGPR:$lhs, imm0_255:$rhs)]>;
def tCMPZi8 : T1I<(outs), (ins tGPR:$lhs, i32imm:$rhs),
                "cmp $lhs, $rhs",
                [(ARMcmpZ tGPR:$lhs, imm0_255:$rhs)]>;

}

// CMP register
let Defs = [CPSR] in {
def tCMPr : T1I<(outs), (ins tGPR:$lhs, tGPR:$rhs),
               "cmp $lhs, $rhs",
               [(ARMcmp tGPR:$lhs, tGPR:$rhs)]>;
def tCMPZr : T1I<(outs), (ins tGPR:$lhs, tGPR:$rhs),
                "cmp $lhs, $rhs",
                [(ARMcmpZ tGPR:$lhs, tGPR:$rhs)]>;
}

// TODO: A7-37: CMP(3) - cmp hi regs

// XOR register
let isCommutable = 1, Defs = [CPSR] in
def tEOR : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
               "eor $dst, $rhs",
               [(set tGPR:$dst, (xor tGPR:$lhs, tGPR:$rhs))]>;

// LSL immediate
let Defs = [CPSR] in
def tLSLri : T1I<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                "lsl $dst, $lhs, $rhs",
                [(set tGPR:$dst, (shl tGPR:$lhs, (i32 imm:$rhs)))]>;

// LSL register
let Defs = [CPSR] in
def tLSLrr : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                 "lsl $dst, $rhs",
                 [(set tGPR:$dst, (shl tGPR:$lhs, tGPR:$rhs))]>;

// LSR immediate
let Defs = [CPSR] in
def tLSRri : T1I<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                "lsr $dst, $lhs, $rhs",
                [(set tGPR:$dst, (srl tGPR:$lhs, (i32 imm:$rhs)))]>;

// LSR register
let Defs = [CPSR] in
def tLSRrr : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                 "lsr $dst, $rhs",
                 [(set tGPR:$dst, (srl tGPR:$lhs, tGPR:$rhs))]>;

// move register
let Defs = [CPSR] in
def tMOVi8 : T1I<(outs tGPR:$dst), (ins i32imm:$src),
                 "mov $dst, $src",
                 [(set tGPR:$dst, imm0_255:$src)]>;

// TODO: A7-73: MOV(2) - mov setting flag.


// Note: MOV(2) of two low regs updates the flags, so we emit this as 'cpy',
// which is MOV(3).  This also supports high registers.
let neverHasSideEffects = 1 in {
def tMOVr       : T1I<(outs tGPR:$dst), (ins tGPR:$src),
                      "cpy $dst, $src", []>;
def tMOVhir2lor : T1I<(outs tGPR:$dst), (ins GPR:$src),
                      "cpy $dst, $src\t@ hir2lor", []>;
def tMOVlor2hir : T1I<(outs GPR:$dst), (ins tGPR:$src),
                      "cpy $dst, $src\t@ lor2hir", []>;
def tMOVhir2hir : T1I<(outs GPR:$dst), (ins GPR:$src),
                      "cpy $dst, $src\t@ hir2hir", []>;
} // neverHasSideEffects

// multiply register
let isCommutable = 1, Defs = [CPSR] in
def tMUL : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
               "mul $dst, $rhs",
               [(set tGPR:$dst, (mul tGPR:$lhs, tGPR:$rhs))]>;

// move inverse register
let Defs = [CPSR] in
def tMVN : T1I<(outs tGPR:$dst), (ins tGPR:$src),
              "mvn $dst, $src",
              [(set tGPR:$dst, (not tGPR:$src))]>;

// negate register
let Defs = [CPSR] in
def tNEG : T1I<(outs tGPR:$dst), (ins tGPR:$src),
              "neg $dst, $src",
              [(set tGPR:$dst, (ineg tGPR:$src))]>;

// bitwise or register
let isCommutable = 1, Defs = [CPSR] in
def tORR : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
               "orr $dst, $rhs",
               [(set tGPR:$dst, (or tGPR:$lhs, tGPR:$rhs))]>;

// swaps
def tREV : T1I<(outs tGPR:$dst), (ins tGPR:$src),
              "rev $dst, $src",
              [(set tGPR:$dst, (bswap tGPR:$src))]>,
              Requires<[IsThumb1Only, HasV6]>;

def tREV16 : T1I<(outs tGPR:$dst), (ins tGPR:$src),
                "rev16 $dst, $src",
                [(set tGPR:$dst,
                    (or (and (srl tGPR:$src, (i32 8)), 0xFF),
                        (or (and (shl tGPR:$src, (i32 8)), 0xFF00),
                            (or (and (srl tGPR:$src, (i32 8)), 0xFF0000),
                                (and (shl tGPR:$src, (i32 8)), 0xFF000000)))))]>,
                Requires<[IsThumb1Only, HasV6]>;

def tREVSH : T1I<(outs tGPR:$dst), (ins tGPR:$src),
                "revsh $dst, $src",
                [(set tGPR:$dst,
                   (sext_inreg
                     (or (srl (and tGPR:$src, 0xFFFF), (i32 8)),
                         (shl tGPR:$src, (i32 8))), i16))]>,
                Requires<[IsThumb1Only, HasV6]>;

// rotate right register
let Defs = [CPSR] in
def tROR : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                "ror $dst, $rhs",
                [(set tGPR:$dst, (rotr tGPR:$lhs, tGPR:$rhs))]>;

// Subtract with carry register
let Defs = [CPSR], Uses = [CPSR] in
def tSBCS : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                "sbc $dst, $rhs",
                [(set tGPR:$dst, (sube tGPR:$lhs, tGPR:$rhs))]>;

// Subtract immediate
let Defs = [CPSR] in
def tSUBi3 : T1I<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                 "sub $dst, $lhs, $rhs",
                 [(set tGPR:$dst, (add tGPR:$lhs, imm0_7_neg:$rhs))]>;

let Defs = [CPSR] in
def tSUBi8 : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                  "sub $dst, $rhs",
                  [(set tGPR:$dst, (add tGPR:$lhs, imm8_255_neg:$rhs))]>;

// subtract register
let Defs = [CPSR] in
def tSUBrr : T1I<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                "sub $dst, $lhs, $rhs",
                [(set tGPR:$dst, (sub tGPR:$lhs, tGPR:$rhs))]>;

// TODO: A7-96: STMIA - store multiple.

def tSUBspi : T1It<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                  "sub $dst, $rhs * 4", []>;

// sign-extend byte
def tSXTB  : T1I<(outs tGPR:$dst), (ins tGPR:$src),
                "sxtb $dst, $src",
                [(set tGPR:$dst, (sext_inreg tGPR:$src, i8))]>,
                Requires<[IsThumb1Only, HasV6]>;

// sign-extend short
def tSXTH  : T1I<(outs tGPR:$dst), (ins tGPR:$src),
                "sxth $dst, $src",
                [(set tGPR:$dst, (sext_inreg tGPR:$src, i16))]>,
                Requires<[IsThumb1Only, HasV6]>;

// test
let isCommutable = 1, Defs = [CPSR] in
def tTST  : T1I<(outs), (ins tGPR:$lhs, tGPR:$rhs),
               "tst $lhs, $rhs",
               [(ARMcmpZ (and tGPR:$lhs, tGPR:$rhs), 0)]>;

// zero-extend byte
def tUXTB  : T1I<(outs tGPR:$dst), (ins tGPR:$src),
                "uxtb $dst, $src",
                [(set tGPR:$dst, (and tGPR:$src, 0xFF))]>,
                Requires<[IsThumb1Only, HasV6]>;

// zero-extend short
def tUXTH  : T1I<(outs tGPR:$dst), (ins tGPR:$src),
                "uxth $dst, $src",
                [(set tGPR:$dst, (and tGPR:$src, 0xFFFF))]>,
                Requires<[IsThumb1Only, HasV6]>;


// Conditional move tMOVCCr - Used to implement the Thumb SELECT_CC DAG operation.
// Expanded by the scheduler into a branch sequence.
let usesCustomDAGSchedInserter = 1 in  // Expanded by the scheduler.
  def tMOVCCr :
  PseudoInst<(outs tGPR:$dst), (ins tGPR:$false, tGPR:$true, pred:$cc),
              "@ tMOVCCr $cc",
              [/*(set tGPR:$dst, (ARMcmov tGPR:$false, tGPR:$true, imm:$cc))*/]>;

// tLEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def tLEApcrel : T1Ix2<(outs tGPR:$dst), (ins i32imm:$label),
                    !strconcat(!strconcat(".set PCRELV${:uid}, ($label-(",
                                          "${:private}PCRELL${:uid}+4))\n"),
                               !strconcat("\tmov $dst, #PCRELV${:uid}\n",
                                  "${:private}PCRELL${:uid}:\n\tadd $dst, pc")),
                    []>;

def tLEApcrelJT : T1Ix2<(outs tGPR:$dst), (ins i32imm:$label, i32imm:$id),
          !strconcat(!strconcat(".set PCRELV${:uid}, (${label}_${id:no_hash}-(",
                                         "${:private}PCRELL${:uid}+4))\n"),
                     !strconcat("\tmov $dst, #PCRELV${:uid}\n",
                                "${:private}PCRELL${:uid}:\n\tadd $dst, pc")),
                    []>;

//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1,
  Defs = [R0, LR] in {
  def tTPsoft  : T1Ix2<(outs), (ins),
               "bl __aeabi_read_tp",
               [(set R0, ARMthread_pointer)]>;
}

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// Add with carry
def : TPat<(addc   tGPR:$lhs, imm0_7:$rhs),
           (tADDi3 tGPR:$lhs, imm0_7:$rhs)>;
def : TPat<(addc   tGPR:$lhs, imm8_255:$rhs),
           (tADDi3 tGPR:$lhs, imm8_255:$rhs)>;
def : TPat<(addc   tGPR:$lhs, tGPR:$rhs),
           (tADDrr tGPR:$lhs, tGPR:$rhs)>;

// Subtract with carry
def : TPat<(addc   tGPR:$lhs, imm0_7_neg:$rhs),
           (tSUBi3 tGPR:$lhs, imm0_7_neg:$rhs)>;
def : TPat<(addc   tGPR:$lhs, imm8_255_neg:$rhs),
           (tSUBi8 tGPR:$lhs, imm8_255_neg:$rhs)>;
def : TPat<(subc   tGPR:$lhs, tGPR:$rhs),
           (tSUBrr tGPR:$lhs, tGPR:$rhs)>;

// ConstantPool, GlobalAddress
def : TPat<(ARMWrapper  tglobaladdr :$dst), (tLEApcrel tglobaladdr :$dst)>;
def : TPat<(ARMWrapper  tconstpool  :$dst), (tLEApcrel tconstpool  :$dst)>;

// JumpTable
def : TPat<(ARMWrapperJT tjumptable:$dst, imm:$id),
           (tLEApcrelJT tjumptable:$dst, imm:$id)>;

// Direct calls
def : TPat<(ARMtcall texternalsym:$func), (tBL texternalsym:$func)>;
def : Tv5Pat<(ARMcall texternalsym:$func), (tBLXi texternalsym:$func)>;

// Indirect calls to ARM routines
def : Tv5Pat<(ARMcall tGPR:$dst), (tBLXr tGPR:$dst)>;

// zextload i1 -> zextload i8
def : T1Pat<(zextloadi1 t_addrmode_s1:$addr),
            (tLDRB t_addrmode_s1:$addr)>;

// extload -> zextload
def : T1Pat<(extloadi1  t_addrmode_s1:$addr),  (tLDRB t_addrmode_s1:$addr)>;
def : T1Pat<(extloadi8  t_addrmode_s1:$addr),  (tLDRB t_addrmode_s1:$addr)>;
def : T1Pat<(extloadi16 t_addrmode_s2:$addr),  (tLDRH t_addrmode_s2:$addr)>;

// Large immediate handling.

// Two piece imms.
def : T1Pat<(i32 thumb_immshifted:$src),
            (tLSLri (tMOVi8 (thumb_immshifted_val imm:$src)),
                    (thumb_immshifted_shamt imm:$src))>;

def : T1Pat<(i32 imm0_255_comp:$src),
            (tMVN (tMOVi8 (imm_comp_XFORM imm:$src)))>;
