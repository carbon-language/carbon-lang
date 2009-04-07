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
           [(ARMcallseq_end imm:$amt1, imm:$amt2)]>, Requires<[IsThumb]>;

def tADJCALLSTACKDOWN :
PseudoInst<(outs), (ins i32imm:$amt),
           "@ tADJCALLSTACKDOWN $amt",
           [(ARMcallseq_start imm:$amt)]>, Requires<[IsThumb]>;
}

let isNotDuplicable = 1 in
def tPICADD : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, pclabel:$cp),
                  "$cp:\n\tadd $dst, pc",
                  [(set tGPR:$dst, (ARMpic_add tGPR:$lhs, imm:$cp))]>;

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1 in {
  def tBX_RET : TI<(outs), (ins), "bx lr", [(ARMretflag)]>;
  // Alternative return instruction used by vararg functions.
  def tBX_RET_vararg : TI<(outs), (ins tGPR:$target), "bx $target", []>;
}

// FIXME: remove when we have a way to marking a MI with these properties.
let isReturn = 1, isTerminator = 1 in
def tPOP_RET : TI<(outs reglist:$dst1, variable_ops), (ins),
                   "pop $dst1", []>;

let isCall = 1,
  Defs = [R0, R1, R2, R3, LR,
          D0, D1, D2, D3, D4, D5, D6, D7] in {
  def tBL  : TIx2<(outs), (ins i32imm:$func, variable_ops),
                   "bl ${func:call}",
                   [(ARMtcall tglobaladdr:$func)]>;
  // ARMv5T and above
  def tBLXi : TIx2<(outs), (ins i32imm:$func, variable_ops),
                    "blx ${func:call}",
                    [(ARMcall tglobaladdr:$func)]>, Requires<[HasV5T]>;
  def tBLXr : TI<(outs), (ins tGPR:$func, variable_ops),
                  "blx $func",
                  [(ARMtcall tGPR:$func)]>, Requires<[HasV5T]>;
  // ARMv4T
  def tBX : TIx2<(outs), (ins tGPR:$func, variable_ops),
                  "cpy lr, pc\n\tbx $func",
                  [(ARMcall_nolink tGPR:$func)]>;
}

let isBranch = 1, isTerminator = 1 in {
  let isBarrier = 1 in {
    let isPredicable = 1 in
    def tB   : TI<(outs), (ins brtarget:$target), "b $target",
                  [(br bb:$target)]>;

  // Far jump
  def tBfar : TIx2<(outs), (ins brtarget:$target), "bl $target\t@ far jump",[]>;

  def tBR_JTr : TJTI<(outs),
                     (ins tGPR:$target, jtblock_operand:$jt, i32imm:$id),
                     "cpy pc, $target \n\t.align\t2\n$jt",
                     [(ARMbrjt tGPR:$target, tjumptable:$jt, imm:$id)]>;
  }
}

// FIXME: should be able to write a pattern for ARMBrcond, but can't use
// a two-value operand where a dag node expects two operands. :(
let isBranch = 1, isTerminator = 1 in
  def tBcc : TI<(outs), (ins brtarget:$target, pred:$cc), "b$cc $target",
                 [/*(ARMbrcond bb:$target, imm:$cc)*/]>;

//===----------------------------------------------------------------------===//
//  Load Store Instructions.
//

let canFoldAsLoad = 1 in
def tLDR : TI4<(outs tGPR:$dst), (ins t_addrmode_s4:$addr),
               "ldr $dst, $addr",
               [(set tGPR:$dst, (load t_addrmode_s4:$addr))]>;

def tLDRB : TI1<(outs tGPR:$dst), (ins t_addrmode_s1:$addr),
                "ldrb $dst, $addr",
                [(set tGPR:$dst, (zextloadi8 t_addrmode_s1:$addr))]>;

def tLDRH : TI2<(outs tGPR:$dst), (ins t_addrmode_s2:$addr),
                "ldrh $dst, $addr",
                [(set tGPR:$dst, (zextloadi16 t_addrmode_s2:$addr))]>;

def tLDRSB : TI1<(outs tGPR:$dst), (ins t_addrmode_rr:$addr),
                 "ldrsb $dst, $addr",
                 [(set tGPR:$dst, (sextloadi8 t_addrmode_rr:$addr))]>;

def tLDRSH : TI2<(outs tGPR:$dst), (ins t_addrmode_rr:$addr),
                 "ldrsh $dst, $addr",
                 [(set tGPR:$dst, (sextloadi16 t_addrmode_rr:$addr))]>;

let canFoldAsLoad = 1 in
def tLDRspi : TIs<(outs tGPR:$dst), (ins t_addrmode_sp:$addr),
                  "ldr $dst, $addr",
                  [(set tGPR:$dst, (load t_addrmode_sp:$addr))]>;

// Special instruction for restore. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
let canFoldAsLoad = 1, mayLoad = 1 in
def tRestore : TIs<(outs tGPR:$dst), (ins t_addrmode_sp:$addr),
                    "ldr $dst, $addr", []>;

// Load tconstpool
let canFoldAsLoad = 1 in
def tLDRpci : TIs<(outs tGPR:$dst), (ins i32imm:$addr),
                  "ldr $dst, $addr",
                  [(set tGPR:$dst, (load (ARMWrapper tconstpool:$addr)))]>;

// Special LDR for loads from non-pc-relative constpools.
let canFoldAsLoad = 1, mayLoad = 1, isReMaterializable = 1 in
def tLDRcp  : TIs<(outs tGPR:$dst), (ins i32imm:$addr),
                  "ldr $dst, $addr", []>;

def tSTR : TI4<(outs), (ins tGPR:$src, t_addrmode_s4:$addr),
               "str $src, $addr",
               [(store tGPR:$src, t_addrmode_s4:$addr)]>;

def tSTRB : TI1<(outs), (ins tGPR:$src, t_addrmode_s1:$addr),
                 "strb $src, $addr",
                 [(truncstorei8 tGPR:$src, t_addrmode_s1:$addr)]>;

def tSTRH : TI2<(outs), (ins tGPR:$src, t_addrmode_s2:$addr),
                 "strh $src, $addr",
                 [(truncstorei16 tGPR:$src, t_addrmode_s2:$addr)]>;

def tSTRspi : TIs<(outs), (ins tGPR:$src, t_addrmode_sp:$addr),
                   "str $src, $addr",
                   [(store tGPR:$src, t_addrmode_sp:$addr)]>;

let mayStore = 1 in {
// Special instruction for spill. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
def tSpill : TIs<(outs), (ins tGPR:$src, t_addrmode_sp:$addr),
                  "str $src, $addr", []>;
}

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

// TODO: A7-44: LDMIA - load multiple

let mayLoad = 1 in
def tPOP : TI<(outs reglist:$dst1, variable_ops), (ins),
               "pop $dst1", []>;

let mayStore = 1 in
def tPUSH : TI<(outs), (ins reglist:$src1, variable_ops),
                "push $src1", []>;

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

// Add with carry
def tADC : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
               "adc $dst, $rhs",
               [(set tGPR:$dst, (adde tGPR:$lhs, tGPR:$rhs))]>;

def tADDS : TI<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
               "add $dst, $lhs, $rhs",
               [(set tGPR:$dst, (addc tGPR:$lhs, tGPR:$rhs))]>;


def tADDi3 : TI<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                "add $dst, $lhs, $rhs",
                [(set tGPR:$dst, (add tGPR:$lhs, imm0_7:$rhs))]>;

def tADDi8 : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                 "add $dst, $rhs",
                 [(set tGPR:$dst, (add tGPR:$lhs, imm8_255:$rhs))]>;

def tADDrr : TI<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                "add $dst, $lhs, $rhs",
                [(set tGPR:$dst, (add tGPR:$lhs, tGPR:$rhs))]>;

def tADDhirr : TIt<(outs tGPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                   "add $dst, $rhs @ addhirr", []>;

def tADDrPCi : TI<(outs tGPR:$dst), (ins i32imm:$rhs),
                  "add $dst, pc, $rhs * 4", []>;

def tADDrSPi : TI<(outs tGPR:$dst), (ins GPR:$sp, i32imm:$rhs),
                  "add $dst, $sp, $rhs * 4 @ addrspi", []>;

def tADDspi : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                  "add $dst, $rhs * 4", []>;

def tAND : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                "and $dst, $rhs",
                [(set tGPR:$dst, (and tGPR:$lhs, tGPR:$rhs))]>;

def tASRri : TI<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                "asr $dst, $lhs, $rhs",
                [(set tGPR:$dst, (sra tGPR:$lhs, imm:$rhs))]>;

def tASRrr : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                 "asr $dst, $rhs",
                 [(set tGPR:$dst, (sra tGPR:$lhs, tGPR:$rhs))]>;

def tBIC : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
               "bic $dst, $rhs",
               [(set tGPR:$dst, (and tGPR:$lhs, (not tGPR:$rhs)))]>;


def tCMN : TI<(outs), (ins tGPR:$lhs, tGPR:$rhs),
              "cmn $lhs, $rhs",
              [(ARMcmp tGPR:$lhs, (ineg tGPR:$rhs))]>;

def tCMPi8 : TI<(outs), (ins tGPR:$lhs, i32imm:$rhs),
               "cmp $lhs, $rhs",
               [(ARMcmp tGPR:$lhs, imm0_255:$rhs)]>;

def tCMPr : TI<(outs), (ins tGPR:$lhs, tGPR:$rhs),
               "cmp $lhs, $rhs",
               [(ARMcmp tGPR:$lhs, tGPR:$rhs)]>;

def tTST  : TI<(outs), (ins tGPR:$lhs, tGPR:$rhs),
               "tst $lhs, $rhs",
               [(ARMcmpNZ (and tGPR:$lhs, tGPR:$rhs), 0)]>;

def tCMNNZ : TI<(outs), (ins tGPR:$lhs, tGPR:$rhs),
                "cmn $lhs, $rhs",
                [(ARMcmpNZ tGPR:$lhs, (ineg tGPR:$rhs))]>;

def tCMPNZi8 : TI<(outs), (ins tGPR:$lhs, i32imm:$rhs),
                 "cmp $lhs, $rhs",
                 [(ARMcmpNZ tGPR:$lhs, imm0_255:$rhs)]>;

def tCMPNZr : TI<(outs), (ins tGPR:$lhs, tGPR:$rhs),
                 "cmp $lhs, $rhs",
                 [(ARMcmpNZ tGPR:$lhs, tGPR:$rhs)]>;

// TODO: A7-37: CMP(3) - cmp hi regs

def tEOR : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
               "eor $dst, $rhs",
               [(set tGPR:$dst, (xor tGPR:$lhs, tGPR:$rhs))]>;

def tLSLri : TI<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                "lsl $dst, $lhs, $rhs",
                [(set tGPR:$dst, (shl tGPR:$lhs, imm:$rhs))]>;

def tLSLrr : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                 "lsl $dst, $rhs",
                 [(set tGPR:$dst, (shl tGPR:$lhs, tGPR:$rhs))]>;

def tLSRri : TI<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                "lsr $dst, $lhs, $rhs",
                [(set tGPR:$dst, (srl tGPR:$lhs, imm:$rhs))]>;

def tLSRrr : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                 "lsr $dst, $rhs",
                 [(set tGPR:$dst, (srl tGPR:$lhs, tGPR:$rhs))]>;

// FIXME: This is not rematerializable because mov changes the condition code.
def tMOVi8 : TI<(outs tGPR:$dst), (ins i32imm:$src),
                 "mov $dst, $src",
                 [(set tGPR:$dst, imm0_255:$src)]>;

// TODO: A7-73: MOV(2) - mov setting flag.


// Note: MOV(2) of two low regs updates the flags, so we emit this as 'cpy',
// which is MOV(3).  This also supports high registers.
def tMOVr       : TI<(outs tGPR:$dst), (ins tGPR:$src),
                      "cpy $dst, $src", []>;
def tMOVhir2lor : TI<(outs tGPR:$dst), (ins GPR:$src),
                      "cpy $dst, $src\t@ hir2lor", []>;
def tMOVlor2hir : TI<(outs GPR:$dst), (ins tGPR:$src),
                      "cpy $dst, $src\t@ lor2hir", []>;
def tMOVhir2hir : TI<(outs GPR:$dst), (ins GPR:$src),
                      "cpy $dst, $src\t@ hir2hir", []>;

def tMUL : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
               "mul $dst, $rhs",
               [(set tGPR:$dst, (mul tGPR:$lhs, tGPR:$rhs))]>;

def tMVN : TI<(outs tGPR:$dst), (ins tGPR:$src),
              "mvn $dst, $src",
              [(set tGPR:$dst, (not tGPR:$src))]>;

def tNEG : TI<(outs tGPR:$dst), (ins tGPR:$src),
              "neg $dst, $src",
              [(set tGPR:$dst, (ineg tGPR:$src))]>;

def tORR : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
               "orr $dst, $rhs",
               [(set tGPR:$dst, (or tGPR:$lhs, tGPR:$rhs))]>;


def tREV : TI<(outs tGPR:$dst), (ins tGPR:$src),
              "rev $dst, $src",
              [(set tGPR:$dst, (bswap tGPR:$src))]>,
              Requires<[IsThumb, HasV6]>;

def tREV16 : TI<(outs tGPR:$dst), (ins tGPR:$src),
                "rev16 $dst, $src",
                [(set tGPR:$dst,
                    (or (and (srl tGPR:$src, 8), 0xFF),
                        (or (and (shl tGPR:$src, 8), 0xFF00),
                            (or (and (srl tGPR:$src, 8), 0xFF0000),
                                (and (shl tGPR:$src, 8), 0xFF000000)))))]>,
                Requires<[IsThumb, HasV6]>;

def tREVSH : TI<(outs tGPR:$dst), (ins tGPR:$src),
                "revsh $dst, $src",
                [(set tGPR:$dst,
                   (sext_inreg
                     (or (srl (and tGPR:$src, 0xFFFF), 8),
                         (shl tGPR:$src, 8)), i16))]>,
                Requires<[IsThumb, HasV6]>;

def tROR : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                "ror $dst, $rhs",
                [(set tGPR:$dst, (rotr tGPR:$lhs, tGPR:$rhs))]>;


// Subtract with carry
def tSBC : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                "sbc $dst, $rhs",
                [(set tGPR:$dst, (sube tGPR:$lhs, tGPR:$rhs))]>;

def tSUBS : TI<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                "sub $dst, $lhs, $rhs",
               [(set tGPR:$dst, (subc tGPR:$lhs, tGPR:$rhs))]>;


// TODO: A7-96: STMIA - store multiple.

def tSUBi3 : TI<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                "sub $dst, $lhs, $rhs",
                [(set tGPR:$dst, (add tGPR:$lhs, imm0_7_neg:$rhs))]>;

def tSUBi8 : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                  "sub $dst, $rhs",
                  [(set tGPR:$dst, (add tGPR:$lhs, imm8_255_neg:$rhs))]>;

def tSUBrr : TI<(outs tGPR:$dst), (ins tGPR:$lhs, tGPR:$rhs),
                "sub $dst, $lhs, $rhs",
                [(set tGPR:$dst, (sub tGPR:$lhs, tGPR:$rhs))]>;

def tSUBspi : TIt<(outs tGPR:$dst), (ins tGPR:$lhs, i32imm:$rhs),
                  "sub $dst, $rhs * 4", []>;

def tSXTB  : TI<(outs tGPR:$dst), (ins tGPR:$src),
                "sxtb $dst, $src",
                [(set tGPR:$dst, (sext_inreg tGPR:$src, i8))]>,
                Requires<[IsThumb, HasV6]>;
def tSXTH  : TI<(outs tGPR:$dst), (ins tGPR:$src),
                "sxth $dst, $src",
                [(set tGPR:$dst, (sext_inreg tGPR:$src, i16))]>,
                Requires<[IsThumb, HasV6]>;


def tUXTB  : TI<(outs tGPR:$dst), (ins tGPR:$src),
                "uxtb $dst, $src",
                [(set tGPR:$dst, (and tGPR:$src, 0xFF))]>,
                Requires<[IsThumb, HasV6]>;
def tUXTH  : TI<(outs tGPR:$dst), (ins tGPR:$src),
                "uxth $dst, $src",
                [(set tGPR:$dst, (and tGPR:$src, 0xFFFF))]>,
                Requires<[IsThumb, HasV6]>;


// Conditional move tMOVCCr - Used to implement the Thumb SELECT_CC DAG operation.
// Expanded by the scheduler into a branch sequence.
let usesCustomDAGSchedInserter = 1 in  // Expanded by the scheduler.
  def tMOVCCr :
  PseudoInst<(outs tGPR:$dst), (ins tGPR:$false, tGPR:$true, pred:$cc),
              "@ tMOVCCr $cc",
              [/*(set tGPR:$dst, (ARMcmov tGPR:$false, tGPR:$true, imm:$cc))*/]>;

// tLEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def tLEApcrel : TIx2<(outs tGPR:$dst), (ins i32imm:$label),
                    !strconcat(!strconcat(".set PCRELV${:uid}, ($label-(",
                                          "${:private}PCRELL${:uid}+4))\n"),
                               !strconcat("\tmov $dst, #PCRELV${:uid}\n",
                                  "${:private}PCRELL${:uid}:\n\tadd $dst, pc")),
                    []>;

def tLEApcrelJT : TIx2<(outs tGPR:$dst), (ins i32imm:$label, i32imm:$id),
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
  def tTPsoft  : TIx2<(outs), (ins),
               "bl __aeabi_read_tp",
               [(set R0, ARMthread_pointer)]>;
}

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// ConstantPool, GlobalAddress
def : ThumbPat<(ARMWrapper  tglobaladdr :$dst), (tLEApcrel tglobaladdr :$dst)>;
def : ThumbPat<(ARMWrapper  tconstpool  :$dst), (tLEApcrel tconstpool  :$dst)>;

// JumpTable
def : ThumbPat<(ARMWrapperJT tjumptable:$dst, imm:$id),
               (tLEApcrelJT tjumptable:$dst, imm:$id)>;

// Direct calls
def : ThumbPat<(ARMtcall texternalsym:$func), (tBL texternalsym:$func)>;
def : ThumbV5Pat<(ARMcall texternalsym:$func), (tBLXi texternalsym:$func)>;

// Indirect calls to ARM routines
def : ThumbV5Pat<(ARMcall tGPR:$dst), (tBLXr tGPR:$dst)>;

// zextload i1 -> zextload i8
def : ThumbPat<(zextloadi1 t_addrmode_s1:$addr),
               (tLDRB t_addrmode_s1:$addr)>;

// extload -> zextload
def : ThumbPat<(extloadi1  t_addrmode_s1:$addr),  (tLDRB t_addrmode_s1:$addr)>;
def : ThumbPat<(extloadi8  t_addrmode_s1:$addr),  (tLDRB t_addrmode_s1:$addr)>;
def : ThumbPat<(extloadi16 t_addrmode_s2:$addr),  (tLDRH t_addrmode_s2:$addr)>;

// Large immediate handling.

// Two piece imms.
def : ThumbPat<(i32 thumb_immshifted:$src),
               (tLSLri (tMOVi8 (thumb_immshifted_val imm:$src)),
                       (thumb_immshifted_shamt imm:$src))>;

def : ThumbPat<(i32 imm0_255_comp:$src),
               (tMVN (tMOVi8 (imm_comp_XFORM imm:$src)))>;
