//===- ARMInstrThumb.td - Thumb support for ARM ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
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

// TI - Thumb instruction.

// ThumbPat - Same as Pat<>, but requires that the compiler be in Thumb mode.
class ThumbPat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsThumb];
}

class ThumbV5Pat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsThumb, HasV5T];
}

class ThumbI<dag ops, AddrMode am, SizeFlagVal sz,
             string asm, string cstr, list<dag> pattern>
  // FIXME: Set all opcodes to 0 for now.
  : InstARM<0, am, sz, IndexModeNone, ops, asm, cstr> {
  let Pattern = pattern;
  list<Predicate> Predicates = [IsThumb];
}

class TI<dag ops, string asm, list<dag> pattern>
  : ThumbI<ops, AddrModeNone, Size2Bytes, asm, "", pattern>;
class TI1<dag ops, string asm, list<dag> pattern>
  : ThumbI<ops, AddrModeT1, Size2Bytes, asm, "", pattern>;
class TI2<dag ops, string asm, list<dag> pattern>
  : ThumbI<ops, AddrModeT2, Size2Bytes, asm, "", pattern>;
class TI4<dag ops, string asm, list<dag> pattern>
  : ThumbI<ops, AddrModeT4, Size2Bytes, asm, "", pattern>;
class TIs<dag ops, string asm, list<dag> pattern>
  : ThumbI<ops, AddrModeTs, Size2Bytes, asm, "", pattern>;

// Two-address instructions
class TIt<dag ops, string asm, list<dag> pattern>
  : ThumbI<ops, AddrModeNone, Size2Bytes, asm, "$lhs = $dst", pattern>;

// BL, BLX(1) are translated by assembler into two instructions
class TIx2<dag ops, string asm, list<dag> pattern>
  : ThumbI<ops, AddrModeNone, Size4Bytes, asm, "", pattern>;

// BR_JT instructions
class TJTI<dag ops, string asm, list<dag> pattern>
  : ThumbI<ops, AddrModeNone, SizeSpecial, asm, "", pattern>;

def imm_neg_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(-(int)N->getValue(), MVT::i32);
}]>;
def imm_comp_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(~((uint32_t)N->getValue()), MVT::i32);
}]>;


/// imm0_7 predicate - True if the 32-bit immediate is in the range [0,7].
def imm0_7 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getValue() < 8;
}]>;
def imm0_7_neg : PatLeaf<(i32 imm), [{
  return (uint32_t)-N->getValue() < 8;
}], imm_neg_XFORM>;

def imm0_255 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getValue() < 256;
}]>;
def imm0_255_comp : PatLeaf<(i32 imm), [{
  return ~((uint32_t)N->getValue()) < 256;
}]>;

def imm8_255 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getValue() >= 8 && (uint32_t)N->getValue() < 256;
}]>;
def imm8_255_neg : PatLeaf<(i32 imm), [{
  unsigned Val = -N->getValue();
  return Val >= 8 && Val < 256;
}], imm_neg_XFORM>;

// Break imm's up into two pieces: an immediate + a left shift.
// This uses thumb_immshifted to match and thumb_immshifted_val and
// thumb_immshifted_shamt to get the val/shift pieces.
def thumb_immshifted : PatLeaf<(imm), [{
  return ARM_AM::isThumbImmShiftedVal((unsigned)N->getValue());
}]>;

def thumb_immshifted_val : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getThumbImmNonShiftedVal((unsigned)N->getValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

def thumb_immshifted_shamt : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getThumbImmValShift((unsigned)N->getValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

// Define Thumb specific addressing modes.

// t_addrmode_rr := reg + reg
//
def t_addrmode_rr : Operand<i32>,
                    ComplexPattern<i32, 2, "SelectThumbAddrModeRR", []> {
  let PrintMethod = "printThumbAddrModeRROperand";
  let MIOperandInfo = (ops GPR:$base, GPR:$offsreg);
}

// t_addrmode_s4 := reg + reg
//                  reg + imm5 * 4
//
def t_addrmode_s4 : Operand<i32>,
                    ComplexPattern<i32, 3, "SelectThumbAddrModeS4", []> {
  let PrintMethod = "printThumbAddrModeS4Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm, GPR:$offsreg);
}

// t_addrmode_s2 := reg + reg
//                  reg + imm5 * 2
//
def t_addrmode_s2 : Operand<i32>,
                    ComplexPattern<i32, 3, "SelectThumbAddrModeS2", []> {
  let PrintMethod = "printThumbAddrModeS2Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm, GPR:$offsreg);
}

// t_addrmode_s1 := reg + reg
//                  reg + imm5
//
def t_addrmode_s1 : Operand<i32>,
                    ComplexPattern<i32, 3, "SelectThumbAddrModeS1", []> {
  let PrintMethod = "printThumbAddrModeS1Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm, GPR:$offsreg);
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

def tPICADD : TIt<(ops GPR:$dst, GPR:$lhs, pclabel:$cp),
                  "$cp:\n\tadd $dst, pc",
                  [(set GPR:$dst, (ARMpic_add GPR:$lhs, imm:$cp))]>;

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1 in {
  def tBX_RET : TI<(ops), "bx lr", [(ARMretflag)]>;
  // Alternative return instruction used by vararg functions.
  def tBX_RET_vararg : TI<(ops GPR:$dst), "bx $dst", []>;
}

// FIXME: remove when we have a way to marking a MI with these properties.
let isLoad = 1, isReturn = 1, isTerminator = 1 in
def tPOP_RET : TI<(ops reglist:$dst1, variable_ops),
                   "pop $dst1", []>;

let isCall = 1, noResults = 1, 
  Defs = [R0, R1, R2, R3, LR,
          D0, D1, D2, D3, D4, D5, D6, D7] in {
  def tBL  : TIx2<(ops i32imm:$func, variable_ops),
                   "bl ${func:call}",
                   [(ARMtcall tglobaladdr:$func)]>;
  // ARMv5T and above
  def tBLXi : TIx2<(ops i32imm:$func, variable_ops),
                    "blx ${func:call}",
                    [(ARMcall tglobaladdr:$func)]>, Requires<[HasV5T]>;
  def tBLXr : TI<(ops GPR:$dst, variable_ops),
                  "blx $dst",
                  [(ARMtcall GPR:$dst)]>, Requires<[HasV5T]>;
  // ARMv4T
  def tBX : TIx2<(ops GPR:$dst, variable_ops),
                  "cpy lr, pc\n\tbx $dst",
                  [(ARMcall_nolink GPR:$dst)]>;
}

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
  def tB   : TI<(ops brtarget:$dst), "b $dst", [(br bb:$dst)]>;

  // Far jump
  def tBfar  : TIx2<(ops brtarget:$dst), "bl $dst\t@ far jump", []>;

  def tBR_JTr : TJTI<(ops GPR:$dst, jtblock_operand:$jt, i32imm:$id),
                     "cpy pc, $dst \n\t.align\t2\n$jt",
                     [(ARMbrjt GPR:$dst, tjumptable:$jt, imm:$id)]>;
}

let isBranch = 1, isTerminator = 1, noResults = 1, isBarrier = 1 in
  def tBcc : TI<(ops brtarget:$dst, CCOp:$cc), "b$cc $dst",
                 [(ARMbrcond bb:$dst, imm:$cc)]>;

//===----------------------------------------------------------------------===//
//  Load Store Instructions.
//

let isLoad = 1 in {
def tLDR : TI4<(ops GPR:$dst, t_addrmode_s4:$addr),
               "ldr $dst, $addr",
               [(set GPR:$dst, (load t_addrmode_s4:$addr))]>;

def tLDRB : TI1<(ops GPR:$dst, t_addrmode_s1:$addr),
                "ldrb $dst, $addr",
                [(set GPR:$dst, (zextloadi8 t_addrmode_s1:$addr))]>;

def tLDRH : TI2<(ops GPR:$dst, t_addrmode_s2:$addr),
                "ldrh $dst, $addr",
                [(set GPR:$dst, (zextloadi16 t_addrmode_s2:$addr))]>;

def tLDRSB : TI1<(ops GPR:$dst, t_addrmode_rr:$addr),
                 "ldrsb $dst, $addr",
                 [(set GPR:$dst, (sextloadi8 t_addrmode_rr:$addr))]>;

def tLDRSH : TI2<(ops GPR:$dst, t_addrmode_rr:$addr),
                 "ldrsh $dst, $addr",
                 [(set GPR:$dst, (sextloadi16 t_addrmode_rr:$addr))]>;

def tLDRspi : TIs<(ops GPR:$dst, t_addrmode_sp:$addr),
                  "ldr $dst, $addr",
                  [(set GPR:$dst, (load t_addrmode_sp:$addr))]>;

// Special instruction for restore. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
def tRestore : TIs<(ops GPR:$dst, t_addrmode_sp:$addr),
                    "ldr $dst, $addr", []>;

// Load tconstpool
def tLDRpci : TIs<(ops GPR:$dst, i32imm:$addr),
                  "ldr $dst, $addr",
                  [(set GPR:$dst, (load (ARMWrapper tconstpool:$addr)))]>;

// Special LDR for loads from non-pc-relative constpools.
let isReMaterializable = 1 in
def tLDRcp  : TIs<(ops GPR:$dst, i32imm:$addr),
                  "ldr $dst, $addr", []>;
} // isLoad

let isStore = 1 in {
def tSTR : TI4<(ops GPR:$src, t_addrmode_s4:$addr),
               "str $src, $addr",
               [(store GPR:$src, t_addrmode_s4:$addr)]>;

def tSTRB : TI1<(ops GPR:$src, t_addrmode_s1:$addr),
                 "strb $src, $addr",
                 [(truncstorei8 GPR:$src, t_addrmode_s1:$addr)]>;

def tSTRH : TI2<(ops GPR:$src, t_addrmode_s2:$addr),
                 "strh $src, $addr",
                 [(truncstorei16 GPR:$src, t_addrmode_s2:$addr)]>;

def tSTRspi : TIs<(ops GPR:$src, t_addrmode_sp:$addr),
                   "str $src, $addr",
                   [(store GPR:$src, t_addrmode_sp:$addr)]>;

// Special instruction for spill. It cannot clobber condition register
// when it's expanded by eliminateCallFramePseudoInstr().
def tSpill : TIs<(ops GPR:$src, t_addrmode_sp:$addr),
                  "str $src, $addr", []>;
}

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

// TODO: A7-44: LDMIA - load multiple

let isLoad = 1 in
def tPOP : TI<(ops reglist:$dst1, variable_ops),
               "pop $dst1", []>;

let isStore = 1 in
def tPUSH : TI<(ops reglist:$src1, variable_ops),
                "push $src1", []>;

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

// Add with carry
def tADC : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
               "adc $dst, $rhs",
               [(set GPR:$dst, (adde GPR:$lhs, GPR:$rhs))]>;

def tADDS : TI<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
               "add $dst, $lhs, $rhs",
               [(set GPR:$dst, (addc GPR:$lhs, GPR:$rhs))]>;


def tADDi3 : TI<(ops GPR:$dst, GPR:$lhs, i32imm:$rhs),
                "add $dst, $lhs, $rhs",
                [(set GPR:$dst, (add GPR:$lhs, imm0_7:$rhs))]>;

def tADDi8 : TIt<(ops GPR:$dst, GPR:$lhs, i32imm:$rhs),
                 "add $dst, $rhs",
                 [(set GPR:$dst, (add GPR:$lhs, imm8_255:$rhs))]>;

def tADDrr : TI<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
                "add $dst, $lhs, $rhs",
                [(set GPR:$dst, (add GPR:$lhs, GPR:$rhs))]>;

def tADDhirr : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
                   "add $dst, $rhs", []>;

def tADDrPCi : TI<(ops GPR:$dst, i32imm:$rhs),
                  "add $dst, pc, $rhs * 4", []>;
def tADDrSPi : TI<(ops GPR:$dst, GPR:$sp, i32imm:$rhs),
                  "add $dst, $sp, $rhs * 4", []>;
def tADDspi : TIt<(ops GPR:$dst, GPR:$lhs, i32imm:$rhs),
                  "add $dst, $rhs * 4", []>;

def tAND : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
                "and $dst, $rhs",
                [(set GPR:$dst, (and GPR:$lhs, GPR:$rhs))]>;

def tASRri : TI<(ops GPR:$dst, GPR:$lhs, i32imm:$rhs),
                "asr $dst, $lhs, $rhs",
                [(set GPR:$dst, (sra GPR:$lhs, imm:$rhs))]>;

def tASRrr : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
                 "asr $dst, $rhs",
                 [(set GPR:$dst, (sra GPR:$lhs, GPR:$rhs))]>;

def tBIC : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
               "bic $dst, $rhs",
               [(set GPR:$dst, (and GPR:$lhs, (not GPR:$rhs)))]>;


def tCMN : TI<(ops GPR:$lhs, GPR:$rhs),
              "cmn $lhs, $rhs",
              [(ARMcmp GPR:$lhs, (ineg GPR:$rhs))]>;

def tCMPi8 : TI<(ops GPR:$lhs, i32imm:$rhs),
               "cmp $lhs, $rhs",
               [(ARMcmp GPR:$lhs, imm0_255:$rhs)]>;

def tCMPr : TI<(ops GPR:$lhs, GPR:$rhs),
               "cmp $lhs, $rhs",
               [(ARMcmp GPR:$lhs, GPR:$rhs)]>;

def tTST  : TI<(ops GPR:$lhs, GPR:$rhs),
               "tst $lhs, $rhs",
               [(ARMcmpNZ (and GPR:$lhs, GPR:$rhs), 0)]>;

def tCMNNZ : TI<(ops GPR:$lhs, GPR:$rhs),
                "cmn $lhs, $rhs",
                [(ARMcmpNZ GPR:$lhs, (ineg GPR:$rhs))]>;

def tCMPNZi8 : TI<(ops GPR:$lhs, i32imm:$rhs),
                 "cmp $lhs, $rhs",
                 [(ARMcmpNZ GPR:$lhs, imm0_255:$rhs)]>;

def tCMPNZr : TI<(ops GPR:$lhs, GPR:$rhs),
                 "cmp $lhs, $rhs",
                 [(ARMcmpNZ GPR:$lhs, GPR:$rhs)]>;

// TODO: A7-37: CMP(3) - cmp hi regs

def tEOR : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
               "eor $dst, $rhs",
               [(set GPR:$dst, (xor GPR:$lhs, GPR:$rhs))]>;

def tLSLri : TI<(ops GPR:$dst, GPR:$lhs, i32imm:$rhs),
                "lsl $dst, $lhs, $rhs",
                [(set GPR:$dst, (shl GPR:$lhs, imm:$rhs))]>;

def tLSLrr : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
                 "lsl $dst, $rhs",
                 [(set GPR:$dst, (shl GPR:$lhs, GPR:$rhs))]>;

def tLSRri : TI<(ops GPR:$dst, GPR:$lhs, i32imm:$rhs),
                "lsr $dst, $lhs, $rhs",
                [(set GPR:$dst, (srl GPR:$lhs, imm:$rhs))]>;

def tLSRrr : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
                 "lsr $dst, $rhs",
                 [(set GPR:$dst, (srl GPR:$lhs, GPR:$rhs))]>;

// FIXME: This is not rematerializable because mov changes the condition code.
def tMOVi8 : TI<(ops GPR:$dst, i32imm:$src),
                 "mov $dst, $src",
                 [(set GPR:$dst, imm0_255:$src)]>;

// TODO: A7-73: MOV(2) - mov setting flag.


// Note: MOV(2) of two low regs updates the flags, so we emit this as 'cpy',
// which is MOV(3).  This also supports high registers.
def tMOVr  : TI<(ops GPR:$dst, GPR:$src),
                 "cpy $dst, $src", []>;

def tMUL : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
               "mul $dst, $rhs",
               [(set GPR:$dst, (mul GPR:$lhs, GPR:$rhs))]>;

def tMVN : TI<(ops GPR:$dst, GPR:$src),
              "mvn $dst, $src",
              [(set GPR:$dst, (not GPR:$src))]>;

def tNEG : TI<(ops GPR:$dst, GPR:$src),
              "neg $dst, $src",
              [(set GPR:$dst, (ineg GPR:$src))]>;

def tORR : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
               "orr $dst, $rhs",
               [(set GPR:$dst, (or GPR:$lhs, GPR:$rhs))]>;


def tREV : TI<(ops GPR:$dst, GPR:$src),
              "rev $dst, $src",
              [(set GPR:$dst, (bswap GPR:$src))]>, 
              Requires<[IsThumb, HasV6]>;

def tREV16 : TI<(ops GPR:$dst, GPR:$src),
                "rev16 $dst, $src",
                [(set GPR:$dst,
                    (or (and (srl GPR:$src, 8), 0xFF),
                        (or (and (shl GPR:$src, 8), 0xFF00),
                            (or (and (srl GPR:$src, 8), 0xFF0000),
                                (and (shl GPR:$src, 8), 0xFF000000)))))]>,
                Requires<[IsThumb, HasV6]>;

def tREVSH : TI<(ops GPR:$dst, GPR:$src),
                "revsh $dst, $src",
                [(set GPR:$dst,
                   (sext_inreg
                     (or (srl (and GPR:$src, 0xFFFF), 8),
                         (shl GPR:$src, 8)), i16))]>,
                Requires<[IsThumb, HasV6]>;

def tROR : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
                "ror $dst, $rhs",
                [(set GPR:$dst, (rotr GPR:$lhs, GPR:$rhs))]>;


// Subtract with carry
def tSBC : TIt<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
                "sbc $dst, $rhs",
                [(set GPR:$dst, (sube GPR:$lhs, GPR:$rhs))]>;

def tSUBS : TI<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
                "sub $dst, $lhs, $rhs",
               [(set GPR:$dst, (subc GPR:$lhs, GPR:$rhs))]>;


// TODO: A7-96: STMIA - store multiple.

def tSUBi3 : TI<(ops GPR:$dst, GPR:$lhs, i32imm:$rhs),
                "sub $dst, $lhs, $rhs",
                [(set GPR:$dst, (add GPR:$lhs, imm0_7_neg:$rhs))]>;
                
def tSUBi8 : TIt<(ops GPR:$dst, GPR:$lhs, i32imm:$rhs),
                  "sub $dst, $rhs",
                  [(set GPR:$dst, (add GPR:$lhs, imm8_255_neg:$rhs))]>;
                
def tSUBrr : TI<(ops GPR:$dst, GPR:$lhs, GPR:$rhs),
                "sub $dst, $lhs, $rhs",
                [(set GPR:$dst, (sub GPR:$lhs, GPR:$rhs))]>;

def tSUBspi : TIt<(ops GPR:$dst, GPR:$lhs, i32imm:$rhs),
                  "sub $dst, $rhs * 4", []>;

def tSXTB  : TI<(ops GPR:$dst, GPR:$src),
                "sxtb $dst, $src",
                [(set GPR:$dst, (sext_inreg GPR:$src, i8))]>,
                Requires<[IsThumb, HasV6]>;
def tSXTH  : TI<(ops GPR:$dst, GPR:$src),
                "sxth $dst, $src",
                [(set GPR:$dst, (sext_inreg GPR:$src, i16))]>,
                Requires<[IsThumb, HasV6]>;


def tUXTB  : TI<(ops GPR:$dst, GPR:$src),
                "uxtb $dst, $src",
                [(set GPR:$dst, (and GPR:$src, 0xFF))]>,
                Requires<[IsThumb, HasV6]>;
def tUXTH  : TI<(ops GPR:$dst, GPR:$src),
                "uxth $dst, $src",
                [(set GPR:$dst, (and GPR:$src, 0xFFFF))]>, 
                Requires<[IsThumb, HasV6]>;


// Conditional move tMOVCCr - Used to implement the Thumb SELECT_CC DAG operation.
// Expanded by the scheduler into a branch sequence.
let usesCustomDAGSchedInserter = 1 in  // Expanded by the scheduler.
  def tMOVCCr :
  PseudoInst<(ops GPR:$dst, GPR:$false, GPR:$true, CCOp:$cc),
              "@ tMOVCCr $cc",
              [(set GPR:$dst, (ARMcmov GPR:$false, GPR:$true, imm:$cc))]>;

// tLEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def tLEApcrel : TIx2<(ops GPR:$dst, i32imm:$label),
                    !strconcat(!strconcat(".set PCRELV${:uid}, ($label-(",
                                          "${:private}PCRELL${:uid}+4))\n"),
                               !strconcat("\tmov $dst, #PCRELV${:uid}\n",
                                  "${:private}PCRELL${:uid}:\n\tadd $dst, pc")),
                    []>;

def tLEApcrelJT : TIx2<(ops GPR:$dst, i32imm:$label, i32imm:$id),
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
  def tTPsoft  : TIx2<(ops),
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
def : ThumbV5Pat<(ARMcall GPR:$dst), (tBLXr GPR:$dst)>;

// zextload i1 -> zextload i8
def : ThumbPat<(zextloadi1 t_addrmode_s1:$addr),
               (tLDRB t_addrmode_s1:$addr)>;
                  
// extload -> zextload
def : ThumbPat<(extloadi1  t_addrmode_s1:$addr),  (tLDRB t_addrmode_s1:$addr)>;
def : ThumbPat<(extloadi8  t_addrmode_s1:$addr),  (tLDRB t_addrmode_s1:$addr)>;
def : ThumbPat<(extloadi16 t_addrmode_s2:$addr),  (tLDRH t_addrmode_s2:$addr)>;

// truncstore i1 -> truncstore i8
def : ThumbPat<(truncstorei1 GPR:$src, t_addrmode_s1:$dst), 
               (tSTRB GPR:$src, t_addrmode_s1:$dst)>;

// Large immediate handling.

// Two piece imms.
def : ThumbPat<(i32 thumb_immshifted:$src),
               (tLSLri (tMOVi8 (thumb_immshifted_val imm:$src)),
                       (thumb_immshifted_shamt imm:$src))>;

def : ThumbPat<(i32 imm0_255_comp:$src),
               (tMVN (tMOVi8 (imm_comp_XFORM imm:$src)))>;
