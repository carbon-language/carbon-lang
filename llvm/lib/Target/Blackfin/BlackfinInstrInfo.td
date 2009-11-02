//===- BlackfinInstrInfo.td - Target Description for Blackfin Target ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Blackfin instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

include "BlackfinInstrFormats.td"

// These are target-independent nodes, but have target-specific formats.
def SDT_BfinCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32> ]>;
def SDT_BfinCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32> ]>;

def BfinCallseqStart : SDNode<"ISD::CALLSEQ_START", SDT_BfinCallSeqStart,
                              [SDNPHasChain, SDNPOutFlag]>;
def BfinCallseqEnd   : SDNode<"ISD::CALLSEQ_END",   SDT_BfinCallSeqEnd,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def SDT_BfinCall  : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;
def BfinCall      : SDNode<"BFISD::CALL", SDT_BfinCall,
                           [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def BfinRet: SDNode<"BFISD::RET_FLAG", SDTNone,
                    [SDNPHasChain, SDNPOptInFlag]>;

def BfinWrapper: SDNode<"BFISD::Wrapper", SDTIntUnaryOp>;

//===----------------------------------------------------------------------===//
// Transformations
//===----------------------------------------------------------------------===//

def trailingZeros_xform : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getAPIntValue().countTrailingZeros(),
                                   MVT::i32);
}]>;

def trailingOnes_xform : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getAPIntValue().countTrailingOnes(),
                                   MVT::i32);
}]>;

def LO16 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((unsigned short)N->getZExtValue(), MVT::i16);
}]>;

def HI16 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return CurDAG->getTargetConstant((unsigned)N->getZExtValue() >> 16, MVT::i16);
}]>;

//===----------------------------------------------------------------------===//
// Immediates
//===----------------------------------------------------------------------===//

def imm3  : PatLeaf<(imm), [{return isInt<3>(N->getSExtValue());}]>;
def uimm3 : PatLeaf<(imm), [{return isUint<3>(N->getZExtValue());}]>;
def uimm4 : PatLeaf<(imm), [{return isUint<4>(N->getZExtValue());}]>;
def uimm5 : PatLeaf<(imm), [{return isUint<5>(N->getZExtValue());}]>;

def uimm5m2 : PatLeaf<(imm), [{
    uint64_t value = N->getZExtValue();
    return value % 2 == 0 && isUint<5>(value);
}]>;

def uimm6m4 : PatLeaf<(imm), [{
    uint64_t value = N->getZExtValue();
    return value % 4 == 0 && isUint<6>(value);
}]>;

def imm7   : PatLeaf<(imm), [{return isInt<7>(N->getSExtValue());}]>;
def imm16  : PatLeaf<(imm), [{return isInt<16>(N->getSExtValue());}]>;
def uimm16 : PatLeaf<(imm), [{return isUint<16>(N->getZExtValue());}]>;

def ximm16 : PatLeaf<(imm), [{
    int64_t value = N->getSExtValue();
    return value < (1<<16) && value >= -(1<<15);
}]>;

def imm17m2 : PatLeaf<(imm), [{
    int64_t value = N->getSExtValue();
    return value % 2 == 0 && isInt<17>(value);
}]>;

def imm18m4 : PatLeaf<(imm), [{
    int64_t value = N->getSExtValue();
    return value % 4 == 0 && isInt<18>(value);
}]>;

// 32-bit bitmask transformed to a bit number
def uimm5mask : Operand<i32>, PatLeaf<(imm), [{
    return isPowerOf2_32(N->getZExtValue());
}], trailingZeros_xform>;

// 32-bit inverse bitmask transformed to a bit number
def uimm5imask : Operand<i32>, PatLeaf<(imm), [{
    return isPowerOf2_32(~N->getZExtValue());
}], trailingOnes_xform>;

//===----------------------------------------------------------------------===//
// Operands
//===----------------------------------------------------------------------===//

def calltarget : Operand<iPTR>;

def brtarget : Operand<OtherVT>;

// Addressing modes
def ADDRspii : ComplexPattern<i32, 2, "SelectADDRspii", [add, frameindex], []>;

// Address operands
def MEMii : Operand<i32> {
  let PrintMethod = "printMemoryOperand";
  let MIOperandInfo = (ops i32imm, i32imm);
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// Pseudo instructions.
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstBfin<outs, ins, asmstr, pattern>;

let Defs = [SP], Uses = [SP] in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt),
                              "${:comment}ADJCALLSTACKDOWN $amt",
                              [(BfinCallseqStart timm:$amt)]>;
def ADJCALLSTACKUP : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                            "${:comment}ADJCALLSTACKUP $amt1 $amt2",
                            [(BfinCallseqEnd timm:$amt1, timm:$amt2)]>;
}

//===----------------------------------------------------------------------===//
// Table C-9. Program Flow Control Instructions
//===----------------------------------------------------------------------===//

let isBranch = 1, isTerminator = 1 in {

let isIndirectBranch = 1 in
def JUMPp : F1<(outs), (ins P:$target),
               "JUMP ($target);",
               [(brind P:$target)]>;

// TODO JUMP (PC-P)

// NOTE: assembler chooses between JUMP.S and JUMP.L
def JUMPa : F1<(outs), (ins brtarget:$target),
               "jump $target;",
               [(br bb:$target)]>;

def JUMPcc : F1<(outs), (ins AnyCC:$cc, brtarget:$target),
               "if $cc jump $target;",
               [(brcond AnyCC:$cc, bb:$target)]>;
}

let isCall = 1,
    Defs   = [R0, R1, R2, R3, P0, P1, P2, LB0, LB1, LC0, LC1, RETS, ASTAT] in {
def CALLa: F1<(outs), (ins calltarget:$func, variable_ops),
              "call $func;", []>;
def CALLp: F1<(outs), (ins P:$func, variable_ops),
              "call ($func);", [(BfinCall P:$func)]>;
}

let isReturn     = 1,
    isTerminator = 1,
    Uses         = [RETS] in
def RTS: F1<(outs), (ins), "rts;", [(BfinRet)]>;

//===----------------------------------------------------------------------===//
// Table C-10. Load / Store Instructions
//===----------------------------------------------------------------------===//

// Immediate constant loads

// sext immediate, i32 D/P regs
def LOADimm7: F1<(outs DP:$dst), (ins i32imm:$src),
                 "$dst = $src (x);",
                 [(set DP:$dst, imm7:$src)]>;

// zext immediate, i32 reg groups 0-3
def LOADuimm16: F2<(outs GR:$dst), (ins i32imm:$src),
                   "$dst = $src (z);",
                   [(set GR:$dst, uimm16:$src)]>;

// sext immediate, i32 reg groups 0-3
def LOADimm16: F2<(outs GR:$dst), (ins i32imm:$src),
                  "$dst = $src (x);",
                  [(set GR:$dst, imm16:$src)]>;

// Pseudo-instruction for loading a general 32-bit constant.
def LOAD32imm: Pseudo<(outs GR:$dst), (ins i32imm:$src),
                      "$dst.h = ($src >> 16); $dst.l = ($src & 0xffff);",
                      [(set GR:$dst, imm:$src)]>;

def LOAD32sym: Pseudo<(outs GR:$dst), (ins i32imm:$src),
                      "$dst.h = $src; $dst.l = $src;", []>;


// 16-bit immediate, i16 reg groups 0-3
def LOAD16i: F2<(outs GR16:$dst), (ins i16imm:$src),
                 "$dst = $src;", []>;

def : Pat<(BfinWrapper (i32 tglobaladdr:$addr)),
          (LOAD32sym tglobaladdr:$addr)>;

def : Pat<(BfinWrapper (i32 tjumptable:$addr)),
          (LOAD32sym tjumptable:$addr)>;

// We cannot copy from GR16 to D16, and codegen wants to insert copies if we
// emit GR16 instructions. As a hack, we use this fake instruction instead.
def LOAD16i_d16: F2<(outs D16:$dst), (ins i16imm:$src),
                    "$dst = $src;",
                    [(set D16:$dst, ximm16:$src)]>;

// Memory loads with patterns

def LOAD32p: F1<(outs DP:$dst), (ins P:$ptr),
                "$dst = [$ptr];",
                [(set DP:$dst, (load P:$ptr))]>;

// Pseudo-instruction for loading a stack slot
def LOAD32fi: Pseudo<(outs DP:$dst), (ins MEMii:$mem),
                     "${:comment}FI $dst = [$mem];",
                     [(set DP:$dst, (load ADDRspii:$mem))]>;

// Note: Expands to multiple insns
def LOAD16fi: Pseudo<(outs D16:$dst), (ins MEMii:$mem),
                     "${:comment}FI $dst = [$mem];",
                     [(set D16:$dst, (load ADDRspii:$mem))]>;

// Pseudo-instruction for loading a stack slot, used for AnyCC regs.
// Replaced with Load D + CC=D
def LOAD8fi: Pseudo<(outs AnyCC:$dst), (ins MEMii:$mem),
                    "${:comment}FI $dst = B[$mem];",
                    [(set AnyCC:$dst, (load ADDRspii:$mem))]>;

def LOAD32p_uimm6m4: F1<(outs DP:$dst), (ins P:$ptr, i32imm:$off),
                        "$dst = [$ptr + $off];",
                        [(set DP:$dst, (load (add P:$ptr, uimm6m4:$off)))]>;

def LOAD32p_imm18m4: F2<(outs DP:$dst), (ins P:$ptr, i32imm:$off),
                         "$dst = [$ptr + $off];",
                         [(set DP:$dst, (load (add P:$ptr, imm18m4:$off)))]>;

def LOAD32p_16z: F1<(outs D:$dst), (ins P:$ptr),
                    "$dst = W[$ptr] (z);",
                    [(set D:$dst, (zextloadi16 P:$ptr))]>;

def : Pat<(i32 (extloadi16 P:$ptr)),(LOAD32p_16z P:$ptr)>;

def LOAD32p_uimm5m2_16z: F1<(outs D:$dst), (ins P:$ptr, i32imm:$off),
                            "$dst = w[$ptr + $off] (z);",
                            [(set D:$dst, (zextloadi16 (add P:$ptr,
                                                        uimm5m2:$off)))]>;

def : Pat<(i32 (extloadi16 (add P:$ptr, uimm5m2:$off))),
          (LOAD32p_uimm5m2_16z P:$ptr, imm:$off)>;

def LOAD32p_imm17m2_16z: F1<(outs D:$dst), (ins P:$ptr, i32imm:$off),
                            "$dst = w[$ptr + $off] (z);",
                            [(set D:$dst,
                                  (zextloadi16 (add P:$ptr, imm17m2:$off)))]>;

def : Pat<(i32 (extloadi16 (add P:$ptr, imm17m2:$off))),
          (LOAD32p_imm17m2_16z P:$ptr, imm:$off)>;

def LOAD32p_16s: F1<(outs D:$dst), (ins P:$ptr),
                    "$dst = w[$ptr] (x);",
                    [(set D:$dst, (sextloadi16 P:$ptr))]>;

def LOAD32p_uimm5m2_16s: F1<(outs D:$dst), (ins P:$ptr, i32imm:$off),
                            "$dst = w[$ptr + $off] (x);",
                            [(set D:$dst,
                                  (sextloadi16 (add P:$ptr, uimm5m2:$off)))]>;

def LOAD32p_imm17m2_16s: F1<(outs D:$dst), (ins P:$ptr, i32imm:$off),
                            "$dst = w[$ptr + $off] (x);",
                            [(set D:$dst,
                                  (sextloadi16 (add P:$ptr, imm17m2:$off)))]>;

def LOAD16pi: F1<(outs D16:$dst), (ins PI:$ptr),
                "$dst = w[$ptr];",
                [(set D16:$dst, (load PI:$ptr))]>;

def LOAD32p_8z: F1<(outs D:$dst), (ins P:$ptr),
                   "$dst = B[$ptr] (z);",
                   [(set D:$dst, (zextloadi8 P:$ptr))]>;

def : Pat<(i32 (extloadi8 P:$ptr)), (LOAD32p_8z P:$ptr)>;
def : Pat<(i16 (extloadi8 P:$ptr)),
          (EXTRACT_SUBREG (LOAD32p_8z P:$ptr), bfin_subreg_lo16)>;
def : Pat<(i16 (zextloadi8 P:$ptr)),
          (EXTRACT_SUBREG (LOAD32p_8z P:$ptr), bfin_subreg_lo16)>;

def LOAD32p_imm16_8z: F1<(outs D:$dst), (ins P:$ptr, i32imm:$off),
                         "$dst = b[$ptr + $off] (z);",
                         [(set D:$dst, (zextloadi8 (add P:$ptr, imm16:$off)))]>;

def : Pat<(i32 (extloadi8 (add P:$ptr, imm16:$off))),
          (LOAD32p_imm16_8z P:$ptr, imm:$off)>;
def : Pat<(i16 (extloadi8 (add P:$ptr, imm16:$off))),
          (EXTRACT_SUBREG (LOAD32p_imm16_8z P:$ptr, imm:$off),
                           bfin_subreg_lo16)>;
def : Pat<(i16 (zextloadi8 (add P:$ptr, imm16:$off))),
          (EXTRACT_SUBREG (LOAD32p_imm16_8z P:$ptr, imm:$off),
                           bfin_subreg_lo16)>;

def LOAD32p_8s: F1<(outs D:$dst), (ins P:$ptr),
                   "$dst = b[$ptr] (x);",
                   [(set D:$dst, (sextloadi8 P:$ptr))]>;

def : Pat<(i16 (sextloadi8 P:$ptr)),
          (EXTRACT_SUBREG (LOAD32p_8s P:$ptr), bfin_subreg_lo16)>;

def LOAD32p_imm16_8s: F1<(outs D:$dst), (ins P:$ptr, i32imm:$off),
                         "$dst = b[$ptr + $off] (x);",
                         [(set D:$dst, (sextloadi8 (add P:$ptr, imm16:$off)))]>;

def : Pat<(i16 (sextloadi8 (add P:$ptr, imm16:$off))),
          (EXTRACT_SUBREG (LOAD32p_imm16_8s P:$ptr, imm:$off),
                           bfin_subreg_lo16)>;
// Memory loads without patterns

let mayLoad = 1 in {

multiclass LOAD_incdec<RegisterClass drc, RegisterClass prc,
                       string mem="", string suf=";"> {
  def _inc : F1<(outs drc:$dst, prc:$ptr_wb), (ins prc:$ptr),
                !strconcat(!subst("M", mem, "$dst = M[$ptr++]"), suf), []>;
  def _dec : F1<(outs drc:$dst, prc:$ptr_wb), (ins prc:$ptr),
                !strconcat(!subst("M", mem, "$dst = M[$ptr--]"), suf), []>;
}
multiclass LOAD_incdecpost<RegisterClass drc, RegisterClass prc,
                           string mem="", string suf=";">
         : LOAD_incdec<drc, prc, mem, suf> {
  def _post : F1<(outs drc:$dst, prc:$ptr_wb), (ins prc:$ptr, prc:$off),
                 !strconcat(!subst("M", mem, "$dst = M[$ptr++$off]"), suf), []>;
}

defm LOAD32p:    LOAD_incdec<DP, P>;
defm LOAD32i:    LOAD_incdec<D, I>;
defm LOAD8z32p:  LOAD_incdec<D, P, "b", " (z);">;
defm LOAD8s32p:  LOAD_incdec<D, P, "b", " (x);">;
defm LOADhi:     LOAD_incdec<D16, I, "w">;
defm LOAD16z32p: LOAD_incdecpost<D, P, "w", " (z);">;
defm LOAD16s32p: LOAD_incdecpost<D, P, "w", " (x);">;

def LOAD32p_post: F1<(outs D:$dst, P:$ptr_wb), (ins P:$ptr, P:$off),
                     "$dst = [$ptr ++ $off];", []>;

// Note: $fp MUST be FP
def LOAD32fp_nimm7m4: F1<(outs DP:$dst), (ins P:$fp, i32imm:$off),
                         "$dst = [$fp - $off];", []>;

def LOAD32i:      F1<(outs D:$dst), (ins I:$ptr),
                     "$dst = [$ptr];", []>;
def LOAD32i_post: F1<(outs D:$dst, I:$ptr_wb), (ins I:$ptr, M:$off),
                     "$dst = [$ptr ++ $off];", []>;



def LOADhp_post: F1<(outs D16:$dst, P:$ptr_wb), (ins P:$ptr, P:$off),
                    "$dst = w[$ptr ++ $off];", []>;


}

// Memory stores with patterns
def STORE32p: F1<(outs), (ins DP:$val, P:$ptr),
                 "[$ptr] = $val;",
                 [(store DP:$val, P:$ptr)]>;

// Pseudo-instructions for storing to a stack slot
def STORE32fi: Pseudo<(outs), (ins DP:$val, MEMii:$mem),
                      "${:comment}FI [$mem] = $val;",
                      [(store DP:$val, ADDRspii:$mem)]>;

// Note: This stack-storing pseudo-instruction is expanded to multiple insns
def STORE16fi: Pseudo<(outs), (ins D16:$val, MEMii:$mem),
                  "${:comment}FI [$mem] = $val;",
                  [(store D16:$val, ADDRspii:$mem)]>;

// Pseudo-instructions for storing AnyCC register to a stack slot.
// Replaced with D=CC + STORE byte
def STORE8fi: Pseudo<(outs), (ins AnyCC:$val, MEMii:$mem),
                      "${:comment}FI b[$mem] = $val;",
                      [(store AnyCC:$val, ADDRspii:$mem)]>;

def STORE32p_uimm6m4: F1<(outs), (ins DP:$val, P:$ptr, i32imm:$off),
                 "[$ptr + $off] = $val;",
                 [(store DP:$val, (add P:$ptr, uimm6m4:$off))]>;

def STORE32p_imm18m4: F1<(outs), (ins DP:$val, P:$ptr, i32imm:$off),
                 "[$ptr + $off] = $val;",
                 [(store DP:$val, (add P:$ptr, imm18m4:$off))]>;

def STORE16pi: F1<(outs), (ins D16:$val, PI:$ptr),
                  "w[$ptr] = $val;",
                  [(store D16:$val, PI:$ptr)]>;

def STORE8p: F1<(outs), (ins D:$val, P:$ptr),
                "b[$ptr] = $val;",
                [(truncstorei8 D:$val, P:$ptr)]>;

def STORE8p_imm16: F1<(outs), (ins D:$val, P:$ptr, i32imm:$off),
                 "b[$ptr + $off] = $val;",
                 [(truncstorei8 D:$val, (add P:$ptr, imm16:$off))]>;

let Constraints = "$ptr = $ptr_wb" in {

multiclass STORE_incdec<RegisterClass drc, RegisterClass prc,
                        int off=4, string pre=""> {
  def _inc : F1<(outs prc:$ptr_wb), (ins drc:$val, prc:$ptr),
                !strconcat(pre, "[$ptr++] = $val;"),
                [(set prc:$ptr_wb, (post_store drc:$val, prc:$ptr, off))]>;
  def _dec : F1<(outs prc:$ptr_wb), (ins drc:$val, prc:$ptr),
                !strconcat(pre, "[$ptr--] = $val;"),
                [(set prc:$ptr_wb, (post_store drc:$val, prc:$ptr,
                                               (ineg off)))]>;
}

defm STORE32p: STORE_incdec<DP, P>;
defm STORE16i: STORE_incdec<D16, I, 2, "w">;
defm STORE8p:  STORE_incdec<D, P, 1, "b">;

def STORE32p_post: F1<(outs P:$ptr_wb), (ins D:$val, P:$ptr, P:$off),
                      "[$ptr ++ $off] = $val;",
                      [(set P:$ptr_wb, (post_store D:$val, P:$ptr, P:$off))]>;

def STORE16p_post: F1<(outs P:$ptr_wb), (ins D16:$val, P:$ptr, P:$off),
                      "w[$ptr ++ $off] = $val;",
                      [(set P:$ptr_wb, (post_store D16:$val, P:$ptr, P:$off))]>;
}

// Memory stores without patterns

let mayStore = 1 in {

// Note: only works for $fp == FP
def STORE32fp_nimm7m4: F1<(outs), (ins DP:$val, P:$fp, i32imm:$off),
                         "[$fp - $off] = $val;", []>;

def STORE32i: F1<(outs), (ins D:$val, I:$ptr),
                 "[$ptr] = $val;", []>;

def STORE32i_inc: F1<(outs I:$ptr_wb), (ins D:$val, I:$ptr),
                 "[$ptr++] = $val;", []>;

def STORE32i_dec: F1<(outs I:$ptr_wb), (ins D:$val, I:$ptr),
                 "[$ptr--] = $val;", []>;

def STORE32i_post: F1<(outs I:$ptr_wb), (ins D:$val, I:$ptr, M:$off),
                      "[$ptr ++ $off] = $val;", []>;
}

def : Pat<(truncstorei16 D:$val, PI:$ptr),
          (STORE16pi (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS D:$val, D)),
                                     bfin_subreg_lo16), PI:$ptr)>;

def : Pat<(truncstorei16 (srl D:$val, (i16 16)), PI:$ptr),
          (STORE16pi (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS D:$val, D)),
                                     bfin_subreg_hi16), PI:$ptr)>;

def : Pat<(truncstorei8 D16L:$val, P:$ptr),
          (STORE8p (INSERT_SUBREG (i32 (IMPLICIT_DEF)),
                                  (i16 (COPY_TO_REGCLASS D16L:$val, D16L)),
                                  bfin_subreg_lo16),
                   P:$ptr)>;

//===----------------------------------------------------------------------===//
// Table C-11. Move Instructions.
//===----------------------------------------------------------------------===//

def MOVE: F1<(outs ALL:$dst), (ins ALL:$src),
             "$dst = $src;",
             []>;

let isTwoAddress = 1 in
def MOVEcc: F1<(outs DP:$dst), (ins DP:$src1, DP:$src2, AnyCC:$cc),
               "if $cc $dst = $src2;",
               [(set DP:$dst, (select AnyCC:$cc, DP:$src2, DP:$src1))]>;

let Defs = [AZ, AN, AC0, V] in {
def MOVEzext: F1<(outs D:$dst), (ins D16L:$src),
                 "$dst = $src (z);",
                 [(set D:$dst, (zext D16L:$src))]>;

def MOVEsext: F1<(outs D:$dst), (ins D16L:$src),
                 "$dst = $src (x);",
                 [(set D:$dst, (sext D16L:$src))]>;

def MOVEzext8: F1<(outs D:$dst), (ins D:$src),
                  "$dst = $src.b (z);",
                  [(set D:$dst, (and D:$src, 0xff))]>;

def MOVEsext8: F1<(outs D:$dst), (ins D:$src),
                  "$dst = $src.b (x);",
                  [(set D:$dst, (sext_inreg D:$src, i8))]>;

}

def : Pat<(sext_inreg D16L:$src, i8),
          (EXTRACT_SUBREG (MOVEsext8
                           (INSERT_SUBREG (i32 (IMPLICIT_DEF)),
                                          D16L:$src,
                                          bfin_subreg_lo16)),
                          bfin_subreg_lo16)>;

def : Pat<(sext_inreg D:$src, i16),
          (MOVEsext (EXTRACT_SUBREG D:$src, bfin_subreg_lo16))>;

def : Pat<(and D:$src, 0xffff),
          (MOVEzext (EXTRACT_SUBREG D:$src, bfin_subreg_lo16))>;

def : Pat<(i32 (anyext D16L:$src)),
          (INSERT_SUBREG (i32 (IMPLICIT_DEF)),
                         (i16 (COPY_TO_REGCLASS D16L:$src, D16L)),
                         bfin_subreg_lo16)>;

// TODO Dreg = Dreg_byte (X/Z)

// TODO Accumulator moves

//===----------------------------------------------------------------------===//
// Table C-12. Stack Control Instructions
//===----------------------------------------------------------------------===//

let Uses = [SP], Defs = [SP] in {
def PUSH: F1<(outs), (ins ALL:$src),
             "[--sp] = $src;", []> { let mayStore = 1; }

// NOTE: POP does not work for DP regs, use LOAD instead
def POP:  F1<(outs ALL:$dst), (ins),
             "$dst = [sp++];", []> { let mayLoad = 1; }
}

// TODO: push/pop multiple

def LINK: F2<(outs), (ins i32imm:$amount),
             "link $amount;", []>;

def UNLINK: F2<(outs), (ins),
               "unlink;", []>;

//===----------------------------------------------------------------------===//
// Table C-13. Control Code Bit Management Instructions
//===----------------------------------------------------------------------===//

multiclass SETCC<PatFrag opnode, PatFrag invnode, string cond, string suf=";"> {
  def dd : F1<(outs JustCC:$cc), (ins D:$a, D:$b),
              !strconcat(!subst("XX", cond, "cc = $a XX $b"), suf),
              [(set JustCC:$cc, (opnode  D:$a, D:$b))]>;

  def ri : F1<(outs JustCC:$cc), (ins DP:$a, i32imm:$b),
              !strconcat(!subst("XX", cond, "cc = $a XX $b"), suf),
              [(set JustCC:$cc, (opnode  DP:$a, imm3:$b))]>;

  def pp : F1<(outs JustCC:$cc), (ins P:$a, P:$b),
              !strconcat(!subst("XX", cond, "cc = $a XX $b"), suf),
              []>;

  def ri_not : F1<(outs NotCC:$cc), (ins DP:$a, i32imm:$b),
                  !strconcat(!subst("XX", cond, "cc = $a XX $b"), suf),
                  [(set NotCC:$cc, (invnode  DP:$a, imm3:$b))]>;
}

defm SETEQ  : SETCC<seteq,  setne,  "==">;
defm SETLT  : SETCC<setlt,  setge,  "<">;
defm SETLE  : SETCC<setle,  setgt,  "<=">;
defm SETULT : SETCC<setult, setuge, "<",  " (iu);">;
defm SETULE : SETCC<setule, setugt, "<=", " (iu);">;

def SETNEdd : F1<(outs NotCC:$cc), (ins D:$a, D:$b),
                 "cc = $a == $b;",
                 [(set NotCC:$cc, (setne  D:$a, D:$b))]>;

def : Pat<(setgt  D:$a, D:$b), (SETLTdd  D:$b, D:$a)>;
def : Pat<(setge  D:$a, D:$b), (SETLEdd  D:$b, D:$a)>;
def : Pat<(setugt D:$a, D:$b), (SETULTdd D:$b, D:$a)>;
def : Pat<(setuge D:$a, D:$b), (SETULEdd D:$b, D:$a)>;

// TODO: compare pointer for P-P comparisons
// TODO: compare accumulator

let Defs = [AC0] in
def OR_ac0_cc : F1<(outs), (ins JustCC:$cc),
                   "ac0 \\|= cc;", []>;

let Uses = [AC0] in
def MOVE_cc_ac0 : F1<(outs JustCC:$cc), (ins),
                   "cc = ac0;", []>;

def MOVE_ccncc : F1<(outs JustCC:$cc), (ins NotCC:$sb),
                    "cc = !cc;", []>;

def MOVE_ncccc : F1<(outs NotCC:$cc), (ins JustCC:$sb),
                    "cc = !cc;", []>;

def MOVECC_zext : F1<(outs D:$dst), (ins JustCC:$cc),
                      "$dst = $cc;",
                      [(set D:$dst, (zext JustCC:$cc))]>;

def MOVENCC_z : F1<(outs D:$dst), (ins NotCC:$cc),
                   "$dst = cc;", []>;

def MOVECC_nz : F1<(outs AnyCC:$cc), (ins D:$src),
                   "cc = $src;",
                   [(set AnyCC:$cc, (setne D:$src, 0))]>;

//===----------------------------------------------------------------------===//
// Table C-14. Logical Operations Instructions
//===----------------------------------------------------------------------===//

def AND: F1<(outs D:$dst), (ins D:$src1, D:$src2),
            "$dst = $src1 & $src2;",
            [(set D:$dst, (and D:$src1, D:$src2))]>;

def NOT: F1<(outs D:$dst), (ins D:$src),
            "$dst = ~$src;",
            [(set D:$dst, (not D:$src))]>;

def OR: F1<(outs D:$dst), (ins D:$src1, D:$src2),
           "$dst = $src1 \\| $src2;",
           [(set D:$dst, (or D:$src1, D:$src2))]>;

def XOR: F1<(outs D:$dst), (ins D:$src1, D:$src2),
            "$dst = $src1 ^ $src2;",
            [(set D:$dst, (xor D:$src1, D:$src2))]>;

// missing: BXOR, BXORSHIFT

//===----------------------------------------------------------------------===//
// Table C-15. Bit Operations Instructions
//===----------------------------------------------------------------------===//

let isTwoAddress = 1 in {
def BITCLR: F1<(outs D:$dst), (ins D:$src1, uimm5imask:$src2),
              "bitclr($dst, $src2);",
              [(set D:$dst, (and D:$src1, uimm5imask:$src2))]>;

def BITSET: F1<(outs D:$dst), (ins D:$src1, uimm5mask:$src2),
              "bitset($dst, $src2);",
              [(set D:$dst, (or D:$src1, uimm5mask:$src2))]>;

def BITTGL: F1<(outs D:$dst), (ins D:$src1, uimm5mask:$src2),
              "bittgl($dst, $src2);",
              [(set D:$dst, (xor D:$src1, uimm5mask:$src2))]>;
}

def BITTST: F1<(outs JustCC:$cc), (ins D:$src1, uimm5mask:$src2),
              "cc = bittst($src1, $src2);",
              [(set JustCC:$cc, (setne (and D:$src1, uimm5mask:$src2),
                                       (i32 0)))]>;

def NBITTST: F1<(outs JustCC:$cc), (ins D:$src1, uimm5mask:$src2),
               "cc = !bittst($src1, $src2);",
               [(set JustCC:$cc, (seteq (and D:$src1, uimm5mask:$src2),
                                        (i32 0)))]>;

// TODO: DEPOSIT, EXTRACT, BITMUX

def ONES: F2<(outs D16L:$dst), (ins D:$src),
              "$dst = ones $src;",
              [(set D16L:$dst, (trunc (ctpop D:$src)))]>;

def : Pat<(ctpop D:$src), (MOVEzext (ONES D:$src))>;

//===----------------------------------------------------------------------===//
// Table C-16. Shift / Rotate Instructions
//===----------------------------------------------------------------------===//

multiclass SHIFT32<SDNode opnode, string ops> {
  def i : F1<(outs D:$dst), (ins D:$src, i16imm:$amount),
             !subst("XX", ops, "$dst XX= $amount;"),
             [(set D:$dst, (opnode D:$src, (i16 uimm5:$amount)))]>;
  def r : F1<(outs D:$dst), (ins D:$src, D:$amount),
             !subst("XX", ops, "$dst XX= $amount;"),
             [(set D:$dst, (opnode D:$src, D:$amount))]>;
}

let Defs = [AZ, AN, V, VS],
    isTwoAddress = 1 in {
defm SRA : SHIFT32<sra, ">>>">;
defm SRL : SHIFT32<srl, ">>">;
defm SLL : SHIFT32<shl, "<<">;
}

// TODO: automatic switching between 2-addr and 3-addr (?)

let Defs = [AZ, AN, V, VS] in {
def SLLr16: F2<(outs D:$dst), (ins D:$src, D16L:$amount),
             "$dst = lshift $src by $amount;",
             [(set D:$dst, (shl D:$src, D16L:$amount))]>;

// Arithmetic left-shift = saturing overflow.
def SLAr16: F2<(outs D:$dst), (ins D:$src, D16L:$amount),
             "$dst = ashift $src by $amount;",
             [(set D:$dst, (sra D:$src, (ineg D16L:$amount)))]>;

def SRA16i: F1<(outs D16:$dst), (ins D16:$src, i16imm:$amount),
              "$dst = $src >>> $amount;",
              [(set D16:$dst, (sra D16:$src, (i16 uimm4:$amount)))]>;

def SRL16i: F1<(outs D16:$dst), (ins D16:$src, i16imm:$amount),
              "$dst = $src >> $amount;",
              [(set D16:$dst, (srl D16:$src, (i16 uimm4:$amount)))]>;

// Arithmetic left-shift = saturing overflow.
def SLA16r: F1<(outs D16:$dst), (ins D16:$src, D16L:$amount),
              "$dst = ashift $src BY $amount;",
              [(set D16:$dst, (srl D16:$src, (ineg D16L:$amount)))]>;

def SLL16i: F1<(outs D16:$dst), (ins D16:$src, i16imm:$amount),
              "$dst = $src << $amount;",
              [(set D16:$dst, (shl D16:$src, (i16 uimm4:$amount)))]>;

def SLL16r: F1<(outs D16:$dst), (ins D16:$src, D16L:$amount),
              "$dst = lshift $src by $amount;",
              [(set D16:$dst, (shl D16:$src, D16L:$amount))]>;

}

//===----------------------------------------------------------------------===//
// Table C-17. Arithmetic Operations Instructions
//===----------------------------------------------------------------------===//

// TODO: ABS

let Defs = [AZ, AN, AC0, V, VS] in {

def ADD: F1<(outs D:$dst), (ins D:$src1, D:$src2),
            "$dst = $src1 + $src2;",
            [(set D:$dst, (add D:$src1, D:$src2))]>;

def ADD16: F2<(outs D16:$dst), (ins D16:$src1, D16:$src2),
              "$dst = $src1 + $src2;",
              [(set D16:$dst, (add D16:$src1, D16:$src2))]>;

let isTwoAddress = 1 in
def ADDimm7: F1<(outs D:$dst), (ins D:$src1, i32imm:$src2),
                "$dst += $src2;",
                [(set D:$dst, (add D:$src1, imm7:$src2))]>;

def SUB: F1<(outs D:$dst), (ins D:$src1, D:$src2),
            "$dst = $src1 - $src2;",
            [(set D:$dst, (sub D:$src1, D:$src2))]>;

def SUB16: F2<(outs D16:$dst), (ins D16:$src1, D16:$src2),
              "$dst = $src1 - $src2;",
              [(set D16:$dst, (sub D16:$src1, D16:$src2))]>;

}

def : Pat<(addc D:$src1, D:$src2), (ADD D:$src1, D:$src2)>;
def : Pat<(subc D:$src1, D:$src2), (SUB D:$src1, D:$src2)>;

let Defs = [AZ, AN, V, VS] in
def NEG: F1<(outs D:$dst), (ins D:$src),
            "$dst = -$src;",
            [(set D:$dst, (ineg D:$src))]>;

// No pattern, it would confuse isel to have two i32 = i32+i32 patterns
def ADDpp: F1<(outs P:$dst), (ins P:$src1, P:$src2),
              "$dst = $src1 + $src2;", []>;

let isTwoAddress = 1 in
def ADDpp_imm7: F1<(outs P:$dst), (ins P:$src1, i32imm:$src2),
                "$dst += $src2;", []>;

let Defs = [AZ, AN, V] in
def ADD_RND20: F2<(outs D16:$dst), (ins D:$src1, D:$src2),
                  "$dst = $src1 + $src2 (rnd20);", []>;

let Defs = [V, VS] in {
def MUL16: F2<(outs D16:$dst), (ins D16:$src1, D16:$src2),
              "$dst = $src1 * $src2 (is);",
              [(set D16:$dst, (mul D16:$src1, D16:$src2))]>;

def MULHS16: F2<(outs D16:$dst), (ins D16:$src1, D16:$src2),
                "$dst = $src1 * $src2 (ih);",
                [(set D16:$dst, (mulhs D16:$src1, D16:$src2))]>;

def MULhh32s: F2<(outs D:$dst), (ins D16:$src1, D16:$src2),
                "$dst = $src1 * $src2 (is);",
                [(set D:$dst, (mul (sext D16:$src1), (sext D16:$src2)))]>;

def MULhh32u: F2<(outs D:$dst), (ins D16:$src1, D16:$src2),
                "$dst = $src1 * $src2 (is);",
                [(set D:$dst, (mul (zext D16:$src1), (zext D16:$src2)))]>;
}


let isTwoAddress = 1 in
def MUL32: F1<(outs D:$dst), (ins D:$src1, D:$src2),
            "$dst *= $src2;",
            [(set D:$dst, (mul D:$src1, D:$src2))]>;

//===----------------------------------------------------------------------===//
// Table C-18. External Exent Management Instructions
//===----------------------------------------------------------------------===//

def IDLE : F1<(outs), (ins), "idle;", [(int_bfin_idle)]>;
def CSYNC : F1<(outs), (ins), "csync;", [(int_bfin_csync)]>;
def SSYNC : F1<(outs), (ins), "ssync;", [(int_bfin_ssync)]>;
def EMUEXCPT : F1<(outs), (ins), "emuexcpt;", []>;
def CLI : F1<(outs D:$mask), (ins), "cli $mask;", []>;
def STI : F1<(outs), (ins D:$mask), "sti $mask;", []>;
def RAISE : F1<(outs), (ins i32imm:$itr), "raise $itr;", []>;
def EXCPT : F1<(outs), (ins i32imm:$exc), "excpt $exc;", []>;
def NOP : F1<(outs), (ins), "nop;", []>;
def MNOP : F2<(outs), (ins), "mnop;", []>;
def ABORT : F1<(outs), (ins), "abort;", []>;

//===----------------------------------------------------------------------===//
// Table C-19. Cache Control Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Table C-20. Video Pixel Operations Instructions
//===----------------------------------------------------------------------===//

def ALIGN8 : F2<(outs D:$dst), (ins D:$src1, D:$src2),
                "$dst = align8($src1, $src2);",
                [(set D:$dst, (or (shl D:$src1, (i32 24)),
                                  (srl D:$src2, (i32 8))))]>;

def ALIGN16 : F2<(outs D:$dst), (ins D:$src1, D:$src2),
                 "$dst = align16($src1, $src2);",
                 [(set D:$dst, (or (shl D:$src1, (i32 16)),
                                   (srl D:$src2, (i32 16))))]>;

def ALIGN24 : F2<(outs D:$dst), (ins D:$src1, D:$src2),
                 "$dst = align16($src1, $src2);",
                 [(set D:$dst, (or (shl D:$src1, (i32 8)),
                                   (srl D:$src2, (i32 24))))]>;

def DISALGNEXCPT : F2<(outs), (ins), "disalignexcpt;", []>;

// TODO: BYTEOP3P, BYTEOP16P, BYTEOP1P, BYTEOP2P, BYTEOP16M, SAA,
//       BYTEPACK, BYTEUNPACK

// Table C-21. Vector Operations Instructions

// Patterns
def : Pat<(BfinCall (i32 tglobaladdr:$dst)),
          (CALLa tglobaladdr:$dst)>;
def : Pat<(BfinCall (i32 texternalsym:$dst)),
          (CALLa texternalsym:$dst)>;

def : Pat<(sext JustCC:$cc),
          (NEG (MOVECC_zext JustCC:$cc))>;
def : Pat<(anyext JustCC:$cc),
          (MOVECC_zext JustCC:$cc)>;
def : Pat<(i16 (zext JustCC:$cc)),
          (EXTRACT_SUBREG (MOVECC_zext JustCC:$cc), bfin_subreg_lo16)>;
def : Pat<(i16 (sext JustCC:$cc)),
          (EXTRACT_SUBREG (NEG (MOVECC_zext JustCC:$cc)), bfin_subreg_lo16)>;
def : Pat<(i16 (anyext JustCC:$cc)),
          (EXTRACT_SUBREG (MOVECC_zext JustCC:$cc), bfin_subreg_lo16)>;

def : Pat<(i16 (trunc D:$src)),
          (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS D:$src, D)), bfin_subreg_lo16)>;
