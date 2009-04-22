//===- PIC16InstrInfo.td - PIC16 Instruction defs -------------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the PIC16 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PIC16 Specific Type Constraints.
//===----------------------------------------------------------------------===//
class SDTCisI8<int OpNum> : SDTCisVT<OpNum, i8>;
class SDTCisI16<int OpNum> : SDTCisVT<OpNum, i16>;

//===----------------------------------------------------------------------===//
// PIC16 Specific Type Profiles.
//===----------------------------------------------------------------------===//

// Generic type profiles for i8/i16 unary/binary operations.
// Taking one i8 or i16 and producing void.
def SDTI8VoidOp : SDTypeProfile<0, 1, [SDTCisI8<0>]>;
def SDTI16VoidOp : SDTypeProfile<0, 1, [SDTCisI16<0>]>;

// Taking one value and producing an output of same type.
def SDTI8UnaryOp : SDTypeProfile<1, 1, [SDTCisI8<0>, SDTCisI8<1>]>;
def SDTI16UnaryOp : SDTypeProfile<1, 1, [SDTCisI16<0>, SDTCisI16<1>]>;

// Taking two values and producing an output of same type.
def SDTI8BinOp : SDTypeProfile<1, 2, [SDTCisI8<0>, SDTCisI8<1>, SDTCisI8<2>]>;
def SDTI16BinOp : SDTypeProfile<1, 2, [SDTCisI16<0>, SDTCisI16<1>, 
                                       SDTCisI16<2>]>;

// Node specific type profiles.
def SDT_PIC16Load : SDTypeProfile<1, 3, [SDTCisI8<0>, SDTCisI8<1>, 
                                          SDTCisI8<2>, SDTCisI8<3>]>;

def SDT_PIC16Store : SDTypeProfile<0, 4, [SDTCisI8<0>, SDTCisI8<1>, 
                                          SDTCisI8<2>, SDTCisI8<3>]>;

def SDT_PIC16Connect : SDTypeProfile<1, 2, [SDTCisI8<0>, SDTCisI8<1>,
                                            SDTCisI8<2>]>;

// PIC16ISD::CALL type prorile
def SDT_PIC16call : SDTypeProfile<0, -1, [SDTCisInt<0>]>;
def SDT_PIC16callw : SDTypeProfile<1, -1, [SDTCisInt<0>]>;

// PIC16ISD::BRCOND
def SDT_PIC16Brcond: SDTypeProfile<0, 2, 
                                   [SDTCisVT<0, OtherVT>, SDTCisI8<1>]>;

// PIC16ISD::BRCOND
def SDT_PIC16Selecticc: SDTypeProfile<1, 3, 
                                   [SDTCisI8<0>, SDTCisI8<1>, SDTCisI8<2>,
                                    SDTCisI8<3>]>;

//===----------------------------------------------------------------------===//
// PIC16 addressing modes matching via DAG.
//===----------------------------------------------------------------------===//
def diraddr : ComplexPattern<i8, 1, "SelectDirectAddr", [], []>;

//===----------------------------------------------------------------------===//
// PIC16 Specific Node Definitions.
//===----------------------------------------------------------------------===//
def PIC16callseq_start : SDNode<"ISD::CALLSEQ_START", SDTI8VoidOp,
                                [SDNPHasChain, SDNPOutFlag]>;
def PIC16callseq_end   : SDNode<"ISD::CALLSEQ_END", SDTI8VoidOp, 
                                [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

// Low 8-bits of GlobalAddress.
def PIC16Lo : SDNode<"PIC16ISD::Lo", SDTI8BinOp>;  

// High 8-bits of GlobalAddress.
def PIC16Hi : SDNode<"PIC16ISD::Hi", SDTI8BinOp>;

// The MTHI and MTLO nodes are used only to match them in the incoming 
// DAG for replacement by corresponding set_fsrhi, set_fsrlo insntructions.
// These nodes are not used for defining any instructions.
def MTLO     : SDNode<"PIC16ISD::MTLO", SDTI8UnaryOp>;
def MTHI     : SDNode<"PIC16ISD::MTHI", SDTI8UnaryOp>;
def MTPCLATH : SDNode<"PIC16ISD::MTPCLATH", SDTI8UnaryOp>;

// Node to generate Bank Select for a GlobalAddress.
def Banksel : SDNode<"PIC16ISD::Banksel", SDTI8UnaryOp>;

// Node to match a direct store operation.
def PIC16Store : SDNode<"PIC16ISD::PIC16Store", SDT_PIC16Store, [SDNPHasChain]>;
def PIC16StWF : SDNode<"PIC16ISD::PIC16StWF", SDT_PIC16Store, 
                       [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;

// Node to match a direct load operation.
def PIC16Load  : SDNode<"PIC16ISD::PIC16Load", SDT_PIC16Load, [SDNPHasChain]>;
def PIC16LdArg  : SDNode<"PIC16ISD::PIC16LdArg", SDT_PIC16Load, [SDNPHasChain]>;
def PIC16LdWF  : SDNode<"PIC16ISD::PIC16LdWF", SDT_PIC16Load, 
                       [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;
def PIC16Connect: SDNode<"PIC16ISD::PIC16Connect", SDT_PIC16Connect, []>;

// Node to match PIC16 call
def PIC16call : SDNode<"PIC16ISD::CALL", SDT_PIC16call,
                              [SDNPHasChain , SDNPOptInFlag, SDNPOutFlag]>;
def PIC16callw : SDNode<"PIC16ISD::CALLW", SDT_PIC16callw,
                              [SDNPHasChain , SDNPOptInFlag, SDNPOutFlag]>;

// Node to match a comparison instruction.
def PIC16Subcc : SDNode<"PIC16ISD::SUBCC", SDTI8BinOp, [SDNPOutFlag]>;

// Node to match a conditional branch.
def PIC16Brcond : SDNode<"PIC16ISD::BRCOND", SDT_PIC16Brcond, 
                         [SDNPHasChain, SDNPInFlag]>;

def PIC16Selecticc : SDNode<"PIC16ISD::SELECT_ICC", SDT_PIC16Selecticc, 
                         [SDNPInFlag]>;

//===----------------------------------------------------------------------===//
// PIC16 Operand Definitions.
//===----------------------------------------------------------------------===//
def i8mem : Operand<i8>;
def brtarget: Operand<OtherVT>;

// Operand for printing out a condition code.
let PrintMethod = "printCCOperand" in
  def CCOp : Operand<i8>;

include "PIC16InstrFormats.td"

//===----------------------------------------------------------------------===//
// PIC16 Common Classes.
//===----------------------------------------------------------------------===//

// W = W Op F : Load the value from F and do Op to W.
let isTwoAddress = 1 in
class BinOpFW<bits<6> OpCode, string OpcStr, SDNode OpNode>:
  ByteFormat<OpCode, (outs GPR:$dst),
             (ins GPR:$src, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
              !strconcat(OpcStr, " $ptrlo + $offset, W"),
             [(set GPR:$dst, (OpNode GPR:$src, (PIC16Load diraddr:$ptrlo,
                                             (i8 imm:$ptrhi),
                                             (i8 imm:$offset))))]>;

// F = F Op W : Load the value from F, do op with W and store in F.
// This insn class is not marked as TwoAddress because the reg is
// being used as a source operand only. (Remember a TwoAddress insn
// needs a copyRegToReg.)
class BinOpWF<bits<6> OpCode, string OpcStr, SDNode OpNode>:
  ByteFormat<OpCode, (outs),
             (ins GPR:$src, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
              !strconcat(OpcStr, " $ptrlo + $offset"),
             [(PIC16Store (OpNode GPR:$src, (PIC16Load diraddr:$ptrlo,
                                             (i8 imm:$ptrhi),
                                             (i8 imm:$offset))),
                                             diraddr:$ptrlo,
                                             (i8 imm:$ptrhi), (i8 imm:$offset)
                                             )]>;

// W = W Op L : Do Op of L with W and place result in W.
let isTwoAddress = 1 in
class BinOpLW<bits<6> opcode, string OpcStr, SDNode OpNode> :
  LiteralFormat<opcode, (outs GPR:$dst),
                (ins GPR:$src, i8imm:$literal),
                !strconcat(OpcStr, " $literal"),
                [(set GPR:$dst, (OpNode GPR:$src, (i8 imm:$literal)))]>;

//===----------------------------------------------------------------------===//
// PIC16 Instructions.
//===----------------------------------------------------------------------===//

// Pseudo-instructions.
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i8imm:$amt),
                       "!ADJCALLSTACKDOWN $amt",
                       [(PIC16callseq_start imm:$amt)]>;

def ADJCALLSTACKUP : Pseudo<(outs), (ins i8imm:$amt),
                       "!ADJCALLSTACKUP $amt", 
                       [(PIC16callseq_end imm:$amt)]>;

//-----------------------------------
// Vaious movlw insn patterns.
//-----------------------------------
let isReMaterializable = 1 in {
// Move 8-bit literal to W.
def movlw : BitFormat<12, (outs GPR:$dst), (ins i8imm:$src),
                      "movlw $src",
                      [(set GPR:$dst, (i8 imm:$src))]>;

// Move a Lo(TGA) to W.
def movlw_lo_1 : BitFormat<12, (outs GPR:$dst), (ins i8imm:$src, i8imm:$src2),
                      "movlw LOW(${src}) + ${src2}",
                      [(set GPR:$dst, (PIC16Lo tglobaladdr:$src, imm:$src2 ))]>;

// Move a Lo(TES) to W.
def movlw_lo_2 : BitFormat<12, (outs GPR:$dst), (ins i8imm:$src, i8imm:$src2),
                      "movlw LOW(${src}) + ${src2}",
                      [(set GPR:$dst, (PIC16Lo texternalsym:$src, imm:$src2 ))]>;

// Move a Hi(TGA) to W.
def movlw_hi_1 : BitFormat<12, (outs GPR:$dst), (ins i8imm:$src, i8imm:$src2),
                      "movlw HIGH(${src}) + ${src2}",
                      [(set GPR:$dst, (PIC16Hi tglobaladdr:$src, imm:$src2))]>;

// Move a Hi(TES) to W.
def movlw_hi_2 : BitFormat<12, (outs GPR:$dst), (ins i8imm:$src, i8imm:$src2),
                      "movlw HIGH(${src}) + ${src2}",
                      [(set GPR:$dst, (PIC16Hi texternalsym:$src, imm:$src2))]>;
}

//-------------------
// FSR setting insns. 
//-------------------
// These insns are matched via a DAG replacement pattern.
def set_fsrlo:
  ByteFormat<0, (outs FSR16:$fsr), 
             (ins GPR:$val),
             "movwf ${fsr}L",
             []>;

let isTwoAddress = 1 in
def set_fsrhi:
  ByteFormat<0, (outs FSR16:$dst), 
             (ins FSR16:$src, GPR:$val),
             "movwf ${dst}H",
             []>;

def set_pclath:
  ByteFormat<0, (outs PCLATHR:$dst), 
             (ins GPR:$val),
             "movwf ${dst}",
             [(set PCLATHR:$dst , (MTPCLATH GPR:$val))]>;

//----------------------------
// copyRegToReg 
// copyRegToReg insns. These are dummy. They should always be deleted
// by the optimizer and never be present in the final generated code.
// if they are, then we have to write correct macros for these insns.
//----------------------------
def copy_fsr:
  Pseudo<(outs FSR16:$dst), (ins FSR16:$src), "copy_fsr $dst, $src", []>;

def copy_w:
  Pseudo<(outs GPR:$dst), (ins GPR:$src), "copy_w $dst, $src", []>;

class SAVE_FSR<string OpcStr>:
  Pseudo<(outs), 
         (ins FSR16:$src, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi), 
         !strconcat(OpcStr, " $ptrlo, $offset"),
         []>; 
 
def save_fsr0: SAVE_FSR<"save_fsr0">;
def save_fsr1: SAVE_FSR<"save_fsr1">;

class RESTORE_FSR<string OpcStr>:
  Pseudo<(outs FSR16:$dst), 
         (ins i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi), 
         !strconcat(OpcStr, " $ptrlo, $offset"),
         []>; 

def restore_fsr0: RESTORE_FSR<"restore_fsr0">;
def restore_fsr1: RESTORE_FSR<"restore_fsr1">;

//--------------------------
// Store to memory
//-------------------------

// Direct store.
// Input operands are: val = W, ptrlo = GA, offset = offset, ptrhi = banksel.
class MOVWF_INSN<bits<6> OpCode, SDNode OpNodeDest, SDNode Op>:
  ByteFormat<0, (outs), 
             (ins GPR:$val, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
             "movwf ${ptrlo} + ${offset}",
             [(Op GPR:$val, OpNodeDest:$ptrlo, (i8 imm:$ptrhi), 
               (i8 imm:$offset))]>;

// Store W to a Global Address.
def movwf : MOVWF_INSN<0, tglobaladdr, PIC16Store>;

// Store W to an External Symobol.
def movwf_1 : MOVWF_INSN<0, texternalsym, PIC16Store>;

// Store with InFlag and OutFlag
// This is same as movwf_1 but has a flag. A flag is required to 
// order the stores while passing the params to function.
def movwf_2 : MOVWF_INSN<0, texternalsym, PIC16StWF>;

// Indirect store. Matched via a DAG replacement pattern.
def store_indirect : 
  ByteFormat<0, (outs), 
             (ins GPR:$val, FSR16:$fsr, i8imm:$offset),
             "movwi $offset[$fsr]",
             []>;

//----------------------------
// Load from memory
//----------------------------
// Direct load.
// Input Operands are: ptrlo = GA, offset = offset, ptrhi = banksel.
// Output: dst = W
class MOVF_INSN<bits<6> OpCode, SDNode OpNodeSrc, SDNode Op>:
  ByteFormat<0, (outs GPR:$dst), 
             (ins i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
             "movf ${ptrlo} + ${offset}, W",
             [(set GPR:$dst, 
               (Op OpNodeSrc:$ptrlo, (i8 imm:$ptrhi),
               (i8 imm:$offset)))]>;

// Load from a GA.
def movf : MOVF_INSN<0, tglobaladdr, PIC16Load>;

// Load from an ES.
def movf_1 : MOVF_INSN<0, texternalsym, PIC16Load>;
def movf_1_1 : MOVF_INSN<0, texternalsym, PIC16LdArg>;

// Load with InFlag and OutFlag
// This is same as movf_1 but has a flag. A flag is required to 
// order the loads while copying the return value of a function.
def movf_2 : MOVF_INSN<0, texternalsym, PIC16LdWF>;

// Indirect load. Matched via a DAG replacement pattern.
def load_indirect : 
  ByteFormat<0, (outs GPR:$dst), 
             (ins FSR16:$fsr, i8imm:$offset),
             "moviw $offset[$fsr]",
             []>;

//-------------------------
// Bitwise operations patterns
//--------------------------
// W = W op [F]
let Defs = [STATUS] in {
def OrFW :  BinOpFW<0, "iorwf", or>;
def XOrFW : BinOpFW<0, "xorwf", xor>;
def AndFW : BinOpFW<0, "andwf", and>;

// F = W op [F]
def OrWF :  BinOpWF<0, "iorwf", or>;
def XOrWF : BinOpWF<0, "xorwf", xor>;
def AndWF : BinOpWF<0, "andwf", and>;

//-------------------------
// Various add/sub patterns.
//-------------------------

// W = W + [F]
def addfw_1: BinOpFW<0, "addwf", add>;
def addfw_2: BinOpFW<0, "addwf", addc>;

let Uses = [STATUS] in
def addfwc: BinOpFW<0, "addwfc", adde>;  // With Carry.

// F = W + [F]
def addwf_1: BinOpWF<0, "addwf", add>;
def addwf_2: BinOpWF<0, "addwf", addc>;
let Uses = [STATUS] in
def addwfc: BinOpWF<0, "addwfc", adde>;  // With Carry.
}

// W -= [F] ; load from F and sub the value from W.
let isTwoAddress = 1 in
class SUBFW<bits<6> OpCode, string OpcStr, SDNode OpNode>:
  ByteFormat<OpCode, (outs GPR:$dst),
             (ins GPR:$src, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
              !strconcat(OpcStr, " $ptrlo + $offset, W"),
             [(set GPR:$dst, (OpNode (PIC16Load diraddr:$ptrlo,
                                      (i8 imm:$ptrhi), (i8 imm:$offset)),
                                      GPR:$src))]>;
let Defs = [STATUS] in {
def subfw_1: SUBFW<0, "subwf", sub>;
def subfw_2: SUBFW<0, "subwf", subc>;

let Uses = [STATUS] in
def subfwb: SUBFW<0, "subwfb", sube>;  // With Borrow.

def subfw_cc: SUBFW<0, "subwf", PIC16Subcc>;
}

// [F] -= W ; 
class SUBWF<bits<6> OpCode, string OpcStr, SDNode OpNode>:
  ByteFormat<OpCode, (outs),
             (ins GPR:$src, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
              !strconcat(OpcStr, " $ptrlo + $offset"),
             [(PIC16Store (OpNode (PIC16Load diraddr:$ptrlo,
                                      (i8 imm:$ptrhi), (i8 imm:$offset)),
                                      GPR:$src), diraddr:$ptrlo,
                                      (i8 imm:$ptrhi), (i8 imm:$offset))]>;

let Defs = [STATUS] in {
def subwf_1: SUBWF<0, "subwf", sub>;
def subwf_2: SUBWF<0, "subwf", subc>;

let Uses = [STATUS] in
  def subwfb: SUBWF<0, "subwfb", sube>;  // With Borrow.

def subwf_cc: SUBWF<0, "subwf", PIC16Subcc>;
}

// addlw 
let Defs = [STATUS] in {
def addlw_1 : BinOpLW<0, "addlw", add>;
def addlw_2 : BinOpLW<0, "addlw", addc>;

let Uses = [STATUS] in
def addlwc : BinOpLW<0, "addlwc", adde>; // With Carry. (Assembler macro).

// bitwise operations involving a literal and w.
def andlw : BinOpLW<0, "andlw", and>;
def xorlw : BinOpLW<0, "xorlw", xor>;
def orlw  : BinOpLW<0, "iorlw", or>;
}

// sublw 
// W = C - W ; sub W from literal. (Without borrow).
let isTwoAddress = 1 in
class SUBLW<bits<6> opcode, SDNode OpNode> :
  LiteralFormat<opcode, (outs GPR:$dst),
                (ins GPR:$src, i8imm:$literal),
                "sublw $literal",
                [(set GPR:$dst, (OpNode (i8 imm:$literal), GPR:$src))]>;

let Defs = [STATUS] in {
def sublw_1 : SUBLW<0, sub>;
def sublw_2 : SUBLW<0, subc>;
def sublw_cc : SUBLW<0, PIC16Subcc>;
}

// Call instruction.
let isCall = 1,
    Defs = [W, FSR0, FSR1] in {
    def CALL: LiteralFormat<0x1, (outs), (ins i8imm:$func),
            //"call ${func} + 2",
            "call ${func}",
            [(PIC16call diraddr:$func)]>;
}

let isCall = 1,
    Defs = [W, FSR0, FSR1] in {
    def CALL_1: LiteralFormat<0x1, (outs), (ins GPR:$func, PCLATHR:$pc),
            "callw",
            [(PIC16call (PIC16Connect GPR:$func, PCLATHR:$pc))]>;
}

let isCall = 1,
    Defs = [FSR0, FSR1] in {
    def CALLW: LiteralFormat<0x1, (outs GPR:$dest), 
                                  (ins GPR:$func, PCLATHR:$pc),
            "callw",
            [(set GPR:$dest, (PIC16callw (PIC16Connect GPR:$func, PCLATHR:$pc)))]>;
}

let Uses = [STATUS] in
def pic16brcond: ControlFormat<0x0, (outs), (ins brtarget:$dst, CCOp:$cc),
                          "b$cc $dst",
                          [(PIC16Brcond bb:$dst, imm:$cc)]>;

// Unconditional branch.
def br_uncond: ControlFormat<0x0, (outs), (ins brtarget:$dst),
                          "goto $dst",
                          [(br bb:$dst)]>;

// SELECT_CC_* - Used to implement the SELECT_CC DAG operation.  Expanded by the
// scheduler into a branch sequence.
let usesCustomDAGSchedInserter = 1 in {   // Expanded by the scheduler.
  def SELECT_CC_Int_ICC
   : Pseudo<(outs GPR:$dst), (ins GPR:$T, GPR:$F, i8imm:$Cond),
            "; SELECT_CC_Int_ICC PSEUDO!",
            [(set GPR:$dst, (PIC16Selecticc GPR:$T, GPR:$F,
                                             imm:$Cond))]>;
}


// Banksel.
let isReMaterializable = 1 in {
def banksel : 
  Pseudo<(outs BSR:$dst),
         (ins i8mem:$ptr),
         "banksel $ptr",
         [(set BSR:$dst, (Banksel tglobaladdr:$ptr))]>;
}

// Return insn.
def Return : 
  ControlFormat<0, (outs), (ins), "return", [(ret)]>;

//===----------------------------------------------------------------------===//
// PIC16 Replacment Patterns.
//===----------------------------------------------------------------------===//

// Identify an indirect store and select insns for it.
def : Pat<(PIC16Store GPR:$val, (MTLO GPR:$loaddr), (MTHI GPR:$hiaddr), 
           imm:$offset),
          (store_indirect GPR:$val, 
           (set_fsrhi (set_fsrlo GPR:$loaddr), GPR:$hiaddr),
           imm:$offset)>;

def : Pat<(PIC16StWF GPR:$val, (MTLO GPR:$loaddr), (MTHI GPR:$hiaddr), 
           imm:$offset),
          (store_indirect GPR:$val, 
           (set_fsrhi (set_fsrlo GPR:$loaddr), GPR:$hiaddr),
           imm:$offset)>;

// Identify an indirect load and select insns for it.
def : Pat<(PIC16Load (MTLO GPR:$loaddr), (MTHI GPR:$hiaddr), 
           imm:$offset),
          (load_indirect  (set_fsrhi (set_fsrlo GPR:$loaddr), GPR:$hiaddr),
           imm:$offset)>;

def : Pat<(PIC16LdWF (MTLO GPR:$loaddr), (MTHI GPR:$hiaddr), 
           imm:$offset),
          (load_indirect  (set_fsrhi (set_fsrlo GPR:$loaddr), GPR:$hiaddr),
           imm:$offset)>;

