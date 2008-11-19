//===- PIC16InstrInfo.td - PIC16 Instruction defs -------------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the ARM instructions in TableGen format.
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
                                [SDNPHasChain, SDNPOutFlag]>;

// Low 8-bits of GlobalAddress.
def PIC16Lo : SDNode<"PIC16ISD::Lo", SDTI8UnaryOp>;  

// High 8-bits of GlobalAddress.
def PIC16Hi : SDNode<"PIC16ISD::Hi", SDTI8UnaryOp>;

// The MTHI and MTLO nodes are used only to match them in the incoming 
// DAG for replacement by corresponding set_fsrhi, set_fsrlo insntructions.
// These nodes are not used for defining any instructions.
def MTLO : SDNode<"PIC16ISD::MTLO", SDTI8UnaryOp>;
def MTHI : SDNode<"PIC16ISD::MTHI", SDTI8UnaryOp>;

// Node to generate Bank Select for a GlobalAddress.
def Banksel : SDNode<"PIC16ISD::Banksel", SDTI8UnaryOp>;

// Node to match a direct store operation.
def PIC16Store : SDNode<"PIC16ISD::PIC16Store", SDT_PIC16Store, [SDNPHasChain]>;

// Node to match a direct load operation.
def PIC16Load : SDNode<"PIC16ISD::PIC16Load", SDT_PIC16Load, [SDNPHasChain]>;

//===----------------------------------------------------------------------===//
// PIC16 Operand Definitions.
//===----------------------------------------------------------------------===//
def i8mem : Operand<i8>;



//===----------------------------------------------------------------------===//
// PIC16 Instructions.
//===----------------------------------------------------------------------===//
include "PIC16InstrFormats.td"

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
def movlw_lo : BitFormat<12, (outs GPR:$dst), (ins i8imm:$src),
                      "movlw LOW(${src})",
                      [(set GPR:$dst, (PIC16Lo tglobaladdr:$src))]>;

// Move a Hi(TGA) to W.
def movlw_hi : BitFormat<12, (outs GPR:$dst), (ins i8imm:$src),
                      "movlw HIGH(${src})",
                      [(set GPR:$dst, (PIC16Hi tglobaladdr:$src))]>;
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

def copy_fsr:
  Pseudo<(outs FSR16:$dst), (ins FSR16:$src), "copy_fsr $dst, $src", []>;

//--------------------------
// Store to memory
//-------------------------
// Direct store.
def movwf : 
  ByteFormat<0, (outs), 
             (ins GPR:$val, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
             "movwf ${ptrlo} + ${offset}",
             [(PIC16Store GPR:$val, tglobaladdr:$ptrlo, (i8 imm:$ptrhi), 
               (i8 imm:$offset))]>;

def movwf_1 : 
  ByteFormat<0, (outs), 
             (ins GPR:$val, i8mem:$ptrlo, i8imm:$ptrhi, i8imm:$offset),
             "movwf ${ptrlo} + ${offset}",
             [(PIC16Store GPR:$val, texternalsym:$ptrlo, (i8 imm:$ptrhi), 
               (i8 imm:$offset))]>;

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
def movf : 
  ByteFormat<0, (outs GPR:$dst), 
             (ins i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
             "movf ${ptrlo} + ${offset}, W",
             [(set GPR:$dst, 
               (PIC16Load tglobaladdr:$ptrlo, (i8 imm:$ptrhi),
               (i8 imm:$offset)))]>;

def movf_1 : 
  ByteFormat<0, (outs GPR:$dst), 
             (ins i8mem:$ptrlo, i8imm:$ptrhi, i8imm:$offset),
             "movf ${ptrlo} + ${offset}, W",
             [(set GPR:$dst, 
               (PIC16Load texternalsym:$ptrlo, (i8 imm:$ptrhi),
               (i8 imm:$offset)))]>;

// Indirect load. Matched via a DAG replacement pattern.
def load_indirect : 
  ByteFormat<0, (outs GPR:$dst), 
             (ins FSR16:$fsr, i8imm:$offset),
             "moviw $offset[$fsr]",
             []>;

//-------------------------
// Various add/sub patterns.
//-------------------------
// W += [F] ; load from F and add the value to W. 
class ADDFW<bits<6> OpCode, string OpcStr, SDNode OpNode>:
  ByteFormat<OpCode, (outs GPR:$dst),
             (ins GPR:$src, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
              !strconcat(OpcStr, " $ptrlo + $offset, W"),
             [(set GPR:$dst, (OpNode GPR:$src, (PIC16Load diraddr:$ptrlo,
                                             (i8 imm:$ptrhi),
                                             (i8 imm:$offset))))]>;
// let isTwoAddress = 1 in {
def addfw_1: ADDFW<0, "addwf", add>;
def addfw_2: ADDFW<0, "addwf", addc>;
def addfwc: ADDFW<0, "addwfc", adde>;  // With Carry.
// }

// [F] += W ; add the value of  W to [F]. 
class ADDWF<bits<6> OpCode, string OpcStr, SDNode OpNode>:
  ByteFormat<OpCode, (outs),
             (ins GPR:$src, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
              !strconcat(OpcStr, " $ptrlo + $offset"),
             [(PIC16Store (OpNode GPR:$src, (PIC16Load diraddr:$ptrlo,
                                             (i8 imm:$ptrhi),
                                             (i8 imm:$offset))),
                                             diraddr:$ptrlo, 
                                             (i8 imm:$ptrhi), (i8 imm:$offset)
                                             )]>;
def addwf_1: ADDWF<0, "addwf", add>;
def addwf_2: ADDWF<0, "addwf", addc>;
def addwfc: ADDWF<0, "addwfc", adde>;  // With Carry.

// W -= [F] ; load from F and sub the value from W.
class SUBFW<bits<6> OpCode, string OpcStr, SDNode OpNode>:
  ByteFormat<OpCode, (outs GPR:$dst),
             (ins GPR:$src, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
              !strconcat(OpcStr, " $ptrlo + $offset, W"),
             [(set GPR:$dst, (OpNode (PIC16Load diraddr:$ptrlo,
                                      (i8 imm:$ptrhi), (i8 imm:$offset)),
                                      GPR:$src))]>;
//let isTwoAddress = 1 in {
def subfw_1: SUBFW<0, "subwf", sub>;
def subfw_2: SUBFW<0, "subwf", subc>;
def subfwb: SUBFW<0, "subwfb", sube>;  // With Borrow.
//}

// [F] -= W ; 
class SUBWF<bits<6> OpCode, string OpcStr, SDNode OpNode>:
  ByteFormat<OpCode, (outs),
             (ins GPR:$src, i8imm:$offset, i8mem:$ptrlo, i8imm:$ptrhi),
              !strconcat(OpcStr, " $ptrlo + $offset"),
             [(PIC16Store (OpNode (PIC16Load diraddr:$ptrlo,
                                      (i8 imm:$ptrhi), (i8 imm:$offset)),
                                      GPR:$src), diraddr:$ptrlo,
                                      (i8 imm:$ptrhi), (i8 imm:$offset))]>;

def subwf_1: SUBWF<0, "subwf", sub>;
def subwf_2: SUBWF<0, "subwf", subc>;
def subwfb: SUBWF<0, "subwfb", sube>;  // With Borrow.

// addlw 
// W += C ; add literal to W. (Without carry). May Produce a carry.
class ADDLW<bits<6> opcode, string OpcStr, SDNode OpNode> :
  LiteralFormat<opcode, (outs GPR:$dst),
                (ins GPR:$src, i8imm:$literal),
                !strconcat(OpcStr, " $literal"),
                [(set GPR:$dst, (OpNode GPR:$src, (i8 imm:$literal)))]>;

// let isTwoAddress = 1 in {
def addlw_1 : ADDLW<0, "addlw", add>;
def addlw_2 : ADDLW<0, "addlw", addc>;
def addlwc : ADDLW<0, "addlwc", adde>; // With Carry. (Assembler macro).
//}

// sublw 
// W = C - W ; sub W from literal. (Without borrow).
class SUBLW<bits<6> opcode, SDNode OpNode> :
  LiteralFormat<opcode, (outs GPR:$dst),
                (ins GPR:$src, i8imm:$literal),
                "addlw $literal",
                [(set GPR:$dst, (OpNode (i8 imm:$literal), GPR:$src))]>;

//let isTwoAddress = 1 in {
def sublw_1 : SUBLW<0, sub>;
def sublw_2 : SUBLW<0, subc>;
//}

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

// Identify an indirect load and select insns for it.
def : Pat<(PIC16Load (MTLO GPR:$loaddr), (MTHI GPR:$hiaddr), 
           imm:$offset),
          (load_indirect  (set_fsrhi (set_fsrlo GPR:$loaddr), GPR:$hiaddr),
           imm:$offset)>;

