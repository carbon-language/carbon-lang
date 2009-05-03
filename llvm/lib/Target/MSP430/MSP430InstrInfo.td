//===- MSP430InstrInfo.td - MSP430 Instruction defs -----------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the MSP430 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "MSP430InstrFormats.td"

//===----------------------------------------------------------------------===//
// Type Constraints.
//===----------------------------------------------------------------------===//
class SDTCisI8<int OpNum> : SDTCisVT<OpNum, i8>;
class SDTCisI16<int OpNum> : SDTCisVT<OpNum, i16>;

//===----------------------------------------------------------------------===//
// Type Profiles.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MSP430 Specific Node Definitions.
//===----------------------------------------------------------------------===//
def MSP430retflag : SDNode<"MSP430ISD::RET_FLAG", SDTNone,
                     [SDNPHasChain, SDNPOptInFlag]>;

def MSP430rra     : SDNode<"MSP430ISD::RRA", SDTIntUnaryOp, []>;

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

def NOP : Pseudo<(outs), (ins), "nop", []>;

//===----------------------------------------------------------------------===//
// Real Instructions
//===----------------------------------------------------------------------===//

// FIXME: Provide proper encoding!
let isReturn = 1, isTerminator = 1 in {
  def RETI : Pseudo<(outs), (ins), "ret", [(MSP430retflag)]>;
}

//===----------------------------------------------------------------------===//
// Move Instructions

// FIXME: Provide proper encoding!
let neverHasSideEffects = 1 in {
def MOV16rr : Pseudo<(outs GR16:$dst), (ins GR16:$src),
                     "mov.w\t{$src, $dst|$dst, $src}",
                     []>;
}

// FIXME: Provide proper encoding!
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def MOV16ri : Pseudo<(outs GR16:$dst), (ins i16imm:$src),
                     "mov.w\t{$src, $dst|$dst, $src}",
                     [(set GR16:$dst, imm:$src)]>;
}

//===----------------------------------------------------------------------===//
// Arithmetic Instructions

let Defs = [SR] in {
let isCommutable = 1 in { // X = ADD Y, Z  == X = ADD Z, Y

// FIXME: Provide proper encoding!
def ADD16rr : Pseudo<(outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                     "add.w\t{$src2, $dst|$dst, $src2}",
                     [(set GR16:$dst, (add GR16:$src1, GR16:$src2)),
                      (implicit SR)]>;
}

// FIXME: Provide proper encoding!
let isTwoAddress = 1 in {
def SAR16r1 : Pseudo<(outs GR16:$dst), (ins GR16:$src),
                     "rra.w\t$dst",
                     [(set GR16:$dst, (MSP430rra GR16:$src)),
                      (implicit SR)]>;
}

} // Defs = [SR]
