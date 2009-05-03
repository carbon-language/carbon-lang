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
def retflag : SDNode<"MSP430ISD::RET_FLAG", SDTNone,
                     [SDNPHasChain, SDNPOptInFlag]>;

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

def NOP : Pseudo<(outs), (ins), "nop", []>;

//===----------------------------------------------------------------------===//
// Real Instructions
//===----------------------------------------------------------------------===//

// FIXME: Provide proper encoding!
let isReturn = 1, isTerminator = 1 in {
  def RETI: Pseudo<(outs), (ins), "reti", [(retflag)]>;
}
