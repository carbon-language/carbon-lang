//===- SystemZInstrInfo.td - SystemZ Instruction defs ---------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the SystemZ instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "SystemZInstrFormats.td"

//===----------------------------------------------------------------------===//
// SystemZ Specific Node Definitions.
//===----------------------------------------------------------------------===//
def SystemZretflag : SDNode<"SystemZISD::RET_FLAG", SDTNone,
                     [SDNPHasChain, SDNPOptInFlag]>;

let neverHasSideEffects = 1 in
def NOP : Pseudo<(outs), (ins), "# no-op", []>;

//===----------------------------------------------------------------------===//
//  Control Flow Instructions...
//

// FIXME: Provide proper encoding!
let isReturn = 1, isTerminator = 1 in {
  def RET : Pseudo<(outs), (ins), "br\t%r14", [(SystemZretflag)]>;
}

//===----------------------------------------------------------------------===//
// Move Instructions

// FIXME: Provide proper encoding!
let neverHasSideEffects = 1 in {
def MOV64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src),
                     "lgr\t{$dst, $src}",
                     []>;
}

// FIXME: Provide proper encoding!
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def MOV64ri : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                     "lghi\t{$dst, $src}",
                     [(set GR64:$dst, imm:$src)]>;
}

//===----------------------------------------------------------------------===//
// Arithmetic Instructions

let isTwoAddress = 1 in {

let Defs = [PSW] in {

let isCommutable = 1 in { // X = ADD Y, Z  == X = ADD Z, Y
// FIXME: Provide proper encoding!
def ADD64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "agr\t{$dst, $src2}",
                     [(set GR64:$dst, (add GR64:$src1, GR64:$src2)),
                      (implicit PSW)]>;
}

// FIXME: Provide proper encoding!
def ADD64ri : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                     "aghi\t{$dst, $src2}",
                     [(set GR64:$dst, (add GR64:$src1, imm:$src2)),
                      (implicit PSW)]>;

let isCommutable = 1 in { // X = OR Y, Z  == X = OR Z, Y
// FIXME: Provide proper encoding!
def OR64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "ogr\t{$dst, $src2}",
                    [(set GR64:$dst, (or GR64:$src1, GR64:$src2))]>;
}
// FIXME: provide patterns for masked or-with-imm

let isCommutable = 1 in { // X = XOR Y, Z  == X = XOR Z, Y
// FIXME: Provide proper encoding!
def XOR64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "xgr\t{$dst, $src2}",
                     [(set GR64:$dst, (xor GR64:$src1, GR64:$src2))]>;
}

} // Defs = [PSW]
} // isTwoAddress = 1
