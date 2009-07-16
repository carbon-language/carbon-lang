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
// Instruction Pattern Stuff.
//===----------------------------------------------------------------------===//
def LL16 : SDNodeXForm<imm, [{
  // Transformation function: return low 16 bits.
  return getI16Imm(N->getZExtValue() & 0x000000000000FFFFULL);
}]>;

def LH16 : SDNodeXForm<imm, [{
  // Transformation function: return bits 16-31.
  return getI16Imm((N->getZExtValue() & 0x00000000FFFF0000ULL) >> 16);
}]>;

def HL16 : SDNodeXForm<imm, [{
  // Transformation function: return bits 32-47.
  return getI16Imm((N->getZExtValue() & 0x0000FFFF00000000ULL) >> 32);
}]>;

def HH16 : SDNodeXForm<imm, [{
  // Transformation function: return bits 48-63.
  return getI16Imm((N->getZExtValue() & 0xFFFF000000000000ULL) >> 48);
}]>;

def LO32 : SDNodeXForm<imm, [{
  // Transformation function: return low 32 bits.
  return getI32Imm(N->getZExtValue() & 0x00000000FFFFFFFFULL);
}]>;

def HI32 : SDNodeXForm<imm, [{
  // Transformation function: return bits 32-63.
  return getI32Imm(N->getZExtValue() >> 32);
}]>;

def i64ll16 : PatLeaf<(i64 imm), [{  
  // i64ll16 predicate - true if the 64-bit immediate has only rightmost 16
  // bits set.
  return ((N->getZExtValue() & 0x000000000000FFFFULL) == N->getZExtValue());
}], LL16>;

def i64lh16 : PatLeaf<(i64 imm), [{  
  // i64lh16 predicate - true if the 64-bit immediate has only bits 16-31 set.
  return ((N->getZExtValue() & 0x00000000FFFF0000ULL) == N->getZExtValue());
}], LH16>;

def i64hl16 : PatLeaf<(i64 imm), [{  
  // i64hl16 predicate - true if the 64-bit immediate has only bits 32-47 set.
  return ((N->getZExtValue() & 0x0000FFFF00000000ULL) == N->getZExtValue());
}], HL16>;

def i64hh16 : PatLeaf<(i64 imm), [{  
  // i64hh16 predicate - true if the 64-bit immediate has only bits 48-63 set.
  return ((N->getZExtValue() & 0xFFFF000000000000ULL) == N->getZExtValue());
}], HH16>;

def immSExt16 : PatLeaf<(i64 imm), [{
  // immSExt16 predicate - true if the immediate fits in a 16-bit sign extended
  // field.
  uint64_t val = N->getZExtValue();
  return ((int64_t)val == (int16_t)val);
}]>;

def immSExt32 : PatLeaf<(i64 imm), [{
  // immSExt32 predicate - true if the immediate fits in a 32-bit sign extended
  // field.
  uint64_t val = N->getZExtValue();
  return ((int64_t)val == (int32_t)val);
}]>;

def i64lo32 : PatLeaf<(i64 imm), [{
  // i64lo32 predicate - true if the 64-bit immediate has only rightmost 32
  // bits set.
  return ((N->getZExtValue() & 0x00000000FFFFFFFFULL) == N->getZExtValue());
}], LO32>;

def i64hi32 : PatLeaf<(i64 imm), [{
  // i64hi32 predicate - true if the 64-bit immediate has only bits 32-63 set.
  return ((N->getZExtValue() & 0xFFFFFFFF00000000ULL) == N->getZExtValue());
}], HI32>;

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
def MOV64ri16 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                       "lghi\t{$dst, $src}",
                       [(set GR64:$dst, immSExt16:$src)]>;

def MOV64rill16 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llill\t{$dst, $src}",
                         [(set GR64:$dst, i64ll16:$src)]>;
def MOV64rilh16 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llilh\t{$dst, $src}",
                         [(set GR64:$dst, i64lh16:$src)]>;
def MOV64rihl16 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llihl\t{$dst, $src}",
                         [(set GR64:$dst, i64hl16:$src)]>;
def MOV64rihh16 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llihh\t{$dst, $src}",
                         [(set GR64:$dst, i64hh16:$src)]>;
// FIXME: these 3 instructions seem to require extimm facility
def MOV64ri32 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                       "lgfi\t{$dst, $src}",
                       [(set GR64:$dst, immSExt32:$src)]>;
def MOV64rilo32 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llilf\t{$dst, $src}",
                         [(set GR64:$dst, i64lo32:$src)]>;
def MOV64rihi32 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llihf\t{$dst, $src}",
                         [(set GR64:$dst, i64hi32:$src)]>;
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
def ADD64ri16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                       "aghi\t{$dst, $src2}",
                       [(set GR64:$dst, (add GR64:$src1, immSExt16:$src2)),
                        (implicit PSW)]>;

let isCommutable = 1 in { // X = AND Y, Z  == X = AND Z, Y
// FIXME: Provide proper encoding!
def AND64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "ngr\t{$dst, $src2}",
                     [(set GR64:$dst, (and GR64:$src1, GR64:$src2))]>;
}

// FIXME: Provide proper encoding!
def AND64rill16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nill\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64ll16:$src2))]>;
def AND64rilh16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nilh\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64lh16:$src2))]>;
def AND64rihl16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nihl\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64hl16:$src2))]>;
def AND64rihh16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nihh\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64hh16:$src2))]>;

let isCommutable = 1 in { // X = OR Y, Z  == X = OR Z, Y
// FIXME: Provide proper encoding!
def OR64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "ogr\t{$dst, $src2}",
                    [(set GR64:$dst, (or GR64:$src1, GR64:$src2))]>;
}
def OR64rill16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oill\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64ll16:$src2))]>;
def OR64rilh16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oilh\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64lh16:$src2))]>;
def OR64rihl16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oihl\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64hl16:$src2))]>;
def OR64rihh16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oihh\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64hh16:$src2))]>;

// FIXME: Provide proper encoding!
def SUB64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "sgr\t{$dst, $src2}",
                     [(set GR64:$dst, (sub GR64:$src1, GR64:$src2))]>;


let isCommutable = 1 in { // X = XOR Y, Z  == X = XOR Z, Y
// FIXME: Provide proper encoding!
def XOR64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "xgr\t{$dst, $src2}",
                     [(set GR64:$dst, (xor GR64:$src1, GR64:$src2))]>;
}

} // Defs = [PSW]
} // isTwoAddress = 1
