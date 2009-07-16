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

def i64ll16 : PatLeaf<(imm), [{  
  // i64ll16 predicate - true if the 64-bit immediate has only rightmost 16
  // bits set.
  return ((N->getZExtValue() & 0x000000000000FFFFULL) == N->getZExtValue());
}], LL16>;

def i64lh16 : PatLeaf<(imm), [{  
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

def immSExt16 : PatLeaf<(imm), [{
  // immSExt16 predicate - true if the immediate fits in a 16-bit sign extended
  // field.
  if (N->getValueType(0) == MVT::i64) {
    uint64_t val = N->getZExtValue();
    return ((int64_t)val == (int16_t)val);
  } else if (N->getValueType(0) == MVT::i32) {
    uint32_t val = N->getZExtValue();
    return ((int32_t)val == (int16_t)val);
  }

  return false;
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

// extloads
def extloadi64i8   : PatFrag<(ops node:$ptr), (i64 (extloadi8  node:$ptr))>;
def extloadi64i16  : PatFrag<(ops node:$ptr), (i64 (extloadi16 node:$ptr))>;
def extloadi64i32  : PatFrag<(ops node:$ptr), (i64 (extloadi32 node:$ptr))>;

def sextloadi64i8   : PatFrag<(ops node:$ptr), (i64 (sextloadi8  node:$ptr))>;
def sextloadi64i16  : PatFrag<(ops node:$ptr), (i64 (sextloadi16 node:$ptr))>;
def sextloadi64i32  : PatFrag<(ops node:$ptr), (i64 (sextloadi32 node:$ptr))>;

def zextloadi64i8   : PatFrag<(ops node:$ptr), (i64 (zextloadi8  node:$ptr))>;
def zextloadi64i16  : PatFrag<(ops node:$ptr), (i64 (zextloadi16 node:$ptr))>;
def zextloadi64i32  : PatFrag<(ops node:$ptr), (i64 (zextloadi32 node:$ptr))>;


//===----------------------------------------------------------------------===//
// SystemZ Operand Definitions.
//===----------------------------------------------------------------------===//

// Address operands

// riaddr := reg + imm
def riaddr32 : Operand<i32>,
               ComplexPattern<i32, 2, "SelectAddrRI", []> {
  let PrintMethod = "printRIAddrOperand";
  let MIOperandInfo = (ops ADDR32:$base, i32imm:$disp);
}

def riaddr : Operand<i64>,
             ComplexPattern<i64, 2, "SelectAddrRI", []> {
  let PrintMethod = "printRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, i32imm:$disp);
}

//===----------------------------------------------------------------------===//

// rriaddr := reg + reg + imm
def rriaddr : Operand<i64>,
              ComplexPattern<i64, 3, "SelectAddrRRI", []> {
  let PrintMethod = "printRRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, ADDR64:$index, i32imm:$disp);
}

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
def MOV32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src),
                     "lr\t{$dst, $src}",
                     []>;
def MOV64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src),
                     "lgr\t{$dst, $src}",
                     []>;
}

def MOVSX64rr32 : Pseudo<(outs GR64:$dst), (ins GR32:$src),
                         "lgfr\t{$dst, $src}",
                         [(set GR64:$dst, (sext GR32:$src))]>;
def MOVZX64rr32 : Pseudo<(outs GR64:$dst), (ins GR32:$src),
                         "llgfr\t{$dst, $src}",
                         [(set GR64:$dst, (zext GR32:$src))]>;

// FIXME: Provide proper encoding!
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def MOV32ri16 : Pseudo<(outs GR32:$dst), (ins i32imm:$src),
                       "lhi\t{$dst, $src}",
                       [(set GR32:$dst, immSExt16:$src)]>;
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

let canFoldAsLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in {
def MOV64rm : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                     "lg\t{$dst, $src}",
                     [(set GR64:$dst, (load rriaddr:$src))]>;

}

def MOV64mr : Pseudo<(outs), (ins rriaddr:$dst, GR64:$src),
                     "stg\t{$src, $dst}",
                     [(store GR64:$src, rriaddr:$dst)]>;

// extloads
def MOVSX64rm8  : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "lgb\t{$dst, $src}",
                         [(set GR64:$dst, (sextloadi64i8 rriaddr:$src))]>;
def MOVSX64rm16 : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "lgh\t{$dst, $src}",
                         [(set GR64:$dst, (sextloadi64i16 rriaddr:$src))]>;
def MOVSX64rm32 : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "lgf\t{$dst, $src}",
                         [(set GR64:$dst, (sextloadi64i32 rriaddr:$src))]>;

def MOVZX64rm8  : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "llgc\t{$dst, $src}",
                         [(set GR64:$dst, (zextloadi64i8 rriaddr:$src))]>;
def MOVZX64rm16 : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "llgh\t{$dst, $src}",
                         [(set GR64:$dst, (zextloadi64i16 rriaddr:$src))]>;
def MOVZX64rm32 : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "llgf\t{$dst, $src}",
                         [(set GR64:$dst, (zextloadi64i32 rriaddr:$src))]>;

// truncstores
// FIXME: Implement 12-bit displacement stuff someday
def MOV32m8r  : Pseudo<(outs), (ins rriaddr:$dst, GR32:$src),
                       "stcy\t{$src, $dst}",
                       [(truncstorei8 GR32:$src, rriaddr:$dst)]>;

def MOV32m16r : Pseudo<(outs), (ins rriaddr:$dst, GR32:$src),
                       "sthy\t{$src, $dst}",
                       [(truncstorei16 GR32:$src, rriaddr:$dst)]>;

def MOV64m8r  : Pseudo<(outs), (ins rriaddr:$dst, GR64:$src),
                       "stcy\t{$src, $dst}",
                       [(truncstorei8 GR64:$src, rriaddr:$dst)]>;

def MOV64m16r : Pseudo<(outs), (ins rriaddr:$dst, GR64:$src),
                       "sthy\t{$src, $dst}",
                       [(truncstorei16 GR64:$src, rriaddr:$dst)]>;

def MOV64m32r : Pseudo<(outs), (ins rriaddr:$dst, GR64:$src),
                       "sty\t{$src, $dst}",
                       [(truncstorei32 GR64:$src, rriaddr:$dst)]>;

//===----------------------------------------------------------------------===//
// Arithmetic Instructions

let isTwoAddress = 1 in {

let Defs = [PSW] in {

let isCommutable = 1 in { // X = ADD Y, Z  == X = ADD Z, Y
// FIXME: Provide proper encoding!
def ADD32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                     "ar\t{$dst, $src2}",
                     [(set GR32:$dst, (add GR32:$src1, GR32:$src2)),
                      (implicit PSW)]>;
def ADD64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "agr\t{$dst, $src2}",
                     [(set GR64:$dst, (add GR64:$src1, GR64:$src2)),
                      (implicit PSW)]>;
}

// FIXME: Provide proper encoding!
def ADD32ri16 : Pseudo<(outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                       "ahi\t{$dst, $src2}",
                       [(set GR32:$dst, (add GR32:$src1, immSExt16:$src2)),
                        (implicit PSW)]>;
def ADD32ri   : Pseudo<(outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                       "afi\t{$dst, $src2}",
                       [(set GR32:$dst, (add GR32:$src1, imm:$src2)),
                        (implicit PSW)]>;
def ADD64ri16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                       "aghi\t{$dst, $src2}",
                       [(set GR64:$dst, (add GR64:$src1, immSExt16:$src2)),
                        (implicit PSW)]>;
def ADD64ri32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                       "agfi\t{$dst, $src2}",
                       [(set GR64:$dst, (add GR64:$src1, immSExt32:$src2)),
                        (implicit PSW)]>;

let isCommutable = 1 in { // X = AND Y, Z  == X = AND Z, Y
// FIXME: Provide proper encoding!
def AND32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                     "nr\t{$dst, $src2}",
                     [(set GR32:$dst, (and GR32:$src1, GR32:$src2))]>;
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
// FIXME: these 2 instructions seem to require extimm facility
def AND64rilo32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nilf\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64lo32:$src2))]>;
def AND64rihi32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nihf\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64hi32:$src2))]>;

let isCommutable = 1 in { // X = OR Y, Z  == X = OR Z, Y
// FIXME: Provide proper encoding!
def OR32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                    "or\t{$dst, $src2}",
                    [(set GR32:$dst, (or GR32:$src1, GR32:$src2))]>;
def OR64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "ogr\t{$dst, $src2}",
                    [(set GR64:$dst, (or GR64:$src1, GR64:$src2))]>;
}

def OR32ri16  : Pseudo<(outs GR32:$dst), (ins GR32:$src1, i16imm:$src2),
                      "oill\t{$dst, $src2}",
                      [(set GR32:$dst, (or GR32:$src1, i64ll16:$src2))]>;
def OR32ri16h : Pseudo<(outs GR32:$dst), (ins GR32:$src1, i16imm:$src2),
                      "oilh\t{$dst, $src2}",
                      [(set GR32:$dst, (or GR32:$src1, i64lh16:$src2))]>;
def OR32ri : Pseudo<(outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                    "oilf\t{$dst, $src2}",
                    [(set GR32:$dst, (or GR32:$src1, imm:$src2))]>;

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
// FIXME: these 2 instructions seem to require extimm facility
def OR64rilo32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oilf\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64lo32:$src2))]>;
def OR64rihi32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oihf\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64hi32:$src2))]>;

// FIXME: Provide proper encoding!
def SUB32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                     "sr\t{$dst, $src2}",
                     [(set GR32:$dst, (sub GR32:$src1, GR32:$src2))]>;
def SUB64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "sgr\t{$dst, $src2}",
                     [(set GR64:$dst, (sub GR64:$src1, GR64:$src2))]>;


let isCommutable = 1 in { // X = XOR Y, Z  == X = XOR Z, Y
// FIXME: Provide proper encoding!
def XOR32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                     "xr\t{$dst, $src2}",
                     [(set GR32:$dst, (xor GR32:$src1, GR32:$src2))]>;
def XOR64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "xgr\t{$dst, $src2}",
                     [(set GR64:$dst, (xor GR64:$src1, GR64:$src2))]>;
}

def XOR32ri : Pseudo<(outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                     "xilf\t{$dst, $src2}",
                     [(set GR32:$dst, (xor GR32:$src1, imm:$src2))]>;

// FIXME: these 2 instructions seem to require extimm facility
def XOR64rilo32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "xilf\t{$dst, $src2}",
                         [(set GR64:$dst, (xor GR64:$src1, i64lo32:$src2))]>;
def XOR64rihi32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "xihf\t{$dst, $src2}",
                         [(set GR64:$dst, (xor GR64:$src1, i64hi32:$src2))]>;

} // Defs = [PSW]
} // isTwoAddress = 1

//===----------------------------------------------------------------------===//
// Shifts

let isTwoAddress = 1 in
def SRL32rri : Pseudo<(outs GR32:$dst), (ins GR32:$src, riaddr32:$amt),
                      "srl\t{$src, $amt}",
                      [(set GR32:$dst, (srl GR32:$src, riaddr32:$amt))]>;
def SRL64rri : Pseudo<(outs GR64:$dst), (ins GR64:$src, riaddr:$amt),
                      "srlg\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (srl GR64:$src, (i32 (trunc riaddr:$amt))))]>;
def SRLA64ri  : Pseudo<(outs GR64:$dst), (ins GR64:$src, i32imm:$amt),
                      "srlg\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (srl GR64:$src, (i32 imm:$amt)))]>;

let isTwoAddress = 1 in
def SHL32rri : Pseudo<(outs GR32:$dst), (ins GR32:$src, riaddr32:$amt),
                      "sll\t{$src, $amt}",
                      [(set GR32:$dst, (shl GR32:$src, riaddr32:$amt))]>;
def SHL64rri : Pseudo<(outs GR64:$dst), (ins GR64:$src, riaddr:$amt),
                      "sllg\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (shl GR64:$src, (i32 (trunc riaddr:$amt))))]>;
def SHL64ri  : Pseudo<(outs GR64:$dst), (ins GR64:$src, i32imm:$amt),
                      "sllg\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (shl GR64:$src, (i32 imm:$amt)))]>;


let Defs = [PSW] in {
let isTwoAddress = 1 in
def SRA32rri : Pseudo<(outs GR32:$dst), (ins GR32:$src, riaddr32:$amt),
                      "sra\t{$src, $amt}",
                      [(set GR32:$dst, (sra GR32:$src, riaddr32:$amt)),
                       (implicit PSW)]>;
def SRA64rri : Pseudo<(outs GR64:$dst), (ins GR64:$src, riaddr:$amt),
                      "srag\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (sra GR64:$src, (i32 (trunc riaddr:$amt)))),
                       (implicit PSW)]>;
def SRA64ri  : Pseudo<(outs GR64:$dst), (ins GR64:$src, i32imm:$amt),
                      "srag\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (sra GR64:$src, (i32 imm:$amt))),
                       (implicit PSW)]>;
} // Defs = [PSW]

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns.
//===----------------------------------------------------------------------===//

// anyext
def : Pat<(i64 (anyext GR32:$src)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR32:$src, subreg_32bit)>;

//===----------------------------------------------------------------------===//
// Peepholes.
//===----------------------------------------------------------------------===//

// FIXME: use add/sub tricks with 32678/-32768

// trunc patterns
def : Pat<(i32 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, subreg_32bit)>;

// sext_inreg patterns
def : Pat<(sext_inreg GR64:$src, i32),
          (MOVSX64rr32 (EXTRACT_SUBREG GR64:$src, subreg_32bit))>;

// extload patterns
def : Pat<(extloadi64i8  rriaddr:$src), (MOVZX64rm8  rriaddr:$src)>;
def : Pat<(extloadi64i16 rriaddr:$src), (MOVZX64rm16 rriaddr:$src)>;
def : Pat<(extloadi64i32 rriaddr:$src), (MOVZX64rm32 rriaddr:$src)>;
