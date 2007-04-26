//====- X86InstrX86-64.td - Describe the X86 Instruction Set ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86-64 instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand Definitions...
//

// 64-bits but only 32 bits are significant.
def i64i32imm  : Operand<i64>;
// 64-bits but only 8 bits are significant.
def i64i8imm   : Operand<i64>;

def lea64mem : Operand<i64> {
  let PrintMethod = "printi64mem";
  let MIOperandInfo = (ops GR64, i8imm, GR64, i32imm);
}

def lea64_32mem : Operand<i32> {
  let PrintMethod = "printlea64_32mem";
  let MIOperandInfo = (ops GR32, i8imm, GR32, i32imm);
}

//===----------------------------------------------------------------------===//
// Complex Pattern Definitions...
//
def lea64addr : ComplexPattern<i64, 4, "SelectLEAAddr",
                               [add, mul, shl, or, frameindex, X86Wrapper],
                               []>;

//===----------------------------------------------------------------------===//
// Instruction templates...
//

class RI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, REX_W;
class RIi8 <bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : Ii8<o, F, ops, asm, pattern>, REX_W;
class RIi32 <bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : Ii32<o, F, ops, asm, pattern>, REX_W;

class RIi64<bits<8> o, Format f, dag ops, string asm, list<dag> pattern>
  : X86Inst<o, f, Imm64, ops, asm>, REX_W {
  let Pattern = pattern;
  let CodeSize = 3;
}

class RSSI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : SSI<o, F, ops, asm, pattern>, REX_W;
class RSDI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : SDI<o, F, ops, asm, pattern>, REX_W;
class RPDI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : PDI<o, F, ops, asm, pattern>, REX_W;

//===----------------------------------------------------------------------===//
// Pattern fragments...
//

def i64immSExt32  : PatLeaf<(i64 imm), [{
  // i64immSExt32 predicate - True if the 64-bit immediate fits in a 32-bit
  // sign extended field.
  return (int64_t)N->getValue() == (int32_t)N->getValue();
}]>;

def i64immZExt32  : PatLeaf<(i64 imm), [{
  // i64immZExt32 predicate - True if the 64-bit immediate fits in a 32-bit
  // unsignedsign extended field.
  return (uint64_t)N->getValue() == (uint32_t)N->getValue();
}]>;

def i64immSExt8  : PatLeaf<(i64 imm), [{
  // i64immSExt8 predicate - True if the 64-bit immediate fits in a 8-bit
  // sign extended field.
  return (int64_t)N->getValue() == (int8_t)N->getValue();
}]>;

def sextloadi64i1  : PatFrag<(ops node:$ptr), (i64 (sextloadi1 node:$ptr))>;
def sextloadi64i8  : PatFrag<(ops node:$ptr), (i64 (sextloadi8 node:$ptr))>;
def sextloadi64i16 : PatFrag<(ops node:$ptr), (i64 (sextloadi16 node:$ptr))>;
def sextloadi64i32 : PatFrag<(ops node:$ptr), (i64 (sextloadi32 node:$ptr))>;

def zextloadi64i1  : PatFrag<(ops node:$ptr), (i64 (zextloadi1 node:$ptr))>;
def zextloadi64i8  : PatFrag<(ops node:$ptr), (i64 (zextloadi8 node:$ptr))>;
def zextloadi64i16 : PatFrag<(ops node:$ptr), (i64 (zextloadi16 node:$ptr))>;
def zextloadi64i32 : PatFrag<(ops node:$ptr), (i64 (zextloadi32 node:$ptr))>;

def extloadi64i1   : PatFrag<(ops node:$ptr), (i64 (extloadi1 node:$ptr))>;
def extloadi64i8   : PatFrag<(ops node:$ptr), (i64 (extloadi8 node:$ptr))>;
def extloadi64i16  : PatFrag<(ops node:$ptr), (i64 (extloadi16 node:$ptr))>;
def extloadi64i32  : PatFrag<(ops node:$ptr), (i64 (extloadi32 node:$ptr))>;

//===----------------------------------------------------------------------===//
// Instruction list...
//

def IMPLICIT_DEF_GR64  : I<0, Pseudo, (ops GR64:$dst),
                         "#IMPLICIT_DEF $dst",
                         [(set GR64:$dst, (undef))]>;

//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1, noResults = 1 in
  // All calls clobber the non-callee saved registers...
  let Defs = [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
              FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0,
              MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
              XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15] in {
    def CALL64pcrel32 : I<0xE8, RawFrm, (ops i64imm:$dst, variable_ops),
                          "call ${dst:call}", []>;
    def CALL64r       : I<0xFF, MRM2r, (ops GR64:$dst, variable_ops),
                          "call {*}$dst", [(X86call GR64:$dst)]>;
    def CALL64m       : I<0xFF, MRM2m, (ops i64mem:$dst, variable_ops),
                          "call {*}$dst", []>;
  }

// Branches
let isBranch = 1, isTerminator = 1, noResults = 1, isBarrier = 1 in {
  def JMP64r     : I<0xFF, MRM4r, (ops GR64:$dst), "jmp{q} {*}$dst",
                     [(brind GR64:$dst)]>;
  def JMP64m     : I<0xFF, MRM4m, (ops i64mem:$dst), "jmp{q} {*}$dst",
                     [(brind (loadi64 addr:$dst))]>;
}

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//
def LEAVE64  : I<0xC9, RawFrm,
                 (ops), "leave", []>, Imp<[RBP,RSP],[RBP,RSP]>;
def POP64r   : I<0x58, AddRegFrm,
                 (ops GR64:$reg), "pop{q} $reg", []>, Imp<[RSP],[RSP]>;

def LEA64_32r : I<0x8D, MRMSrcMem,
                  (ops GR32:$dst, lea64_32mem:$src),
                  "lea{l} {$src|$dst}, {$dst|$src}",
                  [(set GR32:$dst, lea32addr:$src)]>, Requires<[In64BitMode]>;

def LEA64r   : RI<0x8D, MRMSrcMem, (ops GR64:$dst, lea64mem:$src),
                  "lea{q} {$src|$dst}, {$dst|$src}",
                  [(set GR64:$dst, lea64addr:$src)]>;

let isTwoAddress = 1 in
def BSWAP64r : RI<0xC8, AddRegFrm, (ops GR64:$dst, GR64:$src),
                  "bswap{q} $dst", 
                  [(set GR64:$dst, (bswap GR64:$src))]>, TB;
// Exchange
def XCHG64rr : RI<0x87, MRMDestReg, (ops GR64:$src1, GR64:$src2),
                  "xchg{q} {$src2|$src1}, {$src1|$src2}", []>;
def XCHG64mr : RI<0x87, MRMDestMem, (ops i64mem:$src1, GR64:$src2),
                  "xchg{q} {$src2|$src1}, {$src1|$src2}", []>;
def XCHG64rm : RI<0x87, MRMSrcMem, (ops GR64:$src1, i64mem:$src2),
                  "xchg{q} {$src2|$src1}, {$src1|$src2}", []>;

// Repeat string ops
def REP_MOVSQ : RI<0xA5, RawFrm, (ops), "{rep;movsq|rep movsq}",
                   [(X86rep_movs i64)]>,
                Imp<[RCX,RDI,RSI], [RCX,RDI,RSI]>, REP;
def REP_STOSQ : RI<0xAB, RawFrm, (ops), "{rep;stosq|rep stosq}",
                   [(X86rep_stos i64)]>,
                Imp<[RAX,RCX,RDI], [RCX,RDI]>, REP;

//===----------------------------------------------------------------------===//
//  Move Instructions...
//

def MOV64rr : RI<0x89, MRMDestReg, (ops GR64:$dst, GR64:$src),
                 "mov{q} {$src, $dst|$dst, $src}", []>;

def MOV64ri : RIi64<0xB8, AddRegFrm, (ops GR64:$dst, i64imm:$src),
                    "movabs{q} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, imm:$src)]>;
def MOV64ri32 : RIi32<0xC7, MRM0r, (ops GR64:$dst, i64i32imm:$src),
                      "mov{q} {$src, $dst|$dst, $src}",
                      [(set GR64:$dst, i64immSExt32:$src)]>;

def MOV64rm : RI<0x8B, MRMSrcMem, (ops GR64:$dst, i64mem:$src),
                 "mov{q} {$src, $dst|$dst, $src}",
                 [(set GR64:$dst, (load addr:$src))]>;

def MOV64mr : RI<0x89, MRMDestMem, (ops i64mem:$dst, GR64:$src),
                 "mov{q} {$src, $dst|$dst, $src}",
                 [(store GR64:$src, addr:$dst)]>;
def MOV64mi32 : RIi32<0xC7, MRM0m, (ops i64mem:$dst, i64i32imm:$src),
                      "mov{q} {$src, $dst|$dst, $src}",
                      [(store i64immSExt32:$src, addr:$dst)]>;

// Sign/Zero extenders

def MOVSX64rr8 : RI<0xBE, MRMSrcReg, (ops GR64:$dst, GR8 :$src),
                    "movs{bq|x} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR8:$src))]>, TB;
def MOVSX64rm8 : RI<0xBE, MRMSrcMem, (ops GR64:$dst, i8mem :$src),
                    "movs{bq|x} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i8 addr:$src))]>, TB;
def MOVSX64rr16: RI<0xBF, MRMSrcReg, (ops GR64:$dst, GR16:$src),
                    "movs{wq|x} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR16:$src))]>, TB;
def MOVSX64rm16: RI<0xBF, MRMSrcMem, (ops GR64:$dst, i16mem:$src),
                    "movs{wq|x} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i16 addr:$src))]>, TB;
def MOVSX64rr32: RI<0x63, MRMSrcReg, (ops GR64:$dst, GR32:$src),
                    "movs{lq|xd} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR32:$src))]>;
def MOVSX64rm32: RI<0x63, MRMSrcMem, (ops GR64:$dst, i32mem:$src),
                    "movs{lq|xd} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i32 addr:$src))]>;

def MOVZX64rr8 : RI<0xB6, MRMSrcReg, (ops GR64:$dst, GR8 :$src),
                    "movz{bq|x} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (zext GR8:$src))]>, TB;
def MOVZX64rm8 : RI<0xB6, MRMSrcMem, (ops GR64:$dst, i8mem :$src),
                    "movz{bq|x} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (zextloadi64i8 addr:$src))]>, TB;
def MOVZX64rr16: RI<0xB7, MRMSrcReg, (ops GR64:$dst, GR16:$src),
                    "movz{wq|x} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (zext GR16:$src))]>, TB;
def MOVZX64rm16: RI<0xB7, MRMSrcMem, (ops GR64:$dst, i16mem:$src),
                    "movz{wq|x} {$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (zextloadi64i16 addr:$src))]>, TB;

def CDQE : RI<0x98, RawFrm, (ops),
             "{cltq|cdqe}", []>, Imp<[EAX],[RAX]>;     // RAX = signext(EAX)

def CQO  : RI<0x99, RawFrm, (ops),
              "{cqto|cqo}", []>, Imp<[RAX],[RAX,RDX]>; // RDX:RAX = signext(RAX)

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions...
//

let isTwoAddress = 1 in {
let isConvertibleToThreeAddress = 1 in {
let isCommutable = 1 in
def ADD64rr  : RI<0x01, MRMDestReg, (ops GR64:$dst, GR64:$src1, GR64:$src2),
                  "add{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (add GR64:$src1, GR64:$src2))]>;

def ADD64ri32 : RIi32<0x81, MRM0r, (ops GR64:$dst, GR64:$src1, i64i32imm:$src2),
                      "add{q} {$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (add GR64:$src1, i64immSExt32:$src2))]>;
def ADD64ri8 : RIi8<0x83, MRM0r, (ops GR64:$dst, GR64:$src1, i64i8imm:$src2),
                    "add{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (add GR64:$src1, i64immSExt8:$src2))]>;
} // isConvertibleToThreeAddress

def ADD64rm  : RI<0x03, MRMSrcMem, (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                  "add{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (add GR64:$src1, (load addr:$src2)))]>;
} // isTwoAddress

def ADD64mr  : RI<0x01, MRMDestMem, (ops i64mem:$dst, GR64:$src2),
                  "add{q} {$src2, $dst|$dst, $src2}",
                  [(store (add (load addr:$dst), GR64:$src2), addr:$dst)]>;
def ADD64mi32 : RIi32<0x81, MRM0m, (ops i64mem:$dst, i64i32imm :$src2),
                      "add{q} {$src2, $dst|$dst, $src2}",
               [(store (add (load addr:$dst), i64immSExt32:$src2), addr:$dst)]>;
def ADD64mi8 : RIi8<0x83, MRM0m, (ops i64mem:$dst, i64i8imm :$src2),
                    "add{q} {$src2, $dst|$dst, $src2}",
                [(store (add (load addr:$dst), i64immSExt8:$src2), addr:$dst)]>;

let isTwoAddress = 1 in {
let isCommutable = 1 in
def ADC64rr  : RI<0x11, MRMDestReg, (ops GR64:$dst, GR64:$src1, GR64:$src2),
                  "adc{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (adde GR64:$src1, GR64:$src2))]>;

def ADC64rm  : RI<0x13, MRMSrcMem , (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                  "adc{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (adde GR64:$src1, (load addr:$src2)))]>;

def ADC64ri32 : RIi32<0x81, MRM2r, (ops GR64:$dst, GR64:$src1, i64i32imm:$src2),
                      "adc{q} {$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (adde GR64:$src1, i64immSExt32:$src2))]>;
def ADC64ri8 : RIi8<0x83, MRM2r, (ops GR64:$dst, GR64:$src1, i64i8imm:$src2),
                    "adc{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (adde GR64:$src1, i64immSExt8:$src2))]>;
} // isTwoAddress

def ADC64mr  : RI<0x11, MRMDestMem, (ops i64mem:$dst, GR64:$src2),
                  "adc{q} {$src2, $dst|$dst, $src2}",
                  [(store (adde (load addr:$dst), GR64:$src2), addr:$dst)]>;
def ADC64mi32 : RIi32<0x81, MRM2m, (ops i64mem:$dst, i64i32imm:$src2),
                      "adc{q} {$src2, $dst|$dst, $src2}",
               [(store (adde (load addr:$dst), i64immSExt8:$src2), addr:$dst)]>;
def ADC64mi8 : RIi8<0x83, MRM2m, (ops i64mem:$dst, i64i8imm :$src2),
                    "adc{q} {$src2, $dst|$dst, $src2}",
               [(store (adde (load addr:$dst), i64immSExt8:$src2), addr:$dst)]>;

let isTwoAddress = 1 in {
def SUB64rr  : RI<0x29, MRMDestReg, (ops GR64:$dst, GR64:$src1, GR64:$src2),
                  "sub{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (sub GR64:$src1, GR64:$src2))]>;

def SUB64rm  : RI<0x2B, MRMSrcMem, (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                  "sub{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (sub GR64:$src1, (load addr:$src2)))]>;

def SUB64ri32 : RIi32<0x81, MRM5r, (ops GR64:$dst, GR64:$src1, i64i32imm:$src2),
                      "sub{q} {$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (sub GR64:$src1, i64immSExt32:$src2))]>;
def SUB64ri8 : RIi8<0x83, MRM5r, (ops GR64:$dst, GR64:$src1, i64i8imm:$src2),
                    "sub{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sub GR64:$src1, i64immSExt8:$src2))]>;
} // isTwoAddress

def SUB64mr  : RI<0x29, MRMDestMem, (ops i64mem:$dst, GR64:$src2), 
                  "sub{q} {$src2, $dst|$dst, $src2}",
                  [(store (sub (load addr:$dst), GR64:$src2), addr:$dst)]>;
def SUB64mi32 : RIi32<0x81, MRM5m, (ops i64mem:$dst, i64i32imm:$src2), 
                      "sub{q} {$src2, $dst|$dst, $src2}",
               [(store (sub (load addr:$dst), i64immSExt32:$src2), addr:$dst)]>;
def SUB64mi8 : RIi8<0x83, MRM5m, (ops i64mem:$dst, i64i8imm :$src2), 
                    "sub{q} {$src2, $dst|$dst, $src2}",
                [(store (sub (load addr:$dst), i64immSExt8:$src2), addr:$dst)]>;

let isTwoAddress = 1 in {
def SBB64rr    : RI<0x19, MRMDestReg, (ops GR64:$dst, GR64:$src1, GR64:$src2),
                    "sbb{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sube GR64:$src1, GR64:$src2))]>;

def SBB64rm  : RI<0x1B, MRMSrcMem, (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                  "sbb{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (sube GR64:$src1, (load addr:$src2)))]>;

def SBB64ri32 : RIi32<0x81, MRM3r, (ops GR64:$dst, GR64:$src1, i64i32imm:$src2),
                      "sbb{q} {$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (sube GR64:$src1, i64immSExt32:$src2))]>;
def SBB64ri8 : RIi8<0x83, MRM3r, (ops GR64:$dst, GR64:$src1, i64i8imm:$src2),
                    "sbb{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sube GR64:$src1, i64immSExt8:$src2))]>;
} // isTwoAddress

def SBB64mr  : RI<0x19, MRMDestMem, (ops i64mem:$dst, GR64:$src2), 
                  "sbb{q} {$src2, $dst|$dst, $src2}",
                  [(store (sube (load addr:$dst), GR64:$src2), addr:$dst)]>;
def SBB64mi32 : RIi32<0x81, MRM3m, (ops i64mem:$dst, i64i32imm:$src2), 
                      "sbb{q} {$src2, $dst|$dst, $src2}",
              [(store (sube (load addr:$dst), i64immSExt32:$src2), addr:$dst)]>;
def SBB64mi8 : RIi8<0x83, MRM3m, (ops i64mem:$dst, i64i8imm :$src2), 
                    "sbb{q} {$src2, $dst|$dst, $src2}",
               [(store (sube (load addr:$dst), i64immSExt8:$src2), addr:$dst)]>;

// Unsigned multiplication
def MUL64r : RI<0xF7, MRM4r, (ops GR64:$src),
                "mul{q} $src", []>,
             Imp<[RAX],[RAX,RDX]>;         // RAX,RDX = RAX*GR64
def MUL64m : RI<0xF7, MRM4m, (ops i64mem:$src),
                "mul{q} $src", []>,
             Imp<[RAX],[RAX,RDX]>;         // RAX,RDX = RAX*[mem64]

// Signed multiplication
def IMUL64r : RI<0xF7, MRM5r, (ops GR64:$src),
                 "imul{q} $src", []>,
              Imp<[RAX],[RAX,RDX]>;         // RAX,RDX = RAX*GR64
def IMUL64m : RI<0xF7, MRM5m, (ops i64mem:$src),
                 "imul{q} $src", []>,
              Imp<[RAX],[RAX,RDX]>;         // RAX,RDX = RAX*[mem64]

let isTwoAddress = 1 in {
let isCommutable = 1 in
def IMUL64rr : RI<0xAF, MRMSrcReg, (ops GR64:$dst, GR64:$src1, GR64:$src2),
                  "imul{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (mul GR64:$src1, GR64:$src2))]>, TB;

def IMUL64rm : RI<0xAF, MRMSrcMem, (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                  "imul{q} {$src2, $dst|$dst, $src2}",
                 [(set GR64:$dst, (mul GR64:$src1, (load addr:$src2)))]>, TB;
} // isTwoAddress

// Suprisingly enough, these are not two address instructions!
def IMUL64rri32 : RIi32<0x69, MRMSrcReg,                    // GR64 = GR64*I32
                        (ops GR64:$dst, GR64:$src1, i64i32imm:$src2),
                        "imul{q} {$src2, $src1, $dst|$dst, $src1, $src2}",
                       [(set GR64:$dst, (mul GR64:$src1, i64immSExt32:$src2))]>;
def IMUL64rri8 : RIi8<0x6B, MRMSrcReg,                      // GR64 = GR64*I8
                      (ops GR64:$dst, GR64:$src1, i64i8imm:$src2),
                      "imul{q} {$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR64:$dst, (mul GR64:$src1, i64immSExt8:$src2))]>;
def IMUL64rmi32 : RIi32<0x69, MRMSrcMem,                   // GR64 = [mem64]*I32
                        (ops GR64:$dst, i64mem:$src1, i64i32imm:$src2),
                        "imul{q} {$src2, $src1, $dst|$dst, $src1, $src2}",
                [(set GR64:$dst, (mul (load addr:$src1), i64immSExt32:$src2))]>;
def IMUL64rmi8 : RIi8<0x6B, MRMSrcMem,                      // GR64 = [mem64]*I8
                      (ops GR64:$dst, i64mem:$src1, i64i8imm: $src2),
                      "imul{q} {$src2, $src1, $dst|$dst, $src1, $src2}",
                 [(set GR64:$dst, (mul (load addr:$src1), i64immSExt8:$src2))]>;

// Unsigned division / remainder
def DIV64r : RI<0xF7, MRM6r, (ops GR64:$src),        // RDX:RAX/r64 = RAX,RDX
                "div{q} $src", []>, Imp<[RAX,RDX],[RAX,RDX]>;
def DIV64m : RI<0xF7, MRM6m, (ops i64mem:$src),      // RDX:RAX/[mem64] = RAX,RDX
                "div{q} $src", []>, Imp<[RAX,RDX],[RAX,RDX]>;

// Signed division / remainder
def IDIV64r: RI<0xF7, MRM7r, (ops GR64:$src),        // RDX:RAX/r64 = RAX,RDX
                "idiv{q} $src", []>, Imp<[RAX,RDX],[RAX,RDX]>;
def IDIV64m: RI<0xF7, MRM7m, (ops i64mem:$src),      // RDX:RAX/[mem64] = RAX,RDX
                "idiv{q} $src", []>, Imp<[RAX,RDX],[RAX,RDX]>;

// Unary instructions
let CodeSize = 2 in {
let isTwoAddress = 1 in
def NEG64r : RI<0xF7, MRM3r, (ops GR64:$dst, GR64:$src), "neg{q} $dst",
                [(set GR64:$dst, (ineg GR64:$src))]>;
def NEG64m : RI<0xF7, MRM3m, (ops i64mem:$dst), "neg{q} $dst",
                [(store (ineg (loadi64 addr:$dst)), addr:$dst)]>;

let isTwoAddress = 1, isConvertibleToThreeAddress = 1 in
def INC64r : RI<0xFF, MRM0r, (ops GR64:$dst, GR64:$src), "inc{q} $dst",
                [(set GR64:$dst, (add GR64:$src, 1))]>;
def INC64m : RI<0xFF, MRM0m, (ops i64mem:$dst), "inc{q} $dst",
                [(store (add (loadi64 addr:$dst), 1), addr:$dst)]>;

let isTwoAddress = 1, isConvertibleToThreeAddress = 1 in
def DEC64r : RI<0xFF, MRM1r, (ops GR64:$dst, GR64:$src), "dec{q} $dst",
                [(set GR64:$dst, (add GR64:$src, -1))]>;
def DEC64m : RI<0xFF, MRM1m, (ops i64mem:$dst), "dec{q} $dst",
                [(store (add (loadi64 addr:$dst), -1), addr:$dst)]>;

// In 64-bit mode, single byte INC and DEC cannot be encoded.
let isTwoAddress = 1, isConvertibleToThreeAddress = 1 in {
// Can transform into LEA.
def INC64_16r : I<0xFF, MRM0r, (ops GR16:$dst, GR16:$src), "inc{w} $dst",
                  [(set GR16:$dst, (add GR16:$src, 1))]>,
                OpSize, Requires<[In64BitMode]>;
def INC64_32r : I<0xFF, MRM0r, (ops GR32:$dst, GR32:$src), "inc{l} $dst",
                  [(set GR32:$dst, (add GR32:$src, 1))]>,
                Requires<[In64BitMode]>;
def DEC64_16r : I<0xFF, MRM1r, (ops GR16:$dst, GR16:$src), "dec{w} $dst",
                  [(set GR16:$dst, (add GR16:$src, -1))]>,
                OpSize, Requires<[In64BitMode]>;
def DEC64_32r : I<0xFF, MRM1r, (ops GR32:$dst, GR32:$src), "dec{l} $dst",
                  [(set GR32:$dst, (add GR32:$src, -1))]>,
                Requires<[In64BitMode]>;
} // isConvertibleToThreeAddress
} // CodeSize


// Shift instructions
let isTwoAddress = 1 in {
def SHL64rCL : RI<0xD3, MRM4r, (ops GR64:$dst, GR64:$src),
                  "shl{q} {%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (shl GR64:$src, CL))]>,
               Imp<[CL],[]>;
def SHL64ri  : RIi8<0xC1, MRM4r, (ops GR64:$dst, GR64:$src1, i8imm:$src2),
                    "shl{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (shl GR64:$src1, (i8 imm:$src2)))]>;
def SHL64r1  : RI<0xD1, MRM4r, (ops GR64:$dst, GR64:$src1),
                 "shl{q} $dst", []>;
} // isTwoAddress

def SHL64mCL : RI<0xD3, MRM4m, (ops i64mem:$dst),
                  "shl{q} {%cl, $dst|$dst, %CL}",
                  [(store (shl (loadi64 addr:$dst), CL), addr:$dst)]>,
               Imp<[CL],[]>;
def SHL64mi : RIi8<0xC1, MRM4m, (ops i64mem:$dst, i8imm:$src),
                  "shl{q} {$src, $dst|$dst, $src}",
                 [(store (shl (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def SHL64m1 : RI<0xD1, MRM4m, (ops i64mem:$dst),
                  "shl{q} $dst",
                 [(store (shl (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

let isTwoAddress = 1 in {
def SHR64rCL : RI<0xD3, MRM5r, (ops GR64:$dst, GR64:$src),
                  "shr{q} {%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (srl GR64:$src, CL))]>,
               Imp<[CL],[]>;
def SHR64ri : RIi8<0xC1, MRM5r, (ops GR64:$dst, GR64:$src1, i8imm:$src2),
                  "shr{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (srl GR64:$src1, (i8 imm:$src2)))]>;
def SHR64r1  : RI<0xD1, MRM5r, (ops GR64:$dst, GR64:$src1),
                 "shr{q} $dst",
                 [(set GR64:$dst, (srl GR64:$src1, (i8 1)))]>;
} // isTwoAddress

def SHR64mCL : RI<0xD3, MRM5m, (ops i64mem:$dst),
                  "shr{q} {%cl, $dst|$dst, %CL}",
                  [(store (srl (loadi64 addr:$dst), CL), addr:$dst)]>,
               Imp<[CL],[]>;
def SHR64mi : RIi8<0xC1, MRM5m, (ops i64mem:$dst, i8imm:$src),
                  "shr{q} {$src, $dst|$dst, $src}",
                 [(store (srl (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def SHR64m1 : RI<0xD1, MRM5m, (ops i64mem:$dst),
                  "shr{q} $dst",
                 [(store (srl (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

let isTwoAddress = 1 in {
def SAR64rCL : RI<0xD3, MRM7r, (ops GR64:$dst, GR64:$src),
                 "sar{q} {%cl, $dst|$dst, %CL}",
                 [(set GR64:$dst, (sra GR64:$src, CL))]>, Imp<[CL],[]>;
def SAR64ri  : RIi8<0xC1, MRM7r, (ops GR64:$dst, GR64:$src1, i8imm:$src2),
                   "sar{q} {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (sra GR64:$src1, (i8 imm:$src2)))]>;
def SAR64r1  : RI<0xD1, MRM7r, (ops GR64:$dst, GR64:$src1),
                 "sar{q} $dst",
                 [(set GR64:$dst, (sra GR64:$src1, (i8 1)))]>;
} // isTwoAddress

def SAR64mCL : RI<0xD3, MRM7m, (ops i64mem:$dst), 
                 "sar{q} {%cl, $dst|$dst, %CL}",
                 [(store (sra (loadi64 addr:$dst), CL), addr:$dst)]>,
               Imp<[CL],[]>;
def SAR64mi  : RIi8<0xC1, MRM7m, (ops i64mem:$dst, i8imm:$src),
                    "sar{q} {$src, $dst|$dst, $src}",
                 [(store (sra (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def SAR64m1 : RI<0xD1, MRM7m, (ops i64mem:$dst),
                  "sar{q} $dst",
                 [(store (sra (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

// Rotate instructions
let isTwoAddress = 1 in {
def ROL64rCL : RI<0xD3, MRM0r, (ops GR64:$dst, GR64:$src),
                  "rol{q} {%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (rotl GR64:$src, CL))]>, Imp<[CL],[]>;
def ROL64ri  : RIi8<0xC1, MRM0r, (ops GR64:$dst, GR64:$src1, i8imm:$src2),
                    "rol{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (rotl GR64:$src1, (i8 imm:$src2)))]>;
def ROL64r1  : RI<0xD1, MRM0r, (ops GR64:$dst, GR64:$src1),
                  "rol{q} $dst",
                  [(set GR64:$dst, (rotl GR64:$src1, (i8 1)))]>;
} // isTwoAddress

def ROL64mCL :  I<0xD3, MRM0m, (ops i64mem:$dst),
                  "rol{q} {%cl, $dst|$dst, %CL}",
                  [(store (rotl (loadi64 addr:$dst), CL), addr:$dst)]>,
               Imp<[CL],[]>;
def ROL64mi  : RIi8<0xC1, MRM0m, (ops i64mem:$dst, i8imm:$src),
                    "rol{q} {$src, $dst|$dst, $src}",
                [(store (rotl (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def ROL64m1  : RI<0xD1, MRM0m, (ops i64mem:$dst),
                 "rol{q} $dst",
               [(store (rotl (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

let isTwoAddress = 1 in {
def ROR64rCL : RI<0xD3, MRM1r, (ops GR64:$dst, GR64:$src),
                  "ror{q} {%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (rotr GR64:$src, CL))]>, Imp<[CL],[]>;
def ROR64ri  : RIi8<0xC1, MRM1r, (ops GR64:$dst, GR64:$src1, i8imm:$src2),
                    "ror{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (rotr GR64:$src1, (i8 imm:$src2)))]>;
def ROR64r1  : RI<0xD1, MRM1r, (ops GR64:$dst, GR64:$src1),
                  "ror{q} $dst",
                  [(set GR64:$dst, (rotr GR64:$src1, (i8 1)))]>;
} // isTwoAddress

def ROR64mCL : RI<0xD3, MRM1m, (ops i64mem:$dst), 
                  "ror{q} {%cl, $dst|$dst, %CL}",
                  [(store (rotr (loadi64 addr:$dst), CL), addr:$dst)]>,
               Imp<[CL],[]>;
def ROR64mi  : RIi8<0xC1, MRM1m, (ops i64mem:$dst, i8imm:$src),
                    "ror{q} {$src, $dst|$dst, $src}",
                [(store (rotr (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def ROR64m1  : RI<0xD1, MRM1m, (ops i64mem:$dst),
                 "ror{q} $dst",
               [(store (rotr (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

// Double shift instructions (generalizations of rotate)
let isTwoAddress = 1 in {
def SHLD64rrCL : RI<0xA5, MRMDestReg, (ops GR64:$dst, GR64:$src1, GR64:$src2),
                    "shld{q} {%cl, $src2, $dst|$dst, $src2, %CL}", []>,
                 Imp<[CL],[]>, TB;
def SHRD64rrCL : RI<0xAD, MRMDestReg, (ops GR64:$dst, GR64:$src1, GR64:$src2),
                    "shrd{q} {%cl, $src2, $dst|$dst, $src2, %CL}", []>,
                 Imp<[CL],[]>, TB;

let isCommutable = 1 in {  // FIXME: Update X86InstrInfo::commuteInstruction
def SHLD64rri8 : RIi8<0xA4, MRMDestReg,
                      (ops GR64:$dst, GR64:$src1, GR64:$src2, i8imm:$src3),
                      "shld{q} {$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      TB;
def SHRD64rri8 : RIi8<0xAC, MRMDestReg,
                      (ops GR64:$dst, GR64:$src1, GR64:$src2, i8imm:$src3),
                      "shrd{q} {$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                 TB;
} // isCommutable
} // isTwoAddress

// Temporary hack: there is no patterns associated with these instructions
// so we have to tell tblgen that these do not produce results.
let noResults = 1 in {
def SHLD64mrCL : RI<0xA5, MRMDestMem, (ops i64mem:$dst, GR64:$src2),
                    "shld{q} {%cl, $src2, $dst|$dst, $src2, %CL}", []>,
                 Imp<[CL],[]>, TB;
def SHRD64mrCL : RI<0xAD, MRMDestMem, (ops i64mem:$dst, GR64:$src2),
                    "shrd{q} {%cl, $src2, $dst|$dst, $src2, %CL}", []>,
                 Imp<[CL],[]>, TB;
def SHLD64mri8 : RIi8<0xA4, MRMDestMem,
                      (ops i64mem:$dst, GR64:$src2, i8imm:$src3),
                      "shld{q} {$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                 TB;
def SHRD64mri8 : RIi8<0xAC, MRMDestMem, 
                      (ops i64mem:$dst, GR64:$src2, i8imm:$src3),
                      "shrd{q} {$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                 TB;
} // noResults

//===----------------------------------------------------------------------===//
//  Logical Instructions...
//

let isTwoAddress = 1 in
def NOT64r : RI<0xF7, MRM2r, (ops GR64:$dst, GR64:$src), "not{q} $dst",
                [(set GR64:$dst, (not GR64:$src))]>;
def NOT64m : RI<0xF7, MRM2m, (ops i64mem:$dst), "not{q} $dst",
                [(store (not (loadi64 addr:$dst)), addr:$dst)]>;

let isTwoAddress = 1 in {
let isCommutable = 1 in
def AND64rr  : RI<0x21, MRMDestReg, 
                  (ops GR64:$dst, GR64:$src1, GR64:$src2),
                  "and{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (and GR64:$src1, GR64:$src2))]>;
def AND64rm  : RI<0x23, MRMSrcMem,
                  (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                  "and{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (and GR64:$src1, (load addr:$src2)))]>;
def AND64ri32  : RIi32<0x81, MRM4r, 
                       (ops GR64:$dst, GR64:$src1, i64i32imm:$src2),
                       "and{q} {$src2, $dst|$dst, $src2}",
                       [(set GR64:$dst, (and GR64:$src1, i64immSExt32:$src2))]>;
def AND64ri8 : RIi8<0x83, MRM4r, 
                    (ops GR64:$dst, GR64:$src1, i64i8imm:$src2),
                    "and{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (and GR64:$src1, i64immSExt8:$src2))]>;
} // isTwoAddress

def AND64mr  : RI<0x21, MRMDestMem,
                  (ops i64mem:$dst, GR64:$src),
                  "and{q} {$src, $dst|$dst, $src}",
                  [(store (and (load addr:$dst), GR64:$src), addr:$dst)]>;
def AND64mi32  : RIi32<0x81, MRM4m,
                       (ops i64mem:$dst, i64i32imm:$src),
                       "and{q} {$src, $dst|$dst, $src}",
             [(store (and (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst)]>;
def AND64mi8 : RIi8<0x83, MRM4m,
                    (ops i64mem:$dst, i64i8imm :$src),
                    "and{q} {$src, $dst|$dst, $src}",
                 [(store (and (load addr:$dst), i64immSExt8:$src), addr:$dst)]>;

let isTwoAddress = 1 in {
let isCommutable = 1 in
def OR64rr   : RI<0x09, MRMDestReg, (ops GR64:$dst, GR64:$src1, GR64:$src2),
                  "or{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (or GR64:$src1, GR64:$src2))]>;
def OR64rm   : RI<0x0B, MRMSrcMem , (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                  "or{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (or GR64:$src1, (load addr:$src2)))]>;
def OR64ri32 : RIi32<0x81, MRM1r, (ops GR64:$dst, GR64:$src1, i64i32imm:$src2),
                     "or{q} {$src2, $dst|$dst, $src2}",
                     [(set GR64:$dst, (or GR64:$src1, i64immSExt32:$src2))]>;
def OR64ri8  : RIi8<0x83, MRM1r, (ops GR64:$dst, GR64:$src1, i64i8imm:$src2),
                    "or{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (or GR64:$src1, i64immSExt8:$src2))]>;
} // isTwoAddress

def OR64mr : RI<0x09, MRMDestMem, (ops i64mem:$dst, GR64:$src),
                "or{q} {$src, $dst|$dst, $src}",
                [(store (or (load addr:$dst), GR64:$src), addr:$dst)]>;
def OR64mi32 : RIi32<0x81, MRM1m, (ops i64mem:$dst, i64i32imm:$src),
                     "or{q} {$src, $dst|$dst, $src}",
              [(store (or (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst)]>;
def OR64mi8  : RIi8<0x83, MRM1m, (ops i64mem:$dst, i64i8imm:$src),
                    "or{q} {$src, $dst|$dst, $src}",
                  [(store (or (load addr:$dst), i64immSExt8:$src), addr:$dst)]>;

let isTwoAddress = 1 in {
let isCommutable = 1 in
def XOR64rr  : RI<0x31, MRMDestReg,  (ops GR64:$dst, GR64:$src1, GR64:$src2), 
                  "xor{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (xor GR64:$src1, GR64:$src2))]>;
def XOR64rm  : RI<0x33, MRMSrcMem, (ops GR64:$dst, GR64:$src1, i64mem:$src2), 
                  "xor{q} {$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (xor GR64:$src1, (load addr:$src2)))]>;
def XOR64ri32 : RIi32<0x81, MRM6r, 
                      (ops GR64:$dst, GR64:$src1, i64i32imm:$src2), 
                      "xor{q} {$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (xor GR64:$src1, i64immSExt32:$src2))]>;
def XOR64ri8 : RIi8<0x83, MRM6r,  (ops GR64:$dst, GR64:$src1, i64i8imm:$src2),
                    "xor{q} {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (xor GR64:$src1, i64immSExt8:$src2))]>;
} // isTwoAddress

def XOR64mr  : RI<0x31, MRMDestMem, (ops i64mem:$dst, GR64:$src),
                  "xor{q} {$src, $dst|$dst, $src}",
                  [(store (xor (load addr:$dst), GR64:$src), addr:$dst)]>;
def XOR64mi32 : RIi32<0x81, MRM6m, (ops i64mem:$dst, i64i32imm:$src),
                      "xor{q} {$src, $dst|$dst, $src}",
             [(store (xor (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst)]>;
def XOR64mi8 : RIi8<0x83, MRM6m, (ops i64mem:$dst, i64i8imm :$src),
                    "xor{q} {$src, $dst|$dst, $src}",
                 [(store (xor (load addr:$dst), i64immSExt8:$src), addr:$dst)]>;

//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//

// Integer comparison
let isCommutable = 1 in
def TEST64rr : RI<0x85, MRMDestReg, (ops GR64:$src1, GR64:$src2),
                  "test{q} {$src2, $src1|$src1, $src2}",
                  [(X86cmp (and GR64:$src1, GR64:$src2), 0)]>;
def TEST64rm : RI<0x85, MRMSrcMem, (ops GR64:$src1, i64mem:$src2),
                  "test{q} {$src2, $src1|$src1, $src2}",
                  [(X86cmp (and GR64:$src1, (loadi64 addr:$src2)), 0)]>;
def TEST64ri32 : RIi32<0xF7, MRM0r, (ops GR64:$src1, i64i32imm:$src2),
                       "test{q} {$src2, $src1|$src1, $src2}",
                       [(X86cmp (and GR64:$src1, i64immSExt32:$src2), 0)]>;
def TEST64mi32 : RIi32<0xF7, MRM0m, (ops i64mem:$src1, i64i32imm:$src2),
                       "test{q} {$src2, $src1|$src1, $src2}",
                  [(X86cmp (and (loadi64 addr:$src1), i64immSExt32:$src2), 0)]>;

def CMP64rr : RI<0x39, MRMDestReg, (ops GR64:$src1, GR64:$src2),
                 "cmp{q} {$src2, $src1|$src1, $src2}",
                 [(X86cmp GR64:$src1, GR64:$src2)]>;
def CMP64mr : RI<0x39, MRMDestMem, (ops i64mem:$src1, GR64:$src2),
                 "cmp{q} {$src2, $src1|$src1, $src2}",
                 [(X86cmp (loadi64 addr:$src1), GR64:$src2)]>;
def CMP64rm : RI<0x3B, MRMSrcMem, (ops GR64:$src1, i64mem:$src2),
                 "cmp{q} {$src2, $src1|$src1, $src2}",
                 [(X86cmp GR64:$src1, (loadi64 addr:$src2))]>;
def CMP64ri32 : RIi32<0x81, MRM7r, (ops GR64:$src1, i64i32imm:$src2),
                      "cmp{q} {$src2, $src1|$src1, $src2}",
                      [(X86cmp GR64:$src1, i64immSExt32:$src2)]>;
def CMP64mi32 : RIi32<0x81, MRM7m, (ops i64mem:$src1, i64i32imm:$src2),
                      "cmp{q} {$src2, $src1|$src1, $src2}",
                      [(X86cmp (loadi64 addr:$src1), i64immSExt32:$src2)]>;
def CMP64mi8 : RIi8<0x83, MRM7m, (ops i64mem:$src1, i64i8imm:$src2),
                    "cmp{q} {$src2, $src1|$src1, $src2}",
                    [(X86cmp (loadi64 addr:$src1), i64immSExt8:$src2)]>;
def CMP64ri8 : RIi8<0x83, MRM7r, (ops GR64:$src1, i64i8imm:$src2),
                    "cmp{q} {$src2, $src1|$src1, $src2}",
                    [(X86cmp GR64:$src1, i64immSExt8:$src2)]>;

// Conditional moves
let isTwoAddress = 1 in {
def CMOVB64rr : RI<0x42, MRMSrcReg,       // if <u, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovb {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_B))]>, TB;
def CMOVB64rm : RI<0x42, MRMSrcMem,       // if <u, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovb {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_B))]>, TB;
def CMOVAE64rr: RI<0x43, MRMSrcReg,       // if >=u, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovae {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_AE))]>, TB;
def CMOVAE64rm: RI<0x43, MRMSrcMem,       // if >=u, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovae {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_AE))]>, TB;
def CMOVE64rr : RI<0x44, MRMSrcReg,       // if ==, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmove {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_E))]>, TB;
def CMOVE64rm : RI<0x44, MRMSrcMem,       // if ==, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmove {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_E))]>, TB;
def CMOVNE64rr: RI<0x45, MRMSrcReg,       // if !=, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovne {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_NE))]>, TB;
def CMOVNE64rm: RI<0x45, MRMSrcMem,       // if !=, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovne {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_NE))]>, TB;
def CMOVBE64rr: RI<0x46, MRMSrcReg,       // if <=u, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovbe {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_BE))]>, TB;
def CMOVBE64rm: RI<0x46, MRMSrcMem,       // if <=u, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovbe {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_BE))]>, TB;
def CMOVA64rr : RI<0x47, MRMSrcReg,       // if >u, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmova {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_A))]>, TB;
def CMOVA64rm : RI<0x47, MRMSrcMem,       // if >u, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmova {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_A))]>, TB;
def CMOVL64rr : RI<0x4C, MRMSrcReg,       // if <s, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovl {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_L))]>, TB;
def CMOVL64rm : RI<0x4C, MRMSrcMem,       // if <s, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovl {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_L))]>, TB;
def CMOVGE64rr: RI<0x4D, MRMSrcReg,       // if >=s, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovge {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_GE))]>, TB;
def CMOVGE64rm: RI<0x4D, MRMSrcMem,       // if >=s, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovge {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_GE))]>, TB;
def CMOVLE64rr: RI<0x4E, MRMSrcReg,       // if <=s, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovle {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_LE))]>, TB;
def CMOVLE64rm: RI<0x4E, MRMSrcMem,       // if <=s, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovle {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_LE))]>, TB;
def CMOVG64rr : RI<0x4F, MRMSrcReg,       // if >s, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovg {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_G))]>, TB;
def CMOVG64rm : RI<0x4F, MRMSrcMem,       // if >s, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovg {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_G))]>, TB;
def CMOVS64rr : RI<0x48, MRMSrcReg,       // if signed, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovs {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_S))]>, TB;
def CMOVS64rm : RI<0x48, MRMSrcMem,       // if signed, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovs {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_S))]>, TB;
def CMOVNS64rr: RI<0x49, MRMSrcReg,       // if !signed, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovns {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_NS))]>, TB;
def CMOVNS64rm: RI<0x49, MRMSrcMem,       // if !signed, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovns {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_NS))]>, TB;
def CMOVP64rr : RI<0x4A, MRMSrcReg,       // if parity, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovp {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_P))]>, TB;
def CMOVP64rm : RI<0x4A, MRMSrcMem,       // if parity, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovp {$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_P))]>, TB;
def CMOVNP64rr : RI<0x4B, MRMSrcReg,       // if !parity, GR64 = GR64
                   (ops GR64:$dst, GR64:$src1, GR64:$src2),
                   "cmovnp {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_NP))]>, TB;
def CMOVNP64rm : RI<0x4B, MRMSrcMem,       // if !parity, GR64 = [mem64]
                   (ops GR64:$dst, GR64:$src1, i64mem:$src2),
                   "cmovnp {$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_NP))]>, TB;
} // isTwoAddress

//===----------------------------------------------------------------------===//
//  Conversion Instructions...
//

// f64 -> signed i64
def Int_CVTSD2SI64rr: RSDI<0x2D, MRMSrcReg, (ops GR64:$dst, VR128:$src),
                           "cvtsd2si{q} {$src, $dst|$dst, $src}",
                           []>; // TODO: add intrinsic
def Int_CVTSD2SI64rm: RSDI<0x2D, MRMSrcMem, (ops GR64:$dst, f128mem:$src),
                           "cvtsd2si{q} {$src, $dst|$dst, $src}",
                           []>; // TODO: add intrinsic
def CVTTSD2SI64rr: RSDI<0x2C, MRMSrcReg, (ops GR64:$dst, FR64:$src),
                        "cvttsd2si{q} {$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (fp_to_sint FR64:$src))]>;
def CVTTSD2SI64rm: RSDI<0x2C, MRMSrcMem, (ops GR64:$dst, f64mem:$src),
                        "cvttsd2si{q} {$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (fp_to_sint (loadf64 addr:$src)))]>;
def Int_CVTTSD2SI64rr: RSDI<0x2C, MRMSrcReg, (ops GR64:$dst, VR128:$src),
                            "cvttsd2si{q} {$src, $dst|$dst, $src}",
                            []>; // TODO: add intrinsic
def Int_CVTTSD2SI64rm: RSDI<0x2C, MRMSrcMem, (ops GR64:$dst, f128mem:$src),
                            "cvttsd2si{q} {$src, $dst|$dst, $src}",
                            []>; // TODO: add intrinsic

// Signed i64 -> f64
def CVTSI2SD64rr: RSDI<0x2A, MRMSrcReg, (ops FR64:$dst, GR64:$src),
                       "cvtsi2sd{q} {$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (sint_to_fp GR64:$src))]>;
def CVTSI2SD64rm: RSDI<0x2A, MRMSrcMem, (ops FR64:$dst, i64mem:$src),
                       "cvtsi2sd{q} {$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (sint_to_fp (loadi64 addr:$src)))]>;
let isTwoAddress = 1 in {
def Int_CVTSI2SD64rr: RSDI<0x2A, MRMSrcReg,
                           (ops VR128:$dst, VR128:$src1, GR64:$src2),
                           "cvtsi2sd{q} {$src2, $dst|$dst, $src2}",
                           []>; // TODO: add intrinsic
def Int_CVTSI2SD64rm: RSDI<0x2A, MRMSrcMem,
                           (ops VR128:$dst, VR128:$src1, i64mem:$src2),
                           "cvtsi2sd{q} {$src2, $dst|$dst, $src2}",
                           []>; // TODO: add intrinsic
} // isTwoAddress

// Signed i64 -> f32
def CVTSI2SS64rr: RSSI<0x2A, MRMSrcReg, (ops FR32:$dst, GR64:$src),
                       "cvtsi2ss{q} {$src, $dst|$dst, $src}",
                       [(set FR32:$dst, (sint_to_fp GR64:$src))]>;
def CVTSI2SS64rm: RSSI<0x2A, MRMSrcMem, (ops FR32:$dst, i64mem:$src),
                       "cvtsi2ss{q} {$src, $dst|$dst, $src}",
                       [(set FR32:$dst, (sint_to_fp (loadi64 addr:$src)))]>;
let isTwoAddress = 1 in {
def Int_CVTSI2SS64rr: RSSI<0x2A, MRMSrcReg,
                           (ops VR128:$dst, VR128:$src1, GR64:$src2),
                           "cvtsi2ss{q} {$src2, $dst|$dst, $src2}",
                           []>; // TODO: add intrinsic
def Int_CVTSI2SS64rm: RSSI<0x2A, MRMSrcMem,
                           (ops VR128:$dst, VR128:$src1, i64mem:$src2),
                           "cvtsi2ss{q} {$src2, $dst|$dst, $src2}",
                           []>; // TODO: add intrinsic
} // isTwoAddress

// f32 -> signed i64
def Int_CVTSS2SI64rr: RSSI<0x2D, MRMSrcReg, (ops GR64:$dst, VR128:$src),
                           "cvtss2si{q} {$src, $dst|$dst, $src}",
                           []>; // TODO: add intrinsic
def Int_CVTSS2SI64rm: RSSI<0x2D, MRMSrcMem, (ops GR64:$dst, f32mem:$src),
                           "cvtss2si{q} {$src, $dst|$dst, $src}",
                           []>; // TODO: add intrinsic
def CVTTSS2SI64rr: RSSI<0x2C, MRMSrcReg, (ops GR64:$dst, FR32:$src),
                        "cvttss2si{q} {$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (fp_to_sint FR32:$src))]>;
def CVTTSS2SI64rm: RSSI<0x2C, MRMSrcMem, (ops GR64:$dst, f32mem:$src),
                        "cvttss2si{q} {$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (fp_to_sint (loadf32 addr:$src)))]>;
def Int_CVTTSS2SI64rr: RSSI<0x2C, MRMSrcReg, (ops GR64:$dst, VR128:$src),
                            "cvttss2si{q} {$src, $dst|$dst, $src}",
                            []>; // TODO: add intrinsic
def Int_CVTTSS2SI64rm: RSSI<0x2C, MRMSrcMem, (ops GR64:$dst, f32mem:$src),
                            "cvttss2si{q} {$src, $dst|$dst, $src}",
                            []>; // TODO: add intrinsic

//===----------------------------------------------------------------------===//
// Alias Instructions
//===----------------------------------------------------------------------===//

// Truncate
// In 64-mode, each 64-bit and 32-bit registers has a low 8-bit sub-register.
def TRUNC_64to8  : I<0x88, MRMDestReg, (ops GR8:$dst, GR64:$src),
                     "mov{b} {${src:subreg8}, $dst|$dst, ${src:subreg8}",
                     [(set GR8:$dst, (trunc GR64:$src))]>;
def TRUNC_32to8  : I<0x88, MRMDestReg, (ops GR8:$dst, GR32:$src),
                     "mov{b} {${src:subreg8}, $dst|$dst, ${src:subreg8}",
                     [(set GR8:$dst, (trunc GR32:$src))]>,
                   Requires<[In64BitMode]>;
def TRUNC_16to8  : I<0x88, MRMDestReg, (ops GR8:$dst, GR16:$src),
                     "mov{b} {${src:subreg8}, $dst|$dst, ${src:subreg8}}",
                     [(set GR8:$dst, (trunc GR16:$src))]>,
                   Requires<[In64BitMode]>;

def TRUNC_64to16 : I<0x89, MRMDestReg, (ops GR16:$dst, GR64:$src),
                     "mov{w} {${src:subreg16}, $dst|$dst, ${src:subreg16}}",
                     [(set GR16:$dst, (trunc GR64:$src))]>;

def TRUNC_64to32 : I<0x89, MRMDestReg, (ops GR32:$dst, GR64:$src),
                     "mov{l} {${src:subreg32}, $dst|$dst, ${src:subreg32}}",
                     [(set GR32:$dst, (trunc GR64:$src))]>;

// Zero-extension
// TODO: Remove this after proper i32 -> i64 zext support.
def PsMOVZX64rr32: I<0x89, MRMDestReg, (ops GR64:$dst, GR32:$src),
                     "mov{l} {$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                     [(set GR64:$dst, (zext GR32:$src))]>;
def PsMOVZX64rm32: I<0x8B, MRMSrcMem, (ops GR64:$dst, i32mem:$src),
                     "mov{l} {$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                     [(set GR64:$dst, (zextloadi64i32 addr:$src))]>;


// Alias instructions that map movr0 to xor.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
// FIXME: AddedComplexity gives MOV64r0 a higher priority than MOV64ri32. Remove
// when we have a better way to specify isel priority.
let AddedComplexity = 1 in
def MOV64r0  : RI<0x31, MRMInitReg,  (ops GR64:$dst), 
                 "xor{q} $dst, $dst",
                 [(set GR64:$dst, 0)]>;

// Materialize i64 constant where top 32-bits are zero.
let AddedComplexity = 1 in
def MOV64ri64i32 : Ii32<0xB8, AddRegFrm, (ops GR64:$dst, i64i32imm:$src),
                        "mov{l} {$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                        [(set GR64:$dst, i64immZExt32:$src)]>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// ConstantPool GlobalAddress, ExternalSymbol, and JumpTable
def : Pat<(i64 (X86Wrapper tconstpool  :$dst)),
          (MOV64ri tconstpool  :$dst)>, Requires<[NotSmallCode]>;
def : Pat<(i64 (X86Wrapper tjumptable  :$dst)),
          (MOV64ri tjumptable  :$dst)>, Requires<[NotSmallCode]>;
def : Pat<(i64 (X86Wrapper tglobaladdr :$dst)),
          (MOV64ri tglobaladdr :$dst)>, Requires<[NotSmallCode]>;
def : Pat<(i64 (X86Wrapper texternalsym:$dst)),
          (MOV64ri texternalsym:$dst)>, Requires<[NotSmallCode]>;

def : Pat<(store (i64 (X86Wrapper tconstpool:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, tconstpool:$src)>,
          Requires<[SmallCode, IsStatic]>;
def : Pat<(store (i64 (X86Wrapper tjumptable:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, tjumptable:$src)>,
          Requires<[SmallCode, IsStatic]>;
def : Pat<(store (i64 (X86Wrapper tglobaladdr:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, tglobaladdr:$src)>,
          Requires<[SmallCode, IsStatic]>;
def : Pat<(store (i64 (X86Wrapper texternalsym:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, texternalsym:$src)>,
          Requires<[SmallCode, IsStatic]>;

// Calls
// Direct PC relative function call for small code model. 32-bit displacement
// sign extended to 64-bit.
def : Pat<(X86call (i64 tglobaladdr:$dst)),
          (CALL64pcrel32 tglobaladdr:$dst)>;
def : Pat<(X86call (i64 texternalsym:$dst)),
          (CALL64pcrel32 texternalsym:$dst)>;

def : Pat<(X86tailcall (i64 tglobaladdr:$dst)),
          (CALL64pcrel32 tglobaladdr:$dst)>;
def : Pat<(X86tailcall (i64 texternalsym:$dst)),
          (CALL64pcrel32 texternalsym:$dst)>;

def : Pat<(X86tailcall GR64:$dst),
          (CALL64r GR64:$dst)>;

// {s|z}extload bool -> {s|z}extload byte
def : Pat<(sextloadi64i1 addr:$src), (MOVSX64rm8 addr:$src)>;
def : Pat<(zextloadi64i1 addr:$src), (MOVZX64rm8 addr:$src)>;

// extload
def : Pat<(extloadi64i1 addr:$src),  (MOVZX64rm8  addr:$src)>;
def : Pat<(extloadi64i8 addr:$src),  (MOVZX64rm8  addr:$src)>;
def : Pat<(extloadi64i16 addr:$src), (MOVZX64rm16 addr:$src)>;
def : Pat<(extloadi64i32 addr:$src), (PsMOVZX64rm32 addr:$src)>;

// anyext -> zext
def : Pat<(i64 (anyext GR8 :$src)), (MOVZX64rr8  GR8 :$src)>;
def : Pat<(i64 (anyext GR16:$src)), (MOVZX64rr16 GR16:$src)>;
def : Pat<(i64 (anyext GR32:$src)), (PsMOVZX64rr32 GR32:$src)>;
def : Pat<(i64 (anyext (loadi8  addr:$src))), (MOVZX64rm8  addr:$src)>;
def : Pat<(i64 (anyext (loadi16 addr:$src))), (MOVZX64rm16 addr:$src)>;
def : Pat<(i64 (anyext (loadi32 addr:$src))), (PsMOVZX64rm32 addr:$src)>;

//===----------------------------------------------------------------------===//
// Some peepholes
//===----------------------------------------------------------------------===//

// (shl x, 1) ==> (add x, x)
def : Pat<(shl GR64:$src1, (i8 1)), (ADD64rr GR64:$src1, GR64:$src1)>;

// (or (x >> c) | (y << (64 - c))) ==> (shrd64 x, y, c)
def : Pat<(or (srl GR64:$src1, CL:$amt),
              (shl GR64:$src2, (sub 64, CL:$amt))),
          (SHRD64rrCL GR64:$src1, GR64:$src2)>;

def : Pat<(store (or (srl (loadi64 addr:$dst), CL:$amt),
                     (shl GR64:$src2, (sub 64, CL:$amt))), addr:$dst),
          (SHRD64mrCL addr:$dst, GR64:$src2)>;

// (or (x << c) | (y >> (64 - c))) ==> (shld64 x, y, c)
def : Pat<(or (shl GR64:$src1, CL:$amt),
              (srl GR64:$src2, (sub 64, CL:$amt))),
          (SHLD64rrCL GR64:$src1, GR64:$src2)>;

def : Pat<(store (or (shl (loadi64 addr:$dst), CL:$amt),
                     (srl GR64:$src2, (sub 64, CL:$amt))), addr:$dst),
          (SHLD64mrCL addr:$dst, GR64:$src2)>;

//===----------------------------------------------------------------------===//
// X86-64 SSE Instructions
//===----------------------------------------------------------------------===//

// Move instructions...

def MOV64toPQIrr : RPDI<0x6E, MRMSrcReg, (ops VR128:$dst, GR64:$src),
                        "mov{d|q} {$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                          (v2i64 (scalar_to_vector GR64:$src)))]>;
def MOV64toPQIrm : RPDI<0x6E, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                        "mov{d|q} {$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                          (v2i64 (scalar_to_vector (loadi64 addr:$src))))]>;

def MOVPQIto64rr  : RPDI<0x7E, MRMDestReg, (ops GR64:$dst, VR128:$src),
                         "mov{d|q} {$src, $dst|$dst, $src}",
                         [(set GR64:$dst, (vector_extract (v2i64 VR128:$src),
                                           (iPTR 0)))]>;
def MOVPQIto64mr  : RPDI<0x7E, MRMDestMem, (ops i64mem:$dst, VR128:$src),
                         "mov{d|q} {$src, $dst|$dst, $src}",
                         [(store (i64 (vector_extract (v2i64 VR128:$src),
                                       (iPTR 0))), addr:$dst)]>;

def MOV64toSDrr : RPDI<0x6E, MRMSrcReg, (ops FR64:$dst, GR64:$src),
                       "mov{d|q} {$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert GR64:$src))]>;
def MOV64toSDrm : RPDI<0x6E, MRMSrcMem, (ops FR64:$dst, i64mem:$src),
                       "mov{d|q} {$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert (loadi64 addr:$src)))]>;

def MOVSDto64rr  : RPDI<0x7E, MRMDestReg, (ops GR64:$dst, FR64:$src),
                        "mov{d|q} {$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (bitconvert FR64:$src))]>;
def MOVSDto64mr  : RPDI<0x7E, MRMDestMem, (ops i64mem:$dst, FR64:$src),
                        "mov{d|q} {$src, $dst|$dst, $src}",
                        [(store (i64 (bitconvert FR64:$src)), addr:$dst)]>;
