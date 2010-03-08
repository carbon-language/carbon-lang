//=====- SystemZOperands.td - SystemZ Operands defs ---------*- tblgen-*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the various SystemZ instruction operands.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Pattern Stuff.
//===----------------------------------------------------------------------===//

// SystemZ specific condition code. These correspond to CondCode in
// SystemZ.h. They must be kept in synch.
def SYSTEMZ_COND_O   : PatLeaf<(i8 0)>;
def SYSTEMZ_COND_H   : PatLeaf<(i8 1)>;
def SYSTEMZ_COND_NLE : PatLeaf<(i8 2)>;
def SYSTEMZ_COND_L   : PatLeaf<(i8 3)>;
def SYSTEMZ_COND_NHE : PatLeaf<(i8 4)>;
def SYSTEMZ_COND_LH  : PatLeaf<(i8 5)>;
def SYSTEMZ_COND_NE  : PatLeaf<(i8 6)>;
def SYSTEMZ_COND_E   : PatLeaf<(i8 7)>;
def SYSTEMZ_COND_NLH : PatLeaf<(i8 8)>;
def SYSTEMZ_COND_HE  : PatLeaf<(i8 9)>;
def SYSTEMZ_COND_NL  : PatLeaf<(i8 10)>;
def SYSTEMZ_COND_LE  : PatLeaf<(i8 11)>;
def SYSTEMZ_COND_NH  : PatLeaf<(i8 12)>;
def SYSTEMZ_COND_NO  : PatLeaf<(i8 13)>;

def LO8 : SDNodeXForm<imm, [{
  // Transformation function: return low 8 bits.
  return getI8Imm(N->getZExtValue() & 0x00000000000000FFULL);
}]>;

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

def GetI64FromI32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), MVT::i64);
}]>;

def i32ll16 : PatLeaf<(i32 imm), [{
  // i32ll16 predicate - true if the 32-bit immediate has only rightmost 16
  // bits set.
  return ((N->getZExtValue() & 0x000000000000FFFFULL) == N->getZExtValue());
}], LL16>;

def i32lh16 : PatLeaf<(i32 imm), [{
  // i32lh16 predicate - true if the 32-bit immediate has only bits 16-31 set.
  return ((N->getZExtValue() & 0x00000000FFFF0000ULL) == N->getZExtValue());
}], LH16>;

def i32ll16c : PatLeaf<(i32 imm), [{
  // i32ll16c predicate - true if the 32-bit immediate has all bits 16-31 set.
  return ((N->getZExtValue() | 0x00000000FFFF0000ULL) == N->getZExtValue());
}], LL16>;

def i32lh16c : PatLeaf<(i32 imm), [{
  // i32lh16c predicate - true if the 32-bit immediate has all rightmost 16
  //  bits set.
  return ((N->getZExtValue() | 0x000000000000FFFFULL) == N->getZExtValue());
}], LH16>;

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

def i64ll16c : PatLeaf<(i64 imm), [{  
  // i64ll16c predicate - true if the 64-bit immediate has only rightmost 16
  // bits set.
  return ((N->getZExtValue() | 0xFFFFFFFFFFFF0000ULL) == N->getZExtValue());
}], LL16>;

def i64lh16c : PatLeaf<(i64 imm), [{  
  // i64lh16c predicate - true if the 64-bit immediate has only bits 16-31 set.
  return ((N->getZExtValue() | 0xFFFFFFFF0000FFFFULL) == N->getZExtValue());
}], LH16>;

def i64hl16c : PatLeaf<(i64 imm), [{  
  // i64hl16c predicate - true if the 64-bit immediate has only bits 32-47 set.
  return ((N->getZExtValue() | 0xFFFF0000FFFFFFFFULL) == N->getZExtValue());
}], HL16>;

def i64hh16c : PatLeaf<(i64 imm), [{  
  // i64hh16c predicate - true if the 64-bit immediate has only bits 48-63 set.
  return ((N->getZExtValue() | 0x0000FFFFFFFFFFFFULL) == N->getZExtValue());
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
}], LL16>;

def immSExt32 : PatLeaf<(i64 imm), [{
  // immSExt32 predicate - true if the immediate fits in a 32-bit sign extended
  // field.
  uint64_t val = N->getZExtValue();
  return ((int64_t)val == (int32_t)val);
}], LO32>;

def i64lo32 : PatLeaf<(i64 imm), [{
  // i64lo32 predicate - true if the 64-bit immediate has only rightmost 32
  // bits set.
  return ((N->getZExtValue() & 0x00000000FFFFFFFFULL) == N->getZExtValue());
}], LO32>;

def i64hi32 : PatLeaf<(i64 imm), [{
  // i64hi32 predicate - true if the 64-bit immediate has only bits 32-63 set.
  return ((N->getZExtValue() & 0xFFFFFFFF00000000ULL) == N->getZExtValue());
}], HI32>;

def i64lo32c : PatLeaf<(i64 imm), [{
  // i64lo32 predicate - true if the 64-bit immediate has only rightmost 32
  // bits set.
  return ((N->getZExtValue() | 0xFFFFFFFF00000000ULL) == N->getZExtValue());
}], LO32>;

def i64hi32c : PatLeaf<(i64 imm), [{
  // i64hi32 predicate - true if the 64-bit immediate has only bits 32-63 set.
  return ((N->getZExtValue() | 0x00000000FFFFFFFFULL) == N->getZExtValue());
}], HI32>;

def i32immSExt8  : PatLeaf<(i32 imm), [{
  // i32immSExt8 predicate - True if the 32-bit immediate fits in a 8-bit
  // sign extended field.
  return (int32_t)N->getZExtValue() == (int8_t)N->getZExtValue();
}], LO8>;

def i32immSExt16 : PatLeaf<(i32 imm), [{
  // i32immSExt16 predicate - True if the 32-bit immediate fits in a 16-bit
  // sign extended field.
  return (int32_t)N->getZExtValue() == (int16_t)N->getZExtValue();
}], LL16>;

def i64immSExt32 : PatLeaf<(i64 imm), [{
  // i64immSExt32 predicate - True if the 64-bit immediate fits in a 32-bit
  // sign extended field.
  return (int64_t)N->getZExtValue() == (int32_t)N->getZExtValue();
}], LO32>;

def i64immZExt32 : PatLeaf<(i64 imm), [{
  // i64immZExt32 predicate - True if the 64-bit immediate fits in a 32-bit
  // zero extended field.
  return (uint64_t)N->getZExtValue() == (uint32_t)N->getZExtValue();
}], LO32>;

// extloads
def extloadi32i8   : PatFrag<(ops node:$ptr), (i32 (extloadi8  node:$ptr))>;
def extloadi32i16  : PatFrag<(ops node:$ptr), (i32 (extloadi16 node:$ptr))>;
def extloadi64i8   : PatFrag<(ops node:$ptr), (i64 (extloadi8  node:$ptr))>;
def extloadi64i16  : PatFrag<(ops node:$ptr), (i64 (extloadi16 node:$ptr))>;
def extloadi64i32  : PatFrag<(ops node:$ptr), (i64 (extloadi32 node:$ptr))>;

def sextloadi32i8   : PatFrag<(ops node:$ptr), (i32 (sextloadi8  node:$ptr))>;
def sextloadi32i16  : PatFrag<(ops node:$ptr), (i32 (sextloadi16 node:$ptr))>;
def sextloadi64i8   : PatFrag<(ops node:$ptr), (i64 (sextloadi8  node:$ptr))>;
def sextloadi64i16  : PatFrag<(ops node:$ptr), (i64 (sextloadi16 node:$ptr))>;
def sextloadi64i32  : PatFrag<(ops node:$ptr), (i64 (sextloadi32 node:$ptr))>;

def zextloadi32i8   : PatFrag<(ops node:$ptr), (i32 (zextloadi8  node:$ptr))>;
def zextloadi32i16  : PatFrag<(ops node:$ptr), (i32 (zextloadi16 node:$ptr))>;
def zextloadi64i8   : PatFrag<(ops node:$ptr), (i64 (zextloadi8  node:$ptr))>;
def zextloadi64i16  : PatFrag<(ops node:$ptr), (i64 (zextloadi16 node:$ptr))>;
def zextloadi64i32  : PatFrag<(ops node:$ptr), (i64 (zextloadi32 node:$ptr))>;

// A couple of more descriptive operand definitions.
// 32-bits but only 8 bits are significant.
def i32i8imm  : Operand<i32>;
// 32-bits but only 16 bits are significant.
def i32i16imm : Operand<i32>;
// 64-bits but only 32 bits are significant.
def i64i32imm : Operand<i64>;
// Branch targets have OtherVT type.
def brtarget : Operand<OtherVT>;

// Unsigned i12
def u12imm : Operand<i32> {
  let PrintMethod = "printU12ImmOperand";
}
def u12imm64 : Operand<i64> {
  let PrintMethod = "printU12ImmOperand";
}

// Signed i16
def s16imm : Operand<i32> {
  let PrintMethod = "printS16ImmOperand";
}
def s16imm64 : Operand<i64> {
  let PrintMethod = "printS16ImmOperand";
}
// Signed i20
def s20imm : Operand<i32> {
  let PrintMethod = "printS20ImmOperand";
}
def s20imm64 : Operand<i64> {
  let PrintMethod = "printS20ImmOperand";
}
// Signed i32
def s32imm : Operand<i32> {
  let PrintMethod = "printS32ImmOperand";
}
def s32imm64 : Operand<i64> {
  let PrintMethod = "printS32ImmOperand";
}

def imm_pcrel : Operand<i64> {
  let PrintMethod = "printPCRelImmOperand";
}

//===----------------------------------------------------------------------===//
// SystemZ Operand Definitions.
//===----------------------------------------------------------------------===//

// Address operands

// riaddr := reg + imm
def riaddr32 : Operand<i64>,
               ComplexPattern<i64, 2, "SelectAddrRI12Only", []> {
  let PrintMethod = "printRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, u12imm:$disp);
}

def riaddr12 : Operand<i64>,
               ComplexPattern<i64, 2, "SelectAddrRI12", []> {
  let PrintMethod = "printRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, u12imm64:$disp);
}

def riaddr : Operand<i64>,
             ComplexPattern<i64, 2, "SelectAddrRI", []> {
  let PrintMethod = "printRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, s20imm64:$disp);
}

//===----------------------------------------------------------------------===//

// rriaddr := reg + reg + imm
def rriaddr12 : Operand<i64>,
                ComplexPattern<i64, 3, "SelectAddrRRI12", [], []> {
  let PrintMethod = "printRRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, u12imm64:$disp, ADDR64:$index);
}
def rriaddr : Operand<i64>,
              ComplexPattern<i64, 3, "SelectAddrRRI20", [], []> {
  let PrintMethod = "printRRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, s20imm64:$disp, ADDR64:$index);
}
def laaddr : Operand<i64>,
             ComplexPattern<i64, 3, "SelectLAAddr", [add, sub, or, frameindex], []> {
  let PrintMethod = "printRRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, s20imm64:$disp, ADDR64:$index);
}
