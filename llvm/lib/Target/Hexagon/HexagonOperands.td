//===- HexagonOperands.td - Hexagon immediate processing -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illnois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Immediate operands.

let PrintMethod = "printImmOperand" in {
  // f32Ext type is used to identify constant extended floating point immediates.
  def f32Ext : Operand<f32>;
  def s32Imm : Operand<i32>;
  def s26_6Imm : Operand<i32>;
  def s16Imm : Operand<i32>;
  def s12Imm : Operand<i32>;
  def s11Imm : Operand<i32>;
  def s11_0Imm : Operand<i32>;
  def s11_1Imm : Operand<i32>;
  def s11_2Imm : Operand<i32>;
  def s11_3Imm : Operand<i32>;
  def s10Imm : Operand<i32>;
  def s9Imm : Operand<i32>;
  def m9Imm : Operand<i32>;
  def s8Imm : Operand<i32>;
  def s8Imm64 : Operand<i64>;
  def s6Imm : Operand<i32>;
  def s4Imm : Operand<i32>;
  def s4_0Imm : Operand<i32>;
  def s4_1Imm : Operand<i32>;
  def s4_2Imm : Operand<i32>;
  def s4_3Imm : Operand<i32>;
  def u64Imm : Operand<i64>;
  def u32Imm : Operand<i32>;
  def u26_6Imm : Operand<i32>;
  def u16Imm : Operand<i32>;
  def u16_0Imm : Operand<i32>;
  def u16_1Imm : Operand<i32>;
  def u16_2Imm : Operand<i32>;
  def u11_3Imm : Operand<i32>;
  def u10Imm : Operand<i32>;
  def u9Imm : Operand<i32>;
  def u8Imm : Operand<i32>;
  def u7Imm : Operand<i32>;
  def u6Imm : Operand<i32>;
  def u6_0Imm : Operand<i32>;
  def u6_1Imm : Operand<i32>;
  def u6_2Imm : Operand<i32>;
  def u6_3Imm : Operand<i32>;
  def u5Imm : Operand<i32>;
  def u4Imm : Operand<i32>;
  def u3Imm : Operand<i32>;
  def u2Imm : Operand<i32>;
  def u1Imm : Operand<i32>;
  def n8Imm : Operand<i32>;
  def m6Imm : Operand<i32>;
}

let PrintMethod = "printNOneImmOperand" in
def nOneImm : Operand<i32>;

//
// Immediate predicates
//
def s32ImmPred  : PatLeaf<(i32 imm), [{
  // s32ImmPred predicate - True if the immediate fits in a 32-bit sign extended
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<32>(v);
}]>;

def s32_24ImmPred  : PatLeaf<(i32 imm), [{
  // s32_24ImmPred predicate - True if the immediate fits in a 32-bit sign
  // extended field that is a multiple of 0x1000000.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<32,24>(v);
}]>;

def s32_16s8ImmPred  : PatLeaf<(i32 imm), [{
  // s32_16s8ImmPred predicate - True if the immediate fits in a 32-bit sign
  // extended field that is a multiple of 0x10000.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<24,16>(v);
}]>;

def s26_6ImmPred  : PatLeaf<(i32 imm), [{
  // s26_6ImmPred predicate - True if the immediate fits in a 32-bit
  // sign extended field.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<26,6>(v);
}]>;


def s16ImmPred  : PatLeaf<(i32 imm), [{
  // s16ImmPred predicate - True if the immediate fits in a 16-bit sign extended
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<16>(v);
}]>;


def s13ImmPred  : PatLeaf<(i32 imm), [{
  // s13ImmPred predicate - True if the immediate fits in a 13-bit sign extended
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<13>(v);
}]>;


def s12ImmPred  : PatLeaf<(i32 imm), [{
  // s12ImmPred predicate - True if the immediate fits in a 12-bit
  // sign extended field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<12>(v);
}]>;

def s11_0ImmPred  : PatLeaf<(i32 imm), [{
  // s11_0ImmPred predicate - True if the immediate fits in a 11-bit
  // sign extended field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<11>(v);
}]>;


def s11_1ImmPred  : PatLeaf<(i32 imm), [{
  // s11_1ImmPred predicate - True if the immediate fits in a 12-bit
  // sign extended field and is a multiple of 2.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<11,1>(v);
}]>;


def s11_2ImmPred  : PatLeaf<(i32 imm), [{
  // s11_2ImmPred predicate - True if the immediate fits in a 13-bit
  // sign extended field and is a multiple of 4.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<11,2>(v);
}]>;


def s11_3ImmPred  : PatLeaf<(i32 imm), [{
  // s11_3ImmPred predicate - True if the immediate fits in a 14-bit
  // sign extended field and is a multiple of 8.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<11,3>(v);
}]>;


def s10ImmPred  : PatLeaf<(i32 imm), [{
  // s10ImmPred predicate - True if the immediate fits in a 10-bit sign extended
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<10>(v);
}]>;


def s9ImmPred  : PatLeaf<(i32 imm), [{
  // s9ImmPred predicate - True if the immediate fits in a 9-bit sign extended
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<9>(v);
}]>;

def m9ImmPred  : PatLeaf<(i32 imm), [{
  // m9ImmPred predicate - True if the immediate fits in a 9-bit magnitude
  // field. The range of m9 is -255 to 255.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<9>(v) && (v != -256);
}]>;

def s8ImmPred  : PatLeaf<(i32 imm), [{
  // s8ImmPred predicate - True if the immediate fits in a 8-bit sign extended
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<8>(v);
}]>;


def s8Imm64Pred  : PatLeaf<(i64 imm), [{
  // s8ImmPred predicate - True if the immediate fits in a 8-bit sign extended
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<8>(v);
}]>;


def s6ImmPred  : PatLeaf<(i32 imm), [{
  // s6ImmPred predicate - True if the immediate fits in a 6-bit sign extended
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<6>(v);
}]>;


def s4_0ImmPred  : PatLeaf<(i32 imm), [{
  // s4_0ImmPred predicate - True if the immediate fits in a 4-bit sign extended
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<4>(v);
}]>;


def s4_1ImmPred  : PatLeaf<(i32 imm), [{
  // s4_1ImmPred predicate - True if the immediate fits in a 4-bit sign extended
  // field of 2.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<4,1>(v);
}]>;


def s4_2ImmPred  : PatLeaf<(i32 imm), [{
  // s4_2ImmPred predicate - True if the immediate fits in a 4-bit sign extended
  // field that is a multiple of 4.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<4,2>(v);
}]>;


def s4_3ImmPred  : PatLeaf<(i32 imm), [{
  // s4_3ImmPred predicate - True if the immediate fits in a 4-bit sign extended
  // field that is a multiple of 8.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<4,3>(v);
}]>;


def u64ImmPred  : PatLeaf<(i64 imm), [{
  // Adding "N ||" to suppress gcc unused warning.
  return (N || true);
}]>;

def u32ImmPred  : PatLeaf<(i32 imm), [{
  // u32ImmPred predicate - True if the immediate fits in a 32-bit field.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<32>(v);
}]>;

def u26_6ImmPred  : PatLeaf<(i32 imm), [{
  // u26_6ImmPred - True if the immediate fits in a 32-bit field and
  // is a multiple of 64.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedUInt<26,6>(v);
}]>;

def u16ImmPred  : PatLeaf<(i32 imm), [{
  // u16ImmPred predicate - True if the immediate fits in a 16-bit unsigned
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<16>(v);
}]>;

def u16_s8ImmPred  : PatLeaf<(i32 imm), [{
  // u16_s8ImmPred predicate - True if the immediate fits in a 16-bit sign
  // extended s8 field.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedUInt<16,8>(v);
}]>;

def u9ImmPred  : PatLeaf<(i32 imm), [{
  // u9ImmPred predicate - True if the immediate fits in a 9-bit unsigned
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<9>(v);
}]>;


def u8ImmPred  : PatLeaf<(i32 imm), [{
  // u8ImmPred predicate - True if the immediate fits in a 8-bit unsigned
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<8>(v);
}]>;

def u7StrictPosImmPred : ImmLeaf<i32, [{
  // u7StrictPosImmPred predicate - True if the immediate fits in an 7-bit
  // unsigned field and is strictly greater than 0.
  return isUInt<7>(Imm) && Imm > 0;
}]>;

def u7ImmPred  : PatLeaf<(i32 imm), [{
  // u7ImmPred predicate - True if the immediate fits in a 7-bit unsigned
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<7>(v);
}]>;


def u6ImmPred  : PatLeaf<(i32 imm), [{
  // u6ImmPred predicate - True if the immediate fits in a 6-bit unsigned
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<6>(v);
}]>;

def u6_0ImmPred  : PatLeaf<(i32 imm), [{
  // u6_0ImmPred predicate - True if the immediate fits in a 6-bit unsigned
  // field. Same as u6ImmPred.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<6>(v);
}]>;

def u6_1ImmPred  : PatLeaf<(i32 imm), [{
  // u6_1ImmPred predicate - True if the immediate fits in a 7-bit unsigned
  // field that is 1 bit alinged - multiple of 2.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedUInt<6,1>(v);
}]>;

def u6_2ImmPred  : PatLeaf<(i32 imm), [{
  // u6_2ImmPred predicate - True if the immediate fits in a 8-bit unsigned
  // field that is 2 bits alinged - multiple of 4.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedUInt<6,2>(v);
}]>;

def u6_3ImmPred  : PatLeaf<(i32 imm), [{
  // u6_3ImmPred predicate - True if the immediate fits in a 9-bit unsigned
  // field that is 3 bits alinged - multiple of 8.
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedUInt<6,3>(v);
}]>;

def u5ImmPred  : PatLeaf<(i32 imm), [{
  // u5ImmPred predicate - True if the immediate fits in a 5-bit unsigned
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<5>(v);
}]>;


def u3ImmPred  : PatLeaf<(i32 imm), [{
  // u3ImmPred predicate - True if the immediate fits in a 3-bit unsigned
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<3>(v);
}]>;


def u2ImmPred  : PatLeaf<(i32 imm), [{
  // u2ImmPred predicate - True if the immediate fits in a 2-bit unsigned
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<2>(v);
}]>;


def u1ImmPred  : PatLeaf<(i1 imm), [{
  // u1ImmPred predicate - True if the immediate fits in a 1-bit unsigned
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<1>(v);
}]>;

def m5BImmPred  : PatLeaf<(i32 imm), [{
  // m5BImmPred predicate - True if the (char) number is in range -1 .. -31
  // and will fit in a 5 bit field when made positive, for use in memops.
  // this is specific to the zero extending of a negative by CombineInstr
  int8_t v = (int8_t)N->getSExtValue();
  return (-31 <= v && v <= -1);
}]>;

def m5HImmPred  : PatLeaf<(i32 imm), [{
  // m5HImmPred predicate - True if the (short) number is in range -1 .. -31
  // and will fit in a 5 bit field when made positive, for use in memops.
  // this is specific to the zero extending of a negative by CombineInstr
  int16_t v = (int16_t)N->getSExtValue();
  return (-31 <= v && v <= -1);
}]>;

def m5ImmPred  : PatLeaf<(i32 imm), [{
  // m5ImmPred predicate - True if the number is in range -1 .. -31
  // and will fit in a 5 bit field when made positive, for use in memops.
  int64_t v = (int64_t)N->getSExtValue();
  return (-31 <= v && v <= -1);
}]>;

//InN means negative integers in [-(2^N - 1), 0]
def n8ImmPred  : PatLeaf<(i32 imm), [{
  // n8ImmPred predicate - True if the immediate fits in a 8-bit signed
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return (-255 <= v && v <= 0);
}]>;

def nOneImmPred  : PatLeaf<(i32 imm), [{
  // nOneImmPred predicate - True if the immediate is -1.
  int64_t v = (int64_t)N->getSExtValue();
  return (-1 == v);
}]>;

def Set5ImmPred : PatLeaf<(i32 imm), [{
  // Set5ImmPred predicate - True if the number is in the series of values.
  // [ 2^0, 2^1, ... 2^31 ]
  // For use in setbit immediate.
  uint32_t v = (int32_t)N->getSExtValue();
  // Constrain to 32 bits, and then check for single bit.
  return ImmIsSingleBit(v);
}]>;

def Clr5ImmPred : PatLeaf<(i32 imm), [{
  // Clr5ImmPred predicate - True if the number is in the series of
  // bit negated values.
  // [ 2^0, 2^1, ... 2^31 ]
  // For use in clrbit immediate.
  // Note: we are bit NOTing the value.
  uint32_t v = ~ (int32_t)N->getSExtValue();
  // Constrain to 32 bits, and then check for single bit.
  return ImmIsSingleBit(v);
}]>;

def SetClr5ImmPred : PatLeaf<(i32 imm), [{
  // SetClr5ImmPred predicate - True if the immediate is in range 0..31.
  int32_t v = (int32_t)N->getSExtValue();
  return (v >= 0 && v <= 31);
}]>;

def Set4ImmPred : PatLeaf<(i32 imm), [{
  // Set4ImmPred predicate - True if the number is in the series of values:
  // [ 2^0, 2^1, ... 2^15 ].
  // For use in setbit immediate.
  uint16_t v = (int16_t)N->getSExtValue();
  // Constrain to 16 bits, and then check for single bit.
  return ImmIsSingleBit(v);
}]>;

def Clr4ImmPred : PatLeaf<(i32 imm), [{
  // Clr4ImmPred predicate - True if the number is in the series of
  // bit negated values:
  // [ 2^0, 2^1, ... 2^15 ].
  // For use in setbit and clrbit immediate.
  uint16_t v = ~ (int16_t)N->getSExtValue();
  // Constrain to 16 bits, and then check for single bit.
  return ImmIsSingleBit(v);
}]>;

def SetClr4ImmPred : PatLeaf<(i32 imm), [{
  // SetClr4ImmPred predicate - True if the immediate is in the range 0..15.
  int16_t v = (int16_t)N->getSExtValue();
  return (v >= 0 && v <= 15);
}]>;

def Set3ImmPred : PatLeaf<(i32 imm), [{
  // Set3ImmPred predicate - True if the number is in the series of values:
  // [ 2^0, 2^1, ... 2^7 ].
  // For use in setbit immediate.
  uint8_t v = (int8_t)N->getSExtValue();
  // Constrain to 8 bits, and then check for single bit.
  return ImmIsSingleBit(v);
}]>;

def Clr3ImmPred : PatLeaf<(i32 imm), [{
  // Clr3ImmPred predicate - True if the number is in the series of
  // bit negated values:
  // [ 2^0, 2^1, ... 2^7 ].
  // For use in setbit and clrbit immediate.
  uint8_t v = ~ (int8_t)N->getSExtValue();
  // Constrain to 8 bits, and then check for single bit.
  return ImmIsSingleBit(v);
}]>;

def SetClr3ImmPred : PatLeaf<(i32 imm), [{
  // SetClr3ImmPred predicate - True if the immediate is in the range  0..7.
  int8_t v = (int8_t)N->getSExtValue();
  return (v >= 0 && v <= 7);
}]>;


// Extendable immediate operands.

let PrintMethod = "printExtOperand" in {
  def s16Ext : Operand<i32>;
  def s12Ext : Operand<i32>;
  def s10Ext : Operand<i32>;
  def s9Ext : Operand<i32>;
  def s8Ext : Operand<i32>;
  def s6Ext : Operand<i32>;
  def s11_0Ext : Operand<i32>;
  def s11_1Ext : Operand<i32>;
  def s11_2Ext : Operand<i32>;
  def s11_3Ext : Operand<i32>;
}

let PrintMethod = "printImmOperand" in
def u0AlwaysExt : Operand<i32>;

// Predicates for constant extendable operands
def s16ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate can fit in a 16-bit sign extended field.
    return isInt<16>(v);
  else {
    if (isInt<16>(v))
      return true;

    // Return true if extending this immediate is profitable and the value
    // can fit in a 32-bit signed field.
    return isConstExtProfitable(Node) && isInt<32>(v);
  }
}]>;

def s10ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate can fit in a 10-bit sign extended field.
    return isInt<10>(v);
  else {
    if (isInt<10>(v))
      return true;

    // Return true if extending this immediate is profitable and the value
    // can fit in a 32-bit signed field.
    return isConstExtProfitable(Node) && isInt<32>(v);
  }
}]>;

def s9ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate can fit in a 9-bit sign extended field.
    return isInt<9>(v);
  else {
    if (isInt<9>(v))
      return true;

    // Return true if extending this immediate is profitable and the value
    // can fit in a 32-bit unsigned field.
    return isConstExtProfitable(Node) && isInt<32>(v);
  }
}]>;

def s8ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate can fit in a 8-bit sign extended field.
    return isInt<8>(v);
  else {
    if (isInt<8>(v))
      return true;

    // Return true if extending this immediate is profitable and the value
    // can fit in a 32-bit signed field.
    return isConstExtProfitable(Node) && isInt<32>(v);
  }
}]>;

def s8_16ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate fits in a 8-bit sign extended field.
    return isInt<8>(v);
  else {
    if (isInt<8>(v))
      return true;

    // Return true if extending this immediate is profitable and the value
    // can't fit in a 16-bit signed field. This is required to avoid
    // unnecessary constant extenders.
    return isConstExtProfitable(Node) && !isInt<16>(v);
  }
}]>;

def s6ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate can fit in a 6-bit sign extended field.
    return isInt<6>(v);
  else {
    if (isInt<6>(v))
      return true;

    // Return true if extending this immediate is profitable and the value
    // can fit in a 32-bit unsigned field.
    return isConstExtProfitable(Node) && isInt<32>(v);
  }
}]>;

def s6_16ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate fits in a 6-bit sign extended field.
    return isInt<6>(v);
  else {
    if (isInt<6>(v))
      return true;

    // Return true if extending this immediate is profitable and the value
    // can't fit in a 16-bit signed field. This is required to avoid
    // unnecessary constant extenders.
    return isConstExtProfitable(Node) && !isInt<16>(v);
  }
}]>;

def s6_10ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate can fit in a 6-bit sign extended field.
    return isInt<6>(v);
  else {
    if (isInt<6>(v))
      return true;

    // Return true if extending this immediate is profitable and the value
    // can't fit in a 10-bit signed field. This is required to avoid
    // unnecessary constant extenders.
    return isConstExtProfitable(Node) && !isInt<10>(v);
  }
}]>;

def s11_0ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate can fit in a 11-bit sign extended field.
    return isShiftedInt<11,0>(v);
  else {
    if (isInt<11>(v))
      return true;

    // Return true if extending this immediate is profitable and the value
    // can fit in a 32-bit signed field.
    return isConstExtProfitable(Node) && isInt<32>(v);
  }
}]>;

def s11_1ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate can fit in a 12-bit sign extended field and
    // is 2 byte aligned.
    return isShiftedInt<11,1>(v);
  else {
    if (isInt<12>(v))
      return isShiftedInt<11,1>(v);

    // Return true if extending this immediate is profitable and the low 1 bit
    // is zero (2-byte aligned).
    return isConstExtProfitable(Node) && isInt<32>(v) && ((v % 2) == 0);
  }
}]>;

def s11_2ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate can fit in a 13-bit sign extended field and
    // is 4-byte aligned.
    return isShiftedInt<11,2>(v);
  else {
    if (isInt<13>(v))
      return isShiftedInt<11,2>(v);

    // Return true if extending this immediate is profitable and the low 2-bits
    // are zero (4-byte aligned).
    return isConstExtProfitable(Node)  && isInt<32>(v) && ((v % 4) == 0);
  }
}]>;

def s11_3ExtPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (!Subtarget.hasV4TOps())
    // Return true if the immediate can fit in a 14-bit sign extended field and
    // is 8-byte aligned.
    return isShiftedInt<11,3>(v);
  else {
    if (isInt<14>(v))
     return isShiftedInt<11,3>(v);

    // Return true if extending this immediate is profitable and the low 3-bits
    // are zero (8-byte aligned).
    return isConstExtProfitable(Node)  && isInt<32>(v) && ((v % 8) == 0);
  }
}]>;
