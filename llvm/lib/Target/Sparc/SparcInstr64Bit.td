//===-- SparcInstr64Bit.td - 64-bit instructions for Sparc Target ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains instruction definitions and patterns needed for 64-bit
// code generation on SPARC v9.
//
// Some SPARC v9 instructions are defined in SparcInstrInfo.td because they can
// also be used in 32-bit code running on a SPARC v9 CPU.
//
//===----------------------------------------------------------------------===//

let Predicates = [Is64Bit] in {
// The same integer registers are used for i32 and i64 values.
// When registers hold i32 values, the high bits are don't care.
// This give us free trunc and anyext.
def : Pat<(i64 (anyext i32:$val)), (COPY $val)>;
def : Pat<(i32 (trunc i64:$val)), (COPY $val)>;

} // Predicates = [Is64Bit]


//===----------------------------------------------------------------------===//
// 64-bit Shift Instructions.
//===----------------------------------------------------------------------===//
//
// The 32-bit shift instructions are still available. The left shift srl
// instructions shift all 64 bits, but it only accepts a 5-bit shift amount.
//
// The srl instructions only shift the low 32 bits and clear the high 32 bits.
// Finally, sra shifts the low 32 bits and sign-extends to 64 bits.

let Predicates = [Is64Bit] in {

def : Pat<(i64 (zext i32:$val)), (SRLri $val, 0)>;
def : Pat<(i64 (sext i32:$val)), (SRAri $val, 0)>;

defm SLLX : F3_S<"sllx", 0b100101, 1, shl, i64, I64Regs>;
defm SRLX : F3_S<"srlx", 0b100110, 1, srl, i64, I64Regs>;
defm SRAX : F3_S<"srax", 0b100111, 1, sra, i64, I64Regs>;

} // Predicates = [Is64Bit]


//===----------------------------------------------------------------------===//
// 64-bit Immediates.
//===----------------------------------------------------------------------===//
//
// All 32-bit immediates can be materialized with sethi+or, but 64-bit
// immediates may require more code. There may be a point where it is
// preferable to use a constant pool load instead, depending on the
// microarchitecture.

// The %g0 register is constant 0.
// This is useful for stx %g0, [...], for example.
def : Pat<(i64 0), (i64 G0)>, Requires<[Is64Bit]>;

// Single-instruction patterns.

// The ALU instructions want their simm13 operands as i32 immediates.
def as_i32imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), MVT::i32);
}]>;
def : Pat<(i64 simm13:$val), (ORri (i64 G0), (as_i32imm $val))>;
def : Pat<(i64 SETHIimm:$val), (SETHIi (HI22 $val))>;

// Double-instruction patterns.

// All unsigned i32 immediates can be handled by sethi+or.
def uimm32 : PatLeaf<(imm), [{ return isUInt<32>(N->getZExtValue()); }]>;
def : Pat<(i64 uimm32:$val), (ORri (SETHIi (HI22 $val)), (LO10 $val))>,
      Requires<[Is64Bit]>;

// All negative i33 immediates can be handled by sethi+xor.
def nimm33 : PatLeaf<(imm), [{
  int64_t Imm = N->getSExtValue();
  return Imm < 0 && isInt<33>(Imm);
}]>;
// Bits 10-31 inverted. Same as assembler's %hix.
def HIX22 : SDNodeXForm<imm, [{
  uint64_t Val = (~N->getZExtValue() >> 10) & ((1u << 22) - 1);
  return CurDAG->getTargetConstant(Val, MVT::i32);
}]>;
// Bits 0-9 with ones in bits 10-31. Same as assembler's %lox.
def LOX10 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(~(~N->getZExtValue() & 0x3ff), MVT::i32);
}]>;
def : Pat<(i64 nimm33:$val), (XORri (SETHIi (HIX22 $val)), (LOX10 $val))>,
      Requires<[Is64Bit]>;

// More possible patterns:
//
//   (sllx sethi, n)
//   (sllx simm13, n)
//
// 3 instrs:
//
//   (xor (sllx sethi), simm13)
//   (sllx (xor sethi, simm13))
//
// 4 instrs:
//
//   (or sethi, (sllx sethi))
//   (xnor sethi, (sllx sethi))
//
// 5 instrs:
//
//   (or (sllx sethi), (or sethi, simm13))
//   (xnor (sllx sethi), (or sethi, simm13))
//   (or (sllx sethi), (sllx sethi))
//   (xnor (sllx sethi), (sllx sethi))
//
// Worst case is 6 instrs:
//
//   (or (sllx (or sethi, simmm13)), (or sethi, simm13))

// Bits 42-63, same as assembler's %hh.
def HH22 : SDNodeXForm<imm, [{
  uint64_t Val = (N->getZExtValue() >> 42) & ((1u << 22) - 1);
  return CurDAG->getTargetConstant(Val, MVT::i32);
}]>;
// Bits 32-41, same as assembler's %hm.
def HM10 : SDNodeXForm<imm, [{
  uint64_t Val = (N->getZExtValue() >> 32) & ((1u << 10) - 1);
  return CurDAG->getTargetConstant(Val, MVT::i32);
}]>;
def : Pat<(i64 imm:$val),
          (ORrr (SLLXri (ORri (SETHIi (HH22 $val)), (HM10 $val)), (i64 32)),
                (ORri (SETHIi (HI22 $val)), (LO10 $val)))>,
      Requires<[Is64Bit]>;
