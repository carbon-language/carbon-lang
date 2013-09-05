//===-- SystemZPatterns.td - SystemZ-specific pattern rules ---*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Record that INSN performs a 64-bit version of unary operator OPERATOR
// in which the operand is sign-extended from 32 to 64 bits.
multiclass SXU<SDPatternOperator operator, Instruction insn> {
  def : Pat<(operator (sext (i32 GR32:$src))),
            (insn GR32:$src)>;
  def : Pat<(operator (sext_inreg GR64:$src, i32)),
            (insn (EXTRACT_SUBREG GR64:$src, subreg_32bit))>;
}

// Record that INSN performs a 64-bit version of binary operator OPERATOR
// in which the first operand has class CLS and which the second operand
// is sign-extended from a 32-bit register.
multiclass SXB<SDPatternOperator operator, RegisterOperand cls,
               Instruction insn> {
  def : Pat<(operator cls:$src1, (sext GR32:$src2)),
            (insn cls:$src1, GR32:$src2)>;
  def : Pat<(operator cls:$src1, (sext_inreg GR64:$src2, i32)),
            (insn cls:$src1, (EXTRACT_SUBREG GR64:$src2, subreg_32bit))>;
}

// Like SXB, but for zero extension.
multiclass ZXB<SDPatternOperator operator, RegisterOperand cls,
               Instruction insn> {
  def : Pat<(operator cls:$src1, (zext GR32:$src2)),
            (insn cls:$src1, GR32:$src2)>;
  def : Pat<(operator cls:$src1, (and GR64:$src2, 0xffffffff)),
            (insn cls:$src1, (EXTRACT_SUBREG GR64:$src2, subreg_32bit))>;
}

// Record that INSN performs a binary read-modify-write operation,
// with LOAD, OPERATOR and STORE being the read, modify and write
// respectively.  MODE is the addressing mode and IMM is the type
// of the second operand.
class RMWI<SDPatternOperator load, SDPatternOperator operator,
           SDPatternOperator store, AddressingMode mode,
           PatFrag imm, Instruction insn>
  : Pat<(store (operator (load mode:$addr), imm:$src), mode:$addr),
        (insn mode:$addr, (UIMM8 imm:$src))>;

// Record that INSN performs binary operation OPERATION on a byte
// memory location.  IMM is the type of the second operand.
multiclass RMWIByte<SDPatternOperator operator, AddressingMode mode,
                    Instruction insn> {
  def : RMWI<anyextloadi8, operator, truncstorei8, mode, imm32, insn>;
  def : RMWI<anyextloadi8, operator, truncstorei8, mode, imm64, insn>;
}

// Record that INSN performs insertion TYPE into a register of class CLS.
// The inserted operand is loaded using LOAD from an address of mode MODE.
multiclass InsertMem<string type, Instruction insn, RegisterOperand cls,
                     SDPatternOperator load, AddressingMode mode> {
  def : Pat<(!cast<SDPatternOperator>("or_as_"##type)
              cls:$src1, (load mode:$src2)),
            (insn cls:$src1, mode:$src2)>;
  def : Pat<(!cast<SDPatternOperator>("or_as_rev"##type)
              (load mode:$src2), cls:$src1),
            (insn cls:$src1, mode:$src2)>;
}

// Try to use MVC instruction INSN for a load of type LOAD followed by a store
// of the same size.  VT is the type of the intermediate (legalized) value and
// LENGTH is the number of bytes loaded by LOAD.
multiclass MVCLoadStore<SDPatternOperator load, ValueType vt, Instruction insn,
                        bits<5> length> {
  def : Pat<(mvc_store (vt (load bdaddr12only:$src)), bdaddr12only:$dest),
            (insn bdaddr12only:$dest, bdaddr12only:$src, length)>;
}

// Use NC-like instruction INSN for block_op operation OPERATOR.
// The other operand is a load of type LOAD, which accesses LENGTH bytes.
// VT is the intermediate legalized type in which the binary operation
// is actually done.
multiclass BinaryLoadStore<SDPatternOperator operator, SDPatternOperator load,
                           ValueType vt, Instruction insn, bits<5> length> {
  def : Pat<(operator (vt (load bdaddr12only:$src)), bdaddr12only:$dest),
            (insn bdaddr12only:$dest, bdaddr12only:$src, length)>;
}

// A convenient way of generating all block peepholes for a particular
// LOAD/VT/LENGTH combination.
multiclass BlockLoadStore<SDPatternOperator load, ValueType vt,
                          Instruction mvc, Instruction nc, Instruction oc,
                          Instruction xc, bits<5> length> {
  defm : MVCLoadStore<load, vt, mvc, length>;
  defm : BinaryLoadStore<block_and1, load, vt, nc, length>;
  defm : BinaryLoadStore<block_and2, load, vt, nc, length>;
  defm : BinaryLoadStore<block_or1,  load, vt, oc, length>;
  defm : BinaryLoadStore<block_or2,  load, vt, oc, length>;
  defm : BinaryLoadStore<block_xor1, load, vt, xc, length>;
  defm : BinaryLoadStore<block_xor2, load, vt, xc, length>;
}

// Record that INSN is a LOAD AND TEST that can be used to compare
// registers in CLS against zero.  The instruction has separate R1 and R2
// operands, but they must be the same when the instruction is used like this.
class CompareZeroFP<Instruction insn, RegisterOperand cls>
  : Pat<(z_cmp cls:$reg, (fpimm0)), (insn cls:$reg, cls:$reg)>;
