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
  def : RMWI<zextloadi8, operator, truncstorei8, mode, imm32, insn>;
  def : RMWI<zextloadi8, operator, truncstorei8, mode, imm64, insn>;
  def : RMWI<sextloadi8, operator, truncstorei8, mode, imm32, insn>;
  def : RMWI<sextloadi8, operator, truncstorei8, mode, imm64, insn>;
  def : RMWI<extloadi8, operator, truncstorei8, mode, imm32, insn>;
  def : RMWI<extloadi8, operator, truncstorei8, mode, imm64, insn>;
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
