//===- CodeGenInstruction.h - Instruction Class Wrapper ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a wrapper class for the 'Instruction' TableGen class.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_INSTRUCTION_H
#define CODEGEN_INSTRUCTION_H

#include "llvm/CodeGen/ValueTypes.h"
#include <string>
#include <vector>
#include <utility>

namespace llvm {
  class Record;
  class DagInit;

  struct CodeGenInstruction {
    Record *TheDef;            // The actual record defining this instruction.
    std::string Name;          // Contents of the 'Name' field.
    std::string Namespace;     // The namespace the instruction is in.

    /// AsmString - The format string used to emit a .s file for the
    /// instruction.
    std::string AsmString;

    /// OperandInfo - The information we keep track of for each operand in the
    /// operand list for a tablegen instruction.
    struct OperandInfo {
      /// Rec - The definition this operand is declared as.
      ///
      Record *Rec;

      /// Name - If this operand was assigned a symbolic name, this is it,
      /// otherwise, it's empty.
      std::string Name;

      /// PrinterMethodName - The method used to print operands of this type in
      /// the asmprinter.
      std::string PrinterMethodName;

      /// MIOperandNo - Currently (this is meant to be phased out), some logical
      /// operands correspond to multiple MachineInstr operands.  In the X86
      /// target for example, one address operand is represented as 4
      /// MachineOperands.  Because of this, the operand number in the
      /// OperandList may not match the MachineInstr operand num.  Until it
      /// does, this contains the MI operand index of this operand.
      unsigned MIOperandNo;
      unsigned MINumOperands;   // The number of operands.

      /// MIOperandInfo - Default MI operand type. Note an operand may be made
      /// up of multiple MI operands.
      DagInit *MIOperandInfo;

      OperandInfo(Record *R, const std::string &N, const std::string &PMN, 
                  unsigned MION, unsigned MINO, DagInit *MIOI)
        : Rec(R), Name(N), PrinterMethodName(PMN), MIOperandNo(MION),
          MINumOperands(MINO), MIOperandInfo(MIOI) {}
    };

    /// OperandList - The list of declared operands, along with their declared
    /// type (which is a record).
    std::vector<OperandInfo> OperandList;

    // Various boolean values we track for the instruction.
    bool isReturn;
    bool isBranch;
    bool isBarrier;
    bool isCall;
    bool isLoad;
    bool isStore;
    bool isTwoAddress;
    bool isConvertibleToThreeAddress;
    bool isCommutable;
    bool isTerminator;
    bool hasDelaySlot;
    bool usesCustomDAGSchedInserter;
    bool hasVariableNumberOfOperands;

    CodeGenInstruction(Record *R, const std::string &AsmStr);

    /// getOperandNamed - Return the index of the operand with the specified
    /// non-empty name.  If the instruction does not have an operand with the
    /// specified name, throw an exception.
    unsigned getOperandNamed(const std::string &Name) const;
  };
}

#endif
