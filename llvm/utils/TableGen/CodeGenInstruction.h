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

  struct CodeGenInstruction {
    Record *TheDef;            // The actual record defining this instruction.
    std::string Name;          // Contents of the 'Name' field.
    std::string Namespace;     // The namespace the instruction is in.

    /// AsmString - The format string used to emit a .s file for the
    /// instruction.
    std::string AsmString;

    /// OperandInfo - For each operand declared in the OperandList of the
    /// instruction, keep track of its record (which specifies the class of the
    /// operand), its type, and the name given to the operand, if any.
    struct OperandInfo {
      Record *Rec;
      MVT::ValueType Ty;
      std::string Name;
      OperandInfo(Record *R, MVT::ValueType T, const std::string &N)
        : Rec(R), Ty(T), Name(N) {}
    };
    
    /// OperandList - The list of declared operands, along with their declared
    /// type (which is a record).
    std::vector<OperandInfo> OperandList;

    // Various boolean values we track for the instruction.
    bool isReturn;
    bool isBranch;
    bool isBarrier;
    bool isCall;
    bool isTwoAddress;
    bool isTerminator;

    CodeGenInstruction(Record *R);

    /// getOperandNamed - Return the index of the operand with the specified
    /// non-empty name.  If the instruction does not have an operand with the
    /// specified name, throw an exception.
    unsigned getOperandNamed(const std::string &Name) const;
  };
}

#endif
