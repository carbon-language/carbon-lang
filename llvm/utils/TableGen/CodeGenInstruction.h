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
    
    /// OperandList - The list of declared operands, along with their declared
    /// type (which is a record).
    std::vector<std::pair<Record*, std::string> > OperandList;

    // Various boolean values we track for the instruction.
    bool isReturn;
    bool isBranch;
    bool isBarrier;
    bool isCall;
    bool isTwoAddress;
    bool isTerminator;

    CodeGenInstruction(Record *R);
  };
}

#endif
