//===- CodeGenRegisters.h - Register and RegisterClass Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate information gleaned from the
// target register and register class definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_REGISTERS_H
#define CODEGEN_REGISTERS_H

#include <string>
#include <vector>
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {
  class Record;

  /// CodeGenRegister - Represents a register definition.
  struct CodeGenRegister {
    Record *TheDef;
    const std::string &getName() const;
    unsigned DeclaredSpillSize, DeclaredSpillAlignment;
    CodeGenRegister(Record *R);
  };


  struct CodeGenRegisterClass {
    Record *TheDef;
    std::string Namespace;
    std::vector<Record*> Elements;
    unsigned SpillSize;
    unsigned SpillAlignment;
    MVT::ValueType VT;
    std::string MethodProtos, MethodBodies;

    const std::string &getName() const;

    CodeGenRegisterClass(Record *R);
  };
}

#endif
