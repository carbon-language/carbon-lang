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

namespace llvm {
  class Record;

  /// CodeGenRegister - Represents a register definition.
  struct CodeGenRegister {
    Record *TheDef;
    const std::string &getName() const;
    
    CodeGenRegister(Record *R) : TheDef(R) {}
  };


  struct CodeGenRegisterClass {

  };
}

#endif
