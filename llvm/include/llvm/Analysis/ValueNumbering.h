//===- llvm/Analysis/ValueNumbering.h - Value #'ing Interface ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the abstract ValueNumbering interface, which is used as the
// common interface used by all clients of value numbering information, and
// implemented by all value numbering implementations.
//
// Implementations of this interface must implement the various virtual methods,
// which automatically provides functionality for the entire suite of client
// APIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_VALUE_NUMBERING_H
#define LLVM_ANALYSIS_VALUE_NUMBERING_H

#include <vector>
#include "llvm/Pass.h"

namespace llvm {

class Value;
class Instruction;

struct ValueNumbering {
  virtual ~ValueNumbering();    // We want to be subclassed

  /// getEqualNumberNodes - Return nodes with the same value number as the
  /// specified Value.  This fills in the argument vector with any equal values.
  ///
  virtual void getEqualNumberNodes(Value *V1,
                                   std::vector<Value*> &RetVals) const = 0;

  ///===-------------------------------------------------------------------===//
  /// Interfaces to update value numbering analysis information as the client
  /// changes the program
  ///

  /// deleteInstruction - Clients should invoke this method when they delete an
  /// instruction from the program.  This allows the analysis implementations to
  /// avoid having dangling pointers in their representation.
  virtual void deleteInstruction(Instruction *I) {}
};

extern void BasicValueNumberingStub();
static IncludeFile
HDR_INCLUDE_VALUENUMBERING_CPP((void*)&BasicValueNumberingStub);

} // End llvm namespace

#endif
