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
class Value;

struct ValueNumbering {

  /// getEqualNumberNodes - Return nodes with the same value number as the
  /// specified Value.  This fills in the argument vector with any equal values.
  ///
  virtual void getEqualNumberNodes(Value *V1,
                                   std::vector<Value*> &RetVals) const = 0;

  virtual ~ValueNumbering();    // We want to be subclassed
};

#endif
