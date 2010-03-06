//===-- llvm/CodeGen/SDDbgValue.h - SD dbg_value handling--------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SDDbgValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SDDBGVALUE_H
#define LLVM_CODEGEN_SDDBGVALUE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLoc.h"

namespace llvm {

class MDNode;
class SDNode;
class Value;

/// SDDbgValue - Holds the information from a dbg_value node through SDISel.
/// Either Const or Node is nonzero, but not both.
/// We do not use SDValue here to avoid including its header.

class SDDbgValue {
  SDNode *Node;           // valid for non-constants
  unsigned ResNo;         // valid for non-constants
  Value *Const;           // valid for constants
  MDNode *mdPtr;
  uint64_t Offset;
  DebugLoc DL;
public:
  // Constructor for non-constants.
  SDDbgValue(MDNode *mdP, SDNode *N, unsigned R, uint64_t off, DebugLoc dl) :
    Node(N), ResNo(R), Const(0), mdPtr(mdP), Offset(off), DL(dl) {}

  // Constructor for constants.
  SDDbgValue(MDNode *mdP, Value *C, uint64_t off, DebugLoc dl) : Node(0),
    ResNo(0), Const(C), mdPtr(mdP), Offset(off), DL(dl) {}

  // Returns the MDNode pointer.
  MDNode *getMDPtr() { return mdPtr; }

  // Returns the SDNode* (valid for non-constants only).
  SDNode *getSDNode() { assert (!Const); return Node; }

  // Returns the ResNo (valid for non-constants only).
  unsigned getResNo() { assert (!Const); return ResNo; }

  // Returns the Value* for a constant (invalid for non-constants).
  Value *getConst() { assert (!Node); return Const; }

  // Returns the offset.
  uint64_t getOffset() { return Offset; }

  // Returns the DebugLoc.
  DebugLoc getDebugLoc() { return DL; }
};

} // end llvm namespace

#endif
