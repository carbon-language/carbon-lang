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
public:
  enum DbgValueKind {
    SD = 0,
    CNST = 1,
    FX = 2
  };
private:
  enum DbgValueKind kind;
  union {
    struct {
      SDNode *Node;         // valid for non-constants
      unsigned ResNo;       // valid for non-constants
    } s;
    Value *Const;           // valid for constants
    unsigned FrameIx;       // valid for stack objects
  } u;
  MDNode *mdPtr;
  uint64_t Offset;
  DebugLoc DL;
  unsigned Order;
public:
  // Constructor for non-constants.
  SDDbgValue(MDNode *mdP, SDNode *N, unsigned R, uint64_t off, DebugLoc dl,
             unsigned O) : mdPtr(mdP), Offset(off), DL(dl), Order(O) {
    kind = SD;
    u.s.Node = N;
    u.s.ResNo = R;
  }

  // Constructor for constants.
  SDDbgValue(MDNode *mdP, Value *C, uint64_t off, DebugLoc dl, unsigned O) : 
    mdPtr(mdP), Offset(off), DL(dl), Order(O) {
    kind = CNST;
    u.Const = C;
  }

  // Constructor for frame indices.
  SDDbgValue(MDNode *mdP, unsigned FI, uint64_t off, DebugLoc dl, unsigned O) : 
    mdPtr(mdP), Offset(off), DL(dl), Order(O) {
    kind = FX;
    u.FrameIx = FI;
  }

  // Returns the kind.
  DbgValueKind getKind() { return kind; }

  // Returns the MDNode pointer.
  MDNode *getMDPtr() { return mdPtr; }

  // Returns the SDNode* for a register ref
  SDNode *getSDNode() { assert (kind==SD); return u.s.Node; }

  // Returns the ResNo for a register ref
  unsigned getResNo() { assert (kind==SD); return u.s.ResNo; }

  // Returns the Value* for a constant
  Value *getConst() { assert (kind==CNST); return u.Const; }

  // Returns the FrameIx for a stack object
  unsigned getFrameIx() { assert (kind==FX); return u.FrameIx; }

  // Returns the offset.
  uint64_t getOffset() { return Offset; }

  // Returns the DebugLoc.
  DebugLoc getDebugLoc() { return DL; }

  // Returns the SDNodeOrder.  This is the order of the preceding node in the
  // input.
  unsigned getOrder() { return Order; }
};

} // end llvm namespace

#endif
