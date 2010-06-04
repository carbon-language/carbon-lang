//===- NeonEmitter.h - Generate arm_neon.h for use with clang ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting arm_neon.h, which includes
// a declaration and definition of each function specified by the ARM NEON 
// compiler interface.  See ARM document DUI0348B.
//
//===----------------------------------------------------------------------===//

#ifndef NEON_EMITTER_H
#define NEON_EMITTER_H

#include "Record.h"
#include "TableGenBackend.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

enum OpKind {
  OpNone,
  OpAdd,
  OpSub,
  OpMul,
  OpMla,
  OpMls,
  OpEq,
  OpGe,
  OpLe,
  OpGt,
  OpLt,
  OpNeg,
  OpNot,
  OpAnd,
  OpOr,
  OpXor,
  OpAndNot,
  OpOrNot,
  OpCast
};

enum ClassKind {
  ClassNone,
  ClassI,
  ClassS,
  ClassW,
  ClassB
};

namespace llvm {
  
  class NeonEmitter : public TableGenBackend {
    RecordKeeper &Records;
    StringMap<OpKind> OpMap;
    DenseMap<Record*, ClassKind> ClassMap;
    
  public:
    NeonEmitter(RecordKeeper &R) : Records(R) {
      OpMap["OP_NONE"] = OpNone;
      OpMap["OP_ADD"]  = OpAdd;
      OpMap["OP_SUB"]  = OpSub;
      OpMap["OP_MUL"]  = OpMul;
      OpMap["OP_MLA"]  = OpMla;
      OpMap["OP_MLS"]  = OpMls;
      OpMap["OP_EQ"]   = OpEq;
      OpMap["OP_GE"]   = OpGe;
      OpMap["OP_LE"]   = OpLe;
      OpMap["OP_GT"]   = OpGt;
      OpMap["OP_LT"]   = OpLt;
      OpMap["OP_NEG"]  = OpNeg;
      OpMap["OP_NOT"]  = OpNot;
      OpMap["OP_AND"]  = OpAnd;
      OpMap["OP_OR"]   = OpOr;
      OpMap["OP_XOR"]  = OpXor;
      OpMap["OP_ANDN"] = OpAndNot;
      OpMap["OP_ORN"]  = OpOrNot;
      OpMap["OP_CAST"] = OpCast;

      Record *SI = R.getClass("SInst");
      Record *II = R.getClass("IInst");
      Record *WI = R.getClass("WInst");
      Record *BI = R.getClass("BInst");
      ClassMap[SI] = ClassS;
      ClassMap[II] = ClassI;
      ClassMap[WI] = ClassW;
      ClassMap[BI] = ClassB;
    }
    
    // run - Emit arm_neon.h.inc
    void run(raw_ostream &o);

    // runHeader - Emit all the __builtin prototypes used in arm_neon.h
    void runHeader(raw_ostream &o);
  };
  
} // End llvm namespace

#endif
