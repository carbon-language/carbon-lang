//= ProgramPoint.cpp - Program Points for Path-Sensitive Analysis --*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements methods for subclasses of ProgramPoint.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CFG.h"
#include "clang/Analysis/ProgramPoint.h"

using namespace clang;

BlockEdge::BlockEdge(CFG& cfg, const CFGBlock* B1, const CFGBlock* B2) {    
  if (B1->succ_size() == 1) {
    assert (*(B1->succ_begin()) == B2);
    setRawData(B1, BlockEdgeSrcKind);
  }
  else if (B2->pred_size() == 1) {
    assert (*(B2->pred_begin()) == B1);
    setRawData(B2, BlockEdgeDstKind);
  }
  else 
    setRawData(cfg.getBlockEdgeImpl(B1,B2), BlockEdgeAuxKind);
}

CFGBlock* BlockEdge::getSrc() const {
  switch (getKind()) {
    default:
      assert (false && "Invalid BlockEdgeKind.");
      return NULL;
      
    case BlockEdgeSrcKind:
      return reinterpret_cast<CFGBlock*>(getRawPtr());
      
    case BlockEdgeDstKind:
      return *(reinterpret_cast<CFGBlock*>(getRawPtr())->pred_begin());        
      
    case BlockEdgeAuxKind:
      return reinterpret_cast<BPair*>(getRawPtr())->first;
  }
}

CFGBlock* BlockEdge::getDst() const {
  switch (getKind()) {
    default:
      assert (false && "Invalid BlockEdgeKind.");
      return NULL;
      
    case BlockEdgeSrcKind:
      return *(reinterpret_cast<CFGBlock*>(getRawPtr())->succ_begin());
      
    case BlockEdgeDstKind:
      return reinterpret_cast<CFGBlock*>(getRawPtr());
      
    case BlockEdgeAuxKind:
      return reinterpret_cast<BPair*>(getRawPtr())->second;
  }
}
