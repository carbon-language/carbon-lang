//===- DSCallSiteIterator.h - Iterator for DSGraph call sites ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements an iterator for complete call sites in DSGraphs.  This
// code can either iterator over the normal call list or the aux calls list, and
// is used by the TD and BU passes.
//
//===----------------------------------------------------------------------===//

#ifndef DSCALLSITEITERATOR_H
#define DSCALLSITEITERATOR_H

#include "llvm/Analysis/DataStructure/DSGraph.h"
#include "llvm/Function.h"

namespace llvm {

struct DSCallSiteIterator {
  // FCs are the edges out of the current node are the call site targets...
  std::list<DSCallSite> *FCs;
  std::list<DSCallSite>::iterator CallSite;
  unsigned CallSiteEntry;

  DSCallSiteIterator(std::list<DSCallSite> &CS) : FCs(&CS) {
    CallSite = CS.begin(); CallSiteEntry = 0;
    advanceToValidCallee();
  }

  // End iterator ctor.
  DSCallSiteIterator(std::list<DSCallSite> &CS, bool) : FCs(&CS) {
    CallSite = CS.end(); CallSiteEntry = 0;
  }

  static bool isVAHackFn(const Function *F) {
    return F->getName() == "printf"  || F->getName() == "sscanf" ||
      F->getName() == "fprintf" || F->getName() == "open" ||
      F->getName() == "sprintf" || F->getName() == "fputs" ||
      F->getName() == "fscanf";
  }

  // isUnresolvableFunction - Return true if this is an unresolvable
  // external function.  A direct or indirect call to this cannot be resolved.
  // 
  static bool isUnresolvableFunc(const Function* callee) {
    return callee->isExternal() && !isVAHackFn(callee);
  } 

  void advanceToValidCallee() {
    while (CallSite != FCs->end()) {
      if (CallSite->isDirectCall()) {
        if (CallSiteEntry == 0 &&        // direct call only has one target...
            ! isUnresolvableFunc(CallSite->getCalleeFunc()))
          return;                       // and not an unresolvable external func
      } else {
        DSNode *CalleeNode = CallSite->getCalleeNode();
        if (CallSiteEntry || isCompleteNode(CalleeNode)) {
          const std::vector<GlobalValue*> &Callees = CalleeNode->getGlobals();
          while (CallSiteEntry < Callees.size()) {
            if (isa<Function>(Callees[CallSiteEntry]))
              return;
            ++CallSiteEntry;
          }
        }
      }
      CallSiteEntry = 0;
      ++CallSite;
    }
  }
  
  // isCompleteNode - Return true if we know all of the targets of this node,
  // and if the call sites are not external.
  //
  static inline bool isCompleteNode(DSNode *N) {
    if (N->isIncomplete()) return false;
    const std::vector<GlobalValue*> &Callees = N->getGlobals();
    for (unsigned i = 0, e = Callees.size(); i != e; ++i)
      if (isUnresolvableFunc(cast<Function>(Callees[i])))
        return false;               // Unresolvable external function found...
    return true;  // otherwise ok
  }

public:
  static DSCallSiteIterator begin_aux(DSGraph &G) {
    return G.getAuxFunctionCalls();
  }
  static DSCallSiteIterator end_aux(DSGraph &G) {
    return DSCallSiteIterator(G.getAuxFunctionCalls(), true);
  }
  static DSCallSiteIterator begin_std(DSGraph &G) {
    return G.getFunctionCalls();
  }
  static DSCallSiteIterator end_std(DSGraph &G) {
    return DSCallSiteIterator(G.getFunctionCalls(), true);
  }
  static DSCallSiteIterator begin(std::list<DSCallSite> &CSs) { return CSs; }
  static DSCallSiteIterator end(std::list<DSCallSite> &CSs) {
    return DSCallSiteIterator(CSs, true);
  }
  bool operator==(const DSCallSiteIterator &CSI) const {
    return CallSite == CSI.CallSite && CallSiteEntry == CSI.CallSiteEntry;
  }
  bool operator!=(const DSCallSiteIterator &CSI) const {
    return !operator==(CSI);
  }

  std::list<DSCallSite>::iterator getCallSiteIdx() const { return CallSite; }
  const DSCallSite &getCallSite() const { return *CallSite; }

  Function *operator*() const {
    if (CallSite->isDirectCall()) {
      return CallSite->getCalleeFunc();
    } else {
      DSNode *Node = CallSite->getCalleeNode();
      return cast<Function>(Node->getGlobals()[CallSiteEntry]);
    }
  }

  DSCallSiteIterator& operator++() {                // Preincrement
    ++CallSiteEntry;
    advanceToValidCallee();
    return *this;
  }
  DSCallSiteIterator operator++(int) { // Postincrement
    DSCallSiteIterator tmp = *this; ++*this; return tmp; 
  }
};

} // End llvm namespace

#endif
