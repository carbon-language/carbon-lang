//=- llvm/Analysis/CallTargets.h - Resolve Indirect Call Targets --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass uses DSA to map targets of all calls, and reports on if it
// thinks it knows all targets of a given call.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CALLTARGETS_H
#define LLVM_ANALYSIS_CALLTARGETS_H

#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"

#include <set>
#include <list>

namespace llvm {

  class CallTargetFinder : public ModulePass {
    std::map<CallSite, std::vector<Function*> > IndMap;
    std::set<CallSite> CompleteSites;
    std::list<CallSite> AllSites;

    void findIndTargets(Module &M);
  public:
    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;

    virtual void print(std::ostream &O, const Module *M) const;

    // Given a CallSite, get an iterator of callees
    std::vector<Function*>::iterator begin(CallSite cs);
    std::vector<Function*>::iterator end(CallSite cs);

    // Iterate over CallSites in program
    std::list<CallSite>::iterator cs_begin();
    std::list<CallSite>::iterator cs_end();

    // Do we think we have complete knowledge of this site?
    // That is, do we think there are no missing callees
    bool isComplete(CallSite cs) const;
  };
  
}

#endif
