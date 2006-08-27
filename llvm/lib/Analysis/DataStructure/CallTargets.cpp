//=- lib/Analysis/IPA/CallTargets.cpp - Resolve Call Targets --*- C++ -*-=====//
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
// Loop over all callsites, and lookup the DSNode for that site.  Pull the
// Functions from the node as callees.
// This is essentially a utility pass to simplify later passes that only depend
// on call sites and callees to operate (such as a devirtualizer).
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/DataStructure/DataStructure.h"
#include "llvm/Analysis/DataStructure/DSGraph.h"
#include "llvm/Analysis/DataStructure/CallTargets.h"
#include "llvm/ADT/Statistic.h"
#include <iostream>
#include "llvm/Constants.h"

using namespace llvm;

namespace {
  Statistic<> DirCall("calltarget", "Number of direct calls");
  Statistic<> IndCall("calltarget", "Number of indirect calls");
  Statistic<> CompleteInd("calltarget", "Number of complete indirect calls");
  Statistic<> CompleteEmpty("calltarget", "Number of complete empty calls");

  RegisterPass<CallTargetFinder> X("calltarget","Find Call Targets (uses DSA)");
}

void CallTargetFinder::findIndTargets(Module &M)
{
  TDDataStructures* T = &getAnalysis<TDDataStructures>();
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      for (Function::iterator F = I->begin(), FE = I->end(); F != FE; ++F)
        for (BasicBlock::iterator B = F->begin(), BE = F->end(); B != BE; ++B)
          if (isa<CallInst>(B) || isa<InvokeInst>(B)) {
            CallSite cs = CallSite::get(B);
            AllSites.push_back(cs);
            if (!cs.getCalledFunction()) {
              IndCall++;
              DSNode* N = T->getDSGraph(*cs.getCaller())
                .getNodeForValue(cs.getCalledValue()).getNode();
              N->addFullFunctionList(IndMap[cs]);
              if (N->isComplete() && IndMap[cs].size()) {
                CompleteSites.insert(cs);
                ++CompleteInd;
              } 
              if (N->isComplete() && !IndMap[cs].size()) {
                ++CompleteEmpty;
                std::cerr << "Call site empty: '" << cs.getInstruction()->getName() 
                          << "' In '" << cs.getInstruction()->getParent()->getParent()->getName()
                          << "'\n";
              }
            } else {
              ++DirCall;
              IndMap[cs].push_back(cs.getCalledFunction());
              CompleteSites.insert(cs);
            }
          }
}

void CallTargetFinder::print(std::ostream &O, const Module *M) const
{
  return;
  O << "[* = incomplete] CS: func list\n";
  for (std::map<CallSite, std::vector<Function*> >::const_iterator ii = IndMap.begin(),
         ee = IndMap.end(); ii != ee; ++ii) {
    if (!ii->first.getCalledFunction()) { //only print indirect
      if (!isComplete(ii->first)) {
        O << "* ";
        CallSite cs = ii->first;
        cs.getInstruction()->dump();
        O << cs.getInstruction()->getParent()->getParent()->getName() << " "
          << cs.getInstruction()->getName() << " ";
      }
      O << ii->first.getInstruction() << ":";
      for (std::vector<Function*>::const_iterator i = ii->second.begin(),
             e = ii->second.end(); i != e; ++i) {
        O << " " << (*i)->getName();
      }
      O << "\n";
    }
  }
}

bool CallTargetFinder::runOnModule(Module &M) {
  findIndTargets(M);
  return false;
}

void CallTargetFinder::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<TDDataStructures>();
}

std::vector<Function*>::iterator CallTargetFinder::begin(CallSite cs) {
  return IndMap[cs].begin();
}

std::vector<Function*>::iterator CallTargetFinder::end(CallSite cs) {
  return IndMap[cs].end();
}

bool CallTargetFinder::isComplete(CallSite cs) const {
  return CompleteSites.find(cs) != CompleteSites.end();
}

std::list<CallSite>::iterator CallTargetFinder::cs_begin() {
  return AllSites.begin();
}

std::list<CallSite>::iterator CallTargetFinder::cs_end() {
  return AllSites.end();
}
