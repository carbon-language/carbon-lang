//===- llvm/Analysis/MemoryDependenceAnalysis.h - Memory Deps  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an analysis that determines, for a given memory operation,
// what preceding memory operations it depends on.  It builds on alias analysis
// information, and tries to provide a lazy, caching interface to a common kind
// of alias information query.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORY_DEPENDENCE_H
#define LLVM_ANALYSIS_MEMORY_DEPENDENCE_H

#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Compiler.h"
#include <map>

namespace llvm {

class Function;
class FunctionPass;
class Instruction;

class MemoryDependenceAnalysis : public FunctionPass {
  private:
    
    DenseMap<Instruction*, std::pair<Instruction*, bool> > depGraphLocal;
    std::multimap<Instruction*, Instruction*> reverseDep;
  
    Instruction* getCallSiteDependency(CallSite C, Instruction* start,
                                       bool local = true);
  public:
    
    static Instruction* NonLocal;
    static Instruction* None;
    
    static char ID; // Class identification, replacement for typeinfo
    MemoryDependenceAnalysis() : FunctionPass((intptr_t)&ID) {}

    /// Pass Implementation stuff.  This doesn't do any analysis.
    ///
    bool runOnFunction(Function &) {return false; }
    
    /// Clean up memory in between runs
    void releaseMemory() {
      depGraphLocal.clear();
      reverseDep.clear();
    }

    /// getAnalysisUsage - Does not modify anything.  It uses Value Numbering
    /// and Alias Analysis.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    
    /// getDependency - Return the instruction on which a memory operation
    /// depends, starting with start.
    Instruction* getDependency(Instruction* query, Instruction* start = 0,
                               bool local = true);
    
    /// removeInstruction - Remove an instruction from the dependence analysis,
    /// updating the dependence of instructions that previously depended on it.
    void removeInstruction(Instruction* rem);
  };

} // End llvm namespace

#endif
