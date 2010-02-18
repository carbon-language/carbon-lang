//===-- PIC16Cloner.h - PIC16 LLVM Cloner for shared functions --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains declaration of a cloner class clone all functions that 
// are shared between the main line code (ML) and interrupt line code (IL).
//
//===----------------------------------------------------------------------===//

#ifndef PIC16CLONER_H
#define PIC16CLONER_H

#include "llvm/ADT/DenseMap.h"

using namespace llvm;
using std::vector;
using std::string;
using std::map;

namespace llvm {
  // forward classes.
  class Value;
  class Function;
  class Module;
  class ModulePass;
  class CallGraph;
  class CallGraphNode;
  class AnalysisUsage;

  class PIC16Cloner : public ModulePass { 
  public:
    static char ID; // Class identification 
    PIC16Cloner() : ModulePass(&ID)  {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<CallGraph>();
    }
    virtual bool runOnModule(Module &M);

  private: // Functions
    // Mark reachable functions for the MainLine or InterruptLine.
    void markCallGraph(CallGraphNode *CGN, string StringMark);

    // Clone auto variables of function specified.
    void CloneAutos(Function *F);
   
    // Clone the body of a function.
    Function *cloneFunction(Function *F);

    // Clone all shared functions.
    void cloneSharedFunctions(CallGraphNode *isrCGN);

    // Remap all call sites to the shared function.
    void remapAllSites(Function *Caller, Function *OrgF, Function *Clone);

    // Error reporting for PIC16Pass
    void reportError(string ErrorString, vector<string> &Values);
    void reportError(string ErrorString);

  private:  //data
    // Records if the interrupt function has already been found.
    // If more than one interrupt function is found then an error
    // should be thrown.
    bool foundISR;

    // This ValueMap maps the auto variables of the original functions with
    // the corresponding cloned auto variable of the cloned function. 
    // This value map is passed during the function cloning so that all the
    // uses of auto variables be updated properly. 
    DenseMap<const Value*, Value*> ValueMap;

    // Map of a already cloned functions. 
    map<Function *, Function *> ClonedFunctionMap;
    typedef map<Function *, Function *>::iterator cloned_map_iterator;
  };
}  // End of anonymous namespace

#endif
