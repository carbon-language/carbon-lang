//===- Pass.cpp - LLVM Pass Infrastructure Impementation ------------------===//
//
// This file implements the LLVM Pass infrastructure.  It is primarily
// responsible with ensuring that passes are executed and batched together
// optimally.
//
//===----------------------------------------------------------------------===//

#include "llvm/PassManager.h"
#include "PassManagerT.h"         // PassManagerT implementation
#include "llvm/Module.h"
#include "Support/STLExtras.h"
#include "Support/CommandLine.h"
#include <typeinfo>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>

//===----------------------------------------------------------------------===//
//   AnalysisID Class Implementation
//

static std::vector<AnalysisID> CFGOnlyAnalyses;

// Source of unique analysis ID #'s.
unsigned AnalysisID::NextID = 0;

AnalysisID::AnalysisID(const AnalysisID &AID, bool DependsOnlyOnCFG) {
  ID = AID.ID;                    // Implement the copy ctor part...
  Constructor = AID.Constructor;
  
  // If this analysis only depends on the CFG of the function, add it to the CFG
  // only list...
  if (DependsOnlyOnCFG)
    CFGOnlyAnalyses.push_back(AID);
}

//===----------------------------------------------------------------------===//
//   AnalysisResolver Class Implementation
//

void AnalysisResolver::setAnalysisResolver(Pass *P, AnalysisResolver *AR) {
  assert(P->Resolver == 0 && "Pass already in a PassManager!");
  P->Resolver = AR;
}

//===----------------------------------------------------------------------===//
//   AnalysisUsage Class Implementation
//

// preservesCFG - This function should be called to by the pass, iff they do
// not:
//
//  1. Add or remove basic blocks from the function
//  2. Modify terminator instructions in any way.
//
// This function annotates the AnalysisUsage info object to say that analyses
// that only depend on the CFG are preserved by this pass.
//
void AnalysisUsage::preservesCFG() {
  // Since this transformation doesn't modify the CFG, it preserves all analyses
  // that only depend on the CFG (like dominators, loop info, etc...)
  //
  Preserved.insert(Preserved.end(),
                   CFGOnlyAnalyses.begin(), CFGOnlyAnalyses.end());
}


//===----------------------------------------------------------------------===//
// PassManager implementation - The PassManager class is a simple Pimpl class
// that wraps the PassManagerT template.
//
PassManager::PassManager() : PM(new PassManagerT<Module>()) {}
PassManager::~PassManager() { delete PM; }
void PassManager::add(Pass *P) { PM->add(P); }
bool PassManager::run(Module &M) { return PM->run(M); }


//===----------------------------------------------------------------------===//
// TimingInfo Class - This class is used to calculate information about the
// amount of time each pass takes to execute.  This only happens with
// -time-passes is enabled on the command line.
//
static cl::Flag EnableTiming("time-passes", "Time each pass, printing elapsed"
                             " time for each on exit");

static double getTime() {
  struct timeval T;
  gettimeofday(&T, 0);
  return T.tv_sec + T.tv_usec/1000000.0;
}

// Create method.  If Timing is enabled, this creates and returns a new timing
// object, otherwise it returns null.
//
TimingInfo *TimingInfo::create() {
  return EnableTiming ? new TimingInfo() : 0;
}

void TimingInfo::passStarted(Pass *P) { TimingData[P] -= getTime(); }
void TimingInfo::passEnded(Pass *P) { TimingData[P] += getTime(); }

// TimingDtor - Print out information about timing information
TimingInfo::~TimingInfo() {
  // Iterate over all of the data, converting it into the dual of the data map,
  // so that the data is sorted by amount of time taken, instead of pointer.
  //
  std::vector<pair<double, Pass*> > Data;
  double TotalTime = 0;
  for (std::map<Pass*, double>::iterator I = TimingData.begin(),
         E = TimingData.end(); I != E; ++I)
    // Throw out results for "grouping" pass managers...
    if (!dynamic_cast<AnalysisResolver*>(I->first)) {
      Data.push_back(std::make_pair(I->second, I->first));
      TotalTime += I->second;
    }
  
  // Sort the data by time as the primary key, in reverse order...
  std::sort(Data.begin(), Data.end(), greater<pair<double, Pass*> >());

  // Print out timing header...
  cerr << std::string(79, '=') << "\n"
       << "                      ... Pass execution timing report ...\n"
       << std::string(79, '=') << "\n  Total Execution Time: " << TotalTime
       << " seconds\n\n  % Time: Seconds:\tPass Name:\n";

  // Loop through all of the timing data, printing it out...
  for (unsigned i = 0, e = Data.size(); i != e; ++i) {
    fprintf(stderr, "  %6.2f%% %fs\t%s\n", Data[i].first*100 / TotalTime,
            Data[i].first, Data[i].second->getPassName());
  }
  cerr << "  100.00% " << TotalTime << "s\tTOTAL\n"
       << std::string(79, '=') << "\n";
}


//===----------------------------------------------------------------------===//
// Pass debugging information.  Often it is useful to find out what pass is
// running when a crash occurs in a utility.  When this library is compiled with
// debugging on, a command line option (--debug-pass) is enabled that causes the
// pass name to be printed before it executes.
//

// Different debug levels that can be enabled...
enum PassDebugLevel {
  None, PassStructure, PassExecutions, PassDetails
};

static cl::Enum<enum PassDebugLevel> PassDebugging("debug-pass", cl::Hidden,
  "Print PassManager debugging information",
  clEnumVal(None          , "disable debug output"),
  clEnumVal(PassStructure , "print pass structure before run()"),
  clEnumVal(PassExecutions, "print pass name before it is executed"),
  clEnumVal(PassDetails   , "print pass details when it is executed"), 0); 

void PMDebug::PrintPassStructure(Pass *P) {
  if (PassDebugging >= PassStructure)
    P->dumpPassStructure();
}

void PMDebug::PrintPassInformation(unsigned Depth, const char *Action,
                                   Pass *P, Annotable *V) {
  if (PassDebugging >= PassExecutions) {
    std::cerr << (void*)P << std::string(Depth*2+1, ' ') << Action << " '" 
              << P->getPassName();
    if (V) {
      std::cerr << "' on ";

      if (dynamic_cast<Module*>(V)) {
        std::cerr << "Module\n"; return;
      } else if (Function *F = dynamic_cast<Function*>(V))
        std::cerr << "Function '" << F->getName();
      else if (BasicBlock *BB = dynamic_cast<BasicBlock*>(V))
        std::cerr << "BasicBlock '" << BB->getName();
      else if (Value *Val = dynamic_cast<Value*>(V))
        std::cerr << typeid(*Val).name() << " '" << Val->getName();
    }
    std::cerr << "'...\n";
  }
}

void PMDebug::PrintAnalysisSetInfo(unsigned Depth, const char *Msg,
                                   Pass *P, const std::vector<AnalysisID> &Set){
  if (PassDebugging >= PassDetails && !Set.empty()) {
    std::cerr << (void*)P << std::string(Depth*2+3, ' ') << Msg << " Analyses:";
    for (unsigned i = 0; i != Set.size(); ++i) {
      Pass *P = Set[i].createPass();   // Good thing this is just debug code...
      std::cerr << "  " << P->getPassName();
      delete P;
    }
    std::cerr << "\n";
  }
}

// dumpPassStructure - Implement the -debug-passes=PassStructure option
void Pass::dumpPassStructure(unsigned Offset = 0) {
  std::cerr << std::string(Offset*2, ' ') << getPassName() << "\n";
}


//===----------------------------------------------------------------------===//
// Pass Implementation
//

void Pass::addToPassManager(PassManagerT<Module> *PM, AnalysisUsage &AU) {
  PM->addPass(this, AU);
}


// getPassName - Use C++ RTTI to get a SOMEWHAT intelligable name for the pass.
//
const char *Pass::getPassName() const { return typeid(*this).name(); }

//===----------------------------------------------------------------------===//
// FunctionPass Implementation
//

// run - On a module, we run this pass by initializing, runOnFunction'ing once
// for every function in the module, then by finalizing.
//
bool FunctionPass::run(Module &M) {
  bool Changed = doInitialization(M);
  
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())      // Passes are not run on external functions!
    Changed |= runOnFunction(*I);
  
  return Changed | doFinalization(M);
}

// run - On a function, we simply initialize, run the function, then finalize.
//
bool FunctionPass::run(Function &F) {
  if (F.isExternal()) return false;// Passes are not run on external functions!

  return doInitialization(*F.getParent()) | runOnFunction(F)
       | doFinalization(*F.getParent());
}

void FunctionPass::addToPassManager(PassManagerT<Module> *PM,
                                    AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

void FunctionPass::addToPassManager(PassManagerT<Function> *PM,
                                    AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

//===----------------------------------------------------------------------===//
// BasicBlockPass Implementation
//

// To run this pass on a function, we simply call runOnBasicBlock once for each
// function.
//
bool BasicBlockPass::runOnFunction(Function &F) {
  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    Changed |= runOnBasicBlock(*I);
  return Changed;
}

// To run directly on the basic block, we initialize, runOnBasicBlock, then
// finalize.
//
bool BasicBlockPass::run(BasicBlock &BB) {
  Module &M = *BB.getParent()->getParent();
  return doInitialization(M) | runOnBasicBlock(BB) | doFinalization(M);
}

void BasicBlockPass::addToPassManager(PassManagerT<Function> *PM,
                                      AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

void BasicBlockPass::addToPassManager(PassManagerT<BasicBlock> *PM,
                                      AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

