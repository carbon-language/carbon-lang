//===-- CollectorMetadata.cpp - Garbage collector metadata ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CollectorMetadata and CollectorModuleMetadata
// classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/GCs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

namespace {
  
  class VISIBILITY_HIDDEN Printer : public FunctionPass {
    static char ID;
    std::ostream &OS;
    
  public:
    explicit Printer(std::ostream &OS = *cerr);
    
    const char *getPassName() const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
    
    bool runOnFunction(Function &F);
  };
  
  class VISIBILITY_HIDDEN Deleter : public FunctionPass {
    static char ID;
    
  public:
    Deleter();
    
    const char *getPassName() const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
    
    bool runOnFunction(Function &F);
    bool doFinalization(Module &M);
  };
  
}

static RegisterPass<CollectorModuleMetadata>
X("collector-metadata", "Create Garbage Collector Module Metadata");

// -----------------------------------------------------------------------------

CollectorMetadata::CollectorMetadata(const Function &F, Collector &C)
  : F(F), C(C), FrameSize(~0LL) {}

CollectorMetadata::~CollectorMetadata() {}

// -----------------------------------------------------------------------------

char CollectorModuleMetadata::ID = 0;

CollectorModuleMetadata::CollectorModuleMetadata()
  : ImmutablePass((intptr_t)&ID) {}

CollectorModuleMetadata::~CollectorModuleMetadata() {
  clear();
}

Collector *CollectorModuleMetadata::
getOrCreateCollector(const Module *M, const std::string &Name) {
  const char *Start = Name.c_str();
  
  collector_map_type::iterator NMI = NameMap.find(Start, Start + Name.size());
  if (NMI != NameMap.end())
    return NMI->getValue();
  
  for (CollectorRegistry::iterator I = CollectorRegistry::begin(),
                                   E = CollectorRegistry::end(); I != E; ++I) {
    if (strcmp(Start, I->getName()) == 0) {
      Collector *C = I->instantiate();
      C->M = M;
      C->Name = Name;
      NameMap.GetOrCreateValue(Start, Start + Name.size()).setValue(C);
      Collectors.push_back(C);
      return C;
    }
  }
  
  cerr << "unsupported collector: " << Name << "\n";
  abort();
}

CollectorMetadata &CollectorModuleMetadata::get(const Function &F) {
  assert(!F.isDeclaration() && "Can only get GCFunctionInfo for a definition!");
  assert(F.hasCollector());
  
  function_map_type::iterator I = Map.find(&F);
  if (I != Map.end())
    return *I->second;
    
  Collector *C = getOrCreateCollector(F.getParent(), F.getCollector());
  CollectorMetadata *MD = C->insertFunctionMetadata(F);
  Map[&F] = MD;
  return *MD;
}

void CollectorModuleMetadata::clear() {
  Map.clear();
  NameMap.clear();
  
  for (iterator I = begin(), E = end(); I != E; ++I)
    delete *I;
  Collectors.clear();
}

// -----------------------------------------------------------------------------

char Printer::ID = 0;

FunctionPass *llvm::createCollectorMetadataPrinter(std::ostream &OS) {
  return new Printer(OS);
}

Printer::Printer(std::ostream &OS)
  : FunctionPass(intptr_t(&ID)), OS(OS) {}

const char *Printer::getPassName() const {
  return "Print Garbage Collector Information";
}

void Printer::getAnalysisUsage(AnalysisUsage &AU) const {
  FunctionPass::getAnalysisUsage(AU);
  AU.setPreservesAll();
  AU.addRequired<CollectorModuleMetadata>();
}

static const char *DescKind(GC::PointKind Kind) {
  switch (Kind) {
    default: assert(0 && "Unknown GC point kind");
    case GC::Loop:     return "loop";
    case GC::Return:   return "return";
    case GC::PreCall:  return "pre-call";
    case GC::PostCall: return "post-call";
  }
}

bool Printer::runOnFunction(Function &F) {
  if (F.hasCollector()) {
    CollectorMetadata *FD = &getAnalysis<CollectorModuleMetadata>().get(F);
    
    OS << "GC roots for " << FD->getFunction().getNameStart() << ":\n";
    for (CollectorMetadata::roots_iterator RI = FD->roots_begin(),
                                           RE = FD->roots_end();
                                           RI != RE; ++RI)
      OS << "\t" << RI->Num << "\t" << RI->StackOffset << "[sp]\n";
    
    OS << "GC safe points for " << FD->getFunction().getNameStart() << ":\n";
    for (CollectorMetadata::iterator PI = FD->begin(),
                                     PE = FD->end(); PI != PE; ++PI) {
      
      OS << "\tlabel " << PI->Num << ": " << DescKind(PI->Kind) << ", live = {";
      
      for (CollectorMetadata::live_iterator RI = FD->live_begin(PI),
                                            RE = FD->live_end(PI);;) {
        OS << " " << RI->Num;
        if (++RI == RE)
          break;
        OS << ",";
      }
      
      OS << " }\n";
    }
  }
  
  return false;
}

// -----------------------------------------------------------------------------

char Deleter::ID = 0;

FunctionPass *llvm::createCollectorMetadataDeleter() {
  return new Deleter();
}

Deleter::Deleter() : FunctionPass(intptr_t(&ID)) {}

const char *Deleter::getPassName() const {
  return "Delete Garbage Collector Information";
}

void Deleter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<CollectorModuleMetadata>();
}

bool Deleter::runOnFunction(Function &MF) {
  return false;
}

bool Deleter::doFinalization(Module &M) {
  CollectorModuleMetadata *CMM = getAnalysisToUpdate<CollectorModuleMetadata>();
  assert(CMM && "Deleter didn't require CollectorModuleMetadata?!");
  CMM->clear();
  return false;
}
