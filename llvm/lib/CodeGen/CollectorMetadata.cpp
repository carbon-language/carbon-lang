//===-- CollectorMetadata.cpp - Garbage collector metadata ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Gordon Henriksen and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CollectorMetadata and CollectorModuleMetadata
// classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CollectorMetadata.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Function.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

namespace {
  
  class VISIBILITY_HIDDEN Printer : public MachineFunctionPass {
    static char ID;
    std::ostream &OS;
    
  public:
    Printer(std::ostream &OS = *cerr);
    
    const char *getPassName() const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
    
    bool runOnMachineFunction(MachineFunction &MF);
  };
  
  class VISIBILITY_HIDDEN Deleter : public MachineFunctionPass {
    static char ID;
    
  public:
    Deleter();
    
    const char *getPassName() const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
    
    bool runOnMachineFunction(MachineFunction &MF);
    bool doFinalization(Module &M);
  };
  
  RegisterPass<CollectorModuleMetadata>
  X("collector-metadata", "Create Garbage Collector Module Metadata");
  
}

// -----------------------------------------------------------------------------

CollectorMetadata::CollectorMetadata(const Function &F)
  : F(F), FrameSize(~0LL) {}

CollectorMetadata::~CollectorMetadata() {}

// -----------------------------------------------------------------------------

char CollectorModuleMetadata::ID = 0;

CollectorModuleMetadata::CollectorModuleMetadata()
  : ImmutablePass((intptr_t)&ID) {}

CollectorModuleMetadata::~CollectorModuleMetadata() {
  clear();
}

CollectorMetadata& CollectorModuleMetadata::insert(const Function *F) {
  assert(Map.find(F) == Map.end() && "Function GC metadata already exists!");
  CollectorMetadata *FMD = new CollectorMetadata(*F);
  Functions.push_back(FMD);
  Map[F] = FMD;
  return *FMD;
}

CollectorMetadata* CollectorModuleMetadata::get(const Function *F) const {
  map_type::iterator I = Map.find(F);
  if (I == Map.end())
    return 0;
  return I->second;
}

void CollectorModuleMetadata::clear() {
  for (iterator I = begin(), E = end(); I != E; ++I)
    delete *I;
  
  Functions.clear();
  Map.clear();
}

// -----------------------------------------------------------------------------

char Printer::ID = 0;

Pass *llvm::createCollectorMetadataPrinter(std::ostream &OS) {
  return new Printer(OS);
}

Printer::Printer(std::ostream &OS)
  : MachineFunctionPass(intptr_t(&ID)), OS(OS) {}

const char *Printer::getPassName() const {
  return "Print Garbage Collector Information";
}

void Printer::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
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

bool Printer::runOnMachineFunction(MachineFunction &MF) {
  if (CollectorMetadata *FD =
                 getAnalysis<CollectorModuleMetadata>().get(MF.getFunction())) {
    
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

Pass *llvm::createCollectorMetadataDeleter() {
  return new Deleter();
}

Deleter::Deleter() : MachineFunctionPass(intptr_t(&ID)) {}

const char *Deleter::getPassName() const {
  return "Delete Garbage Collector Information";
}

void Deleter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<CollectorModuleMetadata>();
}

bool Deleter::runOnMachineFunction(MachineFunction &MF) {
  return false;
}

bool Deleter::doFinalization(Module &M) {
  getAnalysis<CollectorModuleMetadata>().clear();
  return false;
}
