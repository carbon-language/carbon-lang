//===-- Internalize.cpp - Mark functions internal -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass loops over all of the functions and variables in the input module.
// If the function or variable is not in the list of external names given to
// the pass it is marked as internal.
//
// This transformation would not be legal or profitable in a regular
// compilation, but it gets extra information from the linker about what is safe
// or profitable.
//
// As an example of a normally illegal transformation: Internalizing a function
// with external linkage. Only if we are told it is only used from within this
// module, it is safe to do it.
//
// On the profitability side: It is always legal to internalize a linkonce_odr
// whose address is not used. Doing so normally would introduce code bloat, but
// if we are told by the linker that the only use of this would be for a
// DSO symbol table, it is profitable to hide it.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "internalize"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <fstream>
#include <set>
using namespace llvm;

STATISTIC(NumAliases  , "Number of aliases internalized");
STATISTIC(NumFunctions, "Number of functions internalized");
STATISTIC(NumGlobals  , "Number of global vars internalized");

// APIFile - A file which contains a list of symbols that should not be marked
// external.
static cl::opt<std::string>
APIFile("internalize-public-api-file", cl::value_desc("filename"),
        cl::desc("A file containing list of symbol names to preserve"));

// APIList - A list of symbols that should not be marked internal.
static cl::list<std::string>
APIList("internalize-public-api-list", cl::value_desc("list"),
        cl::desc("A list of symbol names to preserve"),
        cl::CommaSeparated);

static cl::list<std::string>
DSOList("internalize-dso-list", cl::value_desc("list"),
        cl::desc("A list of symbol names need for a dso symbol table"),
        cl::CommaSeparated);

namespace {
  class InternalizePass : public ModulePass {
    std::set<std::string> ExternalNames;
    std::set<std::string> DSONames;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit InternalizePass();
    explicit InternalizePass(ArrayRef<const char *> ExportList,
                             ArrayRef<const char *> DSOList);
    void LoadFile(const char *Filename);
    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addPreserved<CallGraph>();
    }
  };
} // end anonymous namespace

char InternalizePass::ID = 0;
INITIALIZE_PASS(InternalizePass, "internalize",
                "Internalize Global Symbols", false, false)

InternalizePass::InternalizePass()
  : ModulePass(ID) {
  initializeInternalizePassPass(*PassRegistry::getPassRegistry());
  if (!APIFile.empty())           // If a filename is specified, use it.
    LoadFile(APIFile.c_str());
  ExternalNames.insert(APIList.begin(), APIList.end());
  DSONames.insert(DSOList.begin(), DSOList.end());
}

InternalizePass::InternalizePass(ArrayRef<const char *> ExportList,
                                 ArrayRef<const char *> DSOList)
  : ModulePass(ID){
  initializeInternalizePassPass(*PassRegistry::getPassRegistry());
  for(ArrayRef<const char *>::const_iterator itr = ExportList.begin();
        itr != ExportList.end(); itr++) {
    ExternalNames.insert(*itr);
  }
  for(ArrayRef<const char *>::const_iterator itr = DSOList.begin();
        itr != DSOList.end(); itr++) {
    DSONames.insert(*itr);
  }
}

void InternalizePass::LoadFile(const char *Filename) {
  // Load the APIFile...
  std::ifstream In(Filename);
  if (!In.good()) {
    errs() << "WARNING: Internalize couldn't load file '" << Filename
         << "'! Continuing as if it's empty.\n";
    return; // Just continue as if the file were empty
  }
  while (In) {
    std::string Symbol;
    In >> Symbol;
    if (!Symbol.empty())
      ExternalNames.insert(Symbol);
  }
}

static bool shouldInternalize(const GlobalValue &GV,
                              const std::set<std::string> &ExternalNames,
                              const std::set<std::string> &DSONames) {
  // Function must be defined here
  if (GV.isDeclaration())
    return false;

  // Available externally is really just a "declaration with a body".
  if (GV.hasAvailableExternallyLinkage())
    return false;

  // Already has internal linkage
  if (GV.hasLocalLinkage())
    return false;

  // Marked to keep external?
  if (ExternalNames.count(GV.getName()))
    return false;

  // Not needed for the symbol table?
  if (!DSONames.count(GV.getName()))
    return true;

  // Not a linkonce. Someone can depend on it being on the symbol table.
  if (!GV.hasLinkOnceLinkage())
    return false;

  // The address is not important, we can hide it.
  if (GV.hasUnnamedAddr())
    return true;

  GlobalStatus GS;
  if (GlobalStatus::analyzeGlobal(&GV, GS))
    return false;

  return !GS.IsCompared;
}

bool InternalizePass::runOnModule(Module &M) {
  CallGraph *CG = getAnalysisIfAvailable<CallGraph>();
  CallGraphNode *ExternalNode = CG ? CG->getExternalCallingNode() : 0;
  bool Changed = false;

  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(M, Used, false);

  // We must assume that globals in llvm.used have a reference that not even
  // the linker can see, so we don't internalize them.
  // For llvm.compiler.used the situation is a bit fuzzy. The assembler and
  // linker can drop those symbols. If this pass is running as part of LTO,
  // one might think that it could just drop llvm.compiler.used. The problem
  // is that even in LTO llvm doesn't see every reference. For example,
  // we don't see references from function local inline assembly. To be
  // conservative, we internalize symbols in llvm.compiler.used, but we
  // keep llvm.compiler.used so that the symbol is not deleted by llvm.
  for (SmallPtrSet<GlobalValue *, 8>::iterator I = Used.begin(), E = Used.end();
       I != E; ++I) {
    GlobalValue *V = *I;
    ExternalNames.insert(V->getName());
  }

  // Mark all functions not in the api as internal.
  // FIXME: maybe use private linkage?
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    if (!shouldInternalize(*I, ExternalNames, DSONames))
      continue;

    I->setLinkage(GlobalValue::InternalLinkage);

    if (ExternalNode)
      // Remove a callgraph edge from the external node to this function.
      ExternalNode->removeOneAbstractEdgeTo((*CG)[I]);

    Changed = true;
    ++NumFunctions;
    DEBUG(dbgs() << "Internalizing func " << I->getName() << "\n");
  }

  // Never internalize the llvm.used symbol.  It is used to implement
  // attribute((used)).
  // FIXME: Shouldn't this just filter on llvm.metadata section??
  ExternalNames.insert("llvm.used");
  ExternalNames.insert("llvm.compiler.used");

  // Never internalize anchors used by the machine module info, else the info
  // won't find them.  (see MachineModuleInfo.)
  ExternalNames.insert("llvm.global_ctors");
  ExternalNames.insert("llvm.global_dtors");
  ExternalNames.insert("llvm.global.annotations");

  // Never internalize symbols code-gen inserts.
  // FIXME: We should probably add this (and the __stack_chk_guard) via some
  // type of call-back in CodeGen.
  ExternalNames.insert("__stack_chk_fail");
  ExternalNames.insert("__stack_chk_guard");

  // Mark all global variables with initializers that are not in the api as
  // internal as well.
  // FIXME: maybe use private linkage?
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!shouldInternalize(*I, ExternalNames, DSONames))
      continue;

    I->setLinkage(GlobalValue::InternalLinkage);
    Changed = true;
    ++NumGlobals;
    DEBUG(dbgs() << "Internalized gvar " << I->getName() << "\n");
  }

  // Mark all aliases that are not in the api as internal as well.
  for (Module::alias_iterator I = M.alias_begin(), E = M.alias_end();
       I != E; ++I) {
    if (!shouldInternalize(*I, ExternalNames, DSONames))
      continue;

    I->setLinkage(GlobalValue::InternalLinkage);
    Changed = true;
    ++NumAliases;
    DEBUG(dbgs() << "Internalized alias " << I->getName() << "\n");
  }

  return Changed;
}

ModulePass *llvm::createInternalizePass() {
  return new InternalizePass();
}

ModulePass *llvm::createInternalizePass(ArrayRef<const char *> ExportList,
                                        ArrayRef<const char *> DSOList) {
  return new InternalizePass(ExportList, DSOList);
}
