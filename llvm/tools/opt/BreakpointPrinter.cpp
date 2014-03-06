//===- BreakpointPrinter.cpp - Breakpoint location printer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Breakpoint location printer.
///
//===----------------------------------------------------------------------===//
#include "BreakpointPrinter.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct BreakpointPrinter : public ModulePass {
  raw_ostream &Out;
  static char ID;
  DITypeIdentifierMap TypeIdentifierMap;

  BreakpointPrinter(raw_ostream &out) : ModulePass(ID), Out(out) {}

  void getContextName(DIDescriptor Context, std::string &N) {
    if (Context.isNameSpace()) {
      DINameSpace NS(Context);
      if (!NS.getName().empty()) {
        getContextName(NS.getContext(), N);
        N = N + NS.getName().str() + "::";
      }
    } else if (Context.isType()) {
      DIType TY(Context);
      if (!TY.getName().empty()) {
        getContextName(TY.getContext().resolve(TypeIdentifierMap), N);
        N = N + TY.getName().str() + "::";
      }
    }
  }

  virtual bool runOnModule(Module &M) {
    TypeIdentifierMap.clear();
    NamedMDNode *CU_Nodes = M.getNamedMetadata("llvm.dbg.cu");
    if (CU_Nodes)
      TypeIdentifierMap = generateDITypeIdentifierMap(CU_Nodes);

    StringSet<> Processed;
    if (NamedMDNode *NMD = M.getNamedMetadata("llvm.dbg.sp"))
      for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
        std::string Name;
        DISubprogram SP(NMD->getOperand(i));
        assert((!SP || SP.isSubprogram()) &&
               "A MDNode in llvm.dbg.sp should be null or a DISubprogram.");
        if (!SP)
          continue;
        getContextName(SP.getContext().resolve(TypeIdentifierMap), Name);
        Name = Name + SP.getDisplayName().str();
        if (!Name.empty() && Processed.insert(Name)) {
          Out << Name << "\n";
        }
      }
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

char BreakpointPrinter::ID = 0;
}

ModulePass *llvm::createBreakpointPrinter(raw_ostream &out) {
  return new BreakpointPrinter(out);
}
