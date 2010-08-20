//===- DbgInfoPrinter.cpp - Print debug info in a human readable form ------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that prints instructions, and associated debug
// info:
// 
//   - source/line/col information
//   - original variable name
//   - original type name
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Metadata.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
PrintDirectory("print-fullpath",
               cl::desc("Print fullpath when printing debug info"),
               cl::Hidden);

namespace {
  class PrintDbgInfo : public FunctionPass {
    raw_ostream &Out;
    void printVariableDeclaration(const Value *V);
  public:
    static char ID; // Pass identification
    PrintDbgInfo() : FunctionPass(ID), Out(errs()) {}

    virtual bool runOnFunction(Function &F);
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
  char PrintDbgInfo::ID = 0;
  INITIALIZE_PASS(PrintDbgInfo, "print-dbginfo",
                  "Print debug info in human readable form", false, false);
}

FunctionPass *llvm::createDbgInfoPrinterPass() { return new PrintDbgInfo(); }

void PrintDbgInfo::printVariableDeclaration(const Value *V) {
  std::string DisplayName, File, Directory, Type;
  unsigned LineNo;

  if (!getLocationInfo(V, DisplayName, Type, LineNo, File, Directory))
    return;

  Out << "; ";
  WriteAsOperand(Out, V, false, 0);
  Out << " is variable " << DisplayName
      << " of type " << Type << " declared at ";

  if (PrintDirectory)
    Out << Directory << "/";

  Out << File << ":" << LineNo << "\n";
}

bool PrintDbgInfo::runOnFunction(Function &F) {
  if (F.isDeclaration())
    return false;

  Out << "function " << F.getName() << "\n\n";

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *BB = I;

    if (I != F.begin() && (pred_begin(BB) == pred_end(BB)))
      // Skip dead blocks.
      continue;

    Out << BB->getName();
    Out << ":";

    Out << "\n";

    for (BasicBlock::const_iterator i = BB->begin(), e = BB->end();
         i != e; ++i) {

        printVariableDeclaration(i);

        if (const User *U = dyn_cast<User>(i)) {
          for(unsigned i=0;i<U->getNumOperands();i++)
            printVariableDeclaration(U->getOperand(i));
        }
    }
  }
  return false;
}
