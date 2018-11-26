//===- StackSafetyAnalysis.cpp - Stack memory safety analysis -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/StackSafetyAnalysis.h"

#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "stack-safety"

AnalysisKey StackSafetyAnalysis::Key;

void StackSafetyInfo::print(raw_ostream &O) const { O << "Not Implemented\n"; }

StackSafetyInfo StackSafetyAnalysis::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  return StackSafetyInfo();
}

PreservedAnalyses StackSafetyPrinterPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  OS << "'Stack Safety Local Analysis' for function '" << F.getName() << "'\n";
  AM.getResult<StackSafetyAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}

char StackSafetyInfoWrapperPass::ID = 0;

StackSafetyInfoWrapperPass::StackSafetyInfoWrapperPass() : FunctionPass(ID) {
  initializeStackSafetyInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

void StackSafetyInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.setPreservesAll();
}

void StackSafetyInfoWrapperPass::print(raw_ostream &O, const Module *M) const {
  SSI.print(O);
}

bool StackSafetyInfoWrapperPass::runOnFunction(Function &F) { return false; }

static const char LocalPassArg[] = "stack-safety-local";
static const char LocalPassName[] = "Stack Safety Local Analysis";
INITIALIZE_PASS_BEGIN(StackSafetyInfoWrapperPass, LocalPassArg, LocalPassName,
                      false, true)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(StackSafetyInfoWrapperPass, LocalPassArg, LocalPassName,
                    false, true)
