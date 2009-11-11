//===- LazyValueInfo.cpp - Value constraint analysis ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for lazy computation of value constraint
// information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LazyValueInfo.h"
using namespace llvm;

char LazyValueInfo::ID = 0;
static RegisterPass<LazyValueInfo>
X("lazy-value-info", "Lazy Value Information Analysis", false, true);

namespace llvm {
  FunctionPass *createLazyValueInfoPass() { return new LazyValueInfo(); }
}

LazyValueInfo::LazyValueInfo() : FunctionPass(&ID) {
}

void LazyValueInfo::releaseMemory() {
  
}
