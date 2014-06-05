//===-- JumpInstrTableInfo.cpp: Info for Jump-Instruction Tables ----------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Information about jump-instruction tables that have been created by
/// JumpInstrTables pass.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jiti"

#include "llvm/Analysis/JumpInstrTableInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"

using namespace llvm;

INITIALIZE_PASS(JumpInstrTableInfo, "jump-instr-table-info",
                "Jump-Instruction Table Info", true, true)
char JumpInstrTableInfo::ID = 0;

ImmutablePass *llvm::createJumpInstrTableInfoPass() {
  return new JumpInstrTableInfo();
}

JumpInstrTableInfo::JumpInstrTableInfo() : ImmutablePass(ID), Tables() {
  initializeJumpInstrTableInfoPass(*PassRegistry::getPassRegistry());
}

JumpInstrTableInfo::~JumpInstrTableInfo() {}

void JumpInstrTableInfo::insertEntry(FunctionType *TableFunTy, Function *Target,
                                     Function *Jump) {
  Tables[TableFunTy].push_back(JumpPair(Target, Jump));
}
