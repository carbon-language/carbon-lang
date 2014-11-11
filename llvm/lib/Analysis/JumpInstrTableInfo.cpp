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
#include "llvm/Support/MathExtras.h"

using namespace llvm;

INITIALIZE_PASS(JumpInstrTableInfo, "jump-instr-table-info",
                "Jump-Instruction Table Info", true, true)
char JumpInstrTableInfo::ID = 0;

ImmutablePass *llvm::createJumpInstrTableInfoPass() {
  return new JumpInstrTableInfo();
}

ModulePass *llvm::createJumpInstrTableInfoPass(unsigned Bound) {
  // This cast is always safe, since Bound is always in a subset of uint64_t.
  uint64_t B = static_cast<uint64_t>(Bound);
  return new JumpInstrTableInfo(B);
}

JumpInstrTableInfo::JumpInstrTableInfo(uint64_t ByteAlign)
    : ImmutablePass(ID), Tables(), ByteAlignment(ByteAlign) {
  if (!llvm::isPowerOf2_64(ByteAlign)) {
    // Note that we don't explicitly handle overflow here, since we handle the 0
    // case explicitly when a caller actually tries to create jumptable entries,
    // and this is the return value on overflow.
    ByteAlignment = llvm::NextPowerOf2(ByteAlign);
  }

  initializeJumpInstrTableInfoPass(*PassRegistry::getPassRegistry());
}

JumpInstrTableInfo::~JumpInstrTableInfo() {}

void JumpInstrTableInfo::insertEntry(FunctionType *TableFunTy, Function *Target,
                                     Function *Jump) {
  Tables[TableFunTy].push_back(JumpPair(Target, Jump));
}
