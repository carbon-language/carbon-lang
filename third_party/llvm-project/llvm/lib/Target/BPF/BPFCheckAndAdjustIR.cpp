//===------------ BPFCheckAndAdjustIR.cpp - Check and Adjust IR -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Check IR and adjust IR for verifier friendly codes.
// The following are done for IR checking:
//   - no relocation globals in PHI node.
// The following are done for IR adjustment:
//   - remove __builtin_bpf_passthrough builtins. Target independent IR
//     optimizations are done and those builtins can be removed.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFCORE.h"
#include "BPFTargetMachine.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "bpf-check-and-opt-ir"

using namespace llvm;

namespace {

class BPFCheckAndAdjustIR final : public ModulePass {
  bool runOnModule(Module &F) override;

public:
  static char ID;
  BPFCheckAndAdjustIR() : ModulePass(ID) {}

private:
  void checkIR(Module &M);
  bool adjustIR(Module &M);
  bool removePassThroughBuiltin(Module &M);
  bool removeCompareBuiltin(Module &M);
};
} // End anonymous namespace

char BPFCheckAndAdjustIR::ID = 0;
INITIALIZE_PASS(BPFCheckAndAdjustIR, DEBUG_TYPE, "BPF Check And Adjust IR",
                false, false)

ModulePass *llvm::createBPFCheckAndAdjustIR() {
  return new BPFCheckAndAdjustIR();
}

void BPFCheckAndAdjustIR::checkIR(Module &M) {
  // Ensure relocation global won't appear in PHI node
  // This may happen if the compiler generated the following code:
  //   B1:
  //      g1 = @llvm.skb_buff:0:1...
  //      ...
  //      goto B_COMMON
  //   B2:
  //      g2 = @llvm.skb_buff:0:2...
  //      ...
  //      goto B_COMMON
  //   B_COMMON:
  //      g = PHI(g1, g2)
  //      x = load g
  //      ...
  // If anything likes the above "g = PHI(g1, g2)", issue a fatal error.
  for (Function &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        PHINode *PN = dyn_cast<PHINode>(&I);
        if (!PN || PN->use_empty())
          continue;
        for (int i = 0, e = PN->getNumIncomingValues(); i < e; ++i) {
          auto *GV = dyn_cast<GlobalVariable>(PN->getIncomingValue(i));
          if (!GV)
            continue;
          if (GV->hasAttribute(BPFCoreSharedInfo::AmaAttr) ||
              GV->hasAttribute(BPFCoreSharedInfo::TypeIdAttr))
            report_fatal_error("relocation global in PHI node");
        }
      }
}

bool BPFCheckAndAdjustIR::removePassThroughBuiltin(Module &M) {
  // Remove __builtin_bpf_passthrough()'s which are used to prevent
  // certain IR optimizations. Now major IR optimizations are done,
  // remove them.
  bool Changed = false;
  CallInst *ToBeDeleted = nullptr;
  for (Function &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (ToBeDeleted) {
          ToBeDeleted->eraseFromParent();
          ToBeDeleted = nullptr;
        }

        auto *Call = dyn_cast<CallInst>(&I);
        if (!Call)
          continue;
        auto *GV = dyn_cast<GlobalValue>(Call->getCalledOperand());
        if (!GV)
          continue;
        if (!GV->getName().startswith("llvm.bpf.passthrough"))
          continue;
        Changed = true;
        Value *Arg = Call->getArgOperand(1);
        Call->replaceAllUsesWith(Arg);
        ToBeDeleted = Call;
      }
  return Changed;
}

bool BPFCheckAndAdjustIR::removeCompareBuiltin(Module &M) {
  // Remove __builtin_bpf_compare()'s which are used to prevent
  // certain IR optimizations. Now major IR optimizations are done,
  // remove them.
  bool Changed = false;
  CallInst *ToBeDeleted = nullptr;
  for (Function &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (ToBeDeleted) {
          ToBeDeleted->eraseFromParent();
          ToBeDeleted = nullptr;
        }

        auto *Call = dyn_cast<CallInst>(&I);
        if (!Call)
          continue;
        auto *GV = dyn_cast<GlobalValue>(Call->getCalledOperand());
        if (!GV)
          continue;
        if (!GV->getName().startswith("llvm.bpf.compare"))
          continue;

        Changed = true;
        Value *Arg0 = Call->getArgOperand(0);
        Value *Arg1 = Call->getArgOperand(1);
        Value *Arg2 = Call->getArgOperand(2);

        auto OpVal = cast<ConstantInt>(Arg0)->getValue().getZExtValue();
        CmpInst::Predicate Opcode = (CmpInst::Predicate)OpVal;

        auto *ICmp = new ICmpInst(Opcode, Arg1, Arg2);
        BB.getInstList().insert(Call->getIterator(), ICmp);

        Call->replaceAllUsesWith(ICmp);
        ToBeDeleted = Call;
      }
  return Changed;
}

bool BPFCheckAndAdjustIR::adjustIR(Module &M) {
  bool Changed = removePassThroughBuiltin(M);
  Changed = removeCompareBuiltin(M) || Changed;
  return Changed;
}

bool BPFCheckAndAdjustIR::runOnModule(Module &M) {
  checkIR(M);
  return adjustIR(M);
}
