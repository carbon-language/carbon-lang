//===- VerifyUseListOrder.cpp - Use List Order Verifier ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pass to verify use-list order doesn't change after serialization.
//
// Despite it being a verifier, this pass *does* transform the module, since it
// shuffles the use-list of every value.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/IR/UseListOrder.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "use-list-order"

namespace {
class VerifyUseListOrder : public ModulePass {
public:
  static char ID;
  VerifyUseListOrder();
  bool runOnModule(Module &M) override;
};
} // end anonymous namespace

char VerifyUseListOrder::ID = 0;
INITIALIZE_PASS(VerifyUseListOrder, "verify-use-list-order",
                "Verify Use List Order", false, false)
VerifyUseListOrder::VerifyUseListOrder() : ModulePass(ID) {
  initializeVerifyUseListOrderPass(*PassRegistry::getPassRegistry());
}

bool VerifyUseListOrder::runOnModule(Module &M) {
  DEBUG(dbgs() << "*** verify-use-list-order ***\n");
  if (!shouldPreserveBitcodeUseListOrder()) {
    // Can't verify if order isn't preserved.
    DEBUG(dbgs() << "warning: cannot verify bitcode; "
                    "try -preserve-bc-use-list-order\n");
    return false;
  }

  shuffleUseLists(M);
  if (!verifyBitcodeUseListOrder(M))
    report_fatal_error("bitcode use-list order changed");

  return true;
}

ModulePass *llvm::createVerifyUseListOrderPass() {
  return new VerifyUseListOrder;
}
