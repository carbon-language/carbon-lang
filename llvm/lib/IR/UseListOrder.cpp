//===- UseListOrder.cpp - Implement Use List Order functions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement use list order functions to modify use-list order and verify it
// doesn't change after serialization.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/UseListOrder.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <random>
#include <vector>

#define DEBUG_TYPE "use-list-order"

using namespace llvm;

static cl::opt<bool> PreserveBitcodeUseListOrder(
    "preserve-bc-use-list-order",
    cl::desc("Experimental support to preserve bitcode use-list order."),
    cl::init(false), cl::Hidden);

static cl::opt<bool> PreserveAssemblyUseListOrder(
    "preserve-ll-use-list-order",
    cl::desc("Experimental support to preserve assembly use-list order."),
    cl::init(false), cl::Hidden);

bool llvm::shouldPreserveBitcodeUseListOrder() {
  return PreserveBitcodeUseListOrder;
}

bool llvm::shouldPreserveAssemblyUseListOrder() {
  return PreserveAssemblyUseListOrder;
}

static void shuffleValueUseLists(Value *V, std::minstd_rand0 &Gen,
                                 DenseSet<Value *> &Seen) {
  if (!Seen.insert(V).second)
    return;

  if (auto *C = dyn_cast<Constant>(V))
    if (!isa<GlobalValue>(C))
      for (Value *Op : C->operands())
        shuffleValueUseLists(Op, Gen, Seen);

  if (V->use_empty() || std::next(V->use_begin()) == V->use_end())
    // Nothing to shuffle for 0 or 1 users.
    return;

  // Generate random numbers between 10 and 99, which will line up nicely in
  // debug output.  We're not worried about collisons here.
  DEBUG(dbgs() << "V = "; V->dump());
  std::uniform_int_distribution<short> Dist(10, 99);
  SmallDenseMap<const Use *, short, 16> Order;
  for (const Use &U : V->uses()) {
    auto I = Dist(Gen);
    Order[&U] = I;
    DEBUG(dbgs() << " - order: " << I << ", op = " << U.getOperandNo()
                 << ", U = ";
          U.getUser()->dump());
  }

  DEBUG(dbgs() << " => shuffle\n");
  V->sortUseList(
      [&Order](const Use &L, const Use &R) { return Order[&L] < Order[&R]; });

  DEBUG({
    for (const Use &U : V->uses()) {
      dbgs() << " - order: " << Order.lookup(&U)
             << ", op = " << U.getOperandNo() << ", U = ";
      U.getUser()->dump();
    }
  });
}

void llvm::shuffleUseLists(Module &M, unsigned SeedOffset) {
  DEBUG(dbgs() << "*** shuffle-use-lists ***\n");
  std::minstd_rand0 Gen(std::minstd_rand0::default_seed + SeedOffset);
  DenseSet<Value *> Seen;

  // Shuffle the use-list of each value that would be serialized to an IR file
  // (bitcode or assembly).
  auto shuffle = [&](Value *V) { shuffleValueUseLists(V, Gen, Seen); };

  // Globals.
  for (GlobalVariable &G : M.globals())
    shuffle(&G);
  for (GlobalAlias &A : M.aliases())
    shuffle(&A);
  for (Function &F : M)
    shuffle(&F);

  // Constants used by globals.
  for (GlobalVariable &G : M.globals())
    if (G.hasInitializer())
      shuffle(G.getInitializer());
  for (GlobalAlias &A : M.aliases())
    shuffle(A.getAliasee());
  for (Function &F : M)
    if (F.hasPrefixData())
      shuffle(F.getPrefixData());

  // Function bodies.
  for (Function &F : M) {
    for (Argument &A : F.args())
      shuffle(&A);
    for (BasicBlock &BB : F)
      shuffle(&BB);
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        shuffle(&I);

    // Constants used by instructions.
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        for (Value *Op : I.operands())
          if ((isa<Constant>(Op) && !isa<GlobalValue>(*Op)) ||
              isa<InlineAsm>(Op))
            shuffle(Op);
  }

  DEBUG(dbgs() << "\n");
}
