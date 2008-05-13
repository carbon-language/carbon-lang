//===- MachineDominators.cpp - Machine Dominator Calculation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements simple dominator construction algorithms for finding
// forward dominators on machine functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/Passes.h"

using namespace llvm;

TEMPLATE_INSTANTIATION(class DomTreeNodeBase<MachineBasicBlock>);
TEMPLATE_INSTANTIATION(class DominatorTreeBase<MachineBasicBlock>);

char MachineDominatorTree::ID = 0;

static RegisterPass<MachineDominatorTree>
E("machinedomtree", "MachineDominator Tree Construction", true);

const PassInfo *llvm::MachineDominatorsID = E.getPassInfo();
