//===-- AMDGPULowerKernelAttributes.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass does attempts to make use of reqd_work_group_size metadata
/// to eliminate loads from the dispatch packet and to constant fold OpenCL
/// get_local_size-like functions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "amdgpu-lower-kernel-attributes"

using namespace llvm;

namespace {

// Field offsets in hsa_kernel_dispatch_packet_t.
enum DispatchPackedOffsets {
  WORKGROUP_SIZE_X = 4,
  WORKGROUP_SIZE_Y = 6,
  WORKGROUP_SIZE_Z = 8,

  GRID_SIZE_X = 12,
  GRID_SIZE_Y = 16,
  GRID_SIZE_Z = 20
};

class AMDGPULowerKernelAttributes : public ModulePass {
public:
  static char ID;

  AMDGPULowerKernelAttributes() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override {
    return "AMDGPU Kernel Attributes";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
 }
};

} // end anonymous namespace

static bool processUse(CallInst *CI) {
  Function *F = CI->getParent()->getParent();

  auto MD = F->getMetadata("reqd_work_group_size");
  const bool HasReqdWorkGroupSize = MD && MD->getNumOperands() == 3;

  const bool HasUniformWorkGroupSize =
    F->getFnAttribute("uniform-work-group-size").getValueAsBool();

  if (!HasReqdWorkGroupSize && !HasUniformWorkGroupSize)
    return false;

  Value *WorkGroupSizeX = nullptr;
  Value *WorkGroupSizeY = nullptr;
  Value *WorkGroupSizeZ = nullptr;

  Value *GridSizeX = nullptr;
  Value *GridSizeY = nullptr;
  Value *GridSizeZ = nullptr;

  const DataLayout &DL = F->getParent()->getDataLayout();

  // We expect to see several GEP users, casted to the appropriate type and
  // loaded.
  for (User *U : CI->users()) {
    if (!U->hasOneUse())
      continue;

    int64_t Offset = 0;
    if (GetPointerBaseWithConstantOffset(U, Offset, DL) != CI)
      continue;

    auto *BCI = dyn_cast<BitCastInst>(*U->user_begin());
    if (!BCI || !BCI->hasOneUse())
      continue;

    auto *Load = dyn_cast<LoadInst>(*BCI->user_begin());
    if (!Load || !Load->isSimple())
      continue;

    unsigned LoadSize = DL.getTypeStoreSize(Load->getType());

    // TODO: Handle merged loads.
    switch (Offset) {
    case WORKGROUP_SIZE_X:
      if (LoadSize == 2)
        WorkGroupSizeX = Load;
      break;
    case WORKGROUP_SIZE_Y:
      if (LoadSize == 2)
        WorkGroupSizeY = Load;
      break;
    case WORKGROUP_SIZE_Z:
      if (LoadSize == 2)
        WorkGroupSizeZ = Load;
      break;
    case GRID_SIZE_X:
      if (LoadSize == 4)
        GridSizeX = Load;
      break;
    case GRID_SIZE_Y:
      if (LoadSize == 4)
        GridSizeY = Load;
      break;
    case GRID_SIZE_Z:
      if (LoadSize == 4)
        GridSizeZ = Load;
      break;
    default:
      break;
    }
  }

  // Pattern match the code used to handle partial workgroup dispatches in the
  // library implementation of get_local_size, so the entire function can be
  // constant folded with a known group size.
  //
  // uint r = grid_size - group_id * group_size;
  // get_local_size = (r < group_size) ? r : group_size;
  //
  // If we have uniform-work-group-size (which is the default in OpenCL 1.2),
  // the grid_size is required to be a multiple of group_size). In this case:
  //
  // grid_size - (group_id * group_size) < group_size
  // ->
  // grid_size < group_size + (group_id * group_size)
  //
  // (grid_size / group_size) < 1 + group_id
  //
  // grid_size / group_size is at least 1, so we can conclude the select
  // condition is false (except for group_id == 0, where the select result is
  // the same).

  bool MadeChange = false;
  Value *WorkGroupSizes[3] = { WorkGroupSizeX, WorkGroupSizeY, WorkGroupSizeZ };
  Value *GridSizes[3] = { GridSizeX, GridSizeY, GridSizeZ };

  for (int I = 0; HasUniformWorkGroupSize && I < 3; ++I) {
    Value *GroupSize = WorkGroupSizes[I];
    Value *GridSize = GridSizes[I];
    if (!GroupSize || !GridSize)
      continue;

    using namespace llvm::PatternMatch;
    auto GroupIDIntrin =
        I == 0 ? m_Intrinsic<Intrinsic::amdgcn_workgroup_id_x>()
               : (I == 1 ? m_Intrinsic<Intrinsic::amdgcn_workgroup_id_y>()
                         : m_Intrinsic<Intrinsic::amdgcn_workgroup_id_z>());

    for (User *U : GroupSize->users()) {
      auto *ZextGroupSize = dyn_cast<ZExtInst>(U);
      if (!ZextGroupSize)
        continue;

      for (User *UMin : ZextGroupSize->users()) {
        if (match(UMin,
                  m_UMin(m_Sub(m_Specific(GridSize),
                               m_Mul(GroupIDIntrin, m_Specific(ZextGroupSize))),
                         m_Specific(ZextGroupSize)))) {
          if (HasReqdWorkGroupSize) {
            ConstantInt *KnownSize
              = mdconst::extract<ConstantInt>(MD->getOperand(I));
            UMin->replaceAllUsesWith(ConstantExpr::getIntegerCast(
                KnownSize, UMin->getType(), false));
          } else {
            UMin->replaceAllUsesWith(ZextGroupSize);
          }

          MadeChange = true;
        }
      }
    }
  }

  if (!HasReqdWorkGroupSize)
    return MadeChange;

  // Eliminate any other loads we can from the dispatch packet.
  for (int I = 0; I < 3; ++I) {
    Value *GroupSize = WorkGroupSizes[I];
    if (!GroupSize)
      continue;

    ConstantInt *KnownSize = mdconst::extract<ConstantInt>(MD->getOperand(I));
    GroupSize->replaceAllUsesWith(
      ConstantExpr::getIntegerCast(KnownSize,
                                   GroupSize->getType(),
                                   false));
    MadeChange = true;
  }

  return MadeChange;
}

// TODO: Move makeLIDRangeMetadata usage into here. Seem to not get
// TargetPassConfig for subtarget.
bool AMDGPULowerKernelAttributes::runOnModule(Module &M) {
  StringRef DispatchPtrName
    = Intrinsic::getName(Intrinsic::amdgcn_dispatch_ptr);

  Function *DispatchPtr = M.getFunction(DispatchPtrName);
  if (!DispatchPtr) // Dispatch ptr not used.
    return false;

  bool MadeChange = false;

  SmallPtrSet<Instruction *, 4> HandledUses;
  for (auto *U : DispatchPtr->users()) {
    CallInst *CI = cast<CallInst>(U);
    if (HandledUses.insert(CI).second) {
      if (processUse(CI))
        MadeChange = true;
    }
  }

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(AMDGPULowerKernelAttributes, DEBUG_TYPE,
                      "AMDGPU Kernel Attributes", false, false)
INITIALIZE_PASS_END(AMDGPULowerKernelAttributes, DEBUG_TYPE,
                    "AMDGPU Kernel Attributes", false, false)

char AMDGPULowerKernelAttributes::ID = 0;

ModulePass *llvm::createAMDGPULowerKernelAttributesPass() {
  return new AMDGPULowerKernelAttributes();
}

PreservedAnalyses
AMDGPULowerKernelAttributesPass::run(Function &F, FunctionAnalysisManager &AM) {
  StringRef DispatchPtrName =
      Intrinsic::getName(Intrinsic::amdgcn_dispatch_ptr);

  Function *DispatchPtr = F.getParent()->getFunction(DispatchPtrName);
  if (!DispatchPtr) // Dispatch ptr not used.
    return PreservedAnalyses::all();

  for (Instruction &I : instructions(F)) {
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      if (CI->getCalledFunction() == DispatchPtr)
        processUse(CI);
    }
  }

  return PreservedAnalyses::all();
}
