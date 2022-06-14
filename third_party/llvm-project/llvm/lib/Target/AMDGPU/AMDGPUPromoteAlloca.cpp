//===-- AMDGPUPromoteAlloca.cpp - Promote Allocas -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates allocas by either converting them into vectors or
// by migrating them to local address space.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-promote-alloca"

using namespace llvm;

namespace {

static cl::opt<bool> DisablePromoteAllocaToVector(
  "disable-promote-alloca-to-vector",
  cl::desc("Disable promote alloca to vector"),
  cl::init(false));

static cl::opt<bool> DisablePromoteAllocaToLDS(
  "disable-promote-alloca-to-lds",
  cl::desc("Disable promote alloca to LDS"),
  cl::init(false));

static cl::opt<unsigned> PromoteAllocaToVectorLimit(
  "amdgpu-promote-alloca-to-vector-limit",
  cl::desc("Maximum byte size to consider promote alloca to vector"),
  cl::init(0));

// FIXME: This can create globals so should be a module pass.
class AMDGPUPromoteAlloca : public FunctionPass {
public:
  static char ID;

  AMDGPUPromoteAlloca() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "AMDGPU Promote Alloca"; }

  bool handleAlloca(AllocaInst &I, bool SufficientLDS);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    FunctionPass::getAnalysisUsage(AU);
  }
};

class AMDGPUPromoteAllocaImpl {
private:
  const TargetMachine &TM;
  Module *Mod = nullptr;
  const DataLayout *DL = nullptr;

  // FIXME: This should be per-kernel.
  uint32_t LocalMemLimit = 0;
  uint32_t CurrentLocalMemUsage = 0;
  unsigned MaxVGPRs;

  bool IsAMDGCN = false;
  bool IsAMDHSA = false;

  std::pair<Value *, Value *> getLocalSizeYZ(IRBuilder<> &Builder);
  Value *getWorkitemID(IRBuilder<> &Builder, unsigned N);

  /// BaseAlloca is the alloca root the search started from.
  /// Val may be that alloca or a recursive user of it.
  bool collectUsesWithPtrTypes(Value *BaseAlloca,
                               Value *Val,
                               std::vector<Value*> &WorkList) const;

  /// Val is a derived pointer from Alloca. OpIdx0/OpIdx1 are the operand
  /// indices to an instruction with 2 pointer inputs (e.g. select, icmp).
  /// Returns true if both operands are derived from the same alloca. Val should
  /// be the same value as one of the input operands of UseInst.
  bool binaryOpIsDerivedFromSameAlloca(Value *Alloca, Value *Val,
                                       Instruction *UseInst,
                                       int OpIdx0, int OpIdx1) const;

  /// Check whether we have enough local memory for promotion.
  bool hasSufficientLocalMem(const Function &F);

  bool handleAlloca(AllocaInst &I, bool SufficientLDS);

public:
  AMDGPUPromoteAllocaImpl(TargetMachine &TM) : TM(TM) {}
  bool run(Function &F);
};

class AMDGPUPromoteAllocaToVector : public FunctionPass {
public:
  static char ID;

  AMDGPUPromoteAllocaToVector() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override {
    return "AMDGPU Promote Alloca to vector";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    FunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char AMDGPUPromoteAlloca::ID = 0;
char AMDGPUPromoteAllocaToVector::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUPromoteAlloca, DEBUG_TYPE,
                      "AMDGPU promote alloca to vector or LDS", false, false)
// Move LDS uses from functions to kernels before promote alloca for accurate
// estimation of LDS available
INITIALIZE_PASS_DEPENDENCY(AMDGPULowerModuleLDS)
INITIALIZE_PASS_END(AMDGPUPromoteAlloca, DEBUG_TYPE,
                    "AMDGPU promote alloca to vector or LDS", false, false)

INITIALIZE_PASS(AMDGPUPromoteAllocaToVector, DEBUG_TYPE "-to-vector",
                "AMDGPU promote alloca to vector", false, false)

char &llvm::AMDGPUPromoteAllocaID = AMDGPUPromoteAlloca::ID;
char &llvm::AMDGPUPromoteAllocaToVectorID = AMDGPUPromoteAllocaToVector::ID;

bool AMDGPUPromoteAlloca::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  if (auto *TPC = getAnalysisIfAvailable<TargetPassConfig>()) {
    return AMDGPUPromoteAllocaImpl(TPC->getTM<TargetMachine>()).run(F);
  }
  return false;
}

PreservedAnalyses AMDGPUPromoteAllocaPass::run(Function &F,
                                               FunctionAnalysisManager &AM) {
  bool Changed = AMDGPUPromoteAllocaImpl(TM).run(F);
  if (Changed) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }
  return PreservedAnalyses::all();
}

bool AMDGPUPromoteAllocaImpl::run(Function &F) {
  Mod = F.getParent();
  DL = &Mod->getDataLayout();

  const Triple &TT = TM.getTargetTriple();
  IsAMDGCN = TT.getArch() == Triple::amdgcn;
  IsAMDHSA = TT.getOS() == Triple::AMDHSA;

  const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, F);
  if (!ST.isPromoteAllocaEnabled())
    return false;

  if (IsAMDGCN) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    MaxVGPRs = ST.getMaxNumVGPRs(ST.getWavesPerEU(F).first);
    // A non-entry function has only 32 caller preserved registers.
    // Do not promote alloca which will force spilling.
    if (!AMDGPU::isEntryFunctionCC(F.getCallingConv()))
      MaxVGPRs = std::min(MaxVGPRs, 32u);
  } else {
    MaxVGPRs = 128;
  }

  bool SufficientLDS = hasSufficientLocalMem(F);
  bool Changed = false;
  BasicBlock &EntryBB = *F.begin();

  SmallVector<AllocaInst *, 16> Allocas;
  for (Instruction &I : EntryBB) {
    if (AllocaInst *AI = dyn_cast<AllocaInst>(&I))
      Allocas.push_back(AI);
  }

  for (AllocaInst *AI : Allocas) {
    if (handleAlloca(*AI, SufficientLDS))
      Changed = true;
  }

  return Changed;
}

std::pair<Value *, Value *>
AMDGPUPromoteAllocaImpl::getLocalSizeYZ(IRBuilder<> &Builder) {
  Function &F = *Builder.GetInsertBlock()->getParent();
  const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, F);

  if (!IsAMDHSA) {
    Function *LocalSizeYFn
      = Intrinsic::getDeclaration(Mod, Intrinsic::r600_read_local_size_y);
    Function *LocalSizeZFn
      = Intrinsic::getDeclaration(Mod, Intrinsic::r600_read_local_size_z);

    CallInst *LocalSizeY = Builder.CreateCall(LocalSizeYFn, {});
    CallInst *LocalSizeZ = Builder.CreateCall(LocalSizeZFn, {});

    ST.makeLIDRangeMetadata(LocalSizeY);
    ST.makeLIDRangeMetadata(LocalSizeZ);

    return std::make_pair(LocalSizeY, LocalSizeZ);
  }

  // We must read the size out of the dispatch pointer.
  assert(IsAMDGCN);

  // We are indexing into this struct, and want to extract the workgroup_size_*
  // fields.
  //
  //   typedef struct hsa_kernel_dispatch_packet_s {
  //     uint16_t header;
  //     uint16_t setup;
  //     uint16_t workgroup_size_x ;
  //     uint16_t workgroup_size_y;
  //     uint16_t workgroup_size_z;
  //     uint16_t reserved0;
  //     uint32_t grid_size_x ;
  //     uint32_t grid_size_y ;
  //     uint32_t grid_size_z;
  //
  //     uint32_t private_segment_size;
  //     uint32_t group_segment_size;
  //     uint64_t kernel_object;
  //
  // #ifdef HSA_LARGE_MODEL
  //     void *kernarg_address;
  // #elif defined HSA_LITTLE_ENDIAN
  //     void *kernarg_address;
  //     uint32_t reserved1;
  // #else
  //     uint32_t reserved1;
  //     void *kernarg_address;
  // #endif
  //     uint64_t reserved2;
  //     hsa_signal_t completion_signal; // uint64_t wrapper
  //   } hsa_kernel_dispatch_packet_t
  //
  Function *DispatchPtrFn
    = Intrinsic::getDeclaration(Mod, Intrinsic::amdgcn_dispatch_ptr);

  CallInst *DispatchPtr = Builder.CreateCall(DispatchPtrFn, {});
  DispatchPtr->addRetAttr(Attribute::NoAlias);
  DispatchPtr->addRetAttr(Attribute::NonNull);
  F.removeFnAttr("amdgpu-no-dispatch-ptr");

  // Size of the dispatch packet struct.
  DispatchPtr->addDereferenceableRetAttr(64);

  Type *I32Ty = Type::getInt32Ty(Mod->getContext());
  Value *CastDispatchPtr = Builder.CreateBitCast(
    DispatchPtr, PointerType::get(I32Ty, AMDGPUAS::CONSTANT_ADDRESS));

  // We could do a single 64-bit load here, but it's likely that the basic
  // 32-bit and extract sequence is already present, and it is probably easier
  // to CSE this. The loads should be mergeable later anyway.
  Value *GEPXY = Builder.CreateConstInBoundsGEP1_64(I32Ty, CastDispatchPtr, 1);
  LoadInst *LoadXY = Builder.CreateAlignedLoad(I32Ty, GEPXY, Align(4));

  Value *GEPZU = Builder.CreateConstInBoundsGEP1_64(I32Ty, CastDispatchPtr, 2);
  LoadInst *LoadZU = Builder.CreateAlignedLoad(I32Ty, GEPZU, Align(4));

  MDNode *MD = MDNode::get(Mod->getContext(), None);
  LoadXY->setMetadata(LLVMContext::MD_invariant_load, MD);
  LoadZU->setMetadata(LLVMContext::MD_invariant_load, MD);
  ST.makeLIDRangeMetadata(LoadZU);

  // Extract y component. Upper half of LoadZU should be zero already.
  Value *Y = Builder.CreateLShr(LoadXY, 16);

  return std::make_pair(Y, LoadZU);
}

Value *AMDGPUPromoteAllocaImpl::getWorkitemID(IRBuilder<> &Builder,
                                              unsigned N) {
  Function *F = Builder.GetInsertBlock()->getParent();
  const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, *F);
  Intrinsic::ID IntrID = Intrinsic::not_intrinsic;
  StringRef AttrName;

  switch (N) {
  case 0:
    IntrID = IsAMDGCN ? (Intrinsic::ID)Intrinsic::amdgcn_workitem_id_x
                      : (Intrinsic::ID)Intrinsic::r600_read_tidig_x;
    AttrName = "amdgpu-no-workitem-id-x";
    break;
  case 1:
    IntrID = IsAMDGCN ? (Intrinsic::ID)Intrinsic::amdgcn_workitem_id_y
                      : (Intrinsic::ID)Intrinsic::r600_read_tidig_y;
    AttrName = "amdgpu-no-workitem-id-y";
    break;

  case 2:
    IntrID = IsAMDGCN ? (Intrinsic::ID)Intrinsic::amdgcn_workitem_id_z
                      : (Intrinsic::ID)Intrinsic::r600_read_tidig_z;
    AttrName = "amdgpu-no-workitem-id-z";
    break;
  default:
    llvm_unreachable("invalid dimension");
  }

  Function *WorkitemIdFn = Intrinsic::getDeclaration(Mod, IntrID);
  CallInst *CI = Builder.CreateCall(WorkitemIdFn);
  ST.makeLIDRangeMetadata(CI);
  F->removeFnAttr(AttrName);

  return CI;
}

static FixedVectorType *arrayTypeToVecType(ArrayType *ArrayTy) {
  return FixedVectorType::get(ArrayTy->getElementType(),
                              ArrayTy->getNumElements());
}

static Value *
calculateVectorIndex(Value *Ptr,
                     const std::map<GetElementPtrInst *, Value *> &GEPIdx) {
  auto *GEP = dyn_cast<GetElementPtrInst>(Ptr->stripPointerCasts());
  if (!GEP)
    return ConstantInt::getNullValue(Type::getInt32Ty(Ptr->getContext()));

  auto I = GEPIdx.find(GEP);
  assert(I != GEPIdx.end() && "Must have entry for GEP!");
  return I->second;
}

static Value *GEPToVectorIndex(GetElementPtrInst *GEP, AllocaInst *Alloca,
                               Type *VecElemTy, const DataLayout &DL) {
  // TODO: Extracting a "multiple of X" from a GEP might be a useful generic
  // helper.
  unsigned BW = DL.getIndexTypeSizeInBits(GEP->getType());
  MapVector<Value *, APInt> VarOffsets;
  APInt ConstOffset(BW, 0);
  if (GEP->getPointerOperand()->stripPointerCasts() != Alloca ||
      !GEP->collectOffset(DL, BW, VarOffsets, ConstOffset))
    return nullptr;

  unsigned VecElemSize = DL.getTypeAllocSize(VecElemTy);
  if (VarOffsets.size() > 1)
    return nullptr;

  if (VarOffsets.size() == 1) {
    // Only handle cases where we don't need to insert extra arithmetic
    // instructions.
    const auto &VarOffset = VarOffsets.front();
    if (!ConstOffset.isZero() || VarOffset.second != VecElemSize)
      return nullptr;
    return VarOffset.first;
  }

  APInt Quot;
  uint64_t Rem;
  APInt::udivrem(ConstOffset, VecElemSize, Quot, Rem);
  if (Rem != 0)
    return nullptr;

  return ConstantInt::get(GEP->getContext(), Quot);
}

static bool tryPromoteAllocaToVector(AllocaInst *Alloca, const DataLayout &DL,
                                     unsigned MaxVGPRs) {

  if (DisablePromoteAllocaToVector) {
    LLVM_DEBUG(dbgs() << "  Promotion alloca to vector is disabled\n");
    return false;
  }

  Type *AllocaTy = Alloca->getAllocatedType();
  auto *VectorTy = dyn_cast<FixedVectorType>(AllocaTy);
  if (auto *ArrayTy = dyn_cast<ArrayType>(AllocaTy)) {
    if (VectorType::isValidElementType(ArrayTy->getElementType()) &&
        ArrayTy->getNumElements() > 0)
      VectorTy = arrayTypeToVecType(ArrayTy);
  }

  // Use up to 1/4 of available register budget for vectorization.
  unsigned Limit = PromoteAllocaToVectorLimit ? PromoteAllocaToVectorLimit * 8
                                              : (MaxVGPRs * 32);

  if (DL.getTypeSizeInBits(AllocaTy) * 4 > Limit) {
    LLVM_DEBUG(dbgs() << "  Alloca too big for vectorization with "
                      << MaxVGPRs << " registers available\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Alloca candidate for vectorization\n");

  // FIXME: There is no reason why we can't support larger arrays, we
  // are just being conservative for now.
  // FIXME: We also reject alloca's of the form [ 2 x [ 2 x i32 ]] or equivalent. Potentially these
  // could also be promoted but we don't currently handle this case
  if (!VectorTy || VectorTy->getNumElements() > 16 ||
      VectorTy->getNumElements() < 2) {
    LLVM_DEBUG(dbgs() << "  Cannot convert type to vector\n");
    return false;
  }

  std::map<GetElementPtrInst*, Value*> GEPVectorIdx;
  SmallVector<Instruction *> WorkList;
  SmallVector<Use *, 8> Uses;
  for (Use &U : Alloca->uses())
    Uses.push_back(&U);

  Type *VecEltTy = VectorTy->getElementType();
  while (!Uses.empty()) {
    Use *U = Uses.pop_back_val();
    Instruction *Inst = dyn_cast<Instruction>(U->getUser());

    if (Value *Ptr = getLoadStorePointerOperand(Inst)) {
      // This is a store of the pointer, not to the pointer.
      if (isa<StoreInst>(Inst) &&
          U->getOperandNo() != StoreInst::getPointerOperandIndex())
        return false;

      Type *AccessTy = getLoadStoreType(Inst);
      Ptr = Ptr->stripPointerCasts();

      // Alloca already accessed as vector, leave alone.
      if (Ptr == Alloca && DL.getTypeStoreSize(Alloca->getAllocatedType()) ==
                               DL.getTypeStoreSize(AccessTy))
        continue;

      // Check that this is a simple access of a vector element.
      bool IsSimple = isa<LoadInst>(Inst) ? cast<LoadInst>(Inst)->isSimple()
                                          : cast<StoreInst>(Inst)->isSimple();
      if (!IsSimple ||
          !CastInst::isBitOrNoopPointerCastable(VecEltTy, AccessTy, DL))
        return false;

      WorkList.push_back(Inst);
      continue;
    }

    if (isa<BitCastInst>(Inst)) {
      // Look through bitcasts.
      for (Use &U : Inst->uses())
        Uses.push_back(&U);
      continue;
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
      // If we can't compute a vector index from this GEP, then we can't
      // promote this alloca to vector.
      Value *Index = GEPToVectorIndex(GEP, Alloca, VecEltTy, DL);
      if (!Index) {
        LLVM_DEBUG(dbgs() << "  Cannot compute vector index for GEP " << *GEP
                          << '\n');
        return false;
      }

      GEPVectorIdx[GEP] = Index;
      for (Use &U : Inst->uses())
        Uses.push_back(&U);
      continue;
    }

    // Ignore assume-like intrinsics and comparisons used in assumes.
    if (isAssumeLikeIntrinsic(Inst))
      continue;

    if (isa<ICmpInst>(Inst) && all_of(Inst->users(), [](User *U) {
          return isAssumeLikeIntrinsic(cast<Instruction>(U));
        }))
      continue;

    // Unknown user.
    return false;
  }

  LLVM_DEBUG(dbgs() << "  Converting alloca to vector " << *AllocaTy << " -> "
                    << *VectorTy << '\n');

  for (Instruction *Inst : WorkList) {
    IRBuilder<> Builder(Inst);
    switch (Inst->getOpcode()) {
    case Instruction::Load: {
      Value *Ptr = cast<LoadInst>(Inst)->getPointerOperand();
      Value *Index = calculateVectorIndex(Ptr, GEPVectorIdx);
      Type *VecPtrTy = VectorTy->getPointerTo(Alloca->getAddressSpace());
      Value *BitCast = Builder.CreateBitCast(Alloca, VecPtrTy);
      Value *VecValue = Builder.CreateLoad(VectorTy, BitCast);
      Value *ExtractElement = Builder.CreateExtractElement(VecValue, Index);
      if (Inst->getType() != VecEltTy)
        ExtractElement = Builder.CreateBitOrPointerCast(ExtractElement, Inst->getType());
      Inst->replaceAllUsesWith(ExtractElement);
      Inst->eraseFromParent();
      break;
    }
    case Instruction::Store: {
      StoreInst *SI = cast<StoreInst>(Inst);
      Value *Ptr = SI->getPointerOperand();
      Value *Index = calculateVectorIndex(Ptr, GEPVectorIdx);
      Type *VecPtrTy = VectorTy->getPointerTo(Alloca->getAddressSpace());
      Value *BitCast = Builder.CreateBitCast(Alloca, VecPtrTy);
      Value *VecValue = Builder.CreateLoad(VectorTy, BitCast);
      Value *Elt = SI->getValueOperand();
      if (Elt->getType() != VecEltTy)
        Elt = Builder.CreateBitOrPointerCast(Elt, VecEltTy);
      Value *NewVecValue = Builder.CreateInsertElement(VecValue, Elt, Index);
      Builder.CreateStore(NewVecValue, BitCast);
      Inst->eraseFromParent();
      break;
    }

    default:
      llvm_unreachable("Inconsistency in instructions promotable to vector");
    }
  }
  return true;
}

static bool isCallPromotable(CallInst *CI) {
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI);
  if (!II)
    return false;

  switch (II->getIntrinsicID()) {
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset:
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::invariant_start:
  case Intrinsic::invariant_end:
  case Intrinsic::launder_invariant_group:
  case Intrinsic::strip_invariant_group:
  case Intrinsic::objectsize:
    return true;
  default:
    return false;
  }
}

bool AMDGPUPromoteAllocaImpl::binaryOpIsDerivedFromSameAlloca(
    Value *BaseAlloca, Value *Val, Instruction *Inst, int OpIdx0,
    int OpIdx1) const {
  // Figure out which operand is the one we might not be promoting.
  Value *OtherOp = Inst->getOperand(OpIdx0);
  if (Val == OtherOp)
    OtherOp = Inst->getOperand(OpIdx1);

  if (isa<ConstantPointerNull>(OtherOp))
    return true;

  Value *OtherObj = getUnderlyingObject(OtherOp);
  if (!isa<AllocaInst>(OtherObj))
    return false;

  // TODO: We should be able to replace undefs with the right pointer type.

  // TODO: If we know the other base object is another promotable
  // alloca, not necessarily this alloca, we can do this. The
  // important part is both must have the same address space at
  // the end.
  if (OtherObj != BaseAlloca) {
    LLVM_DEBUG(
        dbgs() << "Found a binary instruction with another alloca object\n");
    return false;
  }

  return true;
}

bool AMDGPUPromoteAllocaImpl::collectUsesWithPtrTypes(
    Value *BaseAlloca, Value *Val, std::vector<Value *> &WorkList) const {

  for (User *User : Val->users()) {
    if (is_contained(WorkList, User))
      continue;

    if (CallInst *CI = dyn_cast<CallInst>(User)) {
      if (!isCallPromotable(CI))
        return false;

      WorkList.push_back(User);
      continue;
    }

    Instruction *UseInst = cast<Instruction>(User);
    if (UseInst->getOpcode() == Instruction::PtrToInt)
      return false;

    if (LoadInst *LI = dyn_cast<LoadInst>(UseInst)) {
      if (LI->isVolatile())
        return false;

      continue;
    }

    if (StoreInst *SI = dyn_cast<StoreInst>(UseInst)) {
      if (SI->isVolatile())
        return false;

      // Reject if the stored value is not the pointer operand.
      if (SI->getPointerOperand() != Val)
        return false;
    } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(UseInst)) {
      if (RMW->isVolatile())
        return false;
    } else if (AtomicCmpXchgInst *CAS = dyn_cast<AtomicCmpXchgInst>(UseInst)) {
      if (CAS->isVolatile())
        return false;
    }

    // Only promote a select if we know that the other select operand
    // is from another pointer that will also be promoted.
    if (ICmpInst *ICmp = dyn_cast<ICmpInst>(UseInst)) {
      if (!binaryOpIsDerivedFromSameAlloca(BaseAlloca, Val, ICmp, 0, 1))
        return false;

      // May need to rewrite constant operands.
      WorkList.push_back(ICmp);
    }

    if (UseInst->getOpcode() == Instruction::AddrSpaceCast) {
      // Give up if the pointer may be captured.
      if (PointerMayBeCaptured(UseInst, true, true))
        return false;
      // Don't collect the users of this.
      WorkList.push_back(User);
      continue;
    }

    // Do not promote vector/aggregate type instructions. It is hard to track
    // their users.
    if (isa<InsertValueInst>(User) || isa<InsertElementInst>(User))
      return false;

    if (!User->getType()->isPointerTy())
      continue;

    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(UseInst)) {
      // Be conservative if an address could be computed outside the bounds of
      // the alloca.
      if (!GEP->isInBounds())
        return false;
    }

    // Only promote a select if we know that the other select operand is from
    // another pointer that will also be promoted.
    if (SelectInst *SI = dyn_cast<SelectInst>(UseInst)) {
      if (!binaryOpIsDerivedFromSameAlloca(BaseAlloca, Val, SI, 1, 2))
        return false;
    }

    // Repeat for phis.
    if (PHINode *Phi = dyn_cast<PHINode>(UseInst)) {
      // TODO: Handle more complex cases. We should be able to replace loops
      // over arrays.
      switch (Phi->getNumIncomingValues()) {
      case 1:
        break;
      case 2:
        if (!binaryOpIsDerivedFromSameAlloca(BaseAlloca, Val, Phi, 0, 1))
          return false;
        break;
      default:
        return false;
      }
    }

    WorkList.push_back(User);
    if (!collectUsesWithPtrTypes(BaseAlloca, User, WorkList))
      return false;
  }

  return true;
}

bool AMDGPUPromoteAllocaImpl::hasSufficientLocalMem(const Function &F) {

  FunctionType *FTy = F.getFunctionType();
  const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, F);

  // If the function has any arguments in the local address space, then it's
  // possible these arguments require the entire local memory space, so
  // we cannot use local memory in the pass.
  for (Type *ParamTy : FTy->params()) {
    PointerType *PtrTy = dyn_cast<PointerType>(ParamTy);
    if (PtrTy && PtrTy->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
      LocalMemLimit = 0;
      LLVM_DEBUG(dbgs() << "Function has local memory argument. Promoting to "
                           "local memory disabled.\n");
      return false;
    }
  }

  LocalMemLimit = ST.getLocalMemorySize();
  if (LocalMemLimit == 0)
    return false;

  SmallVector<const Constant *, 16> Stack;
  SmallPtrSet<const Constant *, 8> VisitedConstants;
  SmallPtrSet<const GlobalVariable *, 8> UsedLDS;

  auto visitUsers = [&](const GlobalVariable *GV, const Constant *Val) -> bool {
    for (const User *U : Val->users()) {
      if (const Instruction *Use = dyn_cast<Instruction>(U)) {
        if (Use->getParent()->getParent() == &F)
          return true;
      } else {
        const Constant *C = cast<Constant>(U);
        if (VisitedConstants.insert(C).second)
          Stack.push_back(C);
      }
    }

    return false;
  };

  for (GlobalVariable &GV : Mod->globals()) {
    if (GV.getAddressSpace() != AMDGPUAS::LOCAL_ADDRESS)
      continue;

    if (visitUsers(&GV, &GV)) {
      UsedLDS.insert(&GV);
      Stack.clear();
      continue;
    }

    // For any ConstantExpr uses, we need to recursively search the users until
    // we see a function.
    while (!Stack.empty()) {
      const Constant *C = Stack.pop_back_val();
      if (visitUsers(&GV, C)) {
        UsedLDS.insert(&GV);
        Stack.clear();
        break;
      }
    }
  }

  const DataLayout &DL = Mod->getDataLayout();
  SmallVector<std::pair<uint64_t, Align>, 16> AllocatedSizes;
  AllocatedSizes.reserve(UsedLDS.size());

  for (const GlobalVariable *GV : UsedLDS) {
    Align Alignment =
        DL.getValueOrABITypeAlignment(GV->getAlign(), GV->getValueType());
    uint64_t AllocSize = DL.getTypeAllocSize(GV->getValueType());

    // HIP uses an extern unsized array in local address space for dynamically
    // allocated shared memory.  In that case, we have to disable the promotion.
    if (GV->hasExternalLinkage() && AllocSize == 0) {
      LocalMemLimit = 0;
      LLVM_DEBUG(dbgs() << "Function has a reference to externally allocated "
                           "local memory. Promoting to local memory "
                           "disabled.\n");
      return false;
    }

    AllocatedSizes.emplace_back(AllocSize, Alignment);
  }

  // Sort to try to estimate the worst case alignment padding
  //
  // FIXME: We should really do something to fix the addresses to a more optimal
  // value instead
  llvm::sort(AllocatedSizes, llvm::less_second());

  // Check how much local memory is being used by global objects
  CurrentLocalMemUsage = 0;

  // FIXME: Try to account for padding here. The real padding and address is
  // currently determined from the inverse order of uses in the function when
  // legalizing, which could also potentially change. We try to estimate the
  // worst case here, but we probably should fix the addresses earlier.
  for (auto Alloc : AllocatedSizes) {
    CurrentLocalMemUsage = alignTo(CurrentLocalMemUsage, Alloc.second);
    CurrentLocalMemUsage += Alloc.first;
  }

  unsigned MaxOccupancy = ST.getOccupancyWithLocalMemSize(CurrentLocalMemUsage,
                                                          F);

  // Restrict local memory usage so that we don't drastically reduce occupancy,
  // unless it is already significantly reduced.

  // TODO: Have some sort of hint or other heuristics to guess occupancy based
  // on other factors..
  unsigned OccupancyHint = ST.getWavesPerEU(F).second;
  if (OccupancyHint == 0)
    OccupancyHint = 7;

  // Clamp to max value.
  OccupancyHint = std::min(OccupancyHint, ST.getMaxWavesPerEU());

  // Check the hint but ignore it if it's obviously wrong from the existing LDS
  // usage.
  MaxOccupancy = std::min(OccupancyHint, MaxOccupancy);


  // Round up to the next tier of usage.
  unsigned MaxSizeWithWaveCount
    = ST.getMaxLocalMemSizeWithWaveCount(MaxOccupancy, F);

  // Program is possibly broken by using more local mem than available.
  if (CurrentLocalMemUsage > MaxSizeWithWaveCount)
    return false;

  LocalMemLimit = MaxSizeWithWaveCount;

  LLVM_DEBUG(dbgs() << F.getName() << " uses " << CurrentLocalMemUsage
                    << " bytes of LDS\n"
                    << "  Rounding size to " << MaxSizeWithWaveCount
                    << " with a maximum occupancy of " << MaxOccupancy << '\n'
                    << " and " << (LocalMemLimit - CurrentLocalMemUsage)
                    << " available for promotion\n");

  return true;
}

// FIXME: Should try to pick the most likely to be profitable allocas first.
bool AMDGPUPromoteAllocaImpl::handleAlloca(AllocaInst &I, bool SufficientLDS) {
  // Array allocations are probably not worth handling, since an allocation of
  // the array type is the canonical form.
  if (!I.isStaticAlloca() || I.isArrayAllocation())
    return false;

  const DataLayout &DL = Mod->getDataLayout();
  IRBuilder<> Builder(&I);

  // First try to replace the alloca with a vector
  Type *AllocaTy = I.getAllocatedType();

  LLVM_DEBUG(dbgs() << "Trying to promote " << I << '\n');

  if (tryPromoteAllocaToVector(&I, DL, MaxVGPRs))
    return true; // Promoted to vector.

  if (DisablePromoteAllocaToLDS)
    return false;

  const Function &ContainingFunction = *I.getParent()->getParent();
  CallingConv::ID CC = ContainingFunction.getCallingConv();

  // Don't promote the alloca to LDS for shader calling conventions as the work
  // item ID intrinsics are not supported for these calling conventions.
  // Furthermore not all LDS is available for some of the stages.
  switch (CC) {
  case CallingConv::AMDGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
    break;
  default:
    LLVM_DEBUG(
        dbgs()
        << " promote alloca to LDS not supported with calling convention.\n");
    return false;
  }

  // Not likely to have sufficient local memory for promotion.
  if (!SufficientLDS)
    return false;

  const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, ContainingFunction);
  unsigned WorkGroupSize = ST.getFlatWorkGroupSizes(ContainingFunction).second;

  Align Alignment =
      DL.getValueOrABITypeAlignment(I.getAlign(), I.getAllocatedType());

  // FIXME: This computed padding is likely wrong since it depends on inverse
  // usage order.
  //
  // FIXME: It is also possible that if we're allowed to use all of the memory
  // could end up using more than the maximum due to alignment padding.

  uint32_t NewSize = alignTo(CurrentLocalMemUsage, Alignment);
  uint32_t AllocSize = WorkGroupSize * DL.getTypeAllocSize(AllocaTy);
  NewSize += AllocSize;

  if (NewSize > LocalMemLimit) {
    LLVM_DEBUG(dbgs() << "  " << AllocSize
                      << " bytes of local memory not available to promote\n");
    return false;
  }

  CurrentLocalMemUsage = NewSize;

  std::vector<Value*> WorkList;

  if (!collectUsesWithPtrTypes(&I, &I, WorkList)) {
    LLVM_DEBUG(dbgs() << " Do not know how to convert all uses\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Promoting alloca to local memory\n");

  Function *F = I.getParent()->getParent();

  Type *GVTy = ArrayType::get(I.getAllocatedType(), WorkGroupSize);
  GlobalVariable *GV = new GlobalVariable(
      *Mod, GVTy, false, GlobalValue::InternalLinkage,
      UndefValue::get(GVTy),
      Twine(F->getName()) + Twine('.') + I.getName(),
      nullptr,
      GlobalVariable::NotThreadLocal,
      AMDGPUAS::LOCAL_ADDRESS);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  GV->setAlignment(I.getAlign());

  Value *TCntY, *TCntZ;

  std::tie(TCntY, TCntZ) = getLocalSizeYZ(Builder);
  Value *TIdX = getWorkitemID(Builder, 0);
  Value *TIdY = getWorkitemID(Builder, 1);
  Value *TIdZ = getWorkitemID(Builder, 2);

  Value *Tmp0 = Builder.CreateMul(TCntY, TCntZ, "", true, true);
  Tmp0 = Builder.CreateMul(Tmp0, TIdX);
  Value *Tmp1 = Builder.CreateMul(TIdY, TCntZ, "", true, true);
  Value *TID = Builder.CreateAdd(Tmp0, Tmp1);
  TID = Builder.CreateAdd(TID, TIdZ);

  Value *Indices[] = {
    Constant::getNullValue(Type::getInt32Ty(Mod->getContext())),
    TID
  };

  Value *Offset = Builder.CreateInBoundsGEP(GVTy, GV, Indices);
  I.mutateType(Offset->getType());
  I.replaceAllUsesWith(Offset);
  I.eraseFromParent();

  SmallVector<IntrinsicInst *> DeferredIntrs;

  for (Value *V : WorkList) {
    CallInst *Call = dyn_cast<CallInst>(V);
    if (!Call) {
      if (ICmpInst *CI = dyn_cast<ICmpInst>(V)) {
        Value *Src0 = CI->getOperand(0);
        PointerType *NewTy = PointerType::getWithSamePointeeType(
            cast<PointerType>(Src0->getType()), AMDGPUAS::LOCAL_ADDRESS);

        if (isa<ConstantPointerNull>(CI->getOperand(0)))
          CI->setOperand(0, ConstantPointerNull::get(NewTy));

        if (isa<ConstantPointerNull>(CI->getOperand(1)))
          CI->setOperand(1, ConstantPointerNull::get(NewTy));

        continue;
      }

      // The operand's value should be corrected on its own and we don't want to
      // touch the users.
      if (isa<AddrSpaceCastInst>(V))
        continue;

      PointerType *NewTy = PointerType::getWithSamePointeeType(
          cast<PointerType>(V->getType()), AMDGPUAS::LOCAL_ADDRESS);

      // FIXME: It doesn't really make sense to try to do this for all
      // instructions.
      V->mutateType(NewTy);

      // Adjust the types of any constant operands.
      if (SelectInst *SI = dyn_cast<SelectInst>(V)) {
        if (isa<ConstantPointerNull>(SI->getOperand(1)))
          SI->setOperand(1, ConstantPointerNull::get(NewTy));

        if (isa<ConstantPointerNull>(SI->getOperand(2)))
          SI->setOperand(2, ConstantPointerNull::get(NewTy));
      } else if (PHINode *Phi = dyn_cast<PHINode>(V)) {
        for (unsigned I = 0, E = Phi->getNumIncomingValues(); I != E; ++I) {
          if (isa<ConstantPointerNull>(Phi->getIncomingValue(I)))
            Phi->setIncomingValue(I, ConstantPointerNull::get(NewTy));
        }
      }

      continue;
    }

    IntrinsicInst *Intr = cast<IntrinsicInst>(Call);
    Builder.SetInsertPoint(Intr);
    switch (Intr->getIntrinsicID()) {
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
      // These intrinsics are for address space 0 only
      Intr->eraseFromParent();
      continue;
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
      // These have 2 pointer operands. In case if second pointer also needs
      // to be replaced we defer processing of these intrinsics until all
      // other values are processed.
      DeferredIntrs.push_back(Intr);
      continue;
    case Intrinsic::memset: {
      MemSetInst *MemSet = cast<MemSetInst>(Intr);
      Builder.CreateMemSet(
          MemSet->getRawDest(), MemSet->getValue(), MemSet->getLength(),
          MaybeAlign(MemSet->getDestAlignment()), MemSet->isVolatile());
      Intr->eraseFromParent();
      continue;
    }
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::launder_invariant_group:
    case Intrinsic::strip_invariant_group:
      Intr->eraseFromParent();
      // FIXME: I think the invariant marker should still theoretically apply,
      // but the intrinsics need to be changed to accept pointers with any
      // address space.
      continue;
    case Intrinsic::objectsize: {
      Value *Src = Intr->getOperand(0);
      Function *ObjectSize = Intrinsic::getDeclaration(
          Mod, Intrinsic::objectsize,
          {Intr->getType(),
           PointerType::getWithSamePointeeType(
               cast<PointerType>(Src->getType()), AMDGPUAS::LOCAL_ADDRESS)});

      CallInst *NewCall = Builder.CreateCall(
          ObjectSize,
          {Src, Intr->getOperand(1), Intr->getOperand(2), Intr->getOperand(3)});
      Intr->replaceAllUsesWith(NewCall);
      Intr->eraseFromParent();
      continue;
    }
    default:
      Intr->print(errs());
      llvm_unreachable("Don't know how to promote alloca intrinsic use.");
    }
  }

  for (IntrinsicInst *Intr : DeferredIntrs) {
    Builder.SetInsertPoint(Intr);
    Intrinsic::ID ID = Intr->getIntrinsicID();
    assert(ID == Intrinsic::memcpy || ID == Intrinsic::memmove);

    MemTransferInst *MI = cast<MemTransferInst>(Intr);
    auto *B =
      Builder.CreateMemTransferInst(ID, MI->getRawDest(), MI->getDestAlign(),
                                    MI->getRawSource(), MI->getSourceAlign(),
                                    MI->getLength(), MI->isVolatile());

    for (unsigned I = 0; I != 2; ++I) {
      if (uint64_t Bytes = Intr->getParamDereferenceableBytes(I)) {
        B->addDereferenceableParamAttr(I, Bytes);
      }
    }

    Intr->eraseFromParent();
  }

  return true;
}

bool handlePromoteAllocaToVector(AllocaInst &I, unsigned MaxVGPRs) {
  // Array allocations are probably not worth handling, since an allocation of
  // the array type is the canonical form.
  if (!I.isStaticAlloca() || I.isArrayAllocation())
    return false;

  LLVM_DEBUG(dbgs() << "Trying to promote " << I << '\n');

  Module *Mod = I.getParent()->getParent()->getParent();
  return tryPromoteAllocaToVector(&I, Mod->getDataLayout(), MaxVGPRs);
}

bool promoteAllocasToVector(Function &F, TargetMachine &TM) {
  if (DisablePromoteAllocaToVector)
    return false;

  const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, F);
  if (!ST.isPromoteAllocaEnabled())
    return false;

  unsigned MaxVGPRs;
  if (TM.getTargetTriple().getArch() == Triple::amdgcn) {
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
    MaxVGPRs = ST.getMaxNumVGPRs(ST.getWavesPerEU(F).first);
    // A non-entry function has only 32 caller preserved registers.
    // Do not promote alloca which will force spilling.
    if (!AMDGPU::isEntryFunctionCC(F.getCallingConv()))
      MaxVGPRs = std::min(MaxVGPRs, 32u);
  } else {
    MaxVGPRs = 128;
  }

  bool Changed = false;
  BasicBlock &EntryBB = *F.begin();

  SmallVector<AllocaInst *, 16> Allocas;
  for (Instruction &I : EntryBB) {
    if (AllocaInst *AI = dyn_cast<AllocaInst>(&I))
      Allocas.push_back(AI);
  }

  for (AllocaInst *AI : Allocas) {
    if (handlePromoteAllocaToVector(*AI, MaxVGPRs))
      Changed = true;
  }

  return Changed;
}

bool AMDGPUPromoteAllocaToVector::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  if (auto *TPC = getAnalysisIfAvailable<TargetPassConfig>()) {
    return promoteAllocasToVector(F, TPC->getTM<TargetMachine>());
  }
  return false;
}

PreservedAnalyses
AMDGPUPromoteAllocaToVectorPass::run(Function &F, FunctionAnalysisManager &AM) {
  bool Changed = promoteAllocasToVector(F, TM);
  if (Changed) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }
  return PreservedAnalyses::all();
}

FunctionPass *llvm::createAMDGPUPromoteAlloca() {
  return new AMDGPUPromoteAlloca();
}

FunctionPass *llvm::createAMDGPUPromoteAllocaToVector() {
  return new AMDGPUPromoteAllocaToVector();
}
