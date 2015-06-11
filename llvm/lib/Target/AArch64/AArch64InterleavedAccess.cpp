//=--------------------- AArch64InterleavedAccess.cpp ----------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64InterleavedAccess pass, which identifies
// interleaved memory accesses and Transforms them into an AArch64 ldN/stN
// intrinsics (N = 2, 3, 4).
//
// An interleaved load reads data from memory into several vectors, with
// DE-interleaving the data on factor. An interleaved store writes several
// vectors to memory with RE-interleaving the data on factor. The interleave
// factor is equal to the number of vectors. AArch64 backend supports interleave
// factor of 2, 3 and 4.
//
// E.g. Transform an interleaved load (Factor = 2):
//        %wide.vec = load <8 x i32>, <8 x i32>* %ptr
//        %v0 = shuffle %wide.vec, undef, <0, 2, 4, 6>  ; Extract even elements
//        %v1 = shuffle %wide.vec, undef, <1, 3, 5, 7>  ; Extract odd elements
//      Into:
//        %ld2 = { <4 x i32>, <4 x i32> } call aarch64.neon.ld2(%ptr)
//        %v0 = extractelement { <4 x i32>, <4 x i32> } %ld2, i32 0
//        %v1 = extractelement { <4 x i32>, <4 x i32> } %ld2, i32 1
//
// E.g. Transform an interleaved store (Factor = 2):
//        %i.vec = shuffle %v0, %v1, <0, 4, 1, 5, 2, 6, 3, 7>  ; Interleaved vec
//        store <8 x i32> %i.vec, <8 x i32>* %ptr
//      Into:
//        %v0 = shuffle %i.vec, undef, <0, 1, 2, 3>
//        %v1 = shuffle %i.vec, undef, <4, 5, 6, 7>
//        call void aarch64.neon.st2(%v0, %v1, %ptr)
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-interleaved-access"

static const unsigned MIN_FACTOR = 2;
static const unsigned MAX_FACTOR = 4;

namespace llvm {
static void initializeAArch64InterleavedAccessPass(PassRegistry &);
}

namespace {

class AArch64InterleavedAccess : public FunctionPass {

public:
  static char ID;
  AArch64InterleavedAccess() : FunctionPass(ID) {
    initializeAArch64InterleavedAccessPass(*PassRegistry::getPassRegistry());
  }

  const char *getPassName() const override {
    return "AArch64 Interleaved Access Pass";
  }

  bool runOnFunction(Function &F) override;

private:
  const DataLayout *DL;
  Module *M;

  /// \brief Transform an interleaved load into ldN intrinsic.
  bool matchInterleavedLoad(ShuffleVectorInst *SVI,
                            SmallSetVector<Instruction *, 32> &DeadInsts);

  /// \brief Transform an interleaved store into stN intrinsic.
  bool matchInterleavedStore(ShuffleVectorInst *SVI,
                             SmallSetVector<Instruction *, 32> &DeadInsts);
};
} // end anonymous namespace.

char AArch64InterleavedAccess::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64InterleavedAccess, DEBUG_TYPE,
                      "AArch64 interleaved access Pass", false, false)
INITIALIZE_PASS_END(AArch64InterleavedAccess, DEBUG_TYPE,
                    "AArch64 interleaved access Pass", false, false)

FunctionPass *llvm::createAArch64InterleavedAccessPass() {
  return new AArch64InterleavedAccess();
}

/// \brief Get a ldN/stN intrinsic according to the Factor (2, 3, or 4).
static Intrinsic::ID getLdNStNIntrinsic(unsigned Factor, bool IsLoad) {
  static const Intrinsic::ID LoadInt[3] = {Intrinsic::aarch64_neon_ld2,
                                           Intrinsic::aarch64_neon_ld3,
                                           Intrinsic::aarch64_neon_ld4};
  static const Intrinsic::ID StoreInt[3] = {Intrinsic::aarch64_neon_st2,
                                            Intrinsic::aarch64_neon_st3,
                                            Intrinsic::aarch64_neon_st4};

  assert(Factor >= MIN_FACTOR && Factor <= MAX_FACTOR &&
         "Invalid interleave factor");

  if (IsLoad)
    return LoadInt[Factor - 2];
  else
    return StoreInt[Factor - 2];
}

/// \brief Check if the mask is a DE-interleave mask of the given factor
/// \p Factor like:
///     <Index, Index+Factor, ..., Index+(NumElts-1)*Factor>
static bool isDeInterleaveMaskOfFactor(ArrayRef<int> Mask, unsigned Factor,
                                       unsigned &Index) {
  // Check all potential start indices from 0 to (Factor - 1).
  for (Index = 0; Index < Factor; Index++) {
    unsigned i = 0;

    // Check that elements are in ascending order by Factor.
    for (; i < Mask.size(); i++)
      if (Mask[i] >= 0 && static_cast<unsigned>(Mask[i]) != Index + i * Factor)
        break;

    if (i == Mask.size())
      return true;
  }

  return false;
}

/// \brief Check if the mask is a DE-interleave mask for an interleaved load.
///
/// E.g. DE-interleave masks (Factor = 2) could be:
///     <0, 2, 4, 6>    (mask of index 0 to extract even elements)
///     <1, 3, 5, 7>    (mask of index 1 to extract odd elements)
static bool isDeInterleaveMask(ArrayRef<int> Mask, unsigned &Factor,
                               unsigned &Index) {
  unsigned NumElts = Mask.size();
  if (NumElts < 2)
    return false;

  for (Factor = MIN_FACTOR; Factor <= MAX_FACTOR; Factor++)
    if (isDeInterleaveMaskOfFactor(Mask, Factor, Index))
      return true;

  return false;
}

/// \brief Check if the given mask \p Mask is RE-interleaved mask of the given
/// factor \p Factor.
///
/// I.e. <0, NumSubElts, ... , NumSubElts*(Factor - 1), 1, NumSubElts + 1, ...>
static bool isReInterleaveMaskOfFactor(ArrayRef<int> Mask, unsigned Factor) {
  unsigned NumElts = Mask.size();
  if (NumElts % Factor)
    return false;

  unsigned NumSubElts = NumElts / Factor;
  if (!isPowerOf2_32(NumSubElts))
    return false;

  for (unsigned i = 0; i < NumSubElts; i++)
    for (unsigned j = 0; j < Factor; j++)
      if (Mask[i * Factor + j] >= 0 &&
          static_cast<unsigned>(Mask[i * Factor + j]) != j * NumSubElts + i)
        return false;

  return true;
}

/// \brief Check if the mask is RE-interleave mask for an interleaved store.
///
/// E.g. The RE-interleave mask (Factor = 2) could be:
///     <0, 4, 1, 5, 2, 6, 3, 7>
static bool isReInterleaveMask(ArrayRef<int> Mask, unsigned &Factor) {
  if (Mask.size() < 4)
    return false;

  // Check potential Factors and return true if find a factor for the mask.
  for (Factor = MIN_FACTOR; Factor <= MAX_FACTOR; Factor++)
    if (isReInterleaveMaskOfFactor(Mask, Factor))
      return true;

  return false;
}

/// \brief Get a mask consisting of sequential integers starting from \p Start.
///
/// I.e. <Start, Start + 1, ..., Start + NumElts - 1>
static Constant *getSequentialMask(IRBuilder<> &Builder, unsigned Start,
                                   unsigned NumElts) {
  SmallVector<Constant *, 16> Mask;
  for (unsigned i = 0; i < NumElts; i++)
    Mask.push_back(Builder.getInt32(Start + i));

  return ConstantVector::get(Mask);
}

bool AArch64InterleavedAccess::matchInterleavedLoad(
    ShuffleVectorInst *SVI, SmallSetVector<Instruction *, 32> &DeadInsts) {
  if (DeadInsts.count(SVI))
    return false;

  LoadInst *LI = dyn_cast<LoadInst>(SVI->getOperand(0));
  if (!LI || !LI->isSimple() || !isa<UndefValue>(SVI->getOperand(1)))
    return false;

  SmallVector<ShuffleVectorInst *, 4> Shuffles;

  // Check if all users of this load are shufflevectors.
  for (auto UI = LI->user_begin(), E = LI->user_end(); UI != E; UI++) {
    ShuffleVectorInst *SV = dyn_cast<ShuffleVectorInst>(*UI);
    if (!SV)
      return false;

    Shuffles.push_back(SV);
  }

  // Check if the type of the first shuffle is legal.
  VectorType *VecTy = Shuffles[0]->getType();
  unsigned TypeSize = DL->getTypeAllocSizeInBits(VecTy);
  if (TypeSize != 64 && TypeSize != 128)
    return false;

  // Check if the mask of the first shuffle is strided and get the start index.
  unsigned Factor, Index;
  if (!isDeInterleaveMask(Shuffles[0]->getShuffleMask(), Factor, Index))
    return false;

  // Holds the corresponding index for each strided shuffle.
  SmallVector<unsigned, 4> Indices;
  Indices.push_back(Index);

  // Check if other shufflevectors are of the same type and factor
  for (unsigned i = 1; i < Shuffles.size(); i++) {
    if (Shuffles[i]->getType() != VecTy)
      return false;

    unsigned Index;
    if (!isDeInterleaveMaskOfFactor(Shuffles[i]->getShuffleMask(), Factor,
                                    Index))
      return false;

    Indices.push_back(Index);
  }

  DEBUG(dbgs() << "Found an interleaved load:" << *LI << "\n");

  // A pointer vector can not be the return type of the ldN intrinsics. Need to
  // load integer vectors first and then convert to pointer vectors.
  Type *EltTy = VecTy->getVectorElementType();
  if (EltTy->isPointerTy())
    VecTy = VectorType::get(DL->getIntPtrType(EltTy),
                            VecTy->getVectorNumElements());

  Type *PtrTy = VecTy->getPointerTo(LI->getPointerAddressSpace());
  Type *Tys[2] = {VecTy, PtrTy};
  Function *LdNFunc =
      Intrinsic::getDeclaration(M, getLdNStNIntrinsic(Factor, true), Tys);

  IRBuilder<> Builder(LI);
  Value *Ptr = Builder.CreateBitCast(LI->getPointerOperand(), PtrTy);

  CallInst *LdN = Builder.CreateCall(LdNFunc, Ptr, "ldN");
  DEBUG(dbgs() << "   Created:" << *LdN << "\n");

  // Replace each strided shufflevector with the corresponding vector loaded
  // by ldN.
  for (unsigned i = 0; i < Shuffles.size(); i++) {
    ShuffleVectorInst *SV = Shuffles[i];
    unsigned Index = Indices[i];

    Value *SubVec = Builder.CreateExtractValue(LdN, Index);

    // Convert the integer vector to pointer vector if the element is pointer.
    if (EltTy->isPointerTy())
      SubVec = Builder.CreateIntToPtr(SubVec, SV->getType());

    SV->replaceAllUsesWith(SubVec);

    DEBUG(dbgs() << "  Replaced:" << *SV << "\n"
                 << "      With:" << *SubVec << "\n");

    // Avoid analyzing it twice.
    DeadInsts.insert(SV);
  }

  // Mark this load as dead.
  DeadInsts.insert(LI);
  return true;
}

bool AArch64InterleavedAccess::matchInterleavedStore(
    ShuffleVectorInst *SVI, SmallSetVector<Instruction *, 32> &DeadInsts) {
  if (DeadInsts.count(SVI) || !SVI->hasOneUse())
    return false;

  StoreInst *SI = dyn_cast<StoreInst>(SVI->user_back());
  if (!SI || !SI->isSimple())
    return false;

  // Check if the mask is interleaved and get the interleave factor.
  unsigned Factor;
  if (!isReInterleaveMask(SVI->getShuffleMask(), Factor))
    return false;

  VectorType *VecTy = SVI->getType();
  unsigned NumSubElts = VecTy->getVectorNumElements() / Factor;
  Type *EltTy = VecTy->getVectorElementType();
  VectorType *SubVecTy = VectorType::get(EltTy, NumSubElts);

  // Skip illegal vector types.
  unsigned TypeSize = DL->getTypeAllocSizeInBits(SubVecTy);
  if (TypeSize != 64 && TypeSize != 128)
    return false;

  DEBUG(dbgs() << "Found an interleaved store:" << *SI << "\n");

  Value *Op0 = SVI->getOperand(0);
  Value *Op1 = SVI->getOperand(1);
  IRBuilder<> Builder(SI);

  // StN intrinsics don't support pointer vectors as arguments. Convert pointer
  // vectors to integer vectors.
  if (EltTy->isPointerTy()) {
    Type *IntTy = DL->getIntPtrType(EltTy);
    unsigned NumOpElts =
        dyn_cast<VectorType>(Op0->getType())->getVectorNumElements();

    // The corresponding integer vector type of the same element size.
    Type *IntVecTy = VectorType::get(IntTy, NumOpElts);

    Op0 = Builder.CreatePtrToInt(Op0, IntVecTy);
    Op1 = Builder.CreatePtrToInt(Op1, IntVecTy);
    SubVecTy = VectorType::get(IntTy, NumSubElts);
  }

  Type *PtrTy = SubVecTy->getPointerTo(SI->getPointerAddressSpace());
  Type *Tys[2] = {SubVecTy, PtrTy};
  Function *StNFunc =
      Intrinsic::getDeclaration(M, getLdNStNIntrinsic(Factor, false), Tys);

  SmallVector<Value *, 5> Ops;

  // Split the shufflevector operands into sub vectors for the new stN call.
  for (unsigned i = 0; i < Factor; i++)
    Ops.push_back(Builder.CreateShuffleVector(
        Op0, Op1, getSequentialMask(Builder, NumSubElts * i, NumSubElts)));

  Ops.push_back(Builder.CreateBitCast(SI->getPointerOperand(), PtrTy));
  CallInst *StN = Builder.CreateCall(StNFunc, Ops);

  (void)StN; // silence warning.
  DEBUG(dbgs() << "  Replaced:" << *SI << "'\n");
  DEBUG(dbgs() << "      with:" << *StN << "\n");

  // Mark this shufflevector and store as dead.
  DeadInsts.insert(SI);
  DeadInsts.insert(SVI);
  return true;
}

bool AArch64InterleavedAccess::runOnFunction(Function &F) {
  DEBUG(dbgs() << "*** " << getPassName() << ": " << F.getName() << "\n");

  M = F.getParent();
  DL = &M->getDataLayout();

  // Holds dead instructions that will be erased later.
  SmallSetVector<Instruction *, 32> DeadInsts;
  bool Changed = false;
  for (auto &I : inst_range(F)) {
    if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(&I)) {
      Changed |= matchInterleavedLoad(SVI, DeadInsts);
      Changed |= matchInterleavedStore(SVI, DeadInsts);
    }
  }

  for (auto I : DeadInsts)
    I->eraseFromParent();

  return Changed;
}
