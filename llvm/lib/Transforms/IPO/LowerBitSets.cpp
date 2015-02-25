//===-- LowerBitSets.cpp - Bitset lowering pass ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers bitset metadata and calls to the llvm.bitset.test intrinsic.
// See http://llvm.org/docs/LangRef.html#bitsets for more information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/LowerBitSets.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "lowerbitsets"

STATISTIC(NumBitSetsCreated, "Number of bitsets created");
STATISTIC(NumBitSetCallsLowered, "Number of bitset calls lowered");
STATISTIC(NumBitSetDisjointSets, "Number of disjoint sets of bitsets");

bool BitSetInfo::containsGlobalOffset(uint64_t Offset) const {
  if (Offset < ByteOffset)
    return false;

  if ((Offset - ByteOffset) % (uint64_t(1) << AlignLog2) != 0)
    return false;

  uint64_t BitOffset = (Offset - ByteOffset) >> AlignLog2;
  if (BitOffset >= BitSize)
    return false;

  return (Bits[BitOffset / 8] >> (BitOffset % 8)) & 1;
}

bool BitSetInfo::containsValue(
    const DataLayout *DL,
    const DenseMap<GlobalVariable *, uint64_t> &GlobalLayout, Value *V,
    uint64_t COffset) const {
  if (auto GV = dyn_cast<GlobalVariable>(V)) {
    auto I = GlobalLayout.find(GV);
    if (I == GlobalLayout.end())
      return false;
    return containsGlobalOffset(I->second + COffset);
  }

  if (auto GEP = dyn_cast<GEPOperator>(V)) {
    APInt APOffset(DL->getPointerSizeInBits(0), 0);
    bool Result = GEP->accumulateConstantOffset(*DL, APOffset);
    if (!Result)
      return false;
    COffset += APOffset.getZExtValue();
    return containsValue(DL, GlobalLayout, GEP->getPointerOperand(),
                         COffset);
  }

  if (auto Op = dyn_cast<Operator>(V)) {
    if (Op->getOpcode() == Instruction::BitCast)
      return containsValue(DL, GlobalLayout, Op->getOperand(0), COffset);

    if (Op->getOpcode() == Instruction::Select)
      return containsValue(DL, GlobalLayout, Op->getOperand(1), COffset) &&
             containsValue(DL, GlobalLayout, Op->getOperand(2), COffset);
  }

  return false;
}

BitSetInfo BitSetBuilder::build() {
  if (Min > Max)
    Min = 0;

  // Normalize each offset against the minimum observed offset, and compute
  // the bitwise OR of each of the offsets. The number of trailing zeros
  // in the mask gives us the log2 of the alignment of all offsets, which
  // allows us to compress the bitset by only storing one bit per aligned
  // address.
  uint64_t Mask = 0;
  for (uint64_t &Offset : Offsets) {
    Offset -= Min;
    Mask |= Offset;
  }

  BitSetInfo BSI;
  BSI.ByteOffset = Min;

  BSI.AlignLog2 = 0;
  // FIXME: Can probably do something smarter if all offsets are 0.
  if (Mask != 0)
    BSI.AlignLog2 = countTrailingZeros(Mask, ZB_Undefined);

  // Build the compressed bitset while normalizing the offsets against the
  // computed alignment.
  BSI.BitSize = ((Max - Min) >> BSI.AlignLog2) + 1;
  uint64_t ByteSize = (BSI.BitSize + 7) / 8;
  BSI.Bits.resize(ByteSize);
  for (uint64_t Offset : Offsets) {
    Offset >>= BSI.AlignLog2;
    BSI.Bits[Offset / 8] |= 1 << (Offset % 8);
  }

  return BSI;
}

void GlobalLayoutBuilder::addFragment(const std::set<uint64_t> &F) {
  // Create a new fragment to hold the layout for F.
  Fragments.emplace_back();
  std::vector<uint64_t> &Fragment = Fragments.back();
  uint64_t FragmentIndex = Fragments.size() - 1;

  for (auto ObjIndex : F) {
    uint64_t OldFragmentIndex = FragmentMap[ObjIndex];
    if (OldFragmentIndex == 0) {
      // We haven't seen this object index before, so just add it to the current
      // fragment.
      Fragment.push_back(ObjIndex);
    } else {
      // This index belongs to an existing fragment. Copy the elements of the
      // old fragment into this one and clear the old fragment. We don't update
      // the fragment map just yet, this ensures that any further references to
      // indices from the old fragment in this fragment do not insert any more
      // indices.
      std::vector<uint64_t> &OldFragment = Fragments[OldFragmentIndex];
      Fragment.insert(Fragment.end(), OldFragment.begin(), OldFragment.end());
      OldFragment.clear();
    }
  }

  // Update the fragment map to point our object indices to this fragment.
  for (uint64_t ObjIndex : Fragment)
    FragmentMap[ObjIndex] = FragmentIndex;
}

namespace {

struct LowerBitSets : public ModulePass {
  static char ID;
  LowerBitSets() : ModulePass(ID) {
    initializeLowerBitSetsPass(*PassRegistry::getPassRegistry());
  }

  const DataLayout *DL;
  IntegerType *Int1Ty;
  IntegerType *Int8Ty;
  IntegerType *Int32Ty;
  Type *Int32PtrTy;
  IntegerType *Int64Ty;
  Type *IntPtrTy;

  // The llvm.bitsets named metadata.
  NamedMDNode *BitSetNM;

  // Mapping from bitset mdstrings to the call sites that test them.
  DenseMap<MDString *, std::vector<CallInst *>> BitSetTestCallSites;

  BitSetInfo
  buildBitSet(MDString *BitSet,
              const DenseMap<GlobalVariable *, uint64_t> &GlobalLayout);
  Value *createBitSetTest(IRBuilder<> &B, const BitSetInfo &BSI,
                          GlobalVariable *BitSetGlobal, Value *BitOffset);
  Value *
  lowerBitSetCall(CallInst *CI, const BitSetInfo &BSI,
                  GlobalVariable *BitSetGlobal, GlobalVariable *CombinedGlobal,
                  const DenseMap<GlobalVariable *, uint64_t> &GlobalLayout);
  void buildBitSetsFromGlobals(Module &M,
                               const std::vector<MDString *> &BitSets,
                               const std::vector<GlobalVariable *> &Globals);
  bool buildBitSets(Module &M);
  bool eraseBitSetMetadata(Module &M);

  bool doInitialization(Module &M) override;
  bool runOnModule(Module &M) override;
};

} // namespace

INITIALIZE_PASS_BEGIN(LowerBitSets, "lowerbitsets",
                "Lower bitset metadata", false, false)
INITIALIZE_PASS_END(LowerBitSets, "lowerbitsets",
                "Lower bitset metadata", false, false)
char LowerBitSets::ID = 0;

ModulePass *llvm::createLowerBitSetsPass() { return new LowerBitSets; }

bool LowerBitSets::doInitialization(Module &M) {
  DL = M.getDataLayout();
  if (!DL)
    report_fatal_error("Data layout required");

  Int1Ty = Type::getInt1Ty(M.getContext());
  Int8Ty = Type::getInt8Ty(M.getContext());
  Int32Ty = Type::getInt32Ty(M.getContext());
  Int32PtrTy = PointerType::getUnqual(Int32Ty);
  Int64Ty = Type::getInt64Ty(M.getContext());
  IntPtrTy = DL->getIntPtrType(M.getContext(), 0);

  BitSetNM = M.getNamedMetadata("llvm.bitsets");

  BitSetTestCallSites.clear();

  return false;
}

/// Build a bit set for BitSet using the object layouts in
/// GlobalLayout.
BitSetInfo LowerBitSets::buildBitSet(
    MDString *BitSet,
    const DenseMap<GlobalVariable *, uint64_t> &GlobalLayout) {
  BitSetBuilder BSB;

  // Compute the byte offset of each element of this bitset.
  if (BitSetNM) {
    for (MDNode *Op : BitSetNM->operands()) {
      if (Op->getOperand(0) != BitSet || !Op->getOperand(1))
        continue;
      auto OpGlobal = cast<GlobalVariable>(
          cast<ConstantAsMetadata>(Op->getOperand(1))->getValue());
      uint64_t Offset =
          cast<ConstantInt>(cast<ConstantAsMetadata>(Op->getOperand(2))
                                ->getValue())->getZExtValue();

      Offset += GlobalLayout.find(OpGlobal)->second;

      BSB.addOffset(Offset);
    }
  }

  return BSB.build();
}

/// Build a test that bit BitOffset mod sizeof(Bits)*8 is set in
/// Bits. This pattern matches to the bt instruction on x86.
static Value *createMaskedBitTest(IRBuilder<> &B, Value *Bits,
                                  Value *BitOffset) {
  auto BitsType = cast<IntegerType>(Bits->getType());
  unsigned BitWidth = BitsType->getBitWidth();

  BitOffset = B.CreateZExtOrTrunc(BitOffset, BitsType);
  Value *BitIndex =
      B.CreateAnd(BitOffset, ConstantInt::get(BitsType, BitWidth - 1));
  Value *BitMask = B.CreateShl(ConstantInt::get(BitsType, 1), BitIndex);
  Value *MaskedBits = B.CreateAnd(Bits, BitMask);
  return B.CreateICmpNE(MaskedBits, ConstantInt::get(BitsType, 0));
}

/// Build a test that bit BitOffset is set in BSI, where
/// BitSetGlobal is a global containing the bits in BSI.
Value *LowerBitSets::createBitSetTest(IRBuilder<> &B, const BitSetInfo &BSI,
                                      GlobalVariable *BitSetGlobal,
                                      Value *BitOffset) {
  if (BSI.Bits.size() <= 8) {
    // If the bit set is sufficiently small, we can avoid a load by bit testing
    // a constant.
    IntegerType *BitsTy;
    if (BSI.Bits.size() <= 4)
      BitsTy = Int32Ty;
    else
      BitsTy = Int64Ty;

    uint64_t Bits = 0;
    for (auto I = BSI.Bits.rbegin(), E = BSI.Bits.rend(); I != E; ++I) {
      Bits <<= 8;
      Bits |= *I;
    }
    Constant *BitsConst = ConstantInt::get(BitsTy, Bits);
    return createMaskedBitTest(B, BitsConst, BitOffset);
  } else {
    // TODO: We might want to use the memory variant of the bt instruction
    // with the previously computed bit offset at -Os. This instruction does
    // exactly what we want but has been benchmarked as being slower than open
    // coding the load+bt.
    Value *BitSetGlobalOffset =
        B.CreateLShr(BitOffset, ConstantInt::get(IntPtrTy, 5));
    Value *BitSetEntryAddr = B.CreateGEP(
        ConstantExpr::getBitCast(BitSetGlobal, Int32PtrTy), BitSetGlobalOffset);
    Value *BitSetEntry = B.CreateLoad(BitSetEntryAddr);

    return createMaskedBitTest(B, BitSetEntry, BitOffset);
  }
}

/// Lower a llvm.bitset.test call to its implementation. Returns the value to
/// replace the call with.
Value *LowerBitSets::lowerBitSetCall(
    CallInst *CI, const BitSetInfo &BSI, GlobalVariable *BitSetGlobal,
    GlobalVariable *CombinedGlobal,
    const DenseMap<GlobalVariable *, uint64_t> &GlobalLayout) {
  Value *Ptr = CI->getArgOperand(0);

  if (BSI.containsValue(DL, GlobalLayout, Ptr))
    return ConstantInt::getTrue(BitSetGlobal->getParent()->getContext());

  Constant *GlobalAsInt = ConstantExpr::getPtrToInt(CombinedGlobal, IntPtrTy);
  Constant *OffsetedGlobalAsInt = ConstantExpr::getAdd(
      GlobalAsInt, ConstantInt::get(IntPtrTy, BSI.ByteOffset));

  BasicBlock *InitialBB = CI->getParent();

  IRBuilder<> B(CI);

  Value *PtrAsInt = B.CreatePtrToInt(Ptr, IntPtrTy);

  if (BSI.isSingleOffset())
    return B.CreateICmpEQ(PtrAsInt, OffsetedGlobalAsInt);

  Value *PtrOffset = B.CreateSub(PtrAsInt, OffsetedGlobalAsInt);

  Value *BitOffset;
  if (BSI.AlignLog2 == 0) {
    BitOffset = PtrOffset;
  } else {
    // We need to check that the offset both falls within our range and is
    // suitably aligned. We can check both properties at the same time by
    // performing a right rotate by log2(alignment) followed by an integer
    // comparison against the bitset size. The rotate will move the lower
    // order bits that need to be zero into the higher order bits of the
    // result, causing the comparison to fail if they are nonzero. The rotate
    // also conveniently gives us a bit offset to use during the load from
    // the bitset.
    Value *OffsetSHR =
        B.CreateLShr(PtrOffset, ConstantInt::get(IntPtrTy, BSI.AlignLog2));
    Value *OffsetSHL = B.CreateShl(
        PtrOffset, ConstantInt::get(IntPtrTy, DL->getPointerSizeInBits(0) -
                                                  BSI.AlignLog2));
    BitOffset = B.CreateOr(OffsetSHR, OffsetSHL);
  }

  Constant *BitSizeConst = ConstantInt::get(IntPtrTy, BSI.BitSize);
  Value *OffsetInRange = B.CreateICmpULT(BitOffset, BitSizeConst);

  // If the bit set is all ones, testing against it is unnecessary.
  if (BSI.isAllOnes())
    return OffsetInRange;

  TerminatorInst *Term = SplitBlockAndInsertIfThen(OffsetInRange, CI, false);
  IRBuilder<> ThenB(Term);

  // Now that we know that the offset is in range and aligned, load the
  // appropriate bit from the bitset.
  Value *Bit = createBitSetTest(ThenB, BSI, BitSetGlobal, BitOffset);

  // The value we want is 0 if we came directly from the initial block
  // (having failed the range or alignment checks), or the loaded bit if
  // we came from the block in which we loaded it.
  B.SetInsertPoint(CI);
  PHINode *P = B.CreatePHI(Int1Ty, 2);
  P->addIncoming(ConstantInt::get(Int1Ty, 0), InitialBB);
  P->addIncoming(Bit, ThenB.GetInsertBlock());
  return P;
}

/// Given a disjoint set of bitsets and globals, layout the globals, build the
/// bit sets and lower the llvm.bitset.test calls.
void LowerBitSets::buildBitSetsFromGlobals(
    Module &M,
    const std::vector<MDString *> &BitSets,
    const std::vector<GlobalVariable *> &Globals) {
  // Build a new global with the combined contents of the referenced globals.
  std::vector<Constant *> GlobalInits;
  for (GlobalVariable *G : Globals) {
    GlobalInits.push_back(G->getInitializer());
    uint64_t InitSize = DL->getTypeAllocSize(G->getInitializer()->getType());

    // Compute the amount of padding required to align the next element to the
    // next power of 2.
    uint64_t Padding = NextPowerOf2(InitSize - 1) - InitSize;

    // Cap at 128 was found experimentally to have a good data/instruction
    // overhead tradeoff.
    if (Padding > 128)
      Padding = RoundUpToAlignment(InitSize, 128) - InitSize;

    GlobalInits.push_back(
        ConstantAggregateZero::get(ArrayType::get(Int8Ty, Padding)));
  }
  if (!GlobalInits.empty())
    GlobalInits.pop_back();
  Constant *NewInit = ConstantStruct::getAnon(M.getContext(), GlobalInits);
  auto CombinedGlobal =
      new GlobalVariable(M, NewInit->getType(), /*isConstant=*/true,
                         GlobalValue::PrivateLinkage, NewInit);

  const StructLayout *CombinedGlobalLayout =
      DL->getStructLayout(cast<StructType>(NewInit->getType()));

  // Compute the offsets of the original globals within the new global.
  DenseMap<GlobalVariable *, uint64_t> GlobalLayout;
  for (unsigned I = 0; I != Globals.size(); ++I)
    // Multiply by 2 to account for padding elements.
    GlobalLayout[Globals[I]] = CombinedGlobalLayout->getElementOffset(I * 2);

  // For each bitset in this disjoint set...
  for (MDString *BS : BitSets) {
    // Build the bitset.
    BitSetInfo BSI = buildBitSet(BS, GlobalLayout);

    // Create a global in which to store it.
    ++NumBitSetsCreated;
    Constant *BitsConst = ConstantDataArray::get(M.getContext(), BSI.Bits);
    auto BitSetGlobal = new GlobalVariable(
        M, BitsConst->getType(), /*isConstant=*/true,
        GlobalValue::PrivateLinkage, BitsConst, BS->getString() + ".bits");

    // Lower each call to llvm.bitset.test for this bitset.
    for (CallInst *CI : BitSetTestCallSites[BS]) {
      ++NumBitSetCallsLowered;
      Value *Lowered =
          lowerBitSetCall(CI, BSI, BitSetGlobal, CombinedGlobal, GlobalLayout);
      CI->replaceAllUsesWith(Lowered);
      CI->eraseFromParent();
    }
  }

  // Build aliases pointing to offsets into the combined global for each
  // global from which we built the combined global, and replace references
  // to the original globals with references to the aliases.
  for (unsigned I = 0; I != Globals.size(); ++I) {
    // Multiply by 2 to account for padding elements.
    Constant *CombinedGlobalIdxs[] = {ConstantInt::get(Int32Ty, 0),
                                      ConstantInt::get(Int32Ty, I * 2)};
    Constant *CombinedGlobalElemPtr =
        ConstantExpr::getGetElementPtr(CombinedGlobal, CombinedGlobalIdxs);
    GlobalAlias *GAlias = GlobalAlias::create(
        Globals[I]->getType()->getElementType(),
        Globals[I]->getType()->getAddressSpace(), Globals[I]->getLinkage(),
        "", CombinedGlobalElemPtr, &M);
    GAlias->takeName(Globals[I]);
    Globals[I]->replaceAllUsesWith(GAlias);
    Globals[I]->eraseFromParent();
  }
}

/// Lower all bit sets in this module.
bool LowerBitSets::buildBitSets(Module &M) {
  Function *BitSetTestFunc =
      M.getFunction(Intrinsic::getName(Intrinsic::bitset_test));
  if (!BitSetTestFunc)
    return false;

  // Equivalence class set containing bitsets and the globals they reference.
  // This is used to partition the set of bitsets in the module into disjoint
  // sets.
  typedef EquivalenceClasses<PointerUnion<GlobalVariable *, MDString *>>
      GlobalClassesTy;
  GlobalClassesTy GlobalClasses;

  for (const Use &U : BitSetTestFunc->uses()) {
    auto CI = cast<CallInst>(U.getUser());

    auto BitSetMDVal = dyn_cast<MetadataAsValue>(CI->getArgOperand(1));
    if (!BitSetMDVal || !isa<MDString>(BitSetMDVal->getMetadata()))
      report_fatal_error(
          "Second argument of llvm.bitset.test must be metadata string");
    auto BitSet = cast<MDString>(BitSetMDVal->getMetadata());

    // Add the call site to the list of call sites for this bit set. We also use
    // BitSetTestCallSites to keep track of whether we have seen this bit set
    // before. If we have, we don't need to re-add the referenced globals to the
    // equivalence class.
    std::pair<DenseMap<MDString *, std::vector<CallInst *>>::iterator,
              bool> Ins =
        BitSetTestCallSites.insert(
            std::make_pair(BitSet, std::vector<CallInst *>()));
    Ins.first->second.push_back(CI);
    if (!Ins.second)
      continue;

    // Add the bitset to the equivalence class.
    GlobalClassesTy::iterator GCI = GlobalClasses.insert(BitSet);
    GlobalClassesTy::member_iterator CurSet = GlobalClasses.findLeader(GCI);

    if (!BitSetNM)
      continue;

    // Verify the bitset metadata and add the referenced globals to the bitset's
    // equivalence class.
    for (MDNode *Op : BitSetNM->operands()) {
      if (Op->getNumOperands() != 3)
        report_fatal_error(
            "All operands of llvm.bitsets metadata must have 3 elements");

      if (Op->getOperand(0) != BitSet || !Op->getOperand(1))
        continue;

      auto OpConstMD = dyn_cast<ConstantAsMetadata>(Op->getOperand(1));
      if (!OpConstMD)
        report_fatal_error("Bit set element must be a constant");
      auto OpGlobal = dyn_cast<GlobalVariable>(OpConstMD->getValue());
      if (!OpGlobal)
        report_fatal_error("Bit set element must refer to global");

      auto OffsetConstMD = dyn_cast<ConstantAsMetadata>(Op->getOperand(2));
      if (!OffsetConstMD)
        report_fatal_error("Bit set element offset must be a constant");
      auto OffsetInt = dyn_cast<ConstantInt>(OffsetConstMD->getValue());
      if (!OffsetInt)
        report_fatal_error(
            "Bit set element offset must be an integer constant");

      CurSet = GlobalClasses.unionSets(
          CurSet, GlobalClasses.findLeader(GlobalClasses.insert(OpGlobal)));
    }
  }

  if (GlobalClasses.empty())
    return false;

  // For each disjoint set we found...
  for (GlobalClassesTy::iterator I = GlobalClasses.begin(),
                                 E = GlobalClasses.end();
       I != E; ++I) {
    if (!I->isLeader()) continue;

    ++NumBitSetDisjointSets;

    // Build the list of bitsets and referenced globals in this disjoint set.
    std::vector<MDString *> BitSets;
    std::vector<GlobalVariable *> Globals;
    llvm::DenseMap<MDString *, uint64_t> BitSetIndices;
    llvm::DenseMap<GlobalVariable *, uint64_t> GlobalIndices;
    for (GlobalClassesTy::member_iterator MI = GlobalClasses.member_begin(I);
         MI != GlobalClasses.member_end(); ++MI) {
      if ((*MI).is<MDString *>()) {
        BitSetIndices[MI->get<MDString *>()] = BitSets.size();
        BitSets.push_back(MI->get<MDString *>());
      } else {
        GlobalIndices[MI->get<GlobalVariable *>()] = Globals.size();
        Globals.push_back(MI->get<GlobalVariable *>());
      }
    }

    // For each bitset, build a set of indices that refer to globals referenced
    // by the bitset.
    std::vector<std::set<uint64_t>> BitSetMembers(BitSets.size());
    if (BitSetNM) {
      for (MDNode *Op : BitSetNM->operands()) {
        // Op = { bitset name, global, offset }
        if (!Op->getOperand(1))
          continue;
        auto I = BitSetIndices.find(cast<MDString>(Op->getOperand(0)));
        if (I == BitSetIndices.end())
          continue;

        auto OpGlobal = cast<GlobalVariable>(
            cast<ConstantAsMetadata>(Op->getOperand(1))->getValue());
        BitSetMembers[I->second].insert(GlobalIndices[OpGlobal]);
      }
    }

    // Order the sets of indices by size. The GlobalLayoutBuilder works best
    // when given small index sets first.
    std::stable_sort(
        BitSetMembers.begin(), BitSetMembers.end(),
        [](const std::set<uint64_t> &O1, const std::set<uint64_t> &O2) {
          return O1.size() < O2.size();
        });

    // Create a GlobalLayoutBuilder and provide it with index sets as layout
    // fragments. The GlobalLayoutBuilder tries to lay out members of fragments
    // as close together as possible.
    GlobalLayoutBuilder GLB(Globals.size());
    for (auto &&MemSet : BitSetMembers)
      GLB.addFragment(MemSet);

    // Build a vector of globals with the computed layout.
    std::vector<GlobalVariable *> OrderedGlobals(Globals.size());
    auto OGI = OrderedGlobals.begin();
    for (auto &&F : GLB.Fragments)
      for (auto &&Offset : F)
        *OGI++ = Globals[Offset];

    // Order bitsets by name for determinism.
    std::sort(BitSets.begin(), BitSets.end(), [](MDString *S1, MDString *S2) {
      return S1->getString() < S2->getString();
    });

    // Build the bitsets from this disjoint set.
    buildBitSetsFromGlobals(M, BitSets, OrderedGlobals);
  }

  return true;
}

bool LowerBitSets::eraseBitSetMetadata(Module &M) {
  if (!BitSetNM)
    return false;

  M.eraseNamedMetadata(BitSetNM);
  return true;
}

bool LowerBitSets::runOnModule(Module &M) {
  bool Changed = buildBitSets(M);
  Changed |= eraseBitSetMetadata(M);
  return Changed;
}
