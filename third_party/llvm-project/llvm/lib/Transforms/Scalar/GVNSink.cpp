//===- GVNSink.cpp - sink expressions into successors ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file GVNSink.cpp
/// This pass attempts to sink instructions into successors, reducing static
/// instruction count and enabling if-conversion.
///
/// We use a variant of global value numbering to decide what can be sunk.
/// Consider:
///
/// [ %a1 = add i32 %b, 1  ]   [ %c1 = add i32 %d, 1  ]
/// [ %a2 = xor i32 %a1, 1 ]   [ %c2 = xor i32 %c1, 1 ]
///                  \           /
///            [ %e = phi i32 %a2, %c2 ]
///            [ add i32 %e, 4         ]
///
///
/// GVN would number %a1 and %c1 differently because they compute different
/// results - the VN of an instruction is a function of its opcode and the
/// transitive closure of its operands. This is the key property for hoisting
/// and CSE.
///
/// What we want when sinking however is for a numbering that is a function of
/// the *uses* of an instruction, which allows us to answer the question "if I
/// replace %a1 with %c1, will it contribute in an equivalent way to all
/// successive instructions?". The PostValueTable class in GVN provides this
/// mapping.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ArrayRecycler.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/GVNExpression.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "gvn-sink"

STATISTIC(NumRemoved, "Number of instructions removed");

namespace llvm {
namespace GVNExpression {

LLVM_DUMP_METHOD void Expression::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

} // end namespace GVNExpression
} // end namespace llvm

namespace {

static bool isMemoryInst(const Instruction *I) {
  return isa<LoadInst>(I) || isa<StoreInst>(I) ||
         (isa<InvokeInst>(I) && !cast<InvokeInst>(I)->doesNotAccessMemory()) ||
         (isa<CallInst>(I) && !cast<CallInst>(I)->doesNotAccessMemory());
}

/// Iterates through instructions in a set of blocks in reverse order from the
/// first non-terminator. For example (assume all blocks have size n):
///   LockstepReverseIterator I([B1, B2, B3]);
///   *I-- = [B1[n], B2[n], B3[n]];
///   *I-- = [B1[n-1], B2[n-1], B3[n-1]];
///   *I-- = [B1[n-2], B2[n-2], B3[n-2]];
///   ...
///
/// It continues until all blocks have been exhausted. Use \c getActiveBlocks()
/// to
/// determine which blocks are still going and the order they appear in the
/// list returned by operator*.
class LockstepReverseIterator {
  ArrayRef<BasicBlock *> Blocks;
  SmallSetVector<BasicBlock *, 4> ActiveBlocks;
  SmallVector<Instruction *, 4> Insts;
  bool Fail;

public:
  LockstepReverseIterator(ArrayRef<BasicBlock *> Blocks) : Blocks(Blocks) {
    reset();
  }

  void reset() {
    Fail = false;
    ActiveBlocks.clear();
    for (BasicBlock *BB : Blocks)
      ActiveBlocks.insert(BB);
    Insts.clear();
    for (BasicBlock *BB : Blocks) {
      if (BB->size() <= 1) {
        // Block wasn't big enough - only contained a terminator.
        ActiveBlocks.remove(BB);
        continue;
      }
      Insts.push_back(BB->getTerminator()->getPrevNode());
    }
    if (Insts.empty())
      Fail = true;
  }

  bool isValid() const { return !Fail; }
  ArrayRef<Instruction *> operator*() const { return Insts; }

  // Note: This needs to return a SmallSetVector as the elements of
  // ActiveBlocks will be later copied to Blocks using std::copy. The
  // resultant order of elements in Blocks needs to be deterministic.
  // Using SmallPtrSet instead causes non-deterministic order while
  // copying. And we cannot simply sort Blocks as they need to match the
  // corresponding Values.
  SmallSetVector<BasicBlock *, 4> &getActiveBlocks() { return ActiveBlocks; }

  void restrictToBlocks(SmallSetVector<BasicBlock *, 4> &Blocks) {
    for (auto II = Insts.begin(); II != Insts.end();) {
      if (!llvm::is_contained(Blocks, (*II)->getParent())) {
        ActiveBlocks.remove((*II)->getParent());
        II = Insts.erase(II);
      } else {
        ++II;
      }
    }
  }

  void operator--() {
    if (Fail)
      return;
    SmallVector<Instruction *, 4> NewInsts;
    for (auto *Inst : Insts) {
      if (Inst == &Inst->getParent()->front())
        ActiveBlocks.remove(Inst->getParent());
      else
        NewInsts.push_back(Inst->getPrevNode());
    }
    if (NewInsts.empty()) {
      Fail = true;
      return;
    }
    Insts = NewInsts;
  }
};

//===----------------------------------------------------------------------===//

/// Candidate solution for sinking. There may be different ways to
/// sink instructions, differing in the number of instructions sunk,
/// the number of predecessors sunk from and the number of PHIs
/// required.
struct SinkingInstructionCandidate {
  unsigned NumBlocks;
  unsigned NumInstructions;
  unsigned NumPHIs;
  unsigned NumMemoryInsts;
  int Cost = -1;
  SmallVector<BasicBlock *, 4> Blocks;

  void calculateCost(unsigned NumOrigPHIs, unsigned NumOrigBlocks) {
    unsigned NumExtraPHIs = NumPHIs - NumOrigPHIs;
    unsigned SplitEdgeCost = (NumOrigBlocks > NumBlocks) ? 2 : 0;
    Cost = (NumInstructions * (NumBlocks - 1)) -
           (NumExtraPHIs *
            NumExtraPHIs) // PHIs are expensive, so make sure they're worth it.
           - SplitEdgeCost;
  }

  bool operator>(const SinkingInstructionCandidate &Other) const {
    return Cost > Other.Cost;
  }
};

#ifndef NDEBUG
raw_ostream &operator<<(raw_ostream &OS, const SinkingInstructionCandidate &C) {
  OS << "<Candidate Cost=" << C.Cost << " #Blocks=" << C.NumBlocks
     << " #Insts=" << C.NumInstructions << " #PHIs=" << C.NumPHIs << ">";
  return OS;
}
#endif

//===----------------------------------------------------------------------===//

/// Describes a PHI node that may or may not exist. These track the PHIs
/// that must be created if we sunk a sequence of instructions. It provides
/// a hash function for efficient equality comparisons.
class ModelledPHI {
  SmallVector<Value *, 4> Values;
  SmallVector<BasicBlock *, 4> Blocks;

public:
  ModelledPHI() = default;

  ModelledPHI(const PHINode *PN) {
    // BasicBlock comes first so we sort by basic block pointer order, then by value pointer order.
    SmallVector<std::pair<BasicBlock *, Value *>, 4> Ops;
    for (unsigned I = 0, E = PN->getNumIncomingValues(); I != E; ++I)
      Ops.push_back({PN->getIncomingBlock(I), PN->getIncomingValue(I)});
    llvm::sort(Ops);
    for (auto &P : Ops) {
      Blocks.push_back(P.first);
      Values.push_back(P.second);
    }
  }

  /// Create a dummy ModelledPHI that will compare unequal to any other ModelledPHI
  /// without the same ID.
  /// \note This is specifically for DenseMapInfo - do not use this!
  static ModelledPHI createDummy(size_t ID) {
    ModelledPHI M;
    M.Values.push_back(reinterpret_cast<Value*>(ID));
    return M;
  }

  /// Create a PHI from an array of incoming values and incoming blocks.
  template <typename VArray, typename BArray>
  ModelledPHI(const VArray &V, const BArray &B) {
    llvm::copy(V, std::back_inserter(Values));
    llvm::copy(B, std::back_inserter(Blocks));
  }

  /// Create a PHI from [I[OpNum] for I in Insts].
  template <typename BArray>
  ModelledPHI(ArrayRef<Instruction *> Insts, unsigned OpNum, const BArray &B) {
    llvm::copy(B, std::back_inserter(Blocks));
    for (auto *I : Insts)
      Values.push_back(I->getOperand(OpNum));
  }

  /// Restrict the PHI's contents down to only \c NewBlocks.
  /// \c NewBlocks must be a subset of \c this->Blocks.
  void restrictToBlocks(const SmallSetVector<BasicBlock *, 4> &NewBlocks) {
    auto BI = Blocks.begin();
    auto VI = Values.begin();
    while (BI != Blocks.end()) {
      assert(VI != Values.end());
      if (!llvm::is_contained(NewBlocks, *BI)) {
        BI = Blocks.erase(BI);
        VI = Values.erase(VI);
      } else {
        ++BI;
        ++VI;
      }
    }
    assert(Blocks.size() == NewBlocks.size());
  }

  ArrayRef<Value *> getValues() const { return Values; }

  bool areAllIncomingValuesSame() const {
    return llvm::all_of(Values, [&](Value *V) { return V == Values[0]; });
  }

  bool areAllIncomingValuesSameType() const {
    return llvm::all_of(
        Values, [&](Value *V) { return V->getType() == Values[0]->getType(); });
  }

  bool areAnyIncomingValuesConstant() const {
    return llvm::any_of(Values, [&](Value *V) { return isa<Constant>(V); });
  }

  // Hash functor
  unsigned hash() const {
      return (unsigned)hash_combine_range(Values.begin(), Values.end());
  }

  bool operator==(const ModelledPHI &Other) const {
    return Values == Other.Values && Blocks == Other.Blocks;
  }
};

template <typename ModelledPHI> struct DenseMapInfo {
  static inline ModelledPHI &getEmptyKey() {
    static ModelledPHI Dummy = ModelledPHI::createDummy(0);
    return Dummy;
  }

  static inline ModelledPHI &getTombstoneKey() {
    static ModelledPHI Dummy = ModelledPHI::createDummy(1);
    return Dummy;
  }

  static unsigned getHashValue(const ModelledPHI &V) { return V.hash(); }

  static bool isEqual(const ModelledPHI &LHS, const ModelledPHI &RHS) {
    return LHS == RHS;
  }
};

using ModelledPHISet = DenseSet<ModelledPHI, DenseMapInfo<ModelledPHI>>;

//===----------------------------------------------------------------------===//
//                             ValueTable
//===----------------------------------------------------------------------===//
// This is a value number table where the value number is a function of the
// *uses* of a value, rather than its operands. Thus, if VN(A) == VN(B) we know
// that the program would be equivalent if we replaced A with PHI(A, B).
//===----------------------------------------------------------------------===//

/// A GVN expression describing how an instruction is used. The operands
/// field of BasicExpression is used to store uses, not operands.
///
/// This class also contains fields for discriminators used when determining
/// equivalence of instructions with sideeffects.
class InstructionUseExpr : public GVNExpression::BasicExpression {
  unsigned MemoryUseOrder = -1;
  bool Volatile = false;
  ArrayRef<int> ShuffleMask;

public:
  InstructionUseExpr(Instruction *I, ArrayRecycler<Value *> &R,
                     BumpPtrAllocator &A)
      : GVNExpression::BasicExpression(I->getNumUses()) {
    allocateOperands(R, A);
    setOpcode(I->getOpcode());
    setType(I->getType());

    if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(I))
      ShuffleMask = SVI->getShuffleMask().copy(A);

    for (auto &U : I->uses())
      op_push_back(U.getUser());
    llvm::sort(op_begin(), op_end());
  }

  void setMemoryUseOrder(unsigned MUO) { MemoryUseOrder = MUO; }
  void setVolatile(bool V) { Volatile = V; }

  hash_code getHashValue() const override {
    return hash_combine(GVNExpression::BasicExpression::getHashValue(),
                        MemoryUseOrder, Volatile, ShuffleMask);
  }

  template <typename Function> hash_code getHashValue(Function MapFn) {
    hash_code H = hash_combine(getOpcode(), getType(), MemoryUseOrder, Volatile,
                               ShuffleMask);
    for (auto *V : operands())
      H = hash_combine(H, MapFn(V));
    return H;
  }
};

class ValueTable {
  DenseMap<Value *, uint32_t> ValueNumbering;
  DenseMap<GVNExpression::Expression *, uint32_t> ExpressionNumbering;
  DenseMap<size_t, uint32_t> HashNumbering;
  BumpPtrAllocator Allocator;
  ArrayRecycler<Value *> Recycler;
  uint32_t nextValueNumber = 1;

  /// Create an expression for I based on its opcode and its uses. If I
  /// touches or reads memory, the expression is also based upon its memory
  /// order - see \c getMemoryUseOrder().
  InstructionUseExpr *createExpr(Instruction *I) {
    InstructionUseExpr *E =
        new (Allocator) InstructionUseExpr(I, Recycler, Allocator);
    if (isMemoryInst(I))
      E->setMemoryUseOrder(getMemoryUseOrder(I));

    if (CmpInst *C = dyn_cast<CmpInst>(I)) {
      CmpInst::Predicate Predicate = C->getPredicate();
      E->setOpcode((C->getOpcode() << 8) | Predicate);
    }
    return E;
  }

  /// Helper to compute the value number for a memory instruction
  /// (LoadInst/StoreInst), including checking the memory ordering and
  /// volatility.
  template <class Inst> InstructionUseExpr *createMemoryExpr(Inst *I) {
    if (isStrongerThanUnordered(I->getOrdering()) || I->isAtomic())
      return nullptr;
    InstructionUseExpr *E = createExpr(I);
    E->setVolatile(I->isVolatile());
    return E;
  }

public:
  ValueTable() = default;

  /// Returns the value number for the specified value, assigning
  /// it a new number if it did not have one before.
  uint32_t lookupOrAdd(Value *V) {
    auto VI = ValueNumbering.find(V);
    if (VI != ValueNumbering.end())
      return VI->second;

    if (!isa<Instruction>(V)) {
      ValueNumbering[V] = nextValueNumber;
      return nextValueNumber++;
    }

    Instruction *I = cast<Instruction>(V);
    InstructionUseExpr *exp = nullptr;
    switch (I->getOpcode()) {
    case Instruction::Load:
      exp = createMemoryExpr(cast<LoadInst>(I));
      break;
    case Instruction::Store:
      exp = createMemoryExpr(cast<StoreInst>(I));
      break;
    case Instruction::Call:
    case Instruction::Invoke:
    case Instruction::FNeg:
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::ICmp:
    case Instruction::FCmp:
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::BitCast:
    case Instruction::AddrSpaceCast:
    case Instruction::Select:
    case Instruction::ExtractElement:
    case Instruction::InsertElement:
    case Instruction::ShuffleVector:
    case Instruction::InsertValue:
    case Instruction::GetElementPtr:
      exp = createExpr(I);
      break;
    default:
      break;
    }

    if (!exp) {
      ValueNumbering[V] = nextValueNumber;
      return nextValueNumber++;
    }

    uint32_t e = ExpressionNumbering[exp];
    if (!e) {
      hash_code H = exp->getHashValue([=](Value *V) { return lookupOrAdd(V); });
      auto I = HashNumbering.find(H);
      if (I != HashNumbering.end()) {
        e = I->second;
      } else {
        e = nextValueNumber++;
        HashNumbering[H] = e;
        ExpressionNumbering[exp] = e;
      }
    }
    ValueNumbering[V] = e;
    return e;
  }

  /// Returns the value number of the specified value. Fails if the value has
  /// not yet been numbered.
  uint32_t lookup(Value *V) const {
    auto VI = ValueNumbering.find(V);
    assert(VI != ValueNumbering.end() && "Value not numbered?");
    return VI->second;
  }

  /// Removes all value numberings and resets the value table.
  void clear() {
    ValueNumbering.clear();
    ExpressionNumbering.clear();
    HashNumbering.clear();
    Recycler.clear(Allocator);
    nextValueNumber = 1;
  }

  /// \c Inst uses or touches memory. Return an ID describing the memory state
  /// at \c Inst such that if getMemoryUseOrder(I1) == getMemoryUseOrder(I2),
  /// the exact same memory operations happen after I1 and I2.
  ///
  /// This is a very hard problem in general, so we use domain-specific
  /// knowledge that we only ever check for equivalence between blocks sharing a
  /// single immediate successor that is common, and when determining if I1 ==
  /// I2 we will have already determined that next(I1) == next(I2). This
  /// inductive property allows us to simply return the value number of the next
  /// instruction that defines memory.
  uint32_t getMemoryUseOrder(Instruction *Inst) {
    auto *BB = Inst->getParent();
    for (auto I = std::next(Inst->getIterator()), E = BB->end();
         I != E && !I->isTerminator(); ++I) {
      if (!isMemoryInst(&*I))
        continue;
      if (isa<LoadInst>(&*I))
        continue;
      CallInst *CI = dyn_cast<CallInst>(&*I);
      if (CI && CI->onlyReadsMemory())
        continue;
      InvokeInst *II = dyn_cast<InvokeInst>(&*I);
      if (II && II->onlyReadsMemory())
        continue;
      return lookupOrAdd(&*I);
    }
    return 0;
  }
};

//===----------------------------------------------------------------------===//

class GVNSink {
public:
  GVNSink() = default;

  bool run(Function &F) {
    LLVM_DEBUG(dbgs() << "GVNSink: running on function @" << F.getName()
                      << "\n");

    unsigned NumSunk = 0;
    ReversePostOrderTraversal<Function*> RPOT(&F);
    for (auto *N : RPOT)
      NumSunk += sinkBB(N);

    return NumSunk > 0;
  }

private:
  ValueTable VN;

  bool shouldAvoidSinkingInstruction(Instruction *I) {
    // These instructions may change or break semantics if moved.
    if (isa<PHINode>(I) || I->isEHPad() || isa<AllocaInst>(I) ||
        I->getType()->isTokenTy())
      return true;
    return false;
  }

  /// The main heuristic function. Analyze the set of instructions pointed to by
  /// LRI and return a candidate solution if these instructions can be sunk, or
  /// None otherwise.
  Optional<SinkingInstructionCandidate> analyzeInstructionForSinking(
      LockstepReverseIterator &LRI, unsigned &InstNum, unsigned &MemoryInstNum,
      ModelledPHISet &NeededPHIs, SmallPtrSetImpl<Value *> &PHIContents);

  /// Create a ModelledPHI for each PHI in BB, adding to PHIs.
  void analyzeInitialPHIs(BasicBlock *BB, ModelledPHISet &PHIs,
                          SmallPtrSetImpl<Value *> &PHIContents) {
    for (PHINode &PN : BB->phis()) {
      auto MPHI = ModelledPHI(&PN);
      PHIs.insert(MPHI);
      for (auto *V : MPHI.getValues())
        PHIContents.insert(V);
    }
  }

  /// The main instruction sinking driver. Set up state and try and sink
  /// instructions into BBEnd from its predecessors.
  unsigned sinkBB(BasicBlock *BBEnd);

  /// Perform the actual mechanics of sinking an instruction from Blocks into
  /// BBEnd, which is their only successor.
  void sinkLastInstruction(ArrayRef<BasicBlock *> Blocks, BasicBlock *BBEnd);

  /// Remove PHIs that all have the same incoming value.
  void foldPointlessPHINodes(BasicBlock *BB) {
    auto I = BB->begin();
    while (PHINode *PN = dyn_cast<PHINode>(I++)) {
      if (!llvm::all_of(PN->incoming_values(), [&](const Value *V) {
            return V == PN->getIncomingValue(0);
          }))
        continue;
      if (PN->getIncomingValue(0) != PN)
        PN->replaceAllUsesWith(PN->getIncomingValue(0));
      else
        PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
      PN->eraseFromParent();
    }
  }
};

Optional<SinkingInstructionCandidate> GVNSink::analyzeInstructionForSinking(
  LockstepReverseIterator &LRI, unsigned &InstNum, unsigned &MemoryInstNum,
  ModelledPHISet &NeededPHIs, SmallPtrSetImpl<Value *> &PHIContents) {
  auto Insts = *LRI;
  LLVM_DEBUG(dbgs() << " -- Analyzing instruction set: [\n"; for (auto *I
                                                                  : Insts) {
    I->dump();
  } dbgs() << " ]\n";);

  DenseMap<uint32_t, unsigned> VNums;
  for (auto *I : Insts) {
    uint32_t N = VN.lookupOrAdd(I);
    LLVM_DEBUG(dbgs() << " VN=" << Twine::utohexstr(N) << " for" << *I << "\n");
    if (N == ~0U)
      return None;
    VNums[N]++;
  }
  unsigned VNumToSink =
      std::max_element(VNums.begin(), VNums.end(),
                       [](const std::pair<uint32_t, unsigned> &I,
                          const std::pair<uint32_t, unsigned> &J) {
                         return I.second < J.second;
                       })
          ->first;

  if (VNums[VNumToSink] == 1)
    // Can't sink anything!
    return None;

  // Now restrict the number of incoming blocks down to only those with
  // VNumToSink.
  auto &ActivePreds = LRI.getActiveBlocks();
  unsigned InitialActivePredSize = ActivePreds.size();
  SmallVector<Instruction *, 4> NewInsts;
  for (auto *I : Insts) {
    if (VN.lookup(I) != VNumToSink)
      ActivePreds.remove(I->getParent());
    else
      NewInsts.push_back(I);
  }
  for (auto *I : NewInsts)
    if (shouldAvoidSinkingInstruction(I))
      return None;

  // If we've restricted the incoming blocks, restrict all needed PHIs also
  // to that set.
  bool RecomputePHIContents = false;
  if (ActivePreds.size() != InitialActivePredSize) {
    ModelledPHISet NewNeededPHIs;
    for (auto P : NeededPHIs) {
      P.restrictToBlocks(ActivePreds);
      NewNeededPHIs.insert(P);
    }
    NeededPHIs = NewNeededPHIs;
    LRI.restrictToBlocks(ActivePreds);
    RecomputePHIContents = true;
  }

  // The sunk instruction's results.
  ModelledPHI NewPHI(NewInsts, ActivePreds);

  // Does sinking this instruction render previous PHIs redundant?
  if (NeededPHIs.erase(NewPHI))
    RecomputePHIContents = true;

  if (RecomputePHIContents) {
    // The needed PHIs have changed, so recompute the set of all needed
    // values.
    PHIContents.clear();
    for (auto &PHI : NeededPHIs)
      PHIContents.insert(PHI.getValues().begin(), PHI.getValues().end());
  }

  // Is this instruction required by a later PHI that doesn't match this PHI?
  // if so, we can't sink this instruction.
  for (auto *V : NewPHI.getValues())
    if (PHIContents.count(V))
      // V exists in this PHI, but the whole PHI is different to NewPHI
      // (else it would have been removed earlier). We cannot continue
      // because this isn't representable.
      return None;

  // Which operands need PHIs?
  // FIXME: If any of these fail, we should partition up the candidates to
  // try and continue making progress.
  Instruction *I0 = NewInsts[0];

  // If all instructions that are going to participate don't have the same
  // number of operands, we can't do any useful PHI analysis for all operands.
  auto hasDifferentNumOperands = [&I0](Instruction *I) {
    return I->getNumOperands() != I0->getNumOperands();
  };
  if (any_of(NewInsts, hasDifferentNumOperands))
    return None;

  for (unsigned OpNum = 0, E = I0->getNumOperands(); OpNum != E; ++OpNum) {
    ModelledPHI PHI(NewInsts, OpNum, ActivePreds);
    if (PHI.areAllIncomingValuesSame())
      continue;
    if (!canReplaceOperandWithVariable(I0, OpNum))
      // We can 't create a PHI from this instruction!
      return None;
    if (NeededPHIs.count(PHI))
      continue;
    if (!PHI.areAllIncomingValuesSameType())
      return None;
    // Don't create indirect calls! The called value is the final operand.
    if ((isa<CallInst>(I0) || isa<InvokeInst>(I0)) && OpNum == E - 1 &&
        PHI.areAnyIncomingValuesConstant())
      return None;

    NeededPHIs.reserve(NeededPHIs.size());
    NeededPHIs.insert(PHI);
    PHIContents.insert(PHI.getValues().begin(), PHI.getValues().end());
  }

  if (isMemoryInst(NewInsts[0]))
    ++MemoryInstNum;

  SinkingInstructionCandidate Cand;
  Cand.NumInstructions = ++InstNum;
  Cand.NumMemoryInsts = MemoryInstNum;
  Cand.NumBlocks = ActivePreds.size();
  Cand.NumPHIs = NeededPHIs.size();
  append_range(Cand.Blocks, ActivePreds);

  return Cand;
}

unsigned GVNSink::sinkBB(BasicBlock *BBEnd) {
  LLVM_DEBUG(dbgs() << "GVNSink: running on basic block ";
             BBEnd->printAsOperand(dbgs()); dbgs() << "\n");
  SmallVector<BasicBlock *, 4> Preds;
  for (auto *B : predecessors(BBEnd)) {
    auto *T = B->getTerminator();
    if (isa<BranchInst>(T) || isa<SwitchInst>(T))
      Preds.push_back(B);
    else
      return 0;
  }
  if (Preds.size() < 2)
    return 0;
  llvm::sort(Preds);

  unsigned NumOrigPreds = Preds.size();
  // We can only sink instructions through unconditional branches.
  for (auto I = Preds.begin(); I != Preds.end();) {
    if ((*I)->getTerminator()->getNumSuccessors() != 1)
      I = Preds.erase(I);
    else
      ++I;
  }

  LockstepReverseIterator LRI(Preds);
  SmallVector<SinkingInstructionCandidate, 4> Candidates;
  unsigned InstNum = 0, MemoryInstNum = 0;
  ModelledPHISet NeededPHIs;
  SmallPtrSet<Value *, 4> PHIContents;
  analyzeInitialPHIs(BBEnd, NeededPHIs, PHIContents);
  unsigned NumOrigPHIs = NeededPHIs.size();

  while (LRI.isValid()) {
    auto Cand = analyzeInstructionForSinking(LRI, InstNum, MemoryInstNum,
                                             NeededPHIs, PHIContents);
    if (!Cand)
      break;
    Cand->calculateCost(NumOrigPHIs, Preds.size());
    Candidates.emplace_back(*Cand);
    --LRI;
  }

  llvm::stable_sort(Candidates, std::greater<SinkingInstructionCandidate>());
  LLVM_DEBUG(dbgs() << " -- Sinking candidates:\n"; for (auto &C
                                                         : Candidates) dbgs()
                                                    << "  " << C << "\n";);

  // Pick the top candidate, as long it is positive!
  if (Candidates.empty() || Candidates.front().Cost <= 0)
    return 0;
  auto C = Candidates.front();

  LLVM_DEBUG(dbgs() << " -- Sinking: " << C << "\n");
  BasicBlock *InsertBB = BBEnd;
  if (C.Blocks.size() < NumOrigPreds) {
    LLVM_DEBUG(dbgs() << " -- Splitting edge to ";
               BBEnd->printAsOperand(dbgs()); dbgs() << "\n");
    InsertBB = SplitBlockPredecessors(BBEnd, C.Blocks, ".gvnsink.split");
    if (!InsertBB) {
      LLVM_DEBUG(dbgs() << " -- FAILED to split edge!\n");
      // Edge couldn't be split.
      return 0;
    }
  }

  for (unsigned I = 0; I < C.NumInstructions; ++I)
    sinkLastInstruction(C.Blocks, InsertBB);

  return C.NumInstructions;
}

void GVNSink::sinkLastInstruction(ArrayRef<BasicBlock *> Blocks,
                                  BasicBlock *BBEnd) {
  SmallVector<Instruction *, 4> Insts;
  for (BasicBlock *BB : Blocks)
    Insts.push_back(BB->getTerminator()->getPrevNode());
  Instruction *I0 = Insts.front();

  SmallVector<Value *, 4> NewOperands;
  for (unsigned O = 0, E = I0->getNumOperands(); O != E; ++O) {
    bool NeedPHI = llvm::any_of(Insts, [&I0, O](const Instruction *I) {
      return I->getOperand(O) != I0->getOperand(O);
    });
    if (!NeedPHI) {
      NewOperands.push_back(I0->getOperand(O));
      continue;
    }

    // Create a new PHI in the successor block and populate it.
    auto *Op = I0->getOperand(O);
    assert(!Op->getType()->isTokenTy() && "Can't PHI tokens!");
    auto *PN = PHINode::Create(Op->getType(), Insts.size(),
                               Op->getName() + ".sink", &BBEnd->front());
    for (auto *I : Insts)
      PN->addIncoming(I->getOperand(O), I->getParent());
    NewOperands.push_back(PN);
  }

  // Arbitrarily use I0 as the new "common" instruction; remap its operands
  // and move it to the start of the successor block.
  for (unsigned O = 0, E = I0->getNumOperands(); O != E; ++O)
    I0->getOperandUse(O).set(NewOperands[O]);
  I0->moveBefore(&*BBEnd->getFirstInsertionPt());

  // Update metadata and IR flags.
  for (auto *I : Insts)
    if (I != I0) {
      combineMetadataForCSE(I0, I, true);
      I0->andIRFlags(I);
    }

  for (auto *I : Insts)
    if (I != I0)
      I->replaceAllUsesWith(I0);
  foldPointlessPHINodes(BBEnd);

  // Finally nuke all instructions apart from the common instruction.
  for (auto *I : Insts)
    if (I != I0)
      I->eraseFromParent();

  NumRemoved += Insts.size() - 1;
}

////////////////////////////////////////////////////////////////////////////////
// Pass machinery / boilerplate

class GVNSinkLegacyPass : public FunctionPass {
public:
  static char ID;

  GVNSinkLegacyPass() : FunctionPass(ID) {
    initializeGVNSinkLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    GVNSink G;
    return G.run(F);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};

} // end anonymous namespace

PreservedAnalyses GVNSinkPass::run(Function &F, FunctionAnalysisManager &AM) {
  GVNSink G;
  if (!G.run(F))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

char GVNSinkLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(GVNSinkLegacyPass, "gvn-sink",
                      "Early GVN sinking of Expressions", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(GVNSinkLegacyPass, "gvn-sink",
                    "Early GVN sinking of Expressions", false, false)

FunctionPass *llvm::createGVNSinkPass() { return new GVNSinkLegacyPass(); }
