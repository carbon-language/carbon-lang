//===-- SeparateConstOffsetFromGEP.cpp - ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Loop unrolling may create many similar GEPs for array accesses.
// e.g., a 2-level loop
//
// float a[32][32]; // global variable
//
// for (int i = 0; i < 2; ++i) {
//   for (int j = 0; j < 2; ++j) {
//     ...
//     ... = a[x + i][y + j];
//     ...
//   }
// }
//
// will probably be unrolled to:
//
// gep %a, 0, %x, %y; load
// gep %a, 0, %x, %y + 1; load
// gep %a, 0, %x + 1, %y; load
// gep %a, 0, %x + 1, %y + 1; load
//
// LLVM's GVN does not use partial redundancy elimination yet, and is thus
// unable to reuse (gep %a, 0, %x, %y). As a result, this misoptimization incurs
// significant slowdown in targets with limited addressing modes. For instance,
// because the PTX target does not support the reg+reg addressing mode, the
// NVPTX backend emits PTX code that literally computes the pointer address of
// each GEP, wasting tons of registers. It emits the following PTX for the
// first load and similar PTX for other loads.
//
// mov.u32         %r1, %x;
// mov.u32         %r2, %y;
// mul.wide.u32    %rl2, %r1, 128;
// mov.u64         %rl3, a;
// add.s64         %rl4, %rl3, %rl2;
// mul.wide.u32    %rl5, %r2, 4;
// add.s64         %rl6, %rl4, %rl5;
// ld.global.f32   %f1, [%rl6];
//
// To reduce the register pressure, the optimization implemented in this file
// merges the common part of a group of GEPs, so we can compute each pointer
// address by adding a simple offset to the common part, saving many registers.
//
// It works by splitting each GEP into a variadic base and a constant offset.
// The variadic base can be computed once and reused by multiple GEPs, and the
// constant offsets can be nicely folded into the reg+immediate addressing mode
// (supported by most targets) without using any extra register.
//
// For instance, we transform the four GEPs and four loads in the above example
// into:
//
// base = gep a, 0, x, y
// load base
// laod base + 1  * sizeof(float)
// load base + 32 * sizeof(float)
// load base + 33 * sizeof(float)
//
// Given the transformed IR, a backend that supports the reg+immediate
// addressing mode can easily fold the pointer arithmetics into the loads. For
// example, the NVPTX backend can easily fold the pointer arithmetics into the
// ld.global.f32 instructions, and the resultant PTX uses much fewer registers.
//
// mov.u32         %r1, %tid.x;
// mov.u32         %r2, %tid.y;
// mul.wide.u32    %rl2, %r1, 128;
// mov.u64         %rl3, a;
// add.s64         %rl4, %rl3, %rl2;
// mul.wide.u32    %rl5, %r2, 4;
// add.s64         %rl6, %rl4, %rl5;
// ld.global.f32   %f1, [%rl6]; // so far the same as unoptimized PTX
// ld.global.f32   %f2, [%rl6+4]; // much better
// ld.global.f32   %f3, [%rl6+128]; // much better
// ld.global.f32   %f4, [%rl6+132]; // much better
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

static cl::opt<bool> DisableSeparateConstOffsetFromGEP(
    "disable-separate-const-offset-from-gep", cl::init(false),
    cl::desc("Do not separate the constant offset from a GEP instruction"),
    cl::Hidden);

namespace {

/// \brief A helper class for separating a constant offset from a GEP index.
///
/// In real programs, a GEP index may be more complicated than a simple addition
/// of something and a constant integer which can be trivially splitted. For
/// example, to split ((a << 3) | 5) + b, we need to search deeper for the
/// constant offset, so that we can seperate the index to (a << 3) + b and 5.
///
/// Therefore, this class looks into the expression that computes a given GEP
/// index, and tries to find a constant integer that can be hoisted to the
/// outermost level of the expression as an addition. Not every constant in an
/// expression can jump out. e.g., we cannot transform (b * (a + 5)) to (b * a +
/// 5); nor can we transform (3 * (a + 5)) to (3 * a + 5), however in this case,
/// -instcombine probably already optimized (3 * (a + 5)) to (3 * a + 15).
class ConstantOffsetExtractor {
 public:
  /// Extracts a constant offset from the given GEP index. It outputs the
  /// numeric value of the extracted constant offset (0 if failed), and a
  /// new index representing the remainder (equal to the original index minus
  /// the constant offset).
  /// \p Idx The given GEP index
  /// \p NewIdx The new index to replace
  /// \p DL The datalayout of the module
  /// \p IP Calculating the new index requires new instructions. IP indicates
  /// where to insert them (typically right before the GEP).
  static int64_t Extract(Value *Idx, Value *&NewIdx, const DataLayout *DL,
                         Instruction *IP);
  /// Looks for a constant offset without extracting it. The meaning of the
  /// arguments and the return value are the same as Extract.
  static int64_t Find(Value *Idx, const DataLayout *DL);

 private:
  ConstantOffsetExtractor(const DataLayout *Layout, Instruction *InsertionPt)
      : DL(Layout), IP(InsertionPt) {}
  /// Searches the expression that computes V for a constant offset. If the
  /// searching is successful, update UserChain as a path from V to the constant
  /// offset.
  int64_t find(Value *V);
  /// A helper function to look into both operands of a binary operator U.
  /// \p IsSub Whether U is a sub operator. If so, we need to negate the
  /// constant offset at some point.
  int64_t findInEitherOperand(User *U, bool IsSub);
  /// After finding the constant offset and how it is reached from the GEP
  /// index, we build a new index which is a clone of the old one except the
  /// constant offset is removed. For example, given (a + (b + 5)) and knowning
  /// the constant offset is 5, this function returns (a + b).
  ///
  /// We cannot simply change the constant to zero because the expression that
  /// computes the index or its intermediate result may be used by others.
  Value *rebuildWithoutConstantOffset();
  // A helper function for rebuildWithoutConstantOffset that rebuilds the direct
  // user (U) of the constant offset (C).
  Value *rebuildLeafWithoutConstantOffset(User *U, Value *C);
  /// Returns a clone of U except the first occurrence of From with To.
  Value *cloneAndReplace(User *U, Value *From, Value *To);

  /// Returns true if LHS and RHS have no bits in common, i.e., LHS | RHS == 0.
  bool NoCommonBits(Value *LHS, Value *RHS) const;
  /// Computes which bits are known to be one or zero.
  /// \p KnownOne Mask of all bits that are known to be one.
  /// \p KnownZero Mask of all bits that are known to be zero.
  void ComputeKnownBits(Value *V, APInt &KnownOne, APInt &KnownZero) const;
  /// Finds the first use of Used in U. Returns -1 if not found.
  static unsigned FindFirstUse(User *U, Value *Used);

  /// The path from the constant offset to the old GEP index. e.g., if the GEP
  /// index is "a * b + (c + 5)". After running function find, UserChain[0] will
  /// be the constant 5, UserChain[1] will be the subexpression "c + 5", and
  /// UserChain[2] will be the entire expression "a * b + (c + 5)".
  ///
  /// This path helps rebuildWithoutConstantOffset rebuild the new GEP index.
  SmallVector<User *, 8> UserChain;
  /// The data layout of the module. Used in ComputeKnownBits.
  const DataLayout *DL;
  Instruction *IP;  /// Insertion position of cloned instructions.
};

/// \brief A pass that tries to split every GEP in the function into a variadic
/// base and a constant offset. It is a FuntionPass because searching for the
/// constant offset may inspect other basic blocks.
class SeparateConstOffsetFromGEP : public FunctionPass {
 public:
  static char ID;
  SeparateConstOffsetFromGEP() : FunctionPass(ID) {
    initializeSeparateConstOffsetFromGEPPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DataLayoutPass>();
    AU.addRequired<TargetTransformInfo>();
  }
  bool runOnFunction(Function &F) override;

 private:
  /// Tries to split the given GEP into a variadic base and a constant offset,
  /// and returns true if the splitting succeeds.
  bool splitGEP(GetElementPtrInst *GEP);
  /// Finds the constant offset within each index, and accumulates them. This
  /// function only inspects the GEP without changing it. The output
  /// NeedsExtraction indicates whether we can extract a non-zero constant
  /// offset from any index.
  int64_t accumulateByteOffset(GetElementPtrInst *GEP, const DataLayout *DL,
                               bool &NeedsExtraction);
};
}  // anonymous namespace

char SeparateConstOffsetFromGEP::ID = 0;
INITIALIZE_PASS_BEGIN(
    SeparateConstOffsetFromGEP, "separate-const-offset-from-gep",
    "Split GEPs to a variadic base and a constant offset for better CSE", false,
    false)
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_DEPENDENCY(DataLayoutPass)
INITIALIZE_PASS_END(
    SeparateConstOffsetFromGEP, "separate-const-offset-from-gep",
    "Split GEPs to a variadic base and a constant offset for better CSE", false,
    false)

FunctionPass *llvm::createSeparateConstOffsetFromGEPPass() {
  return new SeparateConstOffsetFromGEP();
}

int64_t ConstantOffsetExtractor::findInEitherOperand(User *U, bool IsSub) {
  assert(U->getNumOperands() == 2);
  int64_t ConstantOffset = find(U->getOperand(0));
  // If we found a constant offset in the left operand, stop and return that.
  // This shortcut might cause us to miss opportunities of combining the
  // constant offsets in both operands, e.g., (a + 4) + (b + 5) => (a + b) + 9.
  // However, such cases are probably already handled by -instcombine,
  // given this pass runs after the standard optimizations.
  if (ConstantOffset != 0) return ConstantOffset;
  ConstantOffset = find(U->getOperand(1));
  // If U is a sub operator, negate the constant offset found in the right
  // operand.
  return IsSub ? -ConstantOffset : ConstantOffset;
}

int64_t ConstantOffsetExtractor::find(Value *V) {
  // TODO(jingyue): We can even trace into integer/pointer casts, such as
  // inttoptr, ptrtoint, bitcast, and addrspacecast. We choose to handle only
  // integers because it gives good enough results for our benchmarks.
  assert(V->getType()->isIntegerTy());

  User *U = dyn_cast<User>(V);
  // We cannot do much with Values that are not a User, such as BasicBlock and
  // MDNode.
  if (U == nullptr) return 0;

  int64_t ConstantOffset = 0;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(U)) {
    // Hooray, we found it!
    ConstantOffset = CI->getSExtValue();
  } else if (Operator *O = dyn_cast<Operator>(U)) {
    // The GEP index may be more complicated than a simple addition of a
    // varaible and a constant. Therefore, we trace into subexpressions for more
    // hoisting opportunities.
    switch (O->getOpcode()) {
      case Instruction::Add: {
        ConstantOffset = findInEitherOperand(U, false);
        break;
      }
      case Instruction::Sub: {
        ConstantOffset = findInEitherOperand(U, true);
        break;
      }
      case Instruction::Or: {
        // If LHS and RHS don't have common bits, (LHS | RHS) is equivalent to
        // (LHS + RHS).
        if (NoCommonBits(U->getOperand(0), U->getOperand(1)))
          ConstantOffset = findInEitherOperand(U, false);
        break;
      }
      case Instruction::SExt: {
        // For safety, we trace into sext only when its operand is marked
        // "nsw" because xxx.nsw guarantees no signed wrap. e.g., we can safely
        // transform "sext (add nsw a, 5)" into "add nsw (sext a), 5".
        if (BinaryOperator *BO = dyn_cast<BinaryOperator>(U->getOperand(0))) {
          if (BO->hasNoSignedWrap())
            ConstantOffset = find(U->getOperand(0));
        }
        break;
      }
      case Instruction::ZExt: {
        // Similarly, we trace into zext only when its operand is marked with
        // "nuw" because zext (add nuw a, b) == add nuw (zext a), (zext b).
        if (BinaryOperator *BO = dyn_cast<BinaryOperator>(U->getOperand(0))) {
          if (BO->hasNoUnsignedWrap())
            ConstantOffset = find(U->getOperand(0));
        }
        break;
      }
    }
  }
  // If we found a non-zero constant offset, adds it to the path for future
  // transformation (rebuildWithoutConstantOffset). Zero is a valid constant
  // offset, but doesn't help this optimization.
  if (ConstantOffset != 0)
    UserChain.push_back(U);
  return ConstantOffset;
}

unsigned ConstantOffsetExtractor::FindFirstUse(User *U, Value *Used) {
  for (unsigned I = 0, E = U->getNumOperands(); I < E; ++I) {
    if (U->getOperand(I) == Used)
      return I;
  }
  return -1;
}

Value *ConstantOffsetExtractor::cloneAndReplace(User *U, Value *From,
                                                Value *To) {
  // Finds in U the first use of From. It is safe to ignore future occurrences
  // of From, because findInEitherOperand similarly stops searching the right
  // operand when the first operand has a non-zero constant offset.
  unsigned OpNo = FindFirstUse(U, From);
  assert(OpNo != (unsigned)-1 && "UserChain wasn't built correctly");

  // ConstantOffsetExtractor::find only follows Operators (i.e., Instructions
  // and ConstantExprs). Therefore, U is either an Instruction or a
  // ConstantExpr.
  if (Instruction *I = dyn_cast<Instruction>(U)) {
    Instruction *Clone = I->clone();
    Clone->setOperand(OpNo, To);
    Clone->insertBefore(IP);
    return Clone;
  }
  // cast<Constant>(To) is safe because a ConstantExpr only uses Constants.
  return cast<ConstantExpr>(U)
      ->getWithOperandReplaced(OpNo, cast<Constant>(To));
}

Value *ConstantOffsetExtractor::rebuildLeafWithoutConstantOffset(User *U,
                                                                 Value *C) {
  assert(U->getNumOperands() <= 2 &&
         "We didn't trace into any operator with more than 2 operands");
  // If U has only one operand which is the constant offset, removing the
  // constant offset leaves U as a null value.
  if (U->getNumOperands() == 1)
    return Constant::getNullValue(U->getType());

  // U->getNumOperands() == 2
  unsigned OpNo = FindFirstUse(U, C); // U->getOperand(OpNo) == C
  assert(OpNo < 2 && "UserChain wasn't built correctly");
  Value *TheOther = U->getOperand(1 - OpNo); // The other operand of U
  // If U = C - X, removing C makes U = -X; otherwise U will simply be X.
  if (!isa<SubOperator>(U) || OpNo == 1)
    return TheOther;
  if (isa<ConstantExpr>(U))
    return ConstantExpr::getNeg(cast<Constant>(TheOther));
  return BinaryOperator::CreateNeg(TheOther, "", IP);
}

Value *ConstantOffsetExtractor::rebuildWithoutConstantOffset() {
  assert(UserChain.size() > 0 && "you at least found a constant, right?");
  // Start with the constant and go up through UserChain, each time building a
  // clone of the subexpression but with the constant removed.
  // e.g., to build a clone of (a + (b + (c + 5)) but with the 5 removed, we
  // first c, then (b + c), and finally (a + (b + c)).
  //
  // Fast path: if the GEP index is a constant, simply returns 0.
  if (UserChain.size() == 1)
    return ConstantInt::get(UserChain[0]->getType(), 0);

  Value *Remainder =
      rebuildLeafWithoutConstantOffset(UserChain[1], UserChain[0]);
  for (size_t I = 2; I < UserChain.size(); ++I)
    Remainder = cloneAndReplace(UserChain[I], UserChain[I - 1], Remainder);
  return Remainder;
}

int64_t ConstantOffsetExtractor::Extract(Value *Idx, Value *&NewIdx,
                                         const DataLayout *DL,
                                         Instruction *IP) {
  ConstantOffsetExtractor Extractor(DL, IP);
  // Find a non-zero constant offset first.
  int64_t ConstantOffset = Extractor.find(Idx);
  if (ConstantOffset == 0)
    return 0;
  // Then rebuild a new index with the constant removed.
  NewIdx = Extractor.rebuildWithoutConstantOffset();
  return ConstantOffset;
}

int64_t ConstantOffsetExtractor::Find(Value *Idx, const DataLayout *DL) {
  return ConstantOffsetExtractor(DL, nullptr).find(Idx);
}

void ConstantOffsetExtractor::ComputeKnownBits(Value *V, APInt &KnownOne,
                                               APInt &KnownZero) const {
  IntegerType *IT = cast<IntegerType>(V->getType());
  KnownOne = APInt(IT->getBitWidth(), 0);
  KnownZero = APInt(IT->getBitWidth(), 0);
  llvm::ComputeMaskedBits(V, KnownZero, KnownOne, DL, 0);
}

bool ConstantOffsetExtractor::NoCommonBits(Value *LHS, Value *RHS) const {
  assert(LHS->getType() == RHS->getType() &&
         "LHS and RHS should have the same type");
  APInt LHSKnownOne, LHSKnownZero, RHSKnownOne, RHSKnownZero;
  ComputeKnownBits(LHS, LHSKnownOne, LHSKnownZero);
  ComputeKnownBits(RHS, RHSKnownOne, RHSKnownZero);
  return (LHSKnownZero | RHSKnownZero).isAllOnesValue();
}

int64_t SeparateConstOffsetFromGEP::accumulateByteOffset(
    GetElementPtrInst *GEP, const DataLayout *DL, bool &NeedsExtraction) {
  NeedsExtraction = false;
  int64_t AccumulativeByteOffset = 0;
  gep_type_iterator GTI = gep_type_begin(*GEP);
  for (unsigned I = 1, E = GEP->getNumOperands(); I != E; ++I, ++GTI) {
    if (isa<SequentialType>(*GTI)) {
      // Tries to extract a constant offset from this GEP index.
      int64_t ConstantOffset =
          ConstantOffsetExtractor::Find(GEP->getOperand(I), DL);
      if (ConstantOffset != 0) {
        NeedsExtraction = true;
        // A GEP may have multiple indices.  We accumulate the extracted
        // constant offset to a byte offset, and later offset the remainder of
        // the original GEP with this byte offset.
        AccumulativeByteOffset +=
            ConstantOffset * DL->getTypeAllocSize(GTI.getIndexedType());
      }
    }
  }
  return AccumulativeByteOffset;
}

bool SeparateConstOffsetFromGEP::splitGEP(GetElementPtrInst *GEP) {
  // Skip vector GEPs.
  if (GEP->getType()->isVectorTy())
    return false;

  // The backend can already nicely handle the case where all indices are
  // constant.
  if (GEP->hasAllConstantIndices())
    return false;

  bool Changed = false;

  // Shortcuts integer casts. Eliminating these explicit casts can make
  // subsequent optimizations more obvious: ConstantOffsetExtractor needn't
  // trace into these casts.
  if (GEP->isInBounds()) {
    // Doing this to inbounds GEPs is safe because their indices are guaranteed
    // to be non-negative and in bounds.
    gep_type_iterator GTI = gep_type_begin(*GEP);
    for (unsigned I = 1, E = GEP->getNumOperands(); I != E; ++I, ++GTI) {
      if (isa<SequentialType>(*GTI)) {
        if (Operator *O = dyn_cast<Operator>(GEP->getOperand(I))) {
          if (O->getOpcode() == Instruction::SExt ||
              O->getOpcode() == Instruction::ZExt) {
            GEP->setOperand(I, O->getOperand(0));
            Changed = true;
          }
        }
      }
    }
  }

  const DataLayout *DL = &getAnalysis<DataLayoutPass>().getDataLayout();
  bool NeedsExtraction;
  int64_t AccumulativeByteOffset =
      accumulateByteOffset(GEP, DL, NeedsExtraction);

  if (!NeedsExtraction)
    return Changed;
  // Before really splitting the GEP, check whether the backend supports the
  // addressing mode we are about to produce. If no, this splitting probably
  // won't be beneficial.
  TargetTransformInfo &TTI = getAnalysis<TargetTransformInfo>();
  if (!TTI.isLegalAddressingMode(GEP->getType()->getElementType(),
                                 /*BaseGV=*/nullptr, AccumulativeByteOffset,
                                 /*HasBaseReg=*/true, /*Scale=*/0)) {
    return Changed;
  }

  // Remove the constant offset in each GEP index. The resultant GEP computes
  // the variadic base.
  gep_type_iterator GTI = gep_type_begin(*GEP);
  for (unsigned I = 1, E = GEP->getNumOperands(); I != E; ++I, ++GTI) {
    if (isa<SequentialType>(*GTI)) {
      Value *NewIdx = nullptr;
      // Tries to extract a constant offset from this GEP index.
      int64_t ConstantOffset =
          ConstantOffsetExtractor::Extract(GEP->getOperand(I), NewIdx, DL, GEP);
      if (ConstantOffset != 0) {
        assert(NewIdx && "ConstantOffset != 0 implies NewIdx is set");
        GEP->setOperand(I, NewIdx);
        // Clear the inbounds attribute because the new index may be off-bound.
        // e.g.,
        //
        // b = add i64 a, 5
        // addr = gep inbounds float* p, i64 b
        //
        // is transformed to:
        //
        // addr2 = gep float* p, i64 a
        // addr = gep float* addr2, i64 5
        //
        // If a is -4, although the old index b is in bounds, the new index a is
        // off-bound. http://llvm.org/docs/LangRef.html#id181 says "if the
        // inbounds keyword is not present, the offsets are added to the base
        // address with silently-wrapping two's complement arithmetic".
        // Therefore, the final code will be a semantically equivalent.
        //
        // TODO(jingyue): do some range analysis to keep as many inbounds as
        // possible. GEPs with inbounds are more friendly to alias analysis.
        GEP->setIsInBounds(false);
        Changed = true;
      }
    }
  }

  // Offsets the base with the accumulative byte offset.
  //
  //   %gep                        ; the base
  //   ... %gep ...
  //
  // => add the offset
  //
  //   %gep2                       ; clone of %gep
  //   %0       = ptrtoint %gep2
  //   %1       = add %0, <offset>
  //   %new.gep = inttoptr %1
  //   %gep                        ; will be removed
  //   ... %gep ...
  //
  // => replace all uses of %gep with %new.gep and remove %gep
  //
  //   %gep2                       ; clone of %gep
  //   %0       = ptrtoint %gep2
  //   %1       = add %0, <offset>
  //   %new.gep = inttoptr %1
  //   ... %new.gep ...
  //
  // TODO(jingyue): Emit a GEP instead of an "uglygep"
  // (http://llvm.org/docs/GetElementPtr.html#what-s-an-uglygep) to make the IR
  // prettier and more alias analysis friendly. One caveat: if the original GEP
  // ends with a StructType, we need to split the GEP at the last
  // SequentialType. For instance, consider the following IR:
  //
  //   %struct.S = type { float, double }
  //   @array = global [1024 x %struct.S]
  //   %p = getelementptr %array, 0, %i + 5, 1
  //
  // To separate the constant 5 from %p, we would need to split %p at the last
  // array index so that we have:
  //
  //   %addr = gep %array, 0, %i
  //   %p = gep %addr, 5, 1
  Instruction *NewGEP = GEP->clone();
  NewGEP->insertBefore(GEP);
  Type *IntPtrTy = DL->getIntPtrType(GEP->getType());
  Value *Addr = new PtrToIntInst(NewGEP, IntPtrTy, "", GEP);
  Addr = BinaryOperator::CreateAdd(
      Addr, ConstantInt::get(IntPtrTy, AccumulativeByteOffset, true), "", GEP);
  Addr = new IntToPtrInst(Addr, GEP->getType(), "", GEP);

  GEP->replaceAllUsesWith(Addr);
  GEP->eraseFromParent();

  return true;
}

bool SeparateConstOffsetFromGEP::runOnFunction(Function &F) {
  if (DisableSeparateConstOffsetFromGEP)
    return false;

  bool Changed = false;
  for (Function::iterator B = F.begin(), BE = F.end(); B != BE; ++B) {
    for (BasicBlock::iterator I = B->begin(), IE = B->end(); I != IE; ) {
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I++)) {
        Changed |= splitGEP(GEP);
      }
      // No need to split GEP ConstantExprs because all its indices are constant
      // already.
    }
  }
  return Changed;
}
