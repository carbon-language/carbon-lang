//===- MergeFunctions.cpp - Merge identical functions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass looks for equivalent functions that are mergable and folds them.
//
// A hash is computed from the function, based on its type and number of
// basic blocks.
//
// Once all hashes are computed, we perform an expensive equality comparison
// on each function pair. This takes n^2/2 comparisons per bucket, so it's
// important that the hash function be high quality. The equality comparison
// iterates through each instruction in each basic block.
//
// When a match is found the functions are folded. If both functions are
// overridable, we move the functionality into a new internal function and
// leave two overridable thunks to it.
//
//===----------------------------------------------------------------------===//
//
// Future work:
//
// * virtual functions.
//
// Many functions have their address taken by the virtual function table for
// the object they belong to. However, as long as it's only used for a lookup
// and call, this is irrelevant, and we'd like to fold such functions.
//
// * switch from n^2 pair-wise comparisons to an n-way comparison for each
// bucket.
//
// * be smarter about bitcasts.
//
// In order to fold functions, we will sometimes add either bitcast instructions
// or bitcast constant expressions. Unfortunately, this can confound further
// analysis since the two functions differ where one has a bitcast and the
// other doesn't. We should learn to look through bitcasts.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "mergefunc"

STATISTIC(NumFunctionsMerged, "Number of functions merged");
STATISTIC(NumThunksWritten, "Number of thunks generated");
STATISTIC(NumAliasesWritten, "Number of aliases generated");
STATISTIC(NumDoubleWeak, "Number of new functions created");

/// Returns the type id for a type to be hashed. We turn pointer types into
/// integers here because the actual compare logic below considers pointers and
/// integers of the same size as equal.
static Type::TypeID getTypeIDForHash(Type *Ty) {
  if (Ty->isPointerTy())
    return Type::IntegerTyID;
  return Ty->getTypeID();
}

/// Creates a hash-code for the function which is the same for any two
/// functions that will compare equal, without looking at the instructions
/// inside the function.
static unsigned profileFunction(const Function *F) {
  FunctionType *FTy = F->getFunctionType();

  FoldingSetNodeID ID;
  ID.AddInteger(F->size());
  ID.AddInteger(F->getCallingConv());
  ID.AddBoolean(F->hasGC());
  ID.AddBoolean(FTy->isVarArg());
  ID.AddInteger(getTypeIDForHash(FTy->getReturnType()));
  for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
    ID.AddInteger(getTypeIDForHash(FTy->getParamType(i)));
  return ID.ComputeHash();
}

namespace {

/// ComparableFunction - A struct that pairs together functions with a
/// DataLayout so that we can keep them together as elements in the DenseSet.
class ComparableFunction {
public:
  static const ComparableFunction EmptyKey;
  static const ComparableFunction TombstoneKey;
  static DataLayout * const LookupOnly;

  ComparableFunction(Function *Func, const DataLayout *DL)
    : Func(Func), Hash(profileFunction(Func)), DL(DL) {}

  Function *getFunc() const { return Func; }
  unsigned getHash() const { return Hash; }
  const DataLayout *getDataLayout() const { return DL; }

  // Drops AssertingVH reference to the function. Outside of debug mode, this
  // does nothing.
  void release() {
    assert(Func &&
           "Attempted to release function twice, or release empty/tombstone!");
    Func = nullptr;
  }

private:
  explicit ComparableFunction(unsigned Hash)
    : Func(nullptr), Hash(Hash), DL(nullptr) {}

  AssertingVH<Function> Func;
  unsigned Hash;
  const DataLayout *DL;
};

const ComparableFunction ComparableFunction::EmptyKey = ComparableFunction(0);
const ComparableFunction ComparableFunction::TombstoneKey =
    ComparableFunction(1);
DataLayout *const ComparableFunction::LookupOnly = (DataLayout*)(-1);

}

namespace llvm {
  template <>
  struct DenseMapInfo<ComparableFunction> {
    static ComparableFunction getEmptyKey() {
      return ComparableFunction::EmptyKey;
    }
    static ComparableFunction getTombstoneKey() {
      return ComparableFunction::TombstoneKey;
    }
    static unsigned getHashValue(const ComparableFunction &CF) {
      return CF.getHash();
    }
    static bool isEqual(const ComparableFunction &LHS,
                        const ComparableFunction &RHS);
  };
}

namespace {

/// FunctionComparator - Compares two functions to determine whether or not
/// they will generate machine code with the same behaviour. DataLayout is
/// used if available. The comparator always fails conservatively (erring on the
/// side of claiming that two functions are different).
class FunctionComparator {
public:
  FunctionComparator(const DataLayout *DL, const Function *F1,
                     const Function *F2)
    : F1(F1), F2(F2), DL(DL) {}

  /// Test whether the two functions have equivalent behaviour.
  bool compare();

private:
  /// Test whether two basic blocks have equivalent behaviour.
  bool compare(const BasicBlock *BB1, const BasicBlock *BB2);

  /// Constants comparison.
  /// Its analog to lexicographical comparison between hypothetical numbers
  /// of next format:
  /// <bitcastability-trait><raw-bit-contents>
  ///
  /// 1. Bitcastability.
  /// Check whether L's type could be losslessly bitcasted to R's type.
  /// On this stage method, in case when lossless bitcast is not possible
  /// method returns -1 or 1, thus also defining which type is greater in
  /// context of bitcastability.
  /// Stage 0: If types are equal in terms of cmpTypes, then we can go straight
  ///          to the contents comparison.
  ///          If types differ, remember types comparison result and check
  ///          whether we still can bitcast types.
  /// Stage 1: Types that satisfies isFirstClassType conditions are always
  ///          greater then others.
  /// Stage 2: Vector is greater then non-vector.
  ///          If both types are vectors, then vector with greater bitwidth is
  ///          greater.
  ///          If both types are vectors with the same bitwidth, then types
  ///          are bitcastable, and we can skip other stages, and go to contents
  ///          comparison.
  /// Stage 3: Pointer types are greater than non-pointers. If both types are
  ///          pointers of the same address space - go to contents comparison.
  ///          Different address spaces: pointer with greater address space is
  ///          greater.
  /// Stage 4: Types are neither vectors, nor pointers. And they differ.
  ///          We don't know how to bitcast them. So, we better don't do it,
  ///          and return types comparison result (so it determines the
  ///          relationship among constants we don't know how to bitcast).
  ///
  /// Just for clearance, let's see how the set of constants could look
  /// on single dimension axis:
  ///
  /// [NFCT], [FCT, "others"], [FCT, pointers], [FCT, vectors]
  /// Where: NFCT - Not a FirstClassType
  ///        FCT - FirstClassTyp:
  ///
  /// 2. Compare raw contents.
  /// It ignores types on this stage and only compares bits from L and R.
  /// Returns 0, if L and R has equivalent contents.
  /// -1 or 1 if values are different.
  /// Pretty trivial:
  /// 2.1. If contents are numbers, compare numbers.
  ///    Ints with greater bitwidth are greater. Ints with same bitwidths
  ///    compared by their contents.
  /// 2.2. "And so on". Just to avoid discrepancies with comments
  /// perhaps it would be better to read the implementation itself.
  /// 3. And again about overall picture. Let's look back at how the ordered set
  /// of constants will look like:
  /// [NFCT], [FCT, "others"], [FCT, pointers], [FCT, vectors]
  ///
  /// Now look, what could be inside [FCT, "others"], for example:
  /// [FCT, "others"] =
  /// [
  ///   [double 0.1], [double 1.23],
  ///   [i32 1], [i32 2],
  ///   { double 1.0 },       ; StructTyID, NumElements = 1
  ///   { i32 1 },            ; StructTyID, NumElements = 1
  ///   { double 1, i32 1 },  ; StructTyID, NumElements = 2
  ///   { i32 1, double 1 }   ; StructTyID, NumElements = 2
  /// ]
  ///
  /// Let's explain the order. Float numbers will be less than integers, just
  /// because of cmpType terms: FloatTyID < IntegerTyID.
  /// Floats (with same fltSemantics) are sorted according to their value.
  /// Then you can see integers, and they are, like a floats,
  /// could be easy sorted among each others.
  /// The structures. Structures are grouped at the tail, again because of their
  /// TypeID: StructTyID > IntegerTyID > FloatTyID.
  /// Structures with greater number of elements are greater. Structures with
  /// greater elements going first are greater.
  /// The same logic with vectors, arrays and other possible complex types.
  ///
  /// Bitcastable constants.
  /// Let's assume, that some constant, belongs to some group of
  /// "so-called-equal" values with different types, and at the same time
  /// belongs to another group of constants with equal types
  /// and "really" equal values.
  ///
  /// Now, prove that this is impossible:
  ///
  /// If constant A with type TyA is bitcastable to B with type TyB, then:
  /// 1. All constants with equal types to TyA, are bitcastable to B. Since
  ///    those should be vectors (if TyA is vector), pointers
  ///    (if TyA is pointer), or else (if TyA equal to TyB), those types should
  ///    be equal to TyB.
  /// 2. All constants with non-equal, but bitcastable types to TyA, are
  ///    bitcastable to B.
  ///    Once again, just because we allow it to vectors and pointers only.
  ///    This statement could be expanded as below:
  /// 2.1. All vectors with equal bitwidth to vector A, has equal bitwidth to
  ///      vector B, and thus bitcastable to B as well.
  /// 2.2. All pointers of the same address space, no matter what they point to,
  ///      bitcastable. So if C is pointer, it could be bitcasted to A and to B.
  /// So any constant equal or bitcastable to A is equal or bitcastable to B.
  /// QED.
  ///
  /// In another words, for pointers and vectors, we ignore top-level type and
  /// look at their particular properties (bit-width for vectors, and
  /// address space for pointers).
  /// If these properties are equal - compare their contents.
  int cmpConstants(const Constant *L, const Constant *R);

  /// Assign or look up previously assigned numbers for the two values, and
  /// return whether the numbers are equal. Numbers are assigned in the order
  /// visited.
  /// Comparison order:
  /// Stage 0: Value that is function itself is always greater then others.
  ///          If left and right values are references to their functions, then
  ///          they are equal.
  /// Stage 1: Constants are greater than non-constants.
  ///          If both left and right are constants, then the result of
  ///          cmpConstants is used as cmpValues result.
  /// Stage 2: InlineAsm instances are greater than others. If both left and
  ///          right are InlineAsm instances, InlineAsm* pointers casted to
  ///          integers and compared as numbers.
  /// Stage 3: For all other cases we compare order we meet these values in
  ///          their functions. If right value was met first during scanning,
  ///          then left value is greater.
  ///          In another words, we compare serial numbers, for more details
  ///          see comments for sn_mapL and sn_mapR.
  int cmpValues(const Value *L, const Value *R);

  bool enumerate(const Value *V1, const Value *V2) {
    return cmpValues(V1, V2) == 0;
  }

  /// Compare two Instructions for equivalence, similar to
  /// Instruction::isSameOperationAs but with modifications to the type
  /// comparison.
  bool isEquivalentOperation(const Instruction *I1,
                             const Instruction *I2) const;

  /// Compare two GEPs for equivalent pointer arithmetic.
  bool isEquivalentGEP(const GEPOperator *GEP1, const GEPOperator *GEP2);
  bool isEquivalentGEP(const GetElementPtrInst *GEP1,
                       const GetElementPtrInst *GEP2) {
    return isEquivalentGEP(cast<GEPOperator>(GEP1), cast<GEPOperator>(GEP2));
  }

  /// cmpType - compares two types,
  /// defines total ordering among the types set.
  ///
  /// Return values:
  /// 0 if types are equal,
  /// -1 if Left is less than Right,
  /// +1 if Left is greater than Right.
  ///
  /// Description:
  /// Comparison is broken onto stages. Like in lexicographical comparison
  /// stage coming first has higher priority.
  /// On each explanation stage keep in mind total ordering properties.
  ///
  /// 0. Before comparison we coerce pointer types of 0 address space to
  /// integer.
  /// We also don't bother with same type at left and right, so
  /// just return 0 in this case.
  ///
  /// 1. If types are of different kind (different type IDs).
  ///    Return result of type IDs comparison, treating them as numbers.
  /// 2. If types are vectors or integers, compare Type* values as numbers.
  /// 3. Types has same ID, so check whether they belongs to the next group:
  /// * Void
  /// * Float
  /// * Double
  /// * X86_FP80
  /// * FP128
  /// * PPC_FP128
  /// * Label
  /// * Metadata
  /// If so - return 0, yes - we can treat these types as equal only because
  /// their IDs are same.
  /// 4. If Left and Right are pointers, return result of address space
  /// comparison (numbers comparison). We can treat pointer types of same
  /// address space as equal.
  /// 5. If types are complex.
  /// Then both Left and Right are to be expanded and their element types will
  /// be checked with the same way. If we get Res != 0 on some stage, return it.
  /// Otherwise return 0.
  /// 6. For all other cases put llvm_unreachable.
  int cmpType(Type *TyL, Type *TyR) const;

  bool isEquivalentType(Type *Ty1, Type *Ty2) const {
    return cmpType(Ty1, Ty2) == 0;
  }

  int cmpNumbers(uint64_t L, uint64_t R) const;

  int cmpAPInt(const APInt &L, const APInt &R) const;
  int cmpAPFloat(const APFloat &L, const APFloat &R) const;
  int cmpStrings(StringRef L, StringRef R) const;
  int cmpAttrs(const AttributeSet L, const AttributeSet R) const;

  // The two functions undergoing comparison.
  const Function *F1, *F2;

  const DataLayout *DL;

  /// Assign serial numbers to values from left function, and values from
  /// right function.
  /// Explanation:
  /// Being comparing functions we need to compare values we meet at left and
  /// right sides.
  /// Its easy to sort things out for external values. It just should be
  /// the same value at left and right.
  /// But for local values (those were introduced inside function body)
  /// we have to ensure they were introduced at exactly the same place,
  /// and plays the same role.
  /// Let's assign serial number to each value when we meet it first time.
  /// Values that were met at same place will be with same serial numbers.
  /// In this case it would be good to explain few points about values assigned
  /// to BBs and other ways of implementation (see below).
  ///
  /// 1. Safety of BB reordering.
  /// It's safe to change the order of BasicBlocks in function.
  /// Relationship with other functions and serial numbering will not be
  /// changed in this case.
  /// As follows from FunctionComparator::compare(), we do CFG walk: we start
  /// from the entry, and then take each terminator. So it doesn't matter how in
  /// fact BBs are ordered in function. And since cmpValues are called during
  /// this walk, the numbering depends only on how BBs located inside the CFG.
  /// So the answer is - yes. We will get the same numbering.
  ///
  /// 2. Impossibility to use dominance properties of values.
  /// If we compare two instruction operands: first is usage of local
  /// variable AL from function FL, and second is usage of local variable AR
  /// from FR, we could compare their origins and check whether they are
  /// defined at the same place.
  /// But, we are still not able to compare operands of PHI nodes, since those
  /// could be operands from further BBs we didn't scan yet.
  /// So it's impossible to use dominance properties in general.
  DenseMap<const Value*, int> sn_mapL, sn_mapR;
};

}

int FunctionComparator::cmpNumbers(uint64_t L, uint64_t R) const {
  if (L < R) return -1;
  if (L > R) return 1;
  return 0;
}

int FunctionComparator::cmpAPInt(const APInt &L, const APInt &R) const {
  if (int Res = cmpNumbers(L.getBitWidth(), R.getBitWidth()))
    return Res;
  if (L.ugt(R)) return 1;
  if (R.ugt(L)) return -1;
  return 0;
}

int FunctionComparator::cmpAPFloat(const APFloat &L, const APFloat &R) const {
  if (int Res = cmpNumbers((uint64_t)&L.getSemantics(),
                           (uint64_t)&R.getSemantics()))
    return Res;
  return cmpAPInt(L.bitcastToAPInt(), R.bitcastToAPInt());
}

int FunctionComparator::cmpStrings(StringRef L, StringRef R) const {
  // Prevent heavy comparison, compare sizes first.
  if (int Res = cmpNumbers(L.size(), R.size()))
    return Res;

  // Compare strings lexicographically only when it is necessary: only when
  // strings are equal in size.
  return L.compare(R);
}

int FunctionComparator::cmpAttrs(const AttributeSet L,
                                 const AttributeSet R) const {
  if (int Res = cmpNumbers(L.getNumSlots(), R.getNumSlots()))
    return Res;

  for (unsigned i = 0, e = L.getNumSlots(); i != e; ++i) {
    AttributeSet::iterator LI = L.begin(i), LE = L.end(i), RI = R.begin(i),
                           RE = R.end(i);
    for (; LI != LE && RI != RE; ++LI, ++RI) {
      Attribute LA = *LI;
      Attribute RA = *RI;
      if (LA < RA)
        return -1;
      if (RA < LA)
        return 1;
    }
    if (LI != LE)
      return 1;
    if (RI != RE)
      return -1;
  }
  return 0;
}

/// Constants comparison:
/// 1. Check whether type of L constant could be losslessly bitcasted to R
/// type.
/// 2. Compare constant contents.
/// For more details see declaration comments.
int FunctionComparator::cmpConstants(const Constant *L, const Constant *R) {

  Type *TyL = L->getType();
  Type *TyR = R->getType();

  // Check whether types are bitcastable. This part is just re-factored
  // Type::canLosslesslyBitCastTo method, but instead of returning true/false,
  // we also pack into result which type is "less" for us.
  int TypesRes = cmpType(TyL, TyR);
  if (TypesRes != 0) {
    // Types are different, but check whether we can bitcast them.
    if (!TyL->isFirstClassType()) {
      if (TyR->isFirstClassType())
        return -1;
      // Neither TyL nor TyR are values of first class type. Return the result
      // of comparing the types
      return TypesRes;
    }
    if (!TyR->isFirstClassType()) {
      if (TyL->isFirstClassType())
        return 1;
      return TypesRes;
    }

    // Vector -> Vector conversions are always lossless if the two vector types
    // have the same size, otherwise not.
    unsigned TyLWidth = 0;
    unsigned TyRWidth = 0;

    if (const VectorType *VecTyL = dyn_cast<VectorType>(TyL))
      TyLWidth = VecTyL->getBitWidth();
    if (const VectorType *VecTyR = dyn_cast<VectorType>(TyR))
      TyRWidth = VecTyR->getBitWidth();

    if (TyLWidth != TyRWidth)
      return cmpNumbers(TyLWidth, TyRWidth);

    // Zero bit-width means neither TyL nor TyR are vectors.
    if (!TyLWidth) {
      PointerType *PTyL = dyn_cast<PointerType>(TyL);
      PointerType *PTyR = dyn_cast<PointerType>(TyR);
      if (PTyL && PTyR) {
        unsigned AddrSpaceL = PTyL->getAddressSpace();
        unsigned AddrSpaceR = PTyR->getAddressSpace();
        if (int Res = cmpNumbers(AddrSpaceL, AddrSpaceR))
          return Res;
      }
      if (PTyL)
        return 1;
      if (PTyR)
        return -1;

      // TyL and TyR aren't vectors, nor pointers. We don't know how to
      // bitcast them.
      return TypesRes;
    }
  }

  // OK, types are bitcastable, now check constant contents.

  if (L->isNullValue() && R->isNullValue())
    return TypesRes;
  if (L->isNullValue() && !R->isNullValue())
    return 1;
  if (!L->isNullValue() && R->isNullValue())
    return -1;

  if (int Res = cmpNumbers(L->getValueID(), R->getValueID()))
    return Res;

  switch (L->getValueID()) {
  case Value::UndefValueVal: return TypesRes;
  case Value::ConstantIntVal: {
    const APInt &LInt = cast<ConstantInt>(L)->getValue();
    const APInt &RInt = cast<ConstantInt>(R)->getValue();
    return cmpAPInt(LInt, RInt);
  }
  case Value::ConstantFPVal: {
    const APFloat &LAPF = cast<ConstantFP>(L)->getValueAPF();
    const APFloat &RAPF = cast<ConstantFP>(R)->getValueAPF();
    return cmpAPFloat(LAPF, RAPF);
  }
  case Value::ConstantArrayVal: {
    const ConstantArray *LA = cast<ConstantArray>(L);
    const ConstantArray *RA = cast<ConstantArray>(R);
    uint64_t NumElementsL = cast<ArrayType>(TyL)->getNumElements();
    uint64_t NumElementsR = cast<ArrayType>(TyR)->getNumElements();
    if (int Res = cmpNumbers(NumElementsL, NumElementsR))
      return Res;
    for (uint64_t i = 0; i < NumElementsL; ++i) {
      if (int Res = cmpConstants(cast<Constant>(LA->getOperand(i)),
                                 cast<Constant>(RA->getOperand(i))))
        return Res;
    }
    return 0;
  }
  case Value::ConstantStructVal: {
    const ConstantStruct *LS = cast<ConstantStruct>(L);
    const ConstantStruct *RS = cast<ConstantStruct>(R);
    unsigned NumElementsL = cast<StructType>(TyL)->getNumElements();
    unsigned NumElementsR = cast<StructType>(TyR)->getNumElements();
    if (int Res = cmpNumbers(NumElementsL, NumElementsR))
      return Res;
    for (unsigned i = 0; i != NumElementsL; ++i) {
      if (int Res = cmpConstants(cast<Constant>(LS->getOperand(i)),
                                 cast<Constant>(RS->getOperand(i))))
        return Res;
    }
    return 0;
  }
  case Value::ConstantVectorVal: {
    const ConstantVector *LV = cast<ConstantVector>(L);
    const ConstantVector *RV = cast<ConstantVector>(R);
    unsigned NumElementsL = cast<VectorType>(TyL)->getNumElements();
    unsigned NumElementsR = cast<VectorType>(TyR)->getNumElements();
    if (int Res = cmpNumbers(NumElementsL, NumElementsR))
      return Res;
    for (uint64_t i = 0; i < NumElementsL; ++i) {
      if (int Res = cmpConstants(cast<Constant>(LV->getOperand(i)),
                                 cast<Constant>(RV->getOperand(i))))
        return Res;
    }
    return 0;
  }
  case Value::ConstantExprVal: {
    const ConstantExpr *LE = cast<ConstantExpr>(L);
    const ConstantExpr *RE = cast<ConstantExpr>(R);
    unsigned NumOperandsL = LE->getNumOperands();
    unsigned NumOperandsR = RE->getNumOperands();
    if (int Res = cmpNumbers(NumOperandsL, NumOperandsR))
      return Res;
    for (unsigned i = 0; i < NumOperandsL; ++i) {
      if (int Res = cmpConstants(cast<Constant>(LE->getOperand(i)),
                                 cast<Constant>(RE->getOperand(i))))
        return Res;
    }
    return 0;
  }
  case Value::FunctionVal:
  case Value::GlobalVariableVal:
  case Value::GlobalAliasVal:
  default: // Unknown constant, cast L and R pointers to numbers and compare.
    return cmpNumbers((uint64_t)L, (uint64_t)R);
  }
}

/// cmpType - compares two types,
/// defines total ordering among the types set.
/// See method declaration comments for more details.
int FunctionComparator::cmpType(Type *TyL, Type *TyR) const {

  PointerType *PTyL = dyn_cast<PointerType>(TyL);
  PointerType *PTyR = dyn_cast<PointerType>(TyR);

  if (DL) {
    if (PTyL && PTyL->getAddressSpace() == 0) TyL = DL->getIntPtrType(TyL);
    if (PTyR && PTyR->getAddressSpace() == 0) TyR = DL->getIntPtrType(TyR);
  }

  if (TyL == TyR)
    return 0;

  if (int Res = cmpNumbers(TyL->getTypeID(), TyR->getTypeID()))
    return Res;

  switch (TyL->getTypeID()) {
  default:
    llvm_unreachable("Unknown type!");
    // Fall through in Release mode.
  case Type::IntegerTyID:
  case Type::VectorTyID:
    // TyL == TyR would have returned true earlier.
    return cmpNumbers((uint64_t)TyL, (uint64_t)TyR);

  case Type::VoidTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
  case Type::LabelTyID:
  case Type::MetadataTyID:
    return 0;

  case Type::PointerTyID: {
    assert(PTyL && PTyR && "Both types must be pointers here.");
    return cmpNumbers(PTyL->getAddressSpace(), PTyR->getAddressSpace());
  }

  case Type::StructTyID: {
    StructType *STyL = cast<StructType>(TyL);
    StructType *STyR = cast<StructType>(TyR);
    if (STyL->getNumElements() != STyR->getNumElements())
      return cmpNumbers(STyL->getNumElements(), STyR->getNumElements());

    if (STyL->isPacked() != STyR->isPacked())
      return cmpNumbers(STyL->isPacked(), STyR->isPacked());

    for (unsigned i = 0, e = STyL->getNumElements(); i != e; ++i) {
      if (int Res = cmpType(STyL->getElementType(i),
                            STyR->getElementType(i)))
        return Res;
    }
    return 0;
  }

  case Type::FunctionTyID: {
    FunctionType *FTyL = cast<FunctionType>(TyL);
    FunctionType *FTyR = cast<FunctionType>(TyR);
    if (FTyL->getNumParams() != FTyR->getNumParams())
      return cmpNumbers(FTyL->getNumParams(), FTyR->getNumParams());

    if (FTyL->isVarArg() != FTyR->isVarArg())
      return cmpNumbers(FTyL->isVarArg(), FTyR->isVarArg());

    if (int Res = cmpType(FTyL->getReturnType(), FTyR->getReturnType()))
      return Res;

    for (unsigned i = 0, e = FTyL->getNumParams(); i != e; ++i) {
      if (int Res = cmpType(FTyL->getParamType(i), FTyR->getParamType(i)))
        return Res;
    }
    return 0;
  }

  case Type::ArrayTyID: {
    ArrayType *ATyL = cast<ArrayType>(TyL);
    ArrayType *ATyR = cast<ArrayType>(TyR);
    if (ATyL->getNumElements() != ATyR->getNumElements())
      return cmpNumbers(ATyL->getNumElements(), ATyR->getNumElements());
    return cmpType(ATyL->getElementType(), ATyR->getElementType());
  }
  }
}

// Determine whether the two operations are the same except that pointer-to-A
// and pointer-to-B are equivalent. This should be kept in sync with
// Instruction::isSameOperationAs.
bool FunctionComparator::isEquivalentOperation(const Instruction *I1,
                                               const Instruction *I2) const {
  // Differences from Instruction::isSameOperationAs:
  //  * replace type comparison with calls to isEquivalentType.
  //  * we test for I->hasSameSubclassOptionalData (nuw/nsw/tail) at the top
  //  * because of the above, we don't test for the tail bit on calls later on
  if (I1->getOpcode() != I2->getOpcode() ||
      I1->getNumOperands() != I2->getNumOperands() ||
      !isEquivalentType(I1->getType(), I2->getType()) ||
      !I1->hasSameSubclassOptionalData(I2))
    return false;

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same type
  for (unsigned i = 0, e = I1->getNumOperands(); i != e; ++i)
    if (!isEquivalentType(I1->getOperand(i)->getType(),
                          I2->getOperand(i)->getType()))
      return false;

  // Check special state that is a part of some instructions.
  if (const LoadInst *LI = dyn_cast<LoadInst>(I1))
    return LI->isVolatile() == cast<LoadInst>(I2)->isVolatile() &&
           LI->getAlignment() == cast<LoadInst>(I2)->getAlignment() &&
           LI->getOrdering() == cast<LoadInst>(I2)->getOrdering() &&
           LI->getSynchScope() == cast<LoadInst>(I2)->getSynchScope();
  if (const StoreInst *SI = dyn_cast<StoreInst>(I1))
    return SI->isVolatile() == cast<StoreInst>(I2)->isVolatile() &&
           SI->getAlignment() == cast<StoreInst>(I2)->getAlignment() &&
           SI->getOrdering() == cast<StoreInst>(I2)->getOrdering() &&
           SI->getSynchScope() == cast<StoreInst>(I2)->getSynchScope();
  if (const CmpInst *CI = dyn_cast<CmpInst>(I1))
    return CI->getPredicate() == cast<CmpInst>(I2)->getPredicate();
  if (const CallInst *CI = dyn_cast<CallInst>(I1))
    return CI->getCallingConv() == cast<CallInst>(I2)->getCallingConv() &&
           CI->getAttributes() == cast<CallInst>(I2)->getAttributes();
  if (const InvokeInst *CI = dyn_cast<InvokeInst>(I1))
    return CI->getCallingConv() == cast<InvokeInst>(I2)->getCallingConv() &&
           CI->getAttributes() == cast<InvokeInst>(I2)->getAttributes();
  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(I1))
    return IVI->getIndices() == cast<InsertValueInst>(I2)->getIndices();
  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(I1))
    return EVI->getIndices() == cast<ExtractValueInst>(I2)->getIndices();
  if (const FenceInst *FI = dyn_cast<FenceInst>(I1))
    return FI->getOrdering() == cast<FenceInst>(I2)->getOrdering() &&
           FI->getSynchScope() == cast<FenceInst>(I2)->getSynchScope();
  if (const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(I1))
    return CXI->isVolatile() == cast<AtomicCmpXchgInst>(I2)->isVolatile() &&
           CXI->getSuccessOrdering() ==
               cast<AtomicCmpXchgInst>(I2)->getSuccessOrdering() &&
           CXI->getFailureOrdering() ==
               cast<AtomicCmpXchgInst>(I2)->getFailureOrdering() &&
           CXI->getSynchScope() == cast<AtomicCmpXchgInst>(I2)->getSynchScope();
  if (const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I1))
    return RMWI->getOperation() == cast<AtomicRMWInst>(I2)->getOperation() &&
           RMWI->isVolatile() == cast<AtomicRMWInst>(I2)->isVolatile() &&
           RMWI->getOrdering() == cast<AtomicRMWInst>(I2)->getOrdering() &&
           RMWI->getSynchScope() == cast<AtomicRMWInst>(I2)->getSynchScope();

  return true;
}

// Determine whether two GEP operations perform the same underlying arithmetic.
bool FunctionComparator::isEquivalentGEP(const GEPOperator *GEP1,
                                         const GEPOperator *GEP2) {
  unsigned AS = GEP1->getPointerAddressSpace();
  if (AS != GEP2->getPointerAddressSpace())
    return false;

  if (DL) {
    // When we have target data, we can reduce the GEP down to the value in bytes
    // added to the address.
    unsigned BitWidth = DL ? DL->getPointerSizeInBits(AS) : 1;
    APInt Offset1(BitWidth, 0), Offset2(BitWidth, 0);
    if (GEP1->accumulateConstantOffset(*DL, Offset1) &&
        GEP2->accumulateConstantOffset(*DL, Offset2)) {
      return Offset1 == Offset2;
    }
  }

  if (GEP1->getPointerOperand()->getType() !=
      GEP2->getPointerOperand()->getType())
    return false;

  if (GEP1->getNumOperands() != GEP2->getNumOperands())
    return false;

  for (unsigned i = 0, e = GEP1->getNumOperands(); i != e; ++i) {
    if (!enumerate(GEP1->getOperand(i), GEP2->getOperand(i)))
      return false;
  }

  return true;
}

/// Compare two values used by the two functions under pair-wise comparison. If
/// this is the first time the values are seen, they're added to the mapping so
/// that we will detect mismatches on next use.
/// See comments in declaration for more details.
int FunctionComparator::cmpValues(const Value *L, const Value *R) {
  // Catch self-reference case.
  if (L == F1) {
    if (R == F2)
      return 0;
    return -1;
  }
  if (R == F2) {
    if (L == F1)
      return 0;
    return 1;
  }

  const Constant *ConstL = dyn_cast<Constant>(L);
  const Constant *ConstR = dyn_cast<Constant>(R);
  if (ConstL && ConstR) {
    if (L == R)
      return 0;
    return cmpConstants(ConstL, ConstR);
  }

  if (ConstL)
    return 1;
  if (ConstR)
    return -1;

  const InlineAsm *InlineAsmL = dyn_cast<InlineAsm>(L);
  const InlineAsm *InlineAsmR = dyn_cast<InlineAsm>(R);

  if (InlineAsmL && InlineAsmR)
    return cmpNumbers((uint64_t)L, (uint64_t)R);
  if (InlineAsmL)
    return 1;
  if (InlineAsmR)
    return -1;

  auto LeftSN = sn_mapL.insert(std::make_pair(L, sn_mapL.size())),
       RightSN = sn_mapR.insert(std::make_pair(R, sn_mapR.size()));

  return cmpNumbers(LeftSN.first->second, RightSN.first->second);
}
// Test whether two basic blocks have equivalent behaviour.
bool FunctionComparator::compare(const BasicBlock *BB1, const BasicBlock *BB2) {
  BasicBlock::const_iterator F1I = BB1->begin(), F1E = BB1->end();
  BasicBlock::const_iterator F2I = BB2->begin(), F2E = BB2->end();

  do {
    if (!enumerate(F1I, F2I))
      return false;

    if (const GetElementPtrInst *GEP1 = dyn_cast<GetElementPtrInst>(F1I)) {
      const GetElementPtrInst *GEP2 = dyn_cast<GetElementPtrInst>(F2I);
      if (!GEP2)
        return false;

      if (!enumerate(GEP1->getPointerOperand(), GEP2->getPointerOperand()))
        return false;

      if (!isEquivalentGEP(GEP1, GEP2))
        return false;
    } else {
      if (!isEquivalentOperation(F1I, F2I))
        return false;

      assert(F1I->getNumOperands() == F2I->getNumOperands());
      for (unsigned i = 0, e = F1I->getNumOperands(); i != e; ++i) {
        Value *OpF1 = F1I->getOperand(i);
        Value *OpF2 = F2I->getOperand(i);

        if (!enumerate(OpF1, OpF2))
          return false;

        if (OpF1->getValueID() != OpF2->getValueID() ||
            !isEquivalentType(OpF1->getType(), OpF2->getType()))
          return false;
      }
    }

    ++F1I, ++F2I;
  } while (F1I != F1E && F2I != F2E);

  return F1I == F1E && F2I == F2E;
}

// Test whether the two functions have equivalent behaviour.
bool FunctionComparator::compare() {
  // We need to recheck everything, but check the things that weren't included
  // in the hash first.

  sn_mapL.clear();
  sn_mapR.clear();

  if (F1->getAttributes() != F2->getAttributes())
    return false;

  if (F1->hasGC() != F2->hasGC())
    return false;

  if (F1->hasGC() && F1->getGC() != F2->getGC())
    return false;

  if (F1->hasSection() != F2->hasSection())
    return false;

  if (F1->hasSection() && F1->getSection() != F2->getSection())
    return false;

  if (F1->isVarArg() != F2->isVarArg())
    return false;

  // TODO: if it's internal and only used in direct calls, we could handle this
  // case too.
  if (F1->getCallingConv() != F2->getCallingConv())
    return false;

  if (!isEquivalentType(F1->getFunctionType(), F2->getFunctionType()))
    return false;

  assert(F1->arg_size() == F2->arg_size() &&
         "Identically typed functions have different numbers of args!");

  // Visit the arguments so that they get enumerated in the order they're
  // passed in.
  for (Function::const_arg_iterator f1i = F1->arg_begin(),
         f2i = F2->arg_begin(), f1e = F1->arg_end(); f1i != f1e; ++f1i, ++f2i) {
    if (!enumerate(f1i, f2i))
      llvm_unreachable("Arguments repeat!");
  }

  // We do a CFG-ordered walk since the actual ordering of the blocks in the
  // linked list is immaterial. Our walk starts at the entry block for both
  // functions, then takes each block from each terminator in order. As an
  // artifact, this also means that unreachable blocks are ignored.
  SmallVector<const BasicBlock *, 8> F1BBs, F2BBs;
  SmallSet<const BasicBlock *, 128> VisitedBBs; // in terms of F1.

  F1BBs.push_back(&F1->getEntryBlock());
  F2BBs.push_back(&F2->getEntryBlock());

  VisitedBBs.insert(F1BBs[0]);
  while (!F1BBs.empty()) {
    const BasicBlock *F1BB = F1BBs.pop_back_val();
    const BasicBlock *F2BB = F2BBs.pop_back_val();

    if (!enumerate(F1BB, F2BB) || !compare(F1BB, F2BB))
      return false;

    const TerminatorInst *F1TI = F1BB->getTerminator();
    const TerminatorInst *F2TI = F2BB->getTerminator();

    assert(F1TI->getNumSuccessors() == F2TI->getNumSuccessors());
    for (unsigned i = 0, e = F1TI->getNumSuccessors(); i != e; ++i) {
      if (!VisitedBBs.insert(F1TI->getSuccessor(i)))
        continue;

      F1BBs.push_back(F1TI->getSuccessor(i));
      F2BBs.push_back(F2TI->getSuccessor(i));
    }
  }
  return true;
}

namespace {

/// MergeFunctions finds functions which will generate identical machine code,
/// by considering all pointer types to be equivalent. Once identified,
/// MergeFunctions will fold them by replacing a call to one to a call to a
/// bitcast of the other.
///
class MergeFunctions : public ModulePass {
public:
  static char ID;
  MergeFunctions()
    : ModulePass(ID), HasGlobalAliases(false) {
    initializeMergeFunctionsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

private:
  typedef DenseSet<ComparableFunction> FnSetType;

  /// A work queue of functions that may have been modified and should be
  /// analyzed again.
  std::vector<WeakVH> Deferred;

  /// Insert a ComparableFunction into the FnSet, or merge it away if it's
  /// equal to one that's already present.
  bool insert(ComparableFunction &NewF);

  /// Remove a Function from the FnSet and queue it up for a second sweep of
  /// analysis.
  void remove(Function *F);

  /// Find the functions that use this Value and remove them from FnSet and
  /// queue the functions.
  void removeUsers(Value *V);

  /// Replace all direct calls of Old with calls of New. Will bitcast New if
  /// necessary to make types match.
  void replaceDirectCallers(Function *Old, Function *New);

  /// Merge two equivalent functions. Upon completion, G may be deleted, or may
  /// be converted into a thunk. In either case, it should never be visited
  /// again.
  void mergeTwoFunctions(Function *F, Function *G);

  /// Replace G with a thunk or an alias to F. Deletes G.
  void writeThunkOrAlias(Function *F, Function *G);

  /// Replace G with a simple tail call to bitcast(F). Also replace direct uses
  /// of G with bitcast(F). Deletes G.
  void writeThunk(Function *F, Function *G);

  /// Replace G with an alias to F. Deletes G.
  void writeAlias(Function *F, Function *G);

  /// The set of all distinct functions. Use the insert() and remove() methods
  /// to modify it.
  FnSetType FnSet;

  /// DataLayout for more accurate GEP comparisons. May be NULL.
  const DataLayout *DL;

  /// Whether or not the target supports global aliases.
  bool HasGlobalAliases;
};

}  // end anonymous namespace

char MergeFunctions::ID = 0;
INITIALIZE_PASS(MergeFunctions, "mergefunc", "Merge Functions", false, false)

ModulePass *llvm::createMergeFunctionsPass() {
  return new MergeFunctions();
}

bool MergeFunctions::runOnModule(Module &M) {
  bool Changed = false;
  DataLayoutPass *DLP = getAnalysisIfAvailable<DataLayoutPass>();
  DL = DLP ? &DLP->getDataLayout() : nullptr;

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    if (!I->isDeclaration() && !I->hasAvailableExternallyLinkage())
      Deferred.push_back(WeakVH(I));
  }
  FnSet.resize(Deferred.size());

  do {
    std::vector<WeakVH> Worklist;
    Deferred.swap(Worklist);

    DEBUG(dbgs() << "size of module: " << M.size() << '\n');
    DEBUG(dbgs() << "size of worklist: " << Worklist.size() << '\n');

    // Insert only strong functions and merge them. Strong function merging
    // always deletes one of them.
    for (std::vector<WeakVH>::iterator I = Worklist.begin(),
           E = Worklist.end(); I != E; ++I) {
      if (!*I) continue;
      Function *F = cast<Function>(*I);
      if (!F->isDeclaration() && !F->hasAvailableExternallyLinkage() &&
          !F->mayBeOverridden()) {
        ComparableFunction CF = ComparableFunction(F, DL);
        Changed |= insert(CF);
      }
    }

    // Insert only weak functions and merge them. By doing these second we
    // create thunks to the strong function when possible. When two weak
    // functions are identical, we create a new strong function with two weak
    // weak thunks to it which are identical but not mergable.
    for (std::vector<WeakVH>::iterator I = Worklist.begin(),
           E = Worklist.end(); I != E; ++I) {
      if (!*I) continue;
      Function *F = cast<Function>(*I);
      if (!F->isDeclaration() && !F->hasAvailableExternallyLinkage() &&
          F->mayBeOverridden()) {
        ComparableFunction CF = ComparableFunction(F, DL);
        Changed |= insert(CF);
      }
    }
    DEBUG(dbgs() << "size of FnSet: " << FnSet.size() << '\n');
  } while (!Deferred.empty());

  FnSet.clear();

  return Changed;
}

bool DenseMapInfo<ComparableFunction>::isEqual(const ComparableFunction &LHS,
                                               const ComparableFunction &RHS) {
  if (LHS.getFunc() == RHS.getFunc() &&
      LHS.getHash() == RHS.getHash())
    return true;
  if (!LHS.getFunc() || !RHS.getFunc())
    return false;

  // One of these is a special "underlying pointer comparison only" object.
  if (LHS.getDataLayout() == ComparableFunction::LookupOnly ||
      RHS.getDataLayout() == ComparableFunction::LookupOnly)
    return false;

  assert(LHS.getDataLayout() == RHS.getDataLayout() &&
         "Comparing functions for different targets");

  return FunctionComparator(LHS.getDataLayout(), LHS.getFunc(),
                            RHS.getFunc()).compare();
}

// Replace direct callers of Old with New.
void MergeFunctions::replaceDirectCallers(Function *Old, Function *New) {
  Constant *BitcastNew = ConstantExpr::getBitCast(New, Old->getType());
  for (auto UI = Old->use_begin(), UE = Old->use_end(); UI != UE;) {
    Use *U = &*UI;
    ++UI;
    CallSite CS(U->getUser());
    if (CS && CS.isCallee(U)) {
      remove(CS.getInstruction()->getParent()->getParent());
      U->set(BitcastNew);
    }
  }
}

// Replace G with an alias to F if possible, or else a thunk to F. Deletes G.
void MergeFunctions::writeThunkOrAlias(Function *F, Function *G) {
  if (HasGlobalAliases && G->hasUnnamedAddr()) {
    if (G->hasExternalLinkage() || G->hasLocalLinkage() ||
        G->hasWeakLinkage()) {
      writeAlias(F, G);
      return;
    }
  }

  writeThunk(F, G);
}

// Helper for writeThunk,
// Selects proper bitcast operation,
// but a bit simpler then CastInst::getCastOpcode.
static Value *createCast(IRBuilder<false> &Builder, Value *V, Type *DestTy) {
  Type *SrcTy = V->getType();
  if (SrcTy->isStructTy()) {
    assert(DestTy->isStructTy());
    assert(SrcTy->getStructNumElements() == DestTy->getStructNumElements());
    Value *Result = UndefValue::get(DestTy);
    for (unsigned int I = 0, E = SrcTy->getStructNumElements(); I < E; ++I) {
      Value *Element = createCast(
          Builder, Builder.CreateExtractValue(V, ArrayRef<unsigned int>(I)),
          DestTy->getStructElementType(I));

      Result =
          Builder.CreateInsertValue(Result, Element, ArrayRef<unsigned int>(I));
    }
    return Result;
  }
  assert(!DestTy->isStructTy());
  if (SrcTy->isIntegerTy() && DestTy->isPointerTy())
    return Builder.CreateIntToPtr(V, DestTy);
  else if (SrcTy->isPointerTy() && DestTy->isIntegerTy())
    return Builder.CreatePtrToInt(V, DestTy);
  else
    return Builder.CreateBitCast(V, DestTy);
}

// Replace G with a simple tail call to bitcast(F). Also replace direct uses
// of G with bitcast(F). Deletes G.
void MergeFunctions::writeThunk(Function *F, Function *G) {
  if (!G->mayBeOverridden()) {
    // Redirect direct callers of G to F.
    replaceDirectCallers(G, F);
  }

  // If G was internal then we may have replaced all uses of G with F. If so,
  // stop here and delete G. There's no need for a thunk.
  if (G->hasLocalLinkage() && G->use_empty()) {
    G->eraseFromParent();
    return;
  }

  Function *NewG = Function::Create(G->getFunctionType(), G->getLinkage(), "",
                                    G->getParent());
  BasicBlock *BB = BasicBlock::Create(F->getContext(), "", NewG);
  IRBuilder<false> Builder(BB);

  SmallVector<Value *, 16> Args;
  unsigned i = 0;
  FunctionType *FFTy = F->getFunctionType();
  for (Function::arg_iterator AI = NewG->arg_begin(), AE = NewG->arg_end();
       AI != AE; ++AI) {
    Args.push_back(createCast(Builder, (Value*)AI, FFTy->getParamType(i)));
    ++i;
  }

  CallInst *CI = Builder.CreateCall(F, Args);
  CI->setTailCall();
  CI->setCallingConv(F->getCallingConv());
  if (NewG->getReturnType()->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    Builder.CreateRet(createCast(Builder, CI, NewG->getReturnType()));
  }

  NewG->copyAttributesFrom(G);
  NewG->takeName(G);
  removeUsers(G);
  G->replaceAllUsesWith(NewG);
  G->eraseFromParent();

  DEBUG(dbgs() << "writeThunk: " << NewG->getName() << '\n');
  ++NumThunksWritten;
}

// Replace G with an alias to F and delete G.
void MergeFunctions::writeAlias(Function *F, Function *G) {
  Constant *BitcastF = ConstantExpr::getBitCast(F, G->getType());
  GlobalAlias *GA = new GlobalAlias(G->getType(), G->getLinkage(), "",
                                    BitcastF, G->getParent());
  F->setAlignment(std::max(F->getAlignment(), G->getAlignment()));
  GA->takeName(G);
  GA->setVisibility(G->getVisibility());
  removeUsers(G);
  G->replaceAllUsesWith(GA);
  G->eraseFromParent();

  DEBUG(dbgs() << "writeAlias: " << GA->getName() << '\n');
  ++NumAliasesWritten;
}

// Merge two equivalent functions. Upon completion, Function G is deleted.
void MergeFunctions::mergeTwoFunctions(Function *F, Function *G) {
  if (F->mayBeOverridden()) {
    assert(G->mayBeOverridden());

    if (HasGlobalAliases) {
      // Make them both thunks to the same internal function.
      Function *H = Function::Create(F->getFunctionType(), F->getLinkage(), "",
                                     F->getParent());
      H->copyAttributesFrom(F);
      H->takeName(F);
      removeUsers(F);
      F->replaceAllUsesWith(H);

      unsigned MaxAlignment = std::max(G->getAlignment(), H->getAlignment());

      writeAlias(F, G);
      writeAlias(F, H);

      F->setAlignment(MaxAlignment);
      F->setLinkage(GlobalValue::PrivateLinkage);
    } else {
      // We can't merge them. Instead, pick one and update all direct callers
      // to call it and hope that we improve the instruction cache hit rate.
      replaceDirectCallers(G, F);
    }

    ++NumDoubleWeak;
  } else {
    writeThunkOrAlias(F, G);
  }

  ++NumFunctionsMerged;
}

// Insert a ComparableFunction into the FnSet, or merge it away if equal to one
// that was already inserted.
bool MergeFunctions::insert(ComparableFunction &NewF) {
  std::pair<FnSetType::iterator, bool> Result = FnSet.insert(NewF);
  if (Result.second) {
    DEBUG(dbgs() << "Inserting as unique: " << NewF.getFunc()->getName() << '\n');
    return false;
  }

  const ComparableFunction &OldF = *Result.first;

  // Don't merge tiny functions, since it can just end up making the function
  // larger.
  // FIXME: Should still merge them if they are unnamed_addr and produce an
  // alias.
  if (NewF.getFunc()->size() == 1) {
    if (NewF.getFunc()->front().size() <= 2) {
      DEBUG(dbgs() << NewF.getFunc()->getName()
            << " is to small to bother merging\n");
      return false;
    }
  }

  // Never thunk a strong function to a weak function.
  assert(!OldF.getFunc()->mayBeOverridden() ||
         NewF.getFunc()->mayBeOverridden());

  DEBUG(dbgs() << "  " << OldF.getFunc()->getName() << " == "
               << NewF.getFunc()->getName() << '\n');

  Function *DeleteF = NewF.getFunc();
  NewF.release();
  mergeTwoFunctions(OldF.getFunc(), DeleteF);
  return true;
}

// Remove a function from FnSet. If it was already in FnSet, add it to Deferred
// so that we'll look at it in the next round.
void MergeFunctions::remove(Function *F) {
  // We need to make sure we remove F, not a function "equal" to F per the
  // function equality comparator.
  //
  // The special "lookup only" ComparableFunction bypasses the expensive
  // function comparison in favour of a pointer comparison on the underlying
  // Function*'s.
  ComparableFunction CF = ComparableFunction(F, ComparableFunction::LookupOnly);
  if (FnSet.erase(CF)) {
    DEBUG(dbgs() << "Removed " << F->getName() << " from set and deferred it.\n");
    Deferred.push_back(F);
  }
}

// For each instruction used by the value, remove() the function that contains
// the instruction. This should happen right before a call to RAUW.
void MergeFunctions::removeUsers(Value *V) {
  std::vector<Value *> Worklist;
  Worklist.push_back(V);
  while (!Worklist.empty()) {
    Value *V = Worklist.back();
    Worklist.pop_back();

    for (User *U : V->users()) {
      if (Instruction *I = dyn_cast<Instruction>(U)) {
        remove(I->getParent()->getParent());
      } else if (isa<GlobalValue>(U)) {
        // do nothing
      } else if (Constant *C = dyn_cast<Constant>(U)) {
        for (User *UU : C->users())
          Worklist.push_back(UU);
      }
    }
  }
}
