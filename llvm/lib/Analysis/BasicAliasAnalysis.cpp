//===- BasicAliasAnalysis.cpp - Local Alias Analysis Impl -----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the default implementation of the Alias Analysis interface
// that simply implements a few identities (two different globals cannot alias,
// etc), but otherwise does no analysis.
//
// FIXME: This could be extended for a very simple form of mod/ref information.
// If a pointer is locally allocated (either malloc or alloca) and never passed
// into a call or stored to memory, then we know that calls will not mod/ref the
// memory.  This can be important for tailcallelim, and can support CSE of loads
// and dead store elimination across calls.  This is particularly important for
// stack allocated arrays.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
using namespace llvm;

// Make sure that anything that uses AliasAnalysis pulls in this file...
void llvm::BasicAAStub() {}

namespace {
  /// NoAA - This class implements the -no-aa pass, which always returns "I
  /// don't know" for alias queries.  NoAA is unlike other alias analysis
  /// implementations, in that it does not chain to a previous analysis.  As
  /// such it doesn't follow many of the rules that other alias analyses must.
  ///
  struct NoAA : public ImmutablePass, public AliasAnalysis {
    virtual AliasResult alias(const Value *V1, unsigned V1Size,
                              const Value *V2, unsigned V2Size) {
      return MayAlias;
    }

    virtual void getMustAliases(Value *P, std::vector<Value*> &RetVals) { }
    virtual bool pointsToConstantMemory(const Value *P) { return false; }
    virtual bool doesNotAccessMemory(Function *F) { return false; }
    virtual bool onlyReadsMemory(Function *F) { return false; }
    virtual ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size) {
      return ModRef;
    }
    virtual ModRefResult getModRefInfo(CallSite CS1, CallSite CS2) {
      return ModRef;
    }
    virtual bool hasNoModRefInfoForCalls() const { return true; }

    virtual void deleteValue(Value *V) {}
    virtual void copyValue(Value *From, Value *To) {}
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {}
  };
 
  // Register this pass...
  RegisterOpt<NoAA>
  U("no-aa", "No Alias Analysis (always returns 'may' alias)");

  // Declare that we implement the AliasAnalysis interface
  RegisterAnalysisGroup<AliasAnalysis, NoAA> V;
}  // End of anonymous namespace


namespace {
  /// BasicAliasAnalysis - This is the default alias analysis implementation.
  /// Because it doesn't chain to a previous alias analysis (like -no-aa), it
  /// derives from the NoAA class.
  struct BasicAliasAnalysis : public NoAA {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
    }
    
    virtual void initializePass() {
      TD = &getAnalysis<TargetData>();
    }

    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);

    ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size);

    /// hasNoModRefInfoForCalls - We have no way to test one call against
    /// another, unless they are pure or const.
    virtual bool hasNoModRefInfoForCalls() const { return true; }

    /// pointsToConstantMemory - Chase pointers until we find a (constant
    /// global) or not.
    bool pointsToConstantMemory(const Value *P);

    virtual bool doesNotAccessMemory(Function *F);
    virtual bool onlyReadsMemory(Function *F);

  private:
    // CheckGEPInstructions - Check two GEP instructions with known
    // must-aliasing base pointers.  This checks to see if the index expressions
    // preclude the pointers from aliasing...
    AliasResult
    CheckGEPInstructions(const Type* BasePtr1Ty, std::vector<Value*> &GEP1Ops,
                         unsigned G1Size,
                         const Type *BasePtr2Ty, std::vector<Value*> &GEP2Ops,
                         unsigned G2Size);
  };
 
  // Register this pass...
  RegisterOpt<BasicAliasAnalysis>
  X("basicaa", "Basic Alias Analysis (default AA impl)");

  // Declare that we implement the AliasAnalysis interface
  RegisterAnalysisGroup<AliasAnalysis, BasicAliasAnalysis, true> Y;
}  // End of anonymous namespace

// hasUniqueAddress - Return true if the specified value points to something
// with a unique, discernable, address.
static inline bool hasUniqueAddress(const Value *V) {
  return isa<GlobalValue>(V) || isa<AllocationInst>(V);
}

// getUnderlyingObject - This traverses the use chain to figure out what object
// the specified value points to.  If the value points to, or is derived from, a
// unique object or an argument, return it.
static const Value *getUnderlyingObject(const Value *V) {
  if (!isa<PointerType>(V->getType())) return 0;

  // If we are at some type of object... return it.
  if (hasUniqueAddress(V) || isa<Argument>(V)) return V;
  
  // Traverse through different addressing mechanisms...
  if (const Instruction *I = dyn_cast<Instruction>(V)) {
    if (isa<CastInst>(I) || isa<GetElementPtrInst>(I))
      return getUnderlyingObject(I->getOperand(0));
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::Cast ||
        CE->getOpcode() == Instruction::GetElementPtr)
      return getUnderlyingObject(CE->getOperand(0));
  } else if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V)) {
    return CPR->getValue();
  }
  return 0;
}

static const User *isGEP(const Value *V) {
  if (isa<GetElementPtrInst>(V) ||
      (isa<ConstantExpr>(V) &&
       cast<ConstantExpr>(V)->getOpcode() == Instruction::GetElementPtr))
    return cast<User>(V);
  return 0;
}

static const Value *GetGEPOperands(const Value *V, std::vector<Value*> &GEPOps){
  assert(GEPOps.empty() && "Expect empty list to populate!");
  GEPOps.insert(GEPOps.end(), cast<User>(V)->op_begin()+1,
                cast<User>(V)->op_end());

  // Accumulate all of the chained indexes into the operand array
  V = cast<User>(V)->getOperand(0);

  while (const User *G = isGEP(V)) {
    if (!isa<Constant>(GEPOps[0]) ||
        !cast<Constant>(GEPOps[0])->isNullValue())
      break;  // Don't handle folding arbitrary pointer offsets yet...
    GEPOps.erase(GEPOps.begin());   // Drop the zero index
    GEPOps.insert(GEPOps.begin(), G->op_begin()+1, G->op_end());
    V = G->getOperand(0);
  }
  return V;
}

/// pointsToConstantMemory - Chase pointers until we find a (constant
/// global) or not.
bool BasicAliasAnalysis::pointsToConstantMemory(const Value *P) {
  if (const Value *V = getUnderlyingObject(P))
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
      return GV->isConstant();
  return false;
}

static bool AddressMightEscape(const Value *V) {
  for (Value::use_const_iterator UI = V->use_begin(), E = V->use_end();
       UI != E; ++UI) {
    const Instruction *I = cast<Instruction>(*UI);
    switch (I->getOpcode()) {
    case Instruction::Load: break;
    case Instruction::Store:
      if (I->getOperand(0) == V)
        return true; // Escapes if the pointer is stored.
      break;
    case Instruction::GetElementPtr:
      if (AddressMightEscape(I)) return true;
      break;
    case Instruction::Cast:
      if (!isa<PointerType>(I->getType()))
        return true;
      if (AddressMightEscape(I)) return true;
      break;
    default:
      return true;
    }
  }
  return false;
}

// getModRefInfo - Check to see if the specified callsite can clobber the
// specified memory object.  Since we only look at local properties of this
// function, we really can't say much about this query.  We do, however, use
// simple "address taken" analysis on local objects.
//
AliasAnalysis::ModRefResult
BasicAliasAnalysis::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  if (!isa<Constant>(P) && !isa<GlobalValue>(P))
    if (const AllocationInst *AI =
                  dyn_cast_or_null<AllocationInst>(getUnderlyingObject(P))) {
      // Okay, the pointer is to a stack allocated object.  If we can prove that
      // the pointer never "escapes", then we know the call cannot clobber it,
      // because it simply can't get its address.
      if (!AddressMightEscape(AI))
        return NoModRef;
    }

  // The AliasAnalysis base class has some smarts, lets use them.
  return AliasAnalysis::getModRefInfo(CS, P, Size);
}

// alias - Provide a bunch of ad-hoc rules to disambiguate in common cases, such
// as array references.  Note that this function is heavily tail recursive.
// Hopefully we have a smart C++ compiler.  :)
//
AliasAnalysis::AliasResult
BasicAliasAnalysis::alias(const Value *V1, unsigned V1Size,
                          const Value *V2, unsigned V2Size) {
  // Strip off any constant expression casts if they exist
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V1))
    if (CE->getOpcode() == Instruction::Cast)
      V1 = CE->getOperand(0);
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V2))
    if (CE->getOpcode() == Instruction::Cast)
      V2 = CE->getOperand(0);

  // Strip off constant pointer refs if they exist
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V1))
    V1 = CPR->getValue();
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V2))
    V2 = CPR->getValue();

  // Are we checking for alias of the same value?
  if (V1 == V2) return MustAlias;

  if ((!isa<PointerType>(V1->getType()) || !isa<PointerType>(V2->getType())) &&
      V1->getType() != Type::LongTy && V2->getType() != Type::LongTy)
    return NoAlias;  // Scalars cannot alias each other

  // Strip off cast instructions...
  if (const Instruction *I = dyn_cast<CastInst>(V1))
    return alias(I->getOperand(0), V1Size, V2, V2Size);
  if (const Instruction *I = dyn_cast<CastInst>(V2))
    return alias(V1, V1Size, I->getOperand(0), V2Size);

  // Figure out what objects these things are pointing to if we can...
  const Value *O1 = getUnderlyingObject(V1);
  const Value *O2 = getUnderlyingObject(V2);

  // Pointing at a discernible object?
  if (O1 && O2) {
    if (isa<Argument>(O1)) {
      // Incoming argument cannot alias locally allocated object!
      if (isa<AllocationInst>(O2)) return NoAlias;
      // Otherwise, nothing is known...
    } else if (isa<Argument>(O2)) {
      // Incoming argument cannot alias locally allocated object!
      if (isa<AllocationInst>(O1)) return NoAlias;
      // Otherwise, nothing is known...
    } else {
      // If they are two different objects, we know that we have no alias...
      if (O1 != O2) return NoAlias;
    }

    // If they are the same object, they we can look at the indexes.  If they
    // index off of the object is the same for both pointers, they must alias.
    // If they are provably different, they must not alias.  Otherwise, we can't
    // tell anything.
  } else if (O1 && !isa<Argument>(O1) && isa<ConstantPointerNull>(V2)) {
    return NoAlias;                    // Unique values don't alias null
  } else if (O2 && !isa<Argument>(O2) && isa<ConstantPointerNull>(V1)) {
    return NoAlias;                    // Unique values don't alias null
  }

  // If we have two gep instructions with must-alias'ing base pointers, figure
  // out if the indexes to the GEP tell us anything about the derived pointer.
  // Note that we also handle chains of getelementptr instructions as well as
  // constant expression getelementptrs here.
  //
  if (isGEP(V1) && isGEP(V2)) {
    // Drill down into the first non-gep value, to test for must-aliasing of
    // the base pointers.
    const Value *BasePtr1 = V1, *BasePtr2 = V2;
    do {
      BasePtr1 = cast<User>(BasePtr1)->getOperand(0);
    } while (isGEP(BasePtr1) &&
             cast<User>(BasePtr1)->getOperand(1) == 
       Constant::getNullValue(cast<User>(BasePtr1)->getOperand(1)->getType()));
    do {
      BasePtr2 = cast<User>(BasePtr2)->getOperand(0);
    } while (isGEP(BasePtr2) &&
             cast<User>(BasePtr2)->getOperand(1) == 
       Constant::getNullValue(cast<User>(BasePtr2)->getOperand(1)->getType()));

    // Do the base pointers alias?
    AliasResult BaseAlias = alias(BasePtr1, V1Size, BasePtr2, V2Size);
    if (BaseAlias == NoAlias) return NoAlias;
    if (BaseAlias == MustAlias) {
      // If the base pointers alias each other exactly, check to see if we can
      // figure out anything about the resultant pointers, to try to prove
      // non-aliasing.

      // Collect all of the chained GEP operands together into one simple place
      std::vector<Value*> GEP1Ops, GEP2Ops;
      BasePtr1 = GetGEPOperands(V1, GEP1Ops);
      BasePtr2 = GetGEPOperands(V2, GEP2Ops);

      AliasResult GAlias =
        CheckGEPInstructions(BasePtr1->getType(), GEP1Ops, V1Size,
                             BasePtr2->getType(), GEP2Ops, V2Size);
      if (GAlias != MayAlias)
        return GAlias;
    }
  }

  // Check to see if these two pointers are related by a getelementptr
  // instruction.  If one pointer is a GEP with a non-zero index of the other
  // pointer, we know they cannot alias.
  //
  if (isGEP(V2)) {
    std::swap(V1, V2);
    std::swap(V1Size, V2Size);
  }

  if (V1Size != ~0U && V2Size != ~0U)
    if (const User *GEP = isGEP(V1)) {
      std::vector<Value*> GEPOperands;
      const Value *BasePtr = GetGEPOperands(V1, GEPOperands);

      AliasResult R = alias(BasePtr, V1Size, V2, V2Size);
      if (R == MustAlias) {
        // If there is at least one non-zero constant index, we know they cannot
        // alias.
        bool ConstantFound = false;
        bool AllZerosFound = true;
        for (unsigned i = 0, e = GEPOperands.size(); i != e; ++i)
          if (const Constant *C = dyn_cast<Constant>(GEPOperands[i])) {
            if (!C->isNullValue()) {
              ConstantFound = true;
              AllZerosFound = false;
              break;
            }
          } else {
            AllZerosFound = false;
          }

        // If we have getelementptr <ptr>, 0, 0, 0, 0, ... and V2 must aliases
        // the ptr, the end result is a must alias also.
        if (AllZerosFound)
          return MustAlias;

        if (ConstantFound) {
          if (V2Size <= 1 && V1Size <= 1)  // Just pointer check?
            return NoAlias;
          
          // Otherwise we have to check to see that the distance is more than
          // the size of the argument... build an index vector that is equal to
          // the arguments provided, except substitute 0's for any variable
          // indexes we find...
          for (unsigned i = 0; i != GEPOperands.size(); ++i)
            if (!isa<Constant>(GEPOperands[i]) ||
                isa<ConstantExpr>(GEPOperands[i]))
              GEPOperands[i] =Constant::getNullValue(GEPOperands[i]->getType());
          int64_t Offset = getTargetData().getIndexedOffset(BasePtr->getType(),
                                                            GEPOperands);
          if (Offset >= (int64_t)V2Size || Offset <= -(int64_t)V1Size)
            return NoAlias;
        }
      }
    }
  
  return MayAlias;
}

static bool ValuesEqual(Value *V1, Value *V2) {
  if (V1->getType() == V2->getType())
    return V1 == V2;
  if (Constant *C1 = dyn_cast<Constant>(V1))
    if (Constant *C2 = dyn_cast<Constant>(V2)) {
      // Sign extend the constants to long types.
      C1 = ConstantExpr::getSignExtend(C1, Type::LongTy);
      C2 = ConstantExpr::getSignExtend(C2, Type::LongTy);
      return C1 == C2;
    }
  return false;
}

/// CheckGEPInstructions - Check two GEP instructions with known must-aliasing
/// base pointers.  This checks to see if the index expressions preclude the
/// pointers from aliasing...
AliasAnalysis::AliasResult BasicAliasAnalysis::
CheckGEPInstructions(const Type* BasePtr1Ty, std::vector<Value*> &GEP1Ops,
                     unsigned G1S,
                     const Type *BasePtr2Ty, std::vector<Value*> &GEP2Ops,
                     unsigned G2S) {
  // We currently can't handle the case when the base pointers have different
  // primitive types.  Since this is uncommon anyway, we are happy being
  // extremely conservative.
  if (BasePtr1Ty != BasePtr2Ty)
    return MayAlias;

  const Type *GEPPointerTy = BasePtr1Ty;

  // Find the (possibly empty) initial sequence of equal values... which are not
  // necessarily constants.
  unsigned NumGEP1Operands = GEP1Ops.size(), NumGEP2Operands = GEP2Ops.size();
  unsigned MinOperands = std::min(NumGEP1Operands, NumGEP2Operands);
  unsigned MaxOperands = std::max(NumGEP1Operands, NumGEP2Operands);
  unsigned UnequalOper = 0;
  while (UnequalOper != MinOperands &&
         ValuesEqual(GEP1Ops[UnequalOper], GEP2Ops[UnequalOper])) {
    // Advance through the type as we go...
    ++UnequalOper;
    if (const CompositeType *CT = dyn_cast<CompositeType>(BasePtr1Ty))
      BasePtr1Ty = CT->getTypeAtIndex(GEP1Ops[UnequalOper-1]);
    else {
      // If all operands equal each other, then the derived pointers must
      // alias each other...
      BasePtr1Ty = 0;
      assert(UnequalOper == NumGEP1Operands && UnequalOper == NumGEP2Operands &&
             "Ran out of type nesting, but not out of operands?");
      return MustAlias;
    }
  }

  // If we have seen all constant operands, and run out of indexes on one of the
  // getelementptrs, check to see if the tail of the leftover one is all zeros.
  // If so, return mustalias.
  if (UnequalOper == MinOperands) {
    if (GEP1Ops.size() < GEP2Ops.size()) std::swap(GEP1Ops, GEP2Ops);
    
    bool AllAreZeros = true;
    for (unsigned i = UnequalOper; i != MaxOperands; ++i)
      if (!isa<Constant>(GEP1Ops[i]) ||
          !cast<Constant>(GEP1Ops[i])->isNullValue()) {
        AllAreZeros = false;
        break;
      }
    if (AllAreZeros) return MustAlias;
  }

    
  // So now we know that the indexes derived from the base pointers,
  // which are known to alias, are different.  We can still determine a
  // no-alias result if there are differing constant pairs in the index
  // chain.  For example:
  //        A[i][0] != A[j][1] iff (&A[0][1]-&A[0][0] >= std::max(G1S, G2S))
  //
  unsigned SizeMax = std::max(G1S, G2S);
  if (SizeMax == ~0U) return MayAlias; // Avoid frivolous work...

  // Scan for the first operand that is constant and unequal in the
  // two getelementptrs...
  unsigned FirstConstantOper = UnequalOper;
  for (; FirstConstantOper != MinOperands; ++FirstConstantOper) {
    const Value *G1Oper = GEP1Ops[FirstConstantOper];
    const Value *G2Oper = GEP2Ops[FirstConstantOper];
    
    if (G1Oper != G2Oper)   // Found non-equal constant indexes...
      if (Constant *G1OC = dyn_cast<Constant>(const_cast<Value*>(G1Oper)))
        if (Constant *G2OC = dyn_cast<Constant>(const_cast<Value*>(G2Oper))) {
          if (G1OC->getType() != G2OC->getType()) {
            // Sign extend both operands to long.
            G1OC = ConstantExpr::getSignExtend(G1OC, Type::LongTy);
            G2OC = ConstantExpr::getSignExtend(G2OC, Type::LongTy);
            GEP1Ops[FirstConstantOper] = G1OC;
            GEP2Ops[FirstConstantOper] = G2OC;
          }

          if (G1OC != G2OC) {
            // Make sure they are comparable (ie, not constant expressions)...
            // and make sure the GEP with the smaller leading constant is GEP1.
            Constant *Compare = ConstantExpr::getSetGT(G1OC, G2OC);
            if (ConstantBool *CV = dyn_cast<ConstantBool>(Compare)) {
              if (CV->getValue())   // If they are comparable and G2 > G1
                std::swap(GEP1Ops, GEP2Ops);  // Make GEP1 < GEP2
              break;
            }
          }
        }
    BasePtr1Ty = cast<CompositeType>(BasePtr1Ty)->getTypeAtIndex(G1Oper);
  }
  
  // No shared constant operands, and we ran out of common operands.  At this
  // point, the GEP instructions have run through all of their operands, and we
  // haven't found evidence that there are any deltas between the GEP's.
  // However, one GEP may have more operands than the other.  If this is the
  // case, there may still be hope.  Check this now.
  if (FirstConstantOper == MinOperands) {
    // Make GEP1Ops be the longer one if there is a longer one.
    if (GEP1Ops.size() < GEP2Ops.size())
      std::swap(GEP1Ops, GEP2Ops);

    // Is there anything to check?
    if (GEP1Ops.size() > MinOperands) {
      for (unsigned i = FirstConstantOper; i != MaxOperands; ++i)
        if (isa<Constant>(GEP1Ops[i]) && !isa<ConstantExpr>(GEP1Ops[i]) &&
            !cast<Constant>(GEP1Ops[i])->isNullValue()) {
          // Yup, there's a constant in the tail.  Set all variables to
          // constants in the GEP instruction to make it suiteable for
          // TargetData::getIndexedOffset.
          for (i = 0; i != MaxOperands; ++i)
            if (!isa<Constant>(GEP1Ops[i]) || isa<ConstantExpr>(GEP1Ops[i]))
              GEP1Ops[i] = Constant::getNullValue(GEP1Ops[i]->getType());
          // Okay, now get the offset.  This is the relative offset for the full
          // instruction.
          const TargetData &TD = getTargetData();
          int64_t Offset1 = TD.getIndexedOffset(GEPPointerTy, GEP1Ops);

          // Now crop off any constants from the end...
          GEP1Ops.resize(MinOperands);
          int64_t Offset2 = TD.getIndexedOffset(GEPPointerTy, GEP1Ops);
        
          // If the tail provided a bit enough offset, return noalias!
          if ((uint64_t)(Offset2-Offset1) >= SizeMax)
            return NoAlias;
        }
    }
    
    // Couldn't find anything useful.
    return MayAlias;
  }

  // If there are non-equal constants arguments, then we can figure
  // out a minimum known delta between the two index expressions... at
  // this point we know that the first constant index of GEP1 is less
  // than the first constant index of GEP2.

  // Advance BasePtr[12]Ty over this first differing constant operand.
  BasePtr2Ty = cast<CompositeType>(BasePtr1Ty)->getTypeAtIndex(GEP2Ops[FirstConstantOper]);
  BasePtr1Ty = cast<CompositeType>(BasePtr1Ty)->getTypeAtIndex(GEP1Ops[FirstConstantOper]);
  
  // We are going to be using TargetData::getIndexedOffset to determine the
  // offset that each of the GEP's is reaching.  To do this, we have to convert
  // all variable references to constant references.  To do this, we convert the
  // initial equal sequence of variables into constant zeros to start with.
  for (unsigned i = 0; i != FirstConstantOper; ++i) {
    if (!isa<Constant>(GEP1Ops[i]) || isa<ConstantExpr>(GEP1Ops[i]) ||
        !isa<Constant>(GEP2Ops[i]) || isa<ConstantExpr>(GEP2Ops[i]))
      GEP1Ops[i] = GEP2Ops[i] = Constant::getNullValue(Type::UIntTy);
  }

  // We know that GEP1Ops[FirstConstantOper] & GEP2Ops[FirstConstantOper] are ok
  
  // Loop over the rest of the operands...
  for (unsigned i = FirstConstantOper+1; i != MaxOperands; ++i) {
    const Value *Op1 = i < GEP1Ops.size() ? GEP1Ops[i] : 0;
    const Value *Op2 = i < GEP2Ops.size() ? GEP2Ops[i] : 0;
    // If they are equal, use a zero index...
    if (Op1 == Op2 && BasePtr1Ty == BasePtr2Ty) {
      if (!isa<Constant>(Op1) || isa<ConstantExpr>(Op1))
        GEP1Ops[i] = GEP2Ops[i] = Constant::getNullValue(Op1->getType());
      // Otherwise, just keep the constants we have.
    } else {
      if (Op1) {
        if (const ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
          // If this is an array index, make sure the array element is in range.
          if (const ArrayType *AT = dyn_cast<ArrayType>(BasePtr1Ty))
            if (Op1C->getRawValue() >= AT->getNumElements())
              return MayAlias;  // Be conservative with out-of-range accesses
          
        } else {
          // GEP1 is known to produce a value less than GEP2.  To be
          // conservatively correct, we must assume the largest possible
          // constant is used in this position.  This cannot be the initial
          // index to the GEP instructions (because we know we have at least one
          // element before this one with the different constant arguments), so
          // we know that the current index must be into either a struct or
          // array.  Because we know it's not constant, this cannot be a
          // structure index.  Because of this, we can calculate the maximum
          // value possible.
          //
          if (const ArrayType *AT = dyn_cast<ArrayType>(BasePtr1Ty))
            GEP1Ops[i] = ConstantSInt::get(Type::LongTy,AT->getNumElements()-1);
        }
      }
      
      if (Op2) {
        if (const ConstantInt *Op2C = dyn_cast<ConstantInt>(Op2)) {
          // If this is an array index, make sure the array element is in range.
          if (const ArrayType *AT = dyn_cast<ArrayType>(BasePtr1Ty))
            if (Op2C->getRawValue() >= AT->getNumElements())
              return MayAlias;  // Be conservative with out-of-range accesses
        } else {  // Conservatively assume the minimum value for this index
          GEP2Ops[i] = Constant::getNullValue(Op2->getType());
        }
      }
    }

    if (BasePtr1Ty && Op1) {
      if (const CompositeType *CT = dyn_cast<CompositeType>(BasePtr1Ty))
        BasePtr1Ty = CT->getTypeAtIndex(GEP1Ops[i]);
      else
        BasePtr1Ty = 0;
    }

    if (BasePtr2Ty && Op2) {
      if (const CompositeType *CT = dyn_cast<CompositeType>(BasePtr2Ty))
        BasePtr2Ty = CT->getTypeAtIndex(GEP2Ops[i]);
      else
        BasePtr2Ty = 0;
    }
  }
  
  int64_t Offset1 = getTargetData().getIndexedOffset(GEPPointerTy, GEP1Ops);
  int64_t Offset2 = getTargetData().getIndexedOffset(GEPPointerTy, GEP2Ops);
  assert(Offset1 < Offset2 &&"There is at least one different constant here!");

  if ((uint64_t)(Offset2-Offset1) >= SizeMax) {
    //std::cerr << "Determined that these two GEP's don't alias [" 
    //          << SizeMax << " bytes]: \n" << *GEP1 << *GEP2;
    return NoAlias;
  }
  return MayAlias;
}

namespace {
  struct StringCompare {
    bool operator()(const char *LHS, const char *RHS) {
      return strcmp(LHS, RHS) < 0;
    }
  };
}

// Note that this list cannot contain libm functions (such as acos and sqrt)
// that set errno on a domain or other error.
static const char *DoesntAccessMemoryTable[] = {
  // LLVM intrinsics:
  "llvm.frameaddress", "llvm.returnaddress", "llvm.readport", "llvm.isnan",

  "abs", "labs", "llabs", "imaxabs", "fabs", "fabsf", "fabsl",
  "trunc", "truncf", "truncl", "ldexp",
  
  "atan", "atanf", "atanl",   "atan2", "atan2f", "atan2l",
  "cbrt",
  "cos", "cosf", "cosl",      "cosh", "coshf", "coshl",
  "exp", "expf", "expl", 
  "hypot",
  "sin", "sinf", "sinl",      "sinh", "sinhf", "sinhl",
  "tan", "tanf", "tanl",      "tanh", "tanhf", "tanhl",

  // ctype.h
  "isalnum", "isalpha", "iscntrl", "isdigit", "isgraph", "islower", "isprint"
  "ispunct", "isspace", "isupper", "isxdigit", "tolower", "toupper",

  // wctype.h"
  "iswalnum", "iswalpha", "iswcntrl", "iswdigit", "iswgraph", "iswlower",
  "iswprint", "iswpunct", "iswspace", "iswupper", "iswxdigit",

  "iswctype", "towctrans", "towlower", "towupper", 

  "btowc", "wctob", 

  "isinf", "isnan", "finite",

  // C99 math functions
  "copysign", "copysignf", "copysignd",
  "nexttoward", "nexttowardf", "nexttowardd",
  "nextafter", "nextafterf", "nextafterd",

  // glibc functions:
  "__fpclassify", "__fpclassifyf", "__fpclassifyl",
  "__signbit", "__signbitf", "__signbitl",
};

static const unsigned DAMTableSize =
    sizeof(DoesntAccessMemoryTable)/sizeof(DoesntAccessMemoryTable[0]);

/// doesNotAccessMemory - Return true if we know that the function does not
/// access memory at all.  Since basicaa does no analysis, we can only do simple
/// things here.  In particular, if we have an external function with the name
/// of a standard C library function, we are allowed to assume it will be
/// resolved by libc, so we can hardcode some entries in here.
bool BasicAliasAnalysis::doesNotAccessMemory(Function *F) {
  if (!F->isExternal()) return false;

  static bool Initialized = false;
  if (!Initialized) {
    // Sort the table the first time through.
    std::sort(DoesntAccessMemoryTable, DoesntAccessMemoryTable+DAMTableSize,
              StringCompare());
    Initialized = true;
  }

  const char **Ptr = std::lower_bound(DoesntAccessMemoryTable,
                                      DoesntAccessMemoryTable+DAMTableSize,
                                      F->getName().c_str(), StringCompare());
  return Ptr != DoesntAccessMemoryTable+DAMTableSize && *Ptr == F->getName();
}


static const char *OnlyReadsMemoryTable[] = {
  "atoi", "atol", "atof", "atoll", "atoq", "a64l",
  "bcmp", "memcmp", "memchr", "memrchr", "wmemcmp", "wmemchr", 

  // Strings
  "strcmp", "strcasecmp", "strcoll", "strncmp", "strncasecmp",
  "strchr", "strcspn", "strlen", "strpbrk", "strrchr", "strspn", "strstr", 
  "index", "rindex",

  // Wide char strings
  "wcschr", "wcscmp", "wcscoll", "wcscspn", "wcslen", "wcsncmp", "wcspbrk",
  "wcsrchr", "wcsspn", "wcsstr", 

  // glibc
  "alphasort", "alphasort64", "versionsort", "versionsort64",

  // C99
  "nan", "nanf", "nand",

  // File I/O
  "feof", "ferror", "fileno",
  "feof_unlocked", "ferror_unlocked", "fileno_unlocked"
};

static const unsigned ORMTableSize =
    sizeof(OnlyReadsMemoryTable)/sizeof(OnlyReadsMemoryTable[0]);

bool BasicAliasAnalysis::onlyReadsMemory(Function *F) {
  if (doesNotAccessMemory(F)) return true;
  if (!F->isExternal()) return false;

  static bool Initialized = false;
  if (!Initialized) {
    // Sort the table the first time through.
    std::sort(OnlyReadsMemoryTable, OnlyReadsMemoryTable+ORMTableSize,
              StringCompare());
    Initialized = true;
  }

  const char **Ptr = std::lower_bound(OnlyReadsMemoryTable,
                                      OnlyReadsMemoryTable+ORMTableSize,
                                      F->getName().c_str(), StringCompare());
  return Ptr != OnlyReadsMemoryTable+ORMTableSize && *Ptr == F->getName();
}


