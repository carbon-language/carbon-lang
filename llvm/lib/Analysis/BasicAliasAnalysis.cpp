//===- llvm/Analysis/BasicAliasAnalysis.h - Alias Analysis Impl -*- C++ -*-===//
//
// This file defines the default implementation of the Alias Analysis interface
// that simply implements a few identities (two different globals cannot alias,
// etc), but otherwise does no analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/ConstantHandling.h"
#include "llvm/GlobalValue.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"

// Make sure that anything that uses AliasAnalysis pulls in this file...
void BasicAAStub() {}


namespace {
  struct BasicAliasAnalysis : public ImmutablePass, public AliasAnalysis {
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AliasAnalysis::getAnalysisUsage(AU);
    }
    
    virtual void initializePass();

    // alias - This is the only method here that does anything interesting...
    //
    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);
  private:
    // CheckGEPInstructions - Check two GEP instructions of compatible types and
    // equal number of arguments.  This checks to see if the index expressions
    // preclude the pointers from aliasing...
    AliasResult CheckGEPInstructions(GetElementPtrInst *GEP1, unsigned G1Size,
                                     GetElementPtrInst *GEP2, unsigned G2Size);
  };
 
  // Register this pass...
  RegisterOpt<BasicAliasAnalysis>
  X("basicaa", "Basic Alias Analysis (default AA impl)");

  // Declare that we implement the AliasAnalysis interface
  RegisterAnalysisGroup<AliasAnalysis, BasicAliasAnalysis, true> Y;
}  // End of anonymous namespace

void BasicAliasAnalysis::initializePass() {
  InitializeAliasAnalysis(this);
}



// hasUniqueAddress - Return true if the 
static inline bool hasUniqueAddress(const Value *V) {
  return isa<GlobalValue>(V) || isa<MallocInst>(V) || isa<AllocaInst>(V);
}

static const Value *getUnderlyingObject(const Value *V) {
  if (!isa<PointerType>(V->getType())) return 0;

  // If we are at some type of object... return it.
  if (hasUniqueAddress(V)) return V;
  
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


// alias - Provide a bunch of ad-hoc rules to disambiguate in common cases, such
// as array references.  Note that this function is heavily tail recursive.
// Hopefully we have a smart C++ compiler.  :)
//
AliasAnalysis::AliasResult
BasicAliasAnalysis::alias(const Value *V1, unsigned V1Size,
                          const Value *V2, unsigned V2Size) {
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
    // If they are two different objects, we know that we have no alias...
    if (O1 != O2) return NoAlias;

    // If they are the same object, they we can look at the indexes.  If they
    // index off of the object is the same for both pointers, they must alias.
    // If they are provably different, they must not alias.  Otherwise, we can't
    // tell anything.
  } else if (O1 && isa<ConstantPointerNull>(V2)) {
    return NoAlias;                    // Unique values don't alias null
  } else if (O2 && isa<ConstantPointerNull>(V1)) {
    return NoAlias;                    // Unique values don't alias null
  }

  // If we have two gep instructions with identical indices, return an alias
  // result equal to the alias result of the original pointer...
  //
  if (const GetElementPtrInst *GEP1 = dyn_cast<GetElementPtrInst>(V1))
    if (const GetElementPtrInst *GEP2 = dyn_cast<GetElementPtrInst>(V2))
      if (GEP1->getNumOperands() == GEP2->getNumOperands() &&
          GEP1->getOperand(0)->getType() == GEP2->getOperand(0)->getType()) {
        AliasResult GAlias =
          CheckGEPInstructions((GetElementPtrInst*)GEP1, V1Size,
                               (GetElementPtrInst*)GEP2, V2Size);
        if (GAlias != MayAlias)
          return GAlias;
      }

  // Check to see if these two pointers are related by a getelementptr
  // instruction.  If one pointer is a GEP with a non-zero index of the other
  // pointer, we know they cannot alias.
  //
  if (isa<GetElementPtrInst>(V2)) {
    std::swap(V1, V2);
    std::swap(V1Size, V2Size);
  }

  if (V1Size != ~0U && V2Size != ~0U)
    if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V1)) {
      AliasResult R = alias(GEP->getOperand(0), V1Size, V2, V2Size);
      if (R == MustAlias) {
        // If there is at least one non-zero constant index, we know they cannot
        // alias.
        bool ConstantFound = false;
        for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i)
          if (const Constant *C = dyn_cast<Constant>(GEP->getOperand(i)))
            if (!C->isNullValue()) {
              ConstantFound = true;
              break;
          }
        if (ConstantFound) {
          if (V2Size <= 1 && V1Size <= 1)  // Just pointer check?
            return NoAlias;
          
          // Otherwise we have to check to see that the distance is more than
          // the size of the argument... build an index vector that is equal to
          // the arguments provided, except substitute 0's for any variable
          // indexes we find...
          
          std::vector<Value*> Indices;
          Indices.reserve(GEP->getNumOperands()-1);
          for (unsigned i = 1; i != GEP->getNumOperands(); ++i)
            if (const Constant *C = dyn_cast<Constant>(GEP->getOperand(i)))
              Indices.push_back((Value*)C);
            else
              Indices.push_back(Constant::getNullValue(Type::LongTy));
          const Type *Ty = GEP->getOperand(0)->getType();
          int Offset = getTargetData().getIndexedOffset(Ty, Indices);
          if (Offset >= (int)V2Size || Offset <= -(int)V1Size)
            return NoAlias;
        }
      }
    }
  
  return MayAlias;
}

static Value *CheckArrayIndicesForOverflow(const Type *PtrTy,
                                           const std::vector<Value*> &Indices,
                                           const ConstantInt *Idx) {
  if (const ConstantSInt *IdxS = dyn_cast<ConstantSInt>(Idx)) {
    if (IdxS->getValue() < 0)   // Underflow on the array subscript?
      return Constant::getNullValue(Type::LongTy);
    else {                       // Check for overflow
      const ArrayType *ATy =
        cast<ArrayType>(GetElementPtrInst::getIndexedType(PtrTy, Indices,true));
      if (IdxS->getValue() >= (int64_t)ATy->getNumElements())
        return ConstantSInt::get(Type::LongTy, ATy->getNumElements()-1);
    }
  }
  return (Value*)Idx;  // Everything is acceptable.
}

// CheckGEPInstructions - Check two GEP instructions of compatible types and
// equal number of arguments.  This checks to see if the index expressions
// preclude the pointers from aliasing...
//
AliasAnalysis::AliasResult
BasicAliasAnalysis::CheckGEPInstructions(GetElementPtrInst *GEP1, unsigned G1S, 
                                         GetElementPtrInst *GEP2, unsigned G2S){
  // Do the base pointers alias?
  AliasResult BaseAlias = alias(GEP1->getOperand(0), G1S,
                                GEP2->getOperand(0), G2S);
  if (BaseAlias != MustAlias)   // No or May alias: We cannot add anything...
    return BaseAlias;
  
  // Find the (possibly empty) initial sequence of equal values...
  unsigned NumGEPOperands = GEP1->getNumOperands();
  unsigned UnequalOper = 1;
  while (UnequalOper != NumGEPOperands &&
         GEP1->getOperand(UnequalOper) == GEP2->getOperand(UnequalOper))
    ++UnequalOper;
    
  // If all operands equal each other, then the derived pointers must
  // alias each other...
  if (UnequalOper == NumGEPOperands) return MustAlias;
    
  // So now we know that the indexes derived from the base pointers,
  // which are known to alias, are different.  We can still determine a
  // no-alias result if there are differing constant pairs in the index
  // chain.  For example:
  //        A[i][0] != A[j][1] iff (&A[0][1]-&A[0][0] >= std::max(G1S, G2S))
  //
  unsigned SizeMax = std::max(G1S, G2S);
  if (SizeMax == ~0U) return MayAlias; // Avoid frivolous work...

  // Scan for the first operand that is constant and unequal in the
  // two getelemenptrs...
  unsigned FirstConstantOper = UnequalOper;
  for (; FirstConstantOper != NumGEPOperands; ++FirstConstantOper) {
    const Value *G1Oper = GEP1->getOperand(FirstConstantOper);
    const Value *G2Oper = GEP2->getOperand(FirstConstantOper);
    if (G1Oper != G2Oper &&   // Found non-equal constant indexes...
        isa<Constant>(G1Oper) && isa<Constant>(G2Oper)) {
      // Make sure they are comparable...  and make sure the GEP with
      // the smaller leading constant is GEP1.
      ConstantBool *Compare =
        *cast<Constant>(GEP1->getOperand(FirstConstantOper)) >
        *cast<Constant>(GEP2->getOperand(FirstConstantOper));
      if (Compare) {  // If they are comparable...
        if (Compare->getValue())
          std::swap(GEP1, GEP2);  // Make GEP1 < GEP2
        break;
      }
    }
  }
  
  // No constant operands, we cannot tell anything...
  if (FirstConstantOper == NumGEPOperands) return MayAlias;

  // If there are non-equal constants arguments, then we can figure
  // out a minimum known delta between the two index expressions... at
  // this point we know that the first constant index of GEP1 is less
  // than the first constant index of GEP2.
  //
  std::vector<Value*> Indices1;
  Indices1.reserve(NumGEPOperands-1);
  for (unsigned i = 1; i != FirstConstantOper; ++i)
    if (GEP1->getOperand(i)->getType() == Type::UByteTy)
      Indices1.push_back(GEP1->getOperand(i));
    else
      Indices1.push_back(Constant::getNullValue(Type::LongTy));
  std::vector<Value*> Indices2;
  Indices2.reserve(NumGEPOperands-1);
  Indices2 = Indices1;           // Copy the zeros prefix...
  
  // Add the two known constant operands...
  Indices1.push_back((Value*)GEP1->getOperand(FirstConstantOper));
  Indices2.push_back((Value*)GEP2->getOperand(FirstConstantOper));
  
  const Type *GEPPointerTy = GEP1->getOperand(0)->getType();
  
  // Loop over the rest of the operands...
  for (unsigned i = FirstConstantOper+1; i != NumGEPOperands; ++i) {
    const Value *Op1 = GEP1->getOperand(i);
    const Value *Op2 = GEP2->getOperand(i);
    if (Op1 == Op2) {   // If they are equal, use a zero index...
      if (!isa<Constant>(Op1)) {
        Indices1.push_back(Constant::getNullValue(Op1->getType()));
        Indices2.push_back(Indices1.back());
      } else {
        Indices1.push_back((Value*)Op1);
        Indices2.push_back((Value*)Op2);
      }
    } else {
      if (const ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
        // If this is an array index, make sure the array element is in range...
        if (i != 1)   // The pointer index can be "out of range"
          Op1 = CheckArrayIndicesForOverflow(GEPPointerTy, Indices1, Op1C);

        Indices1.push_back((Value*)Op1);
      } else {
        // GEP1 is known to produce a value less than GEP2.  To be
        // conservatively correct, we must assume the largest possible constant
        // is used in this position.  This cannot be the initial index to the
        // GEP instructions (because we know we have at least one element before
        // this one with the different constant arguments), so we know that the
        // current index must be into either a struct or array.  Because we know
        // it's not constant, this cannot be a structure index.  Because of
        // this, we can calculate the maximum value possible.
        //
        const ArrayType *ElTy =
          cast<ArrayType>(GEP1->getIndexedType(GEPPointerTy, Indices1, true));
        Indices1.push_back(ConstantSInt::get(Type::LongTy,
                                             ElTy->getNumElements()-1));
      }
      
      if (const ConstantInt *Op1C = dyn_cast<ConstantInt>(Op2)) {
        // If this is an array index, make sure the array element is in range...
        if (i != 1)   // The pointer index can be "out of range"
          Op1 = CheckArrayIndicesForOverflow(GEPPointerTy, Indices2, Op1C);

        Indices2.push_back((Value*)Op2);
      }
      else // Conservatively assume the minimum value for this index
        Indices2.push_back(Constant::getNullValue(Op2->getType()));
    }
  }
  
  int64_t Offset1 = getTargetData().getIndexedOffset(GEPPointerTy, Indices1);
  int64_t Offset2 = getTargetData().getIndexedOffset(GEPPointerTy, Indices2);
  assert(Offset1 < Offset2 &&"There is at least one different constant here!");

  if ((uint64_t)(Offset2-Offset1) >= SizeMax) {
    //std::cerr << "Determined that these two GEP's don't alias [" 
    //          << SizeMax << " bytes]: \n" << *GEP1 << *GEP2;
    return NoAlias;
  }
  return MayAlias;
}

