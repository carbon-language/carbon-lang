//===- AliasAnalysis.cpp - Generic Alias Analysis Interface Implementation -==//
//
// This file implements the generic AliasAnalysis interface which is used as the
// common interface used by all clients and implementations of alias analysis.
//
// This file also implements the default version of the AliasAnalysis interface
// that is to be used when no other implementation is specified.  This does some
// simple tests that detect obvious cases: two different global pointers cannot
// alias, a global cannot alias a malloc, two different mallocs cannot alias,
// etc.
//
// This alias analysis implementation really isn't very good for anything, but
// it is very fast, and makes a nice clean default implementation.  Because it
// handles lots of little corner cases, other, more complex, alias analysis
// implementations may choose to rely on this pass to resolve these simple and
// easy cases.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/BasicBlock.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/Constants.h"
#include "llvm/ConstantHandling.h"
#include "llvm/GlobalValue.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"

// Register the AliasAnalysis interface, providing a nice name to refer to.
namespace {
  RegisterAnalysisGroup<AliasAnalysis> Z("Alias Analysis");
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(LoadInst *L, Value *P, unsigned Size) {
  return alias(L->getOperand(0), TD->getTypeSize(L->getType()),
               P, Size) ? Ref : NoModRef;
}

AliasAnalysis::ModRefResult
AliasAnalysis::getModRefInfo(StoreInst *S, Value *P, unsigned Size) {
  return alias(S->getOperand(1), TD->getTypeSize(S->getOperand(0)->getType()),
               P, Size) ? Mod : NoModRef;
}


// AliasAnalysis destructor: DO NOT move this to the header file for
// AliasAnalysis or else clients of the AliasAnalysis class may not depend on
// the AliasAnalysis.o file in the current .a file, causing alias analysis
// support to not be included in the tool correctly!
//
AliasAnalysis::~AliasAnalysis() {}

/// setTargetData - Subclasses must call this method to initialize the
/// AliasAnalysis interface before any other methods are called.
///
void AliasAnalysis::InitializeAliasAnalysis(Pass *P) {
  TD = &P->getAnalysis<TargetData>();
}

// getAnalysisUsage - All alias analysis implementations should invoke this
// directly (using AliasAnalysis::getAnalysisUsage(AU)) to make sure that
// TargetData is required by the pass.
void AliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetData>();            // All AA's need TargetData.
}

/// canBasicBlockModify - Return true if it is possible for execution of the
/// specified basic block to modify the value pointed to by Ptr.
///
bool AliasAnalysis::canBasicBlockModify(const BasicBlock &BB,
                                        const Value *Ptr, unsigned Size) {
  return canInstructionRangeModify(BB.front(), BB.back(), Ptr, Size);
}

/// canInstructionRangeModify - Return true if it is possible for the execution
/// of the specified instructions to modify the value pointed to by Ptr.  The
/// instructions to consider are all of the instructions in the range of [I1,I2]
/// INCLUSIVE.  I1 and I2 must be in the same basic block.
///
bool AliasAnalysis::canInstructionRangeModify(const Instruction &I1,
                                              const Instruction &I2,
                                              const Value *Ptr, unsigned Size) {
  assert(I1.getParent() == I2.getParent() &&
         "Instructions not in same basic block!");
  BasicBlock::iterator I = const_cast<Instruction*>(&I1);
  BasicBlock::iterator E = const_cast<Instruction*>(&I2);
  ++E;  // Convert from inclusive to exclusive range.

  for (; I != E; ++I) // Check every instruction in range
    if (getModRefInfo(I, const_cast<Value*>(Ptr), Size) & Mod)
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// BasicAliasAnalysis Pass Implementation
//===----------------------------------------------------------------------===//
//
// Because of the way .a files work, the implementation of the
// BasicAliasAnalysis class MUST be in the AliasAnalysis file itself, or else we
// run the risk of AliasAnalysis being used, but the default implementation not
// being linked into the tool that uses it.  As such, we register and implement
// the class here.
//
namespace {
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

  // Pointing at a discernable object?
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

  if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V1))
    if (GEP->getOperand(0) == V2) {
      // If there is at least one non-zero constant index, we know they cannot
      // alias.
      for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i)
        if (const Constant *C = dyn_cast<Constant>(GEP->getOperand(i)))
          if (!C->isNullValue())
            return NoAlias;
    }

  return MayAlias;
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
    Indices1.push_back(Constant::getNullValue(GEP1->getOperand(i)
                                              ->getType()));
  std::vector<Value*> Indices2;
  Indices2.reserve(NumGEPOperands-1);
  Indices2 = Indices1;           // Copy the zeros prefix...
  
  // Add the two known constant operands...
  Indices1.push_back((Value*)GEP1->getOperand(FirstConstantOper));
  Indices2.push_back((Value*)GEP2->getOperand(FirstConstantOper));
  
  const Type *GEPPointerTy = GEP1->getOperand(0)->getType();
  
  // Loop over the rest of the operands...
  for (unsigned i = FirstConstantOper+1; i!=NumGEPOperands; ++i){
    const Value *Op1 = GEP1->getOperand(i);
    const Value *Op2 = GEP1->getOperand(i);
    if (Op1 == Op2) {   // If they are equal, use a zero index...
      Indices1.push_back(Constant::getNullValue(Op1->getType()));
      Indices2.push_back(Indices1.back());
    } else {
      if (isa<Constant>(Op1))
        Indices1.push_back((Value*)Op1);
      else {
        // GEP1 is known to produce a value less than GEP2.  To be
        // conservatively correct, we must assume the largest
        // possible constant is used in this position.  This cannot
        // be the initial index to the GEP instructions (because we
        // know we have at least one element before this one with
        // the different constant arguments), so we know that the
        // current index must be into either a struct or array.
        // Because of this, we can calculate the maximum value
        // possible.
        //
        const Type *ElTy = GEP1->getIndexedType(GEPPointerTy,
                                                Indices1, true);
        if (const StructType *STy = dyn_cast<StructType>(ElTy)) {
          Indices1.push_back(ConstantUInt::get(Type::UByteTy,
                                               STy->getNumContainedTypes()));
        } else {
          Indices1.push_back(ConstantSInt::get(Type::LongTy,
                                               cast<ArrayType>(ElTy)->getNumElements()));
        }
      }
      
      if (isa<Constant>(Op2))
        Indices2.push_back((Value*)Op2);
      else // Conservatively assume the minimum value for this index
        Indices2.push_back(Constant::getNullValue(Op1->getType()));
    }
  }
  
  unsigned Offset1 = getTargetData().getIndexedOffset(GEPPointerTy, Indices1);
  unsigned Offset2 = getTargetData().getIndexedOffset(GEPPointerTy, Indices2);
  assert(Offset1 < Offset2 &&"There is at least one different constant here!");

  if (Offset2-Offset1 >= SizeMax) {
    //std::cerr << "Determined that these two GEP's don't alias [" 
    //          << SizeMax << " bytes]: \n" << *GEP1 << *GEP2;
    return NoAlias;
  }
  return MayAlias;
}

