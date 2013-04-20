//===-- Analysis.cpp - CodeGen LLVM IR Analysis Utilities -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines several CodeGen-specific LLVM IR analysis utilties.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Analysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

/// ComputeLinearIndex - Given an LLVM IR aggregate type and a sequence
/// of insertvalue or extractvalue indices that identify a member, return
/// the linearized index of the start of the member.
///
unsigned llvm::ComputeLinearIndex(Type *Ty,
                                  const unsigned *Indices,
                                  const unsigned *IndicesEnd,
                                  unsigned CurIndex) {
  // Base case: We're done.
  if (Indices && Indices == IndicesEnd)
    return CurIndex;

  // Given a struct type, recursively traverse the elements.
  if (StructType *STy = dyn_cast<StructType>(Ty)) {
    for (StructType::element_iterator EB = STy->element_begin(),
                                      EI = EB,
                                      EE = STy->element_end();
        EI != EE; ++EI) {
      if (Indices && *Indices == unsigned(EI - EB))
        return ComputeLinearIndex(*EI, Indices+1, IndicesEnd, CurIndex);
      CurIndex = ComputeLinearIndex(*EI, 0, 0, CurIndex);
    }
    return CurIndex;
  }
  // Given an array type, recursively traverse the elements.
  else if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    Type *EltTy = ATy->getElementType();
    for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i) {
      if (Indices && *Indices == i)
        return ComputeLinearIndex(EltTy, Indices+1, IndicesEnd, CurIndex);
      CurIndex = ComputeLinearIndex(EltTy, 0, 0, CurIndex);
    }
    return CurIndex;
  }
  // We haven't found the type we're looking for, so keep searching.
  return CurIndex + 1;
}

/// ComputeValueVTs - Given an LLVM IR type, compute a sequence of
/// EVTs that represent all the individual underlying
/// non-aggregate types that comprise it.
///
/// If Offsets is non-null, it points to a vector to be filled in
/// with the in-memory offsets of each of the individual values.
///
void llvm::ComputeValueVTs(const TargetLowering &TLI, Type *Ty,
                           SmallVectorImpl<EVT> &ValueVTs,
                           SmallVectorImpl<uint64_t> *Offsets,
                           uint64_t StartingOffset) {
  // Given a struct type, recursively traverse the elements.
  if (StructType *STy = dyn_cast<StructType>(Ty)) {
    const StructLayout *SL = TLI.getDataLayout()->getStructLayout(STy);
    for (StructType::element_iterator EB = STy->element_begin(),
                                      EI = EB,
                                      EE = STy->element_end();
         EI != EE; ++EI)
      ComputeValueVTs(TLI, *EI, ValueVTs, Offsets,
                      StartingOffset + SL->getElementOffset(EI - EB));
    return;
  }
  // Given an array type, recursively traverse the elements.
  if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    Type *EltTy = ATy->getElementType();
    uint64_t EltSize = TLI.getDataLayout()->getTypeAllocSize(EltTy);
    for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i)
      ComputeValueVTs(TLI, EltTy, ValueVTs, Offsets,
                      StartingOffset + i * EltSize);
    return;
  }
  // Interpret void as zero return values.
  if (Ty->isVoidTy())
    return;
  // Base case: we can get an EVT for this LLVM IR type.
  ValueVTs.push_back(TLI.getValueType(Ty));
  if (Offsets)
    Offsets->push_back(StartingOffset);
}

/// ExtractTypeInfo - Returns the type info, possibly bitcast, encoded in V.
GlobalVariable *llvm::ExtractTypeInfo(Value *V) {
  V = V->stripPointerCasts();
  GlobalVariable *GV = dyn_cast<GlobalVariable>(V);

  if (GV && GV->getName() == "llvm.eh.catch.all.value") {
    assert(GV->hasInitializer() &&
           "The EH catch-all value must have an initializer");
    Value *Init = GV->getInitializer();
    GV = dyn_cast<GlobalVariable>(Init);
    if (!GV) V = cast<ConstantPointerNull>(Init);
  }

  assert((GV || isa<ConstantPointerNull>(V)) &&
         "TypeInfo must be a global variable or NULL");
  return GV;
}

/// hasInlineAsmMemConstraint - Return true if the inline asm instruction being
/// processed uses a memory 'm' constraint.
bool
llvm::hasInlineAsmMemConstraint(InlineAsm::ConstraintInfoVector &CInfos,
                                const TargetLowering &TLI) {
  for (unsigned i = 0, e = CInfos.size(); i != e; ++i) {
    InlineAsm::ConstraintInfo &CI = CInfos[i];
    for (unsigned j = 0, ee = CI.Codes.size(); j != ee; ++j) {
      TargetLowering::ConstraintType CType = TLI.getConstraintType(CI.Codes[j]);
      if (CType == TargetLowering::C_Memory)
        return true;
    }

    // Indirect operand accesses access memory.
    if (CI.isIndirect)
      return true;
  }

  return false;
}

/// getFCmpCondCode - Return the ISD condition code corresponding to
/// the given LLVM IR floating-point condition code.  This includes
/// consideration of global floating-point math flags.
///
ISD::CondCode llvm::getFCmpCondCode(FCmpInst::Predicate Pred) {
  switch (Pred) {
  case FCmpInst::FCMP_FALSE: return ISD::SETFALSE;
  case FCmpInst::FCMP_OEQ:   return ISD::SETOEQ;
  case FCmpInst::FCMP_OGT:   return ISD::SETOGT;
  case FCmpInst::FCMP_OGE:   return ISD::SETOGE;
  case FCmpInst::FCMP_OLT:   return ISD::SETOLT;
  case FCmpInst::FCMP_OLE:   return ISD::SETOLE;
  case FCmpInst::FCMP_ONE:   return ISD::SETONE;
  case FCmpInst::FCMP_ORD:   return ISD::SETO;
  case FCmpInst::FCMP_UNO:   return ISD::SETUO;
  case FCmpInst::FCMP_UEQ:   return ISD::SETUEQ;
  case FCmpInst::FCMP_UGT:   return ISD::SETUGT;
  case FCmpInst::FCMP_UGE:   return ISD::SETUGE;
  case FCmpInst::FCMP_ULT:   return ISD::SETULT;
  case FCmpInst::FCMP_ULE:   return ISD::SETULE;
  case FCmpInst::FCMP_UNE:   return ISD::SETUNE;
  case FCmpInst::FCMP_TRUE:  return ISD::SETTRUE;
  default: llvm_unreachable("Invalid FCmp predicate opcode!");
  }
}

ISD::CondCode llvm::getFCmpCodeWithoutNaN(ISD::CondCode CC) {
  switch (CC) {
    case ISD::SETOEQ: case ISD::SETUEQ: return ISD::SETEQ;
    case ISD::SETONE: case ISD::SETUNE: return ISD::SETNE;
    case ISD::SETOLT: case ISD::SETULT: return ISD::SETLT;
    case ISD::SETOLE: case ISD::SETULE: return ISD::SETLE;
    case ISD::SETOGT: case ISD::SETUGT: return ISD::SETGT;
    case ISD::SETOGE: case ISD::SETUGE: return ISD::SETGE;
    default: return CC;
  }
}

/// getICmpCondCode - Return the ISD condition code corresponding to
/// the given LLVM IR integer condition code.
///
ISD::CondCode llvm::getICmpCondCode(ICmpInst::Predicate Pred) {
  switch (Pred) {
  case ICmpInst::ICMP_EQ:  return ISD::SETEQ;
  case ICmpInst::ICMP_NE:  return ISD::SETNE;
  case ICmpInst::ICMP_SLE: return ISD::SETLE;
  case ICmpInst::ICMP_ULE: return ISD::SETULE;
  case ICmpInst::ICMP_SGE: return ISD::SETGE;
  case ICmpInst::ICMP_UGE: return ISD::SETUGE;
  case ICmpInst::ICMP_SLT: return ISD::SETLT;
  case ICmpInst::ICMP_ULT: return ISD::SETULT;
  case ICmpInst::ICMP_SGT: return ISD::SETGT;
  case ICmpInst::ICMP_UGT: return ISD::SETUGT;
  default:
    llvm_unreachable("Invalid ICmp predicate opcode!");
  }
}

static bool isNoopBitcast(Type *T1, Type *T2,
                          const TargetLowering& TLI) {
  return T1 == T2 || (T1->isPointerTy() && T2->isPointerTy()) ||
         (isa<VectorType>(T1) && isa<VectorType>(T2) &&
          TLI.isTypeLegal(EVT::getEVT(T1)) && TLI.isTypeLegal(EVT::getEVT(T2)));
}

/// sameNoopInput - Return true if V1 == V2, else if either V1 or V2 is a noop
/// (i.e., lowers to no machine code), look through it (and any transitive noop
/// operands to it) and check if it has the same noop input value.  This is
/// used to determine if a tail call can be formed.
static bool sameNoopInput(const Value *V1, const Value *V2,
                          SmallVectorImpl<unsigned> &Els1,
                          SmallVectorImpl<unsigned> &Els2,
                          const TargetLowering &TLI) {
  using std::swap;
  bool swapParity = false;
  bool equalEls = Els1 == Els2;
  while (true) {
    if ((equalEls && V1 == V2) || isa<UndefValue>(V1) || isa<UndefValue>(V2)) {
      if (swapParity)
        // Revert to original Els1 and Els2 to avoid confusing recursive calls
        swap(Els1, Els2);
      return true;
    }

    // Try to look through V1; if V1 is not an instruction, it can't be looked
    // through.
    const Instruction *I = dyn_cast<Instruction>(V1);
    const Value *NoopInput = 0;
    if (I != 0 && I->getNumOperands() > 0) {
     Value *Op = I->getOperand(0);
      if (isa<TruncInst>(I)) {
        // Look through truly no-op truncates.
        if (TLI.isTruncateFree(Op->getType(), I->getType()))
          NoopInput = Op;
      } else if (isa<BitCastInst>(I)) {
        // Look through truly no-op bitcasts.
        if (isNoopBitcast(Op->getType(), I->getType(), TLI))
          NoopInput = Op;
      } else if (isa<GetElementPtrInst>(I)) {
        // Look through getelementptr
        if (cast<GetElementPtrInst>(I)->hasAllZeroIndices())
          NoopInput = Op;
      } else if (isa<IntToPtrInst>(I)) {
        // Look through inttoptr.
        // Make sure this isn't a truncating or extending cast.  We could
        // support this eventually, but don't bother for now.
        if (!isa<VectorType>(I->getType()) &&
            TLI.getPointerTy().getSizeInBits() == 
              cast<IntegerType>(Op->getType())->getBitWidth())
          NoopInput = Op;
      } else if (isa<PtrToIntInst>(I)) {
        // Look through ptrtoint.
        // Make sure this isn't a truncating or extending cast.  We could
        // support this eventually, but don't bother for now.
        if (!isa<VectorType>(I->getType()) &&
            TLI.getPointerTy().getSizeInBits() == 
              cast<IntegerType>(I->getType())->getBitWidth())
          NoopInput = Op;
      }
    }

    if (NoopInput) {
      V1 = NoopInput;
      continue;
    }

    // If we already swapped, avoid infinite loop
    if (swapParity)
      break;

    // Otherwise, swap V1<->V2, Els1<->Els2
    swap(V1, V2);
    swap(Els1, Els2);
    swapParity = !swapParity;
  }

  for (unsigned n = 0; n < 2; ++n) {
    if (isa<InsertValueInst>(V1)) {
      if (isa<StructType>(V1->getType())) {
        // Look through insertvalue
        unsigned i, e;
        for (i = 0, e = cast<StructType>(V1->getType())->getNumElements();
             i != e; ++i) {
          const Value *InScalar = FindInsertedValue(const_cast<Value*>(V1), i);
          if (InScalar == 0)
            break;
          Els1.push_back(i);
          if (!sameNoopInput(InScalar, V2, Els1, Els2, TLI)) {
            Els1.pop_back();
            break;
          }
          Els1.pop_back();
        }
        if (i == e) {
          if (swapParity)
            swap(Els1, Els2);
          return true;
        }
      }
    } else if (!Els1.empty() && isa<ExtractValueInst>(V1)) {
      const ExtractValueInst *EVI = cast<ExtractValueInst>(V1);
      unsigned i = Els1.back();
      // If the scalar value being inserted is an extractvalue of the right
      // index from the call, then everything is good.
      if (isa<StructType>(EVI->getOperand(0)->getType()) &&
          EVI->getNumIndices() == 1 && EVI->getIndices()[0] == i) {
        // Look through extractvalue
        Els1.pop_back();
        if (sameNoopInput(EVI->getOperand(0), V2, Els1, Els2, TLI)) {
          Els1.push_back(i);
          if (swapParity)
            swap(Els1, Els2);
          return true;
        }
        Els1.push_back(i);
      }
    }

    swap(V1, V2);
    swap(Els1, Els2);
    swapParity = !swapParity;
  }

  if (swapParity)
    swap(Els1, Els2);
  return false;
}

/// Test if the given instruction is in a position to be optimized
/// with a tail-call. This roughly means that it's in a block with
/// a return and there's nothing that needs to be scheduled
/// between it and the return.
///
/// This function only tests target-independent requirements.
bool llvm::isInTailCallPosition(ImmutableCallSite CS,
                                const TargetLowering &TLI) {
  const Instruction *I = CS.getInstruction();
  const BasicBlock *ExitBB = I->getParent();
  const TerminatorInst *Term = ExitBB->getTerminator();
  const ReturnInst *Ret = dyn_cast<ReturnInst>(Term);

  // The block must end in a return statement or unreachable.
  //
  // FIXME: Decline tailcall if it's not guaranteed and if the block ends in
  // an unreachable, for now. The way tailcall optimization is currently
  // implemented means it will add an epilogue followed by a jump. That is
  // not profitable. Also, if the callee is a special function (e.g.
  // longjmp on x86), it can end up causing miscompilation that has not
  // been fully understood.
  if (!Ret &&
      (!TLI.getTargetMachine().Options.GuaranteedTailCallOpt ||
       !isa<UnreachableInst>(Term)))
    return false;

  // If I will have a chain, make sure no other instruction that will have a
  // chain interposes between I and the return.
  if (I->mayHaveSideEffects() || I->mayReadFromMemory() ||
      !isSafeToSpeculativelyExecute(I))
    for (BasicBlock::const_iterator BBI = prior(prior(ExitBB->end())); ;
         --BBI) {
      if (&*BBI == I)
        break;
      // Debug info intrinsics do not get in the way of tail call optimization.
      if (isa<DbgInfoIntrinsic>(BBI))
        continue;
      if (BBI->mayHaveSideEffects() || BBI->mayReadFromMemory() ||
          !isSafeToSpeculativelyExecute(BBI))
        return false;
    }

  // If the block ends with a void return or unreachable, it doesn't matter
  // what the call's return type is.
  if (!Ret || Ret->getNumOperands() == 0) return true;

  // If the return value is undef, it doesn't matter what the call's
  // return type is.
  if (isa<UndefValue>(Ret->getOperand(0))) return true;

  // Conservatively require the attributes of the call to match those of
  // the return. Ignore noalias because it doesn't affect the call sequence.
  const Function *F = ExitBB->getParent();
  AttributeSet CallerAttrs = F->getAttributes();
  if (AttrBuilder(CallerAttrs, AttributeSet::ReturnIndex).
        removeAttribute(Attribute::NoAlias) !=
      AttrBuilder(CallerAttrs, AttributeSet::ReturnIndex).
        removeAttribute(Attribute::NoAlias))
    return false;

  // It's not safe to eliminate the sign / zero extension of the return value.
  if (CallerAttrs.hasAttribute(AttributeSet::ReturnIndex, Attribute::ZExt) ||
      CallerAttrs.hasAttribute(AttributeSet::ReturnIndex, Attribute::SExt))
    return false;

  // Otherwise, make sure the return value and I have the same value
  SmallVector<unsigned, 4> Els1, Els2;
  return sameNoopInput(Ret->getOperand(0), I, Els1, Els2, TLI);
}
