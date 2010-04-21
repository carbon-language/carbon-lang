//===-- Analysis.cpp - CodeGen LLVM IR Analysis Utilities --*- C++ ------*-===//
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
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

/// ComputeLinearIndex - Given an LLVM IR aggregate type and a sequence
/// of insertvalue or extractvalue indices that identify a member, return
/// the linearized index of the start of the member.
///
unsigned llvm::ComputeLinearIndex(const TargetLowering &TLI, const Type *Ty,
                                  const unsigned *Indices,
                                  const unsigned *IndicesEnd,
                                  unsigned CurIndex) {
  // Base case: We're done.
  if (Indices && Indices == IndicesEnd)
    return CurIndex;

  // Given a struct type, recursively traverse the elements.
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    for (StructType::element_iterator EB = STy->element_begin(),
                                      EI = EB,
                                      EE = STy->element_end();
        EI != EE; ++EI) {
      if (Indices && *Indices == unsigned(EI - EB))
        return ComputeLinearIndex(TLI, *EI, Indices+1, IndicesEnd, CurIndex);
      CurIndex = ComputeLinearIndex(TLI, *EI, 0, 0, CurIndex);
    }
    return CurIndex;
  }
  // Given an array type, recursively traverse the elements.
  else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    const Type *EltTy = ATy->getElementType();
    for (unsigned i = 0, e = ATy->getNumElements(); i != e; ++i) {
      if (Indices && *Indices == i)
        return ComputeLinearIndex(TLI, EltTy, Indices+1, IndicesEnd, CurIndex);
      CurIndex = ComputeLinearIndex(TLI, EltTy, 0, 0, CurIndex);
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
void llvm::ComputeValueVTs(const TargetLowering &TLI, const Type *Ty,
                           SmallVectorImpl<EVT> &ValueVTs,
                           SmallVectorImpl<uint64_t> *Offsets,
                           uint64_t StartingOffset) {
  // Given a struct type, recursively traverse the elements.
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    const StructLayout *SL = TLI.getTargetData()->getStructLayout(STy);
    for (StructType::element_iterator EB = STy->element_begin(),
                                      EI = EB,
                                      EE = STy->element_end();
         EI != EE; ++EI)
      ComputeValueVTs(TLI, *EI, ValueVTs, Offsets,
                      StartingOffset + SL->getElementOffset(EI - EB));
    return;
  }
  // Given an array type, recursively traverse the elements.
  if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    const Type *EltTy = ATy->getElementType();
    uint64_t EltSize = TLI.getTargetData()->getTypeAllocSize(EltTy);
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

  if (GV && GV->getName() == ".llvm.eh.catch.all.value") {
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
llvm::hasInlineAsmMemConstraint(std::vector<InlineAsm::ConstraintInfo> &CInfos,
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
  ISD::CondCode FPC, FOC;
  switch (Pred) {
  case FCmpInst::FCMP_FALSE: FOC = FPC = ISD::SETFALSE; break;
  case FCmpInst::FCMP_OEQ:   FOC = ISD::SETEQ; FPC = ISD::SETOEQ; break;
  case FCmpInst::FCMP_OGT:   FOC = ISD::SETGT; FPC = ISD::SETOGT; break;
  case FCmpInst::FCMP_OGE:   FOC = ISD::SETGE; FPC = ISD::SETOGE; break;
  case FCmpInst::FCMP_OLT:   FOC = ISD::SETLT; FPC = ISD::SETOLT; break;
  case FCmpInst::FCMP_OLE:   FOC = ISD::SETLE; FPC = ISD::SETOLE; break;
  case FCmpInst::FCMP_ONE:   FOC = ISD::SETNE; FPC = ISD::SETONE; break;
  case FCmpInst::FCMP_ORD:   FOC = FPC = ISD::SETO;   break;
  case FCmpInst::FCMP_UNO:   FOC = FPC = ISD::SETUO;  break;
  case FCmpInst::FCMP_UEQ:   FOC = ISD::SETEQ; FPC = ISD::SETUEQ; break;
  case FCmpInst::FCMP_UGT:   FOC = ISD::SETGT; FPC = ISD::SETUGT; break;
  case FCmpInst::FCMP_UGE:   FOC = ISD::SETGE; FPC = ISD::SETUGE; break;
  case FCmpInst::FCMP_ULT:   FOC = ISD::SETLT; FPC = ISD::SETULT; break;
  case FCmpInst::FCMP_ULE:   FOC = ISD::SETLE; FPC = ISD::SETULE; break;
  case FCmpInst::FCMP_UNE:   FOC = ISD::SETNE; FPC = ISD::SETUNE; break;
  case FCmpInst::FCMP_TRUE:  FOC = FPC = ISD::SETTRUE; break;
  default:
    llvm_unreachable("Invalid FCmp predicate opcode!");
    FOC = FPC = ISD::SETFALSE;
    break;
  }
  if (FiniteOnlyFPMath())
    return FOC;
  else
    return FPC;
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
    return ISD::SETNE;
  }
}

/// Test if the given instruction is in a position to be optimized
/// with a tail-call. This roughly means that it's in a block with
/// a return and there's nothing that needs to be scheduled
/// between it and the return.
///
/// This function only tests target-independent requirements.
bool llvm::isInTailCallPosition(ImmutableCallSite CS, Attributes CalleeRetAttr,
                                const TargetLowering &TLI) {
  const Instruction *I = CS.getInstruction();
  const BasicBlock *ExitBB = I->getParent();
  const TerminatorInst *Term = ExitBB->getTerminator();
  const ReturnInst *Ret = dyn_cast<ReturnInst>(Term);
  const Function *F = ExitBB->getParent();

  // The block must end in a return statement or unreachable.
  //
  // FIXME: Decline tailcall if it's not guaranteed and if the block ends in
  // an unreachable, for now. The way tailcall optimization is currently
  // implemented means it will add an epilogue followed by a jump. That is
  // not profitable. Also, if the callee is a special function (e.g.
  // longjmp on x86), it can end up causing miscompilation that has not
  // been fully understood.
  if (!Ret &&
      (!GuaranteedTailCallOpt || !isa<UnreachableInst>(Term))) return false;

  // If I will have a chain, make sure no other instruction that will have a
  // chain interposes between I and the return.
  if (I->mayHaveSideEffects() || I->mayReadFromMemory() ||
      !I->isSafeToSpeculativelyExecute())
    for (BasicBlock::const_iterator BBI = prior(prior(ExitBB->end())); ;
         --BBI) {
      if (&*BBI == I)
        break;
      // Debug info intrinsics do not get in the way of tail call optimization.
      if (isa<DbgInfoIntrinsic>(BBI))
        continue;
      if (BBI->mayHaveSideEffects() || BBI->mayReadFromMemory() ||
          !BBI->isSafeToSpeculativelyExecute())
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
  unsigned CallerRetAttr = F->getAttributes().getRetAttributes();
  if ((CalleeRetAttr ^ CallerRetAttr) & ~Attribute::NoAlias)
    return false;

  // It's not safe to eliminate the sign / zero extension of the return value.
  if ((CallerRetAttr & Attribute::ZExt) || (CallerRetAttr & Attribute::SExt))
    return false;

  // Otherwise, make sure the unmodified return value of I is the return value.
  for (const Instruction *U = dyn_cast<Instruction>(Ret->getOperand(0)); ;
       U = dyn_cast<Instruction>(U->getOperand(0))) {
    if (!U)
      return false;
    if (!U->hasOneUse())
      return false;
    if (U == I)
      break;
    // Check for a truly no-op truncate.
    if (isa<TruncInst>(U) &&
        TLI.isTruncateFree(U->getOperand(0)->getType(), U->getType()))
      continue;
    // Check for a truly no-op bitcast.
    if (isa<BitCastInst>(U) &&
        (U->getOperand(0)->getType() == U->getType() ||
         (U->getOperand(0)->getType()->isPointerTy() &&
          U->getType()->isPointerTy())))
      continue;
    // Otherwise it's not a true no-op.
    return false;
  }

  return true;
}

