//===-- NVPTXFavorNonGenericAddrSpace.cpp - ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// When a load/store accesses the generic address space, checks whether the
// address is casted from a non-generic address space. If so, remove this
// addrspacecast because accessing non-generic address spaces is typically
// faster. Besides seeking addrspacecasts, this optimization also traces into
// the base pointer of a GEP.
//
// For instance, the code below loads a float from an array allocated in
// addrspace(3).
//
// %0 = addrspacecast [10 x float] addrspace(3)* @a to [10 x float]*
// %1 = gep [10 x float]* %0, i64 0, i64 %i
// %2 = load float* %1 ; emits ld.f32
//
// First, function hoistAddrSpaceCastFromGEP reorders the addrspacecast
// and the GEP to expose more optimization opportunities to function
// optimizeMemoryInst. The intermediate code looks like:
//
// %0 = gep [10 x float] addrspace(3)* @a, i64 0, i64 %i
// %1 = addrspacecast float addrspace(3)* %0 to float*
// %2 = load float* %1 ; still emits ld.f32, but will be optimized shortly
//
// Then, function optimizeMemoryInstruction detects a load from addrspacecast'ed
// generic pointers, and folds the load and the addrspacecast into a load from
// the original address space. The final code looks like:
//
// %0 = gep [10 x float] addrspace(3)* @a, i64 0, i64 %i
// %2 = load float addrspace(3)* %0 ; emits ld.shared.f32
//
// This pass may remove an addrspacecast in a different BB. Therefore, we
// implement it as a FunctionPass.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// An option to disable this optimization. Enable it by default.
static cl::opt<bool> DisableFavorNonGeneric(
  "disable-nvptx-favor-non-generic",
  cl::init(false),
  cl::desc("Do not convert generic address space usage "
           "to non-generic address space usage"),
  cl::Hidden);

namespace {
/// \brief NVPTXFavorNonGenericAddrSpaces
class NVPTXFavorNonGenericAddrSpaces : public FunctionPass {
public:
  static char ID;
  NVPTXFavorNonGenericAddrSpaces() : FunctionPass(ID) {}

  virtual bool runOnFunction(Function &F) override;

  /// Optimizes load/store instructions. Idx is the index of the pointer operand
  /// (0 for load, and 1 for store). Returns true if it changes anything.
  bool optimizeMemoryInstruction(Instruction *I, unsigned Idx);
  /// Transforms "gep (addrspacecast X), indices" into "addrspacecast (gep X,
  /// indices)".  This reordering exposes to optimizeMemoryInstruction more
  /// optimization opportunities on loads and stores. Returns true if it changes
  /// the program.
  bool hoistAddrSpaceCastFromGEP(GEPOperator *GEP);
};
}

char NVPTXFavorNonGenericAddrSpaces::ID = 0;

namespace llvm {
void initializeNVPTXFavorNonGenericAddrSpacesPass(PassRegistry &);
}
INITIALIZE_PASS(NVPTXFavorNonGenericAddrSpaces, "nvptx-favor-non-generic",
                "Remove unnecessary non-generic-to-generic addrspacecasts",
                false, false)

// Decides whether removing Cast is valid and beneficial. Cast can be an
// instruction or a constant expression.
static bool IsEliminableAddrSpaceCast(Operator *Cast) {
  // Returns false if not even an addrspacecast.
  if (Cast->getOpcode() != Instruction::AddrSpaceCast)
    return false;

  Value *Src = Cast->getOperand(0);
  PointerType *SrcTy = cast<PointerType>(Src->getType());
  PointerType *DestTy = cast<PointerType>(Cast->getType());
  // TODO: For now, we only handle the case where the addrspacecast only changes
  // the address space but not the type. If the type also changes, we could
  // still get rid of the addrspacecast by adding an extra bitcast, but we
  // rarely see such scenarios.
  if (SrcTy->getElementType() != DestTy->getElementType())
    return false;

  // Checks whether the addrspacecast is from a non-generic address space to the
  // generic address space.
  return (SrcTy->getAddressSpace() != AddressSpace::ADDRESS_SPACE_GENERIC &&
          DestTy->getAddressSpace() == AddressSpace::ADDRESS_SPACE_GENERIC);
}

bool NVPTXFavorNonGenericAddrSpaces::hoistAddrSpaceCastFromGEP(
    GEPOperator *GEP) {
  Operator *Cast = dyn_cast<Operator>(GEP->getPointerOperand());
  if (Cast == nullptr)
    return false;

  if (!IsEliminableAddrSpaceCast(Cast))
    return false;

  SmallVector<Value *, 8> Indices(GEP->idx_begin(), GEP->idx_end());
  if (Instruction *GEPI = dyn_cast<Instruction>(GEP)) {
    // %1 = gep (addrspacecast X), indices
    // =>
    // %0 = gep X, indices
    // %1 = addrspacecast %0
    GetElementPtrInst *NewGEPI = GetElementPtrInst::Create(Cast->getOperand(0),
                                                           Indices,
                                                           GEP->getName(),
                                                           GEPI);
    NewGEPI->setIsInBounds(GEP->isInBounds());
    GEP->replaceAllUsesWith(
        new AddrSpaceCastInst(NewGEPI, GEP->getType(), "", GEPI));
  } else {
    // GEP is a constant expression.
    Constant *NewGEPCE = ConstantExpr::getGetElementPtr(
        cast<Constant>(Cast->getOperand(0)),
        Indices,
        GEP->isInBounds());
    GEP->replaceAllUsesWith(
        ConstantExpr::getAddrSpaceCast(NewGEPCE, GEP->getType()));
  }

  return true;
}

bool NVPTXFavorNonGenericAddrSpaces::optimizeMemoryInstruction(Instruction *MI,
                                                               unsigned Idx) {
  // If the pointer operand is a GEP, hoist the addrspacecast if any from the
  // GEP to expose more optimization opportunites.
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(MI->getOperand(Idx))) {
    hoistAddrSpaceCastFromGEP(GEP);
  }

  // load/store (addrspacecast X) => load/store X if shortcutting the
  // addrspacecast is valid and can improve performance.
  //
  // e.g.,
  // %1 = addrspacecast float addrspace(3)* %0 to float*
  // %2 = load float* %1
  // ->
  // %2 = load float addrspace(3)* %0
  //
  // Note: the addrspacecast can also be a constant expression.
  if (Operator *Cast = dyn_cast<Operator>(MI->getOperand(Idx))) {
    if (IsEliminableAddrSpaceCast(Cast)) {
      MI->setOperand(Idx, Cast->getOperand(0));
      return true;
    }
  }

  return false;
}

bool NVPTXFavorNonGenericAddrSpaces::runOnFunction(Function &F) {
  if (DisableFavorNonGeneric)
    return false;

  bool Changed = false;
  for (Function::iterator B = F.begin(), BE = F.end(); B != BE; ++B) {
    for (BasicBlock::iterator I = B->begin(), IE = B->end(); I != IE; ++I) {
      if (isa<LoadInst>(I)) {
        // V = load P
        Changed |= optimizeMemoryInstruction(I, 0);
      } else if (isa<StoreInst>(I)) {
        // store V, P
        Changed |= optimizeMemoryInstruction(I, 1);
      }
    }
  }
  return Changed;
}

FunctionPass *llvm::createNVPTXFavorNonGenericAddrSpacesPass() {
  return new NVPTXFavorNonGenericAddrSpaces();
}
