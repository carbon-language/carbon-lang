//===-- NVPTXLowerKernelArgs.cpp - Lower kernel arguments -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pointer arguments to kernel functions need to be lowered specially.
//
// 1. Copy byval struct args to local memory. This is a preparation for handling
//    cases like
//
//    kernel void foo(struct A arg, ...)
//    {
//      struct A *p = &arg;
//      ...
//      ... = p->filed1 ...  (this is no generic address for .param)
//      p->filed2 = ...      (this is no write access to .param)
//    }
//
// 2. Convert non-byval pointer arguments of CUDA kernels to pointers in the
//    global address space. This allows later optimizations to emit
//    ld.global.*/st.global.* for accessing these pointer arguments. For
//    example,
//
//    define void @foo(float* %input) {
//      %v = load float, float* %input, align 4
//      ...
//    }
//
//    becomes
//
//    define void @foo(float* %input) {
//      %input2 = addrspacecast float* %input to float addrspace(1)*
//      %input3 = addrspacecast float addrspace(1)* %input2 to float*
//      %v = load float, float* %input3, align 4
//      ...
//    }
//
//    Later, NVPTXFavorNonGenericAddrSpaces will optimize it to
//
//    define void @foo(float* %input) {
//      %input2 = addrspacecast float* %input to float addrspace(1)*
//      %v = load float, float addrspace(1)* %input2, align 4
//      ...
//    }
//
// TODO: merge this pass with NVPTXFavorNonGenericAddrSpace so that other passes
// don't cancel the addrspacecast pair this pass emits.
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXUtilities.h"
#include "NVPTXTargetMachine.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace llvm {
void initializeNVPTXLowerKernelArgsPass(PassRegistry &);
}

namespace {
class NVPTXLowerKernelArgs : public FunctionPass {
  bool runOnFunction(Function &F) override;

  // handle byval parameters
  void handleByValParam(Argument *);
  // handle non-byval pointer parameters
  void handlePointerParam(Argument *);

public:
  static char ID; // Pass identification, replacement for typeid
  NVPTXLowerKernelArgs(const NVPTXTargetMachine *TM = nullptr)
      : FunctionPass(ID), TM(TM) {}
  const char *getPassName() const override {
    return "Lower pointer arguments of CUDA kernels";
  }

private:
  const NVPTXTargetMachine *TM;
};
} // namespace

char NVPTXLowerKernelArgs::ID = 1;

INITIALIZE_PASS(NVPTXLowerKernelArgs, "nvptx-lower-kernel-args",
                "Lower kernel arguments (NVPTX)", false, false)

// =============================================================================
// If the function had a byval struct ptr arg, say foo(%struct.x *byval %d),
// then add the following instructions to the first basic block:
//
// %temp = alloca %struct.x, align 8
// %tempd = addrspacecast %struct.x* %d to %struct.x addrspace(101)*
// %tv = load %struct.x addrspace(101)* %tempd
// store %struct.x %tv, %struct.x* %temp, align 8
//
// The above code allocates some space in the stack and copies the incoming
// struct from param space to local space.
// Then replace all occurences of %d by %temp.
// =============================================================================
void NVPTXLowerKernelArgs::handleByValParam(Argument *Arg) {
  Function *Func = Arg->getParent();
  Instruction *FirstInst = &(Func->getEntryBlock().front());
  PointerType *PType = dyn_cast<PointerType>(Arg->getType());

  assert(PType && "Expecting pointer type in handleByValParam");

  Type *StructType = PType->getElementType();
  AllocaInst *AllocA = new AllocaInst(StructType, Arg->getName(), FirstInst);
  // Set the alignment to alignment of the byval parameter. This is because,
  // later load/stores assume that alignment, and we are going to replace
  // the use of the byval parameter with this alloca instruction.
  AllocA->setAlignment(Func->getParamAlignment(Arg->getArgNo() + 1));
  Arg->replaceAllUsesWith(AllocA);

  Value *ArgInParam = new AddrSpaceCastInst(
      Arg, PointerType::get(StructType, ADDRESS_SPACE_PARAM), Arg->getName(),
      FirstInst);
  LoadInst *LI = new LoadInst(ArgInParam, Arg->getName(), FirstInst);
  new StoreInst(LI, AllocA, FirstInst);
}

void NVPTXLowerKernelArgs::handlePointerParam(Argument *Arg) {
  assert(!Arg->hasByValAttr() &&
         "byval params should be handled by handleByValParam");

  // Do nothing if the argument already points to the global address space.
  if (Arg->getType()->getPointerAddressSpace() == ADDRESS_SPACE_GLOBAL)
    return;

  Instruction *FirstInst = Arg->getParent()->getEntryBlock().begin();
  Instruction *ArgInGlobal = new AddrSpaceCastInst(
      Arg, PointerType::get(Arg->getType()->getPointerElementType(),
                            ADDRESS_SPACE_GLOBAL),
      Arg->getName(), FirstInst);
  Value *ArgInGeneric = new AddrSpaceCastInst(ArgInGlobal, Arg->getType(),
                                              Arg->getName(), FirstInst);
  // Replace with ArgInGeneric all uses of Args except ArgInGlobal.
  Arg->replaceAllUsesWith(ArgInGeneric);
  ArgInGlobal->setOperand(0, Arg);
}


// =============================================================================
// Main function for this pass.
// =============================================================================
bool NVPTXLowerKernelArgs::runOnFunction(Function &F) {
  // Skip non-kernels. See the comments at the top of this file.
  if (!isKernelFunction(F))
    return false;

  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy()) {
      if (Arg.hasByValAttr())
        handleByValParam(&Arg);
      else if (TM && TM->getDrvInterface() == NVPTX::CUDA)
        handlePointerParam(&Arg);
    }
  }
  return true;
}

FunctionPass *
llvm::createNVPTXLowerKernelArgsPass(const NVPTXTargetMachine *TM) {
  return new NVPTXLowerKernelArgs(TM);
}
