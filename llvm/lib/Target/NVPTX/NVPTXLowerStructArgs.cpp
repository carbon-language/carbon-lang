//===-- NVPTXLowerStructArgs.cpp - Copy struct args to local memory =====--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Copy struct args to local memory. This is needed for kernel functions only.
// This is a preparation for handling cases like
//
// kernel void foo(struct A arg, ...)
// {
//     struct A *p = &arg;
//     ...
//     ... = p->filed1 ...  (this is no generic address for .param)
//     p->filed2 = ...      (this is no write access to .param)
// }
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXUtilities.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace llvm {
void initializeNVPTXLowerStructArgsPass(PassRegistry &);
}

class LLVM_LIBRARY_VISIBILITY NVPTXLowerStructArgs : public FunctionPass {
  bool runOnFunction(Function &F) override;

  void handleStructPtrArgs(Function &);
  void handleParam(Argument *);

public:
  static char ID; // Pass identification, replacement for typeid
  NVPTXLowerStructArgs() : FunctionPass(ID) {}
  const char *getPassName() const override {
    return "Copy structure (byval *) arguments to stack";
  }
};

char NVPTXLowerStructArgs::ID = 1;

INITIALIZE_PASS(NVPTXLowerStructArgs, "nvptx-lower-struct-args",
                "Lower structure arguments (NVPTX)", false, false)

void NVPTXLowerStructArgs::handleParam(Argument *Arg) {
  Function *Func = Arg->getParent();
  Instruction *FirstInst = &(Func->getEntryBlock().front());
  const PointerType *PType = dyn_cast<PointerType>(Arg->getType());

  assert(PType && "Expecting pointer type in handleParam");

  const Type *StructType = PType->getElementType();

  AllocaInst *AllocA =
    new AllocaInst((Type *)StructType, Arg->getName(), FirstInst);

  /* Set the alignment to alignment of the byval parameter. This is because,
   * later load/stores assume that alignment, and we are going to replace
   * the use of the byval parameter with this alloca instruction.
   */
  AllocA->setAlignment(Func->getParamAlignment(Arg->getArgNo() + 1));

  Arg->replaceAllUsesWith(AllocA);

  // Get the cvt.gen.to.param intrinsic
  const Type *CvtTypes[2] = {
    Type::getInt8PtrTy(Func->getParent()->getContext(), ADDRESS_SPACE_PARAM),
    Type::getInt8PtrTy(Func->getParent()->getContext(), ADDRESS_SPACE_GENERIC)
  };
  Function *CvtFunc = (Function *)Intrinsic::getDeclaration(
      Func->getParent(), Intrinsic::nvvm_ptr_gen_to_param,
      ArrayRef<Type *>((Type **)CvtTypes, 2));
  std::vector<Value *> BC1;
  BC1.push_back(
      new BitCastInst(Arg, Type::getInt8PtrTy(Func->getParent()->getContext(),
                                              ADDRESS_SPACE_GENERIC),
                      Arg->getName(), FirstInst));
  CallInst *CallCVT = CallInst::Create(CvtFunc, ArrayRef<Value *>(BC1),
                                       "cvt_to_param", FirstInst);

  BitCastInst *BitCast = new BitCastInst(
      CallCVT, PointerType::get((Type *)StructType, ADDRESS_SPACE_PARAM),
      Arg->getName(), FirstInst);
  LoadInst *LI = new LoadInst(BitCast, Arg->getName(), FirstInst);
  new StoreInst(LI, AllocA, FirstInst);
}

/// =============================================================================
/// If the function had a struct ptr arg, say foo(%struct.x *byval %d), then
/// add the following instructions to the first basic block :
///
/// %temp = alloca %struct.x, align 8
/// %tt1 = bitcast %struct.x * %d to i8 *
/// %tt2 = llvm.nvvm.cvt.gen.to.param %tt2
/// %tempd = bitcast i8 addrspace(101) * to %struct.x addrspace(101) *
/// %tv = load %struct.x addrspace(101) * %tempd
/// store %struct.x %tv, %struct.x * %temp, align 8
///
/// The above code allocates some space in the stack and copies the incoming
/// struct from param space to local space.
/// Then replace all occurences of %d by %temp.
/// =============================================================================
void NVPTXLowerStructArgs::handleStructPtrArgs(Function &F) {
  const AttributeSet &PAL = F.getAttributes();

  unsigned Idx = 1;

  for (Argument &Arg : F.args()) {
    const Type *Ty = Arg.getType();

    const PointerType *PTy = dyn_cast<PointerType>(Ty);

    if (PTy) {
      if (PAL.hasAttribute(Idx, Attribute::ByVal)) {
        //  cout << "Has struct ptr args" << std::endl;
        handleParam(&Arg);
      }
    }
    Idx++;
  }
}

/// =============================================================================
/// Main function for this pass.
/// =============================================================================
bool NVPTXLowerStructArgs::runOnFunction(Function &F) {
  // Skip non-kernels. See the comments at the top of this file.
  if (!isKernelFunction(F))
    return false;

  handleStructPtrArgs(F);

  return true;
}

FunctionPass *llvm::createNVPTXLowerStructArgsPass() {
  return new NVPTXLowerStructArgs();
}
