//===---- Mips16HardFloat.cpp for Mips16 Hard Float               --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass needed for Mips16 Hard Float
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips16-hard-float"
#include "Mips16HardFloat.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

//
// Return types that matter for hard float are:
// float, double, complex float, and complex double
//
enum FPReturnVariant {
  FRet, DRet, CFRet, CDRet, NoFPRet
};

//
// Determine which FP return type this function has
//
static FPReturnVariant whichFPReturnVariant(Type *T) {
  switch (T->getTypeID()) {
  case Type::FloatTyID:
    return FRet;
  case Type::DoubleTyID:
    return DRet;
  case Type::StructTyID:
    if (T->getStructNumElements() != 2)
      break;
    if ((T->getContainedType(0)->isFloatTy()) &&
        (T->getContainedType(1)->isFloatTy()))
      return CFRet;
    if ((T->getContainedType(0)->isDoubleTy()) &&
        (T->getContainedType(1)->isDoubleTy()))
      return CDRet;
    break;
  default:
    break;
  }
  return NoFPRet;
}

//
// Returns of float, double and complex need to be handled with a helper
// function. The "AndCal" part is coming in a later patch.
//
static bool fixupFPReturnAndCall
  (Function &F, Module *M,  const MipsSubtarget &Subtarget) {
  bool Modified = false;
  LLVMContext &C = M->getContext();
  Type *MyVoid = Type::getVoidTy(C);
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      Instruction &Inst = *I;
      if (const ReturnInst *RI = dyn_cast<ReturnInst>(I)) {
        Value *RVal = RI->getReturnValue();
        if (!RVal) continue;
        //
        // If there is a return value and it needs a helper function,
        // figure out which one and add a call before the actual
        // return to this helper. The purpose of the helper is to move
        // floating point values from their soft float return mapping to
        // where they would have been mapped to in floating point registers.
        //
        Type *T = RVal->getType();
        FPReturnVariant RV = whichFPReturnVariant(T);
        if (RV == NoFPRet) continue;
        static const char* Helper[NoFPRet] =
          {"__mips16_ret_sf", "__mips16_ret_df", "__mips16_ret_sc",
           "__mips16_ret_dc"};
        const char *Name = Helper[RV];
        AttributeSet A;
        Value *Params[] = {RVal};
        Modified = true;
        //
        // These helper functions have a different calling ABI so
        // this __Mips16RetHelper indicates that so that later
        // during call setup, the proper call lowering to the helper
        // functions will take place.
        //
        A = A.addAttribute(C, AttributeSet::FunctionIndex,
                           "__Mips16RetHelper");
        A = A.addAttribute(C, AttributeSet::FunctionIndex,
                           Attribute::ReadNone);
        Value *F = (M->getOrInsertFunction(Name, A, MyVoid, T, NULL));
        CallInst::Create(F, Params, "", &Inst );
      }
    }
  return Modified;
}

namespace llvm {

//
// This pass only makes sense when the underlying chip has floating point but
// we are compiling as mips16.
// For all mips16 functions (that are not stubs we have already generated), or
// declared via attributes as nomips16, we must:
//    1) fixup all returns of float, double, single and double complex
//       by calling a helper function before the actual return.
//    2) generate helper functions (stubs) that can be called by mips32 functions
//       that will move parameters passed normally passed in floating point
//       registers the soft float equivalents. (Coming in a later patch).
//    3) in the case of static relocation, generate helper functions so that
//       mips16 functions can call extern functions of unknown type (mips16 or
//       mips32). (Coming in a later patch).
//    4) TBD. For pic, calls to extern functions of unknown type are handled by
//       predefined helper functions in libc but this work is currently done
//       during call lowering but it should be moved here in the future.
//
bool Mips16HardFloat::runOnModule(Module &M) {
  DEBUG(errs() << "Run on Module Mips16HardFloat\n");
  bool Modified = false;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration() || F->hasFnAttribute("mips16_fp_stub") ||
        F->hasFnAttribute("nomips16")) continue;
    Modified |= fixupFPReturnAndCall(*F, &M, Subtarget);
  }
  return Modified;
}

char Mips16HardFloat::ID = 0;

}

ModulePass *llvm::createMips16HardFloat(MipsTargetMachine &TM) {
  return new Mips16HardFloat(TM);
}

