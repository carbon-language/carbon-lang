//===-- ARMSjLjLoweringPass.cpp - ARM SjLj Lowering Pass ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that lowers the SjLj exception handling into
// machine instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-sjlj-lowering"
#include "ARM.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

// Hidden options for the new EH stuff.
static cl::opt<bool>
EnableNewSjLjEHPrepare("enable-new-sjlj-eh", cl::Hidden,
                       cl::desc("Use the new SjLj EH preparation pass"));

namespace {

class ARMSjLjLowering : public MachineFunctionPass {
  Type *FunctionCtxTy;
  LLVMContext *Context;

  MachineFunction *MF;
  const Function *Fn;
  const TargetLowering *TLI;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;

  /// createFunctionContext - Create the function context on the stack. This
  /// returns the nonnegative identifier representing it in the FrameInfo.
  int createFunctionContext();

public:
  static char ID;
  ARMSjLjLowering() : MachineFunctionPass(ID) {}

  virtual bool runOnMachineFunction(MachineFunction &mf);

  virtual const char *getPassName() const {
    return "ARM setjmp/longjmp exception handling lowering pass";
  }
};

char ARMSjLjLowering::ID = 0;

} // end anonymous namespace

FunctionPass *llvm::createARMSjLjLoweringPass() {
  return new ARMSjLjLowering();
}

bool ARMSjLjLowering::runOnMachineFunction(MachineFunction &mf) {
  if (!EnableNewSjLjEHPrepare) return false;

  MF = &mf;
  Fn = MF->getFunction();
  Context = &Fn->getContext();
  TLI = MF->getTarget().getTargetLowering();
  TII = MF->getTarget().getInstrInfo();
  TRI = MF->getTarget().getRegisterInfo();

  int FrameIdx = createFunctionContext(); (void)FrameIdx;

  return true;
}

/// createFunctionContext - Create the function context on the stack.
int ARMSjLjLowering::createFunctionContext() {
  // struct _Unwind_FunctionContext {
  //   // next function in stack of handlers.
  //   struct _Unwind_FunctionContext *prev;
  //
  //   // set by calling function before registering to be the landing pad.
  //   uintptr_t resumeLocation;
  //
  //   // set by personality handler to be parameters passed to landing pad
  //   // function.
  //   uintptr_t resumeParameters[4];
  //
  //   // set by calling function before registering
  //   __personality_routine personality;  // arm offset=24
  //
  //   uintptr_t lsda                      // arm offset=28
  //
  //   // variable length array, contains registers to restore
  //   // 0 = r7, 1 = pc, 2 = sp
  //   void *jbuf[];  // 5 for GCC compatibility.
  // };
  Type *VoidPtrTy = Type::getInt8PtrTy(*Context);
  Type *Int32Ty = Type::getInt32Ty(*Context);
  FunctionCtxTy =
    StructType::get(VoidPtrTy,                        // prev
                    Int32Ty,                          // resumeLocation
                    ArrayType::get(Int32Ty, 4),       // resumeParameters
                    VoidPtrTy,                        // personality
                    VoidPtrTy,                        // lsda
                    ArrayType::get(VoidPtrTy, 5),     // jbuf
                    NULL);

  uint64_t TySize = TLI->getTargetData()->getTypeAllocSize(FunctionCtxTy);
  unsigned Align = TLI->getTargetData()->getPrefTypeAlignment(FunctionCtxTy);

  return MF->getFrameInfo()->CreateStackObject(TySize, Align, false, false);
}
