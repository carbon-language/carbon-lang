//===-- XCoreLowerThreadLocal - Lower thread local variables --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains a pass that lowers thread local variables on the
///        XCore.
///
//===----------------------------------------------------------------------===//

#include "XCore.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "xcore-lower-thread-local"

using namespace llvm;

static cl::opt<unsigned> MaxThreads(
  "xcore-max-threads", cl::Optional,
  cl::desc("Maximum number of threads (for emulation thread-local storage)"),
  cl::Hidden, cl::value_desc("number"), cl::init(8));

namespace {
  /// Lowers thread local variables on the XCore. Each thread local variable is
  /// expanded to an array of n elements indexed by the thread ID where n is the
  /// fixed number hardware threads supported by the device.
  struct XCoreLowerThreadLocal : public ModulePass {
    static char ID;

    XCoreLowerThreadLocal() : ModulePass(ID) {
      initializeXCoreLowerThreadLocalPass(*PassRegistry::getPassRegistry());
    }

    bool lowerGlobal(GlobalVariable *GV);

    bool runOnModule(Module &M);
  };
}

char XCoreLowerThreadLocal::ID = 0;

INITIALIZE_PASS(XCoreLowerThreadLocal, "xcore-lower-thread-local",
                "Lower thread local variables", false, false)

ModulePass *llvm::createXCoreLowerThreadLocalPass() {
  return new XCoreLowerThreadLocal();
}

static ArrayType *createLoweredType(Type *OriginalType) {
  return ArrayType::get(OriginalType, MaxThreads);
}

static Constant *
createLoweredInitializer(ArrayType *NewType, Constant *OriginalInitializer) {
  SmallVector<Constant *, 8> Elements(MaxThreads);
  for (unsigned i = 0; i != MaxThreads; ++i) {
    Elements[i] = OriginalInitializer;
  }
  return ConstantArray::get(NewType, Elements);
}

static bool hasNonInstructionUse(GlobalVariable *GV) {
  for (Value::use_iterator UI = GV->use_begin(), E = GV->use_end(); UI != E;
       ++UI)
    if (!isa<Instruction>(*UI))
      return true;

  return false;
}

static bool isZeroLengthArray(Type *Ty) {
  ArrayType *AT = dyn_cast<ArrayType>(Ty);
  return AT && (AT->getNumElements() == 0);
}

bool XCoreLowerThreadLocal::lowerGlobal(GlobalVariable *GV) {
  Module *M = GV->getParent();
  LLVMContext &Ctx = M->getContext();
  if (!GV->isThreadLocal())
    return false;

  // Skip globals that we can't lower and leave it for the backend to error.
  if (hasNonInstructionUse(GV) ||
      !GV->getType()->isSized() || isZeroLengthArray(GV->getType()))
    return false;

  // Create replacement global.
  ArrayType *NewType = createLoweredType(GV->getType()->getElementType());
  Constant *NewInitializer = createLoweredInitializer(NewType,
                                                      GV->getInitializer());
  GlobalVariable *NewGV =
    new GlobalVariable(*M, NewType, GV->isConstant(), GV->getLinkage(),
                       NewInitializer, "", 0, GlobalVariable::NotThreadLocal,
                       GV->getType()->getAddressSpace(),
                       GV->isExternallyInitialized());

  // Update uses.
  SmallVector<User *, 16> Users(GV->use_begin(), GV->use_end());
  for (unsigned I = 0, E = Users.size(); I != E; ++I) {
    User *U = Users[I];
    Instruction *Inst = cast<Instruction>(U);
    IRBuilder<> Builder(Inst);
    Function *GetID = Intrinsic::getDeclaration(GV->getParent(),
                                                Intrinsic::xcore_getid);
    Value *ThreadID = Builder.CreateCall(GetID);
    SmallVector<Value *, 2> Indices;
    Indices.push_back(Constant::getNullValue(Type::getInt64Ty(Ctx)));
    Indices.push_back(ThreadID);
    Value *Addr = Builder.CreateInBoundsGEP(NewGV, Indices);
    U->replaceUsesOfWith(GV, Addr);
  }

  // Remove old global.
  NewGV->takeName(GV);
  GV->eraseFromParent();
  return true;
}

bool XCoreLowerThreadLocal::runOnModule(Module &M) {
  // Find thread local globals.
  bool MadeChange = false;
  SmallVector<GlobalVariable *, 16> ThreadLocalGlobals;
  for (Module::global_iterator GVI = M.global_begin(), E = M.global_end();
       GVI != E; ++GVI) {
    GlobalVariable *GV = GVI;
    if (GV->isThreadLocal())
      ThreadLocalGlobals.push_back(GV);
  }
  for (unsigned I = 0, E = ThreadLocalGlobals.size(); I != E; ++I) {
    MadeChange |= lowerGlobal(ThreadLocalGlobals[I]);
  }
  return MadeChange;
}
