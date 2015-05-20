//===-- X86WinEHState - Insert EH state updates for win32 exceptions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// All functions using an MSVC EH personality use an explicitly updated state
// number stored in an exception registration stack object. The registration
// object is linked into a thread-local chain of registrations stored at fs:00.
// This pass adds the registration object and EH state updates.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/Analysis/LibCallSemantics.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "winehstate"

namespace {
class WinEHStatePass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid.

  WinEHStatePass() : FunctionPass(ID) {}

  bool runOnFunction(Function &Fn) override;

  bool doInitialization(Module &M) override;

  bool doFinalization(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  const char *getPassName() const override {
    return "Windows 32-bit x86 EH state insertion";
  }

private:
  void emitExceptionRegistrationRecord(Function *F);

  void linkExceptionRegistration(IRBuilder<> &Builder, Value *RegNode,
                                 Value *Handler);
  void unlinkExceptionRegistration(IRBuilder<> &Builder, Value *RegNode);

  Value *emitEHLSDA(IRBuilder<> &Builder, Function *F);

  Function *generateLSDAInEAXThunk(Function *ParentFunc);

  // Module-level type getters.
  Type *getEHRegistrationType();
  Type *getSEH3RegistrationType();
  Type *getSEH4RegistrationType();
  Type *getCXXEH3RegistrationType();

  // Per-module data.
  Module *TheModule = nullptr;
  StructType *EHRegistrationTy = nullptr;
  StructType *CXXEH3RegistrationTy = nullptr;
  StructType *SEH3RegistrationTy = nullptr;
  StructType *SEH4RegistrationTy = nullptr;

  // Per-function state
  EHPersonality Personality = EHPersonality::Unknown;
  Function *PersonalityFn = nullptr;
};
}

FunctionPass *llvm::createX86WinEHStatePass() { return new WinEHStatePass(); }

char WinEHStatePass::ID = 0;

bool WinEHStatePass::doInitialization(Module &M) {
  TheModule = &M;
  return false;
}

bool WinEHStatePass::doFinalization(Module &M) {
  assert(TheModule == &M);
  TheModule = nullptr;
  EHRegistrationTy = nullptr;
  CXXEH3RegistrationTy = nullptr;
  SEH3RegistrationTy = nullptr;
  SEH4RegistrationTy = nullptr;
  return false;
}

void WinEHStatePass::getAnalysisUsage(AnalysisUsage &AU) const {
  // This pass should only insert a stack allocation, memory accesses, and
  // framerecovers.
  AU.setPreservesCFG();
}

bool WinEHStatePass::runOnFunction(Function &F) {
  // If this is an outlined handler, don't do anything. We'll do state insertion
  // for it in the parent.
  StringRef WinEHParentName =
      F.getFnAttribute("wineh-parent").getValueAsString();
  if (WinEHParentName != F.getName() && !WinEHParentName.empty())
    return false;

  // Check the personality. Do nothing if this is not an MSVC personality.
  LandingPadInst *LP = nullptr;
  for (BasicBlock &BB : F) {
    LP = BB.getLandingPadInst();
    if (LP)
      break;
  }
  if (!LP)
    return false;
  PersonalityFn =
      dyn_cast<Function>(LP->getPersonalityFn()->stripPointerCasts());
  if (!PersonalityFn)
    return false;
  Personality = classifyEHPersonality(PersonalityFn);
  if (!isMSVCEHPersonality(Personality))
    return false;

  emitExceptionRegistrationRecord(&F);
  // FIXME: State insertion.

  // Reset per-function state.
  PersonalityFn = nullptr;
  Personality = EHPersonality::Unknown;
  return true;
}

/// Get the common EH registration subobject:
///   typedef _EXCEPTION_DISPOSITION (*PEXCEPTION_ROUTINE)(
///       _EXCEPTION_RECORD *, void *, _CONTEXT *, void *);
///   struct EHRegistrationNode {
///     EHRegistrationNode *Next;
///     PEXCEPTION_ROUTINE Handler;
///   };
Type *WinEHStatePass::getEHRegistrationType() {
  if (EHRegistrationTy)
    return EHRegistrationTy;
  LLVMContext &Context = TheModule->getContext();
  EHRegistrationTy = StructType::create(Context, "EHRegistrationNode");
  Type *FieldTys[] = {
      EHRegistrationTy->getPointerTo(0), // EHRegistrationNode *Next
      Type::getInt8PtrTy(Context) // EXCEPTION_DISPOSITION (*Handler)(...)
  };
  EHRegistrationTy->setBody(FieldTys, false);
  return EHRegistrationTy;
}

/// The __CxxFrameHandler3 registration node:
///   struct CXXExceptionRegistration {
///     void *SavedESP;
///     EHRegistrationNode SubRecord;
///     int32_t TryLevel;
///   };
Type *WinEHStatePass::getCXXEH3RegistrationType() {
  if (CXXEH3RegistrationTy)
    return CXXEH3RegistrationTy;
  LLVMContext &Context = TheModule->getContext();
  Type *FieldTys[] = {
      Type::getInt8PtrTy(Context), // void *SavedESP
      getEHRegistrationType(),     // EHRegistrationNode SubRecord
      Type::getInt32Ty(Context)    // int32_t TryLevel
  };
  CXXEH3RegistrationTy =
      StructType::create(FieldTys, "CXXExceptionRegistration");
  return CXXEH3RegistrationTy;
}

/// The _except_handler3 registration node:
///   struct EH3ExceptionRegistration {
///     EHRegistrationNode SubRecord;
///     void *ScopeTable;
///     int32_t TryLevel;
///   };
Type *WinEHStatePass::getSEH3RegistrationType() {
  if (SEH3RegistrationTy)
    return SEH3RegistrationTy;
  LLVMContext &Context = TheModule->getContext();
  Type *FieldTys[] = {
      getEHRegistrationType(),     // EHRegistrationNode SubRecord
      Type::getInt8PtrTy(Context), // void *ScopeTable
      Type::getInt32Ty(Context)    // int32_t TryLevel
  };
  SEH3RegistrationTy = StructType::create(FieldTys, "EH3ExceptionRegistration");
  return SEH3RegistrationTy;
}

/// The _except_handler4 registration node:
///   struct EH4ExceptionRegistration {
///     void *SavedESP;
///     _EXCEPTION_POINTERS *ExceptionPointers;
///     EHRegistrationNode SubRecord;
///     int32_t EncodedScopeTable;
///     int32_t TryLevel;
///   };
Type *WinEHStatePass::getSEH4RegistrationType() {
  if (SEH4RegistrationTy)
    return SEH4RegistrationTy;
  LLVMContext &Context = TheModule->getContext();
  Type *FieldTys[] = {
      Type::getInt8PtrTy(Context), // void *SavedESP
      Type::getInt8PtrTy(Context), // void *ExceptionPointers
      getEHRegistrationType(),     // EHRegistrationNode SubRecord
      Type::getInt32Ty(Context),   // int32_t EncodedScopeTable
      Type::getInt32Ty(Context)    // int32_t TryLevel
  };
  SEH4RegistrationTy = StructType::create(FieldTys, "EH4ExceptionRegistration");
  return SEH4RegistrationTy;
}

// Emit an exception registration record. These are stack allocations with the
// common subobject of two pointers: the previous registration record (the old
// fs:00) and the personality function for the current frame. The data before
// and after that is personality function specific.
void WinEHStatePass::emitExceptionRegistrationRecord(Function *F) {
  assert(Personality == EHPersonality::MSVC_CXX ||
         Personality == EHPersonality::MSVC_X86SEH);

  StringRef PersonalityName = PersonalityFn->getName();
  IRBuilder<> Builder(&F->getEntryBlock(), F->getEntryBlock().begin());
  Type *Int8PtrType = Builder.getInt8PtrTy();
  Value *SubRecord = nullptr;
  if (PersonalityName == "__CxxFrameHandler3") {
    Type *RegNodeTy = getCXXEH3RegistrationType();
    Value *RegNode = Builder.CreateAlloca(RegNodeTy);
    // FIXME: We can skip this in -GS- mode, when we figure that out.
    // SavedESP = llvm.stacksave()
    Value *SP = Builder.CreateCall(
        Intrinsic::getDeclaration(TheModule, Intrinsic::stacksave), {});
    Builder.CreateStore(SP, Builder.CreateStructGEP(RegNodeTy, RegNode, 0));
    // TryLevel = -1
    Builder.CreateStore(Builder.getInt32(-1),
                        Builder.CreateStructGEP(RegNodeTy, RegNode, 2));
    // Handler = __ehhandler$F
    Function *Trampoline = generateLSDAInEAXThunk(F);
    SubRecord = Builder.CreateStructGEP(RegNodeTy, RegNode, 1);
    linkExceptionRegistration(Builder, SubRecord, Trampoline);
  } else if (PersonalityName == "_except_handler3") {
    Type *RegNodeTy = getSEH3RegistrationType();
    Value *RegNode = Builder.CreateAlloca(RegNodeTy);
    // TryLevel = -1
    Builder.CreateStore(Builder.getInt32(-1),
                        Builder.CreateStructGEP(RegNodeTy, RegNode, 2));
    // ScopeTable = llvm.x86.seh.lsda(F)
    Value *LSDA = emitEHLSDA(Builder, F);
    Builder.CreateStore(LSDA, Builder.CreateStructGEP(RegNodeTy, RegNode, 1));
    SubRecord = Builder.CreateStructGEP(RegNodeTy, RegNode, 0);
    linkExceptionRegistration(Builder, SubRecord, PersonalityFn);
  } else if (PersonalityName == "_except_handler4") {
    Type *RegNodeTy = getSEH4RegistrationType();
    Value *RegNode = Builder.CreateAlloca(RegNodeTy);
    // SavedESP = llvm.stacksave()
    Value *SP = Builder.CreateCall(
        Intrinsic::getDeclaration(TheModule, Intrinsic::stacksave), {});
    Builder.CreateStore(SP, Builder.CreateStructGEP(RegNodeTy, RegNode, 0));
    // TryLevel = -2
    Builder.CreateStore(Builder.getInt32(-2),
                        Builder.CreateStructGEP(RegNodeTy, RegNode, 4));
    // FIXME: XOR the LSDA with __security_cookie.
    // ScopeTable = llvm.x86.seh.lsda(F)
    Value *FI8 = Builder.CreateBitCast(F, Int8PtrType);
    Value *LSDA = Builder.CreateCall(
        Intrinsic::getDeclaration(TheModule, Intrinsic::x86_seh_lsda), FI8);
    Builder.CreateStore(LSDA, Builder.CreateStructGEP(RegNodeTy, RegNode, 1));
    SubRecord = Builder.CreateStructGEP(RegNodeTy, RegNode, 2);
    linkExceptionRegistration(Builder, SubRecord, PersonalityFn);
  } else {
    llvm_unreachable("unexpected personality function");
  }

  // FIXME: Insert an unlink before all returns.
  for (BasicBlock &BB : *F) {
    TerminatorInst *T = BB.getTerminator();
    if (!isa<ReturnInst>(T))
      continue;
    Builder.SetInsertPoint(T);
    unlinkExceptionRegistration(Builder, SubRecord);
  }
}

Value *WinEHStatePass::emitEHLSDA(IRBuilder<> &Builder, Function *F) {
  Value *FI8 = Builder.CreateBitCast(F, Type::getInt8PtrTy(F->getContext()));
  return Builder.CreateCall(
      Intrinsic::getDeclaration(TheModule, Intrinsic::x86_seh_lsda), FI8);
}

/// Generate a thunk that puts the LSDA of ParentFunc in EAX and then calls
/// PersonalityFn, forwarding the parameters passed to PEXCEPTION_ROUTINE:
///   typedef _EXCEPTION_DISPOSITION (*PEXCEPTION_ROUTINE)(
///       _EXCEPTION_RECORD *, void *, _CONTEXT *, void *);
/// We essentially want this code:
///   movl $lsda, %eax
///   jmpl ___CxxFrameHandler3
Function *WinEHStatePass::generateLSDAInEAXThunk(Function *ParentFunc) {
  LLVMContext &Context = ParentFunc->getContext();
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int8PtrType = Type::getInt8PtrTy(Context);
  Type *ArgTys[5] = {Int8PtrType, Int8PtrType, Int8PtrType, Int8PtrType,
                     Int8PtrType};
  FunctionType *TrampolineTy =
      FunctionType::get(Int32Ty, makeArrayRef(&ArgTys[0], 4),
                        /*isVarArg=*/false);
  FunctionType *TargetFuncTy =
      FunctionType::get(Int32Ty, makeArrayRef(&ArgTys[0], 5),
                        /*isVarArg=*/false);
  Function *Trampoline = Function::Create(
      TrampolineTy, GlobalValue::InternalLinkage,
      Twine("__ehhandler$") + ParentFunc->getName(), TheModule);
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", Trampoline);
  IRBuilder<> Builder(EntryBB);
  Value *LSDA = emitEHLSDA(Builder, ParentFunc);
  Value *CastPersonality =
      Builder.CreateBitCast(PersonalityFn, TargetFuncTy->getPointerTo());
  auto AI = Trampoline->arg_begin();
  Value *Args[5] = {LSDA, AI++, AI++, AI++, AI++};
  CallInst *Call = Builder.CreateCall(CastPersonality, Args);
  // Can't use musttail due to prototype mismatch, but we can use tail.
  Call->setTailCall(true);
  // Set inreg so we pass it in EAX.
  Call->addAttribute(1, Attribute::InReg);
  Builder.CreateRet(Call);
  return Trampoline;
}

void WinEHStatePass::linkExceptionRegistration(IRBuilder<> &Builder,
                                               Value *RegNode, Value *Handler) {
  Type *RegNodeTy = getEHRegistrationType();
  // Handler = Handler
  Handler = Builder.CreateBitCast(Handler, Builder.getInt8PtrTy());
  Builder.CreateStore(Handler, Builder.CreateStructGEP(RegNodeTy, RegNode, 1));
  // Next = [fs:00]
  Constant *FSZero =
      Constant::getNullValue(RegNodeTy->getPointerTo()->getPointerTo(257));
  Value *Next = Builder.CreateLoad(FSZero);
  Builder.CreateStore(Next, Builder.CreateStructGEP(RegNodeTy, RegNode, 0));
  // [fs:00] = RegNode
  Builder.CreateStore(RegNode, FSZero);
}

void WinEHStatePass::unlinkExceptionRegistration(IRBuilder<> &Builder,
                                                 Value *RegNode) {
  // Clone RegNode into the current BB for better address mode folding.
  if (auto *GEP = dyn_cast<GetElementPtrInst>(RegNode)) {
    GEP = cast<GetElementPtrInst>(GEP->clone());
    Builder.Insert(GEP);
    RegNode = GEP;
  }
  Type *RegNodeTy = getEHRegistrationType();
  // [fs:00] = RegNode->Next
  Value *Next =
      Builder.CreateLoad(Builder.CreateStructGEP(RegNodeTy, RegNode, 0));
  Constant *FSZero =
      Constant::getNullValue(RegNodeTy->getPointerTo()->getPointerTo(257));
  Builder.CreateStore(Next, FSZero);
}
