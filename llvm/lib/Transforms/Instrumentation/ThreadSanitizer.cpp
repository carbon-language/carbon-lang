//===-- ThreadSanitizer.cpp - race detector -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer, a race detector.
//
// The tool is under development, for the details about previous versions see
// http://code.google.com/p/data-race-test
//
// The instrumentation phase is quite simple:
//   - Insert calls to run-time library before every memory access.
//      - Optimizations may apply to avoid instrumenting some of the accesses.
//   - Insert calls at function entry/exit.
// The rest is handled by the run-time library.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "tsan"

#include "FunctionBlackList.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Intrinsics.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Type.h"

using namespace llvm;

static cl::opt<std::string>  ClBlackListFile("tsan-blacklist",
       cl::desc("Blacklist file"), cl::Hidden);

namespace {
/// ThreadSanitizer: instrument the code in module to find races.
struct ThreadSanitizer : public FunctionPass {
  ThreadSanitizer();
  bool runOnFunction(Function &F);
  bool doInitialization(Module &M);
  bool instrumentLoadOrStore(Instruction *I);
  static char ID;  // Pass identification, replacement for typeid.

 private:
  TargetData *TD;
  OwningPtr<FunctionBlackList> BL;
  // Callbacks to run-time library are computed in doInitialization.
  Value *TsanFuncEntry;
  Value *TsanFuncExit;
  // Accesses sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t kNumberOfAccessSizes = 5;
  Value *TsanRead[kNumberOfAccessSizes];
  Value *TsanWrite[kNumberOfAccessSizes];
  Value *TsanVptrUpdate;
};
}  // namespace

char ThreadSanitizer::ID = 0;
INITIALIZE_PASS(ThreadSanitizer, "tsan",
    "ThreadSanitizer: detects data races.",
    false, false)

ThreadSanitizer::ThreadSanitizer()
  : FunctionPass(ID),
  TD(NULL) {
}

FunctionPass *llvm::createThreadSanitizerPass() {
  return new ThreadSanitizer();
}

bool ThreadSanitizer::doInitialization(Module &M) {
  TD = getAnalysisIfAvailable<TargetData>();
  if (!TD)
    return false;
  BL.reset(new FunctionBlackList(ClBlackListFile));

  // Always insert a call to __tsan_init into the module's CTORs.
  IRBuilder<> IRB(M.getContext());
  Value *TsanInit = M.getOrInsertFunction("__tsan_init",
                                          IRB.getVoidTy(), NULL);
  appendToGlobalCtors(M, cast<Function>(TsanInit), 0);

  // Initialize the callbacks.
  TsanFuncEntry = M.getOrInsertFunction("__tsan_func_entry", IRB.getVoidTy(),
                                        IRB.getInt8PtrTy(), NULL);
  TsanFuncExit = M.getOrInsertFunction("__tsan_func_exit", IRB.getVoidTy(),
                                       NULL);
  for (size_t i = 0; i < kNumberOfAccessSizes; ++i) {
    SmallString<32> ReadName("__tsan_read");
    ReadName += itostr(1 << i);
    TsanRead[i] = M.getOrInsertFunction(ReadName, IRB.getVoidTy(),
                                        IRB.getInt8PtrTy(), NULL);
    SmallString<32> WriteName("__tsan_write");
    WriteName += itostr(1 << i);
    TsanWrite[i] = M.getOrInsertFunction(WriteName, IRB.getVoidTy(),
                                         IRB.getInt8PtrTy(), NULL);
  }
  TsanVptrUpdate = M.getOrInsertFunction("__tsan_vptr_update", IRB.getVoidTy(),
                                         IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                                         NULL);
  return true;
}

bool ThreadSanitizer::runOnFunction(Function &F) {
  if (!TD) return false;
  if (BL->isIn(F)) return false;
  SmallVector<Instruction*, 8> RetVec;
  SmallVector<Instruction*, 8> LoadsAndStores;
  bool Res = false;
  bool HasCalls = false;

  // Traverse all instructions, collect loads/stores/returns, check for calls.
  for (Function::iterator FI = F.begin(), FE = F.end();
       FI != FE; ++FI) {
    BasicBlock &BB = *FI;
    for (BasicBlock::iterator BI = BB.begin(), BE = BB.end();
         BI != BE; ++BI) {
      if (isa<LoadInst>(BI) || isa<StoreInst>(BI))
        LoadsAndStores.push_back(BI);
      else if (isa<ReturnInst>(BI))
        RetVec.push_back(BI);
      else if (isa<CallInst>(BI) || isa<InvokeInst>(BI))
        HasCalls = true;
    }
  }

  // We have collected all loads and stores.
  // FIXME: many of these accesses do not need to be checked for races
  // (e.g. variables that do not escape, etc).

  // Instrument memory accesses.
  for (size_t i = 0, n = LoadsAndStores.size(); i < n; ++i) {
    Res |= instrumentLoadOrStore(LoadsAndStores[i]);
  }

  // Instrument function entry/exit points if there were instrumented accesses.
  if (Res || HasCalls) {
    IRBuilder<> IRB(F.getEntryBlock().getFirstNonPHI());
    Value *ReturnAddress = IRB.CreateCall(
        Intrinsic::getDeclaration(F.getParent(), Intrinsic::returnaddress),
        IRB.getInt32(0));
    IRB.CreateCall(TsanFuncEntry, ReturnAddress);
    for (size_t i = 0, n = RetVec.size(); i < n; ++i) {
      IRBuilder<> IRBRet(RetVec[i]);
      IRBRet.CreateCall(TsanFuncExit);
    }
    Res = true;
  }
  return Res;
}

static bool isVtableAccess(Instruction *I) {
  if (MDNode *Tag = I->getMetadata(LLVMContext::MD_tbaa)) {
    if (Tag->getNumOperands() < 1) return false;
    if (MDString *Tag1 = dyn_cast<MDString>(Tag->getOperand(0))) {
      if (Tag1->getString() == "vtable pointer") return true;
    }
  }
  return false;
}

bool ThreadSanitizer::instrumentLoadOrStore(Instruction *I) {
  IRBuilder<> IRB(I);
  bool IsWrite = isa<StoreInst>(*I);
  Value *Addr = IsWrite
      ? cast<StoreInst>(I)->getPointerOperand()
      : cast<LoadInst>(I)->getPointerOperand();
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  uint32_t TypeSize = TD->getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8  && TypeSize != 16 &&
      TypeSize != 32 && TypeSize != 64 && TypeSize != 128) {
    // Ignore all unusual sizes.
    return false;
  }
  if (IsWrite && isVtableAccess(I)) {
    Value *StoredValue = cast<StoreInst>(I)->getValueOperand();
    IRB.CreateCall2(TsanVptrUpdate,
                    IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                    IRB.CreatePointerCast(StoredValue, IRB.getInt8PtrTy()));
    return true;
  }
  size_t Idx = CountTrailingZeros_32(TypeSize / 8);
  assert(Idx < kNumberOfAccessSizes);
  Value *OnAccessFunc = IsWrite ? TsanWrite[Idx] : TsanRead[Idx];
  IRB.CreateCall(OnAccessFunc, IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()));
  return true;
}
