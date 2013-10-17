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

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/SpecialCaseList.h"

using namespace llvm;

static cl::opt<std::string>  ClBlacklistFile("tsan-blacklist",
       cl::desc("Blacklist file"), cl::Hidden);
static cl::opt<bool>  ClInstrumentMemoryAccesses(
    "tsan-instrument-memory-accesses", cl::init(true),
    cl::desc("Instrument memory accesses"), cl::Hidden);
static cl::opt<bool>  ClInstrumentFuncEntryExit(
    "tsan-instrument-func-entry-exit", cl::init(true),
    cl::desc("Instrument function entry and exit"), cl::Hidden);
static cl::opt<bool>  ClInstrumentAtomics(
    "tsan-instrument-atomics", cl::init(true),
    cl::desc("Instrument atomics"), cl::Hidden);
static cl::opt<bool>  ClInstrumentMemIntrinsics(
    "tsan-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumOmittedReadsBeforeWrite,
          "Number of reads ignored due to following writes");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");
STATISTIC(NumInstrumentedVtableWrites, "Number of vtable ptr writes");
STATISTIC(NumInstrumentedVtableReads, "Number of vtable ptr reads");
STATISTIC(NumOmittedReadsFromConstantGlobals,
          "Number of reads from constant globals");
STATISTIC(NumOmittedReadsFromVtable, "Number of vtable reads");

namespace {

/// ThreadSanitizer: instrument the code in module to find races.
struct ThreadSanitizer : public FunctionPass {
  ThreadSanitizer(StringRef BlacklistFile = StringRef())
      : FunctionPass(ID),
        TD(0),
        BlacklistFile(BlacklistFile.empty() ? ClBlacklistFile
                                            : BlacklistFile) { }
  const char *getPassName() const;
  bool runOnFunction(Function &F);
  bool doInitialization(Module &M);
  static char ID;  // Pass identification, replacement for typeid.

 private:
  void initializeCallbacks(Module &M);
  bool instrumentLoadOrStore(Instruction *I);
  bool instrumentAtomic(Instruction *I);
  bool instrumentMemIntrinsic(Instruction *I);
  void chooseInstructionsToInstrument(SmallVectorImpl<Instruction*> &Local,
                                      SmallVectorImpl<Instruction*> &All);
  bool addrPointsToConstantData(Value *Addr);
  int getMemoryAccessFuncIndex(Value *Addr);

  DataLayout *TD;
  Type *IntptrTy;
  SmallString<64> BlacklistFile;
  OwningPtr<SpecialCaseList> BL;
  IntegerType *OrdTy;
  // Callbacks to run-time library are computed in doInitialization.
  Function *TsanFuncEntry;
  Function *TsanFuncExit;
  // Accesses sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t kNumberOfAccessSizes = 5;
  Function *TsanRead[kNumberOfAccessSizes];
  Function *TsanWrite[kNumberOfAccessSizes];
  Function *TsanAtomicLoad[kNumberOfAccessSizes];
  Function *TsanAtomicStore[kNumberOfAccessSizes];
  Function *TsanAtomicRMW[AtomicRMWInst::LAST_BINOP + 1][kNumberOfAccessSizes];
  Function *TsanAtomicCAS[kNumberOfAccessSizes];
  Function *TsanAtomicThreadFence;
  Function *TsanAtomicSignalFence;
  Function *TsanVptrUpdate;
  Function *TsanVptrLoad;
  Function *MemmoveFn, *MemcpyFn, *MemsetFn;
};
}  // namespace

char ThreadSanitizer::ID = 0;
INITIALIZE_PASS(ThreadSanitizer, "tsan",
    "ThreadSanitizer: detects data races.",
    false, false)

const char *ThreadSanitizer::getPassName() const {
  return "ThreadSanitizer";
}

FunctionPass *llvm::createThreadSanitizerPass(StringRef BlacklistFile) {
  return new ThreadSanitizer(BlacklistFile);
}

static Function *checkInterfaceFunction(Constant *FuncOrBitcast) {
  if (Function *F = dyn_cast<Function>(FuncOrBitcast))
     return F;
  FuncOrBitcast->dump();
  report_fatal_error("ThreadSanitizer interface function redefined");
}

void ThreadSanitizer::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  // Initialize the callbacks.
  TsanFuncEntry = checkInterfaceFunction(M.getOrInsertFunction(
      "__tsan_func_entry", IRB.getVoidTy(), IRB.getInt8PtrTy(), NULL));
  TsanFuncExit = checkInterfaceFunction(M.getOrInsertFunction(
      "__tsan_func_exit", IRB.getVoidTy(), NULL));
  OrdTy = IRB.getInt32Ty();
  for (size_t i = 0; i < kNumberOfAccessSizes; ++i) {
    const size_t ByteSize = 1 << i;
    const size_t BitSize = ByteSize * 8;
    SmallString<32> ReadName("__tsan_read" + itostr(ByteSize));
    TsanRead[i] = checkInterfaceFunction(M.getOrInsertFunction(
        ReadName, IRB.getVoidTy(), IRB.getInt8PtrTy(), NULL));

    SmallString<32> WriteName("__tsan_write" + itostr(ByteSize));
    TsanWrite[i] = checkInterfaceFunction(M.getOrInsertFunction(
        WriteName, IRB.getVoidTy(), IRB.getInt8PtrTy(), NULL));

    Type *Ty = Type::getIntNTy(M.getContext(), BitSize);
    Type *PtrTy = Ty->getPointerTo();
    SmallString<32> AtomicLoadName("__tsan_atomic" + itostr(BitSize) +
                                   "_load");
    TsanAtomicLoad[i] = checkInterfaceFunction(M.getOrInsertFunction(
        AtomicLoadName, Ty, PtrTy, OrdTy, NULL));

    SmallString<32> AtomicStoreName("__tsan_atomic" + itostr(BitSize) +
                                    "_store");
    TsanAtomicStore[i] = checkInterfaceFunction(M.getOrInsertFunction(
        AtomicStoreName, IRB.getVoidTy(), PtrTy, Ty, OrdTy,
        NULL));

    for (int op = AtomicRMWInst::FIRST_BINOP;
        op <= AtomicRMWInst::LAST_BINOP; ++op) {
      TsanAtomicRMW[op][i] = NULL;
      const char *NamePart = NULL;
      if (op == AtomicRMWInst::Xchg)
        NamePart = "_exchange";
      else if (op == AtomicRMWInst::Add)
        NamePart = "_fetch_add";
      else if (op == AtomicRMWInst::Sub)
        NamePart = "_fetch_sub";
      else if (op == AtomicRMWInst::And)
        NamePart = "_fetch_and";
      else if (op == AtomicRMWInst::Or)
        NamePart = "_fetch_or";
      else if (op == AtomicRMWInst::Xor)
        NamePart = "_fetch_xor";
      else if (op == AtomicRMWInst::Nand)
        NamePart = "_fetch_nand";
      else
        continue;
      SmallString<32> RMWName("__tsan_atomic" + itostr(BitSize) + NamePart);
      TsanAtomicRMW[op][i] = checkInterfaceFunction(M.getOrInsertFunction(
          RMWName, Ty, PtrTy, Ty, OrdTy, NULL));
    }

    SmallString<32> AtomicCASName("__tsan_atomic" + itostr(BitSize) +
                                  "_compare_exchange_val");
    TsanAtomicCAS[i] = checkInterfaceFunction(M.getOrInsertFunction(
        AtomicCASName, Ty, PtrTy, Ty, Ty, OrdTy, OrdTy, NULL));
  }
  TsanVptrUpdate = checkInterfaceFunction(M.getOrInsertFunction(
      "__tsan_vptr_update", IRB.getVoidTy(), IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), NULL));
  TsanVptrLoad = checkInterfaceFunction(M.getOrInsertFunction(
      "__tsan_vptr_read", IRB.getVoidTy(), IRB.getInt8PtrTy(), NULL));
  TsanAtomicThreadFence = checkInterfaceFunction(M.getOrInsertFunction(
      "__tsan_atomic_thread_fence", IRB.getVoidTy(), OrdTy, NULL));
  TsanAtomicSignalFence = checkInterfaceFunction(M.getOrInsertFunction(
      "__tsan_atomic_signal_fence", IRB.getVoidTy(), OrdTy, NULL));

  MemmoveFn = checkInterfaceFunction(M.getOrInsertFunction(
    "memmove", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
    IRB.getInt8PtrTy(), IntptrTy, NULL));
  MemcpyFn = checkInterfaceFunction(M.getOrInsertFunction(
    "memcpy", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
    IntptrTy, NULL));
  MemsetFn = checkInterfaceFunction(M.getOrInsertFunction(
    "memset", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IRB.getInt32Ty(),
    IntptrTy, NULL));
}

bool ThreadSanitizer::doInitialization(Module &M) {
  TD = getAnalysisIfAvailable<DataLayout>();
  if (!TD)
    return false;
  BL.reset(SpecialCaseList::createOrDie(BlacklistFile));

  // Always insert a call to __tsan_init into the module's CTORs.
  IRBuilder<> IRB(M.getContext());
  IntptrTy = IRB.getIntPtrTy(TD);
  Value *TsanInit = M.getOrInsertFunction("__tsan_init",
                                          IRB.getVoidTy(), NULL);
  appendToGlobalCtors(M, cast<Function>(TsanInit), 0);

  return true;
}

static bool isVtableAccess(Instruction *I) {
  if (MDNode *Tag = I->getMetadata(LLVMContext::MD_tbaa))
    return Tag->isTBAAVtableAccess();
  return false;
}

bool ThreadSanitizer::addrPointsToConstantData(Value *Addr) {
  // If this is a GEP, just analyze its pointer operand.
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr))
    Addr = GEP->getPointerOperand();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->isConstant()) {
      // Reads from constant globals can not race with any writes.
      NumOmittedReadsFromConstantGlobals++;
      return true;
    }
  } else if (LoadInst *L = dyn_cast<LoadInst>(Addr)) {
    if (isVtableAccess(L)) {
      // Reads from a vtable pointer can not race with any writes.
      NumOmittedReadsFromVtable++;
      return true;
    }
  }
  return false;
}

// Instrumenting some of the accesses may be proven redundant.
// Currently handled:
//  - read-before-write (within same BB, no calls between)
//
// We do not handle some of the patterns that should not survive
// after the classic compiler optimizations.
// E.g. two reads from the same temp should be eliminated by CSE,
// two writes should be eliminated by DSE, etc.
//
// 'Local' is a vector of insns within the same BB (no calls between).
// 'All' is a vector of insns that will be instrumented.
void ThreadSanitizer::chooseInstructionsToInstrument(
    SmallVectorImpl<Instruction*> &Local,
    SmallVectorImpl<Instruction*> &All) {
  SmallSet<Value*, 8> WriteTargets;
  // Iterate from the end.
  for (SmallVectorImpl<Instruction*>::reverse_iterator It = Local.rbegin(),
       E = Local.rend(); It != E; ++It) {
    Instruction *I = *It;
    if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
      WriteTargets.insert(Store->getPointerOperand());
    } else {
      LoadInst *Load = cast<LoadInst>(I);
      Value *Addr = Load->getPointerOperand();
      if (WriteTargets.count(Addr)) {
        // We will write to this temp, so no reason to analyze the read.
        NumOmittedReadsBeforeWrite++;
        continue;
      }
      if (addrPointsToConstantData(Addr)) {
        // Addr points to some constant data -- it can not race with any writes.
        continue;
      }
    }
    All.push_back(I);
  }
  Local.clear();
}

static bool isAtomic(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->isAtomic() && LI->getSynchScope() == CrossThread;
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isAtomic() && SI->getSynchScope() == CrossThread;
  if (isa<AtomicRMWInst>(I))
    return true;
  if (isa<AtomicCmpXchgInst>(I))
    return true;
  if (isa<FenceInst>(I))
    return true;
  return false;
}

bool ThreadSanitizer::runOnFunction(Function &F) {
  if (!TD) return false;
  if (BL->isIn(F)) return false;
  initializeCallbacks(*F.getParent());
  SmallVector<Instruction*, 8> RetVec;
  SmallVector<Instruction*, 8> AllLoadsAndStores;
  SmallVector<Instruction*, 8> LocalLoadsAndStores;
  SmallVector<Instruction*, 8> AtomicAccesses;
  SmallVector<Instruction*, 8> MemIntrinCalls;
  bool Res = false;
  bool HasCalls = false;

  // Traverse all instructions, collect loads/stores/returns, check for calls.
  for (Function::iterator FI = F.begin(), FE = F.end();
       FI != FE; ++FI) {
    BasicBlock &BB = *FI;
    for (BasicBlock::iterator BI = BB.begin(), BE = BB.end();
         BI != BE; ++BI) {
      if (isAtomic(BI))
        AtomicAccesses.push_back(BI);
      else if (isa<LoadInst>(BI) || isa<StoreInst>(BI))
        LocalLoadsAndStores.push_back(BI);
      else if (isa<ReturnInst>(BI))
        RetVec.push_back(BI);
      else if (isa<CallInst>(BI) || isa<InvokeInst>(BI)) {
        if (isa<MemIntrinsic>(BI))
          MemIntrinCalls.push_back(BI);
        HasCalls = true;
        chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores);
      }
    }
    chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores);
  }

  // We have collected all loads and stores.
  // FIXME: many of these accesses do not need to be checked for races
  // (e.g. variables that do not escape, etc).

  // Instrument memory accesses.
  if (ClInstrumentMemoryAccesses && F.hasFnAttribute(Attribute::SanitizeThread))
    for (size_t i = 0, n = AllLoadsAndStores.size(); i < n; ++i) {
      Res |= instrumentLoadOrStore(AllLoadsAndStores[i]);
    }

  // Instrument atomic memory accesses.
  if (ClInstrumentAtomics)
    for (size_t i = 0, n = AtomicAccesses.size(); i < n; ++i) {
      Res |= instrumentAtomic(AtomicAccesses[i]);
    }

  if (ClInstrumentMemIntrinsics)
    for (size_t i = 0, n = MemIntrinCalls.size(); i < n; ++i) {
      Res |= instrumentMemIntrinsic(MemIntrinCalls[i]);
    }

  // Instrument function entry/exit points if there were instrumented accesses.
  if ((Res || HasCalls) && ClInstrumentFuncEntryExit) {
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

bool ThreadSanitizer::instrumentLoadOrStore(Instruction *I) {
  IRBuilder<> IRB(I);
  bool IsWrite = isa<StoreInst>(*I);
  Value *Addr = IsWrite
      ? cast<StoreInst>(I)->getPointerOperand()
      : cast<LoadInst>(I)->getPointerOperand();
  int Idx = getMemoryAccessFuncIndex(Addr);
  if (Idx < 0)
    return false;
  if (IsWrite && isVtableAccess(I)) {
    DEBUG(dbgs() << "  VPTR : " << *I << "\n");
    Value *StoredValue = cast<StoreInst>(I)->getValueOperand();
    // StoredValue does not necessary have a pointer type.
    if (isa<IntegerType>(StoredValue->getType()))
      StoredValue = IRB.CreateIntToPtr(StoredValue, IRB.getInt8PtrTy());
    // Call TsanVptrUpdate.
    IRB.CreateCall2(TsanVptrUpdate,
                    IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                    IRB.CreatePointerCast(StoredValue, IRB.getInt8PtrTy()));
    NumInstrumentedVtableWrites++;
    return true;
  }
  if (!IsWrite && isVtableAccess(I)) {
    IRB.CreateCall(TsanVptrLoad,
                   IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()));
    NumInstrumentedVtableReads++;
    return true;
  }
  Value *OnAccessFunc = IsWrite ? TsanWrite[Idx] : TsanRead[Idx];
  IRB.CreateCall(OnAccessFunc, IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()));
  if (IsWrite) NumInstrumentedWrites++;
  else         NumInstrumentedReads++;
  return true;
}

static ConstantInt *createOrdering(IRBuilder<> *IRB, AtomicOrdering ord) {
  uint32_t v = 0;
  switch (ord) {
    case NotAtomic:              assert(false);
    case Unordered:              // Fall-through.
    case Monotonic:              v = 0; break;
    // case Consume:                v = 1; break;  // Not specified yet.
    case Acquire:                v = 2; break;
    case Release:                v = 3; break;
    case AcquireRelease:         v = 4; break;
    case SequentiallyConsistent: v = 5; break;
  }
  return IRB->getInt32(v);
}

static ConstantInt *createFailOrdering(IRBuilder<> *IRB, AtomicOrdering ord) {
  uint32_t v = 0;
  switch (ord) {
    case NotAtomic:              assert(false);
    case Unordered:              // Fall-through.
    case Monotonic:              v = 0; break;
    // case Consume:                v = 1; break;  // Not specified yet.
    case Acquire:                v = 2; break;
    case Release:                v = 0; break;
    case AcquireRelease:         v = 2; break;
    case SequentiallyConsistent: v = 5; break;
  }
  return IRB->getInt32(v);
}

// If a memset intrinsic gets inlined by the code gen, we will miss races on it.
// So, we either need to ensure the intrinsic is not inlined, or instrument it.
// We do not instrument memset/memmove/memcpy intrinsics (too complicated),
// instead we simply replace them with regular function calls, which are then
// intercepted by the run-time.
// Since tsan is running after everyone else, the calls should not be
// replaced back with intrinsics. If that becomes wrong at some point,
// we will need to call e.g. __tsan_memset to avoid the intrinsics.
bool ThreadSanitizer::instrumentMemIntrinsic(Instruction *I) {
  IRBuilder<> IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    IRB.CreateCall3(MemsetFn,
      IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
      IRB.CreateIntCast(M->getArgOperand(1), IRB.getInt32Ty(), false),
      IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false));
    I->eraseFromParent();
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    IRB.CreateCall3(isa<MemCpyInst>(M) ? MemcpyFn : MemmoveFn,
      IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
      IRB.CreatePointerCast(M->getArgOperand(1), IRB.getInt8PtrTy()),
      IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false));
    I->eraseFromParent();
  }
  return false;
}

// Both llvm and ThreadSanitizer atomic operations are based on C++11/C1x
// standards.  For background see C++11 standard.  A slightly older, publically
// available draft of the standard (not entirely up-to-date, but close enough
// for casual browsing) is available here:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3242.pdf
// The following page contains more background information:
// http://www.hpl.hp.com/personal/Hans_Boehm/c++mm/

bool ThreadSanitizer::instrumentAtomic(Instruction *I) {
  IRBuilder<> IRB(I);
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Value *Addr = LI->getPointerOperand();
    int Idx = getMemoryAccessFuncIndex(Addr);
    if (Idx < 0)
      return false;
    const size_t ByteSize = 1 << Idx;
    const size_t BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Type *PtrTy = Ty->getPointerTo();
    Value *Args[] = {IRB.CreatePointerCast(Addr, PtrTy),
                     createOrdering(&IRB, LI->getOrdering())};
    CallInst *C = CallInst::Create(TsanAtomicLoad[Idx],
                                   ArrayRef<Value*>(Args));
    ReplaceInstWithInst(I, C);

  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Value *Addr = SI->getPointerOperand();
    int Idx = getMemoryAccessFuncIndex(Addr);
    if (Idx < 0)
      return false;
    const size_t ByteSize = 1 << Idx;
    const size_t BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Type *PtrTy = Ty->getPointerTo();
    Value *Args[] = {IRB.CreatePointerCast(Addr, PtrTy),
                     IRB.CreateIntCast(SI->getValueOperand(), Ty, false),
                     createOrdering(&IRB, SI->getOrdering())};
    CallInst *C = CallInst::Create(TsanAtomicStore[Idx],
                                   ArrayRef<Value*>(Args));
    ReplaceInstWithInst(I, C);
  } else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
    Value *Addr = RMWI->getPointerOperand();
    int Idx = getMemoryAccessFuncIndex(Addr);
    if (Idx < 0)
      return false;
    Function *F = TsanAtomicRMW[RMWI->getOperation()][Idx];
    if (F == NULL)
      return false;
    const size_t ByteSize = 1 << Idx;
    const size_t BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Type *PtrTy = Ty->getPointerTo();
    Value *Args[] = {IRB.CreatePointerCast(Addr, PtrTy),
                     IRB.CreateIntCast(RMWI->getValOperand(), Ty, false),
                     createOrdering(&IRB, RMWI->getOrdering())};
    CallInst *C = CallInst::Create(F, ArrayRef<Value*>(Args));
    ReplaceInstWithInst(I, C);
  } else if (AtomicCmpXchgInst *CASI = dyn_cast<AtomicCmpXchgInst>(I)) {
    Value *Addr = CASI->getPointerOperand();
    int Idx = getMemoryAccessFuncIndex(Addr);
    if (Idx < 0)
      return false;
    const size_t ByteSize = 1 << Idx;
    const size_t BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Type *PtrTy = Ty->getPointerTo();
    Value *Args[] = {IRB.CreatePointerCast(Addr, PtrTy),
                     IRB.CreateIntCast(CASI->getCompareOperand(), Ty, false),
                     IRB.CreateIntCast(CASI->getNewValOperand(), Ty, false),
                     createOrdering(&IRB, CASI->getOrdering()),
                     createFailOrdering(&IRB, CASI->getOrdering())};
    CallInst *C = CallInst::Create(TsanAtomicCAS[Idx], ArrayRef<Value*>(Args));
    ReplaceInstWithInst(I, C);
  } else if (FenceInst *FI = dyn_cast<FenceInst>(I)) {
    Value *Args[] = {createOrdering(&IRB, FI->getOrdering())};
    Function *F = FI->getSynchScope() == SingleThread ?
        TsanAtomicSignalFence : TsanAtomicThreadFence;
    CallInst *C = CallInst::Create(F, ArrayRef<Value*>(Args));
    ReplaceInstWithInst(I, C);
  }
  return true;
}

int ThreadSanitizer::getMemoryAccessFuncIndex(Value *Addr) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  uint32_t TypeSize = TD->getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8  && TypeSize != 16 &&
      TypeSize != 32 && TypeSize != 64 && TypeSize != 128) {
    NumAccessesWithBadSize++;
    // Ignore all unusual sizes.
    return -1;
  }
  size_t Idx = countTrailingZeros(TypeSize / 8);
  assert(Idx < kNumberOfAccessSizes);
  return Idx;
}
