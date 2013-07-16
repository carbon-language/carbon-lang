//===-- AddressSanitizer.cpp - memory error detector ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
// Details of the algorithm:
//  http://code.google.com/p/address-sanitizer/wiki/AddressSanitizerAlgorithm
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asan"

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InstVisitor.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/SpecialCaseList.h"
#include <algorithm>
#include <string>

using namespace llvm;

static const uint64_t kDefaultShadowScale = 3;
static const uint64_t kDefaultShadowOffset32 = 1ULL << 29;
static const uint64_t kDefaultShadowOffset64 = 1ULL << 44;
static const uint64_t kDefaultShort64bitShadowOffset = 0x7FFF8000;  // < 2G.
static const uint64_t kPPC64_ShadowOffset64 = 1ULL << 41;
static const uint64_t kMIPS32_ShadowOffset32 = 0x0aaa8000;

static const size_t kMaxStackMallocSize = 1 << 16;  // 64K
static const uintptr_t kCurrentStackFrameMagic = 0x41B58AB3;
static const uintptr_t kRetiredStackFrameMagic = 0x45E0360E;

static const char *const kAsanModuleCtorName = "asan.module_ctor";
static const char *const kAsanModuleDtorName = "asan.module_dtor";
static const int         kAsanCtorAndCtorPriority = 1;
static const char *const kAsanReportErrorTemplate = "__asan_report_";
static const char *const kAsanReportLoadN = "__asan_report_load_n";
static const char *const kAsanReportStoreN = "__asan_report_store_n";
static const char *const kAsanRegisterGlobalsName = "__asan_register_globals";
static const char *const kAsanUnregisterGlobalsName = "__asan_unregister_globals";
static const char *const kAsanPoisonGlobalsName = "__asan_before_dynamic_init";
static const char *const kAsanUnpoisonGlobalsName = "__asan_after_dynamic_init";
static const char *const kAsanInitName = "__asan_init_v3";
static const char *const kAsanHandleNoReturnName = "__asan_handle_no_return";
static const char *const kAsanMappingOffsetName = "__asan_mapping_offset";
static const char *const kAsanMappingScaleName = "__asan_mapping_scale";
static const char *const kAsanStackMallocName = "__asan_stack_malloc";
static const char *const kAsanStackFreeName = "__asan_stack_free";
static const char *const kAsanGenPrefix = "__asan_gen_";
static const char *const kAsanPoisonStackMemoryName =
    "__asan_poison_stack_memory";
static const char *const kAsanUnpoisonStackMemoryName =
    "__asan_unpoison_stack_memory";

static const int kAsanStackLeftRedzoneMagic = 0xf1;
static const int kAsanStackMidRedzoneMagic = 0xf2;
static const int kAsanStackRightRedzoneMagic = 0xf3;
static const int kAsanStackPartialRedzoneMagic = 0xf4;

// Accesses sizes are powers of two: 1, 2, 4, 8, 16.
static const size_t kNumberOfAccessSizes = 5;

// Command-line flags.

// This flag may need to be replaced with -f[no-]asan-reads.
static cl::opt<bool> ClInstrumentReads("asan-instrument-reads",
       cl::desc("instrument read instructions"), cl::Hidden, cl::init(true));
static cl::opt<bool> ClInstrumentWrites("asan-instrument-writes",
       cl::desc("instrument write instructions"), cl::Hidden, cl::init(true));
static cl::opt<bool> ClInstrumentAtomics("asan-instrument-atomics",
       cl::desc("instrument atomic instructions (rmw, cmpxchg)"),
       cl::Hidden, cl::init(true));
static cl::opt<bool> ClAlwaysSlowPath("asan-always-slow-path",
       cl::desc("use instrumentation with slow path for all accesses"),
       cl::Hidden, cl::init(false));
// This flag limits the number of instructions to be instrumented
// in any given BB. Normally, this should be set to unlimited (INT_MAX),
// but due to http://llvm.org/bugs/show_bug.cgi?id=12652 we temporary
// set it to 10000.
static cl::opt<int> ClMaxInsnsToInstrumentPerBB("asan-max-ins-per-bb",
       cl::init(10000),
       cl::desc("maximal number of instructions to instrument in any given BB"),
       cl::Hidden);
// This flag may need to be replaced with -f[no]asan-stack.
static cl::opt<bool> ClStack("asan-stack",
       cl::desc("Handle stack memory"), cl::Hidden, cl::init(true));
// This flag may need to be replaced with -f[no]asan-use-after-return.
static cl::opt<bool> ClUseAfterReturn("asan-use-after-return",
       cl::desc("Check return-after-free"), cl::Hidden, cl::init(false));
// This flag may need to be replaced with -f[no]asan-globals.
static cl::opt<bool> ClGlobals("asan-globals",
       cl::desc("Handle global objects"), cl::Hidden, cl::init(true));
static cl::opt<bool> ClInitializers("asan-initialization-order",
       cl::desc("Handle C++ initializer order"), cl::Hidden, cl::init(false));
static cl::opt<bool> ClMemIntrin("asan-memintrin",
       cl::desc("Handle memset/memcpy/memmove"), cl::Hidden, cl::init(true));
static cl::opt<bool> ClRealignStack("asan-realign-stack",
       cl::desc("Realign stack to 32"), cl::Hidden, cl::init(true));
static cl::opt<std::string> ClBlacklistFile("asan-blacklist",
       cl::desc("File containing the list of objects to ignore "
                "during instrumentation"), cl::Hidden);

// This is an experimental feature that will allow to choose between
// instrumented and non-instrumented code at link-time.
// If this option is on, just before instrumenting a function we create its
// clone; if the function is not changed by asan the clone is deleted.
// If we end up with a clone, we put the instrumented function into a section
// called "ASAN" and the uninstrumented function into a section called "NOASAN".
//
// This is still a prototype, we need to figure out a way to keep two copies of
// a function so that the linker can easily choose one of them.
static cl::opt<bool> ClKeepUninstrumented("asan-keep-uninstrumented-functions",
       cl::desc("Keep uninstrumented copies of functions"),
       cl::Hidden, cl::init(false));

// These flags allow to change the shadow mapping.
// The shadow mapping looks like
//    Shadow = (Mem >> scale) + (1 << offset_log)
static cl::opt<int> ClMappingScale("asan-mapping-scale",
       cl::desc("scale of asan shadow mapping"), cl::Hidden, cl::init(0));
static cl::opt<int> ClMappingOffsetLog("asan-mapping-offset-log",
       cl::desc("offset of asan shadow mapping"), cl::Hidden, cl::init(-1));
static cl::opt<bool> ClShort64BitOffset("asan-short-64bit-mapping-offset",
       cl::desc("Use short immediate constant as the mapping offset for 64bit"),
       cl::Hidden, cl::init(true));

// Optimization flags. Not user visible, used mostly for testing
// and benchmarking the tool.
static cl::opt<bool> ClOpt("asan-opt",
       cl::desc("Optimize instrumentation"), cl::Hidden, cl::init(true));
static cl::opt<bool> ClOptSameTemp("asan-opt-same-temp",
       cl::desc("Instrument the same temp just once"), cl::Hidden,
       cl::init(true));
static cl::opt<bool> ClOptGlobals("asan-opt-globals",
       cl::desc("Don't instrument scalar globals"), cl::Hidden, cl::init(true));

static cl::opt<bool> ClCheckLifetime("asan-check-lifetime",
       cl::desc("Use llvm.lifetime intrinsics to insert extra checks"),
       cl::Hidden, cl::init(false));

// Debug flags.
static cl::opt<int> ClDebug("asan-debug", cl::desc("debug"), cl::Hidden,
                            cl::init(0));
static cl::opt<int> ClDebugStack("asan-debug-stack", cl::desc("debug stack"),
                                 cl::Hidden, cl::init(0));
static cl::opt<std::string> ClDebugFunc("asan-debug-func",
                                        cl::Hidden, cl::desc("Debug func"));
static cl::opt<int> ClDebugMin("asan-debug-min", cl::desc("Debug min inst"),
                               cl::Hidden, cl::init(-1));
static cl::opt<int> ClDebugMax("asan-debug-max", cl::desc("Debug man inst"),
                               cl::Hidden, cl::init(-1));

namespace {
/// A set of dynamically initialized globals extracted from metadata.
class SetOfDynamicallyInitializedGlobals {
 public:
  void Init(Module& M) {
    // Clang generates metadata identifying all dynamically initialized globals.
    NamedMDNode *DynamicGlobals =
        M.getNamedMetadata("llvm.asan.dynamically_initialized_globals");
    if (!DynamicGlobals)
      return;
    for (int i = 0, n = DynamicGlobals->getNumOperands(); i < n; ++i) {
      MDNode *MDN = DynamicGlobals->getOperand(i);
      assert(MDN->getNumOperands() == 1);
      Value *VG = MDN->getOperand(0);
      // The optimizer may optimize away a global entirely, in which case we
      // cannot instrument access to it.
      if (!VG)
        continue;
      DynInitGlobals.insert(cast<GlobalVariable>(VG));
    }
  }
  bool Contains(GlobalVariable *G) { return DynInitGlobals.count(G) != 0; }
 private:
  SmallSet<GlobalValue*, 32> DynInitGlobals;
};

/// This struct defines the shadow mapping using the rule:
///   shadow = (mem >> Scale) ADD-or-OR Offset.
struct ShadowMapping {
  int Scale;
  uint64_t Offset;
  bool OrShadowOffset;
};

static ShadowMapping getShadowMapping(const Module &M, int LongSize,
                                      bool ZeroBaseShadow) {
  llvm::Triple TargetTriple(M.getTargetTriple());
  bool IsAndroid = TargetTriple.getEnvironment() == llvm::Triple::Android;
  bool IsMacOSX = TargetTriple.getOS() == llvm::Triple::MacOSX;
  bool IsPPC64 = TargetTriple.getArch() == llvm::Triple::ppc64;
  bool IsX86_64 = TargetTriple.getArch() == llvm::Triple::x86_64;
  bool IsMIPS32 = TargetTriple.getArch() == llvm::Triple::mips ||
                  TargetTriple.getArch() == llvm::Triple::mipsel;

  ShadowMapping Mapping;

  // OR-ing shadow offset if more efficient (at least on x86),
  // but on ppc64 we have to use add since the shadow offset is not neccesary
  // 1/8-th of the address space.
  Mapping.OrShadowOffset = !IsPPC64 && !ClShort64BitOffset;

  Mapping.Offset = (IsAndroid || ZeroBaseShadow) ? 0 :
      (LongSize == 32 ?
       (IsMIPS32 ? kMIPS32_ShadowOffset32 : kDefaultShadowOffset32) :
       IsPPC64 ? kPPC64_ShadowOffset64 : kDefaultShadowOffset64);
  if (!ZeroBaseShadow && ClShort64BitOffset && IsX86_64 && !IsMacOSX) {
    assert(LongSize == 64);
    Mapping.Offset = kDefaultShort64bitShadowOffset;
  }
  if (!ZeroBaseShadow && ClMappingOffsetLog >= 0) {
    // Zero offset log is the special case.
    Mapping.Offset = (ClMappingOffsetLog == 0) ? 0 : 1ULL << ClMappingOffsetLog;
  }

  Mapping.Scale = kDefaultShadowScale;
  if (ClMappingScale) {
    Mapping.Scale = ClMappingScale;
  }

  return Mapping;
}

static size_t RedzoneSizeForScale(int MappingScale) {
  // Redzone used for stack and globals is at least 32 bytes.
  // For scales 6 and 7, the redzone has to be 64 and 128 bytes respectively.
  return std::max(32U, 1U << MappingScale);
}

/// AddressSanitizer: instrument the code in module to find memory bugs.
struct AddressSanitizer : public FunctionPass {
  AddressSanitizer(bool CheckInitOrder = true,
                   bool CheckUseAfterReturn = false,
                   bool CheckLifetime = false,
                   StringRef BlacklistFile = StringRef(),
                   bool ZeroBaseShadow = false)
      : FunctionPass(ID),
        CheckInitOrder(CheckInitOrder || ClInitializers),
        CheckUseAfterReturn(CheckUseAfterReturn || ClUseAfterReturn),
        CheckLifetime(CheckLifetime || ClCheckLifetime),
        BlacklistFile(BlacklistFile.empty() ? ClBlacklistFile
                                            : BlacklistFile),
        ZeroBaseShadow(ZeroBaseShadow) {}
  virtual const char *getPassName() const {
    return "AddressSanitizerFunctionPass";
  }
  void instrumentMop(Instruction *I);
  void instrumentAddress(Instruction *OrigIns, Instruction *InsertBefore,
                         Value *Addr, uint32_t TypeSize, bool IsWrite,
                         Value *SizeArgument);
  Value *createSlowPathCmp(IRBuilder<> &IRB, Value *AddrLong,
                           Value *ShadowValue, uint32_t TypeSize);
  Instruction *generateCrashCode(Instruction *InsertBefore, Value *Addr,
                                 bool IsWrite, size_t AccessSizeIndex,
                                 Value *SizeArgument);
  bool instrumentMemIntrinsic(MemIntrinsic *MI);
  void instrumentMemIntrinsicParam(Instruction *OrigIns, Value *Addr,
                                   Value *Size,
                                   Instruction *InsertBefore, bool IsWrite);
  Value *memToShadow(Value *Shadow, IRBuilder<> &IRB);
  bool runOnFunction(Function &F);
  bool maybeInsertAsanInitAtFunctionEntry(Function &F);
  void emitShadowMapping(Module &M, IRBuilder<> &IRB) const;
  virtual bool doInitialization(Module &M);
  static char ID;  // Pass identification, replacement for typeid

 private:
  void initializeCallbacks(Module &M);

  bool ShouldInstrumentGlobal(GlobalVariable *G);
  bool LooksLikeCodeInBug11395(Instruction *I);
  void FindDynamicInitializers(Module &M);

  bool CheckInitOrder;
  bool CheckUseAfterReturn;
  bool CheckLifetime;
  SmallString<64> BlacklistFile;
  bool ZeroBaseShadow;

  LLVMContext *C;
  DataLayout *TD;
  int LongSize;
  Type *IntptrTy;
  ShadowMapping Mapping;
  Function *AsanCtorFunction;
  Function *AsanInitFunction;
  Function *AsanHandleNoReturnFunc;
  OwningPtr<SpecialCaseList> BL;
  // This array is indexed by AccessIsWrite and log2(AccessSize).
  Function *AsanErrorCallback[2][kNumberOfAccessSizes];
  // This array is indexed by AccessIsWrite.
  Function *AsanErrorCallbackSized[2];
  InlineAsm *EmptyAsm;
  SetOfDynamicallyInitializedGlobals DynamicallyInitializedGlobals;

  friend struct FunctionStackPoisoner;
};

class AddressSanitizerModule : public ModulePass {
 public:
  AddressSanitizerModule(bool CheckInitOrder = true,
                         StringRef BlacklistFile = StringRef(),
                         bool ZeroBaseShadow = false)
      : ModulePass(ID),
        CheckInitOrder(CheckInitOrder || ClInitializers),
        BlacklistFile(BlacklistFile.empty() ? ClBlacklistFile
                                            : BlacklistFile),
        ZeroBaseShadow(ZeroBaseShadow) {}
  bool runOnModule(Module &M);
  static char ID;  // Pass identification, replacement for typeid
  virtual const char *getPassName() const {
    return "AddressSanitizerModule";
  }

 private:
  void initializeCallbacks(Module &M);

  bool ShouldInstrumentGlobal(GlobalVariable *G);
  void createInitializerPoisonCalls(Module &M, GlobalValue *ModuleName);
  size_t RedzoneSize() const {
    return RedzoneSizeForScale(Mapping.Scale);
  }

  bool CheckInitOrder;
  SmallString<64> BlacklistFile;
  bool ZeroBaseShadow;

  OwningPtr<SpecialCaseList> BL;
  SetOfDynamicallyInitializedGlobals DynamicallyInitializedGlobals;
  Type *IntptrTy;
  LLVMContext *C;
  DataLayout *TD;
  ShadowMapping Mapping;
  Function *AsanPoisonGlobals;
  Function *AsanUnpoisonGlobals;
  Function *AsanRegisterGlobals;
  Function *AsanUnregisterGlobals;
};

// Stack poisoning does not play well with exception handling.
// When an exception is thrown, we essentially bypass the code
// that unpoisones the stack. This is why the run-time library has
// to intercept __cxa_throw (as well as longjmp, etc) and unpoison the entire
// stack in the interceptor. This however does not work inside the
// actual function which catches the exception. Most likely because the
// compiler hoists the load of the shadow value somewhere too high.
// This causes asan to report a non-existing bug on 453.povray.
// It sounds like an LLVM bug.
struct FunctionStackPoisoner : public InstVisitor<FunctionStackPoisoner> {
  Function &F;
  AddressSanitizer &ASan;
  DIBuilder DIB;
  LLVMContext *C;
  Type *IntptrTy;
  Type *IntptrPtrTy;
  ShadowMapping Mapping;

  SmallVector<AllocaInst*, 16> AllocaVec;
  SmallVector<Instruction*, 8> RetVec;
  uint64_t TotalStackSize;
  unsigned StackAlignment;

  Function *AsanStackMallocFunc, *AsanStackFreeFunc;
  Function *AsanPoisonStackMemoryFunc, *AsanUnpoisonStackMemoryFunc;

  // Stores a place and arguments of poisoning/unpoisoning call for alloca.
  struct AllocaPoisonCall {
    IntrinsicInst *InsBefore;
    uint64_t Size;
    bool DoPoison;
  };
  SmallVector<AllocaPoisonCall, 8> AllocaPoisonCallVec;

  // Maps Value to an AllocaInst from which the Value is originated.
  typedef DenseMap<Value*, AllocaInst*> AllocaForValueMapTy;
  AllocaForValueMapTy AllocaForValue;

  FunctionStackPoisoner(Function &F, AddressSanitizer &ASan)
      : F(F), ASan(ASan), DIB(*F.getParent()), C(ASan.C),
        IntptrTy(ASan.IntptrTy), IntptrPtrTy(PointerType::get(IntptrTy, 0)),
        Mapping(ASan.Mapping),
        TotalStackSize(0), StackAlignment(1 << Mapping.Scale) {}

  bool runOnFunction() {
    if (!ClStack) return false;
    // Collect alloca, ret, lifetime instructions etc.
    for (df_iterator<BasicBlock*> DI = df_begin(&F.getEntryBlock()),
         DE = df_end(&F.getEntryBlock()); DI != DE; ++DI) {
      BasicBlock *BB = *DI;
      visit(*BB);
    }
    if (AllocaVec.empty()) return false;

    initializeCallbacks(*F.getParent());

    poisonStack();

    if (ClDebugStack) {
      DEBUG(dbgs() << F);
    }
    return true;
  }

  // Finds all static Alloca instructions and puts
  // poisoned red zones around all of them.
  // Then unpoison everything back before the function returns.
  void poisonStack();

  // ----------------------- Visitors.
  /// \brief Collect all Ret instructions.
  void visitReturnInst(ReturnInst &RI) {
    RetVec.push_back(&RI);
  }

  /// \brief Collect Alloca instructions we want (and can) handle.
  void visitAllocaInst(AllocaInst &AI) {
    if (!isInterestingAlloca(AI)) return;

    StackAlignment = std::max(StackAlignment, AI.getAlignment());
    AllocaVec.push_back(&AI);
    uint64_t AlignedSize = getAlignedAllocaSize(&AI);
    TotalStackSize += AlignedSize;
  }

  /// \brief Collect lifetime intrinsic calls to check for use-after-scope
  /// errors.
  void visitIntrinsicInst(IntrinsicInst &II) {
    if (!ASan.CheckLifetime) return;
    Intrinsic::ID ID = II.getIntrinsicID();
    if (ID != Intrinsic::lifetime_start &&
        ID != Intrinsic::lifetime_end)
      return;
    // Found lifetime intrinsic, add ASan instrumentation if necessary.
    ConstantInt *Size = dyn_cast<ConstantInt>(II.getArgOperand(0));
    // If size argument is undefined, don't do anything.
    if (Size->isMinusOne()) return;
    // Check that size doesn't saturate uint64_t and can
    // be stored in IntptrTy.
    const uint64_t SizeValue = Size->getValue().getLimitedValue();
    if (SizeValue == ~0ULL ||
        !ConstantInt::isValueValidForType(IntptrTy, SizeValue))
      return;
    // Find alloca instruction that corresponds to llvm.lifetime argument.
    AllocaInst *AI = findAllocaForValue(II.getArgOperand(1));
    if (!AI) return;
    bool DoPoison = (ID == Intrinsic::lifetime_end);
    AllocaPoisonCall APC = {&II, SizeValue, DoPoison};
    AllocaPoisonCallVec.push_back(APC);
  }

  // ---------------------- Helpers.
  void initializeCallbacks(Module &M);

  // Check if we want (and can) handle this alloca.
  bool isInterestingAlloca(AllocaInst &AI) {
    return (!AI.isArrayAllocation() &&
            AI.isStaticAlloca() &&
            AI.getAlignment() <= RedzoneSize() &&
            AI.getAllocatedType()->isSized());
  }

  size_t RedzoneSize() const {
    return RedzoneSizeForScale(Mapping.Scale);
  }
  uint64_t getAllocaSizeInBytes(AllocaInst *AI) {
    Type *Ty = AI->getAllocatedType();
    uint64_t SizeInBytes = ASan.TD->getTypeAllocSize(Ty);
    return SizeInBytes;
  }
  uint64_t getAlignedSize(uint64_t SizeInBytes) {
    size_t RZ = RedzoneSize();
    return ((SizeInBytes + RZ - 1) / RZ) * RZ;
  }
  uint64_t getAlignedAllocaSize(AllocaInst *AI) {
    uint64_t SizeInBytes = getAllocaSizeInBytes(AI);
    return getAlignedSize(SizeInBytes);
  }
  /// Finds alloca where the value comes from.
  AllocaInst *findAllocaForValue(Value *V);
  void poisonRedZones(const ArrayRef<AllocaInst*> &AllocaVec, IRBuilder<> IRB,
                      Value *ShadowBase, bool DoPoison);
  void poisonAlloca(Value *V, uint64_t Size, IRBuilder<> IRB, bool DoPoison);
};

}  // namespace

char AddressSanitizer::ID = 0;
INITIALIZE_PASS(AddressSanitizer, "asan",
    "AddressSanitizer: detects use-after-free and out-of-bounds bugs.",
    false, false)
FunctionPass *llvm::createAddressSanitizerFunctionPass(
    bool CheckInitOrder, bool CheckUseAfterReturn, bool CheckLifetime,
    StringRef BlacklistFile, bool ZeroBaseShadow) {
  return new AddressSanitizer(CheckInitOrder, CheckUseAfterReturn,
                              CheckLifetime, BlacklistFile, ZeroBaseShadow);
}

char AddressSanitizerModule::ID = 0;
INITIALIZE_PASS(AddressSanitizerModule, "asan-module",
    "AddressSanitizer: detects use-after-free and out-of-bounds bugs."
    "ModulePass", false, false)
ModulePass *llvm::createAddressSanitizerModulePass(
    bool CheckInitOrder, StringRef BlacklistFile, bool ZeroBaseShadow) {
  return new AddressSanitizerModule(CheckInitOrder, BlacklistFile,
                                    ZeroBaseShadow);
}

static size_t TypeSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = countTrailingZeros(TypeSize / 8);
  assert(Res < kNumberOfAccessSizes);
  return Res;
}

// Create a constant for Str so that we can pass it to the run-time lib.
static GlobalVariable *createPrivateGlobalForString(Module &M, StringRef Str) {
  Constant *StrConst = ConstantDataArray::getString(M.getContext(), Str);
  GlobalVariable *GV = new GlobalVariable(M, StrConst->getType(), true,
                            GlobalValue::PrivateLinkage, StrConst,
                            kAsanGenPrefix);
  GV->setUnnamedAddr(true);  // Ok to merge these.
  GV->setAlignment(1);  // Strings may not be merged w/o setting align 1.
  return GV;
}

static bool GlobalWasGeneratedByAsan(GlobalVariable *G) {
  return G->getName().find(kAsanGenPrefix) == 0;
}

Value *AddressSanitizer::memToShadow(Value *Shadow, IRBuilder<> &IRB) {
  // Shadow >> scale
  Shadow = IRB.CreateLShr(Shadow, Mapping.Scale);
  if (Mapping.Offset == 0)
    return Shadow;
  // (Shadow >> scale) | offset
  if (Mapping.OrShadowOffset)
    return IRB.CreateOr(Shadow, ConstantInt::get(IntptrTy, Mapping.Offset));
  else
    return IRB.CreateAdd(Shadow, ConstantInt::get(IntptrTy, Mapping.Offset));
}

void AddressSanitizer::instrumentMemIntrinsicParam(
    Instruction *OrigIns,
    Value *Addr, Value *Size, Instruction *InsertBefore, bool IsWrite) {
  IRBuilder<> IRB(InsertBefore);
  if (Size->getType() != IntptrTy)
    Size = IRB.CreateIntCast(Size, IntptrTy, false);
  // Check the first byte.
  instrumentAddress(OrigIns, InsertBefore, Addr, 8, IsWrite, Size);
  // Check the last byte.
  IRB.SetInsertPoint(InsertBefore);
  Value *SizeMinusOne = IRB.CreateSub(Size, ConstantInt::get(IntptrTy, 1));
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  Value *AddrLast = IRB.CreateAdd(AddrLong, SizeMinusOne);
  instrumentAddress(OrigIns, InsertBefore, AddrLast, 8, IsWrite, Size);
}

// Instrument memset/memmove/memcpy
bool AddressSanitizer::instrumentMemIntrinsic(MemIntrinsic *MI) {
  Value *Dst = MI->getDest();
  MemTransferInst *MemTran = dyn_cast<MemTransferInst>(MI);
  Value *Src = MemTran ? MemTran->getSource() : 0;
  Value *Length = MI->getLength();

  Constant *ConstLength = dyn_cast<Constant>(Length);
  Instruction *InsertBefore = MI;
  if (ConstLength) {
    if (ConstLength->isNullValue()) return false;
  } else {
    // The size is not a constant so it could be zero -- check at run-time.
    IRBuilder<> IRB(InsertBefore);

    Value *Cmp = IRB.CreateICmpNE(Length,
                                  Constant::getNullValue(Length->getType()));
    InsertBefore = SplitBlockAndInsertIfThen(cast<Instruction>(Cmp), false);
  }

  instrumentMemIntrinsicParam(MI, Dst, Length, InsertBefore, true);
  if (Src)
    instrumentMemIntrinsicParam(MI, Src, Length, InsertBefore, false);
  return true;
}

// If I is an interesting memory access, return the PointerOperand
// and set IsWrite. Otherwise return NULL.
static Value *isInterestingMemoryAccess(Instruction *I, bool *IsWrite) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (!ClInstrumentReads) return NULL;
    *IsWrite = false;
    return LI->getPointerOperand();
  }
  if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (!ClInstrumentWrites) return NULL;
    *IsWrite = true;
    return SI->getPointerOperand();
  }
  if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    if (!ClInstrumentAtomics) return NULL;
    *IsWrite = true;
    return RMW->getPointerOperand();
  }
  if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    if (!ClInstrumentAtomics) return NULL;
    *IsWrite = true;
    return XCHG->getPointerOperand();
  }
  return NULL;
}

void AddressSanitizer::instrumentMop(Instruction *I) {
  bool IsWrite = false;
  Value *Addr = isInterestingMemoryAccess(I, &IsWrite);
  assert(Addr);
  if (ClOpt && ClOptGlobals) {
    if (GlobalVariable *G = dyn_cast<GlobalVariable>(Addr)) {
      // If initialization order checking is disabled, a simple access to a
      // dynamically initialized global is always valid.
      if (!CheckInitOrder)
        return;
      // If a global variable does not have dynamic initialization we don't
      // have to instrument it.  However, if a global does not have initailizer
      // at all, we assume it has dynamic initializer (in other TU).
      if (G->hasInitializer() && !DynamicallyInitializedGlobals.Contains(G))
        return;
    }
  }

  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();

  assert(OrigTy->isSized());
  uint32_t TypeSize = TD->getTypeStoreSizeInBits(OrigTy);

  assert((TypeSize % 8) == 0);

  // Instrument a 1-, 2-, 4-, 8-, or 16- byte access with one check.
  if (TypeSize == 8  || TypeSize == 16 ||
      TypeSize == 32 || TypeSize == 64 || TypeSize == 128)
    return instrumentAddress(I, I, Addr, TypeSize, IsWrite, 0);
  // Instrument unusual size (but still multiple of 8).
  // We can not do it with a single check, so we do 1-byte check for the first
  // and the last bytes. We call __asan_report_*_n(addr, real_size) to be able
  // to report the actual access size.
  IRBuilder<> IRB(I);
  Value *LastByte =  IRB.CreateIntToPtr(
      IRB.CreateAdd(IRB.CreatePointerCast(Addr, IntptrTy),
                    ConstantInt::get(IntptrTy, TypeSize / 8 - 1)),
      OrigPtrTy);
  Value *Size = ConstantInt::get(IntptrTy, TypeSize / 8);
  instrumentAddress(I, I, Addr, 8, IsWrite, Size);
  instrumentAddress(I, I, LastByte, 8, IsWrite, Size);
}

// Validate the result of Module::getOrInsertFunction called for an interface
// function of AddressSanitizer. If the instrumented module defines a function
// with the same name, their prototypes must match, otherwise
// getOrInsertFunction returns a bitcast.
static Function *checkInterfaceFunction(Constant *FuncOrBitcast) {
  if (isa<Function>(FuncOrBitcast)) return cast<Function>(FuncOrBitcast);
  FuncOrBitcast->dump();
  report_fatal_error("trying to redefine an AddressSanitizer "
                     "interface function");
}

Instruction *AddressSanitizer::generateCrashCode(
    Instruction *InsertBefore, Value *Addr,
    bool IsWrite, size_t AccessSizeIndex, Value *SizeArgument) {
  IRBuilder<> IRB(InsertBefore);
  CallInst *Call = SizeArgument
    ? IRB.CreateCall2(AsanErrorCallbackSized[IsWrite], Addr, SizeArgument)
    : IRB.CreateCall(AsanErrorCallback[IsWrite][AccessSizeIndex], Addr);

  // We don't do Call->setDoesNotReturn() because the BB already has
  // UnreachableInst at the end.
  // This EmptyAsm is required to avoid callback merge.
  IRB.CreateCall(EmptyAsm);
  return Call;
}

Value *AddressSanitizer::createSlowPathCmp(IRBuilder<> &IRB, Value *AddrLong,
                                            Value *ShadowValue,
                                            uint32_t TypeSize) {
  size_t Granularity = 1 << Mapping.Scale;
  // Addr & (Granularity - 1)
  Value *LastAccessedByte = IRB.CreateAnd(
      AddrLong, ConstantInt::get(IntptrTy, Granularity - 1));
  // (Addr & (Granularity - 1)) + size - 1
  if (TypeSize / 8 > 1)
    LastAccessedByte = IRB.CreateAdd(
        LastAccessedByte, ConstantInt::get(IntptrTy, TypeSize / 8 - 1));
  // (uint8_t) ((Addr & (Granularity-1)) + size - 1)
  LastAccessedByte = IRB.CreateIntCast(
      LastAccessedByte, ShadowValue->getType(), false);
  // ((uint8_t) ((Addr & (Granularity-1)) + size - 1)) >= ShadowValue
  return IRB.CreateICmpSGE(LastAccessedByte, ShadowValue);
}

void AddressSanitizer::instrumentAddress(Instruction *OrigIns,
                                         Instruction *InsertBefore,
                                         Value *Addr, uint32_t TypeSize,
                                         bool IsWrite, Value *SizeArgument) {
  IRBuilder<> IRB(InsertBefore);
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);

  Type *ShadowTy  = IntegerType::get(
      *C, std::max(8U, TypeSize >> Mapping.Scale));
  Type *ShadowPtrTy = PointerType::get(ShadowTy, 0);
  Value *ShadowPtr = memToShadow(AddrLong, IRB);
  Value *CmpVal = Constant::getNullValue(ShadowTy);
  Value *ShadowValue = IRB.CreateLoad(
      IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy));

  Value *Cmp = IRB.CreateICmpNE(ShadowValue, CmpVal);
  size_t AccessSizeIndex = TypeSizeToSizeIndex(TypeSize);
  size_t Granularity = 1 << Mapping.Scale;
  TerminatorInst *CrashTerm = 0;

  if (ClAlwaysSlowPath || (TypeSize < 8 * Granularity)) {
    TerminatorInst *CheckTerm =
        SplitBlockAndInsertIfThen(cast<Instruction>(Cmp), false);
    assert(dyn_cast<BranchInst>(CheckTerm)->isUnconditional());
    BasicBlock *NextBB = CheckTerm->getSuccessor(0);
    IRB.SetInsertPoint(CheckTerm);
    Value *Cmp2 = createSlowPathCmp(IRB, AddrLong, ShadowValue, TypeSize);
    BasicBlock *CrashBlock =
        BasicBlock::Create(*C, "", NextBB->getParent(), NextBB);
    CrashTerm = new UnreachableInst(*C, CrashBlock);
    BranchInst *NewTerm = BranchInst::Create(CrashBlock, NextBB, Cmp2);
    ReplaceInstWithInst(CheckTerm, NewTerm);
  } else {
    CrashTerm = SplitBlockAndInsertIfThen(cast<Instruction>(Cmp), true);
  }

  Instruction *Crash = generateCrashCode(
      CrashTerm, AddrLong, IsWrite, AccessSizeIndex, SizeArgument);
  Crash->setDebugLoc(OrigIns->getDebugLoc());
}

void AddressSanitizerModule::createInitializerPoisonCalls(
    Module &M, GlobalValue *ModuleName) {
  // We do all of our poisoning and unpoisoning within _GLOBAL__I_a.
  Function *GlobalInit = M.getFunction("_GLOBAL__I_a");
  // If that function is not present, this TU contains no globals, or they have
  // all been optimized away
  if (!GlobalInit)
    return;

  // Set up the arguments to our poison/unpoison functions.
  IRBuilder<> IRB(GlobalInit->begin()->getFirstInsertionPt());

  // Add a call to poison all external globals before the given function starts.
  Value *ModuleNameAddr = ConstantExpr::getPointerCast(ModuleName, IntptrTy);
  IRB.CreateCall(AsanPoisonGlobals, ModuleNameAddr);

  // Add calls to unpoison all globals before each return instruction.
  for (Function::iterator I = GlobalInit->begin(), E = GlobalInit->end();
      I != E; ++I) {
    if (ReturnInst *RI = dyn_cast<ReturnInst>(I->getTerminator())) {
      CallInst::Create(AsanUnpoisonGlobals, "", RI);
    }
  }
}

bool AddressSanitizerModule::ShouldInstrumentGlobal(GlobalVariable *G) {
  Type *Ty = cast<PointerType>(G->getType())->getElementType();
  DEBUG(dbgs() << "GLOBAL: " << *G << "\n");

  if (BL->isIn(*G)) return false;
  if (!Ty->isSized()) return false;
  if (!G->hasInitializer()) return false;
  if (GlobalWasGeneratedByAsan(G)) return false;  // Our own global.
  // Touch only those globals that will not be defined in other modules.
  // Don't handle ODR type linkages since other modules may be built w/o asan.
  if (G->getLinkage() != GlobalVariable::ExternalLinkage &&
      G->getLinkage() != GlobalVariable::PrivateLinkage &&
      G->getLinkage() != GlobalVariable::InternalLinkage)
    return false;
  // Two problems with thread-locals:
  //   - The address of the main thread's copy can't be computed at link-time.
  //   - Need to poison all copies, not just the main thread's one.
  if (G->isThreadLocal())
    return false;
  // For now, just ignore this Alloca if the alignment is large.
  if (G->getAlignment() > RedzoneSize()) return false;

  // Ignore all the globals with the names starting with "\01L_OBJC_".
  // Many of those are put into the .cstring section. The linker compresses
  // that section by removing the spare \0s after the string terminator, so
  // our redzones get broken.
  if ((G->getName().find("\01L_OBJC_") == 0) ||
      (G->getName().find("\01l_OBJC_") == 0)) {
    DEBUG(dbgs() << "Ignoring \\01L_OBJC_* global: " << *G);
    return false;
  }

  if (G->hasSection()) {
    StringRef Section(G->getSection());
    // Ignore the globals from the __OBJC section. The ObjC runtime assumes
    // those conform to /usr/lib/objc/runtime.h, so we can't add redzones to
    // them.
    if ((Section.find("__OBJC,") == 0) ||
        (Section.find("__DATA, __objc_") == 0)) {
      DEBUG(dbgs() << "Ignoring ObjC runtime global: " << *G);
      return false;
    }
    // See http://code.google.com/p/address-sanitizer/issues/detail?id=32
    // Constant CFString instances are compiled in the following way:
    //  -- the string buffer is emitted into
    //     __TEXT,__cstring,cstring_literals
    //  -- the constant NSConstantString structure referencing that buffer
    //     is placed into __DATA,__cfstring
    // Therefore there's no point in placing redzones into __DATA,__cfstring.
    // Moreover, it causes the linker to crash on OS X 10.7
    if (Section.find("__DATA,__cfstring") == 0) {
      DEBUG(dbgs() << "Ignoring CFString: " << *G);
      return false;
    }
  }

  return true;
}

void AddressSanitizerModule::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(*C);
  // Declare our poisoning and unpoisoning functions.
  AsanPoisonGlobals = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanPoisonGlobalsName, IRB.getVoidTy(), IntptrTy, NULL));
  AsanPoisonGlobals->setLinkage(Function::ExternalLinkage);
  AsanUnpoisonGlobals = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanUnpoisonGlobalsName, IRB.getVoidTy(), NULL));
  AsanUnpoisonGlobals->setLinkage(Function::ExternalLinkage);
  // Declare functions that register/unregister globals.
  AsanRegisterGlobals = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanRegisterGlobalsName, IRB.getVoidTy(),
      IntptrTy, IntptrTy, NULL));
  AsanRegisterGlobals->setLinkage(Function::ExternalLinkage);
  AsanUnregisterGlobals = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanUnregisterGlobalsName,
      IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
  AsanUnregisterGlobals->setLinkage(Function::ExternalLinkage);
}

// This function replaces all global variables with new variables that have
// trailing redzones. It also creates a function that poisons
// redzones and inserts this function into llvm.global_ctors.
bool AddressSanitizerModule::runOnModule(Module &M) {
  if (!ClGlobals) return false;
  TD = getAnalysisIfAvailable<DataLayout>();
  if (!TD)
    return false;
  BL.reset(new SpecialCaseList(BlacklistFile));
  if (BL->isIn(M)) return false;
  C = &(M.getContext());
  int LongSize = TD->getPointerSizeInBits();
  IntptrTy = Type::getIntNTy(*C, LongSize);
  Mapping = getShadowMapping(M, LongSize, ZeroBaseShadow);
  initializeCallbacks(M);
  DynamicallyInitializedGlobals.Init(M);

  SmallVector<GlobalVariable *, 16> GlobalsToChange;

  for (Module::GlobalListType::iterator G = M.global_begin(),
       E = M.global_end(); G != E; ++G) {
    if (ShouldInstrumentGlobal(G))
      GlobalsToChange.push_back(G);
  }

  size_t n = GlobalsToChange.size();
  if (n == 0) return false;

  // A global is described by a structure
  //   size_t beg;
  //   size_t size;
  //   size_t size_with_redzone;
  //   const char *name;
  //   const char *module_name;
  //   size_t has_dynamic_init;
  // We initialize an array of such structures and pass it to a run-time call.
  StructType *GlobalStructTy = StructType::get(IntptrTy, IntptrTy,
                                               IntptrTy, IntptrTy,
                                               IntptrTy, IntptrTy, NULL);
  SmallVector<Constant *, 16> Initializers(n), DynamicInit;


  Function *CtorFunc = M.getFunction(kAsanModuleCtorName);
  assert(CtorFunc);
  IRBuilder<> IRB(CtorFunc->getEntryBlock().getTerminator());

  bool HasDynamicallyInitializedGlobals = false;

  GlobalVariable *ModuleName = createPrivateGlobalForString(
      M, M.getModuleIdentifier());
  // We shouldn't merge same module names, as this string serves as unique
  // module ID in runtime.
  ModuleName->setUnnamedAddr(false);

  for (size_t i = 0; i < n; i++) {
    static const uint64_t kMaxGlobalRedzone = 1 << 18;
    GlobalVariable *G = GlobalsToChange[i];
    PointerType *PtrTy = cast<PointerType>(G->getType());
    Type *Ty = PtrTy->getElementType();
    uint64_t SizeInBytes = TD->getTypeAllocSize(Ty);
    uint64_t MinRZ = RedzoneSize();
    // MinRZ <= RZ <= kMaxGlobalRedzone
    // and trying to make RZ to be ~ 1/4 of SizeInBytes.
    uint64_t RZ = std::max(MinRZ,
                         std::min(kMaxGlobalRedzone,
                                  (SizeInBytes / MinRZ / 4) * MinRZ));
    uint64_t RightRedzoneSize = RZ;
    // Round up to MinRZ
    if (SizeInBytes % MinRZ)
      RightRedzoneSize += MinRZ - (SizeInBytes % MinRZ);
    assert(((RightRedzoneSize + SizeInBytes) % MinRZ) == 0);
    Type *RightRedZoneTy = ArrayType::get(IRB.getInt8Ty(), RightRedzoneSize);
    // Determine whether this global should be poisoned in initialization.
    bool GlobalHasDynamicInitializer =
        DynamicallyInitializedGlobals.Contains(G);
    // Don't check initialization order if this global is blacklisted.
    GlobalHasDynamicInitializer &= !BL->isIn(*G, "init");

    StructType *NewTy = StructType::get(Ty, RightRedZoneTy, NULL);
    Constant *NewInitializer = ConstantStruct::get(
        NewTy, G->getInitializer(),
        Constant::getNullValue(RightRedZoneTy), NULL);

    GlobalVariable *Name = createPrivateGlobalForString(M, G->getName());

    // Create a new global variable with enough space for a redzone.
    GlobalVariable *NewGlobal = new GlobalVariable(
        M, NewTy, G->isConstant(), G->getLinkage(),
        NewInitializer, "", G, G->getThreadLocalMode());
    NewGlobal->copyAttributesFrom(G);
    NewGlobal->setAlignment(MinRZ);

    Value *Indices2[2];
    Indices2[0] = IRB.getInt32(0);
    Indices2[1] = IRB.getInt32(0);

    G->replaceAllUsesWith(
        ConstantExpr::getGetElementPtr(NewGlobal, Indices2, true));
    NewGlobal->takeName(G);
    G->eraseFromParent();

    Initializers[i] = ConstantStruct::get(
        GlobalStructTy,
        ConstantExpr::getPointerCast(NewGlobal, IntptrTy),
        ConstantInt::get(IntptrTy, SizeInBytes),
        ConstantInt::get(IntptrTy, SizeInBytes + RightRedzoneSize),
        ConstantExpr::getPointerCast(Name, IntptrTy),
        ConstantExpr::getPointerCast(ModuleName, IntptrTy),
        ConstantInt::get(IntptrTy, GlobalHasDynamicInitializer),
        NULL);

    // Populate the first and last globals declared in this TU.
    if (CheckInitOrder && GlobalHasDynamicInitializer)
      HasDynamicallyInitializedGlobals = true;

    DEBUG(dbgs() << "NEW GLOBAL: " << *NewGlobal << "\n");
  }

  ArrayType *ArrayOfGlobalStructTy = ArrayType::get(GlobalStructTy, n);
  GlobalVariable *AllGlobals = new GlobalVariable(
      M, ArrayOfGlobalStructTy, false, GlobalVariable::PrivateLinkage,
      ConstantArray::get(ArrayOfGlobalStructTy, Initializers), "");

  // Create calls for poisoning before initializers run and unpoisoning after.
  if (CheckInitOrder && HasDynamicallyInitializedGlobals)
    createInitializerPoisonCalls(M, ModuleName);
  IRB.CreateCall2(AsanRegisterGlobals,
                  IRB.CreatePointerCast(AllGlobals, IntptrTy),
                  ConstantInt::get(IntptrTy, n));

  // We also need to unregister globals at the end, e.g. when a shared library
  // gets closed.
  Function *AsanDtorFunction = Function::Create(
      FunctionType::get(Type::getVoidTy(*C), false),
      GlobalValue::InternalLinkage, kAsanModuleDtorName, &M);
  BasicBlock *AsanDtorBB = BasicBlock::Create(*C, "", AsanDtorFunction);
  IRBuilder<> IRB_Dtor(ReturnInst::Create(*C, AsanDtorBB));
  IRB_Dtor.CreateCall2(AsanUnregisterGlobals,
                       IRB.CreatePointerCast(AllGlobals, IntptrTy),
                       ConstantInt::get(IntptrTy, n));
  appendToGlobalDtors(M, AsanDtorFunction, kAsanCtorAndCtorPriority);

  DEBUG(dbgs() << M);
  return true;
}

void AddressSanitizer::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(*C);
  // Create __asan_report* callbacks.
  for (size_t AccessIsWrite = 0; AccessIsWrite <= 1; AccessIsWrite++) {
    for (size_t AccessSizeIndex = 0; AccessSizeIndex < kNumberOfAccessSizes;
         AccessSizeIndex++) {
      // IsWrite and TypeSize are encoded in the function name.
      std::string FunctionName = std::string(kAsanReportErrorTemplate) +
          (AccessIsWrite ? "store" : "load") + itostr(1 << AccessSizeIndex);
      // If we are merging crash callbacks, they have two parameters.
      AsanErrorCallback[AccessIsWrite][AccessSizeIndex] =
          checkInterfaceFunction(M.getOrInsertFunction(
              FunctionName, IRB.getVoidTy(), IntptrTy, NULL));
    }
  }
  AsanErrorCallbackSized[0] = checkInterfaceFunction(M.getOrInsertFunction(
              kAsanReportLoadN, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
  AsanErrorCallbackSized[1] = checkInterfaceFunction(M.getOrInsertFunction(
              kAsanReportStoreN, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));

  AsanHandleNoReturnFunc = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanHandleNoReturnName, IRB.getVoidTy(), NULL));
  // We insert an empty inline asm after __asan_report* to avoid callback merge.
  EmptyAsm = InlineAsm::get(FunctionType::get(IRB.getVoidTy(), false),
                            StringRef(""), StringRef(""),
                            /*hasSideEffects=*/true);
}

void AddressSanitizer::emitShadowMapping(Module &M, IRBuilder<> &IRB) const {
  // Tell the values of mapping offset and scale to the run-time.
  GlobalValue *asan_mapping_offset =
      new GlobalVariable(M, IntptrTy, true, GlobalValue::LinkOnceODRLinkage,
                     ConstantInt::get(IntptrTy, Mapping.Offset),
                     kAsanMappingOffsetName);
  // Read the global, otherwise it may be optimized away.
  IRB.CreateLoad(asan_mapping_offset, true);

  GlobalValue *asan_mapping_scale =
      new GlobalVariable(M, IntptrTy, true, GlobalValue::LinkOnceODRLinkage,
                         ConstantInt::get(IntptrTy, Mapping.Scale),
                         kAsanMappingScaleName);
  // Read the global, otherwise it may be optimized away.
  IRB.CreateLoad(asan_mapping_scale, true);
}

// virtual
bool AddressSanitizer::doInitialization(Module &M) {
  // Initialize the private fields. No one has accessed them before.
  TD = getAnalysisIfAvailable<DataLayout>();

  if (!TD)
    return false;
  BL.reset(new SpecialCaseList(BlacklistFile));
  DynamicallyInitializedGlobals.Init(M);

  C = &(M.getContext());
  LongSize = TD->getPointerSizeInBits();
  IntptrTy = Type::getIntNTy(*C, LongSize);

  AsanCtorFunction = Function::Create(
      FunctionType::get(Type::getVoidTy(*C), false),
      GlobalValue::InternalLinkage, kAsanModuleCtorName, &M);
  BasicBlock *AsanCtorBB = BasicBlock::Create(*C, "", AsanCtorFunction);
  // call __asan_init in the module ctor.
  IRBuilder<> IRB(ReturnInst::Create(*C, AsanCtorBB));
  AsanInitFunction = checkInterfaceFunction(
      M.getOrInsertFunction(kAsanInitName, IRB.getVoidTy(), NULL));
  AsanInitFunction->setLinkage(Function::ExternalLinkage);
  IRB.CreateCall(AsanInitFunction);

  Mapping = getShadowMapping(M, LongSize, ZeroBaseShadow);
  emitShadowMapping(M, IRB);

  appendToGlobalCtors(M, AsanCtorFunction, kAsanCtorAndCtorPriority);
  return true;
}

bool AddressSanitizer::maybeInsertAsanInitAtFunctionEntry(Function &F) {
  // For each NSObject descendant having a +load method, this method is invoked
  // by the ObjC runtime before any of the static constructors is called.
  // Therefore we need to instrument such methods with a call to __asan_init
  // at the beginning in order to initialize our runtime before any access to
  // the shadow memory.
  // We cannot just ignore these methods, because they may call other
  // instrumented functions.
  if (F.getName().find(" load]") != std::string::npos) {
    IRBuilder<> IRB(F.begin()->begin());
    IRB.CreateCall(AsanInitFunction);
    return true;
  }
  return false;
}

bool AddressSanitizer::runOnFunction(Function &F) {
  if (BL->isIn(F)) return false;
  if (&F == AsanCtorFunction) return false;
  if (F.getLinkage() == GlobalValue::AvailableExternallyLinkage) return false;
  DEBUG(dbgs() << "ASAN instrumenting:\n" << F << "\n");
  initializeCallbacks(*F.getParent());

  // If needed, insert __asan_init before checking for SanitizeAddress attr.
  maybeInsertAsanInitAtFunctionEntry(F);

  if (!F.hasFnAttribute(Attribute::SanitizeAddress))
    return false;

  if (!ClDebugFunc.empty() && ClDebugFunc != F.getName())
    return false;

  // We want to instrument every address only once per basic block (unless there
  // are calls between uses).
  SmallSet<Value*, 16> TempsToInstrument;
  SmallVector<Instruction*, 16> ToInstrument;
  SmallVector<Instruction*, 8> NoReturnCalls;
  int NumAllocas = 0;
  bool IsWrite;

  // Fill the set of memory operations to instrument.
  for (Function::iterator FI = F.begin(), FE = F.end();
       FI != FE; ++FI) {
    TempsToInstrument.clear();
    int NumInsnsPerBB = 0;
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end();
         BI != BE; ++BI) {
      if (LooksLikeCodeInBug11395(BI)) return false;
      if (Value *Addr = isInterestingMemoryAccess(BI, &IsWrite)) {
        if (ClOpt && ClOptSameTemp) {
          if (!TempsToInstrument.insert(Addr))
            continue;  // We've seen this temp in the current BB.
        }
      } else if (isa<MemIntrinsic>(BI) && ClMemIntrin) {
        // ok, take it.
      } else {
        if (isa<AllocaInst>(BI))
          NumAllocas++;
        CallSite CS(BI);
        if (CS) {
          // A call inside BB.
          TempsToInstrument.clear();
          if (CS.doesNotReturn())
            NoReturnCalls.push_back(CS.getInstruction());
        }
        continue;
      }
      ToInstrument.push_back(BI);
      NumInsnsPerBB++;
      if (NumInsnsPerBB >= ClMaxInsnsToInstrumentPerBB)
        break;
    }
  }

  Function *UninstrumentedDuplicate = 0;
  bool LikelyToInstrument =
      !NoReturnCalls.empty() || !ToInstrument.empty() || (NumAllocas > 0);
  if (ClKeepUninstrumented && LikelyToInstrument) {
    ValueToValueMapTy VMap;
    UninstrumentedDuplicate = CloneFunction(&F, VMap, false);
    UninstrumentedDuplicate->removeFnAttr(Attribute::SanitizeAddress);
    UninstrumentedDuplicate->setName("NOASAN_" + F.getName());
    F.getParent()->getFunctionList().push_back(UninstrumentedDuplicate);
  }

  // Instrument.
  int NumInstrumented = 0;
  for (size_t i = 0, n = ToInstrument.size(); i != n; i++) {
    Instruction *Inst = ToInstrument[i];
    if (ClDebugMin < 0 || ClDebugMax < 0 ||
        (NumInstrumented >= ClDebugMin && NumInstrumented <= ClDebugMax)) {
      if (isInterestingMemoryAccess(Inst, &IsWrite))
        instrumentMop(Inst);
      else
        instrumentMemIntrinsic(cast<MemIntrinsic>(Inst));
    }
    NumInstrumented++;
  }

  FunctionStackPoisoner FSP(F, *this);
  bool ChangedStack = FSP.runOnFunction();

  // We must unpoison the stack before every NoReturn call (throw, _exit, etc).
  // See e.g. http://code.google.com/p/address-sanitizer/issues/detail?id=37
  for (size_t i = 0, n = NoReturnCalls.size(); i != n; i++) {
    Instruction *CI = NoReturnCalls[i];
    IRBuilder<> IRB(CI);
    IRB.CreateCall(AsanHandleNoReturnFunc);
  }

  bool res = NumInstrumented > 0 || ChangedStack || !NoReturnCalls.empty();
  DEBUG(dbgs() << "ASAN done instrumenting: " << res << " " << F << "\n");

  if (ClKeepUninstrumented) {
    if (!res) {
      // No instrumentation is done, no need for the duplicate.
      if (UninstrumentedDuplicate)
        UninstrumentedDuplicate->eraseFromParent();
    } else {
      // The function was instrumented. We must have the duplicate.
      assert(UninstrumentedDuplicate);
      UninstrumentedDuplicate->setSection("NOASAN");
      assert(!F.hasSection());
      F.setSection("ASAN");
    }
  }

  return res;
}

static uint64_t ValueForPoison(uint64_t PoisonByte, size_t ShadowRedzoneSize) {
  if (ShadowRedzoneSize == 1) return PoisonByte;
  if (ShadowRedzoneSize == 2) return (PoisonByte << 8) + PoisonByte;
  if (ShadowRedzoneSize == 4)
    return (PoisonByte << 24) + (PoisonByte << 16) +
        (PoisonByte << 8) + (PoisonByte);
  llvm_unreachable("ShadowRedzoneSize is either 1, 2 or 4");
}

static void PoisonShadowPartialRightRedzone(uint8_t *Shadow,
                                            size_t Size,
                                            size_t RZSize,
                                            size_t ShadowGranularity,
                                            uint8_t Magic) {
  for (size_t i = 0; i < RZSize;
       i+= ShadowGranularity, Shadow++) {
    if (i + ShadowGranularity <= Size) {
      *Shadow = 0;  // fully addressable
    } else if (i >= Size) {
      *Shadow = Magic;  // unaddressable
    } else {
      *Shadow = Size - i;  // first Size-i bytes are addressable
    }
  }
}

// Workaround for bug 11395: we don't want to instrument stack in functions
// with large assembly blobs (32-bit only), otherwise reg alloc may crash.
// FIXME: remove once the bug 11395 is fixed.
bool AddressSanitizer::LooksLikeCodeInBug11395(Instruction *I) {
  if (LongSize != 32) return false;
  CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI || !CI->isInlineAsm()) return false;
  if (CI->getNumArgOperands() <= 5) return false;
  // We have inline assembly with quite a few arguments.
  return true;
}

void FunctionStackPoisoner::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(*C);
  AsanStackMallocFunc = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanStackMallocName, IntptrTy, IntptrTy, IntptrTy, NULL));
  AsanStackFreeFunc = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanStackFreeName, IRB.getVoidTy(),
      IntptrTy, IntptrTy, IntptrTy, NULL));
  AsanPoisonStackMemoryFunc = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanPoisonStackMemoryName, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
  AsanUnpoisonStackMemoryFunc = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanUnpoisonStackMemoryName, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
}

void FunctionStackPoisoner::poisonRedZones(
  const ArrayRef<AllocaInst*> &AllocaVec, IRBuilder<> IRB, Value *ShadowBase,
  bool DoPoison) {
  size_t ShadowRZSize = RedzoneSize() >> Mapping.Scale;
  assert(ShadowRZSize >= 1 && ShadowRZSize <= 4);
  Type *RZTy = Type::getIntNTy(*C, ShadowRZSize * 8);
  Type *RZPtrTy = PointerType::get(RZTy, 0);

  Value *PoisonLeft  = ConstantInt::get(RZTy,
    ValueForPoison(DoPoison ? kAsanStackLeftRedzoneMagic : 0LL, ShadowRZSize));
  Value *PoisonMid   = ConstantInt::get(RZTy,
    ValueForPoison(DoPoison ? kAsanStackMidRedzoneMagic : 0LL, ShadowRZSize));
  Value *PoisonRight = ConstantInt::get(RZTy,
    ValueForPoison(DoPoison ? kAsanStackRightRedzoneMagic : 0LL, ShadowRZSize));

  // poison the first red zone.
  IRB.CreateStore(PoisonLeft, IRB.CreateIntToPtr(ShadowBase, RZPtrTy));

  // poison all other red zones.
  uint64_t Pos = RedzoneSize();
  for (size_t i = 0, n = AllocaVec.size(); i < n; i++) {
    AllocaInst *AI = AllocaVec[i];
    uint64_t SizeInBytes = getAllocaSizeInBytes(AI);
    uint64_t AlignedSize = getAlignedAllocaSize(AI);
    assert(AlignedSize - SizeInBytes < RedzoneSize());
    Value *Ptr = NULL;

    Pos += AlignedSize;

    assert(ShadowBase->getType() == IntptrTy);
    if (SizeInBytes < AlignedSize) {
      // Poison the partial redzone at right
      Ptr = IRB.CreateAdd(
          ShadowBase, ConstantInt::get(IntptrTy,
                                       (Pos >> Mapping.Scale) - ShadowRZSize));
      size_t AddressableBytes = RedzoneSize() - (AlignedSize - SizeInBytes);
      uint32_t Poison = 0;
      if (DoPoison) {
        PoisonShadowPartialRightRedzone((uint8_t*)&Poison, AddressableBytes,
                                        RedzoneSize(),
                                        1ULL << Mapping.Scale,
                                        kAsanStackPartialRedzoneMagic);
        Poison =
            ASan.TD->isLittleEndian()
                ? support::endian::byte_swap<uint32_t, support::little>(Poison)
                : support::endian::byte_swap<uint32_t, support::big>(Poison);
      }
      Value *PartialPoison = ConstantInt::get(RZTy, Poison);
      IRB.CreateStore(PartialPoison, IRB.CreateIntToPtr(Ptr, RZPtrTy));
    }

    // Poison the full redzone at right.
    Ptr = IRB.CreateAdd(ShadowBase,
                        ConstantInt::get(IntptrTy, Pos >> Mapping.Scale));
    bool LastAlloca = (i == AllocaVec.size() - 1);
    Value *Poison = LastAlloca ? PoisonRight : PoisonMid;
    IRB.CreateStore(Poison, IRB.CreateIntToPtr(Ptr, RZPtrTy));

    Pos += RedzoneSize();
  }
}

void FunctionStackPoisoner::poisonStack() {
  uint64_t LocalStackSize = TotalStackSize +
                            (AllocaVec.size() + 1) * RedzoneSize();

  bool DoStackMalloc = ASan.CheckUseAfterReturn
      && LocalStackSize <= kMaxStackMallocSize;

  assert(AllocaVec.size() > 0);
  Instruction *InsBefore = AllocaVec[0];
  IRBuilder<> IRB(InsBefore);


  Type *ByteArrayTy = ArrayType::get(IRB.getInt8Ty(), LocalStackSize);
  AllocaInst *MyAlloca =
      new AllocaInst(ByteArrayTy, "MyAlloca", InsBefore);
  if (ClRealignStack && StackAlignment < RedzoneSize())
    StackAlignment = RedzoneSize();
  MyAlloca->setAlignment(StackAlignment);
  assert(MyAlloca->isStaticAlloca());
  Value *OrigStackBase = IRB.CreatePointerCast(MyAlloca, IntptrTy);
  Value *LocalStackBase = OrigStackBase;

  if (DoStackMalloc) {
    LocalStackBase = IRB.CreateCall2(AsanStackMallocFunc,
        ConstantInt::get(IntptrTy, LocalStackSize), OrigStackBase);
  }

  // This string will be parsed by the run-time (DescribeAddressIfStack).
  SmallString<2048> StackDescriptionStorage;
  raw_svector_ostream StackDescription(StackDescriptionStorage);
  StackDescription << AllocaVec.size() << " ";

  // Insert poison calls for lifetime intrinsics for alloca.
  bool HavePoisonedAllocas = false;
  for (size_t i = 0, n = AllocaPoisonCallVec.size(); i < n; i++) {
    const AllocaPoisonCall &APC = AllocaPoisonCallVec[i];
    IntrinsicInst *II = APC.InsBefore;
    AllocaInst *AI = findAllocaForValue(II->getArgOperand(1));
    assert(AI);
    IRBuilder<> IRB(II);
    poisonAlloca(AI, APC.Size, IRB, APC.DoPoison);
    HavePoisonedAllocas |= APC.DoPoison;
  }

  uint64_t Pos = RedzoneSize();
  // Replace Alloca instructions with base+offset.
  for (size_t i = 0, n = AllocaVec.size(); i < n; i++) {
    AllocaInst *AI = AllocaVec[i];
    uint64_t SizeInBytes = getAllocaSizeInBytes(AI);
    StringRef Name = AI->getName();
    StackDescription << Pos << " " << SizeInBytes << " "
                     << Name.size() << " " << Name << " ";
    uint64_t AlignedSize = getAlignedAllocaSize(AI);
    assert((AlignedSize % RedzoneSize()) == 0);
    Value *NewAllocaPtr = IRB.CreateIntToPtr(
            IRB.CreateAdd(LocalStackBase, ConstantInt::get(IntptrTy, Pos)),
            AI->getType());
    replaceDbgDeclareForAlloca(AI, NewAllocaPtr, DIB);
    AI->replaceAllUsesWith(NewAllocaPtr);
    Pos += AlignedSize + RedzoneSize();
  }
  assert(Pos == LocalStackSize);

  // The left-most redzone has enough space for at least 4 pointers.
  // Write the Magic value to redzone[0].
  Value *BasePlus0 = IRB.CreateIntToPtr(LocalStackBase, IntptrPtrTy);
  IRB.CreateStore(ConstantInt::get(IntptrTy, kCurrentStackFrameMagic),
                  BasePlus0);
  // Write the frame description constant to redzone[1].
  Value *BasePlus1 = IRB.CreateIntToPtr(
    IRB.CreateAdd(LocalStackBase, ConstantInt::get(IntptrTy, ASan.LongSize/8)),
    IntptrPtrTy);
  GlobalVariable *StackDescriptionGlobal =
      createPrivateGlobalForString(*F.getParent(), StackDescription.str());
  Value *Description = IRB.CreatePointerCast(StackDescriptionGlobal,
                                             IntptrTy);
  IRB.CreateStore(Description, BasePlus1);
  // Write the PC to redzone[2].
  Value *BasePlus2 = IRB.CreateIntToPtr(
    IRB.CreateAdd(LocalStackBase, ConstantInt::get(IntptrTy,
                                                   2 * ASan.LongSize/8)),
    IntptrPtrTy);
  IRB.CreateStore(IRB.CreatePointerCast(&F, IntptrTy), BasePlus2);

  // Poison the stack redzones at the entry.
  Value *ShadowBase = ASan.memToShadow(LocalStackBase, IRB);
  poisonRedZones(AllocaVec, IRB, ShadowBase, true);

  // Unpoison the stack before all ret instructions.
  for (size_t i = 0, n = RetVec.size(); i < n; i++) {
    Instruction *Ret = RetVec[i];
    IRBuilder<> IRBRet(Ret);
    // Mark the current frame as retired.
    IRBRet.CreateStore(ConstantInt::get(IntptrTy, kRetiredStackFrameMagic),
                       BasePlus0);
    // Unpoison the stack.
    poisonRedZones(AllocaVec, IRBRet, ShadowBase, false);
    if (DoStackMalloc) {
      // In use-after-return mode, mark the whole stack frame unaddressable.
      IRBRet.CreateCall3(AsanStackFreeFunc, LocalStackBase,
                         ConstantInt::get(IntptrTy, LocalStackSize),
                         OrigStackBase);
    } else if (HavePoisonedAllocas) {
      // If we poisoned some allocas in llvm.lifetime analysis,
      // unpoison whole stack frame now.
      assert(LocalStackBase == OrigStackBase);
      poisonAlloca(LocalStackBase, LocalStackSize, IRBRet, false);
    }
  }

  // We are done. Remove the old unused alloca instructions.
  for (size_t i = 0, n = AllocaVec.size(); i < n; i++)
    AllocaVec[i]->eraseFromParent();
}

void FunctionStackPoisoner::poisonAlloca(Value *V, uint64_t Size,
                                         IRBuilder<> IRB, bool DoPoison) {
  // For now just insert the call to ASan runtime.
  Value *AddrArg = IRB.CreatePointerCast(V, IntptrTy);
  Value *SizeArg = ConstantInt::get(IntptrTy, Size);
  IRB.CreateCall2(DoPoison ? AsanPoisonStackMemoryFunc
                           : AsanUnpoisonStackMemoryFunc,
                  AddrArg, SizeArg);
}

// Handling llvm.lifetime intrinsics for a given %alloca:
// (1) collect all llvm.lifetime.xxx(%size, %value) describing the alloca.
// (2) if %size is constant, poison memory for llvm.lifetime.end (to detect
//     invalid accesses) and unpoison it for llvm.lifetime.start (the memory
//     could be poisoned by previous llvm.lifetime.end instruction, as the
//     variable may go in and out of scope several times, e.g. in loops).
// (3) if we poisoned at least one %alloca in a function,
//     unpoison the whole stack frame at function exit.

AllocaInst *FunctionStackPoisoner::findAllocaForValue(Value *V) {
  if (AllocaInst *AI = dyn_cast<AllocaInst>(V))
    // We're intested only in allocas we can handle.
    return isInterestingAlloca(*AI) ? AI : 0;
  // See if we've already calculated (or started to calculate) alloca for a
  // given value.
  AllocaForValueMapTy::iterator I = AllocaForValue.find(V);
  if (I != AllocaForValue.end())
    return I->second;
  // Store 0 while we're calculating alloca for value V to avoid
  // infinite recursion if the value references itself.
  AllocaForValue[V] = 0;
  AllocaInst *Res = 0;
  if (CastInst *CI = dyn_cast<CastInst>(V))
    Res = findAllocaForValue(CI->getOperand(0));
  else if (PHINode *PN = dyn_cast<PHINode>(V)) {
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      Value *IncValue = PN->getIncomingValue(i);
      // Allow self-referencing phi-nodes.
      if (IncValue == PN) continue;
      AllocaInst *IncValueAI = findAllocaForValue(IncValue);
      // AI for incoming values should exist and should all be equal.
      if (IncValueAI == 0 || (Res != 0 && IncValueAI != Res))
        return 0;
      Res = IncValueAI;
    }
  }
  if (Res != 0)
    AllocaForValue[V] = Res;
  return Res;
}
