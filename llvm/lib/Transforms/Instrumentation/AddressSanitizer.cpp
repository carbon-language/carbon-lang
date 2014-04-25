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

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/system_error.h"
#include "llvm/Transforms/Utils/ASanStackFrameLayout.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/SpecialCaseList.h"
#include <algorithm>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "asan"

static const uint64_t kDefaultShadowScale = 3;
static const uint64_t kDefaultShadowOffset32 = 1ULL << 29;
static const uint64_t kIOSShadowOffset32 = 1ULL << 30;
static const uint64_t kDefaultShadowOffset64 = 1ULL << 44;
static const uint64_t kSmallX86_64ShadowOffset = 0x7FFF8000;  // < 2G.
static const uint64_t kPPC64_ShadowOffset64 = 1ULL << 41;
static const uint64_t kMIPS32_ShadowOffset32 = 0x0aaa8000;
static const uint64_t kFreeBSD_ShadowOffset32 = 1ULL << 30;
static const uint64_t kFreeBSD_ShadowOffset64 = 1ULL << 46;

static const size_t kMinStackMallocSize = 1 << 6;  // 64B
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
static const char *const kAsanUnregisterGlobalsName =
    "__asan_unregister_globals";
static const char *const kAsanPoisonGlobalsName = "__asan_before_dynamic_init";
static const char *const kAsanUnpoisonGlobalsName = "__asan_after_dynamic_init";
static const char *const kAsanInitName = "__asan_init_v3";
static const char *const kAsanCovName = "__sanitizer_cov";
static const char *const kAsanPtrCmp = "__sanitizer_ptr_cmp";
static const char *const kAsanPtrSub = "__sanitizer_ptr_sub";
static const char *const kAsanHandleNoReturnName = "__asan_handle_no_return";
static const int         kMaxAsanStackMallocSizeClass = 10;
static const char *const kAsanStackMallocNameTemplate = "__asan_stack_malloc_";
static const char *const kAsanStackFreeNameTemplate = "__asan_stack_free_";
static const char *const kAsanGenPrefix = "__asan_gen_";
static const char *const kAsanPoisonStackMemoryName =
    "__asan_poison_stack_memory";
static const char *const kAsanUnpoisonStackMemoryName =
    "__asan_unpoison_stack_memory";

static const char *const kAsanOptionDetectUAR =
    "__asan_option_detect_stack_use_after_return";

#ifndef NDEBUG
static const int kAsanStackAfterReturnMagic = 0xf5;
#endif

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
static cl::opt<int> ClCoverage("asan-coverage",
       cl::desc("ASan coverage. 0: none, 1: entry block, 2: all blocks"),
       cl::Hidden, cl::init(false));
static cl::opt<int> ClCoverageBlockThreshold("asan-coverage-block-threshold",
       cl::desc("Add coverage instrumentation only to the entry block if there "
                "are more than this number of blocks."),
       cl::Hidden, cl::init(1500));
static cl::opt<bool> ClInitializers("asan-initialization-order",
       cl::desc("Handle C++ initializer order"), cl::Hidden, cl::init(false));
static cl::opt<bool> ClInvalidPointerPairs("asan-detect-invalid-pointer-pair",
       cl::desc("Instrument <, <=, >, >=, - with pointer operands"),
       cl::Hidden, cl::init(false));
static cl::opt<unsigned> ClRealignStack("asan-realign-stack",
       cl::desc("Realign stack to the value of this flag (power of two)"),
       cl::Hidden, cl::init(32));
static cl::opt<std::string> ClBlacklistFile("asan-blacklist",
       cl::desc("File containing the list of objects to ignore "
                "during instrumentation"), cl::Hidden);
static cl::opt<int> ClInstrumentationWithCallsThreshold(
    "asan-instrumentation-with-call-threshold",
       cl::desc("If the function being instrumented contains more than "
                "this number of memory accesses, use callbacks instead of "
                "inline checks (-1 means never use callbacks)."),
       cl::Hidden, cl::init(10000));
static cl::opt<std::string> ClMemoryAccessCallbackPrefix(
       "asan-memory-access-callback-prefix",
       cl::desc("Prefix for memory access callbacks"), cl::Hidden,
       cl::init("__asan_"));

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

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumOptimizedAccessesToGlobalArray,
          "Number of optimized accesses to global arrays");
STATISTIC(NumOptimizedAccessesToGlobalVar,
          "Number of optimized accesses to global vars");

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

static ShadowMapping getShadowMapping(const Module &M, int LongSize) {
  llvm::Triple TargetTriple(M.getTargetTriple());
  bool IsAndroid = TargetTriple.getEnvironment() == llvm::Triple::Android;
  bool IsIOS = TargetTriple.getOS() == llvm::Triple::IOS;
  bool IsFreeBSD = TargetTriple.getOS() == llvm::Triple::FreeBSD;
  bool IsLinux = TargetTriple.getOS() == llvm::Triple::Linux;
  bool IsPPC64 = TargetTriple.getArch() == llvm::Triple::ppc64 ||
                 TargetTriple.getArch() == llvm::Triple::ppc64le;
  bool IsX86_64 = TargetTriple.getArch() == llvm::Triple::x86_64;
  bool IsMIPS32 = TargetTriple.getArch() == llvm::Triple::mips ||
                  TargetTriple.getArch() == llvm::Triple::mipsel;

  ShadowMapping Mapping;

  if (LongSize == 32) {
    if (IsAndroid)
      Mapping.Offset = 0;
    else if (IsMIPS32)
      Mapping.Offset = kMIPS32_ShadowOffset32;
    else if (IsFreeBSD)
      Mapping.Offset = kFreeBSD_ShadowOffset32;
    else if (IsIOS)
      Mapping.Offset = kIOSShadowOffset32;
    else
      Mapping.Offset = kDefaultShadowOffset32;
  } else {  // LongSize == 64
    if (IsPPC64)
      Mapping.Offset = kPPC64_ShadowOffset64;
    else if (IsFreeBSD)
      Mapping.Offset = kFreeBSD_ShadowOffset64;
    else if (IsLinux && IsX86_64)
      Mapping.Offset = kSmallX86_64ShadowOffset;
    else
      Mapping.Offset = kDefaultShadowOffset64;
  }

  Mapping.Scale = kDefaultShadowScale;
  if (ClMappingScale) {
    Mapping.Scale = ClMappingScale;
  }

  // OR-ing shadow offset if more efficient (at least on x86) if the offset
  // is a power of two, but on ppc64 we have to use add since the shadow
  // offset is not necessary 1/8-th of the address space.
  Mapping.OrShadowOffset = !IsPPC64 && !(Mapping.Offset & (Mapping.Offset - 1));

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
                   StringRef BlacklistFile = StringRef())
      : FunctionPass(ID),
        CheckInitOrder(CheckInitOrder || ClInitializers),
        CheckUseAfterReturn(CheckUseAfterReturn || ClUseAfterReturn),
        CheckLifetime(CheckLifetime || ClCheckLifetime),
        BlacklistFile(BlacklistFile.empty() ? ClBlacklistFile
                                            : BlacklistFile) {}
  const char *getPassName() const override {
    return "AddressSanitizerFunctionPass";
  }
  void instrumentMop(Instruction *I, bool UseCalls);
  void instrumentPointerComparisonOrSubtraction(Instruction *I);
  void instrumentAddress(Instruction *OrigIns, Instruction *InsertBefore,
                         Value *Addr, uint32_t TypeSize, bool IsWrite,
                         Value *SizeArgument, bool UseCalls);
  Value *createSlowPathCmp(IRBuilder<> &IRB, Value *AddrLong,
                           Value *ShadowValue, uint32_t TypeSize);
  Instruction *generateCrashCode(Instruction *InsertBefore, Value *Addr,
                                 bool IsWrite, size_t AccessSizeIndex,
                                 Value *SizeArgument);
  void instrumentMemIntrinsic(MemIntrinsic *MI);
  Value *memToShadow(Value *Shadow, IRBuilder<> &IRB);
  bool runOnFunction(Function &F) override;
  bool maybeInsertAsanInitAtFunctionEntry(Function &F);
  bool doInitialization(Module &M) override;
  static char ID;  // Pass identification, replacement for typeid

 private:
  void initializeCallbacks(Module &M);

  bool LooksLikeCodeInBug11395(Instruction *I);
  bool GlobalIsLinkerInitialized(GlobalVariable *G);
  bool InjectCoverage(Function &F, const ArrayRef<BasicBlock*> AllBlocks);
  void InjectCoverageAtBlock(Function &F, BasicBlock &BB);

  bool CheckInitOrder;
  bool CheckUseAfterReturn;
  bool CheckLifetime;
  SmallString<64> BlacklistFile;

  LLVMContext *C;
  const DataLayout *DL;
  int LongSize;
  Type *IntptrTy;
  ShadowMapping Mapping;
  Function *AsanCtorFunction;
  Function *AsanInitFunction;
  Function *AsanHandleNoReturnFunc;
  Function *AsanCovFunction;
  Function *AsanPtrCmpFunction, *AsanPtrSubFunction;
  std::unique_ptr<SpecialCaseList> BL;
  // This array is indexed by AccessIsWrite and log2(AccessSize).
  Function *AsanErrorCallback[2][kNumberOfAccessSizes];
  Function *AsanMemoryAccessCallback[2][kNumberOfAccessSizes];
  // This array is indexed by AccessIsWrite.
  Function *AsanErrorCallbackSized[2],
           *AsanMemoryAccessCallbackSized[2];
  Function *AsanMemmove, *AsanMemcpy, *AsanMemset;
  InlineAsm *EmptyAsm;
  SetOfDynamicallyInitializedGlobals DynamicallyInitializedGlobals;

  friend struct FunctionStackPoisoner;
};

class AddressSanitizerModule : public ModulePass {
 public:
  AddressSanitizerModule(bool CheckInitOrder = true,
                         StringRef BlacklistFile = StringRef())
      : ModulePass(ID),
        CheckInitOrder(CheckInitOrder || ClInitializers),
        BlacklistFile(BlacklistFile.empty() ? ClBlacklistFile
                                            : BlacklistFile) {}
  bool runOnModule(Module &M) override;
  static char ID;  // Pass identification, replacement for typeid
  const char *getPassName() const override {
    return "AddressSanitizerModule";
  }

 private:
  void initializeCallbacks(Module &M);

  bool ShouldInstrumentGlobal(GlobalVariable *G);
  void createInitializerPoisonCalls(Module &M, GlobalValue *ModuleName);
  size_t MinRedzoneSizeForGlobal() const {
    return RedzoneSizeForScale(Mapping.Scale);
  }

  bool CheckInitOrder;
  SmallString<64> BlacklistFile;

  std::unique_ptr<SpecialCaseList> BL;
  SetOfDynamicallyInitializedGlobals DynamicallyInitializedGlobals;
  Type *IntptrTy;
  LLVMContext *C;
  const DataLayout *DL;
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
  unsigned StackAlignment;

  Function *AsanStackMallocFunc[kMaxAsanStackMallocSizeClass + 1],
           *AsanStackFreeFunc[kMaxAsanStackMallocSizeClass + 1];
  Function *AsanPoisonStackMemoryFunc, *AsanUnpoisonStackMemoryFunc;

  // Stores a place and arguments of poisoning/unpoisoning call for alloca.
  struct AllocaPoisonCall {
    IntrinsicInst *InsBefore;
    AllocaInst *AI;
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
        StackAlignment(1 << Mapping.Scale) {}

  bool runOnFunction() {
    if (!ClStack) return false;
    // Collect alloca, ret, lifetime instructions etc.
    for (BasicBlock *BB : depth_first(&F.getEntryBlock()))
      visit(*BB);

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
    AllocaPoisonCall APC = {&II, AI, SizeValue, DoPoison};
    AllocaPoisonCallVec.push_back(APC);
  }

  // ---------------------- Helpers.
  void initializeCallbacks(Module &M);

  // Check if we want (and can) handle this alloca.
  bool isInterestingAlloca(AllocaInst &AI) const {
    return (!AI.isArrayAllocation() && AI.isStaticAlloca() &&
            AI.getAllocatedType()->isSized() &&
            // alloca() may be called with 0 size, ignore it.
            getAllocaSizeInBytes(&AI) > 0);
  }

  uint64_t getAllocaSizeInBytes(AllocaInst *AI) const {
    Type *Ty = AI->getAllocatedType();
    uint64_t SizeInBytes = ASan.DL->getTypeAllocSize(Ty);
    return SizeInBytes;
  }
  /// Finds alloca where the value comes from.
  AllocaInst *findAllocaForValue(Value *V);
  void poisonRedZones(const ArrayRef<uint8_t> ShadowBytes, IRBuilder<> &IRB,
                      Value *ShadowBase, bool DoPoison);
  void poisonAlloca(Value *V, uint64_t Size, IRBuilder<> &IRB, bool DoPoison);

  void SetShadowToStackAfterReturnInlined(IRBuilder<> &IRB, Value *ShadowBase,
                                          int Size);
};

}  // namespace

char AddressSanitizer::ID = 0;
INITIALIZE_PASS(AddressSanitizer, "asan",
    "AddressSanitizer: detects use-after-free and out-of-bounds bugs.",
    false, false)
FunctionPass *llvm::createAddressSanitizerFunctionPass(
    bool CheckInitOrder, bool CheckUseAfterReturn, bool CheckLifetime,
    StringRef BlacklistFile) {
  return new AddressSanitizer(CheckInitOrder, CheckUseAfterReturn,
                              CheckLifetime, BlacklistFile);
}

char AddressSanitizerModule::ID = 0;
INITIALIZE_PASS(AddressSanitizerModule, "asan-module",
    "AddressSanitizer: detects use-after-free and out-of-bounds bugs."
    "ModulePass", false, false)
ModulePass *llvm::createAddressSanitizerModulePass(
    bool CheckInitOrder, StringRef BlacklistFile) {
  return new AddressSanitizerModule(CheckInitOrder, BlacklistFile);
}

static size_t TypeSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = countTrailingZeros(TypeSize / 8);
  assert(Res < kNumberOfAccessSizes);
  return Res;
}

// \brief Create a constant for Str so that we can pass it to the run-time lib.
static GlobalVariable *createPrivateGlobalForString(
    Module &M, StringRef Str, bool AllowMerging) {
  Constant *StrConst = ConstantDataArray::getString(M.getContext(), Str);
  // We use private linkage for module-local strings. If they can be merged
  // with another one, we set the unnamed_addr attribute.
  GlobalVariable *GV =
      new GlobalVariable(M, StrConst->getType(), true,
                         GlobalValue::PrivateLinkage, StrConst, kAsanGenPrefix);
  if (AllowMerging)
    GV->setUnnamedAddr(true);
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

// Instrument memset/memmove/memcpy
void AddressSanitizer::instrumentMemIntrinsic(MemIntrinsic *MI) {
  IRBuilder<> IRB(MI);
  Instruction *Call = nullptr;
  if (isa<MemTransferInst>(MI)) {
    Call = IRB.CreateCall3(
        isa<MemMoveInst>(MI) ? AsanMemmove : AsanMemcpy,
        IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
        IRB.CreatePointerCast(MI->getOperand(1), IRB.getInt8PtrTy()),
        IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false));
  } else if (isa<MemSetInst>(MI)) {
    Call = IRB.CreateCall3(
        AsanMemset,
        IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
        IRB.CreateIntCast(MI->getOperand(1), IRB.getInt32Ty(), false),
        IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false));
  }
  Call->setDebugLoc(MI->getDebugLoc());
  MI->eraseFromParent();
}

// If I is an interesting memory access, return the PointerOperand
// and set IsWrite. Otherwise return NULL.
static Value *isInterestingMemoryAccess(Instruction *I, bool *IsWrite) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (!ClInstrumentReads) return nullptr;
    *IsWrite = false;
    return LI->getPointerOperand();
  }
  if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (!ClInstrumentWrites) return nullptr;
    *IsWrite = true;
    return SI->getPointerOperand();
  }
  if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    if (!ClInstrumentAtomics) return nullptr;
    *IsWrite = true;
    return RMW->getPointerOperand();
  }
  if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    if (!ClInstrumentAtomics) return nullptr;
    *IsWrite = true;
    return XCHG->getPointerOperand();
  }
  return nullptr;
}

static bool isPointerOperand(Value *V) {
  return V->getType()->isPointerTy() || isa<PtrToIntInst>(V);
}

// This is a rough heuristic; it may cause both false positives and
// false negatives. The proper implementation requires cooperation with
// the frontend.
static bool isInterestingPointerComparisonOrSubtraction(Instruction *I) {
  if (ICmpInst *Cmp = dyn_cast<ICmpInst>(I)) {
    if (!Cmp->isRelational())
      return false;
  } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
    if (BO->getOpcode() != Instruction::Sub)
      return false;
  } else {
    return false;
  }
  if (!isPointerOperand(I->getOperand(0)) ||
      !isPointerOperand(I->getOperand(1)))
      return false;
  return true;
}

bool AddressSanitizer::GlobalIsLinkerInitialized(GlobalVariable *G) {
  // If a global variable does not have dynamic initialization we don't
  // have to instrument it.  However, if a global does not have initializer
  // at all, we assume it has dynamic initializer (in other TU).
  return G->hasInitializer() && !DynamicallyInitializedGlobals.Contains(G);
}

void
AddressSanitizer::instrumentPointerComparisonOrSubtraction(Instruction *I) {
  IRBuilder<> IRB(I);
  Function *F = isa<ICmpInst>(I) ? AsanPtrCmpFunction : AsanPtrSubFunction;
  Value *Param[2] = {I->getOperand(0), I->getOperand(1)};
  for (int i = 0; i < 2; i++) {
    if (Param[i]->getType()->isPointerTy())
      Param[i] = IRB.CreatePointerCast(Param[i], IntptrTy);
  }
  IRB.CreateCall2(F, Param[0], Param[1]);
}

void AddressSanitizer::instrumentMop(Instruction *I, bool UseCalls) {
  bool IsWrite = false;
  Value *Addr = isInterestingMemoryAccess(I, &IsWrite);
  assert(Addr);
  if (ClOpt && ClOptGlobals) {
    if (GlobalVariable *G = dyn_cast<GlobalVariable>(Addr)) {
      // If initialization order checking is disabled, a simple access to a
      // dynamically initialized global is always valid.
      if (!CheckInitOrder || GlobalIsLinkerInitialized(G)) {
        NumOptimizedAccessesToGlobalVar++;
        return;
      }
    }
    ConstantExpr *CE = dyn_cast<ConstantExpr>(Addr);
    if (CE && CE->isGEPWithNoNotionalOverIndexing()) {
      if (GlobalVariable *G = dyn_cast<GlobalVariable>(CE->getOperand(0))) {
        if (CE->getOperand(1)->isNullValue() && GlobalIsLinkerInitialized(G)) {
          NumOptimizedAccessesToGlobalArray++;
          return;
        }
      }
    }
  }

  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();

  assert(OrigTy->isSized());
  uint32_t TypeSize = DL->getTypeStoreSizeInBits(OrigTy);

  assert((TypeSize % 8) == 0);

  if (IsWrite)
    NumInstrumentedWrites++;
  else
    NumInstrumentedReads++;

  // Instrument a 1-, 2-, 4-, 8-, or 16- byte access with one check.
  if (TypeSize == 8  || TypeSize == 16 ||
      TypeSize == 32 || TypeSize == 64 || TypeSize == 128)
    return instrumentAddress(I, I, Addr, TypeSize, IsWrite, nullptr, UseCalls);
  // Instrument unusual size (but still multiple of 8).
  // We can not do it with a single check, so we do 1-byte check for the first
  // and the last bytes. We call __asan_report_*_n(addr, real_size) to be able
  // to report the actual access size.
  IRBuilder<> IRB(I);
  Value *Size = ConstantInt::get(IntptrTy, TypeSize / 8);
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  if (UseCalls) {
    CallInst *Check =
        IRB.CreateCall2(AsanMemoryAccessCallbackSized[IsWrite], AddrLong, Size);
    Check->setDebugLoc(I->getDebugLoc());
  } else {
    Value *LastByte = IRB.CreateIntToPtr(
        IRB.CreateAdd(AddrLong, ConstantInt::get(IntptrTy, TypeSize / 8 - 1)),
        OrigPtrTy);
    instrumentAddress(I, I, Addr, 8, IsWrite, Size, false);
    instrumentAddress(I, I, LastByte, 8, IsWrite, Size, false);
  }
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
                                         Instruction *InsertBefore, Value *Addr,
                                         uint32_t TypeSize, bool IsWrite,
                                         Value *SizeArgument, bool UseCalls) {
  IRBuilder<> IRB(InsertBefore);
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  size_t AccessSizeIndex = TypeSizeToSizeIndex(TypeSize);

  if (UseCalls) {
    IRB.CreateCall(AsanMemoryAccessCallback[IsWrite][AccessSizeIndex],
                   AddrLong);
    return;
  }

  Type *ShadowTy  = IntegerType::get(
      *C, std::max(8U, TypeSize >> Mapping.Scale));
  Type *ShadowPtrTy = PointerType::get(ShadowTy, 0);
  Value *ShadowPtr = memToShadow(AddrLong, IRB);
  Value *CmpVal = Constant::getNullValue(ShadowTy);
  Value *ShadowValue = IRB.CreateLoad(
      IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy));

  Value *Cmp = IRB.CreateICmpNE(ShadowValue, CmpVal);
  size_t Granularity = 1 << Mapping.Scale;
  TerminatorInst *CrashTerm = nullptr;

  if (ClAlwaysSlowPath || (TypeSize < 8 * Granularity)) {
    TerminatorInst *CheckTerm =
        SplitBlockAndInsertIfThen(Cmp, InsertBefore, false);
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
    CrashTerm = SplitBlockAndInsertIfThen(Cmp, InsertBefore, true);
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
  // For now, just ignore this Global if the alignment is large.
  if (G->getAlignment() > MinRedzoneSizeForGlobal()) return false;

  // Ignore all the globals with the names starting with "\01L_OBJC_".
  // Many of those are put into the .cstring section. The linker compresses
  // that section by removing the spare \0s after the string terminator, so
  // our redzones get broken.
  if ((G->getName().find("\01L_OBJC_") == 0) ||
      (G->getName().find("\01l_OBJC_") == 0)) {
    DEBUG(dbgs() << "Ignoring \\01L_OBJC_* global: " << *G << "\n");
    return false;
  }

  if (G->hasSection()) {
    StringRef Section(G->getSection());
    // Ignore the globals from the __OBJC section. The ObjC runtime assumes
    // those conform to /usr/lib/objc/runtime.h, so we can't add redzones to
    // them.
    if ((Section.find("__OBJC,") == 0) ||
        (Section.find("__DATA, __objc_") == 0)) {
      DEBUG(dbgs() << "Ignoring ObjC runtime global: " << *G << "\n");
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
      DEBUG(dbgs() << "Ignoring CFString: " << *G << "\n");
      return false;
    }
    // The linker merges the contents of cstring_literals and removes the
    // trailing zeroes.
    if (Section.find("__TEXT,__cstring,cstring_literals") == 0) {
      DEBUG(dbgs() << "Ignoring a cstring literal: " << *G << "\n");
      return false;
    }
    // Globals from llvm.metadata aren't emitted, do not instrument them.
    if (Section == "llvm.metadata") return false;
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

  DataLayoutPass *DLP = getAnalysisIfAvailable<DataLayoutPass>();
  if (!DLP)
    return false;
  DL = &DLP->getDataLayout();

  BL.reset(SpecialCaseList::createOrDie(BlacklistFile));
  if (BL->isIn(M)) return false;
  C = &(M.getContext());
  int LongSize = DL->getPointerSizeInBits();
  IntptrTy = Type::getIntNTy(*C, LongSize);
  Mapping = getShadowMapping(M, LongSize);
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
  SmallVector<Constant *, 16> Initializers(n);

  Function *CtorFunc = M.getFunction(kAsanModuleCtorName);
  assert(CtorFunc);
  IRBuilder<> IRB(CtorFunc->getEntryBlock().getTerminator());

  bool HasDynamicallyInitializedGlobals = false;

  // We shouldn't merge same module names, as this string serves as unique
  // module ID in runtime.
  GlobalVariable *ModuleName = createPrivateGlobalForString(
      M, M.getModuleIdentifier(), /*AllowMerging*/false);

  for (size_t i = 0; i < n; i++) {
    static const uint64_t kMaxGlobalRedzone = 1 << 18;
    GlobalVariable *G = GlobalsToChange[i];
    PointerType *PtrTy = cast<PointerType>(G->getType());
    Type *Ty = PtrTy->getElementType();
    uint64_t SizeInBytes = DL->getTypeAllocSize(Ty);
    uint64_t MinRZ = MinRedzoneSizeForGlobal();
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

    GlobalVariable *Name =
        createPrivateGlobalForString(M, G->getName(), /*AllowMerging*/true);

    // Create a new global variable with enough space for a redzone.
    GlobalValue::LinkageTypes Linkage = G->getLinkage();
    if (G->isConstant() && Linkage == GlobalValue::PrivateLinkage)
      Linkage = GlobalValue::InternalLinkage;
    GlobalVariable *NewGlobal = new GlobalVariable(
        M, NewTy, G->isConstant(), Linkage,
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
      M, ArrayOfGlobalStructTy, false, GlobalVariable::InternalLinkage,
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
      std::string Suffix =
          (AccessIsWrite ? "store" : "load") + itostr(1 << AccessSizeIndex);
      AsanErrorCallback[AccessIsWrite][AccessSizeIndex] =
          checkInterfaceFunction(
              M.getOrInsertFunction(kAsanReportErrorTemplate + Suffix,
                                    IRB.getVoidTy(), IntptrTy, NULL));
      AsanMemoryAccessCallback[AccessIsWrite][AccessSizeIndex] =
          checkInterfaceFunction(
              M.getOrInsertFunction(ClMemoryAccessCallbackPrefix + Suffix,
                                    IRB.getVoidTy(), IntptrTy, NULL));
    }
  }
  AsanErrorCallbackSized[0] = checkInterfaceFunction(M.getOrInsertFunction(
              kAsanReportLoadN, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
  AsanErrorCallbackSized[1] = checkInterfaceFunction(M.getOrInsertFunction(
              kAsanReportStoreN, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));

  AsanMemoryAccessCallbackSized[0] = checkInterfaceFunction(
      M.getOrInsertFunction(ClMemoryAccessCallbackPrefix + "loadN",
                            IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
  AsanMemoryAccessCallbackSized[1] = checkInterfaceFunction(
      M.getOrInsertFunction(ClMemoryAccessCallbackPrefix + "storeN",
                            IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));

  AsanMemmove = checkInterfaceFunction(M.getOrInsertFunction(
      ClMemoryAccessCallbackPrefix + "memmove", IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IntptrTy, NULL));
  AsanMemcpy = checkInterfaceFunction(M.getOrInsertFunction(
      ClMemoryAccessCallbackPrefix + "memcpy", IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IntptrTy, NULL));
  AsanMemset = checkInterfaceFunction(M.getOrInsertFunction(
      ClMemoryAccessCallbackPrefix + "memset", IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), IRB.getInt32Ty(), IntptrTy, NULL));

  AsanHandleNoReturnFunc = checkInterfaceFunction(
      M.getOrInsertFunction(kAsanHandleNoReturnName, IRB.getVoidTy(), NULL));
  AsanCovFunction = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanCovName, IRB.getVoidTy(), NULL));
  AsanPtrCmpFunction = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanPtrCmp, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
  AsanPtrSubFunction = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanPtrSub, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
  // We insert an empty inline asm after __asan_report* to avoid callback merge.
  EmptyAsm = InlineAsm::get(FunctionType::get(IRB.getVoidTy(), false),
                            StringRef(""), StringRef(""),
                            /*hasSideEffects=*/true);
}

// virtual
bool AddressSanitizer::doInitialization(Module &M) {
  // Initialize the private fields. No one has accessed them before.
  DataLayoutPass *DLP = getAnalysisIfAvailable<DataLayoutPass>();
  if (!DLP)
    report_fatal_error("data layout missing");
  DL = &DLP->getDataLayout();

  BL.reset(SpecialCaseList::createOrDie(BlacklistFile));
  DynamicallyInitializedGlobals.Init(M);

  C = &(M.getContext());
  LongSize = DL->getPointerSizeInBits();
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

  Mapping = getShadowMapping(M, LongSize);

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

void AddressSanitizer::InjectCoverageAtBlock(Function &F, BasicBlock &BB) {
  BasicBlock::iterator IP = BB.getFirstInsertionPt(), BE = BB.end();
  // Skip static allocas at the top of the entry block so they don't become
  // dynamic when we split the block.  If we used our optimized stack layout,
  // then there will only be one alloca and it will come first.
  for (; IP != BE; ++IP) {
    AllocaInst *AI = dyn_cast<AllocaInst>(IP);
    if (!AI || !AI->isStaticAlloca())
      break;
  }

  IRBuilder<> IRB(IP);
  Type *Int8Ty = IRB.getInt8Ty();
  GlobalVariable *Guard = new GlobalVariable(
      *F.getParent(), Int8Ty, false, GlobalValue::PrivateLinkage,
      Constant::getNullValue(Int8Ty), "__asan_gen_cov_" + F.getName());
  LoadInst *Load = IRB.CreateLoad(Guard);
  Load->setAtomic(Monotonic);
  Load->setAlignment(1);
  Value *Cmp = IRB.CreateICmpEQ(Constant::getNullValue(Int8Ty), Load);
  Instruction *Ins = SplitBlockAndInsertIfThen(
      Cmp, IP, false, MDBuilder(*C).createBranchWeights(1, 100000));
  IRB.SetInsertPoint(Ins);
  // We pass &F to __sanitizer_cov. We could avoid this and rely on
  // GET_CALLER_PC, but having the PC of the first instruction is just nice.
  Instruction *Call = IRB.CreateCall(AsanCovFunction);
  Call->setDebugLoc(IP->getDebugLoc());
  StoreInst *Store = IRB.CreateStore(ConstantInt::get(Int8Ty, 1), Guard);
  Store->setAtomic(Monotonic);
  Store->setAlignment(1);
}

// Poor man's coverage that works with ASan.
// We create a Guard boolean variable with the same linkage
// as the function and inject this code into the entry block (-asan-coverage=1)
// or all blocks (-asan-coverage=2):
// if (*Guard) {
//    __sanitizer_cov(&F);
//    *Guard = 1;
// }
// The accesses to Guard are atomic. The rest of the logic is
// in __sanitizer_cov (it's fine to call it more than once).
//
// This coverage implementation provides very limited data:
// it only tells if a given function (block) was ever executed.
// No counters, no per-edge data.
// But for many use cases this is what we need and the added slowdown
// is negligible. This simple implementation will probably be obsoleted
// by the upcoming Clang-based coverage implementation.
// By having it here and now we hope to
//  a) get the functionality to users earlier and
//  b) collect usage statistics to help improve Clang coverage design.
bool AddressSanitizer::InjectCoverage(Function &F,
                                      const ArrayRef<BasicBlock *> AllBlocks) {
  if (!ClCoverage) return false;

  if (ClCoverage == 1 ||
      (unsigned)ClCoverageBlockThreshold < AllBlocks.size()) {
    InjectCoverageAtBlock(F, F.getEntryBlock());
  } else {
    for (size_t i = 0, n = AllBlocks.size(); i < n; i++)
      InjectCoverageAtBlock(F, *AllBlocks[i]);
  }
  return true;
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
  SmallVector<BasicBlock*, 16> AllBlocks;
  SmallVector<Instruction*, 16> PointerComparisonsOrSubtracts;
  int NumAllocas = 0;
  bool IsWrite;

  // Fill the set of memory operations to instrument.
  for (Function::iterator FI = F.begin(), FE = F.end();
       FI != FE; ++FI) {
    AllBlocks.push_back(FI);
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
      } else if (ClInvalidPointerPairs &&
                 isInterestingPointerComparisonOrSubtraction(BI)) {
        PointerComparisonsOrSubtracts.push_back(BI);
        continue;
      } else if (isa<MemIntrinsic>(BI)) {
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

  Function *UninstrumentedDuplicate = nullptr;
  bool LikelyToInstrument =
      !NoReturnCalls.empty() || !ToInstrument.empty() || (NumAllocas > 0);
  if (ClKeepUninstrumented && LikelyToInstrument) {
    ValueToValueMapTy VMap;
    UninstrumentedDuplicate = CloneFunction(&F, VMap, false);
    UninstrumentedDuplicate->removeFnAttr(Attribute::SanitizeAddress);
    UninstrumentedDuplicate->setName("NOASAN_" + F.getName());
    F.getParent()->getFunctionList().push_back(UninstrumentedDuplicate);
  }

  bool UseCalls = false;
  if (ClInstrumentationWithCallsThreshold >= 0 &&
      ToInstrument.size() > (unsigned)ClInstrumentationWithCallsThreshold)
    UseCalls = true;

  // Instrument.
  int NumInstrumented = 0;
  for (size_t i = 0, n = ToInstrument.size(); i != n; i++) {
    Instruction *Inst = ToInstrument[i];
    if (ClDebugMin < 0 || ClDebugMax < 0 ||
        (NumInstrumented >= ClDebugMin && NumInstrumented <= ClDebugMax)) {
      if (isInterestingMemoryAccess(Inst, &IsWrite))
        instrumentMop(Inst, UseCalls);
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

  for (size_t i = 0, n = PointerComparisonsOrSubtracts.size(); i != n; i++) {
    instrumentPointerComparisonOrSubtraction(PointerComparisonsOrSubtracts[i]);
    NumInstrumented++;
  }

  bool res = NumInstrumented > 0 || ChangedStack || !NoReturnCalls.empty();

  if (InjectCoverage(F, AllBlocks))
    res = true;

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
  for (int i = 0; i <= kMaxAsanStackMallocSizeClass; i++) {
    std::string Suffix = itostr(i);
    AsanStackMallocFunc[i] = checkInterfaceFunction(
        M.getOrInsertFunction(kAsanStackMallocNameTemplate + Suffix, IntptrTy,
                              IntptrTy, IntptrTy, NULL));
    AsanStackFreeFunc[i] = checkInterfaceFunction(M.getOrInsertFunction(
        kAsanStackFreeNameTemplate + Suffix, IRB.getVoidTy(), IntptrTy,
        IntptrTy, IntptrTy, NULL));
  }
  AsanPoisonStackMemoryFunc = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanPoisonStackMemoryName, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
  AsanUnpoisonStackMemoryFunc = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanUnpoisonStackMemoryName, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
}

void
FunctionStackPoisoner::poisonRedZones(const ArrayRef<uint8_t> ShadowBytes,
                                      IRBuilder<> &IRB, Value *ShadowBase,
                                      bool DoPoison) {
  size_t n = ShadowBytes.size();
  size_t i = 0;
  // We need to (un)poison n bytes of stack shadow. Poison as many as we can
  // using 64-bit stores (if we are on 64-bit arch), then poison the rest
  // with 32-bit stores, then with 16-byte stores, then with 8-byte stores.
  for (size_t LargeStoreSizeInBytes = ASan.LongSize / 8;
       LargeStoreSizeInBytes != 0; LargeStoreSizeInBytes /= 2) {
    for (; i + LargeStoreSizeInBytes - 1 < n; i += LargeStoreSizeInBytes) {
      uint64_t Val = 0;
      for (size_t j = 0; j < LargeStoreSizeInBytes; j++) {
        if (ASan.DL->isLittleEndian())
          Val |= (uint64_t)ShadowBytes[i + j] << (8 * j);
        else
          Val = (Val << 8) | ShadowBytes[i + j];
      }
      if (!Val) continue;
      Value *Ptr = IRB.CreateAdd(ShadowBase, ConstantInt::get(IntptrTy, i));
      Type *StoreTy = Type::getIntNTy(*C, LargeStoreSizeInBytes * 8);
      Value *Poison = ConstantInt::get(StoreTy, DoPoison ? Val : 0);
      IRB.CreateStore(Poison, IRB.CreateIntToPtr(Ptr, StoreTy->getPointerTo()));
    }
  }
}

// Fake stack allocator (asan_fake_stack.h) has 11 size classes
// for every power of 2 from kMinStackMallocSize to kMaxAsanStackMallocSizeClass
static int StackMallocSizeClass(uint64_t LocalStackSize) {
  assert(LocalStackSize <= kMaxStackMallocSize);
  uint64_t MaxSize = kMinStackMallocSize;
  for (int i = 0; ; i++, MaxSize *= 2)
    if (LocalStackSize <= MaxSize)
      return i;
  llvm_unreachable("impossible LocalStackSize");
}

// Set Size bytes starting from ShadowBase to kAsanStackAfterReturnMagic.
// We can not use MemSet intrinsic because it may end up calling the actual
// memset. Size is a multiple of 8.
// Currently this generates 8-byte stores on x86_64; it may be better to
// generate wider stores.
void FunctionStackPoisoner::SetShadowToStackAfterReturnInlined(
    IRBuilder<> &IRB, Value *ShadowBase, int Size) {
  assert(!(Size % 8));
  assert(kAsanStackAfterReturnMagic == 0xf5);
  for (int i = 0; i < Size; i += 8) {
    Value *p = IRB.CreateAdd(ShadowBase, ConstantInt::get(IntptrTy, i));
    IRB.CreateStore(ConstantInt::get(IRB.getInt64Ty(), 0xf5f5f5f5f5f5f5f5ULL),
                    IRB.CreateIntToPtr(p, IRB.getInt64Ty()->getPointerTo()));
  }
}

void FunctionStackPoisoner::poisonStack() {
  int StackMallocIdx = -1;

  assert(AllocaVec.size() > 0);
  Instruction *InsBefore = AllocaVec[0];
  IRBuilder<> IRB(InsBefore);

  SmallVector<ASanStackVariableDescription, 16> SVD;
  SVD.reserve(AllocaVec.size());
  for (size_t i = 0, n = AllocaVec.size(); i < n; i++) {
    AllocaInst *AI = AllocaVec[i];
    ASanStackVariableDescription D = { AI->getName().data(),
                                   getAllocaSizeInBytes(AI),
                                   AI->getAlignment(), AI, 0};
    SVD.push_back(D);
  }
  // Minimal header size (left redzone) is 4 pointers,
  // i.e. 32 bytes on 64-bit platforms and 16 bytes in 32-bit platforms.
  size_t MinHeaderSize = ASan.LongSize / 2;
  ASanStackFrameLayout L;
  ComputeASanStackFrameLayout(SVD, 1UL << Mapping.Scale, MinHeaderSize, &L);
  DEBUG(dbgs() << L.DescriptionString << " --- " << L.FrameSize << "\n");
  uint64_t LocalStackSize = L.FrameSize;
  bool DoStackMalloc =
      ASan.CheckUseAfterReturn && LocalStackSize <= kMaxStackMallocSize;

  Type *ByteArrayTy = ArrayType::get(IRB.getInt8Ty(), LocalStackSize);
  AllocaInst *MyAlloca =
      new AllocaInst(ByteArrayTy, "MyAlloca", InsBefore);
  assert((ClRealignStack & (ClRealignStack - 1)) == 0);
  size_t FrameAlignment = std::max(L.FrameAlignment, (size_t)ClRealignStack);
  MyAlloca->setAlignment(FrameAlignment);
  assert(MyAlloca->isStaticAlloca());
  Value *OrigStackBase = IRB.CreatePointerCast(MyAlloca, IntptrTy);
  Value *LocalStackBase = OrigStackBase;

  if (DoStackMalloc) {
    // LocalStackBase = OrigStackBase
    // if (__asan_option_detect_stack_use_after_return)
    //   LocalStackBase = __asan_stack_malloc_N(LocalStackBase, OrigStackBase);
    StackMallocIdx = StackMallocSizeClass(LocalStackSize);
    assert(StackMallocIdx <= kMaxAsanStackMallocSizeClass);
    Constant *OptionDetectUAR = F.getParent()->getOrInsertGlobal(
        kAsanOptionDetectUAR, IRB.getInt32Ty());
    Value *Cmp = IRB.CreateICmpNE(IRB.CreateLoad(OptionDetectUAR),
                                  Constant::getNullValue(IRB.getInt32Ty()));
    Instruction *Term = SplitBlockAndInsertIfThen(Cmp, InsBefore, false);
    BasicBlock *CmpBlock = cast<Instruction>(Cmp)->getParent();
    IRBuilder<> IRBIf(Term);
    LocalStackBase = IRBIf.CreateCall2(
        AsanStackMallocFunc[StackMallocIdx],
        ConstantInt::get(IntptrTy, LocalStackSize), OrigStackBase);
    BasicBlock *SetBlock = cast<Instruction>(LocalStackBase)->getParent();
    IRB.SetInsertPoint(InsBefore);
    PHINode *Phi = IRB.CreatePHI(IntptrTy, 2);
    Phi->addIncoming(OrigStackBase, CmpBlock);
    Phi->addIncoming(LocalStackBase, SetBlock);
    LocalStackBase = Phi;
  }

  // Insert poison calls for lifetime intrinsics for alloca.
  bool HavePoisonedAllocas = false;
  for (size_t i = 0, n = AllocaPoisonCallVec.size(); i < n; i++) {
    const AllocaPoisonCall &APC = AllocaPoisonCallVec[i];
    assert(APC.InsBefore);
    assert(APC.AI);
    IRBuilder<> IRB(APC.InsBefore);
    poisonAlloca(APC.AI, APC.Size, IRB, APC.DoPoison);
    HavePoisonedAllocas |= APC.DoPoison;
  }

  // Replace Alloca instructions with base+offset.
  for (size_t i = 0, n = SVD.size(); i < n; i++) {
    AllocaInst *AI = SVD[i].AI;
    Value *NewAllocaPtr = IRB.CreateIntToPtr(
        IRB.CreateAdd(LocalStackBase,
                      ConstantInt::get(IntptrTy, SVD[i].Offset)),
        AI->getType());
    replaceDbgDeclareForAlloca(AI, NewAllocaPtr, DIB);
    AI->replaceAllUsesWith(NewAllocaPtr);
  }

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
      createPrivateGlobalForString(*F.getParent(), L.DescriptionString,
                                   /*AllowMerging*/true);
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
  poisonRedZones(L.ShadowBytes, IRB, ShadowBase, true);

  // (Un)poison the stack before all ret instructions.
  for (size_t i = 0, n = RetVec.size(); i < n; i++) {
    Instruction *Ret = RetVec[i];
    IRBuilder<> IRBRet(Ret);
    // Mark the current frame as retired.
    IRBRet.CreateStore(ConstantInt::get(IntptrTy, kRetiredStackFrameMagic),
                       BasePlus0);
    if (DoStackMalloc) {
      assert(StackMallocIdx >= 0);
      // if LocalStackBase != OrigStackBase:
      //     // In use-after-return mode, poison the whole stack frame.
      //     if StackMallocIdx <= 4
      //         // For small sizes inline the whole thing:
      //         memset(ShadowBase, kAsanStackAfterReturnMagic, ShadowSize);
      //         **SavedFlagPtr(LocalStackBase) = 0
      //     else
      //         __asan_stack_free_N(LocalStackBase, OrigStackBase)
      // else
      //     <This is not a fake stack; unpoison the redzones>
      Value *Cmp = IRBRet.CreateICmpNE(LocalStackBase, OrigStackBase);
      TerminatorInst *ThenTerm, *ElseTerm;
      SplitBlockAndInsertIfThenElse(Cmp, Ret, &ThenTerm, &ElseTerm);

      IRBuilder<> IRBPoison(ThenTerm);
      if (StackMallocIdx <= 4) {
        int ClassSize = kMinStackMallocSize << StackMallocIdx;
        SetShadowToStackAfterReturnInlined(IRBPoison, ShadowBase,
                                           ClassSize >> Mapping.Scale);
        Value *SavedFlagPtrPtr = IRBPoison.CreateAdd(
            LocalStackBase,
            ConstantInt::get(IntptrTy, ClassSize - ASan.LongSize / 8));
        Value *SavedFlagPtr = IRBPoison.CreateLoad(
            IRBPoison.CreateIntToPtr(SavedFlagPtrPtr, IntptrPtrTy));
        IRBPoison.CreateStore(
            Constant::getNullValue(IRBPoison.getInt8Ty()),
            IRBPoison.CreateIntToPtr(SavedFlagPtr, IRBPoison.getInt8PtrTy()));
      } else {
        // For larger frames call __asan_stack_free_*.
        IRBPoison.CreateCall3(AsanStackFreeFunc[StackMallocIdx], LocalStackBase,
                              ConstantInt::get(IntptrTy, LocalStackSize),
                              OrigStackBase);
      }

      IRBuilder<> IRBElse(ElseTerm);
      poisonRedZones(L.ShadowBytes, IRBElse, ShadowBase, false);
    } else if (HavePoisonedAllocas) {
      // If we poisoned some allocas in llvm.lifetime analysis,
      // unpoison whole stack frame now.
      assert(LocalStackBase == OrigStackBase);
      poisonAlloca(LocalStackBase, LocalStackSize, IRBRet, false);
    } else {
      poisonRedZones(L.ShadowBytes, IRBRet, ShadowBase, false);
    }
  }

  // We are done. Remove the old unused alloca instructions.
  for (size_t i = 0, n = AllocaVec.size(); i < n; i++)
    AllocaVec[i]->eraseFromParent();
}

void FunctionStackPoisoner::poisonAlloca(Value *V, uint64_t Size,
                                         IRBuilder<> &IRB, bool DoPoison) {
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
    return isInterestingAlloca(*AI) ? AI : nullptr;
  // See if we've already calculated (or started to calculate) alloca for a
  // given value.
  AllocaForValueMapTy::iterator I = AllocaForValue.find(V);
  if (I != AllocaForValue.end())
    return I->second;
  // Store 0 while we're calculating alloca for value V to avoid
  // infinite recursion if the value references itself.
  AllocaForValue[V] = nullptr;
  AllocaInst *Res = nullptr;
  if (CastInst *CI = dyn_cast<CastInst>(V))
    Res = findAllocaForValue(CI->getOperand(0));
  else if (PHINode *PN = dyn_cast<PHINode>(V)) {
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      Value *IncValue = PN->getIncomingValue(i);
      // Allow self-referencing phi-nodes.
      if (IncValue == PN) continue;
      AllocaInst *IncValueAI = findAllocaForValue(IncValue);
      // AI for incoming values should exist and should all be equal.
      if (IncValueAI == nullptr || (Res != nullptr && IncValueAI != Res))
        return nullptr;
      Res = IncValueAI;
    }
  }
  if (Res)
    AllocaForValue[V] = Res;
  return Res;
}
