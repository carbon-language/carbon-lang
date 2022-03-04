//===- MemorySanitizer.cpp - detector of uninitialized reads --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file is a part of MemorySanitizer, a detector of uninitialized
/// reads.
///
/// The algorithm of the tool is similar to Memcheck
/// (http://goo.gl/QKbem). We associate a few shadow bits with every
/// byte of the application memory, poison the shadow of the malloc-ed
/// or alloca-ed memory, load the shadow bits on every memory read,
/// propagate the shadow bits through some of the arithmetic
/// instruction (including MOV), store the shadow bits on every memory
/// write, report a bug on some other instructions (e.g. JMP) if the
/// associated shadow is poisoned.
///
/// But there are differences too. The first and the major one:
/// compiler instrumentation instead of binary instrumentation. This
/// gives us much better register allocation, possible compiler
/// optimizations and a fast start-up. But this brings the major issue
/// as well: msan needs to see all program events, including system
/// calls and reads/writes in system libraries, so we either need to
/// compile *everything* with msan or use a binary translation
/// component (e.g. DynamoRIO) to instrument pre-built libraries.
/// Another difference from Memcheck is that we use 8 shadow bits per
/// byte of application memory and use a direct shadow mapping. This
/// greatly simplifies the instrumentation code and avoids races on
/// shadow updates (Memcheck is single-threaded so races are not a
/// concern there. Memcheck uses 2 shadow bits per byte with a slow
/// path storage that uses 8 bits per byte).
///
/// The default value of shadow is 0, which means "clean" (not poisoned).
///
/// Every module initializer should call __msan_init to ensure that the
/// shadow memory is ready. On error, __msan_warning is called. Since
/// parameters and return values may be passed via registers, we have a
/// specialized thread-local shadow for return values
/// (__msan_retval_tls) and parameters (__msan_param_tls).
///
///                           Origin tracking.
///
/// MemorySanitizer can track origins (allocation points) of all uninitialized
/// values. This behavior is controlled with a flag (msan-track-origins) and is
/// disabled by default.
///
/// Origins are 4-byte values created and interpreted by the runtime library.
/// They are stored in a second shadow mapping, one 4-byte value for 4 bytes
/// of application memory. Propagation of origins is basically a bunch of
/// "select" instructions that pick the origin of a dirty argument, if an
/// instruction has one.
///
/// Every 4 aligned, consecutive bytes of application memory have one origin
/// value associated with them. If these bytes contain uninitialized data
/// coming from 2 different allocations, the last store wins. Because of this,
/// MemorySanitizer reports can show unrelated origins, but this is unlikely in
/// practice.
///
/// Origins are meaningless for fully initialized values, so MemorySanitizer
/// avoids storing origin to memory when a fully initialized value is stored.
/// This way it avoids needless overwriting origin of the 4-byte region on
/// a short (i.e. 1 byte) clean store, and it is also good for performance.
///
///                            Atomic handling.
///
/// Ideally, every atomic store of application value should update the
/// corresponding shadow location in an atomic way. Unfortunately, atomic store
/// of two disjoint locations can not be done without severe slowdown.
///
/// Therefore, we implement an approximation that may err on the safe side.
/// In this implementation, every atomically accessed location in the program
/// may only change from (partially) uninitialized to fully initialized, but
/// not the other way around. We load the shadow _after_ the application load,
/// and we store the shadow _before_ the app store. Also, we always store clean
/// shadow (if the application store is atomic). This way, if the store-load
/// pair constitutes a happens-before arc, shadow store and load are correctly
/// ordered such that the load will get either the value that was stored, or
/// some later value (which is always clean).
///
/// This does not work very well with Compare-And-Swap (CAS) and
/// Read-Modify-Write (RMW) operations. To follow the above logic, CAS and RMW
/// must store the new shadow before the app operation, and load the shadow
/// after the app operation. Computers don't work this way. Current
/// implementation ignores the load aspect of CAS/RMW, always returning a clean
/// value. It implements the store part as a simple atomic store by storing a
/// clean shadow.
///
///                      Instrumenting inline assembly.
///
/// For inline assembly code LLVM has little idea about which memory locations
/// become initialized depending on the arguments. It can be possible to figure
/// out which arguments are meant to point to inputs and outputs, but the
/// actual semantics can be only visible at runtime. In the Linux kernel it's
/// also possible that the arguments only indicate the offset for a base taken
/// from a segment register, so it's dangerous to treat any asm() arguments as
/// pointers. We take a conservative approach generating calls to
///   __msan_instrument_asm_store(ptr, size)
/// , which defer the memory unpoisoning to the runtime library.
/// The latter can perform more complex address checks to figure out whether
/// it's safe to touch the shadow memory.
/// Like with atomic operations, we call __msan_instrument_asm_store() before
/// the assembly call, so that changes to the shadow memory will be seen by
/// other threads together with main memory initialization.
///
///                  KernelMemorySanitizer (KMSAN) implementation.
///
/// The major differences between KMSAN and MSan instrumentation are:
///  - KMSAN always tracks the origins and implies msan-keep-going=true;
///  - KMSAN allocates shadow and origin memory for each page separately, so
///    there are no explicit accesses to shadow and origin in the
///    instrumentation.
///    Shadow and origin values for a particular X-byte memory location
///    (X=1,2,4,8) are accessed through pointers obtained via the
///      __msan_metadata_ptr_for_load_X(ptr)
///      __msan_metadata_ptr_for_store_X(ptr)
///    functions. The corresponding functions check that the X-byte accesses
///    are possible and returns the pointers to shadow and origin memory.
///    Arbitrary sized accesses are handled with:
///      __msan_metadata_ptr_for_load_n(ptr, size)
///      __msan_metadata_ptr_for_store_n(ptr, size);
///  - TLS variables are stored in a single per-task struct. A call to a
///    function __msan_get_context_state() returning a pointer to that struct
///    is inserted into every instrumented function before the entry block;
///  - __msan_warning() takes a 32-bit origin parameter;
///  - local variables are poisoned with __msan_poison_alloca() upon function
///    entry and unpoisoned with __msan_unpoison_alloca() before leaving the
///    function;
///  - the pass doesn't declare any global variables or add global constructors
///    to the translation unit.
///
/// Also, KMSAN currently ignores uninitialized memory passed into inline asm
/// calls, making sure we're on the safe side wrt. possible false positives.
///
///  KernelMemorySanitizer only supports X86_64 at the moment.
///
//
// FIXME: This sanitizer does not yet handle scalable vectors
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "msan"

static const unsigned kOriginSize = 4;
static const Align kMinOriginAlignment = Align(4);
static const Align kShadowTLSAlignment = Align(8);

// These constants must be kept in sync with the ones in msan.h.
static const unsigned kParamTLSSize = 800;
static const unsigned kRetvalTLSSize = 800;

// Accesses sizes are powers of two: 1, 2, 4, 8.
static const size_t kNumberOfAccessSizes = 4;

/// Track origins of uninitialized values.
///
/// Adds a section to MemorySanitizer report that points to the allocation
/// (stack or heap) the uninitialized bits came from originally.
static cl::opt<int> ClTrackOrigins("msan-track-origins",
       cl::desc("Track origins (allocation sites) of poisoned memory"),
       cl::Hidden, cl::init(0));

static cl::opt<bool> ClKeepGoing("msan-keep-going",
       cl::desc("keep going after reporting a UMR"),
       cl::Hidden, cl::init(false));

static cl::opt<bool> ClPoisonStack("msan-poison-stack",
       cl::desc("poison uninitialized stack variables"),
       cl::Hidden, cl::init(true));

static cl::opt<bool> ClPoisonStackWithCall("msan-poison-stack-with-call",
       cl::desc("poison uninitialized stack variables with a call"),
       cl::Hidden, cl::init(false));

static cl::opt<int> ClPoisonStackPattern("msan-poison-stack-pattern",
       cl::desc("poison uninitialized stack variables with the given pattern"),
       cl::Hidden, cl::init(0xff));

static cl::opt<bool> ClPoisonUndef("msan-poison-undef",
       cl::desc("poison undef temps"),
       cl::Hidden, cl::init(true));

static cl::opt<bool> ClHandleICmp("msan-handle-icmp",
       cl::desc("propagate shadow through ICmpEQ and ICmpNE"),
       cl::Hidden, cl::init(true));

static cl::opt<bool> ClHandleICmpExact("msan-handle-icmp-exact",
       cl::desc("exact handling of relational integer ICmp"),
       cl::Hidden, cl::init(false));

static cl::opt<bool> ClHandleLifetimeIntrinsics(
    "msan-handle-lifetime-intrinsics",
    cl::desc(
        "when possible, poison scoped variables at the beginning of the scope "
        "(slower, but more precise)"),
    cl::Hidden, cl::init(true));

// When compiling the Linux kernel, we sometimes see false positives related to
// MSan being unable to understand that inline assembly calls may initialize
// local variables.
// This flag makes the compiler conservatively unpoison every memory location
// passed into an assembly call. Note that this may cause false positives.
// Because it's impossible to figure out the array sizes, we can only unpoison
// the first sizeof(type) bytes for each type* pointer.
// The instrumentation is only enabled in KMSAN builds, and only if
// -msan-handle-asm-conservative is on. This is done because we may want to
// quickly disable assembly instrumentation when it breaks.
static cl::opt<bool> ClHandleAsmConservative(
    "msan-handle-asm-conservative",
    cl::desc("conservative handling of inline assembly"), cl::Hidden,
    cl::init(true));

// This flag controls whether we check the shadow of the address
// operand of load or store. Such bugs are very rare, since load from
// a garbage address typically results in SEGV, but still happen
// (e.g. only lower bits of address are garbage, or the access happens
// early at program startup where malloc-ed memory is more likely to
// be zeroed. As of 2012-08-28 this flag adds 20% slowdown.
static cl::opt<bool> ClCheckAccessAddress("msan-check-access-address",
       cl::desc("report accesses through a pointer which has poisoned shadow"),
       cl::Hidden, cl::init(true));

static cl::opt<bool> ClEagerChecks(
    "msan-eager-checks",
    cl::desc("check arguments and return values at function call boundaries"),
    cl::Hidden, cl::init(false));

static cl::opt<bool> ClDumpStrictInstructions("msan-dump-strict-instructions",
       cl::desc("print out instructions with default strict semantics"),
       cl::Hidden, cl::init(false));

static cl::opt<int> ClInstrumentationWithCallThreshold(
    "msan-instrumentation-with-call-threshold",
    cl::desc(
        "If the function being instrumented requires more than "
        "this number of checks and origin stores, use callbacks instead of "
        "inline checks (-1 means never use callbacks)."),
    cl::Hidden, cl::init(3500));

static cl::opt<bool>
    ClEnableKmsan("msan-kernel",
                  cl::desc("Enable KernelMemorySanitizer instrumentation"),
                  cl::Hidden, cl::init(false));

static cl::opt<bool>
    ClDisableChecks("msan-disable-checks",
                    cl::desc("Apply no_sanitize to the whole file"), cl::Hidden,
                    cl::init(false));

// This is an experiment to enable handling of cases where shadow is a non-zero
// compile-time constant. For some unexplainable reason they were silently
// ignored in the instrumentation.
static cl::opt<bool> ClCheckConstantShadow("msan-check-constant-shadow",
       cl::desc("Insert checks for constant shadow values"),
       cl::Hidden, cl::init(false));

// This is off by default because of a bug in gold:
// https://sourceware.org/bugzilla/show_bug.cgi?id=19002
static cl::opt<bool> ClWithComdat("msan-with-comdat",
       cl::desc("Place MSan constructors in comdat sections"),
       cl::Hidden, cl::init(false));

// These options allow to specify custom memory map parameters
// See MemoryMapParams for details.
static cl::opt<uint64_t> ClAndMask("msan-and-mask",
                                   cl::desc("Define custom MSan AndMask"),
                                   cl::Hidden, cl::init(0));

static cl::opt<uint64_t> ClXorMask("msan-xor-mask",
                                   cl::desc("Define custom MSan XorMask"),
                                   cl::Hidden, cl::init(0));

static cl::opt<uint64_t> ClShadowBase("msan-shadow-base",
                                      cl::desc("Define custom MSan ShadowBase"),
                                      cl::Hidden, cl::init(0));

static cl::opt<uint64_t> ClOriginBase("msan-origin-base",
                                      cl::desc("Define custom MSan OriginBase"),
                                      cl::Hidden, cl::init(0));

const char kMsanModuleCtorName[] = "msan.module_ctor";
const char kMsanInitName[] = "__msan_init";

namespace {

// Memory map parameters used in application-to-shadow address calculation.
// Offset = (Addr & ~AndMask) ^ XorMask
// Shadow = ShadowBase + Offset
// Origin = OriginBase + Offset
struct MemoryMapParams {
  uint64_t AndMask;
  uint64_t XorMask;
  uint64_t ShadowBase;
  uint64_t OriginBase;
};

struct PlatformMemoryMapParams {
  const MemoryMapParams *bits32;
  const MemoryMapParams *bits64;
};

} // end anonymous namespace

// i386 Linux
static const MemoryMapParams Linux_I386_MemoryMapParams = {
  0x000080000000,  // AndMask
  0,               // XorMask (not used)
  0,               // ShadowBase (not used)
  0x000040000000,  // OriginBase
};

// x86_64 Linux
static const MemoryMapParams Linux_X86_64_MemoryMapParams = {
#ifdef MSAN_LINUX_X86_64_OLD_MAPPING
  0x400000000000,  // AndMask
  0,               // XorMask (not used)
  0,               // ShadowBase (not used)
  0x200000000000,  // OriginBase
#else
  0,               // AndMask (not used)
  0x500000000000,  // XorMask
  0,               // ShadowBase (not used)
  0x100000000000,  // OriginBase
#endif
};

// mips64 Linux
static const MemoryMapParams Linux_MIPS64_MemoryMapParams = {
  0,               // AndMask (not used)
  0x008000000000,  // XorMask
  0,               // ShadowBase (not used)
  0x002000000000,  // OriginBase
};

// ppc64 Linux
static const MemoryMapParams Linux_PowerPC64_MemoryMapParams = {
  0xE00000000000,  // AndMask
  0x100000000000,  // XorMask
  0x080000000000,  // ShadowBase
  0x1C0000000000,  // OriginBase
};

// s390x Linux
static const MemoryMapParams Linux_S390X_MemoryMapParams = {
    0xC00000000000, // AndMask
    0,              // XorMask (not used)
    0x080000000000, // ShadowBase
    0x1C0000000000, // OriginBase
};

// aarch64 Linux
static const MemoryMapParams Linux_AArch64_MemoryMapParams = {
  0,               // AndMask (not used)
  0x06000000000,   // XorMask
  0,               // ShadowBase (not used)
  0x01000000000,   // OriginBase
};

// i386 FreeBSD
static const MemoryMapParams FreeBSD_I386_MemoryMapParams = {
  0x000180000000,  // AndMask
  0x000040000000,  // XorMask
  0x000020000000,  // ShadowBase
  0x000700000000,  // OriginBase
};

// x86_64 FreeBSD
static const MemoryMapParams FreeBSD_X86_64_MemoryMapParams = {
  0xc00000000000,  // AndMask
  0x200000000000,  // XorMask
  0x100000000000,  // ShadowBase
  0x380000000000,  // OriginBase
};

// x86_64 NetBSD
static const MemoryMapParams NetBSD_X86_64_MemoryMapParams = {
  0,               // AndMask
  0x500000000000,  // XorMask
  0,               // ShadowBase
  0x100000000000,  // OriginBase
};

static const PlatformMemoryMapParams Linux_X86_MemoryMapParams = {
  &Linux_I386_MemoryMapParams,
  &Linux_X86_64_MemoryMapParams,
};

static const PlatformMemoryMapParams Linux_MIPS_MemoryMapParams = {
  nullptr,
  &Linux_MIPS64_MemoryMapParams,
};

static const PlatformMemoryMapParams Linux_PowerPC_MemoryMapParams = {
  nullptr,
  &Linux_PowerPC64_MemoryMapParams,
};

static const PlatformMemoryMapParams Linux_S390_MemoryMapParams = {
    nullptr,
    &Linux_S390X_MemoryMapParams,
};

static const PlatformMemoryMapParams Linux_ARM_MemoryMapParams = {
  nullptr,
  &Linux_AArch64_MemoryMapParams,
};

static const PlatformMemoryMapParams FreeBSD_X86_MemoryMapParams = {
  &FreeBSD_I386_MemoryMapParams,
  &FreeBSD_X86_64_MemoryMapParams,
};

static const PlatformMemoryMapParams NetBSD_X86_MemoryMapParams = {
  nullptr,
  &NetBSD_X86_64_MemoryMapParams,
};

namespace {

/// Instrument functions of a module to detect uninitialized reads.
///
/// Instantiating MemorySanitizer inserts the msan runtime library API function
/// declarations into the module if they don't exist already. Instantiating
/// ensures the __msan_init function is in the list of global constructors for
/// the module.
class MemorySanitizer {
public:
  MemorySanitizer(Module &M, MemorySanitizerOptions Options)
      : CompileKernel(Options.Kernel), TrackOrigins(Options.TrackOrigins),
        Recover(Options.Recover), EagerChecks(Options.EagerChecks) {
    initializeModule(M);
  }

  // MSan cannot be moved or copied because of MapParams.
  MemorySanitizer(MemorySanitizer &&) = delete;
  MemorySanitizer &operator=(MemorySanitizer &&) = delete;
  MemorySanitizer(const MemorySanitizer &) = delete;
  MemorySanitizer &operator=(const MemorySanitizer &) = delete;

  bool sanitizeFunction(Function &F, TargetLibraryInfo &TLI);

private:
  friend struct MemorySanitizerVisitor;
  friend struct VarArgAMD64Helper;
  friend struct VarArgMIPS64Helper;
  friend struct VarArgAArch64Helper;
  friend struct VarArgPowerPC64Helper;
  friend struct VarArgSystemZHelper;

  void initializeModule(Module &M);
  void initializeCallbacks(Module &M);
  void createKernelApi(Module &M);
  void createUserspaceApi(Module &M);

  /// True if we're compiling the Linux kernel.
  bool CompileKernel;
  /// Track origins (allocation points) of uninitialized values.
  int TrackOrigins;
  bool Recover;
  bool EagerChecks;

  LLVMContext *C;
  Type *IntptrTy;
  Type *OriginTy;

  // XxxTLS variables represent the per-thread state in MSan and per-task state
  // in KMSAN.
  // For the userspace these point to thread-local globals. In the kernel land
  // they point to the members of a per-task struct obtained via a call to
  // __msan_get_context_state().

  /// Thread-local shadow storage for function parameters.
  Value *ParamTLS;

  /// Thread-local origin storage for function parameters.
  Value *ParamOriginTLS;

  /// Thread-local shadow storage for function return value.
  Value *RetvalTLS;

  /// Thread-local origin storage for function return value.
  Value *RetvalOriginTLS;

  /// Thread-local shadow storage for in-register va_arg function
  /// parameters (x86_64-specific).
  Value *VAArgTLS;

  /// Thread-local shadow storage for in-register va_arg function
  /// parameters (x86_64-specific).
  Value *VAArgOriginTLS;

  /// Thread-local shadow storage for va_arg overflow area
  /// (x86_64-specific).
  Value *VAArgOverflowSizeTLS;

  /// Are the instrumentation callbacks set up?
  bool CallbacksInitialized = false;

  /// The run-time callback to print a warning.
  FunctionCallee WarningFn;

  // These arrays are indexed by log2(AccessSize).
  FunctionCallee MaybeWarningFn[kNumberOfAccessSizes];
  FunctionCallee MaybeStoreOriginFn[kNumberOfAccessSizes];

  /// Run-time helper that generates a new origin value for a stack
  /// allocation.
  FunctionCallee MsanSetAllocaOrigin4Fn;

  /// Run-time helper that poisons stack on function entry.
  FunctionCallee MsanPoisonStackFn;

  /// Run-time helper that records a store (or any event) of an
  /// uninitialized value and returns an updated origin id encoding this info.
  FunctionCallee MsanChainOriginFn;

  /// Run-time helper that paints an origin over a region.
  FunctionCallee MsanSetOriginFn;

  /// MSan runtime replacements for memmove, memcpy and memset.
  FunctionCallee MemmoveFn, MemcpyFn, MemsetFn;

  /// KMSAN callback for task-local function argument shadow.
  StructType *MsanContextStateTy;
  FunctionCallee MsanGetContextStateFn;

  /// Functions for poisoning/unpoisoning local variables
  FunctionCallee MsanPoisonAllocaFn, MsanUnpoisonAllocaFn;

  /// Each of the MsanMetadataPtrXxx functions returns a pair of shadow/origin
  /// pointers.
  FunctionCallee MsanMetadataPtrForLoadN, MsanMetadataPtrForStoreN;
  FunctionCallee MsanMetadataPtrForLoad_1_8[4];
  FunctionCallee MsanMetadataPtrForStore_1_8[4];
  FunctionCallee MsanInstrumentAsmStoreFn;

  /// Helper to choose between different MsanMetadataPtrXxx().
  FunctionCallee getKmsanShadowOriginAccessFn(bool isStore, int size);

  /// Memory map parameters used in application-to-shadow calculation.
  const MemoryMapParams *MapParams;

  /// Custom memory map parameters used when -msan-shadow-base or
  // -msan-origin-base is provided.
  MemoryMapParams CustomMapParams;

  MDNode *ColdCallWeights;

  /// Branch weights for origin store.
  MDNode *OriginStoreWeights;
};

void insertModuleCtor(Module &M) {
  getOrCreateSanitizerCtorAndInitFunctions(
      M, kMsanModuleCtorName, kMsanInitName,
      /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      // This callback is invoked when the functions are created the first
      // time. Hook them into the global ctors list in that case:
      [&](Function *Ctor, FunctionCallee) {
        if (!ClWithComdat) {
          appendToGlobalCtors(M, Ctor, 0);
          return;
        }
        Comdat *MsanCtorComdat = M.getOrInsertComdat(kMsanModuleCtorName);
        Ctor->setComdat(MsanCtorComdat);
        appendToGlobalCtors(M, Ctor, 0, Ctor);
      });
}

/// A legacy function pass for msan instrumentation.
///
/// Instruments functions to detect uninitialized reads.
struct MemorySanitizerLegacyPass : public FunctionPass {
  // Pass identification, replacement for typeid.
  static char ID;

  MemorySanitizerLegacyPass(MemorySanitizerOptions Options = {})
      : FunctionPass(ID), Options(Options) {
    initializeMemorySanitizerLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override { return "MemorySanitizerLegacyPass"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    return MSan->sanitizeFunction(
        F, getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F));
  }
  bool doInitialization(Module &M) override;

  Optional<MemorySanitizer> MSan;
  MemorySanitizerOptions Options;
};

template <class T> T getOptOrDefault(const cl::opt<T> &Opt, T Default) {
  return (Opt.getNumOccurrences() > 0) ? Opt : Default;
}

} // end anonymous namespace

MemorySanitizerOptions::MemorySanitizerOptions(int TO, bool R, bool K,
                                               bool EagerChecks)
    : Kernel(getOptOrDefault(ClEnableKmsan, K)),
      TrackOrigins(getOptOrDefault(ClTrackOrigins, Kernel ? 2 : TO)),
      Recover(getOptOrDefault(ClKeepGoing, Kernel || R)),
      EagerChecks(getOptOrDefault(ClEagerChecks, EagerChecks)) {}

PreservedAnalyses MemorySanitizerPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  MemorySanitizer Msan(*F.getParent(), Options);
  if (Msan.sanitizeFunction(F, FAM.getResult<TargetLibraryAnalysis>(F)))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

PreservedAnalyses
ModuleMemorySanitizerPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (Options.Kernel)
    return PreservedAnalyses::all();
  insertModuleCtor(M);
  return PreservedAnalyses::none();
}

void MemorySanitizerPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<MemorySanitizerPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  OS << "<";
  if (Options.Recover)
    OS << "recover;";
  if (Options.Kernel)
    OS << "kernel;";
  if (Options.EagerChecks)
    OS << "eager-checks;";
  OS << "track-origins=" << Options.TrackOrigins;
  OS << ">";
}

char MemorySanitizerLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(MemorySanitizerLegacyPass, "msan",
                      "MemorySanitizer: detects uninitialized reads.", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(MemorySanitizerLegacyPass, "msan",
                    "MemorySanitizer: detects uninitialized reads.", false,
                    false)

FunctionPass *
llvm::createMemorySanitizerLegacyPassPass(MemorySanitizerOptions Options) {
  return new MemorySanitizerLegacyPass(Options);
}

/// Create a non-const global initialized with the given string.
///
/// Creates a writable global for Str so that we can pass it to the
/// run-time lib. Runtime uses first 4 bytes of the string to store the
/// frame ID, so the string needs to be mutable.
static GlobalVariable *createPrivateNonConstGlobalForString(Module &M,
                                                            StringRef Str) {
  Constant *StrConst = ConstantDataArray::getString(M.getContext(), Str);
  return new GlobalVariable(M, StrConst->getType(), /*isConstant=*/false,
                            GlobalValue::PrivateLinkage, StrConst, "");
}

/// Create KMSAN API callbacks.
void MemorySanitizer::createKernelApi(Module &M) {
  IRBuilder<> IRB(*C);

  // These will be initialized in insertKmsanPrologue().
  RetvalTLS = nullptr;
  RetvalOriginTLS = nullptr;
  ParamTLS = nullptr;
  ParamOriginTLS = nullptr;
  VAArgTLS = nullptr;
  VAArgOriginTLS = nullptr;
  VAArgOverflowSizeTLS = nullptr;

  WarningFn = M.getOrInsertFunction("__msan_warning", IRB.getVoidTy(),
                                    IRB.getInt32Ty());
  // Requests the per-task context state (kmsan_context_state*) from the
  // runtime library.
  MsanContextStateTy = StructType::get(
      ArrayType::get(IRB.getInt64Ty(), kParamTLSSize / 8),
      ArrayType::get(IRB.getInt64Ty(), kRetvalTLSSize / 8),
      ArrayType::get(IRB.getInt64Ty(), kParamTLSSize / 8),
      ArrayType::get(IRB.getInt64Ty(), kParamTLSSize / 8), /* va_arg_origin */
      IRB.getInt64Ty(), ArrayType::get(OriginTy, kParamTLSSize / 4), OriginTy,
      OriginTy);
  MsanGetContextStateFn = M.getOrInsertFunction(
      "__msan_get_context_state", PointerType::get(MsanContextStateTy, 0));

  Type *RetTy = StructType::get(PointerType::get(IRB.getInt8Ty(), 0),
                                PointerType::get(IRB.getInt32Ty(), 0));

  for (int ind = 0, size = 1; ind < 4; ind++, size <<= 1) {
    std::string name_load =
        "__msan_metadata_ptr_for_load_" + std::to_string(size);
    std::string name_store =
        "__msan_metadata_ptr_for_store_" + std::to_string(size);
    MsanMetadataPtrForLoad_1_8[ind] = M.getOrInsertFunction(
        name_load, RetTy, PointerType::get(IRB.getInt8Ty(), 0));
    MsanMetadataPtrForStore_1_8[ind] = M.getOrInsertFunction(
        name_store, RetTy, PointerType::get(IRB.getInt8Ty(), 0));
  }

  MsanMetadataPtrForLoadN = M.getOrInsertFunction(
      "__msan_metadata_ptr_for_load_n", RetTy,
      PointerType::get(IRB.getInt8Ty(), 0), IRB.getInt64Ty());
  MsanMetadataPtrForStoreN = M.getOrInsertFunction(
      "__msan_metadata_ptr_for_store_n", RetTy,
      PointerType::get(IRB.getInt8Ty(), 0), IRB.getInt64Ty());

  // Functions for poisoning and unpoisoning memory.
  MsanPoisonAllocaFn =
      M.getOrInsertFunction("__msan_poison_alloca", IRB.getVoidTy(),
                            IRB.getInt8PtrTy(), IntptrTy, IRB.getInt8PtrTy());
  MsanUnpoisonAllocaFn = M.getOrInsertFunction(
      "__msan_unpoison_alloca", IRB.getVoidTy(), IRB.getInt8PtrTy(), IntptrTy);
}

static Constant *getOrInsertGlobal(Module &M, StringRef Name, Type *Ty) {
  return M.getOrInsertGlobal(Name, Ty, [&] {
    return new GlobalVariable(M, Ty, false, GlobalVariable::ExternalLinkage,
                              nullptr, Name, nullptr,
                              GlobalVariable::InitialExecTLSModel);
  });
}

/// Insert declarations for userspace-specific functions and globals.
void MemorySanitizer::createUserspaceApi(Module &M) {
  IRBuilder<> IRB(*C);

  // Create the callback.
  // FIXME: this function should have "Cold" calling conv,
  // which is not yet implemented.
  StringRef WarningFnName = Recover ? "__msan_warning_with_origin"
                                    : "__msan_warning_with_origin_noreturn";
  WarningFn =
      M.getOrInsertFunction(WarningFnName, IRB.getVoidTy(), IRB.getInt32Ty());

  // Create the global TLS variables.
  RetvalTLS =
      getOrInsertGlobal(M, "__msan_retval_tls",
                        ArrayType::get(IRB.getInt64Ty(), kRetvalTLSSize / 8));

  RetvalOriginTLS = getOrInsertGlobal(M, "__msan_retval_origin_tls", OriginTy);

  ParamTLS =
      getOrInsertGlobal(M, "__msan_param_tls",
                        ArrayType::get(IRB.getInt64Ty(), kParamTLSSize / 8));

  ParamOriginTLS =
      getOrInsertGlobal(M, "__msan_param_origin_tls",
                        ArrayType::get(OriginTy, kParamTLSSize / 4));

  VAArgTLS =
      getOrInsertGlobal(M, "__msan_va_arg_tls",
                        ArrayType::get(IRB.getInt64Ty(), kParamTLSSize / 8));

  VAArgOriginTLS =
      getOrInsertGlobal(M, "__msan_va_arg_origin_tls",
                        ArrayType::get(OriginTy, kParamTLSSize / 4));

  VAArgOverflowSizeTLS =
      getOrInsertGlobal(M, "__msan_va_arg_overflow_size_tls", IRB.getInt64Ty());

  for (size_t AccessSizeIndex = 0; AccessSizeIndex < kNumberOfAccessSizes;
       AccessSizeIndex++) {
    unsigned AccessSize = 1 << AccessSizeIndex;
    std::string FunctionName = "__msan_maybe_warning_" + itostr(AccessSize);
    SmallVector<std::pair<unsigned, Attribute>, 2> MaybeWarningFnAttrs;
    MaybeWarningFnAttrs.push_back(std::make_pair(
        AttributeList::FirstArgIndex, Attribute::get(*C, Attribute::ZExt)));
    MaybeWarningFnAttrs.push_back(std::make_pair(
        AttributeList::FirstArgIndex + 1, Attribute::get(*C, Attribute::ZExt)));
    MaybeWarningFn[AccessSizeIndex] = M.getOrInsertFunction(
        FunctionName, AttributeList::get(*C, MaybeWarningFnAttrs),
        IRB.getVoidTy(), IRB.getIntNTy(AccessSize * 8), IRB.getInt32Ty());

    FunctionName = "__msan_maybe_store_origin_" + itostr(AccessSize);
    SmallVector<std::pair<unsigned, Attribute>, 2> MaybeStoreOriginFnAttrs;
    MaybeStoreOriginFnAttrs.push_back(std::make_pair(
        AttributeList::FirstArgIndex, Attribute::get(*C, Attribute::ZExt)));
    MaybeStoreOriginFnAttrs.push_back(std::make_pair(
        AttributeList::FirstArgIndex + 2, Attribute::get(*C, Attribute::ZExt)));
    MaybeStoreOriginFn[AccessSizeIndex] = M.getOrInsertFunction(
        FunctionName, AttributeList::get(*C, MaybeStoreOriginFnAttrs),
        IRB.getVoidTy(), IRB.getIntNTy(AccessSize * 8), IRB.getInt8PtrTy(),
        IRB.getInt32Ty());
  }

  MsanSetAllocaOrigin4Fn = M.getOrInsertFunction(
    "__msan_set_alloca_origin4", IRB.getVoidTy(), IRB.getInt8PtrTy(), IntptrTy,
    IRB.getInt8PtrTy(), IntptrTy);
  MsanPoisonStackFn =
      M.getOrInsertFunction("__msan_poison_stack", IRB.getVoidTy(),
                            IRB.getInt8PtrTy(), IntptrTy);
}

/// Insert extern declaration of runtime-provided functions and globals.
void MemorySanitizer::initializeCallbacks(Module &M) {
  // Only do this once.
  if (CallbacksInitialized)
    return;

  IRBuilder<> IRB(*C);
  // Initialize callbacks that are common for kernel and userspace
  // instrumentation.
  MsanChainOriginFn = M.getOrInsertFunction(
    "__msan_chain_origin", IRB.getInt32Ty(), IRB.getInt32Ty());
  MsanSetOriginFn =
      M.getOrInsertFunction("__msan_set_origin", IRB.getVoidTy(),
                            IRB.getInt8PtrTy(), IntptrTy, IRB.getInt32Ty());
  MemmoveFn = M.getOrInsertFunction(
    "__msan_memmove", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
    IRB.getInt8PtrTy(), IntptrTy);
  MemcpyFn = M.getOrInsertFunction(
    "__msan_memcpy", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
    IntptrTy);
  MemsetFn = M.getOrInsertFunction(
    "__msan_memset", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IRB.getInt32Ty(),
    IntptrTy);

  MsanInstrumentAsmStoreFn =
      M.getOrInsertFunction("__msan_instrument_asm_store", IRB.getVoidTy(),
                            PointerType::get(IRB.getInt8Ty(), 0), IntptrTy);

  if (CompileKernel) {
    createKernelApi(M);
  } else {
    createUserspaceApi(M);
  }
  CallbacksInitialized = true;
}

FunctionCallee MemorySanitizer::getKmsanShadowOriginAccessFn(bool isStore,
                                                             int size) {
  FunctionCallee *Fns =
      isStore ? MsanMetadataPtrForStore_1_8 : MsanMetadataPtrForLoad_1_8;
  switch (size) {
  case 1:
    return Fns[0];
  case 2:
    return Fns[1];
  case 4:
    return Fns[2];
  case 8:
    return Fns[3];
  default:
    return nullptr;
  }
}

/// Module-level initialization.
///
/// inserts a call to __msan_init to the module's constructor list.
void MemorySanitizer::initializeModule(Module &M) {
  auto &DL = M.getDataLayout();

  bool ShadowPassed = ClShadowBase.getNumOccurrences() > 0;
  bool OriginPassed = ClOriginBase.getNumOccurrences() > 0;
  // Check the overrides first
  if (ShadowPassed || OriginPassed) {
    CustomMapParams.AndMask = ClAndMask;
    CustomMapParams.XorMask = ClXorMask;
    CustomMapParams.ShadowBase = ClShadowBase;
    CustomMapParams.OriginBase = ClOriginBase;
    MapParams = &CustomMapParams;
  } else {
    Triple TargetTriple(M.getTargetTriple());
    switch (TargetTriple.getOS()) {
      case Triple::FreeBSD:
        switch (TargetTriple.getArch()) {
          case Triple::x86_64:
            MapParams = FreeBSD_X86_MemoryMapParams.bits64;
            break;
          case Triple::x86:
            MapParams = FreeBSD_X86_MemoryMapParams.bits32;
            break;
          default:
            report_fatal_error("unsupported architecture");
        }
        break;
      case Triple::NetBSD:
        switch (TargetTriple.getArch()) {
          case Triple::x86_64:
            MapParams = NetBSD_X86_MemoryMapParams.bits64;
            break;
          default:
            report_fatal_error("unsupported architecture");
        }
        break;
      case Triple::Linux:
        switch (TargetTriple.getArch()) {
          case Triple::x86_64:
            MapParams = Linux_X86_MemoryMapParams.bits64;
            break;
          case Triple::x86:
            MapParams = Linux_X86_MemoryMapParams.bits32;
            break;
          case Triple::mips64:
          case Triple::mips64el:
            MapParams = Linux_MIPS_MemoryMapParams.bits64;
            break;
          case Triple::ppc64:
          case Triple::ppc64le:
            MapParams = Linux_PowerPC_MemoryMapParams.bits64;
            break;
          case Triple::systemz:
            MapParams = Linux_S390_MemoryMapParams.bits64;
            break;
          case Triple::aarch64:
          case Triple::aarch64_be:
            MapParams = Linux_ARM_MemoryMapParams.bits64;
            break;
          default:
            report_fatal_error("unsupported architecture");
        }
        break;
      default:
        report_fatal_error("unsupported operating system");
    }
  }

  C = &(M.getContext());
  IRBuilder<> IRB(*C);
  IntptrTy = IRB.getIntPtrTy(DL);
  OriginTy = IRB.getInt32Ty();

  ColdCallWeights = MDBuilder(*C).createBranchWeights(1, 1000);
  OriginStoreWeights = MDBuilder(*C).createBranchWeights(1, 1000);

  if (!CompileKernel) {
    if (TrackOrigins)
      M.getOrInsertGlobal("__msan_track_origins", IRB.getInt32Ty(), [&] {
        return new GlobalVariable(
            M, IRB.getInt32Ty(), true, GlobalValue::WeakODRLinkage,
            IRB.getInt32(TrackOrigins), "__msan_track_origins");
      });

    if (Recover)
      M.getOrInsertGlobal("__msan_keep_going", IRB.getInt32Ty(), [&] {
        return new GlobalVariable(M, IRB.getInt32Ty(), true,
                                  GlobalValue::WeakODRLinkage,
                                  IRB.getInt32(Recover), "__msan_keep_going");
      });
}
}

bool MemorySanitizerLegacyPass::doInitialization(Module &M) {
  if (!Options.Kernel)
    insertModuleCtor(M);
  MSan.emplace(M, Options);
  return true;
}

namespace {

/// A helper class that handles instrumentation of VarArg
/// functions on a particular platform.
///
/// Implementations are expected to insert the instrumentation
/// necessary to propagate argument shadow through VarArg function
/// calls. Visit* methods are called during an InstVisitor pass over
/// the function, and should avoid creating new basic blocks. A new
/// instance of this class is created for each instrumented function.
struct VarArgHelper {
  virtual ~VarArgHelper() = default;

  /// Visit a CallBase.
  virtual void visitCallBase(CallBase &CB, IRBuilder<> &IRB) = 0;

  /// Visit a va_start call.
  virtual void visitVAStartInst(VAStartInst &I) = 0;

  /// Visit a va_copy call.
  virtual void visitVACopyInst(VACopyInst &I) = 0;

  /// Finalize function instrumentation.
  ///
  /// This method is called after visiting all interesting (see above)
  /// instructions in a function.
  virtual void finalizeInstrumentation() = 0;
};

struct MemorySanitizerVisitor;

} // end anonymous namespace

static VarArgHelper *CreateVarArgHelper(Function &Func, MemorySanitizer &Msan,
                                        MemorySanitizerVisitor &Visitor);

static unsigned TypeSizeToSizeIndex(unsigned TypeSize) {
  if (TypeSize <= 8) return 0;
  return Log2_32_Ceil((TypeSize + 7) / 8);
}

namespace {

/// This class does all the work for a given function. Store and Load
/// instructions store and load corresponding shadow and origin
/// values. Most instructions propagate shadow from arguments to their
/// return values. Certain instructions (most importantly, BranchInst)
/// test their argument shadow and print reports (with a runtime call) if it's
/// non-zero.
struct MemorySanitizerVisitor : public InstVisitor<MemorySanitizerVisitor> {
  Function &F;
  MemorySanitizer &MS;
  SmallVector<PHINode *, 16> ShadowPHINodes, OriginPHINodes;
  ValueMap<Value*, Value*> ShadowMap, OriginMap;
  std::unique_ptr<VarArgHelper> VAHelper;
  const TargetLibraryInfo *TLI;
  Instruction *FnPrologueEnd;

  // The following flags disable parts of MSan instrumentation based on
  // exclusion list contents and command-line options.
  bool InsertChecks;
  bool PropagateShadow;
  bool PoisonStack;
  bool PoisonUndef;

  struct ShadowOriginAndInsertPoint {
    Value *Shadow;
    Value *Origin;
    Instruction *OrigIns;

    ShadowOriginAndInsertPoint(Value *S, Value *O, Instruction *I)
      : Shadow(S), Origin(O), OrigIns(I) {}
  };
  SmallVector<ShadowOriginAndInsertPoint, 16> InstrumentationList;
  bool InstrumentLifetimeStart = ClHandleLifetimeIntrinsics;
  SmallSet<AllocaInst *, 16> AllocaSet;
  SmallVector<std::pair<IntrinsicInst *, AllocaInst *>, 16> LifetimeStartList;
  SmallVector<StoreInst *, 16> StoreList;

  MemorySanitizerVisitor(Function &F, MemorySanitizer &MS,
                         const TargetLibraryInfo &TLI)
      : F(F), MS(MS), VAHelper(CreateVarArgHelper(F, MS, *this)), TLI(&TLI) {
    bool SanitizeFunction =
        F.hasFnAttribute(Attribute::SanitizeMemory) && !ClDisableChecks;
    InsertChecks = SanitizeFunction;
    PropagateShadow = SanitizeFunction;
    PoisonStack = SanitizeFunction && ClPoisonStack;
    PoisonUndef = SanitizeFunction && ClPoisonUndef;

    // In the presence of unreachable blocks, we may see Phi nodes with
    // incoming nodes from such blocks. Since InstVisitor skips unreachable
    // blocks, such nodes will not have any shadow value associated with them.
    // It's easier to remove unreachable blocks than deal with missing shadow.
    removeUnreachableBlocks(F);

    MS.initializeCallbacks(*F.getParent());
    FnPrologueEnd = IRBuilder<>(F.getEntryBlock().getFirstNonPHI())
                        .CreateIntrinsic(Intrinsic::donothing, {}, {});

    if (MS.CompileKernel) {
      IRBuilder<> IRB(FnPrologueEnd);
      insertKmsanPrologue(IRB);
    }

    LLVM_DEBUG(if (!InsertChecks) dbgs()
               << "MemorySanitizer is not inserting checks into '"
               << F.getName() << "'\n");
  }

  bool isInPrologue(Instruction &I) {
    return I.getParent() == FnPrologueEnd->getParent() &&
           (&I == FnPrologueEnd || I.comesBefore(FnPrologueEnd));
  }

  Value *updateOrigin(Value *V, IRBuilder<> &IRB) {
    if (MS.TrackOrigins <= 1) return V;
    return IRB.CreateCall(MS.MsanChainOriginFn, V);
  }

  Value *originToIntptr(IRBuilder<> &IRB, Value *Origin) {
    const DataLayout &DL = F.getParent()->getDataLayout();
    unsigned IntptrSize = DL.getTypeStoreSize(MS.IntptrTy);
    if (IntptrSize == kOriginSize) return Origin;
    assert(IntptrSize == kOriginSize * 2);
    Origin = IRB.CreateIntCast(Origin, MS.IntptrTy, /* isSigned */ false);
    return IRB.CreateOr(Origin, IRB.CreateShl(Origin, kOriginSize * 8));
  }

  /// Fill memory range with the given origin value.
  void paintOrigin(IRBuilder<> &IRB, Value *Origin, Value *OriginPtr,
                   unsigned Size, Align Alignment) {
    const DataLayout &DL = F.getParent()->getDataLayout();
    const Align IntptrAlignment = DL.getABITypeAlign(MS.IntptrTy);
    unsigned IntptrSize = DL.getTypeStoreSize(MS.IntptrTy);
    assert(IntptrAlignment >= kMinOriginAlignment);
    assert(IntptrSize >= kOriginSize);

    unsigned Ofs = 0;
    Align CurrentAlignment = Alignment;
    if (Alignment >= IntptrAlignment && IntptrSize > kOriginSize) {
      Value *IntptrOrigin = originToIntptr(IRB, Origin);
      Value *IntptrOriginPtr =
          IRB.CreatePointerCast(OriginPtr, PointerType::get(MS.IntptrTy, 0));
      for (unsigned i = 0; i < Size / IntptrSize; ++i) {
        Value *Ptr = i ? IRB.CreateConstGEP1_32(MS.IntptrTy, IntptrOriginPtr, i)
                       : IntptrOriginPtr;
        IRB.CreateAlignedStore(IntptrOrigin, Ptr, CurrentAlignment);
        Ofs += IntptrSize / kOriginSize;
        CurrentAlignment = IntptrAlignment;
      }
    }

    for (unsigned i = Ofs; i < (Size + kOriginSize - 1) / kOriginSize; ++i) {
      Value *GEP =
          i ? IRB.CreateConstGEP1_32(MS.OriginTy, OriginPtr, i) : OriginPtr;
      IRB.CreateAlignedStore(Origin, GEP, CurrentAlignment);
      CurrentAlignment = kMinOriginAlignment;
    }
  }

  void storeOrigin(IRBuilder<> &IRB, Value *Addr, Value *Shadow, Value *Origin,
                   Value *OriginPtr, Align Alignment, bool AsCall) {
    const DataLayout &DL = F.getParent()->getDataLayout();
    const Align OriginAlignment = std::max(kMinOriginAlignment, Alignment);
    unsigned StoreSize = DL.getTypeStoreSize(Shadow->getType());
    Value *ConvertedShadow = convertShadowToScalar(Shadow, IRB);
    if (auto *ConstantShadow = dyn_cast<Constant>(ConvertedShadow)) {
      if (ClCheckConstantShadow && !ConstantShadow->isZeroValue())
        paintOrigin(IRB, updateOrigin(Origin, IRB), OriginPtr, StoreSize,
                    OriginAlignment);
      return;
    }

    unsigned TypeSizeInBits = DL.getTypeSizeInBits(ConvertedShadow->getType());
    unsigned SizeIndex = TypeSizeToSizeIndex(TypeSizeInBits);
    if (AsCall && SizeIndex < kNumberOfAccessSizes && !MS.CompileKernel) {
      FunctionCallee Fn = MS.MaybeStoreOriginFn[SizeIndex];
      Value *ConvertedShadow2 =
          IRB.CreateZExt(ConvertedShadow, IRB.getIntNTy(8 * (1 << SizeIndex)));
      CallBase *CB = IRB.CreateCall(
          Fn, {ConvertedShadow2,
               IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()), Origin});
      CB->addParamAttr(0, Attribute::ZExt);
      CB->addParamAttr(2, Attribute::ZExt);
    } else {
      Value *Cmp = convertToBool(ConvertedShadow, IRB, "_mscmp");
      Instruction *CheckTerm = SplitBlockAndInsertIfThen(
          Cmp, &*IRB.GetInsertPoint(), false, MS.OriginStoreWeights);
      IRBuilder<> IRBNew(CheckTerm);
      paintOrigin(IRBNew, updateOrigin(Origin, IRBNew), OriginPtr, StoreSize,
                  OriginAlignment);
    }
  }

  void materializeStores(bool InstrumentWithCalls) {
    for (StoreInst *SI : StoreList) {
      IRBuilder<> IRB(SI);
      Value *Val = SI->getValueOperand();
      Value *Addr = SI->getPointerOperand();
      Value *Shadow = SI->isAtomic() ? getCleanShadow(Val) : getShadow(Val);
      Value *ShadowPtr, *OriginPtr;
      Type *ShadowTy = Shadow->getType();
      const Align Alignment = SI->getAlign();
      const Align OriginAlignment = std::max(kMinOriginAlignment, Alignment);
      std::tie(ShadowPtr, OriginPtr) =
          getShadowOriginPtr(Addr, IRB, ShadowTy, Alignment, /*isStore*/ true);

      StoreInst *NewSI = IRB.CreateAlignedStore(Shadow, ShadowPtr, Alignment);
      LLVM_DEBUG(dbgs() << "  STORE: " << *NewSI << "\n");
      (void)NewSI;

      if (SI->isAtomic())
        SI->setOrdering(addReleaseOrdering(SI->getOrdering()));

      if (MS.TrackOrigins && !SI->isAtomic())
        storeOrigin(IRB, Addr, Shadow, getOrigin(Val), OriginPtr,
                    OriginAlignment, InstrumentWithCalls);
    }
  }

  /// Helper function to insert a warning at IRB's current insert point.
  void insertWarningFn(IRBuilder<> &IRB, Value *Origin) {
    if (!Origin)
      Origin = (Value *)IRB.getInt32(0);
    assert(Origin->getType()->isIntegerTy());
    IRB.CreateCall(MS.WarningFn, Origin)->setCannotMerge();
    // FIXME: Insert UnreachableInst if !MS.Recover?
    // This may invalidate some of the following checks and needs to be done
    // at the very end.
  }

  void materializeOneCheck(Instruction *OrigIns, Value *Shadow, Value *Origin,
                           bool AsCall) {
    IRBuilder<> IRB(OrigIns);
    LLVM_DEBUG(dbgs() << "  SHAD0 : " << *Shadow << "\n");
    Value *ConvertedShadow = convertShadowToScalar(Shadow, IRB);
    LLVM_DEBUG(dbgs() << "  SHAD1 : " << *ConvertedShadow << "\n");

    if (auto *ConstantShadow = dyn_cast<Constant>(ConvertedShadow)) {
      if (ClCheckConstantShadow && !ConstantShadow->isZeroValue()) {
        insertWarningFn(IRB, Origin);
      }
      return;
    }

    const DataLayout &DL = OrigIns->getModule()->getDataLayout();

    unsigned TypeSizeInBits = DL.getTypeSizeInBits(ConvertedShadow->getType());
    unsigned SizeIndex = TypeSizeToSizeIndex(TypeSizeInBits);
    if (AsCall && SizeIndex < kNumberOfAccessSizes && !MS.CompileKernel) {
      FunctionCallee Fn = MS.MaybeWarningFn[SizeIndex];
      Value *ConvertedShadow2 =
          IRB.CreateZExt(ConvertedShadow, IRB.getIntNTy(8 * (1 << SizeIndex)));
      CallBase *CB = IRB.CreateCall(
          Fn, {ConvertedShadow2,
               MS.TrackOrigins && Origin ? Origin : (Value *)IRB.getInt32(0)});
      CB->addParamAttr(0, Attribute::ZExt);
      CB->addParamAttr(1, Attribute::ZExt);
    } else {
      Value *Cmp = convertToBool(ConvertedShadow, IRB, "_mscmp");
      Instruction *CheckTerm = SplitBlockAndInsertIfThen(
          Cmp, OrigIns,
          /* Unreachable */ !MS.Recover, MS.ColdCallWeights);

      IRB.SetInsertPoint(CheckTerm);
      insertWarningFn(IRB, Origin);
      LLVM_DEBUG(dbgs() << "  CHECK: " << *Cmp << "\n");
    }
  }

  void materializeChecks(bool InstrumentWithCalls) {
    for (const auto &ShadowData : InstrumentationList) {
      Instruction *OrigIns = ShadowData.OrigIns;
      Value *Shadow = ShadowData.Shadow;
      Value *Origin = ShadowData.Origin;
      materializeOneCheck(OrigIns, Shadow, Origin, InstrumentWithCalls);
    }
    LLVM_DEBUG(dbgs() << "DONE:\n" << F);
  }

  // Returns the last instruction in the new prologue
  void insertKmsanPrologue(IRBuilder<> &IRB) {
    Value *ContextState = IRB.CreateCall(MS.MsanGetContextStateFn, {});
    Constant *Zero = IRB.getInt32(0);
    MS.ParamTLS = IRB.CreateGEP(MS.MsanContextStateTy, ContextState,
                                {Zero, IRB.getInt32(0)}, "param_shadow");
    MS.RetvalTLS = IRB.CreateGEP(MS.MsanContextStateTy, ContextState,
                                 {Zero, IRB.getInt32(1)}, "retval_shadow");
    MS.VAArgTLS = IRB.CreateGEP(MS.MsanContextStateTy, ContextState,
                                {Zero, IRB.getInt32(2)}, "va_arg_shadow");
    MS.VAArgOriginTLS = IRB.CreateGEP(MS.MsanContextStateTy, ContextState,
                                      {Zero, IRB.getInt32(3)}, "va_arg_origin");
    MS.VAArgOverflowSizeTLS =
        IRB.CreateGEP(MS.MsanContextStateTy, ContextState,
                      {Zero, IRB.getInt32(4)}, "va_arg_overflow_size");
    MS.ParamOriginTLS = IRB.CreateGEP(MS.MsanContextStateTy, ContextState,
                                      {Zero, IRB.getInt32(5)}, "param_origin");
    MS.RetvalOriginTLS =
        IRB.CreateGEP(MS.MsanContextStateTy, ContextState,
                      {Zero, IRB.getInt32(6)}, "retval_origin");
  }

  /// Add MemorySanitizer instrumentation to a function.
  bool runOnFunction() {
    // Iterate all BBs in depth-first order and create shadow instructions
    // for all instructions (where applicable).
    // For PHI nodes we create dummy shadow PHIs which will be finalized later.
    for (BasicBlock *BB : depth_first(FnPrologueEnd->getParent()))
      visit(*BB);

    // Finalize PHI nodes.
    for (PHINode *PN : ShadowPHINodes) {
      PHINode *PNS = cast<PHINode>(getShadow(PN));
      PHINode *PNO = MS.TrackOrigins ? cast<PHINode>(getOrigin(PN)) : nullptr;
      size_t NumValues = PN->getNumIncomingValues();
      for (size_t v = 0; v < NumValues; v++) {
        PNS->addIncoming(getShadow(PN, v), PN->getIncomingBlock(v));
        if (PNO) PNO->addIncoming(getOrigin(PN, v), PN->getIncomingBlock(v));
      }
    }

    VAHelper->finalizeInstrumentation();

    // Poison llvm.lifetime.start intrinsics, if we haven't fallen back to
    // instrumenting only allocas.
    if (InstrumentLifetimeStart) {
      for (auto Item : LifetimeStartList) {
        instrumentAlloca(*Item.second, Item.first);
        AllocaSet.erase(Item.second);
      }
    }
    // Poison the allocas for which we didn't instrument the corresponding
    // lifetime intrinsics.
    for (AllocaInst *AI : AllocaSet)
      instrumentAlloca(*AI);

    bool InstrumentWithCalls = ClInstrumentationWithCallThreshold >= 0 &&
                               InstrumentationList.size() + StoreList.size() >
                                   (unsigned)ClInstrumentationWithCallThreshold;

    // Insert shadow value checks.
    materializeChecks(InstrumentWithCalls);

    // Delayed instrumentation of StoreInst.
    // This may not add new address checks.
    materializeStores(InstrumentWithCalls);

    return true;
  }

  /// Compute the shadow type that corresponds to a given Value.
  Type *getShadowTy(Value *V) {
    return getShadowTy(V->getType());
  }

  /// Compute the shadow type that corresponds to a given Type.
  Type *getShadowTy(Type *OrigTy) {
    if (!OrigTy->isSized()) {
      return nullptr;
    }
    // For integer type, shadow is the same as the original type.
    // This may return weird-sized types like i1.
    if (IntegerType *IT = dyn_cast<IntegerType>(OrigTy))
      return IT;
    const DataLayout &DL = F.getParent()->getDataLayout();
    if (VectorType *VT = dyn_cast<VectorType>(OrigTy)) {
      uint32_t EltSize = DL.getTypeSizeInBits(VT->getElementType());
      return FixedVectorType::get(IntegerType::get(*MS.C, EltSize),
                                  cast<FixedVectorType>(VT)->getNumElements());
    }
    if (ArrayType *AT = dyn_cast<ArrayType>(OrigTy)) {
      return ArrayType::get(getShadowTy(AT->getElementType()),
                            AT->getNumElements());
    }
    if (StructType *ST = dyn_cast<StructType>(OrigTy)) {
      SmallVector<Type*, 4> Elements;
      for (unsigned i = 0, n = ST->getNumElements(); i < n; i++)
        Elements.push_back(getShadowTy(ST->getElementType(i)));
      StructType *Res = StructType::get(*MS.C, Elements, ST->isPacked());
      LLVM_DEBUG(dbgs() << "getShadowTy: " << *ST << " ===> " << *Res << "\n");
      return Res;
    }
    uint32_t TypeSize = DL.getTypeSizeInBits(OrigTy);
    return IntegerType::get(*MS.C, TypeSize);
  }

  /// Flatten a vector type.
  Type *getShadowTyNoVec(Type *ty) {
    if (VectorType *vt = dyn_cast<VectorType>(ty))
      return IntegerType::get(*MS.C,
                              vt->getPrimitiveSizeInBits().getFixedSize());
    return ty;
  }

  /// Extract combined shadow of struct elements as a bool
  Value *collapseStructShadow(StructType *Struct, Value *Shadow,
                              IRBuilder<> &IRB) {
    Value *FalseVal = IRB.getIntN(/* width */ 1, /* value */ 0);
    Value *Aggregator = FalseVal;

    for (unsigned Idx = 0; Idx < Struct->getNumElements(); Idx++) {
      // Combine by ORing together each element's bool shadow
      Value *ShadowItem = IRB.CreateExtractValue(Shadow, Idx);
      Value *ShadowInner = convertShadowToScalar(ShadowItem, IRB);
      Value *ShadowBool = convertToBool(ShadowInner, IRB);

      if (Aggregator != FalseVal)
        Aggregator = IRB.CreateOr(Aggregator, ShadowBool);
      else
        Aggregator = ShadowBool;
    }

    return Aggregator;
  }

  // Extract combined shadow of array elements
  Value *collapseArrayShadow(ArrayType *Array, Value *Shadow,
                             IRBuilder<> &IRB) {
    if (!Array->getNumElements())
      return IRB.getIntN(/* width */ 1, /* value */ 0);

    Value *FirstItem = IRB.CreateExtractValue(Shadow, 0);
    Value *Aggregator = convertShadowToScalar(FirstItem, IRB);

    for (unsigned Idx = 1; Idx < Array->getNumElements(); Idx++) {
      Value *ShadowItem = IRB.CreateExtractValue(Shadow, Idx);
      Value *ShadowInner = convertShadowToScalar(ShadowItem, IRB);
      Aggregator = IRB.CreateOr(Aggregator, ShadowInner);
    }
    return Aggregator;
  }

  /// Convert a shadow value to it's flattened variant. The resulting
  /// shadow may not necessarily have the same bit width as the input
  /// value, but it will always be comparable to zero.
  Value *convertShadowToScalar(Value *V, IRBuilder<> &IRB) {
    if (StructType *Struct = dyn_cast<StructType>(V->getType()))
      return collapseStructShadow(Struct, V, IRB);
    if (ArrayType *Array = dyn_cast<ArrayType>(V->getType()))
      return collapseArrayShadow(Array, V, IRB);
    Type *Ty = V->getType();
    Type *NoVecTy = getShadowTyNoVec(Ty);
    if (Ty == NoVecTy) return V;
    return IRB.CreateBitCast(V, NoVecTy);
  }

  // Convert a scalar value to an i1 by comparing with 0
  Value *convertToBool(Value *V, IRBuilder<> &IRB, const Twine &name = "") {
    Type *VTy = V->getType();
    assert(VTy->isIntegerTy());
    if (VTy->getIntegerBitWidth() == 1)
      // Just converting a bool to a bool, so do nothing.
      return V;
    return IRB.CreateICmpNE(V, ConstantInt::get(VTy, 0), name);
  }

  /// Compute the integer shadow offset that corresponds to a given
  /// application address.
  ///
  /// Offset = (Addr & ~AndMask) ^ XorMask
  Value *getShadowPtrOffset(Value *Addr, IRBuilder<> &IRB) {
    Value *OffsetLong = IRB.CreatePointerCast(Addr, MS.IntptrTy);

    uint64_t AndMask = MS.MapParams->AndMask;
    if (AndMask)
      OffsetLong =
          IRB.CreateAnd(OffsetLong, ConstantInt::get(MS.IntptrTy, ~AndMask));

    uint64_t XorMask = MS.MapParams->XorMask;
    if (XorMask)
      OffsetLong =
          IRB.CreateXor(OffsetLong, ConstantInt::get(MS.IntptrTy, XorMask));
    return OffsetLong;
  }

  /// Compute the shadow and origin addresses corresponding to a given
  /// application address.
  ///
  /// Shadow = ShadowBase + Offset
  /// Origin = (OriginBase + Offset) & ~3ULL
  std::pair<Value *, Value *>
  getShadowOriginPtrUserspace(Value *Addr, IRBuilder<> &IRB, Type *ShadowTy,
                              MaybeAlign Alignment) {
    Value *ShadowOffset = getShadowPtrOffset(Addr, IRB);
    Value *ShadowLong = ShadowOffset;
    uint64_t ShadowBase = MS.MapParams->ShadowBase;
    if (ShadowBase != 0) {
      ShadowLong =
        IRB.CreateAdd(ShadowLong,
                      ConstantInt::get(MS.IntptrTy, ShadowBase));
    }
    Value *ShadowPtr =
        IRB.CreateIntToPtr(ShadowLong, PointerType::get(ShadowTy, 0));
    Value *OriginPtr = nullptr;
    if (MS.TrackOrigins) {
      Value *OriginLong = ShadowOffset;
      uint64_t OriginBase = MS.MapParams->OriginBase;
      if (OriginBase != 0)
        OriginLong = IRB.CreateAdd(OriginLong,
                                   ConstantInt::get(MS.IntptrTy, OriginBase));
      if (!Alignment || *Alignment < kMinOriginAlignment) {
        uint64_t Mask = kMinOriginAlignment.value() - 1;
        OriginLong =
            IRB.CreateAnd(OriginLong, ConstantInt::get(MS.IntptrTy, ~Mask));
      }
      OriginPtr =
          IRB.CreateIntToPtr(OriginLong, PointerType::get(MS.OriginTy, 0));
    }
    return std::make_pair(ShadowPtr, OriginPtr);
  }

  std::pair<Value *, Value *> getShadowOriginPtrKernel(Value *Addr,
                                                       IRBuilder<> &IRB,
                                                       Type *ShadowTy,
                                                       bool isStore) {
    Value *ShadowOriginPtrs;
    const DataLayout &DL = F.getParent()->getDataLayout();
    int Size = DL.getTypeStoreSize(ShadowTy);

    FunctionCallee Getter = MS.getKmsanShadowOriginAccessFn(isStore, Size);
    Value *AddrCast =
        IRB.CreatePointerCast(Addr, PointerType::get(IRB.getInt8Ty(), 0));
    if (Getter) {
      ShadowOriginPtrs = IRB.CreateCall(Getter, AddrCast);
    } else {
      Value *SizeVal = ConstantInt::get(MS.IntptrTy, Size);
      ShadowOriginPtrs = IRB.CreateCall(isStore ? MS.MsanMetadataPtrForStoreN
                                                : MS.MsanMetadataPtrForLoadN,
                                        {AddrCast, SizeVal});
    }
    Value *ShadowPtr = IRB.CreateExtractValue(ShadowOriginPtrs, 0);
    ShadowPtr = IRB.CreatePointerCast(ShadowPtr, PointerType::get(ShadowTy, 0));
    Value *OriginPtr = IRB.CreateExtractValue(ShadowOriginPtrs, 1);

    return std::make_pair(ShadowPtr, OriginPtr);
  }

  std::pair<Value *, Value *> getShadowOriginPtr(Value *Addr, IRBuilder<> &IRB,
                                                 Type *ShadowTy,
                                                 MaybeAlign Alignment,
                                                 bool isStore) {
    if (MS.CompileKernel)
      return getShadowOriginPtrKernel(Addr, IRB, ShadowTy, isStore);
    return getShadowOriginPtrUserspace(Addr, IRB, ShadowTy, Alignment);
  }

  /// Compute the shadow address for a given function argument.
  ///
  /// Shadow = ParamTLS+ArgOffset.
  Value *getShadowPtrForArgument(Value *A, IRBuilder<> &IRB,
                                 int ArgOffset) {
    Value *Base = IRB.CreatePointerCast(MS.ParamTLS, MS.IntptrTy);
    if (ArgOffset)
      Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(getShadowTy(A), 0),
                              "_msarg");
  }

  /// Compute the origin address for a given function argument.
  Value *getOriginPtrForArgument(Value *A, IRBuilder<> &IRB,
                                 int ArgOffset) {
    if (!MS.TrackOrigins)
      return nullptr;
    Value *Base = IRB.CreatePointerCast(MS.ParamOriginTLS, MS.IntptrTy);
    if (ArgOffset)
      Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(MS.OriginTy, 0),
                              "_msarg_o");
  }

  /// Compute the shadow address for a retval.
  Value *getShadowPtrForRetval(Value *A, IRBuilder<> &IRB) {
    return IRB.CreatePointerCast(MS.RetvalTLS,
                                 PointerType::get(getShadowTy(A), 0),
                                 "_msret");
  }

  /// Compute the origin address for a retval.
  Value *getOriginPtrForRetval(IRBuilder<> &IRB) {
    // We keep a single origin for the entire retval. Might be too optimistic.
    return MS.RetvalOriginTLS;
  }

  /// Set SV to be the shadow value for V.
  void setShadow(Value *V, Value *SV) {
    assert(!ShadowMap.count(V) && "Values may only have one shadow");
    ShadowMap[V] = PropagateShadow ? SV : getCleanShadow(V);
  }

  /// Set Origin to be the origin value for V.
  void setOrigin(Value *V, Value *Origin) {
    if (!MS.TrackOrigins) return;
    assert(!OriginMap.count(V) && "Values may only have one origin");
    LLVM_DEBUG(dbgs() << "ORIGIN: " << *V << "  ==> " << *Origin << "\n");
    OriginMap[V] = Origin;
  }

  Constant *getCleanShadow(Type *OrigTy) {
    Type *ShadowTy = getShadowTy(OrigTy);
    if (!ShadowTy)
      return nullptr;
    return Constant::getNullValue(ShadowTy);
  }

  /// Create a clean shadow value for a given value.
  ///
  /// Clean shadow (all zeroes) means all bits of the value are defined
  /// (initialized).
  Constant *getCleanShadow(Value *V) {
    return getCleanShadow(V->getType());
  }

  /// Create a dirty shadow of a given shadow type.
  Constant *getPoisonedShadow(Type *ShadowTy) {
    assert(ShadowTy);
    if (isa<IntegerType>(ShadowTy) || isa<VectorType>(ShadowTy))
      return Constant::getAllOnesValue(ShadowTy);
    if (ArrayType *AT = dyn_cast<ArrayType>(ShadowTy)) {
      SmallVector<Constant *, 4> Vals(AT->getNumElements(),
                                      getPoisonedShadow(AT->getElementType()));
      return ConstantArray::get(AT, Vals);
    }
    if (StructType *ST = dyn_cast<StructType>(ShadowTy)) {
      SmallVector<Constant *, 4> Vals;
      for (unsigned i = 0, n = ST->getNumElements(); i < n; i++)
        Vals.push_back(getPoisonedShadow(ST->getElementType(i)));
      return ConstantStruct::get(ST, Vals);
    }
    llvm_unreachable("Unexpected shadow type");
  }

  /// Create a dirty shadow for a given value.
  Constant *getPoisonedShadow(Value *V) {
    Type *ShadowTy = getShadowTy(V);
    if (!ShadowTy)
      return nullptr;
    return getPoisonedShadow(ShadowTy);
  }

  /// Create a clean (zero) origin.
  Value *getCleanOrigin() {
    return Constant::getNullValue(MS.OriginTy);
  }

  /// Get the shadow value for a given Value.
  ///
  /// This function either returns the value set earlier with setShadow,
  /// or extracts if from ParamTLS (for function arguments).
  Value *getShadow(Value *V) {
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      if (!PropagateShadow || I->getMetadata("nosanitize"))
        return getCleanShadow(V);
      // For instructions the shadow is already stored in the map.
      Value *Shadow = ShadowMap[V];
      if (!Shadow) {
        LLVM_DEBUG(dbgs() << "No shadow: " << *V << "\n" << *(I->getParent()));
        (void)I;
        assert(Shadow && "No shadow for a value");
      }
      return Shadow;
    }
    if (UndefValue *U = dyn_cast<UndefValue>(V)) {
      Value *AllOnes = (PropagateShadow && PoisonUndef) ? getPoisonedShadow(V)
                                                        : getCleanShadow(V);
      LLVM_DEBUG(dbgs() << "Undef: " << *U << " ==> " << *AllOnes << "\n");
      (void)U;
      return AllOnes;
    }
    if (Argument *A = dyn_cast<Argument>(V)) {
      // For arguments we compute the shadow on demand and store it in the map.
      Value **ShadowPtr = &ShadowMap[V];
      if (*ShadowPtr)
        return *ShadowPtr;
      Function *F = A->getParent();
      IRBuilder<> EntryIRB(FnPrologueEnd);
      unsigned ArgOffset = 0;
      const DataLayout &DL = F->getParent()->getDataLayout();
      for (auto &FArg : F->args()) {
        if (!FArg.getType()->isSized()) {
          LLVM_DEBUG(dbgs() << "Arg is not sized\n");
          continue;
        }

        unsigned Size = FArg.hasByValAttr()
                            ? DL.getTypeAllocSize(FArg.getParamByValType())
                            : DL.getTypeAllocSize(FArg.getType());

        if (A == &FArg) {
          bool Overflow = ArgOffset + Size > kParamTLSSize;
          if (FArg.hasByValAttr()) {
            // ByVal pointer itself has clean shadow. We copy the actual
            // argument shadow to the underlying memory.
            // Figure out maximal valid memcpy alignment.
            const Align ArgAlign = DL.getValueOrABITypeAlignment(
                MaybeAlign(FArg.getParamAlignment()), FArg.getParamByValType());
            Value *CpShadowPtr, *CpOriginPtr;
            std::tie(CpShadowPtr, CpOriginPtr) =
                getShadowOriginPtr(V, EntryIRB, EntryIRB.getInt8Ty(), ArgAlign,
                                   /*isStore*/ true);
            if (!PropagateShadow || Overflow) {
              // ParamTLS overflow.
              EntryIRB.CreateMemSet(
                  CpShadowPtr, Constant::getNullValue(EntryIRB.getInt8Ty()),
                  Size, ArgAlign);
            } else {
              Value *Base = getShadowPtrForArgument(&FArg, EntryIRB, ArgOffset);
              const Align CopyAlign = std::min(ArgAlign, kShadowTLSAlignment);
              Value *Cpy = EntryIRB.CreateMemCpy(CpShadowPtr, CopyAlign, Base,
                                                 CopyAlign, Size);
              LLVM_DEBUG(dbgs() << "  ByValCpy: " << *Cpy << "\n");
              (void)Cpy;

              if (MS.TrackOrigins) {
                Value *OriginPtr =
                    getOriginPtrForArgument(&FArg, EntryIRB, ArgOffset);
                // FIXME: OriginSize should be:
                // alignTo(V % kMinOriginAlignment + Size, kMinOriginAlignment)
                unsigned OriginSize = alignTo(Size, kMinOriginAlignment);
                EntryIRB.CreateMemCpy(
                    CpOriginPtr,
                    /* by getShadowOriginPtr */ kMinOriginAlignment, OriginPtr,
                    /* by origin_tls[ArgOffset] */ kMinOriginAlignment,
                    OriginSize);
              }
            }
          }

          if (!PropagateShadow || Overflow || FArg.hasByValAttr() ||
              (MS.EagerChecks && FArg.hasAttribute(Attribute::NoUndef))) {
            *ShadowPtr = getCleanShadow(V);
            setOrigin(A, getCleanOrigin());
          } else {
            // Shadow over TLS
            Value *Base = getShadowPtrForArgument(&FArg, EntryIRB, ArgOffset);
            *ShadowPtr = EntryIRB.CreateAlignedLoad(getShadowTy(&FArg), Base,
                                                    kShadowTLSAlignment);
            if (MS.TrackOrigins) {
              Value *OriginPtr =
                  getOriginPtrForArgument(&FArg, EntryIRB, ArgOffset);
              setOrigin(A, EntryIRB.CreateLoad(MS.OriginTy, OriginPtr));
            }
          }
          LLVM_DEBUG(dbgs()
                     << "  ARG:    " << FArg << " ==> " << **ShadowPtr << "\n");
          break;
        }

        ArgOffset += alignTo(Size, kShadowTLSAlignment);
      }
      assert(*ShadowPtr && "Could not find shadow for an argument");
      return *ShadowPtr;
    }
    // For everything else the shadow is zero.
    return getCleanShadow(V);
  }

  /// Get the shadow for i-th argument of the instruction I.
  Value *getShadow(Instruction *I, int i) {
    return getShadow(I->getOperand(i));
  }

  /// Get the origin for a value.
  Value *getOrigin(Value *V) {
    if (!MS.TrackOrigins) return nullptr;
    if (!PropagateShadow) return getCleanOrigin();
    if (isa<Constant>(V)) return getCleanOrigin();
    assert((isa<Instruction>(V) || isa<Argument>(V)) &&
           "Unexpected value type in getOrigin()");
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      if (I->getMetadata("nosanitize"))
        return getCleanOrigin();
    }
    Value *Origin = OriginMap[V];
    assert(Origin && "Missing origin");
    return Origin;
  }

  /// Get the origin for i-th argument of the instruction I.
  Value *getOrigin(Instruction *I, int i) {
    return getOrigin(I->getOperand(i));
  }

  /// Remember the place where a shadow check should be inserted.
  ///
  /// This location will be later instrumented with a check that will print a
  /// UMR warning in runtime if the shadow value is not 0.
  void insertShadowCheck(Value *Shadow, Value *Origin, Instruction *OrigIns) {
    assert(Shadow);
    if (!InsertChecks) return;
#ifndef NDEBUG
    Type *ShadowTy = Shadow->getType();
    assert((isa<IntegerType>(ShadowTy) || isa<VectorType>(ShadowTy) ||
            isa<StructType>(ShadowTy) || isa<ArrayType>(ShadowTy)) &&
           "Can only insert checks for integer, vector, and aggregate shadow "
           "types");
#endif
    InstrumentationList.push_back(
        ShadowOriginAndInsertPoint(Shadow, Origin, OrigIns));
  }

  /// Remember the place where a shadow check should be inserted.
  ///
  /// This location will be later instrumented with a check that will print a
  /// UMR warning in runtime if the value is not fully defined.
  void insertShadowCheck(Value *Val, Instruction *OrigIns) {
    assert(Val);
    Value *Shadow, *Origin;
    if (ClCheckConstantShadow) {
      Shadow = getShadow(Val);
      if (!Shadow) return;
      Origin = getOrigin(Val);
    } else {
      Shadow = dyn_cast_or_null<Instruction>(getShadow(Val));
      if (!Shadow) return;
      Origin = dyn_cast_or_null<Instruction>(getOrigin(Val));
    }
    insertShadowCheck(Shadow, Origin, OrigIns);
  }

  AtomicOrdering addReleaseOrdering(AtomicOrdering a) {
    switch (a) {
      case AtomicOrdering::NotAtomic:
        return AtomicOrdering::NotAtomic;
      case AtomicOrdering::Unordered:
      case AtomicOrdering::Monotonic:
      case AtomicOrdering::Release:
        return AtomicOrdering::Release;
      case AtomicOrdering::Acquire:
      case AtomicOrdering::AcquireRelease:
        return AtomicOrdering::AcquireRelease;
      case AtomicOrdering::SequentiallyConsistent:
        return AtomicOrdering::SequentiallyConsistent;
    }
    llvm_unreachable("Unknown ordering");
  }

  Value *makeAddReleaseOrderingTable(IRBuilder<> &IRB) {
    constexpr int NumOrderings = (int)AtomicOrderingCABI::seq_cst + 1;
    uint32_t OrderingTable[NumOrderings] = {};

    OrderingTable[(int)AtomicOrderingCABI::relaxed] =
        OrderingTable[(int)AtomicOrderingCABI::release] =
            (int)AtomicOrderingCABI::release;
    OrderingTable[(int)AtomicOrderingCABI::consume] =
        OrderingTable[(int)AtomicOrderingCABI::acquire] =
            OrderingTable[(int)AtomicOrderingCABI::acq_rel] =
                (int)AtomicOrderingCABI::acq_rel;
    OrderingTable[(int)AtomicOrderingCABI::seq_cst] =
        (int)AtomicOrderingCABI::seq_cst;

    return ConstantDataVector::get(IRB.getContext(),
                                   makeArrayRef(OrderingTable, NumOrderings));
  }

  AtomicOrdering addAcquireOrdering(AtomicOrdering a) {
    switch (a) {
      case AtomicOrdering::NotAtomic:
        return AtomicOrdering::NotAtomic;
      case AtomicOrdering::Unordered:
      case AtomicOrdering::Monotonic:
      case AtomicOrdering::Acquire:
        return AtomicOrdering::Acquire;
      case AtomicOrdering::Release:
      case AtomicOrdering::AcquireRelease:
        return AtomicOrdering::AcquireRelease;
      case AtomicOrdering::SequentiallyConsistent:
        return AtomicOrdering::SequentiallyConsistent;
    }
    llvm_unreachable("Unknown ordering");
  }

  Value *makeAddAcquireOrderingTable(IRBuilder<> &IRB) {
    constexpr int NumOrderings = (int)AtomicOrderingCABI::seq_cst + 1;
    uint32_t OrderingTable[NumOrderings] = {};

    OrderingTable[(int)AtomicOrderingCABI::relaxed] =
        OrderingTable[(int)AtomicOrderingCABI::acquire] =
            OrderingTable[(int)AtomicOrderingCABI::consume] =
                (int)AtomicOrderingCABI::acquire;
    OrderingTable[(int)AtomicOrderingCABI::release] =
        OrderingTable[(int)AtomicOrderingCABI::acq_rel] =
            (int)AtomicOrderingCABI::acq_rel;
    OrderingTable[(int)AtomicOrderingCABI::seq_cst] =
        (int)AtomicOrderingCABI::seq_cst;

    return ConstantDataVector::get(IRB.getContext(),
                                   makeArrayRef(OrderingTable, NumOrderings));
  }

  // ------------------- Visitors.
  using InstVisitor<MemorySanitizerVisitor>::visit;
  void visit(Instruction &I) {
    if (I.getMetadata("nosanitize"))
      return;
    // Don't want to visit if we're in the prologue
    if (isInPrologue(I))
      return;
    InstVisitor<MemorySanitizerVisitor>::visit(I);
  }

  /// Instrument LoadInst
  ///
  /// Loads the corresponding shadow and (optionally) origin.
  /// Optionally, checks that the load address is fully defined.
  void visitLoadInst(LoadInst &I) {
    assert(I.getType()->isSized() && "Load type must have size");
    assert(!I.getMetadata("nosanitize"));
    IRBuilder<> IRB(I.getNextNode());
    Type *ShadowTy = getShadowTy(&I);
    Value *Addr = I.getPointerOperand();
    Value *ShadowPtr = nullptr, *OriginPtr = nullptr;
    const Align Alignment = assumeAligned(I.getAlignment());
    if (PropagateShadow) {
      std::tie(ShadowPtr, OriginPtr) =
          getShadowOriginPtr(Addr, IRB, ShadowTy, Alignment, /*isStore*/ false);
      setShadow(&I,
                IRB.CreateAlignedLoad(ShadowTy, ShadowPtr, Alignment, "_msld"));
    } else {
      setShadow(&I, getCleanShadow(&I));
    }

    if (ClCheckAccessAddress)
      insertShadowCheck(I.getPointerOperand(), &I);

    if (I.isAtomic())
      I.setOrdering(addAcquireOrdering(I.getOrdering()));

    if (MS.TrackOrigins) {
      if (PropagateShadow) {
        const Align OriginAlignment = std::max(kMinOriginAlignment, Alignment);
        setOrigin(
            &I, IRB.CreateAlignedLoad(MS.OriginTy, OriginPtr, OriginAlignment));
      } else {
        setOrigin(&I, getCleanOrigin());
      }
    }
  }

  /// Instrument StoreInst
  ///
  /// Stores the corresponding shadow and (optionally) origin.
  /// Optionally, checks that the store address is fully defined.
  void visitStoreInst(StoreInst &I) {
    StoreList.push_back(&I);
    if (ClCheckAccessAddress)
      insertShadowCheck(I.getPointerOperand(), &I);
  }

  void handleCASOrRMW(Instruction &I) {
    assert(isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I));

    IRBuilder<> IRB(&I);
    Value *Addr = I.getOperand(0);
    Value *Val = I.getOperand(1);
    Value *ShadowPtr = getShadowOriginPtr(Addr, IRB, Val->getType(), Align(1),
                                          /*isStore*/ true)
                           .first;

    if (ClCheckAccessAddress)
      insertShadowCheck(Addr, &I);

    // Only test the conditional argument of cmpxchg instruction.
    // The other argument can potentially be uninitialized, but we can not
    // detect this situation reliably without possible false positives.
    if (isa<AtomicCmpXchgInst>(I))
      insertShadowCheck(Val, &I);

    IRB.CreateStore(getCleanShadow(Val), ShadowPtr);

    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
  }

  void visitAtomicRMWInst(AtomicRMWInst &I) {
    handleCASOrRMW(I);
    I.setOrdering(addReleaseOrdering(I.getOrdering()));
  }

  void visitAtomicCmpXchgInst(AtomicCmpXchgInst &I) {
    handleCASOrRMW(I);
    I.setSuccessOrdering(addReleaseOrdering(I.getSuccessOrdering()));
  }

  // Vector manipulation.
  void visitExtractElementInst(ExtractElementInst &I) {
    insertShadowCheck(I.getOperand(1), &I);
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateExtractElement(getShadow(&I, 0), I.getOperand(1),
              "_msprop"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitInsertElementInst(InsertElementInst &I) {
    insertShadowCheck(I.getOperand(2), &I);
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateInsertElement(getShadow(&I, 0), getShadow(&I, 1),
              I.getOperand(2), "_msprop"));
    setOriginForNaryOp(I);
  }

  void visitShuffleVectorInst(ShuffleVectorInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateShuffleVector(getShadow(&I, 0), getShadow(&I, 1),
                                          I.getShuffleMask(), "_msprop"));
    setOriginForNaryOp(I);
  }

  // Casts.
  void visitSExtInst(SExtInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateSExt(getShadow(&I, 0), I.getType(), "_msprop"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitZExtInst(ZExtInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateZExt(getShadow(&I, 0), I.getType(), "_msprop"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitTruncInst(TruncInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateTrunc(getShadow(&I, 0), I.getType(), "_msprop"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitBitCastInst(BitCastInst &I) {
    // Special case: if this is the bitcast (there is exactly 1 allowed) between
    // a musttail call and a ret, don't instrument. New instructions are not
    // allowed after a musttail call.
    if (auto *CI = dyn_cast<CallInst>(I.getOperand(0)))
      if (CI->isMustTailCall())
        return;
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateBitCast(getShadow(&I, 0), getShadowTy(&I)));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitPtrToIntInst(PtrToIntInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateIntCast(getShadow(&I, 0), getShadowTy(&I), false,
             "_msprop_ptrtoint"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitIntToPtrInst(IntToPtrInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateIntCast(getShadow(&I, 0), getShadowTy(&I), false,
             "_msprop_inttoptr"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitFPToSIInst(CastInst& I) { handleShadowOr(I); }
  void visitFPToUIInst(CastInst& I) { handleShadowOr(I); }
  void visitSIToFPInst(CastInst& I) { handleShadowOr(I); }
  void visitUIToFPInst(CastInst& I) { handleShadowOr(I); }
  void visitFPExtInst(CastInst& I) { handleShadowOr(I); }
  void visitFPTruncInst(CastInst& I) { handleShadowOr(I); }

  /// Propagate shadow for bitwise AND.
  ///
  /// This code is exact, i.e. if, for example, a bit in the left argument
  /// is defined and 0, then neither the value not definedness of the
  /// corresponding bit in B don't affect the resulting shadow.
  void visitAnd(BinaryOperator &I) {
    IRBuilder<> IRB(&I);
    //  "And" of 0 and a poisoned value results in unpoisoned value.
    //  1&1 => 1;     0&1 => 0;     p&1 => p;
    //  1&0 => 0;     0&0 => 0;     p&0 => 0;
    //  1&p => p;     0&p => 0;     p&p => p;
    //  S = (S1 & S2) | (V1 & S2) | (S1 & V2)
    Value *S1 = getShadow(&I, 0);
    Value *S2 = getShadow(&I, 1);
    Value *V1 = I.getOperand(0);
    Value *V2 = I.getOperand(1);
    if (V1->getType() != S1->getType()) {
      V1 = IRB.CreateIntCast(V1, S1->getType(), false);
      V2 = IRB.CreateIntCast(V2, S2->getType(), false);
    }
    Value *S1S2 = IRB.CreateAnd(S1, S2);
    Value *V1S2 = IRB.CreateAnd(V1, S2);
    Value *S1V2 = IRB.CreateAnd(S1, V2);
    setShadow(&I, IRB.CreateOr({S1S2, V1S2, S1V2}));
    setOriginForNaryOp(I);
  }

  void visitOr(BinaryOperator &I) {
    IRBuilder<> IRB(&I);
    //  "Or" of 1 and a poisoned value results in unpoisoned value.
    //  1|1 => 1;     0|1 => 1;     p|1 => 1;
    //  1|0 => 1;     0|0 => 0;     p|0 => p;
    //  1|p => 1;     0|p => p;     p|p => p;
    //  S = (S1 & S2) | (~V1 & S2) | (S1 & ~V2)
    Value *S1 = getShadow(&I, 0);
    Value *S2 = getShadow(&I, 1);
    Value *V1 = IRB.CreateNot(I.getOperand(0));
    Value *V2 = IRB.CreateNot(I.getOperand(1));
    if (V1->getType() != S1->getType()) {
      V1 = IRB.CreateIntCast(V1, S1->getType(), false);
      V2 = IRB.CreateIntCast(V2, S2->getType(), false);
    }
    Value *S1S2 = IRB.CreateAnd(S1, S2);
    Value *V1S2 = IRB.CreateAnd(V1, S2);
    Value *S1V2 = IRB.CreateAnd(S1, V2);
    setShadow(&I, IRB.CreateOr({S1S2, V1S2, S1V2}));
    setOriginForNaryOp(I);
  }

  /// Default propagation of shadow and/or origin.
  ///
  /// This class implements the general case of shadow propagation, used in all
  /// cases where we don't know and/or don't care about what the operation
  /// actually does. It converts all input shadow values to a common type
  /// (extending or truncating as necessary), and bitwise OR's them.
  ///
  /// This is much cheaper than inserting checks (i.e. requiring inputs to be
  /// fully initialized), and less prone to false positives.
  ///
  /// This class also implements the general case of origin propagation. For a
  /// Nary operation, result origin is set to the origin of an argument that is
  /// not entirely initialized. If there is more than one such arguments, the
  /// rightmost of them is picked. It does not matter which one is picked if all
  /// arguments are initialized.
  template <bool CombineShadow>
  class Combiner {
    Value *Shadow = nullptr;
    Value *Origin = nullptr;
    IRBuilder<> &IRB;
    MemorySanitizerVisitor *MSV;

  public:
    Combiner(MemorySanitizerVisitor *MSV, IRBuilder<> &IRB)
        : IRB(IRB), MSV(MSV) {}

    /// Add a pair of shadow and origin values to the mix.
    Combiner &Add(Value *OpShadow, Value *OpOrigin) {
      if (CombineShadow) {
        assert(OpShadow);
        if (!Shadow)
          Shadow = OpShadow;
        else {
          OpShadow = MSV->CreateShadowCast(IRB, OpShadow, Shadow->getType());
          Shadow = IRB.CreateOr(Shadow, OpShadow, "_msprop");
        }
      }

      if (MSV->MS.TrackOrigins) {
        assert(OpOrigin);
        if (!Origin) {
          Origin = OpOrigin;
        } else {
          Constant *ConstOrigin = dyn_cast<Constant>(OpOrigin);
          // No point in adding something that might result in 0 origin value.
          if (!ConstOrigin || !ConstOrigin->isNullValue()) {
            Value *FlatShadow = MSV->convertShadowToScalar(OpShadow, IRB);
            Value *Cond =
                IRB.CreateICmpNE(FlatShadow, MSV->getCleanShadow(FlatShadow));
            Origin = IRB.CreateSelect(Cond, OpOrigin, Origin);
          }
        }
      }
      return *this;
    }

    /// Add an application value to the mix.
    Combiner &Add(Value *V) {
      Value *OpShadow = MSV->getShadow(V);
      Value *OpOrigin = MSV->MS.TrackOrigins ? MSV->getOrigin(V) : nullptr;
      return Add(OpShadow, OpOrigin);
    }

    /// Set the current combined values as the given instruction's shadow
    /// and origin.
    void Done(Instruction *I) {
      if (CombineShadow) {
        assert(Shadow);
        Shadow = MSV->CreateShadowCast(IRB, Shadow, MSV->getShadowTy(I));
        MSV->setShadow(I, Shadow);
      }
      if (MSV->MS.TrackOrigins) {
        assert(Origin);
        MSV->setOrigin(I, Origin);
      }
    }
  };

  using ShadowAndOriginCombiner = Combiner<true>;
  using OriginCombiner = Combiner<false>;

  /// Propagate origin for arbitrary operation.
  void setOriginForNaryOp(Instruction &I) {
    if (!MS.TrackOrigins) return;
    IRBuilder<> IRB(&I);
    OriginCombiner OC(this, IRB);
    for (Use &Op : I.operands())
      OC.Add(Op.get());
    OC.Done(&I);
  }

  size_t VectorOrPrimitiveTypeSizeInBits(Type *Ty) {
    assert(!(Ty->isVectorTy() && Ty->getScalarType()->isPointerTy()) &&
           "Vector of pointers is not a valid shadow type");
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getNumElements() *
                                  Ty->getScalarSizeInBits()
                            : Ty->getPrimitiveSizeInBits();
  }

  /// Cast between two shadow types, extending or truncating as
  /// necessary.
  Value *CreateShadowCast(IRBuilder<> &IRB, Value *V, Type *dstTy,
                          bool Signed = false) {
    Type *srcTy = V->getType();
    size_t srcSizeInBits = VectorOrPrimitiveTypeSizeInBits(srcTy);
    size_t dstSizeInBits = VectorOrPrimitiveTypeSizeInBits(dstTy);
    if (srcSizeInBits > 1 && dstSizeInBits == 1)
      return IRB.CreateICmpNE(V, getCleanShadow(V));

    if (dstTy->isIntegerTy() && srcTy->isIntegerTy())
      return IRB.CreateIntCast(V, dstTy, Signed);
    if (dstTy->isVectorTy() && srcTy->isVectorTy() &&
        cast<FixedVectorType>(dstTy)->getNumElements() ==
            cast<FixedVectorType>(srcTy)->getNumElements())
      return IRB.CreateIntCast(V, dstTy, Signed);
    Value *V1 = IRB.CreateBitCast(V, Type::getIntNTy(*MS.C, srcSizeInBits));
    Value *V2 =
      IRB.CreateIntCast(V1, Type::getIntNTy(*MS.C, dstSizeInBits), Signed);
    return IRB.CreateBitCast(V2, dstTy);
    // TODO: handle struct types.
  }

  /// Cast an application value to the type of its own shadow.
  Value *CreateAppToShadowCast(IRBuilder<> &IRB, Value *V) {
    Type *ShadowTy = getShadowTy(V);
    if (V->getType() == ShadowTy)
      return V;
    if (V->getType()->isPtrOrPtrVectorTy())
      return IRB.CreatePtrToInt(V, ShadowTy);
    else
      return IRB.CreateBitCast(V, ShadowTy);
  }

  /// Propagate shadow for arbitrary operation.
  void handleShadowOr(Instruction &I) {
    IRBuilder<> IRB(&I);
    ShadowAndOriginCombiner SC(this, IRB);
    for (Use &Op : I.operands())
      SC.Add(Op.get());
    SC.Done(&I);
  }

  void visitFNeg(UnaryOperator &I) { handleShadowOr(I); }

  // Handle multiplication by constant.
  //
  // Handle a special case of multiplication by constant that may have one or
  // more zeros in the lower bits. This makes corresponding number of lower bits
  // of the result zero as well. We model it by shifting the other operand
  // shadow left by the required number of bits. Effectively, we transform
  // (X * (A * 2**B)) to ((X << B) * A) and instrument (X << B) as (Sx << B).
  // We use multiplication by 2**N instead of shift to cover the case of
  // multiplication by 0, which may occur in some elements of a vector operand.
  void handleMulByConstant(BinaryOperator &I, Constant *ConstArg,
                           Value *OtherArg) {
    Constant *ShadowMul;
    Type *Ty = ConstArg->getType();
    if (auto *VTy = dyn_cast<VectorType>(Ty)) {
      unsigned NumElements = cast<FixedVectorType>(VTy)->getNumElements();
      Type *EltTy = VTy->getElementType();
      SmallVector<Constant *, 16> Elements;
      for (unsigned Idx = 0; Idx < NumElements; ++Idx) {
        if (ConstantInt *Elt =
                dyn_cast<ConstantInt>(ConstArg->getAggregateElement(Idx))) {
          const APInt &V = Elt->getValue();
          APInt V2 = APInt(V.getBitWidth(), 1) << V.countTrailingZeros();
          Elements.push_back(ConstantInt::get(EltTy, V2));
        } else {
          Elements.push_back(ConstantInt::get(EltTy, 1));
        }
      }
      ShadowMul = ConstantVector::get(Elements);
    } else {
      if (ConstantInt *Elt = dyn_cast<ConstantInt>(ConstArg)) {
        const APInt &V = Elt->getValue();
        APInt V2 = APInt(V.getBitWidth(), 1) << V.countTrailingZeros();
        ShadowMul = ConstantInt::get(Ty, V2);
      } else {
        ShadowMul = ConstantInt::get(Ty, 1);
      }
    }

    IRBuilder<> IRB(&I);
    setShadow(&I,
              IRB.CreateMul(getShadow(OtherArg), ShadowMul, "msprop_mul_cst"));
    setOrigin(&I, getOrigin(OtherArg));
  }

  void visitMul(BinaryOperator &I) {
    Constant *constOp0 = dyn_cast<Constant>(I.getOperand(0));
    Constant *constOp1 = dyn_cast<Constant>(I.getOperand(1));
    if (constOp0 && !constOp1)
      handleMulByConstant(I, constOp0, I.getOperand(1));
    else if (constOp1 && !constOp0)
      handleMulByConstant(I, constOp1, I.getOperand(0));
    else
      handleShadowOr(I);
  }

  void visitFAdd(BinaryOperator &I) { handleShadowOr(I); }
  void visitFSub(BinaryOperator &I) { handleShadowOr(I); }
  void visitFMul(BinaryOperator &I) { handleShadowOr(I); }
  void visitAdd(BinaryOperator &I) { handleShadowOr(I); }
  void visitSub(BinaryOperator &I) { handleShadowOr(I); }
  void visitXor(BinaryOperator &I) { handleShadowOr(I); }

  void handleIntegerDiv(Instruction &I) {
    IRBuilder<> IRB(&I);
    // Strict on the second argument.
    insertShadowCheck(I.getOperand(1), &I);
    setShadow(&I, getShadow(&I, 0));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitUDiv(BinaryOperator &I) { handleIntegerDiv(I); }
  void visitSDiv(BinaryOperator &I) { handleIntegerDiv(I); }
  void visitURem(BinaryOperator &I) { handleIntegerDiv(I); }
  void visitSRem(BinaryOperator &I) { handleIntegerDiv(I); }

  // Floating point division is side-effect free. We can not require that the
  // divisor is fully initialized and must propagate shadow. See PR37523.
  void visitFDiv(BinaryOperator &I) { handleShadowOr(I); }
  void visitFRem(BinaryOperator &I) { handleShadowOr(I); }

  /// Instrument == and != comparisons.
  ///
  /// Sometimes the comparison result is known even if some of the bits of the
  /// arguments are not.
  void handleEqualityComparison(ICmpInst &I) {
    IRBuilder<> IRB(&I);
    Value *A = I.getOperand(0);
    Value *B = I.getOperand(1);
    Value *Sa = getShadow(A);
    Value *Sb = getShadow(B);

    // Get rid of pointers and vectors of pointers.
    // For ints (and vectors of ints), types of A and Sa match,
    // and this is a no-op.
    A = IRB.CreatePointerCast(A, Sa->getType());
    B = IRB.CreatePointerCast(B, Sb->getType());

    // A == B  <==>  (C = A^B) == 0
    // A != B  <==>  (C = A^B) != 0
    // Sc = Sa | Sb
    Value *C = IRB.CreateXor(A, B);
    Value *Sc = IRB.CreateOr(Sa, Sb);
    // Now dealing with i = (C == 0) comparison (or C != 0, does not matter now)
    // Result is defined if one of the following is true
    // * there is a defined 1 bit in C
    // * C is fully defined
    // Si = !(C & ~Sc) && Sc
    Value *Zero = Constant::getNullValue(Sc->getType());
    Value *MinusOne = Constant::getAllOnesValue(Sc->getType());
    Value *Si =
      IRB.CreateAnd(IRB.CreateICmpNE(Sc, Zero),
                    IRB.CreateICmpEQ(
                      IRB.CreateAnd(IRB.CreateXor(Sc, MinusOne), C), Zero));
    Si->setName("_msprop_icmp");
    setShadow(&I, Si);
    setOriginForNaryOp(I);
  }

  /// Build the lowest possible value of V, taking into account V's
  ///        uninitialized bits.
  Value *getLowestPossibleValue(IRBuilder<> &IRB, Value *A, Value *Sa,
                                bool isSigned) {
    if (isSigned) {
      // Split shadow into sign bit and other bits.
      Value *SaOtherBits = IRB.CreateLShr(IRB.CreateShl(Sa, 1), 1);
      Value *SaSignBit = IRB.CreateXor(Sa, SaOtherBits);
      // Maximise the undefined shadow bit, minimize other undefined bits.
      return
        IRB.CreateOr(IRB.CreateAnd(A, IRB.CreateNot(SaOtherBits)), SaSignBit);
    } else {
      // Minimize undefined bits.
      return IRB.CreateAnd(A, IRB.CreateNot(Sa));
    }
  }

  /// Build the highest possible value of V, taking into account V's
  ///        uninitialized bits.
  Value *getHighestPossibleValue(IRBuilder<> &IRB, Value *A, Value *Sa,
                                bool isSigned) {
    if (isSigned) {
      // Split shadow into sign bit and other bits.
      Value *SaOtherBits = IRB.CreateLShr(IRB.CreateShl(Sa, 1), 1);
      Value *SaSignBit = IRB.CreateXor(Sa, SaOtherBits);
      // Minimise the undefined shadow bit, maximise other undefined bits.
      return
        IRB.CreateOr(IRB.CreateAnd(A, IRB.CreateNot(SaSignBit)), SaOtherBits);
    } else {
      // Maximize undefined bits.
      return IRB.CreateOr(A, Sa);
    }
  }

  /// Instrument relational comparisons.
  ///
  /// This function does exact shadow propagation for all relational
  /// comparisons of integers, pointers and vectors of those.
  /// FIXME: output seems suboptimal when one of the operands is a constant
  void handleRelationalComparisonExact(ICmpInst &I) {
    IRBuilder<> IRB(&I);
    Value *A = I.getOperand(0);
    Value *B = I.getOperand(1);
    Value *Sa = getShadow(A);
    Value *Sb = getShadow(B);

    // Get rid of pointers and vectors of pointers.
    // For ints (and vectors of ints), types of A and Sa match,
    // and this is a no-op.
    A = IRB.CreatePointerCast(A, Sa->getType());
    B = IRB.CreatePointerCast(B, Sb->getType());

    // Let [a0, a1] be the interval of possible values of A, taking into account
    // its undefined bits. Let [b0, b1] be the interval of possible values of B.
    // Then (A cmp B) is defined iff (a0 cmp b1) == (a1 cmp b0).
    bool IsSigned = I.isSigned();
    Value *S1 = IRB.CreateICmp(I.getPredicate(),
                               getLowestPossibleValue(IRB, A, Sa, IsSigned),
                               getHighestPossibleValue(IRB, B, Sb, IsSigned));
    Value *S2 = IRB.CreateICmp(I.getPredicate(),
                               getHighestPossibleValue(IRB, A, Sa, IsSigned),
                               getLowestPossibleValue(IRB, B, Sb, IsSigned));
    Value *Si = IRB.CreateXor(S1, S2);
    setShadow(&I, Si);
    setOriginForNaryOp(I);
  }

  /// Instrument signed relational comparisons.
  ///
  /// Handle sign bit tests: x<0, x>=0, x<=-1, x>-1 by propagating the highest
  /// bit of the shadow. Everything else is delegated to handleShadowOr().
  void handleSignedRelationalComparison(ICmpInst &I) {
    Constant *constOp;
    Value *op = nullptr;
    CmpInst::Predicate pre;
    if ((constOp = dyn_cast<Constant>(I.getOperand(1)))) {
      op = I.getOperand(0);
      pre = I.getPredicate();
    } else if ((constOp = dyn_cast<Constant>(I.getOperand(0)))) {
      op = I.getOperand(1);
      pre = I.getSwappedPredicate();
    } else {
      handleShadowOr(I);
      return;
    }

    if ((constOp->isNullValue() &&
         (pre == CmpInst::ICMP_SLT || pre == CmpInst::ICMP_SGE)) ||
        (constOp->isAllOnesValue() &&
         (pre == CmpInst::ICMP_SGT || pre == CmpInst::ICMP_SLE))) {
      IRBuilder<> IRB(&I);
      Value *Shadow = IRB.CreateICmpSLT(getShadow(op), getCleanShadow(op),
                                        "_msprop_icmp_s");
      setShadow(&I, Shadow);
      setOrigin(&I, getOrigin(op));
    } else {
      handleShadowOr(I);
    }
  }

  void visitICmpInst(ICmpInst &I) {
    if (!ClHandleICmp) {
      handleShadowOr(I);
      return;
    }
    if (I.isEquality()) {
      handleEqualityComparison(I);
      return;
    }

    assert(I.isRelational());
    if (ClHandleICmpExact) {
      handleRelationalComparisonExact(I);
      return;
    }
    if (I.isSigned()) {
      handleSignedRelationalComparison(I);
      return;
    }

    assert(I.isUnsigned());
    if ((isa<Constant>(I.getOperand(0)) || isa<Constant>(I.getOperand(1)))) {
      handleRelationalComparisonExact(I);
      return;
    }

    handleShadowOr(I);
  }

  void visitFCmpInst(FCmpInst &I) {
    handleShadowOr(I);
  }

  void handleShift(BinaryOperator &I) {
    IRBuilder<> IRB(&I);
    // If any of the S2 bits are poisoned, the whole thing is poisoned.
    // Otherwise perform the same shift on S1.
    Value *S1 = getShadow(&I, 0);
    Value *S2 = getShadow(&I, 1);
    Value *S2Conv = IRB.CreateSExt(IRB.CreateICmpNE(S2, getCleanShadow(S2)),
                                   S2->getType());
    Value *V2 = I.getOperand(1);
    Value *Shift = IRB.CreateBinOp(I.getOpcode(), S1, V2);
    setShadow(&I, IRB.CreateOr(Shift, S2Conv));
    setOriginForNaryOp(I);
  }

  void visitShl(BinaryOperator &I) { handleShift(I); }
  void visitAShr(BinaryOperator &I) { handleShift(I); }
  void visitLShr(BinaryOperator &I) { handleShift(I); }

  void handleFunnelShift(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    // If any of the S2 bits are poisoned, the whole thing is poisoned.
    // Otherwise perform the same shift on S0 and S1.
    Value *S0 = getShadow(&I, 0);
    Value *S1 = getShadow(&I, 1);
    Value *S2 = getShadow(&I, 2);
    Value *S2Conv =
        IRB.CreateSExt(IRB.CreateICmpNE(S2, getCleanShadow(S2)), S2->getType());
    Value *V2 = I.getOperand(2);
    Function *Intrin = Intrinsic::getDeclaration(
        I.getModule(), I.getIntrinsicID(), S2Conv->getType());
    Value *Shift = IRB.CreateCall(Intrin, {S0, S1, V2});
    setShadow(&I, IRB.CreateOr(Shift, S2Conv));
    setOriginForNaryOp(I);
  }

  /// Instrument llvm.memmove
  ///
  /// At this point we don't know if llvm.memmove will be inlined or not.
  /// If we don't instrument it and it gets inlined,
  /// our interceptor will not kick in and we will lose the memmove.
  /// If we instrument the call here, but it does not get inlined,
  /// we will memove the shadow twice: which is bad in case
  /// of overlapping regions. So, we simply lower the intrinsic to a call.
  ///
  /// Similar situation exists for memcpy and memset.
  void visitMemMoveInst(MemMoveInst &I) {
    IRBuilder<> IRB(&I);
    IRB.CreateCall(
        MS.MemmoveFn,
        {IRB.CreatePointerCast(I.getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(I.getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(I.getArgOperand(2), MS.IntptrTy, false)});
    I.eraseFromParent();
  }

  // Similar to memmove: avoid copying shadow twice.
  // This is somewhat unfortunate as it may slowdown small constant memcpys.
  // FIXME: consider doing manual inline for small constant sizes and proper
  // alignment.
  void visitMemCpyInst(MemCpyInst &I) {
    IRBuilder<> IRB(&I);
    IRB.CreateCall(
        MS.MemcpyFn,
        {IRB.CreatePointerCast(I.getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(I.getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(I.getArgOperand(2), MS.IntptrTy, false)});
    I.eraseFromParent();
  }

  // Same as memcpy.
  void visitMemSetInst(MemSetInst &I) {
    IRBuilder<> IRB(&I);
    IRB.CreateCall(
        MS.MemsetFn,
        {IRB.CreatePointerCast(I.getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(I.getArgOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(I.getArgOperand(2), MS.IntptrTy, false)});
    I.eraseFromParent();
  }

  void visitVAStartInst(VAStartInst &I) {
    VAHelper->visitVAStartInst(I);
  }

  void visitVACopyInst(VACopyInst &I) {
    VAHelper->visitVACopyInst(I);
  }

  /// Handle vector store-like intrinsics.
  ///
  /// Instrument intrinsics that look like a simple SIMD store: writes memory,
  /// has 1 pointer argument and 1 vector argument, returns void.
  bool handleVectorStoreIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value* Addr = I.getArgOperand(0);
    Value *Shadow = getShadow(&I, 1);
    Value *ShadowPtr, *OriginPtr;

    // We don't know the pointer alignment (could be unaligned SSE store!).
    // Have to assume to worst case.
    std::tie(ShadowPtr, OriginPtr) = getShadowOriginPtr(
        Addr, IRB, Shadow->getType(), Align(1), /*isStore*/ true);
    IRB.CreateAlignedStore(Shadow, ShadowPtr, Align(1));

    if (ClCheckAccessAddress)
      insertShadowCheck(Addr, &I);

    // FIXME: factor out common code from materializeStores
    if (MS.TrackOrigins) IRB.CreateStore(getOrigin(&I, 1), OriginPtr);
    return true;
  }

  /// Handle vector load-like intrinsics.
  ///
  /// Instrument intrinsics that look like a simple SIMD load: reads memory,
  /// has 1 pointer argument, returns a vector.
  bool handleVectorLoadIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *Addr = I.getArgOperand(0);

    Type *ShadowTy = getShadowTy(&I);
    Value *ShadowPtr = nullptr, *OriginPtr = nullptr;
    if (PropagateShadow) {
      // We don't know the pointer alignment (could be unaligned SSE load!).
      // Have to assume to worst case.
      const Align Alignment = Align(1);
      std::tie(ShadowPtr, OriginPtr) =
          getShadowOriginPtr(Addr, IRB, ShadowTy, Alignment, /*isStore*/ false);
      setShadow(&I,
                IRB.CreateAlignedLoad(ShadowTy, ShadowPtr, Alignment, "_msld"));
    } else {
      setShadow(&I, getCleanShadow(&I));
    }

    if (ClCheckAccessAddress)
      insertShadowCheck(Addr, &I);

    if (MS.TrackOrigins) {
      if (PropagateShadow)
        setOrigin(&I, IRB.CreateLoad(MS.OriginTy, OriginPtr));
      else
        setOrigin(&I, getCleanOrigin());
    }
    return true;
  }

  /// Handle (SIMD arithmetic)-like intrinsics.
  ///
  /// Instrument intrinsics with any number of arguments of the same type,
  /// equal to the return type. The type should be simple (no aggregates or
  /// pointers; vectors are fine).
  /// Caller guarantees that this intrinsic does not access memory.
  bool maybeHandleSimpleNomemIntrinsic(IntrinsicInst &I) {
    Type *RetTy = I.getType();
    if (!(RetTy->isIntOrIntVectorTy() ||
          RetTy->isFPOrFPVectorTy() ||
          RetTy->isX86_MMXTy()))
      return false;

    unsigned NumArgOperands = I.arg_size();
    for (unsigned i = 0; i < NumArgOperands; ++i) {
      Type *Ty = I.getArgOperand(i)->getType();
      if (Ty != RetTy)
        return false;
    }

    IRBuilder<> IRB(&I);
    ShadowAndOriginCombiner SC(this, IRB);
    for (unsigned i = 0; i < NumArgOperands; ++i)
      SC.Add(I.getArgOperand(i));
    SC.Done(&I);

    return true;
  }

  /// Heuristically instrument unknown intrinsics.
  ///
  /// The main purpose of this code is to do something reasonable with all
  /// random intrinsics we might encounter, most importantly - SIMD intrinsics.
  /// We recognize several classes of intrinsics by their argument types and
  /// ModRefBehaviour and apply special instrumentation when we are reasonably
  /// sure that we know what the intrinsic does.
  ///
  /// We special-case intrinsics where this approach fails. See llvm.bswap
  /// handling as an example of that.
  bool handleUnknownIntrinsic(IntrinsicInst &I) {
    unsigned NumArgOperands = I.arg_size();
    if (NumArgOperands == 0)
      return false;

    if (NumArgOperands == 2 &&
        I.getArgOperand(0)->getType()->isPointerTy() &&
        I.getArgOperand(1)->getType()->isVectorTy() &&
        I.getType()->isVoidTy() &&
        !I.onlyReadsMemory()) {
      // This looks like a vector store.
      return handleVectorStoreIntrinsic(I);
    }

    if (NumArgOperands == 1 &&
        I.getArgOperand(0)->getType()->isPointerTy() &&
        I.getType()->isVectorTy() &&
        I.onlyReadsMemory()) {
      // This looks like a vector load.
      return handleVectorLoadIntrinsic(I);
    }

    if (I.doesNotAccessMemory())
      if (maybeHandleSimpleNomemIntrinsic(I))
        return true;

    // FIXME: detect and handle SSE maskstore/maskload
    return false;
  }

  void handleInvariantGroup(IntrinsicInst &I) {
    setShadow(&I, getShadow(&I, 0));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void handleLifetimeStart(IntrinsicInst &I) {
    if (!PoisonStack)
      return;
    AllocaInst *AI = llvm::findAllocaForValue(I.getArgOperand(1));
    if (!AI)
      InstrumentLifetimeStart = false;
    LifetimeStartList.push_back(std::make_pair(&I, AI));
  }

  void handleBswap(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *Op = I.getArgOperand(0);
    Type *OpType = Op->getType();
    Function *BswapFunc = Intrinsic::getDeclaration(
      F.getParent(), Intrinsic::bswap, makeArrayRef(&OpType, 1));
    setShadow(&I, IRB.CreateCall(BswapFunc, getShadow(Op)));
    setOrigin(&I, getOrigin(Op));
  }

  // Instrument vector convert intrinsic.
  //
  // This function instruments intrinsics like cvtsi2ss:
  // %Out = int_xxx_cvtyyy(%ConvertOp)
  // or
  // %Out = int_xxx_cvtyyy(%CopyOp, %ConvertOp)
  // Intrinsic converts \p NumUsedElements elements of \p ConvertOp to the same
  // number \p Out elements, and (if has 2 arguments) copies the rest of the
  // elements from \p CopyOp.
  // In most cases conversion involves floating-point value which may trigger a
  // hardware exception when not fully initialized. For this reason we require
  // \p ConvertOp[0:NumUsedElements] to be fully initialized and trap otherwise.
  // We copy the shadow of \p CopyOp[NumUsedElements:] to \p
  // Out[NumUsedElements:]. This means that intrinsics without \p CopyOp always
  // return a fully initialized value.
  void handleVectorConvertIntrinsic(IntrinsicInst &I, int NumUsedElements,
                                    bool HasRoundingMode = false) {
    IRBuilder<> IRB(&I);
    Value *CopyOp, *ConvertOp;

    assert((!HasRoundingMode ||
            isa<ConstantInt>(I.getArgOperand(I.arg_size() - 1))) &&
           "Invalid rounding mode");

    switch (I.arg_size() - HasRoundingMode) {
    case 2:
      CopyOp = I.getArgOperand(0);
      ConvertOp = I.getArgOperand(1);
      break;
    case 1:
      ConvertOp = I.getArgOperand(0);
      CopyOp = nullptr;
      break;
    default:
      llvm_unreachable("Cvt intrinsic with unsupported number of arguments.");
    }

    // The first *NumUsedElements* elements of ConvertOp are converted to the
    // same number of output elements. The rest of the output is copied from
    // CopyOp, or (if not available) filled with zeroes.
    // Combine shadow for elements of ConvertOp that are used in this operation,
    // and insert a check.
    // FIXME: consider propagating shadow of ConvertOp, at least in the case of
    // int->any conversion.
    Value *ConvertShadow = getShadow(ConvertOp);
    Value *AggShadow = nullptr;
    if (ConvertOp->getType()->isVectorTy()) {
      AggShadow = IRB.CreateExtractElement(
          ConvertShadow, ConstantInt::get(IRB.getInt32Ty(), 0));
      for (int i = 1; i < NumUsedElements; ++i) {
        Value *MoreShadow = IRB.CreateExtractElement(
            ConvertShadow, ConstantInt::get(IRB.getInt32Ty(), i));
        AggShadow = IRB.CreateOr(AggShadow, MoreShadow);
      }
    } else {
      AggShadow = ConvertShadow;
    }
    assert(AggShadow->getType()->isIntegerTy());
    insertShadowCheck(AggShadow, getOrigin(ConvertOp), &I);

    // Build result shadow by zero-filling parts of CopyOp shadow that come from
    // ConvertOp.
    if (CopyOp) {
      assert(CopyOp->getType() == I.getType());
      assert(CopyOp->getType()->isVectorTy());
      Value *ResultShadow = getShadow(CopyOp);
      Type *EltTy = cast<VectorType>(ResultShadow->getType())->getElementType();
      for (int i = 0; i < NumUsedElements; ++i) {
        ResultShadow = IRB.CreateInsertElement(
            ResultShadow, ConstantInt::getNullValue(EltTy),
            ConstantInt::get(IRB.getInt32Ty(), i));
      }
      setShadow(&I, ResultShadow);
      setOrigin(&I, getOrigin(CopyOp));
    } else {
      setShadow(&I, getCleanShadow(&I));
      setOrigin(&I, getCleanOrigin());
    }
  }

  // Given a scalar or vector, extract lower 64 bits (or less), and return all
  // zeroes if it is zero, and all ones otherwise.
  Value *Lower64ShadowExtend(IRBuilder<> &IRB, Value *S, Type *T) {
    if (S->getType()->isVectorTy())
      S = CreateShadowCast(IRB, S, IRB.getInt64Ty(), /* Signed */ true);
    assert(S->getType()->getPrimitiveSizeInBits() <= 64);
    Value *S2 = IRB.CreateICmpNE(S, getCleanShadow(S));
    return CreateShadowCast(IRB, S2, T, /* Signed */ true);
  }

  // Given a vector, extract its first element, and return all
  // zeroes if it is zero, and all ones otherwise.
  Value *LowerElementShadowExtend(IRBuilder<> &IRB, Value *S, Type *T) {
    Value *S1 = IRB.CreateExtractElement(S, (uint64_t)0);
    Value *S2 = IRB.CreateICmpNE(S1, getCleanShadow(S1));
    return CreateShadowCast(IRB, S2, T, /* Signed */ true);
  }

  Value *VariableShadowExtend(IRBuilder<> &IRB, Value *S) {
    Type *T = S->getType();
    assert(T->isVectorTy());
    Value *S2 = IRB.CreateICmpNE(S, getCleanShadow(S));
    return IRB.CreateSExt(S2, T);
  }

  // Instrument vector shift intrinsic.
  //
  // This function instruments intrinsics like int_x86_avx2_psll_w.
  // Intrinsic shifts %In by %ShiftSize bits.
  // %ShiftSize may be a vector. In that case the lower 64 bits determine shift
  // size, and the rest is ignored. Behavior is defined even if shift size is
  // greater than register (or field) width.
  void handleVectorShiftIntrinsic(IntrinsicInst &I, bool Variable) {
    assert(I.arg_size() == 2);
    IRBuilder<> IRB(&I);
    // If any of the S2 bits are poisoned, the whole thing is poisoned.
    // Otherwise perform the same shift on S1.
    Value *S1 = getShadow(&I, 0);
    Value *S2 = getShadow(&I, 1);
    Value *S2Conv = Variable ? VariableShadowExtend(IRB, S2)
                             : Lower64ShadowExtend(IRB, S2, getShadowTy(&I));
    Value *V1 = I.getOperand(0);
    Value *V2 = I.getOperand(1);
    Value *Shift = IRB.CreateCall(I.getFunctionType(), I.getCalledOperand(),
                                  {IRB.CreateBitCast(S1, V1->getType()), V2});
    Shift = IRB.CreateBitCast(Shift, getShadowTy(&I));
    setShadow(&I, IRB.CreateOr(Shift, S2Conv));
    setOriginForNaryOp(I);
  }

  // Get an X86_MMX-sized vector type.
  Type *getMMXVectorTy(unsigned EltSizeInBits) {
    const unsigned X86_MMXSizeInBits = 64;
    assert(EltSizeInBits != 0 && (X86_MMXSizeInBits % EltSizeInBits) == 0 &&
           "Illegal MMX vector element size");
    return FixedVectorType::get(IntegerType::get(*MS.C, EltSizeInBits),
                                X86_MMXSizeInBits / EltSizeInBits);
  }

  // Returns a signed counterpart for an (un)signed-saturate-and-pack
  // intrinsic.
  Intrinsic::ID getSignedPackIntrinsic(Intrinsic::ID id) {
    switch (id) {
      case Intrinsic::x86_sse2_packsswb_128:
      case Intrinsic::x86_sse2_packuswb_128:
        return Intrinsic::x86_sse2_packsswb_128;

      case Intrinsic::x86_sse2_packssdw_128:
      case Intrinsic::x86_sse41_packusdw:
        return Intrinsic::x86_sse2_packssdw_128;

      case Intrinsic::x86_avx2_packsswb:
      case Intrinsic::x86_avx2_packuswb:
        return Intrinsic::x86_avx2_packsswb;

      case Intrinsic::x86_avx2_packssdw:
      case Intrinsic::x86_avx2_packusdw:
        return Intrinsic::x86_avx2_packssdw;

      case Intrinsic::x86_mmx_packsswb:
      case Intrinsic::x86_mmx_packuswb:
        return Intrinsic::x86_mmx_packsswb;

      case Intrinsic::x86_mmx_packssdw:
        return Intrinsic::x86_mmx_packssdw;
      default:
        llvm_unreachable("unexpected intrinsic id");
    }
  }

  // Instrument vector pack intrinsic.
  //
  // This function instruments intrinsics like x86_mmx_packsswb, that
  // packs elements of 2 input vectors into half as many bits with saturation.
  // Shadow is propagated with the signed variant of the same intrinsic applied
  // to sext(Sa != zeroinitializer), sext(Sb != zeroinitializer).
  // EltSizeInBits is used only for x86mmx arguments.
  void handleVectorPackIntrinsic(IntrinsicInst &I, unsigned EltSizeInBits = 0) {
    assert(I.arg_size() == 2);
    bool isX86_MMX = I.getOperand(0)->getType()->isX86_MMXTy();
    IRBuilder<> IRB(&I);
    Value *S1 = getShadow(&I, 0);
    Value *S2 = getShadow(&I, 1);
    assert(isX86_MMX || S1->getType()->isVectorTy());

    // SExt and ICmpNE below must apply to individual elements of input vectors.
    // In case of x86mmx arguments, cast them to appropriate vector types and
    // back.
    Type *T = isX86_MMX ? getMMXVectorTy(EltSizeInBits) : S1->getType();
    if (isX86_MMX) {
      S1 = IRB.CreateBitCast(S1, T);
      S2 = IRB.CreateBitCast(S2, T);
    }
    Value *S1_ext = IRB.CreateSExt(
        IRB.CreateICmpNE(S1, Constant::getNullValue(T)), T);
    Value *S2_ext = IRB.CreateSExt(
        IRB.CreateICmpNE(S2, Constant::getNullValue(T)), T);
    if (isX86_MMX) {
      Type *X86_MMXTy = Type::getX86_MMXTy(*MS.C);
      S1_ext = IRB.CreateBitCast(S1_ext, X86_MMXTy);
      S2_ext = IRB.CreateBitCast(S2_ext, X86_MMXTy);
    }

    Function *ShadowFn = Intrinsic::getDeclaration(
        F.getParent(), getSignedPackIntrinsic(I.getIntrinsicID()));

    Value *S =
        IRB.CreateCall(ShadowFn, {S1_ext, S2_ext}, "_msprop_vector_pack");
    if (isX86_MMX) S = IRB.CreateBitCast(S, getShadowTy(&I));
    setShadow(&I, S);
    setOriginForNaryOp(I);
  }

  // Instrument sum-of-absolute-differences intrinsic.
  void handleVectorSadIntrinsic(IntrinsicInst &I) {
    const unsigned SignificantBitsPerResultElement = 16;
    bool isX86_MMX = I.getOperand(0)->getType()->isX86_MMXTy();
    Type *ResTy = isX86_MMX ? IntegerType::get(*MS.C, 64) : I.getType();
    unsigned ZeroBitsPerResultElement =
        ResTy->getScalarSizeInBits() - SignificantBitsPerResultElement;

    IRBuilder<> IRB(&I);
    Value *S = IRB.CreateOr(getShadow(&I, 0), getShadow(&I, 1));
    S = IRB.CreateBitCast(S, ResTy);
    S = IRB.CreateSExt(IRB.CreateICmpNE(S, Constant::getNullValue(ResTy)),
                       ResTy);
    S = IRB.CreateLShr(S, ZeroBitsPerResultElement);
    S = IRB.CreateBitCast(S, getShadowTy(&I));
    setShadow(&I, S);
    setOriginForNaryOp(I);
  }

  // Instrument multiply-add intrinsic.
  void handleVectorPmaddIntrinsic(IntrinsicInst &I,
                                  unsigned EltSizeInBits = 0) {
    bool isX86_MMX = I.getOperand(0)->getType()->isX86_MMXTy();
    Type *ResTy = isX86_MMX ? getMMXVectorTy(EltSizeInBits * 2) : I.getType();
    IRBuilder<> IRB(&I);
    Value *S = IRB.CreateOr(getShadow(&I, 0), getShadow(&I, 1));
    S = IRB.CreateBitCast(S, ResTy);
    S = IRB.CreateSExt(IRB.CreateICmpNE(S, Constant::getNullValue(ResTy)),
                       ResTy);
    S = IRB.CreateBitCast(S, getShadowTy(&I));
    setShadow(&I, S);
    setOriginForNaryOp(I);
  }

  // Instrument compare-packed intrinsic.
  // Basically, an or followed by sext(icmp ne 0) to end up with all-zeros or
  // all-ones shadow.
  void handleVectorComparePackedIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Type *ResTy = getShadowTy(&I);
    Value *S0 = IRB.CreateOr(getShadow(&I, 0), getShadow(&I, 1));
    Value *S = IRB.CreateSExt(
        IRB.CreateICmpNE(S0, Constant::getNullValue(ResTy)), ResTy);
    setShadow(&I, S);
    setOriginForNaryOp(I);
  }

  // Instrument compare-scalar intrinsic.
  // This handles both cmp* intrinsics which return the result in the first
  // element of a vector, and comi* which return the result as i32.
  void handleVectorCompareScalarIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *S0 = IRB.CreateOr(getShadow(&I, 0), getShadow(&I, 1));
    Value *S = LowerElementShadowExtend(IRB, S0, getShadowTy(&I));
    setShadow(&I, S);
    setOriginForNaryOp(I);
  }

  // Instrument generic vector reduction intrinsics
  // by ORing together all their fields.
  void handleVectorReduceIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *S = IRB.CreateOrReduce(getShadow(&I, 0));
    setShadow(&I, S);
    setOrigin(&I, getOrigin(&I, 0));
  }

  // Instrument vector.reduce.or intrinsic.
  // Valid (non-poisoned) set bits in the operand pull low the
  // corresponding shadow bits.
  void handleVectorReduceOrIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *OperandShadow = getShadow(&I, 0);
    Value *OperandUnsetBits = IRB.CreateNot(I.getOperand(0));
    Value *OperandUnsetOrPoison = IRB.CreateOr(OperandUnsetBits, OperandShadow);
    // Bit N is clean if any field's bit N is 1 and unpoison
    Value *OutShadowMask = IRB.CreateAndReduce(OperandUnsetOrPoison);
    // Otherwise, it is clean if every field's bit N is unpoison
    Value *OrShadow = IRB.CreateOrReduce(OperandShadow);
    Value *S = IRB.CreateAnd(OutShadowMask, OrShadow);

    setShadow(&I, S);
    setOrigin(&I, getOrigin(&I, 0));
  }

  // Instrument vector.reduce.and intrinsic.
  // Valid (non-poisoned) unset bits in the operand pull down the
  // corresponding shadow bits.
  void handleVectorReduceAndIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *OperandShadow = getShadow(&I, 0);
    Value *OperandSetOrPoison = IRB.CreateOr(I.getOperand(0), OperandShadow);
    // Bit N is clean if any field's bit N is 0 and unpoison
    Value *OutShadowMask = IRB.CreateAndReduce(OperandSetOrPoison);
    // Otherwise, it is clean if every field's bit N is unpoison
    Value *OrShadow = IRB.CreateOrReduce(OperandShadow);
    Value *S = IRB.CreateAnd(OutShadowMask, OrShadow);

    setShadow(&I, S);
    setOrigin(&I, getOrigin(&I, 0));
  }

  void handleStmxcsr(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value* Addr = I.getArgOperand(0);
    Type *Ty = IRB.getInt32Ty();
    Value *ShadowPtr =
        getShadowOriginPtr(Addr, IRB, Ty, Align(1), /*isStore*/ true).first;

    IRB.CreateStore(getCleanShadow(Ty),
                    IRB.CreatePointerCast(ShadowPtr, Ty->getPointerTo()));

    if (ClCheckAccessAddress)
      insertShadowCheck(Addr, &I);
  }

  void handleLdmxcsr(IntrinsicInst &I) {
    if (!InsertChecks) return;

    IRBuilder<> IRB(&I);
    Value *Addr = I.getArgOperand(0);
    Type *Ty = IRB.getInt32Ty();
    const Align Alignment = Align(1);
    Value *ShadowPtr, *OriginPtr;
    std::tie(ShadowPtr, OriginPtr) =
        getShadowOriginPtr(Addr, IRB, Ty, Alignment, /*isStore*/ false);

    if (ClCheckAccessAddress)
      insertShadowCheck(Addr, &I);

    Value *Shadow = IRB.CreateAlignedLoad(Ty, ShadowPtr, Alignment, "_ldmxcsr");
    Value *Origin = MS.TrackOrigins ? IRB.CreateLoad(MS.OriginTy, OriginPtr)
                                    : getCleanOrigin();
    insertShadowCheck(Shadow, Origin, &I);
  }

  void handleMaskedStore(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *V = I.getArgOperand(0);
    Value *Addr = I.getArgOperand(1);
    const Align Alignment(
        cast<ConstantInt>(I.getArgOperand(2))->getZExtValue());
    Value *Mask = I.getArgOperand(3);
    Value *Shadow = getShadow(V);

    Value *ShadowPtr;
    Value *OriginPtr;
    std::tie(ShadowPtr, OriginPtr) = getShadowOriginPtr(
        Addr, IRB, Shadow->getType(), Alignment, /*isStore*/ true);

    if (ClCheckAccessAddress) {
      insertShadowCheck(Addr, &I);
      // Uninitialized mask is kind of like uninitialized address, but not as
      // scary.
      insertShadowCheck(Mask, &I);
    }

    IRB.CreateMaskedStore(Shadow, ShadowPtr, Alignment, Mask);

    if (MS.TrackOrigins) {
      auto &DL = F.getParent()->getDataLayout();
      paintOrigin(IRB, getOrigin(V), OriginPtr,
                  DL.getTypeStoreSize(Shadow->getType()),
                  std::max(Alignment, kMinOriginAlignment));
    }
  }

  bool handleMaskedLoad(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *Addr = I.getArgOperand(0);
    const Align Alignment(
        cast<ConstantInt>(I.getArgOperand(1))->getZExtValue());
    Value *Mask = I.getArgOperand(2);
    Value *PassThru = I.getArgOperand(3);

    Type *ShadowTy = getShadowTy(&I);
    Value *ShadowPtr, *OriginPtr;
    if (PropagateShadow) {
      std::tie(ShadowPtr, OriginPtr) =
          getShadowOriginPtr(Addr, IRB, ShadowTy, Alignment, /*isStore*/ false);
      setShadow(&I, IRB.CreateMaskedLoad(ShadowTy, ShadowPtr, Alignment, Mask,
                                         getShadow(PassThru), "_msmaskedld"));
    } else {
      setShadow(&I, getCleanShadow(&I));
    }

    if (ClCheckAccessAddress) {
      insertShadowCheck(Addr, &I);
      insertShadowCheck(Mask, &I);
    }

    if (MS.TrackOrigins) {
      if (PropagateShadow) {
        // Choose between PassThru's and the loaded value's origins.
        Value *MaskedPassThruShadow = IRB.CreateAnd(
            getShadow(PassThru), IRB.CreateSExt(IRB.CreateNeg(Mask), ShadowTy));

        Value *Acc = IRB.CreateExtractElement(
            MaskedPassThruShadow, ConstantInt::get(IRB.getInt32Ty(), 0));
        for (int i = 1, N = cast<FixedVectorType>(PassThru->getType())
                                ->getNumElements();
             i < N; ++i) {
          Value *More = IRB.CreateExtractElement(
              MaskedPassThruShadow, ConstantInt::get(IRB.getInt32Ty(), i));
          Acc = IRB.CreateOr(Acc, More);
        }

        Value *Origin = IRB.CreateSelect(
            IRB.CreateICmpNE(Acc, Constant::getNullValue(Acc->getType())),
            getOrigin(PassThru), IRB.CreateLoad(MS.OriginTy, OriginPtr));

        setOrigin(&I, Origin);
      } else {
        setOrigin(&I, getCleanOrigin());
      }
    }
    return true;
  }

  // Instrument BMI / BMI2 intrinsics.
  // All of these intrinsics are Z = I(X, Y)
  // where the types of all operands and the result match, and are either i32 or i64.
  // The following instrumentation happens to work for all of them:
  //   Sz = I(Sx, Y) | (sext (Sy != 0))
  void handleBmiIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Type *ShadowTy = getShadowTy(&I);

    // If any bit of the mask operand is poisoned, then the whole thing is.
    Value *SMask = getShadow(&I, 1);
    SMask = IRB.CreateSExt(IRB.CreateICmpNE(SMask, getCleanShadow(ShadowTy)),
                           ShadowTy);
    // Apply the same intrinsic to the shadow of the first operand.
    Value *S = IRB.CreateCall(I.getCalledFunction(),
                              {getShadow(&I, 0), I.getOperand(1)});
    S = IRB.CreateOr(SMask, S);
    setShadow(&I, S);
    setOriginForNaryOp(I);
  }

  SmallVector<int, 8> getPclmulMask(unsigned Width, bool OddElements) {
    SmallVector<int, 8> Mask;
    for (unsigned X = OddElements ? 1 : 0; X < Width; X += 2) {
      Mask.append(2, X);
    }
    return Mask;
  }

  // Instrument pclmul intrinsics.
  // These intrinsics operate either on odd or on even elements of the input
  // vectors, depending on the constant in the 3rd argument, ignoring the rest.
  // Replace the unused elements with copies of the used ones, ex:
  //   (0, 1, 2, 3) -> (0, 0, 2, 2) (even case)
  // or
  //   (0, 1, 2, 3) -> (1, 1, 3, 3) (odd case)
  // and then apply the usual shadow combining logic.
  void handlePclmulIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    unsigned Width =
        cast<FixedVectorType>(I.getArgOperand(0)->getType())->getNumElements();
    assert(isa<ConstantInt>(I.getArgOperand(2)) &&
           "pclmul 3rd operand must be a constant");
    unsigned Imm = cast<ConstantInt>(I.getArgOperand(2))->getZExtValue();
    Value *Shuf0 = IRB.CreateShuffleVector(getShadow(&I, 0),
                                           getPclmulMask(Width, Imm & 0x01));
    Value *Shuf1 = IRB.CreateShuffleVector(getShadow(&I, 1),
                                           getPclmulMask(Width, Imm & 0x10));
    ShadowAndOriginCombiner SOC(this, IRB);
    SOC.Add(Shuf0, getOrigin(&I, 0));
    SOC.Add(Shuf1, getOrigin(&I, 1));
    SOC.Done(&I);
  }

  // Instrument _mm_*_sd intrinsics
  void handleUnarySdIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *First = getShadow(&I, 0);
    Value *Second = getShadow(&I, 1);
    // High word of first operand, low word of second
    Value *Shadow =
        IRB.CreateShuffleVector(First, Second, llvm::makeArrayRef<int>({2, 1}));

    setShadow(&I, Shadow);
    setOriginForNaryOp(I);
  }

  void handleBinarySdIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *First = getShadow(&I, 0);
    Value *Second = getShadow(&I, 1);
    Value *OrShadow = IRB.CreateOr(First, Second);
    // High word of first operand, low word of both OR'd together
    Value *Shadow = IRB.CreateShuffleVector(First, OrShadow,
                                            llvm::makeArrayRef<int>({2, 1}));

    setShadow(&I, Shadow);
    setOriginForNaryOp(I);
  }

  // Instrument abs intrinsic.
  // handleUnknownIntrinsic can't handle it because of the last
  // is_int_min_poison argument which does not match the result type.
  void handleAbsIntrinsic(IntrinsicInst &I) {
    assert(I.getType()->isIntOrIntVectorTy());
    assert(I.getArgOperand(0)->getType() == I.getType());

    // FIXME: Handle is_int_min_poison.
    IRBuilder<> IRB(&I);
    setShadow(&I, getShadow(&I, 0));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitIntrinsicInst(IntrinsicInst &I) {
    switch (I.getIntrinsicID()) {
    case Intrinsic::abs:
      handleAbsIntrinsic(I);
      break;
    case Intrinsic::lifetime_start:
      handleLifetimeStart(I);
      break;
    case Intrinsic::launder_invariant_group:
    case Intrinsic::strip_invariant_group:
      handleInvariantGroup(I);
      break;
    case Intrinsic::bswap:
      handleBswap(I);
      break;
    case Intrinsic::masked_store:
      handleMaskedStore(I);
      break;
    case Intrinsic::masked_load:
      handleMaskedLoad(I);
      break;
    case Intrinsic::vector_reduce_and:
      handleVectorReduceAndIntrinsic(I);
      break;
    case Intrinsic::vector_reduce_or:
      handleVectorReduceOrIntrinsic(I);
      break;
    case Intrinsic::vector_reduce_add:
    case Intrinsic::vector_reduce_xor:
    case Intrinsic::vector_reduce_mul:
      handleVectorReduceIntrinsic(I);
      break;
    case Intrinsic::x86_sse_stmxcsr:
      handleStmxcsr(I);
      break;
    case Intrinsic::x86_sse_ldmxcsr:
      handleLdmxcsr(I);
      break;
    case Intrinsic::x86_avx512_vcvtsd2usi64:
    case Intrinsic::x86_avx512_vcvtsd2usi32:
    case Intrinsic::x86_avx512_vcvtss2usi64:
    case Intrinsic::x86_avx512_vcvtss2usi32:
    case Intrinsic::x86_avx512_cvttss2usi64:
    case Intrinsic::x86_avx512_cvttss2usi:
    case Intrinsic::x86_avx512_cvttsd2usi64:
    case Intrinsic::x86_avx512_cvttsd2usi:
    case Intrinsic::x86_avx512_cvtusi2ss:
    case Intrinsic::x86_avx512_cvtusi642sd:
    case Intrinsic::x86_avx512_cvtusi642ss:
      handleVectorConvertIntrinsic(I, 1, true);
      break;
    case Intrinsic::x86_sse2_cvtsd2si64:
    case Intrinsic::x86_sse2_cvtsd2si:
    case Intrinsic::x86_sse2_cvtsd2ss:
    case Intrinsic::x86_sse2_cvttsd2si64:
    case Intrinsic::x86_sse2_cvttsd2si:
    case Intrinsic::x86_sse_cvtss2si64:
    case Intrinsic::x86_sse_cvtss2si:
    case Intrinsic::x86_sse_cvttss2si64:
    case Intrinsic::x86_sse_cvttss2si:
      handleVectorConvertIntrinsic(I, 1);
      break;
    case Intrinsic::x86_sse_cvtps2pi:
    case Intrinsic::x86_sse_cvttps2pi:
      handleVectorConvertIntrinsic(I, 2);
      break;

    case Intrinsic::x86_avx512_psll_w_512:
    case Intrinsic::x86_avx512_psll_d_512:
    case Intrinsic::x86_avx512_psll_q_512:
    case Intrinsic::x86_avx512_pslli_w_512:
    case Intrinsic::x86_avx512_pslli_d_512:
    case Intrinsic::x86_avx512_pslli_q_512:
    case Intrinsic::x86_avx512_psrl_w_512:
    case Intrinsic::x86_avx512_psrl_d_512:
    case Intrinsic::x86_avx512_psrl_q_512:
    case Intrinsic::x86_avx512_psra_w_512:
    case Intrinsic::x86_avx512_psra_d_512:
    case Intrinsic::x86_avx512_psra_q_512:
    case Intrinsic::x86_avx512_psrli_w_512:
    case Intrinsic::x86_avx512_psrli_d_512:
    case Intrinsic::x86_avx512_psrli_q_512:
    case Intrinsic::x86_avx512_psrai_w_512:
    case Intrinsic::x86_avx512_psrai_d_512:
    case Intrinsic::x86_avx512_psrai_q_512:
    case Intrinsic::x86_avx512_psra_q_256:
    case Intrinsic::x86_avx512_psra_q_128:
    case Intrinsic::x86_avx512_psrai_q_256:
    case Intrinsic::x86_avx512_psrai_q_128:
    case Intrinsic::x86_avx2_psll_w:
    case Intrinsic::x86_avx2_psll_d:
    case Intrinsic::x86_avx2_psll_q:
    case Intrinsic::x86_avx2_pslli_w:
    case Intrinsic::x86_avx2_pslli_d:
    case Intrinsic::x86_avx2_pslli_q:
    case Intrinsic::x86_avx2_psrl_w:
    case Intrinsic::x86_avx2_psrl_d:
    case Intrinsic::x86_avx2_psrl_q:
    case Intrinsic::x86_avx2_psra_w:
    case Intrinsic::x86_avx2_psra_d:
    case Intrinsic::x86_avx2_psrli_w:
    case Intrinsic::x86_avx2_psrli_d:
    case Intrinsic::x86_avx2_psrli_q:
    case Intrinsic::x86_avx2_psrai_w:
    case Intrinsic::x86_avx2_psrai_d:
    case Intrinsic::x86_sse2_psll_w:
    case Intrinsic::x86_sse2_psll_d:
    case Intrinsic::x86_sse2_psll_q:
    case Intrinsic::x86_sse2_pslli_w:
    case Intrinsic::x86_sse2_pslli_d:
    case Intrinsic::x86_sse2_pslli_q:
    case Intrinsic::x86_sse2_psrl_w:
    case Intrinsic::x86_sse2_psrl_d:
    case Intrinsic::x86_sse2_psrl_q:
    case Intrinsic::x86_sse2_psra_w:
    case Intrinsic::x86_sse2_psra_d:
    case Intrinsic::x86_sse2_psrli_w:
    case Intrinsic::x86_sse2_psrli_d:
    case Intrinsic::x86_sse2_psrli_q:
    case Intrinsic::x86_sse2_psrai_w:
    case Intrinsic::x86_sse2_psrai_d:
    case Intrinsic::x86_mmx_psll_w:
    case Intrinsic::x86_mmx_psll_d:
    case Intrinsic::x86_mmx_psll_q:
    case Intrinsic::x86_mmx_pslli_w:
    case Intrinsic::x86_mmx_pslli_d:
    case Intrinsic::x86_mmx_pslli_q:
    case Intrinsic::x86_mmx_psrl_w:
    case Intrinsic::x86_mmx_psrl_d:
    case Intrinsic::x86_mmx_psrl_q:
    case Intrinsic::x86_mmx_psra_w:
    case Intrinsic::x86_mmx_psra_d:
    case Intrinsic::x86_mmx_psrli_w:
    case Intrinsic::x86_mmx_psrli_d:
    case Intrinsic::x86_mmx_psrli_q:
    case Intrinsic::x86_mmx_psrai_w:
    case Intrinsic::x86_mmx_psrai_d:
      handleVectorShiftIntrinsic(I, /* Variable */ false);
      break;
    case Intrinsic::x86_avx2_psllv_d:
    case Intrinsic::x86_avx2_psllv_d_256:
    case Intrinsic::x86_avx512_psllv_d_512:
    case Intrinsic::x86_avx2_psllv_q:
    case Intrinsic::x86_avx2_psllv_q_256:
    case Intrinsic::x86_avx512_psllv_q_512:
    case Intrinsic::x86_avx2_psrlv_d:
    case Intrinsic::x86_avx2_psrlv_d_256:
    case Intrinsic::x86_avx512_psrlv_d_512:
    case Intrinsic::x86_avx2_psrlv_q:
    case Intrinsic::x86_avx2_psrlv_q_256:
    case Intrinsic::x86_avx512_psrlv_q_512:
    case Intrinsic::x86_avx2_psrav_d:
    case Intrinsic::x86_avx2_psrav_d_256:
    case Intrinsic::x86_avx512_psrav_d_512:
    case Intrinsic::x86_avx512_psrav_q_128:
    case Intrinsic::x86_avx512_psrav_q_256:
    case Intrinsic::x86_avx512_psrav_q_512:
      handleVectorShiftIntrinsic(I, /* Variable */ true);
      break;

    case Intrinsic::x86_sse2_packsswb_128:
    case Intrinsic::x86_sse2_packssdw_128:
    case Intrinsic::x86_sse2_packuswb_128:
    case Intrinsic::x86_sse41_packusdw:
    case Intrinsic::x86_avx2_packsswb:
    case Intrinsic::x86_avx2_packssdw:
    case Intrinsic::x86_avx2_packuswb:
    case Intrinsic::x86_avx2_packusdw:
      handleVectorPackIntrinsic(I);
      break;

    case Intrinsic::x86_mmx_packsswb:
    case Intrinsic::x86_mmx_packuswb:
      handleVectorPackIntrinsic(I, 16);
      break;

    case Intrinsic::x86_mmx_packssdw:
      handleVectorPackIntrinsic(I, 32);
      break;

    case Intrinsic::x86_mmx_psad_bw:
    case Intrinsic::x86_sse2_psad_bw:
    case Intrinsic::x86_avx2_psad_bw:
      handleVectorSadIntrinsic(I);
      break;

    case Intrinsic::x86_sse2_pmadd_wd:
    case Intrinsic::x86_avx2_pmadd_wd:
    case Intrinsic::x86_ssse3_pmadd_ub_sw_128:
    case Intrinsic::x86_avx2_pmadd_ub_sw:
      handleVectorPmaddIntrinsic(I);
      break;

    case Intrinsic::x86_ssse3_pmadd_ub_sw:
      handleVectorPmaddIntrinsic(I, 8);
      break;

    case Intrinsic::x86_mmx_pmadd_wd:
      handleVectorPmaddIntrinsic(I, 16);
      break;

    case Intrinsic::x86_sse_cmp_ss:
    case Intrinsic::x86_sse2_cmp_sd:
    case Intrinsic::x86_sse_comieq_ss:
    case Intrinsic::x86_sse_comilt_ss:
    case Intrinsic::x86_sse_comile_ss:
    case Intrinsic::x86_sse_comigt_ss:
    case Intrinsic::x86_sse_comige_ss:
    case Intrinsic::x86_sse_comineq_ss:
    case Intrinsic::x86_sse_ucomieq_ss:
    case Intrinsic::x86_sse_ucomilt_ss:
    case Intrinsic::x86_sse_ucomile_ss:
    case Intrinsic::x86_sse_ucomigt_ss:
    case Intrinsic::x86_sse_ucomige_ss:
    case Intrinsic::x86_sse_ucomineq_ss:
    case Intrinsic::x86_sse2_comieq_sd:
    case Intrinsic::x86_sse2_comilt_sd:
    case Intrinsic::x86_sse2_comile_sd:
    case Intrinsic::x86_sse2_comigt_sd:
    case Intrinsic::x86_sse2_comige_sd:
    case Intrinsic::x86_sse2_comineq_sd:
    case Intrinsic::x86_sse2_ucomieq_sd:
    case Intrinsic::x86_sse2_ucomilt_sd:
    case Intrinsic::x86_sse2_ucomile_sd:
    case Intrinsic::x86_sse2_ucomigt_sd:
    case Intrinsic::x86_sse2_ucomige_sd:
    case Intrinsic::x86_sse2_ucomineq_sd:
      handleVectorCompareScalarIntrinsic(I);
      break;

    case Intrinsic::x86_sse_cmp_ps:
    case Intrinsic::x86_sse2_cmp_pd:
      // FIXME: For x86_avx_cmp_pd_256 and x86_avx_cmp_ps_256 this function
      // generates reasonably looking IR that fails in the backend with "Do not
      // know how to split the result of this operator!".
      handleVectorComparePackedIntrinsic(I);
      break;

    case Intrinsic::x86_bmi_bextr_32:
    case Intrinsic::x86_bmi_bextr_64:
    case Intrinsic::x86_bmi_bzhi_32:
    case Intrinsic::x86_bmi_bzhi_64:
    case Intrinsic::x86_bmi_pdep_32:
    case Intrinsic::x86_bmi_pdep_64:
    case Intrinsic::x86_bmi_pext_32:
    case Intrinsic::x86_bmi_pext_64:
      handleBmiIntrinsic(I);
      break;

    case Intrinsic::x86_pclmulqdq:
    case Intrinsic::x86_pclmulqdq_256:
    case Intrinsic::x86_pclmulqdq_512:
      handlePclmulIntrinsic(I);
      break;

    case Intrinsic::x86_sse41_round_sd:
      handleUnarySdIntrinsic(I);
      break;
    case Intrinsic::x86_sse2_max_sd:
    case Intrinsic::x86_sse2_min_sd:
      handleBinarySdIntrinsic(I);
      break;

    case Intrinsic::fshl:
    case Intrinsic::fshr:
      handleFunnelShift(I);
      break;

    case Intrinsic::is_constant:
      // The result of llvm.is.constant() is always defined.
      setShadow(&I, getCleanShadow(&I));
      setOrigin(&I, getCleanOrigin());
      break;

    default:
      if (!handleUnknownIntrinsic(I))
        visitInstruction(I);
      break;
    }
  }

  void visitLibAtomicLoad(CallBase &CB) {
    // Since we use getNextNode here, we can't have CB terminate the BB.
    assert(isa<CallInst>(CB));

    IRBuilder<> IRB(&CB);
    Value *Size = CB.getArgOperand(0);
    Value *SrcPtr = CB.getArgOperand(1);
    Value *DstPtr = CB.getArgOperand(2);
    Value *Ordering = CB.getArgOperand(3);
    // Convert the call to have at least Acquire ordering to make sure
    // the shadow operations aren't reordered before it.
    Value *NewOrdering =
        IRB.CreateExtractElement(makeAddAcquireOrderingTable(IRB), Ordering);
    CB.setArgOperand(3, NewOrdering);

    IRBuilder<> NextIRB(CB.getNextNode());
    NextIRB.SetCurrentDebugLocation(CB.getDebugLoc());

    Value *SrcShadowPtr, *SrcOriginPtr;
    std::tie(SrcShadowPtr, SrcOriginPtr) =
        getShadowOriginPtr(SrcPtr, NextIRB, NextIRB.getInt8Ty(), Align(1),
                           /*isStore*/ false);
    Value *DstShadowPtr =
        getShadowOriginPtr(DstPtr, NextIRB, NextIRB.getInt8Ty(), Align(1),
                           /*isStore*/ true)
            .first;

    NextIRB.CreateMemCpy(DstShadowPtr, Align(1), SrcShadowPtr, Align(1), Size);
    if (MS.TrackOrigins) {
      Value *SrcOrigin = NextIRB.CreateAlignedLoad(MS.OriginTy, SrcOriginPtr,
                                                   kMinOriginAlignment);
      Value *NewOrigin = updateOrigin(SrcOrigin, NextIRB);
      NextIRB.CreateCall(MS.MsanSetOriginFn, {DstPtr, Size, NewOrigin});
    }
  }

  void visitLibAtomicStore(CallBase &CB) {
    IRBuilder<> IRB(&CB);
    Value *Size = CB.getArgOperand(0);
    Value *DstPtr = CB.getArgOperand(2);
    Value *Ordering = CB.getArgOperand(3);
    // Convert the call to have at least Release ordering to make sure
    // the shadow operations aren't reordered after it.
    Value *NewOrdering =
        IRB.CreateExtractElement(makeAddReleaseOrderingTable(IRB), Ordering);
    CB.setArgOperand(3, NewOrdering);

    Value *DstShadowPtr =
        getShadowOriginPtr(DstPtr, IRB, IRB.getInt8Ty(), Align(1),
                           /*isStore*/ true)
            .first;

    // Atomic store always paints clean shadow/origin. See file header.
    IRB.CreateMemSet(DstShadowPtr, getCleanShadow(IRB.getInt8Ty()), Size,
                     Align(1));
  }

  void visitCallBase(CallBase &CB) {
    assert(!CB.getMetadata("nosanitize"));
    if (CB.isInlineAsm()) {
      // For inline asm (either a call to asm function, or callbr instruction),
      // do the usual thing: check argument shadow and mark all outputs as
      // clean. Note that any side effects of the inline asm that are not
      // immediately visible in its constraints are not handled.
      if (ClHandleAsmConservative && MS.CompileKernel)
        visitAsmInstruction(CB);
      else
        visitInstruction(CB);
      return;
    }
    LibFunc LF;
    if (TLI->getLibFunc(CB, LF)) {
      // libatomic.a functions need to have special handling because there isn't
      // a good way to intercept them or compile the library with
      // instrumentation.
      switch (LF) {
      case LibFunc_atomic_load:
        if (!isa<CallInst>(CB)) {
          llvm::errs() << "MSAN -- cannot instrument invoke of libatomic load."
                          "Ignoring!\n";
          break;
        }
        visitLibAtomicLoad(CB);
        return;
      case LibFunc_atomic_store:
        visitLibAtomicStore(CB);
        return;
      default:
        break;
      }
    }

    if (auto *Call = dyn_cast<CallInst>(&CB)) {
      assert(!isa<IntrinsicInst>(Call) && "intrinsics are handled elsewhere");

      // We are going to insert code that relies on the fact that the callee
      // will become a non-readonly function after it is instrumented by us. To
      // prevent this code from being optimized out, mark that function
      // non-readonly in advance.
      AttributeMask B;
      B.addAttribute(Attribute::ReadOnly)
          .addAttribute(Attribute::ReadNone)
          .addAttribute(Attribute::WriteOnly)
          .addAttribute(Attribute::ArgMemOnly)
          .addAttribute(Attribute::Speculatable);

      Call->removeFnAttrs(B);
      if (Function *Func = Call->getCalledFunction()) {
        Func->removeFnAttrs(B);
      }

      maybeMarkSanitizerLibraryCallNoBuiltin(Call, TLI);
    }
    IRBuilder<> IRB(&CB);
    bool MayCheckCall = MS.EagerChecks;
    if (Function *Func = CB.getCalledFunction()) {
      // __sanitizer_unaligned_{load,store} functions may be called by users
      // and always expects shadows in the TLS. So don't check them.
      MayCheckCall &= !Func->getName().startswith("__sanitizer_unaligned_");
    }

    unsigned ArgOffset = 0;
    LLVM_DEBUG(dbgs() << "  CallSite: " << CB << "\n");
    for (auto ArgIt = CB.arg_begin(), End = CB.arg_end(); ArgIt != End;
         ++ArgIt) {
      Value *A = *ArgIt;
      unsigned i = ArgIt - CB.arg_begin();
      if (!A->getType()->isSized()) {
        LLVM_DEBUG(dbgs() << "Arg " << i << " is not sized: " << CB << "\n");
        continue;
      }
      unsigned Size = 0;
      const DataLayout &DL = F.getParent()->getDataLayout();

      bool ByVal = CB.paramHasAttr(i, Attribute::ByVal);
      bool NoUndef = CB.paramHasAttr(i, Attribute::NoUndef);
      bool EagerCheck = MayCheckCall && !ByVal && NoUndef;

      if (EagerCheck) {
        insertShadowCheck(A, &CB);
        Size = DL.getTypeAllocSize(A->getType());
      } else {
        Value *Store = nullptr;
        // Compute the Shadow for arg even if it is ByVal, because
        // in that case getShadow() will copy the actual arg shadow to
        // __msan_param_tls.
        Value *ArgShadow = getShadow(A);
        Value *ArgShadowBase = getShadowPtrForArgument(A, IRB, ArgOffset);
        LLVM_DEBUG(dbgs() << "  Arg#" << i << ": " << *A
                          << " Shadow: " << *ArgShadow << "\n");
        if (ByVal) {
          // ByVal requires some special handling as it's too big for a single
          // load
          assert(A->getType()->isPointerTy() &&
                 "ByVal argument is not a pointer!");
          Size = DL.getTypeAllocSize(CB.getParamByValType(i));
          if (ArgOffset + Size > kParamTLSSize)
            break;
          const MaybeAlign ParamAlignment(CB.getParamAlign(i));
          MaybeAlign Alignment = llvm::None;
          if (ParamAlignment)
            Alignment = std::min(*ParamAlignment, kShadowTLSAlignment);
          Value *AShadowPtr, *AOriginPtr;
          std::tie(AShadowPtr, AOriginPtr) =
              getShadowOriginPtr(A, IRB, IRB.getInt8Ty(), Alignment,
                                 /*isStore*/ false);
          if (!PropagateShadow) {
            Store = IRB.CreateMemSet(ArgShadowBase,
                                     Constant::getNullValue(IRB.getInt8Ty()),
                                     Size, Alignment);
          } else {
            Store = IRB.CreateMemCpy(ArgShadowBase, Alignment, AShadowPtr,
                                     Alignment, Size);
            if (MS.TrackOrigins) {
              Value *ArgOriginBase = getOriginPtrForArgument(A, IRB, ArgOffset);
              // FIXME: OriginSize should be:
              // alignTo(A % kMinOriginAlignment + Size, kMinOriginAlignment)
              unsigned OriginSize = alignTo(Size, kMinOriginAlignment);
              IRB.CreateMemCpy(
                  ArgOriginBase,
                  /* by origin_tls[ArgOffset] */ kMinOriginAlignment,
                  AOriginPtr,
                  /* by getShadowOriginPtr */ kMinOriginAlignment, OriginSize);
            }
          }
        } else {
          // Any other parameters mean we need bit-grained tracking of uninit
          // data
          Size = DL.getTypeAllocSize(A->getType());
          if (ArgOffset + Size > kParamTLSSize)
            break;
          Store = IRB.CreateAlignedStore(ArgShadow, ArgShadowBase,
                                         kShadowTLSAlignment);
          Constant *Cst = dyn_cast<Constant>(ArgShadow);
          if (MS.TrackOrigins && !(Cst && Cst->isNullValue())) {
            IRB.CreateStore(getOrigin(A),
                            getOriginPtrForArgument(A, IRB, ArgOffset));
          }
        }
        (void)Store;
        assert(Store != nullptr);
        LLVM_DEBUG(dbgs() << "  Param:" << *Store << "\n");
      }
      assert(Size != 0);
      ArgOffset += alignTo(Size, kShadowTLSAlignment);
    }
    LLVM_DEBUG(dbgs() << "  done with call args\n");

    FunctionType *FT = CB.getFunctionType();
    if (FT->isVarArg()) {
      VAHelper->visitCallBase(CB, IRB);
    }

    // Now, get the shadow for the RetVal.
    if (!CB.getType()->isSized())
      return;
    // Don't emit the epilogue for musttail call returns.
    if (isa<CallInst>(CB) && cast<CallInst>(CB).isMustTailCall())
      return;

    if (MayCheckCall && CB.hasRetAttr(Attribute::NoUndef)) {
      setShadow(&CB, getCleanShadow(&CB));
      setOrigin(&CB, getCleanOrigin());
      return;
    }

    IRBuilder<> IRBBefore(&CB);
    // Until we have full dynamic coverage, make sure the retval shadow is 0.
    Value *Base = getShadowPtrForRetval(&CB, IRBBefore);
    IRBBefore.CreateAlignedStore(getCleanShadow(&CB), Base,
                                 kShadowTLSAlignment);
    BasicBlock::iterator NextInsn;
    if (isa<CallInst>(CB)) {
      NextInsn = ++CB.getIterator();
      assert(NextInsn != CB.getParent()->end());
    } else {
      BasicBlock *NormalDest = cast<InvokeInst>(CB).getNormalDest();
      if (!NormalDest->getSinglePredecessor()) {
        // FIXME: this case is tricky, so we are just conservative here.
        // Perhaps we need to split the edge between this BB and NormalDest,
        // but a naive attempt to use SplitEdge leads to a crash.
        setShadow(&CB, getCleanShadow(&CB));
        setOrigin(&CB, getCleanOrigin());
        return;
      }
      // FIXME: NextInsn is likely in a basic block that has not been visited yet.
      // Anything inserted there will be instrumented by MSan later!
      NextInsn = NormalDest->getFirstInsertionPt();
      assert(NextInsn != NormalDest->end() &&
             "Could not find insertion point for retval shadow load");
    }
    IRBuilder<> IRBAfter(&*NextInsn);
    Value *RetvalShadow = IRBAfter.CreateAlignedLoad(
        getShadowTy(&CB), getShadowPtrForRetval(&CB, IRBAfter),
        kShadowTLSAlignment, "_msret");
    setShadow(&CB, RetvalShadow);
    if (MS.TrackOrigins)
      setOrigin(&CB, IRBAfter.CreateLoad(MS.OriginTy,
                                         getOriginPtrForRetval(IRBAfter)));
  }

  bool isAMustTailRetVal(Value *RetVal) {
    if (auto *I = dyn_cast<BitCastInst>(RetVal)) {
      RetVal = I->getOperand(0);
    }
    if (auto *I = dyn_cast<CallInst>(RetVal)) {
      return I->isMustTailCall();
    }
    return false;
  }

  void visitReturnInst(ReturnInst &I) {
    IRBuilder<> IRB(&I);
    Value *RetVal = I.getReturnValue();
    if (!RetVal) return;
    // Don't emit the epilogue for musttail call returns.
    if (isAMustTailRetVal(RetVal)) return;
    Value *ShadowPtr = getShadowPtrForRetval(RetVal, IRB);
    bool HasNoUndef =
        F.hasRetAttribute(Attribute::NoUndef);
    bool StoreShadow = !(MS.EagerChecks && HasNoUndef);
    // FIXME: Consider using SpecialCaseList to specify a list of functions that
    // must always return fully initialized values. For now, we hardcode "main".
    bool EagerCheck = (MS.EagerChecks && HasNoUndef) || (F.getName() == "main");

    Value *Shadow = getShadow(RetVal);
    bool StoreOrigin = true;
    if (EagerCheck) {
      insertShadowCheck(RetVal, &I);
      Shadow = getCleanShadow(RetVal);
      StoreOrigin = false;
    }

    // The caller may still expect information passed over TLS if we pass our
    // check
    if (StoreShadow) {
      IRB.CreateAlignedStore(Shadow, ShadowPtr, kShadowTLSAlignment);
      if (MS.TrackOrigins && StoreOrigin)
        IRB.CreateStore(getOrigin(RetVal), getOriginPtrForRetval(IRB));
    }
  }

  void visitPHINode(PHINode &I) {
    IRBuilder<> IRB(&I);
    if (!PropagateShadow) {
      setShadow(&I, getCleanShadow(&I));
      setOrigin(&I, getCleanOrigin());
      return;
    }

    ShadowPHINodes.push_back(&I);
    setShadow(&I, IRB.CreatePHI(getShadowTy(&I), I.getNumIncomingValues(),
                                "_msphi_s"));
    if (MS.TrackOrigins)
      setOrigin(&I, IRB.CreatePHI(MS.OriginTy, I.getNumIncomingValues(),
                                  "_msphi_o"));
  }

  Value *getLocalVarDescription(AllocaInst &I) {
    SmallString<2048> StackDescriptionStorage;
    raw_svector_ostream StackDescription(StackDescriptionStorage);
    // We create a string with a description of the stack allocation and
    // pass it into __msan_set_alloca_origin.
    // It will be printed by the run-time if stack-originated UMR is found.
    // The first 4 bytes of the string are set to '----' and will be replaced
    // by __msan_va_arg_overflow_size_tls at the first call.
    StackDescription << "----" << I.getName() << "@" << F.getName();
    return createPrivateNonConstGlobalForString(*F.getParent(),
                                                StackDescription.str());
  }

  void poisonAllocaUserspace(AllocaInst &I, IRBuilder<> &IRB, Value *Len) {
    if (PoisonStack && ClPoisonStackWithCall) {
      IRB.CreateCall(MS.MsanPoisonStackFn,
                     {IRB.CreatePointerCast(&I, IRB.getInt8PtrTy()), Len});
    } else {
      Value *ShadowBase, *OriginBase;
      std::tie(ShadowBase, OriginBase) = getShadowOriginPtr(
          &I, IRB, IRB.getInt8Ty(), Align(1), /*isStore*/ true);

      Value *PoisonValue = IRB.getInt8(PoisonStack ? ClPoisonStackPattern : 0);
      IRB.CreateMemSet(ShadowBase, PoisonValue, Len, I.getAlign());
    }

    if (PoisonStack && MS.TrackOrigins) {
      Value *Descr = getLocalVarDescription(I);
      IRB.CreateCall(MS.MsanSetAllocaOrigin4Fn,
                     {IRB.CreatePointerCast(&I, IRB.getInt8PtrTy()), Len,
                      IRB.CreatePointerCast(Descr, IRB.getInt8PtrTy()),
                      IRB.CreatePointerCast(&F, MS.IntptrTy)});
    }
  }

  void poisonAllocaKmsan(AllocaInst &I, IRBuilder<> &IRB, Value *Len) {
    Value *Descr = getLocalVarDescription(I);
    if (PoisonStack) {
      IRB.CreateCall(MS.MsanPoisonAllocaFn,
                     {IRB.CreatePointerCast(&I, IRB.getInt8PtrTy()), Len,
                      IRB.CreatePointerCast(Descr, IRB.getInt8PtrTy())});
    } else {
      IRB.CreateCall(MS.MsanUnpoisonAllocaFn,
                     {IRB.CreatePointerCast(&I, IRB.getInt8PtrTy()), Len});
    }
  }

  void instrumentAlloca(AllocaInst &I, Instruction *InsPoint = nullptr) {
    if (!InsPoint)
      InsPoint = &I;
    IRBuilder<> IRB(InsPoint->getNextNode());
    const DataLayout &DL = F.getParent()->getDataLayout();
    uint64_t TypeSize = DL.getTypeAllocSize(I.getAllocatedType());
    Value *Len = ConstantInt::get(MS.IntptrTy, TypeSize);
    if (I.isArrayAllocation())
      Len = IRB.CreateMul(Len, I.getArraySize());

    if (MS.CompileKernel)
      poisonAllocaKmsan(I, IRB, Len);
    else
      poisonAllocaUserspace(I, IRB, Len);
  }

  void visitAllocaInst(AllocaInst &I) {
    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
    // We'll get to this alloca later unless it's poisoned at the corresponding
    // llvm.lifetime.start.
    AllocaSet.insert(&I);
  }

  void visitSelectInst(SelectInst& I) {
    IRBuilder<> IRB(&I);
    // a = select b, c, d
    Value *B = I.getCondition();
    Value *C = I.getTrueValue();
    Value *D = I.getFalseValue();
    Value *Sb = getShadow(B);
    Value *Sc = getShadow(C);
    Value *Sd = getShadow(D);

    // Result shadow if condition shadow is 0.
    Value *Sa0 = IRB.CreateSelect(B, Sc, Sd);
    Value *Sa1;
    if (I.getType()->isAggregateType()) {
      // To avoid "sign extending" i1 to an arbitrary aggregate type, we just do
      // an extra "select". This results in much more compact IR.
      // Sa = select Sb, poisoned, (select b, Sc, Sd)
      Sa1 = getPoisonedShadow(getShadowTy(I.getType()));
    } else {
      // Sa = select Sb, [ (c^d) | Sc | Sd ], [ b ? Sc : Sd ]
      // If Sb (condition is poisoned), look for bits in c and d that are equal
      // and both unpoisoned.
      // If !Sb (condition is unpoisoned), simply pick one of Sc and Sd.

      // Cast arguments to shadow-compatible type.
      C = CreateAppToShadowCast(IRB, C);
      D = CreateAppToShadowCast(IRB, D);

      // Result shadow if condition shadow is 1.
      Sa1 = IRB.CreateOr({IRB.CreateXor(C, D), Sc, Sd});
    }
    Value *Sa = IRB.CreateSelect(Sb, Sa1, Sa0, "_msprop_select");
    setShadow(&I, Sa);
    if (MS.TrackOrigins) {
      // Origins are always i32, so any vector conditions must be flattened.
      // FIXME: consider tracking vector origins for app vectors?
      if (B->getType()->isVectorTy()) {
        Type *FlatTy = getShadowTyNoVec(B->getType());
        B = IRB.CreateICmpNE(IRB.CreateBitCast(B, FlatTy),
                                ConstantInt::getNullValue(FlatTy));
        Sb = IRB.CreateICmpNE(IRB.CreateBitCast(Sb, FlatTy),
                                      ConstantInt::getNullValue(FlatTy));
      }
      // a = select b, c, d
      // Oa = Sb ? Ob : (b ? Oc : Od)
      setOrigin(
          &I, IRB.CreateSelect(Sb, getOrigin(I.getCondition()),
                               IRB.CreateSelect(B, getOrigin(I.getTrueValue()),
                                                getOrigin(I.getFalseValue()))));
    }
  }

  void visitLandingPadInst(LandingPadInst &I) {
    // Do nothing.
    // See https://github.com/google/sanitizers/issues/504
    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
  }

  void visitCatchSwitchInst(CatchSwitchInst &I) {
    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
  }

  void visitFuncletPadInst(FuncletPadInst &I) {
    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
  }

  void visitGetElementPtrInst(GetElementPtrInst &I) {
    handleShadowOr(I);
  }

  void visitExtractValueInst(ExtractValueInst &I) {
    IRBuilder<> IRB(&I);
    Value *Agg = I.getAggregateOperand();
    LLVM_DEBUG(dbgs() << "ExtractValue:  " << I << "\n");
    Value *AggShadow = getShadow(Agg);
    LLVM_DEBUG(dbgs() << "   AggShadow:  " << *AggShadow << "\n");
    Value *ResShadow = IRB.CreateExtractValue(AggShadow, I.getIndices());
    LLVM_DEBUG(dbgs() << "   ResShadow:  " << *ResShadow << "\n");
    setShadow(&I, ResShadow);
    setOriginForNaryOp(I);
  }

  void visitInsertValueInst(InsertValueInst &I) {
    IRBuilder<> IRB(&I);
    LLVM_DEBUG(dbgs() << "InsertValue:  " << I << "\n");
    Value *AggShadow = getShadow(I.getAggregateOperand());
    Value *InsShadow = getShadow(I.getInsertedValueOperand());
    LLVM_DEBUG(dbgs() << "   AggShadow:  " << *AggShadow << "\n");
    LLVM_DEBUG(dbgs() << "   InsShadow:  " << *InsShadow << "\n");
    Value *Res = IRB.CreateInsertValue(AggShadow, InsShadow, I.getIndices());
    LLVM_DEBUG(dbgs() << "   Res:        " << *Res << "\n");
    setShadow(&I, Res);
    setOriginForNaryOp(I);
  }

  void dumpInst(Instruction &I) {
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      errs() << "ZZZ call " << CI->getCalledFunction()->getName() << "\n";
    } else {
      errs() << "ZZZ " << I.getOpcodeName() << "\n";
    }
    errs() << "QQQ " << I << "\n";
  }

  void visitResumeInst(ResumeInst &I) {
    LLVM_DEBUG(dbgs() << "Resume: " << I << "\n");
    // Nothing to do here.
  }

  void visitCleanupReturnInst(CleanupReturnInst &CRI) {
    LLVM_DEBUG(dbgs() << "CleanupReturn: " << CRI << "\n");
    // Nothing to do here.
  }

  void visitCatchReturnInst(CatchReturnInst &CRI) {
    LLVM_DEBUG(dbgs() << "CatchReturn: " << CRI << "\n");
    // Nothing to do here.
  }

  void instrumentAsmArgument(Value *Operand, Type *ElemTy, Instruction &I,
                             IRBuilder<> &IRB, const DataLayout &DL,
                             bool isOutput) {
    // For each assembly argument, we check its value for being initialized.
    // If the argument is a pointer, we assume it points to a single element
    // of the corresponding type (or to a 8-byte word, if the type is unsized).
    // Each such pointer is instrumented with a call to the runtime library.
    Type *OpType = Operand->getType();
    // Check the operand value itself.
    insertShadowCheck(Operand, &I);
    if (!OpType->isPointerTy() || !isOutput) {
      assert(!isOutput);
      return;
    }
    if (!ElemTy->isSized())
      return;
    int Size = DL.getTypeStoreSize(ElemTy);
    Value *Ptr = IRB.CreatePointerCast(Operand, IRB.getInt8PtrTy());
    Value *SizeVal = ConstantInt::get(MS.IntptrTy, Size);
    IRB.CreateCall(MS.MsanInstrumentAsmStoreFn, {Ptr, SizeVal});
  }

  /// Get the number of output arguments returned by pointers.
  int getNumOutputArgs(InlineAsm *IA, CallBase *CB) {
    int NumRetOutputs = 0;
    int NumOutputs = 0;
    Type *RetTy = cast<Value>(CB)->getType();
    if (!RetTy->isVoidTy()) {
      // Register outputs are returned via the CallInst return value.
      auto *ST = dyn_cast<StructType>(RetTy);
      if (ST)
        NumRetOutputs = ST->getNumElements();
      else
        NumRetOutputs = 1;
    }
    InlineAsm::ConstraintInfoVector Constraints = IA->ParseConstraints();
    for (const InlineAsm::ConstraintInfo &Info : Constraints) {
      switch (Info.Type) {
      case InlineAsm::isOutput:
        NumOutputs++;
        break;
      default:
        break;
      }
    }
    return NumOutputs - NumRetOutputs;
  }

  void visitAsmInstruction(Instruction &I) {
    // Conservative inline assembly handling: check for poisoned shadow of
    // asm() arguments, then unpoison the result and all the memory locations
    // pointed to by those arguments.
    // An inline asm() statement in C++ contains lists of input and output
    // arguments used by the assembly code. These are mapped to operands of the
    // CallInst as follows:
    //  - nR register outputs ("=r) are returned by value in a single structure
    //  (SSA value of the CallInst);
    //  - nO other outputs ("=m" and others) are returned by pointer as first
    // nO operands of the CallInst;
    //  - nI inputs ("r", "m" and others) are passed to CallInst as the
    // remaining nI operands.
    // The total number of asm() arguments in the source is nR+nO+nI, and the
    // corresponding CallInst has nO+nI+1 operands (the last operand is the
    // function to be called).
    const DataLayout &DL = F.getParent()->getDataLayout();
    CallBase *CB = cast<CallBase>(&I);
    IRBuilder<> IRB(&I);
    InlineAsm *IA = cast<InlineAsm>(CB->getCalledOperand());
    int OutputArgs = getNumOutputArgs(IA, CB);
    // The last operand of a CallInst is the function itself.
    int NumOperands = CB->getNumOperands() - 1;

    // Check input arguments. Doing so before unpoisoning output arguments, so
    // that we won't overwrite uninit values before checking them.
    for (int i = OutputArgs; i < NumOperands; i++) {
      Value *Operand = CB->getOperand(i);
      instrumentAsmArgument(Operand, CB->getParamElementType(i), I, IRB, DL,
                            /*isOutput*/ false);
    }
    // Unpoison output arguments. This must happen before the actual InlineAsm
    // call, so that the shadow for memory published in the asm() statement
    // remains valid.
    for (int i = 0; i < OutputArgs; i++) {
      Value *Operand = CB->getOperand(i);
      instrumentAsmArgument(Operand, CB->getParamElementType(i), I, IRB, DL,
                            /*isOutput*/ true);
    }

    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
  }

  void visitFreezeInst(FreezeInst &I) {
    // Freeze always returns a fully defined value.
    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
  }

  void visitInstruction(Instruction &I) {
    // Everything else: stop propagating and check for poisoned shadow.
    if (ClDumpStrictInstructions)
      dumpInst(I);
    LLVM_DEBUG(dbgs() << "DEFAULT: " << I << "\n");
    for (size_t i = 0, n = I.getNumOperands(); i < n; i++) {
      Value *Operand = I.getOperand(i);
      if (Operand->getType()->isSized())
        insertShadowCheck(Operand, &I);
    }
    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
  }
};

/// AMD64-specific implementation of VarArgHelper.
struct VarArgAMD64Helper : public VarArgHelper {
  // An unfortunate workaround for asymmetric lowering of va_arg stuff.
  // See a comment in visitCallBase for more details.
  static const unsigned AMD64GpEndOffset = 48;  // AMD64 ABI Draft 0.99.6 p3.5.7
  static const unsigned AMD64FpEndOffsetSSE = 176;
  // If SSE is disabled, fp_offset in va_list is zero.
  static const unsigned AMD64FpEndOffsetNoSSE = AMD64GpEndOffset;

  unsigned AMD64FpEndOffset;
  Function &F;
  MemorySanitizer &MS;
  MemorySanitizerVisitor &MSV;
  Value *VAArgTLSCopy = nullptr;
  Value *VAArgTLSOriginCopy = nullptr;
  Value *VAArgOverflowSize = nullptr;

  SmallVector<CallInst*, 16> VAStartInstrumentationList;

  enum ArgKind { AK_GeneralPurpose, AK_FloatingPoint, AK_Memory };

  VarArgAMD64Helper(Function &F, MemorySanitizer &MS,
                    MemorySanitizerVisitor &MSV)
      : F(F), MS(MS), MSV(MSV) {
    AMD64FpEndOffset = AMD64FpEndOffsetSSE;
    for (const auto &Attr : F.getAttributes().getFnAttrs()) {
      if (Attr.isStringAttribute() &&
          (Attr.getKindAsString() == "target-features")) {
        if (Attr.getValueAsString().contains("-sse"))
          AMD64FpEndOffset = AMD64FpEndOffsetNoSSE;
        break;
      }
    }
  }

  ArgKind classifyArgument(Value* arg) {
    // A very rough approximation of X86_64 argument classification rules.
    Type *T = arg->getType();
    if (T->isFPOrFPVectorTy() || T->isX86_MMXTy())
      return AK_FloatingPoint;
    if (T->isIntegerTy() && T->getPrimitiveSizeInBits() <= 64)
      return AK_GeneralPurpose;
    if (T->isPointerTy())
      return AK_GeneralPurpose;
    return AK_Memory;
  }

  // For VarArg functions, store the argument shadow in an ABI-specific format
  // that corresponds to va_list layout.
  // We do this because Clang lowers va_arg in the frontend, and this pass
  // only sees the low level code that deals with va_list internals.
  // A much easier alternative (provided that Clang emits va_arg instructions)
  // would have been to associate each live instance of va_list with a copy of
  // MSanParamTLS, and extract shadow on va_arg() call in the argument list
  // order.
  void visitCallBase(CallBase &CB, IRBuilder<> &IRB) override {
    unsigned GpOffset = 0;
    unsigned FpOffset = AMD64GpEndOffset;
    unsigned OverflowOffset = AMD64FpEndOffset;
    const DataLayout &DL = F.getParent()->getDataLayout();
    for (auto ArgIt = CB.arg_begin(), End = CB.arg_end(); ArgIt != End;
         ++ArgIt) {
      Value *A = *ArgIt;
      unsigned ArgNo = CB.getArgOperandNo(ArgIt);
      bool IsFixed = ArgNo < CB.getFunctionType()->getNumParams();
      bool IsByVal = CB.paramHasAttr(ArgNo, Attribute::ByVal);
      if (IsByVal) {
        // ByVal arguments always go to the overflow area.
        // Fixed arguments passed through the overflow area will be stepped
        // over by va_start, so don't count them towards the offset.
        if (IsFixed)
          continue;
        assert(A->getType()->isPointerTy());
        Type *RealTy = CB.getParamByValType(ArgNo);
        uint64_t ArgSize = DL.getTypeAllocSize(RealTy);
        Value *ShadowBase = getShadowPtrForVAArgument(
            RealTy, IRB, OverflowOffset, alignTo(ArgSize, 8));
        Value *OriginBase = nullptr;
        if (MS.TrackOrigins)
          OriginBase = getOriginPtrForVAArgument(RealTy, IRB, OverflowOffset);
        OverflowOffset += alignTo(ArgSize, 8);
        if (!ShadowBase)
          continue;
        Value *ShadowPtr, *OriginPtr;
        std::tie(ShadowPtr, OriginPtr) =
            MSV.getShadowOriginPtr(A, IRB, IRB.getInt8Ty(), kShadowTLSAlignment,
                                   /*isStore*/ false);

        IRB.CreateMemCpy(ShadowBase, kShadowTLSAlignment, ShadowPtr,
                         kShadowTLSAlignment, ArgSize);
        if (MS.TrackOrigins)
          IRB.CreateMemCpy(OriginBase, kShadowTLSAlignment, OriginPtr,
                           kShadowTLSAlignment, ArgSize);
      } else {
        ArgKind AK = classifyArgument(A);
        if (AK == AK_GeneralPurpose && GpOffset >= AMD64GpEndOffset)
          AK = AK_Memory;
        if (AK == AK_FloatingPoint && FpOffset >= AMD64FpEndOffset)
          AK = AK_Memory;
        Value *ShadowBase, *OriginBase = nullptr;
        switch (AK) {
          case AK_GeneralPurpose:
            ShadowBase =
                getShadowPtrForVAArgument(A->getType(), IRB, GpOffset, 8);
            if (MS.TrackOrigins)
              OriginBase =
                  getOriginPtrForVAArgument(A->getType(), IRB, GpOffset);
            GpOffset += 8;
            break;
          case AK_FloatingPoint:
            ShadowBase =
                getShadowPtrForVAArgument(A->getType(), IRB, FpOffset, 16);
            if (MS.TrackOrigins)
              OriginBase =
                  getOriginPtrForVAArgument(A->getType(), IRB, FpOffset);
            FpOffset += 16;
            break;
          case AK_Memory:
            if (IsFixed)
              continue;
            uint64_t ArgSize = DL.getTypeAllocSize(A->getType());
            ShadowBase =
                getShadowPtrForVAArgument(A->getType(), IRB, OverflowOffset, 8);
            if (MS.TrackOrigins)
              OriginBase =
                  getOriginPtrForVAArgument(A->getType(), IRB, OverflowOffset);
            OverflowOffset += alignTo(ArgSize, 8);
        }
        // Take fixed arguments into account for GpOffset and FpOffset,
        // but don't actually store shadows for them.
        // TODO(glider): don't call get*PtrForVAArgument() for them.
        if (IsFixed)
          continue;
        if (!ShadowBase)
          continue;
        Value *Shadow = MSV.getShadow(A);
        IRB.CreateAlignedStore(Shadow, ShadowBase, kShadowTLSAlignment);
        if (MS.TrackOrigins) {
          Value *Origin = MSV.getOrigin(A);
          unsigned StoreSize = DL.getTypeStoreSize(Shadow->getType());
          MSV.paintOrigin(IRB, Origin, OriginBase, StoreSize,
                          std::max(kShadowTLSAlignment, kMinOriginAlignment));
        }
      }
    }
    Constant *OverflowSize =
      ConstantInt::get(IRB.getInt64Ty(), OverflowOffset - AMD64FpEndOffset);
    IRB.CreateStore(OverflowSize, MS.VAArgOverflowSizeTLS);
  }

  /// Compute the shadow address for a given va_arg.
  Value *getShadowPtrForVAArgument(Type *Ty, IRBuilder<> &IRB,
                                   unsigned ArgOffset, unsigned ArgSize) {
    // Make sure we don't overflow __msan_va_arg_tls.
    if (ArgOffset + ArgSize > kParamTLSSize)
      return nullptr;
    Value *Base = IRB.CreatePointerCast(MS.VAArgTLS, MS.IntptrTy);
    Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(MSV.getShadowTy(Ty), 0),
                              "_msarg_va_s");
  }

  /// Compute the origin address for a given va_arg.
  Value *getOriginPtrForVAArgument(Type *Ty, IRBuilder<> &IRB, int ArgOffset) {
    Value *Base = IRB.CreatePointerCast(MS.VAArgOriginTLS, MS.IntptrTy);
    // getOriginPtrForVAArgument() is always called after
    // getShadowPtrForVAArgument(), so __msan_va_arg_origin_tls can never
    // overflow.
    Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(MS.OriginTy, 0),
                              "_msarg_va_o");
  }

  void unpoisonVAListTagForInst(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr, *OriginPtr;
    const Align Alignment = Align(8);
    std::tie(ShadowPtr, OriginPtr) =
        MSV.getShadowOriginPtr(VAListTag, IRB, IRB.getInt8Ty(), Alignment,
                               /*isStore*/ true);

    // Unpoison the whole __va_list_tag.
    // FIXME: magic ABI constants.
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */ 24, Alignment, false);
    // We shouldn't need to zero out the origins, as they're only checked for
    // nonzero shadow.
  }

  void visitVAStartInst(VAStartInst &I) override {
    if (F.getCallingConv() == CallingConv::Win64)
      return;
    VAStartInstrumentationList.push_back(&I);
    unpoisonVAListTagForInst(I);
  }

  void visitVACopyInst(VACopyInst &I) override {
    if (F.getCallingConv() == CallingConv::Win64) return;
    unpoisonVAListTagForInst(I);
  }

  void finalizeInstrumentation() override {
    assert(!VAArgOverflowSize && !VAArgTLSCopy &&
           "finalizeInstrumentation called twice");
    if (!VAStartInstrumentationList.empty()) {
      // If there is a va_start in this function, make a backup copy of
      // va_arg_tls somewhere in the function entry block.
      IRBuilder<> IRB(MSV.FnPrologueEnd);
      VAArgOverflowSize =
          IRB.CreateLoad(IRB.getInt64Ty(), MS.VAArgOverflowSizeTLS);
      Value *CopySize =
        IRB.CreateAdd(ConstantInt::get(MS.IntptrTy, AMD64FpEndOffset),
                      VAArgOverflowSize);
      VAArgTLSCopy = IRB.CreateAlloca(Type::getInt8Ty(*MS.C), CopySize);
      IRB.CreateMemCpy(VAArgTLSCopy, Align(8), MS.VAArgTLS, Align(8), CopySize);
      if (MS.TrackOrigins) {
        VAArgTLSOriginCopy = IRB.CreateAlloca(Type::getInt8Ty(*MS.C), CopySize);
        IRB.CreateMemCpy(VAArgTLSOriginCopy, Align(8), MS.VAArgOriginTLS,
                         Align(8), CopySize);
      }
    }

    // Instrument va_start.
    // Copy va_list shadow from the backup copy of the TLS contents.
    for (size_t i = 0, n = VAStartInstrumentationList.size(); i < n; i++) {
      CallInst *OrigInst = VAStartInstrumentationList[i];
      IRBuilder<> IRB(OrigInst->getNextNode());
      Value *VAListTag = OrigInst->getArgOperand(0);

      Type *RegSaveAreaPtrTy = Type::getInt64PtrTy(*MS.C);
      Value *RegSaveAreaPtrPtr = IRB.CreateIntToPtr(
          IRB.CreateAdd(IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
                        ConstantInt::get(MS.IntptrTy, 16)),
          PointerType::get(RegSaveAreaPtrTy, 0));
      Value *RegSaveAreaPtr =
          IRB.CreateLoad(RegSaveAreaPtrTy, RegSaveAreaPtrPtr);
      Value *RegSaveAreaShadowPtr, *RegSaveAreaOriginPtr;
      const Align Alignment = Align(16);
      std::tie(RegSaveAreaShadowPtr, RegSaveAreaOriginPtr) =
          MSV.getShadowOriginPtr(RegSaveAreaPtr, IRB, IRB.getInt8Ty(),
                                 Alignment, /*isStore*/ true);
      IRB.CreateMemCpy(RegSaveAreaShadowPtr, Alignment, VAArgTLSCopy, Alignment,
                       AMD64FpEndOffset);
      if (MS.TrackOrigins)
        IRB.CreateMemCpy(RegSaveAreaOriginPtr, Alignment, VAArgTLSOriginCopy,
                         Alignment, AMD64FpEndOffset);
      Type *OverflowArgAreaPtrTy = Type::getInt64PtrTy(*MS.C);
      Value *OverflowArgAreaPtrPtr = IRB.CreateIntToPtr(
          IRB.CreateAdd(IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
                        ConstantInt::get(MS.IntptrTy, 8)),
          PointerType::get(OverflowArgAreaPtrTy, 0));
      Value *OverflowArgAreaPtr =
          IRB.CreateLoad(OverflowArgAreaPtrTy, OverflowArgAreaPtrPtr);
      Value *OverflowArgAreaShadowPtr, *OverflowArgAreaOriginPtr;
      std::tie(OverflowArgAreaShadowPtr, OverflowArgAreaOriginPtr) =
          MSV.getShadowOriginPtr(OverflowArgAreaPtr, IRB, IRB.getInt8Ty(),
                                 Alignment, /*isStore*/ true);
      Value *SrcPtr = IRB.CreateConstGEP1_32(IRB.getInt8Ty(), VAArgTLSCopy,
                                             AMD64FpEndOffset);
      IRB.CreateMemCpy(OverflowArgAreaShadowPtr, Alignment, SrcPtr, Alignment,
                       VAArgOverflowSize);
      if (MS.TrackOrigins) {
        SrcPtr = IRB.CreateConstGEP1_32(IRB.getInt8Ty(), VAArgTLSOriginCopy,
                                        AMD64FpEndOffset);
        IRB.CreateMemCpy(OverflowArgAreaOriginPtr, Alignment, SrcPtr, Alignment,
                         VAArgOverflowSize);
      }
    }
  }
};

/// MIPS64-specific implementation of VarArgHelper.
struct VarArgMIPS64Helper : public VarArgHelper {
  Function &F;
  MemorySanitizer &MS;
  MemorySanitizerVisitor &MSV;
  Value *VAArgTLSCopy = nullptr;
  Value *VAArgSize = nullptr;

  SmallVector<CallInst*, 16> VAStartInstrumentationList;

  VarArgMIPS64Helper(Function &F, MemorySanitizer &MS,
                    MemorySanitizerVisitor &MSV) : F(F), MS(MS), MSV(MSV) {}

  void visitCallBase(CallBase &CB, IRBuilder<> &IRB) override {
    unsigned VAArgOffset = 0;
    const DataLayout &DL = F.getParent()->getDataLayout();
    for (auto ArgIt = CB.arg_begin() + CB.getFunctionType()->getNumParams(),
              End = CB.arg_end();
         ArgIt != End; ++ArgIt) {
      Triple TargetTriple(F.getParent()->getTargetTriple());
      Value *A = *ArgIt;
      Value *Base;
      uint64_t ArgSize = DL.getTypeAllocSize(A->getType());
      if (TargetTriple.getArch() == Triple::mips64) {
        // Adjusting the shadow for argument with size < 8 to match the placement
        // of bits in big endian system
        if (ArgSize < 8)
          VAArgOffset += (8 - ArgSize);
      }
      Base = getShadowPtrForVAArgument(A->getType(), IRB, VAArgOffset, ArgSize);
      VAArgOffset += ArgSize;
      VAArgOffset = alignTo(VAArgOffset, 8);
      if (!Base)
        continue;
      IRB.CreateAlignedStore(MSV.getShadow(A), Base, kShadowTLSAlignment);
    }

    Constant *TotalVAArgSize = ConstantInt::get(IRB.getInt64Ty(), VAArgOffset);
    // Here using VAArgOverflowSizeTLS as VAArgSizeTLS to avoid creation of
    // a new class member i.e. it is the total size of all VarArgs.
    IRB.CreateStore(TotalVAArgSize, MS.VAArgOverflowSizeTLS);
  }

  /// Compute the shadow address for a given va_arg.
  Value *getShadowPtrForVAArgument(Type *Ty, IRBuilder<> &IRB,
                                   unsigned ArgOffset, unsigned ArgSize) {
    // Make sure we don't overflow __msan_va_arg_tls.
    if (ArgOffset + ArgSize > kParamTLSSize)
      return nullptr;
    Value *Base = IRB.CreatePointerCast(MS.VAArgTLS, MS.IntptrTy);
    Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(MSV.getShadowTy(Ty), 0),
                              "_msarg");
  }

  void visitVAStartInst(VAStartInst &I) override {
    IRBuilder<> IRB(&I);
    VAStartInstrumentationList.push_back(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr, *OriginPtr;
    const Align Alignment = Align(8);
    std::tie(ShadowPtr, OriginPtr) = MSV.getShadowOriginPtr(
        VAListTag, IRB, IRB.getInt8Ty(), Alignment, /*isStore*/ true);
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */ 8, Alignment, false);
  }

  void visitVACopyInst(VACopyInst &I) override {
    IRBuilder<> IRB(&I);
    VAStartInstrumentationList.push_back(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr, *OriginPtr;
    const Align Alignment = Align(8);
    std::tie(ShadowPtr, OriginPtr) = MSV.getShadowOriginPtr(
        VAListTag, IRB, IRB.getInt8Ty(), Alignment, /*isStore*/ true);
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */ 8, Alignment, false);
  }

  void finalizeInstrumentation() override {
    assert(!VAArgSize && !VAArgTLSCopy &&
           "finalizeInstrumentation called twice");
    IRBuilder<> IRB(MSV.FnPrologueEnd);
    VAArgSize = IRB.CreateLoad(IRB.getInt64Ty(), MS.VAArgOverflowSizeTLS);
    Value *CopySize = IRB.CreateAdd(ConstantInt::get(MS.IntptrTy, 0),
                                    VAArgSize);

    if (!VAStartInstrumentationList.empty()) {
      // If there is a va_start in this function, make a backup copy of
      // va_arg_tls somewhere in the function entry block.
      VAArgTLSCopy = IRB.CreateAlloca(Type::getInt8Ty(*MS.C), CopySize);
      IRB.CreateMemCpy(VAArgTLSCopy, Align(8), MS.VAArgTLS, Align(8), CopySize);
    }

    // Instrument va_start.
    // Copy va_list shadow from the backup copy of the TLS contents.
    for (size_t i = 0, n = VAStartInstrumentationList.size(); i < n; i++) {
      CallInst *OrigInst = VAStartInstrumentationList[i];
      IRBuilder<> IRB(OrigInst->getNextNode());
      Value *VAListTag = OrigInst->getArgOperand(0);
      Type *RegSaveAreaPtrTy = Type::getInt64PtrTy(*MS.C);
      Value *RegSaveAreaPtrPtr =
          IRB.CreateIntToPtr(IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
                             PointerType::get(RegSaveAreaPtrTy, 0));
      Value *RegSaveAreaPtr =
          IRB.CreateLoad(RegSaveAreaPtrTy, RegSaveAreaPtrPtr);
      Value *RegSaveAreaShadowPtr, *RegSaveAreaOriginPtr;
      const Align Alignment = Align(8);
      std::tie(RegSaveAreaShadowPtr, RegSaveAreaOriginPtr) =
          MSV.getShadowOriginPtr(RegSaveAreaPtr, IRB, IRB.getInt8Ty(),
                                 Alignment, /*isStore*/ true);
      IRB.CreateMemCpy(RegSaveAreaShadowPtr, Alignment, VAArgTLSCopy, Alignment,
                       CopySize);
    }
  }
};

/// AArch64-specific implementation of VarArgHelper.
struct VarArgAArch64Helper : public VarArgHelper {
  static const unsigned kAArch64GrArgSize = 64;
  static const unsigned kAArch64VrArgSize = 128;

  static const unsigned AArch64GrBegOffset = 0;
  static const unsigned AArch64GrEndOffset = kAArch64GrArgSize;
  // Make VR space aligned to 16 bytes.
  static const unsigned AArch64VrBegOffset = AArch64GrEndOffset;
  static const unsigned AArch64VrEndOffset = AArch64VrBegOffset
                                             + kAArch64VrArgSize;
  static const unsigned AArch64VAEndOffset = AArch64VrEndOffset;

  Function &F;
  MemorySanitizer &MS;
  MemorySanitizerVisitor &MSV;
  Value *VAArgTLSCopy = nullptr;
  Value *VAArgOverflowSize = nullptr;

  SmallVector<CallInst*, 16> VAStartInstrumentationList;

  enum ArgKind { AK_GeneralPurpose, AK_FloatingPoint, AK_Memory };

  VarArgAArch64Helper(Function &F, MemorySanitizer &MS,
                    MemorySanitizerVisitor &MSV) : F(F), MS(MS), MSV(MSV) {}

  ArgKind classifyArgument(Value* arg) {
    Type *T = arg->getType();
    if (T->isFPOrFPVectorTy())
      return AK_FloatingPoint;
    if ((T->isIntegerTy() && T->getPrimitiveSizeInBits() <= 64)
        || (T->isPointerTy()))
      return AK_GeneralPurpose;
    return AK_Memory;
  }

  // The instrumentation stores the argument shadow in a non ABI-specific
  // format because it does not know which argument is named (since Clang,
  // like x86_64 case, lowers the va_args in the frontend and this pass only
  // sees the low level code that deals with va_list internals).
  // The first seven GR registers are saved in the first 56 bytes of the
  // va_arg tls arra, followers by the first 8 FP/SIMD registers, and then
  // the remaining arguments.
  // Using constant offset within the va_arg TLS array allows fast copy
  // in the finalize instrumentation.
  void visitCallBase(CallBase &CB, IRBuilder<> &IRB) override {
    unsigned GrOffset = AArch64GrBegOffset;
    unsigned VrOffset = AArch64VrBegOffset;
    unsigned OverflowOffset = AArch64VAEndOffset;

    const DataLayout &DL = F.getParent()->getDataLayout();
    for (auto ArgIt = CB.arg_begin(), End = CB.arg_end(); ArgIt != End;
         ++ArgIt) {
      Value *A = *ArgIt;
      unsigned ArgNo = CB.getArgOperandNo(ArgIt);
      bool IsFixed = ArgNo < CB.getFunctionType()->getNumParams();
      ArgKind AK = classifyArgument(A);
      if (AK == AK_GeneralPurpose && GrOffset >= AArch64GrEndOffset)
        AK = AK_Memory;
      if (AK == AK_FloatingPoint && VrOffset >= AArch64VrEndOffset)
        AK = AK_Memory;
      Value *Base;
      switch (AK) {
        case AK_GeneralPurpose:
          Base = getShadowPtrForVAArgument(A->getType(), IRB, GrOffset, 8);
          GrOffset += 8;
          break;
        case AK_FloatingPoint:
          Base = getShadowPtrForVAArgument(A->getType(), IRB, VrOffset, 8);
          VrOffset += 16;
          break;
        case AK_Memory:
          // Don't count fixed arguments in the overflow area - va_start will
          // skip right over them.
          if (IsFixed)
            continue;
          uint64_t ArgSize = DL.getTypeAllocSize(A->getType());
          Base = getShadowPtrForVAArgument(A->getType(), IRB, OverflowOffset,
                                           alignTo(ArgSize, 8));
          OverflowOffset += alignTo(ArgSize, 8);
          break;
      }
      // Count Gp/Vr fixed arguments to their respective offsets, but don't
      // bother to actually store a shadow.
      if (IsFixed)
        continue;
      if (!Base)
        continue;
      IRB.CreateAlignedStore(MSV.getShadow(A), Base, kShadowTLSAlignment);
    }
    Constant *OverflowSize =
      ConstantInt::get(IRB.getInt64Ty(), OverflowOffset - AArch64VAEndOffset);
    IRB.CreateStore(OverflowSize, MS.VAArgOverflowSizeTLS);
  }

  /// Compute the shadow address for a given va_arg.
  Value *getShadowPtrForVAArgument(Type *Ty, IRBuilder<> &IRB,
                                   unsigned ArgOffset, unsigned ArgSize) {
    // Make sure we don't overflow __msan_va_arg_tls.
    if (ArgOffset + ArgSize > kParamTLSSize)
      return nullptr;
    Value *Base = IRB.CreatePointerCast(MS.VAArgTLS, MS.IntptrTy);
    Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(MSV.getShadowTy(Ty), 0),
                              "_msarg");
  }

  void visitVAStartInst(VAStartInst &I) override {
    IRBuilder<> IRB(&I);
    VAStartInstrumentationList.push_back(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr, *OriginPtr;
    const Align Alignment = Align(8);
    std::tie(ShadowPtr, OriginPtr) = MSV.getShadowOriginPtr(
        VAListTag, IRB, IRB.getInt8Ty(), Alignment, /*isStore*/ true);
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */ 32, Alignment, false);
  }

  void visitVACopyInst(VACopyInst &I) override {
    IRBuilder<> IRB(&I);
    VAStartInstrumentationList.push_back(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr, *OriginPtr;
    const Align Alignment = Align(8);
    std::tie(ShadowPtr, OriginPtr) = MSV.getShadowOriginPtr(
        VAListTag, IRB, IRB.getInt8Ty(), Alignment, /*isStore*/ true);
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */ 32, Alignment, false);
  }

  // Retrieve a va_list field of 'void*' size.
  Value* getVAField64(IRBuilder<> &IRB, Value *VAListTag, int offset) {
    Value *SaveAreaPtrPtr =
      IRB.CreateIntToPtr(
        IRB.CreateAdd(IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
                      ConstantInt::get(MS.IntptrTy, offset)),
        Type::getInt64PtrTy(*MS.C));
    return IRB.CreateLoad(Type::getInt64Ty(*MS.C), SaveAreaPtrPtr);
  }

  // Retrieve a va_list field of 'int' size.
  Value* getVAField32(IRBuilder<> &IRB, Value *VAListTag, int offset) {
    Value *SaveAreaPtr =
      IRB.CreateIntToPtr(
        IRB.CreateAdd(IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
                      ConstantInt::get(MS.IntptrTy, offset)),
        Type::getInt32PtrTy(*MS.C));
    Value *SaveArea32 = IRB.CreateLoad(IRB.getInt32Ty(), SaveAreaPtr);
    return IRB.CreateSExt(SaveArea32, MS.IntptrTy);
  }

  void finalizeInstrumentation() override {
    assert(!VAArgOverflowSize && !VAArgTLSCopy &&
           "finalizeInstrumentation called twice");
    if (!VAStartInstrumentationList.empty()) {
      // If there is a va_start in this function, make a backup copy of
      // va_arg_tls somewhere in the function entry block.
      IRBuilder<> IRB(MSV.FnPrologueEnd);
      VAArgOverflowSize =
          IRB.CreateLoad(IRB.getInt64Ty(), MS.VAArgOverflowSizeTLS);
      Value *CopySize =
        IRB.CreateAdd(ConstantInt::get(MS.IntptrTy, AArch64VAEndOffset),
                      VAArgOverflowSize);
      VAArgTLSCopy = IRB.CreateAlloca(Type::getInt8Ty(*MS.C), CopySize);
      IRB.CreateMemCpy(VAArgTLSCopy, Align(8), MS.VAArgTLS, Align(8), CopySize);
    }

    Value *GrArgSize = ConstantInt::get(MS.IntptrTy, kAArch64GrArgSize);
    Value *VrArgSize = ConstantInt::get(MS.IntptrTy, kAArch64VrArgSize);

    // Instrument va_start, copy va_list shadow from the backup copy of
    // the TLS contents.
    for (size_t i = 0, n = VAStartInstrumentationList.size(); i < n; i++) {
      CallInst *OrigInst = VAStartInstrumentationList[i];
      IRBuilder<> IRB(OrigInst->getNextNode());

      Value *VAListTag = OrigInst->getArgOperand(0);

      // The variadic ABI for AArch64 creates two areas to save the incoming
      // argument registers (one for 64-bit general register xn-x7 and another
      // for 128-bit FP/SIMD vn-v7).
      // We need then to propagate the shadow arguments on both regions
      // 'va::__gr_top + va::__gr_offs' and 'va::__vr_top + va::__vr_offs'.
      // The remaining arguments are saved on shadow for 'va::stack'.
      // One caveat is it requires only to propagate the non-named arguments,
      // however on the call site instrumentation 'all' the arguments are
      // saved. So to copy the shadow values from the va_arg TLS array
      // we need to adjust the offset for both GR and VR fields based on
      // the __{gr,vr}_offs value (since they are stores based on incoming
      // named arguments).

      // Read the stack pointer from the va_list.
      Value *StackSaveAreaPtr = getVAField64(IRB, VAListTag, 0);

      // Read both the __gr_top and __gr_off and add them up.
      Value *GrTopSaveAreaPtr = getVAField64(IRB, VAListTag, 8);
      Value *GrOffSaveArea = getVAField32(IRB, VAListTag, 24);

      Value *GrRegSaveAreaPtr = IRB.CreateAdd(GrTopSaveAreaPtr, GrOffSaveArea);

      // Read both the __vr_top and __vr_off and add them up.
      Value *VrTopSaveAreaPtr = getVAField64(IRB, VAListTag, 16);
      Value *VrOffSaveArea = getVAField32(IRB, VAListTag, 28);

      Value *VrRegSaveAreaPtr = IRB.CreateAdd(VrTopSaveAreaPtr, VrOffSaveArea);

      // It does not know how many named arguments is being used and, on the
      // callsite all the arguments were saved.  Since __gr_off is defined as
      // '0 - ((8 - named_gr) * 8)', the idea is to just propagate the variadic
      // argument by ignoring the bytes of shadow from named arguments.
      Value *GrRegSaveAreaShadowPtrOff =
        IRB.CreateAdd(GrArgSize, GrOffSaveArea);

      Value *GrRegSaveAreaShadowPtr =
          MSV.getShadowOriginPtr(GrRegSaveAreaPtr, IRB, IRB.getInt8Ty(),
                                 Align(8), /*isStore*/ true)
              .first;

      Value *GrSrcPtr = IRB.CreateInBoundsGEP(IRB.getInt8Ty(), VAArgTLSCopy,
                                              GrRegSaveAreaShadowPtrOff);
      Value *GrCopySize = IRB.CreateSub(GrArgSize, GrRegSaveAreaShadowPtrOff);

      IRB.CreateMemCpy(GrRegSaveAreaShadowPtr, Align(8), GrSrcPtr, Align(8),
                       GrCopySize);

      // Again, but for FP/SIMD values.
      Value *VrRegSaveAreaShadowPtrOff =
          IRB.CreateAdd(VrArgSize, VrOffSaveArea);

      Value *VrRegSaveAreaShadowPtr =
          MSV.getShadowOriginPtr(VrRegSaveAreaPtr, IRB, IRB.getInt8Ty(),
                                 Align(8), /*isStore*/ true)
              .first;

      Value *VrSrcPtr = IRB.CreateInBoundsGEP(
        IRB.getInt8Ty(),
        IRB.CreateInBoundsGEP(IRB.getInt8Ty(), VAArgTLSCopy,
                              IRB.getInt32(AArch64VrBegOffset)),
        VrRegSaveAreaShadowPtrOff);
      Value *VrCopySize = IRB.CreateSub(VrArgSize, VrRegSaveAreaShadowPtrOff);

      IRB.CreateMemCpy(VrRegSaveAreaShadowPtr, Align(8), VrSrcPtr, Align(8),
                       VrCopySize);

      // And finally for remaining arguments.
      Value *StackSaveAreaShadowPtr =
          MSV.getShadowOriginPtr(StackSaveAreaPtr, IRB, IRB.getInt8Ty(),
                                 Align(16), /*isStore*/ true)
              .first;

      Value *StackSrcPtr =
        IRB.CreateInBoundsGEP(IRB.getInt8Ty(), VAArgTLSCopy,
                              IRB.getInt32(AArch64VAEndOffset));

      IRB.CreateMemCpy(StackSaveAreaShadowPtr, Align(16), StackSrcPtr,
                       Align(16), VAArgOverflowSize);
    }
  }
};

/// PowerPC64-specific implementation of VarArgHelper.
struct VarArgPowerPC64Helper : public VarArgHelper {
  Function &F;
  MemorySanitizer &MS;
  MemorySanitizerVisitor &MSV;
  Value *VAArgTLSCopy = nullptr;
  Value *VAArgSize = nullptr;

  SmallVector<CallInst*, 16> VAStartInstrumentationList;

  VarArgPowerPC64Helper(Function &F, MemorySanitizer &MS,
                    MemorySanitizerVisitor &MSV) : F(F), MS(MS), MSV(MSV) {}

  void visitCallBase(CallBase &CB, IRBuilder<> &IRB) override {
    // For PowerPC, we need to deal with alignment of stack arguments -
    // they are mostly aligned to 8 bytes, but vectors and i128 arrays
    // are aligned to 16 bytes, byvals can be aligned to 8 or 16 bytes,
    // For that reason, we compute current offset from stack pointer (which is
    // always properly aligned), and offset for the first vararg, then subtract
    // them.
    unsigned VAArgBase;
    Triple TargetTriple(F.getParent()->getTargetTriple());
    // Parameter save area starts at 48 bytes from frame pointer for ABIv1,
    // and 32 bytes for ABIv2.  This is usually determined by target
    // endianness, but in theory could be overridden by function attribute.
    if (TargetTriple.getArch() == Triple::ppc64)
      VAArgBase = 48;
    else
      VAArgBase = 32;
    unsigned VAArgOffset = VAArgBase;
    const DataLayout &DL = F.getParent()->getDataLayout();
    for (auto ArgIt = CB.arg_begin(), End = CB.arg_end(); ArgIt != End;
         ++ArgIt) {
      Value *A = *ArgIt;
      unsigned ArgNo = CB.getArgOperandNo(ArgIt);
      bool IsFixed = ArgNo < CB.getFunctionType()->getNumParams();
      bool IsByVal = CB.paramHasAttr(ArgNo, Attribute::ByVal);
      if (IsByVal) {
        assert(A->getType()->isPointerTy());
        Type *RealTy = CB.getParamByValType(ArgNo);
        uint64_t ArgSize = DL.getTypeAllocSize(RealTy);
        MaybeAlign ArgAlign = CB.getParamAlign(ArgNo);
        if (!ArgAlign || *ArgAlign < Align(8))
          ArgAlign = Align(8);
        VAArgOffset = alignTo(VAArgOffset, ArgAlign);
        if (!IsFixed) {
          Value *Base = getShadowPtrForVAArgument(
              RealTy, IRB, VAArgOffset - VAArgBase, ArgSize);
          if (Base) {
            Value *AShadowPtr, *AOriginPtr;
            std::tie(AShadowPtr, AOriginPtr) =
                MSV.getShadowOriginPtr(A, IRB, IRB.getInt8Ty(),
                                       kShadowTLSAlignment, /*isStore*/ false);

            IRB.CreateMemCpy(Base, kShadowTLSAlignment, AShadowPtr,
                             kShadowTLSAlignment, ArgSize);
          }
        }
        VAArgOffset += alignTo(ArgSize, 8);
      } else {
        Value *Base;
        uint64_t ArgSize = DL.getTypeAllocSize(A->getType());
        uint64_t ArgAlign = 8;
        if (A->getType()->isArrayTy()) {
          // Arrays are aligned to element size, except for long double
          // arrays, which are aligned to 8 bytes.
          Type *ElementTy = A->getType()->getArrayElementType();
          if (!ElementTy->isPPC_FP128Ty())
            ArgAlign = DL.getTypeAllocSize(ElementTy);
        } else if (A->getType()->isVectorTy()) {
          // Vectors are naturally aligned.
          ArgAlign = DL.getTypeAllocSize(A->getType());
        }
        if (ArgAlign < 8)
          ArgAlign = 8;
        VAArgOffset = alignTo(VAArgOffset, ArgAlign);
        if (DL.isBigEndian()) {
          // Adjusting the shadow for argument with size < 8 to match the placement
          // of bits in big endian system
          if (ArgSize < 8)
            VAArgOffset += (8 - ArgSize);
        }
        if (!IsFixed) {
          Base = getShadowPtrForVAArgument(A->getType(), IRB,
                                           VAArgOffset - VAArgBase, ArgSize);
          if (Base)
            IRB.CreateAlignedStore(MSV.getShadow(A), Base, kShadowTLSAlignment);
        }
        VAArgOffset += ArgSize;
        VAArgOffset = alignTo(VAArgOffset, 8);
      }
      if (IsFixed)
        VAArgBase = VAArgOffset;
    }

    Constant *TotalVAArgSize = ConstantInt::get(IRB.getInt64Ty(),
                                                VAArgOffset - VAArgBase);
    // Here using VAArgOverflowSizeTLS as VAArgSizeTLS to avoid creation of
    // a new class member i.e. it is the total size of all VarArgs.
    IRB.CreateStore(TotalVAArgSize, MS.VAArgOverflowSizeTLS);
  }

  /// Compute the shadow address for a given va_arg.
  Value *getShadowPtrForVAArgument(Type *Ty, IRBuilder<> &IRB,
                                   unsigned ArgOffset, unsigned ArgSize) {
    // Make sure we don't overflow __msan_va_arg_tls.
    if (ArgOffset + ArgSize > kParamTLSSize)
      return nullptr;
    Value *Base = IRB.CreatePointerCast(MS.VAArgTLS, MS.IntptrTy);
    Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(MSV.getShadowTy(Ty), 0),
                              "_msarg");
  }

  void visitVAStartInst(VAStartInst &I) override {
    IRBuilder<> IRB(&I);
    VAStartInstrumentationList.push_back(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr, *OriginPtr;
    const Align Alignment = Align(8);
    std::tie(ShadowPtr, OriginPtr) = MSV.getShadowOriginPtr(
        VAListTag, IRB, IRB.getInt8Ty(), Alignment, /*isStore*/ true);
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */ 8, Alignment, false);
  }

  void visitVACopyInst(VACopyInst &I) override {
    IRBuilder<> IRB(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr, *OriginPtr;
    const Align Alignment = Align(8);
    std::tie(ShadowPtr, OriginPtr) = MSV.getShadowOriginPtr(
        VAListTag, IRB, IRB.getInt8Ty(), Alignment, /*isStore*/ true);
    // Unpoison the whole __va_list_tag.
    // FIXME: magic ABI constants.
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */ 8, Alignment, false);
  }

  void finalizeInstrumentation() override {
    assert(!VAArgSize && !VAArgTLSCopy &&
           "finalizeInstrumentation called twice");
    IRBuilder<> IRB(MSV.FnPrologueEnd);
    VAArgSize = IRB.CreateLoad(IRB.getInt64Ty(), MS.VAArgOverflowSizeTLS);
    Value *CopySize = IRB.CreateAdd(ConstantInt::get(MS.IntptrTy, 0),
                                    VAArgSize);

    if (!VAStartInstrumentationList.empty()) {
      // If there is a va_start in this function, make a backup copy of
      // va_arg_tls somewhere in the function entry block.
      VAArgTLSCopy = IRB.CreateAlloca(Type::getInt8Ty(*MS.C), CopySize);
      IRB.CreateMemCpy(VAArgTLSCopy, Align(8), MS.VAArgTLS, Align(8), CopySize);
    }

    // Instrument va_start.
    // Copy va_list shadow from the backup copy of the TLS contents.
    for (size_t i = 0, n = VAStartInstrumentationList.size(); i < n; i++) {
      CallInst *OrigInst = VAStartInstrumentationList[i];
      IRBuilder<> IRB(OrigInst->getNextNode());
      Value *VAListTag = OrigInst->getArgOperand(0);
      Type *RegSaveAreaPtrTy = Type::getInt64PtrTy(*MS.C);
      Value *RegSaveAreaPtrPtr =
          IRB.CreateIntToPtr(IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
                             PointerType::get(RegSaveAreaPtrTy, 0));
      Value *RegSaveAreaPtr =
          IRB.CreateLoad(RegSaveAreaPtrTy, RegSaveAreaPtrPtr);
      Value *RegSaveAreaShadowPtr, *RegSaveAreaOriginPtr;
      const Align Alignment = Align(8);
      std::tie(RegSaveAreaShadowPtr, RegSaveAreaOriginPtr) =
          MSV.getShadowOriginPtr(RegSaveAreaPtr, IRB, IRB.getInt8Ty(),
                                 Alignment, /*isStore*/ true);
      IRB.CreateMemCpy(RegSaveAreaShadowPtr, Alignment, VAArgTLSCopy, Alignment,
                       CopySize);
    }
  }
};

/// SystemZ-specific implementation of VarArgHelper.
struct VarArgSystemZHelper : public VarArgHelper {
  static const unsigned SystemZGpOffset = 16;
  static const unsigned SystemZGpEndOffset = 56;
  static const unsigned SystemZFpOffset = 128;
  static const unsigned SystemZFpEndOffset = 160;
  static const unsigned SystemZMaxVrArgs = 8;
  static const unsigned SystemZRegSaveAreaSize = 160;
  static const unsigned SystemZOverflowOffset = 160;
  static const unsigned SystemZVAListTagSize = 32;
  static const unsigned SystemZOverflowArgAreaPtrOffset = 16;
  static const unsigned SystemZRegSaveAreaPtrOffset = 24;

  Function &F;
  MemorySanitizer &MS;
  MemorySanitizerVisitor &MSV;
  Value *VAArgTLSCopy = nullptr;
  Value *VAArgTLSOriginCopy = nullptr;
  Value *VAArgOverflowSize = nullptr;

  SmallVector<CallInst *, 16> VAStartInstrumentationList;

  enum class ArgKind {
    GeneralPurpose,
    FloatingPoint,
    Vector,
    Memory,
    Indirect,
  };

  enum class ShadowExtension { None, Zero, Sign };

  VarArgSystemZHelper(Function &F, MemorySanitizer &MS,
                      MemorySanitizerVisitor &MSV)
      : F(F), MS(MS), MSV(MSV) {}

  ArgKind classifyArgument(Type *T, bool IsSoftFloatABI) {
    // T is a SystemZABIInfo::classifyArgumentType() output, and there are
    // only a few possibilities of what it can be. In particular, enums, single
    // element structs and large types have already been taken care of.

    // Some i128 and fp128 arguments are converted to pointers only in the
    // back end.
    if (T->isIntegerTy(128) || T->isFP128Ty())
      return ArgKind::Indirect;
    if (T->isFloatingPointTy())
      return IsSoftFloatABI ? ArgKind::GeneralPurpose : ArgKind::FloatingPoint;
    if (T->isIntegerTy() || T->isPointerTy())
      return ArgKind::GeneralPurpose;
    if (T->isVectorTy())
      return ArgKind::Vector;
    return ArgKind::Memory;
  }

  ShadowExtension getShadowExtension(const CallBase &CB, unsigned ArgNo) {
    // ABI says: "One of the simple integer types no more than 64 bits wide.
    // ... If such an argument is shorter than 64 bits, replace it by a full
    // 64-bit integer representing the same number, using sign or zero
    // extension". Shadow for an integer argument has the same type as the
    // argument itself, so it can be sign or zero extended as well.
    bool ZExt = CB.paramHasAttr(ArgNo, Attribute::ZExt);
    bool SExt = CB.paramHasAttr(ArgNo, Attribute::SExt);
    if (ZExt) {
      assert(!SExt);
      return ShadowExtension::Zero;
    }
    if (SExt) {
      assert(!ZExt);
      return ShadowExtension::Sign;
    }
    return ShadowExtension::None;
  }

  void visitCallBase(CallBase &CB, IRBuilder<> &IRB) override {
    bool IsSoftFloatABI = CB.getCalledFunction()
                              ->getFnAttribute("use-soft-float")
                              .getValueAsBool();
    unsigned GpOffset = SystemZGpOffset;
    unsigned FpOffset = SystemZFpOffset;
    unsigned VrIndex = 0;
    unsigned OverflowOffset = SystemZOverflowOffset;
    const DataLayout &DL = F.getParent()->getDataLayout();
    for (auto ArgIt = CB.arg_begin(), End = CB.arg_end(); ArgIt != End;
         ++ArgIt) {
      Value *A = *ArgIt;
      unsigned ArgNo = CB.getArgOperandNo(ArgIt);
      bool IsFixed = ArgNo < CB.getFunctionType()->getNumParams();
      // SystemZABIInfo does not produce ByVal parameters.
      assert(!CB.paramHasAttr(ArgNo, Attribute::ByVal));
      Type *T = A->getType();
      ArgKind AK = classifyArgument(T, IsSoftFloatABI);
      if (AK == ArgKind::Indirect) {
        T = PointerType::get(T, 0);
        AK = ArgKind::GeneralPurpose;
      }
      if (AK == ArgKind::GeneralPurpose && GpOffset >= SystemZGpEndOffset)
        AK = ArgKind::Memory;
      if (AK == ArgKind::FloatingPoint && FpOffset >= SystemZFpEndOffset)
        AK = ArgKind::Memory;
      if (AK == ArgKind::Vector && (VrIndex >= SystemZMaxVrArgs || !IsFixed))
        AK = ArgKind::Memory;
      Value *ShadowBase = nullptr;
      Value *OriginBase = nullptr;
      ShadowExtension SE = ShadowExtension::None;
      switch (AK) {
      case ArgKind::GeneralPurpose: {
        // Always keep track of GpOffset, but store shadow only for varargs.
        uint64_t ArgSize = 8;
        if (GpOffset + ArgSize <= kParamTLSSize) {
          if (!IsFixed) {
            SE = getShadowExtension(CB, ArgNo);
            uint64_t GapSize = 0;
            if (SE == ShadowExtension::None) {
              uint64_t ArgAllocSize = DL.getTypeAllocSize(T);
              assert(ArgAllocSize <= ArgSize);
              GapSize = ArgSize - ArgAllocSize;
            }
            ShadowBase = getShadowAddrForVAArgument(IRB, GpOffset + GapSize);
            if (MS.TrackOrigins)
              OriginBase = getOriginPtrForVAArgument(IRB, GpOffset + GapSize);
          }
          GpOffset += ArgSize;
        } else {
          GpOffset = kParamTLSSize;
        }
        break;
      }
      case ArgKind::FloatingPoint: {
        // Always keep track of FpOffset, but store shadow only for varargs.
        uint64_t ArgSize = 8;
        if (FpOffset + ArgSize <= kParamTLSSize) {
          if (!IsFixed) {
            // PoP says: "A short floating-point datum requires only the
            // left-most 32 bit positions of a floating-point register".
            // Therefore, in contrast to AK_GeneralPurpose and AK_Memory,
            // don't extend shadow and don't mind the gap.
            ShadowBase = getShadowAddrForVAArgument(IRB, FpOffset);
            if (MS.TrackOrigins)
              OriginBase = getOriginPtrForVAArgument(IRB, FpOffset);
          }
          FpOffset += ArgSize;
        } else {
          FpOffset = kParamTLSSize;
        }
        break;
      }
      case ArgKind::Vector: {
        // Keep track of VrIndex. No need to store shadow, since vector varargs
        // go through AK_Memory.
        assert(IsFixed);
        VrIndex++;
        break;
      }
      case ArgKind::Memory: {
        // Keep track of OverflowOffset and store shadow only for varargs.
        // Ignore fixed args, since we need to copy only the vararg portion of
        // the overflow area shadow.
        if (!IsFixed) {
          uint64_t ArgAllocSize = DL.getTypeAllocSize(T);
          uint64_t ArgSize = alignTo(ArgAllocSize, 8);
          if (OverflowOffset + ArgSize <= kParamTLSSize) {
            SE = getShadowExtension(CB, ArgNo);
            uint64_t GapSize =
                SE == ShadowExtension::None ? ArgSize - ArgAllocSize : 0;
            ShadowBase =
                getShadowAddrForVAArgument(IRB, OverflowOffset + GapSize);
            if (MS.TrackOrigins)
              OriginBase =
                  getOriginPtrForVAArgument(IRB, OverflowOffset + GapSize);
            OverflowOffset += ArgSize;
          } else {
            OverflowOffset = kParamTLSSize;
          }
        }
        break;
      }
      case ArgKind::Indirect:
        llvm_unreachable("Indirect must be converted to GeneralPurpose");
      }
      if (ShadowBase == nullptr)
        continue;
      Value *Shadow = MSV.getShadow(A);
      if (SE != ShadowExtension::None)
        Shadow = MSV.CreateShadowCast(IRB, Shadow, IRB.getInt64Ty(),
                                      /*Signed*/ SE == ShadowExtension::Sign);
      ShadowBase = IRB.CreateIntToPtr(
          ShadowBase, PointerType::get(Shadow->getType(), 0), "_msarg_va_s");
      IRB.CreateStore(Shadow, ShadowBase);
      if (MS.TrackOrigins) {
        Value *Origin = MSV.getOrigin(A);
        unsigned StoreSize = DL.getTypeStoreSize(Shadow->getType());
        MSV.paintOrigin(IRB, Origin, OriginBase, StoreSize,
                        kMinOriginAlignment);
      }
    }
    Constant *OverflowSize = ConstantInt::get(
        IRB.getInt64Ty(), OverflowOffset - SystemZOverflowOffset);
    IRB.CreateStore(OverflowSize, MS.VAArgOverflowSizeTLS);
  }

  Value *getShadowAddrForVAArgument(IRBuilder<> &IRB, unsigned ArgOffset) {
    Value *Base = IRB.CreatePointerCast(MS.VAArgTLS, MS.IntptrTy);
    return IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
  }

  Value *getOriginPtrForVAArgument(IRBuilder<> &IRB, int ArgOffset) {
    Value *Base = IRB.CreatePointerCast(MS.VAArgOriginTLS, MS.IntptrTy);
    Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(MS.OriginTy, 0),
                              "_msarg_va_o");
  }

  void unpoisonVAListTagForInst(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr, *OriginPtr;
    const Align Alignment = Align(8);
    std::tie(ShadowPtr, OriginPtr) =
        MSV.getShadowOriginPtr(VAListTag, IRB, IRB.getInt8Ty(), Alignment,
                               /*isStore*/ true);
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     SystemZVAListTagSize, Alignment, false);
  }

  void visitVAStartInst(VAStartInst &I) override {
    VAStartInstrumentationList.push_back(&I);
    unpoisonVAListTagForInst(I);
  }

  void visitVACopyInst(VACopyInst &I) override { unpoisonVAListTagForInst(I); }

  void copyRegSaveArea(IRBuilder<> &IRB, Value *VAListTag) {
    Type *RegSaveAreaPtrTy = Type::getInt64PtrTy(*MS.C);
    Value *RegSaveAreaPtrPtr = IRB.CreateIntToPtr(
        IRB.CreateAdd(
            IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
            ConstantInt::get(MS.IntptrTy, SystemZRegSaveAreaPtrOffset)),
        PointerType::get(RegSaveAreaPtrTy, 0));
    Value *RegSaveAreaPtr = IRB.CreateLoad(RegSaveAreaPtrTy, RegSaveAreaPtrPtr);
    Value *RegSaveAreaShadowPtr, *RegSaveAreaOriginPtr;
    const Align Alignment = Align(8);
    std::tie(RegSaveAreaShadowPtr, RegSaveAreaOriginPtr) =
        MSV.getShadowOriginPtr(RegSaveAreaPtr, IRB, IRB.getInt8Ty(), Alignment,
                               /*isStore*/ true);
    // TODO(iii): copy only fragments filled by visitCallBase()
    IRB.CreateMemCpy(RegSaveAreaShadowPtr, Alignment, VAArgTLSCopy, Alignment,
                     SystemZRegSaveAreaSize);
    if (MS.TrackOrigins)
      IRB.CreateMemCpy(RegSaveAreaOriginPtr, Alignment, VAArgTLSOriginCopy,
                       Alignment, SystemZRegSaveAreaSize);
  }

  void copyOverflowArea(IRBuilder<> &IRB, Value *VAListTag) {
    Type *OverflowArgAreaPtrTy = Type::getInt64PtrTy(*MS.C);
    Value *OverflowArgAreaPtrPtr = IRB.CreateIntToPtr(
        IRB.CreateAdd(
            IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
            ConstantInt::get(MS.IntptrTy, SystemZOverflowArgAreaPtrOffset)),
        PointerType::get(OverflowArgAreaPtrTy, 0));
    Value *OverflowArgAreaPtr =
        IRB.CreateLoad(OverflowArgAreaPtrTy, OverflowArgAreaPtrPtr);
    Value *OverflowArgAreaShadowPtr, *OverflowArgAreaOriginPtr;
    const Align Alignment = Align(8);
    std::tie(OverflowArgAreaShadowPtr, OverflowArgAreaOriginPtr) =
        MSV.getShadowOriginPtr(OverflowArgAreaPtr, IRB, IRB.getInt8Ty(),
                               Alignment, /*isStore*/ true);
    Value *SrcPtr = IRB.CreateConstGEP1_32(IRB.getInt8Ty(), VAArgTLSCopy,
                                           SystemZOverflowOffset);
    IRB.CreateMemCpy(OverflowArgAreaShadowPtr, Alignment, SrcPtr, Alignment,
                     VAArgOverflowSize);
    if (MS.TrackOrigins) {
      SrcPtr = IRB.CreateConstGEP1_32(IRB.getInt8Ty(), VAArgTLSOriginCopy,
                                      SystemZOverflowOffset);
      IRB.CreateMemCpy(OverflowArgAreaOriginPtr, Alignment, SrcPtr, Alignment,
                       VAArgOverflowSize);
    }
  }

  void finalizeInstrumentation() override {
    assert(!VAArgOverflowSize && !VAArgTLSCopy &&
           "finalizeInstrumentation called twice");
    if (!VAStartInstrumentationList.empty()) {
      // If there is a va_start in this function, make a backup copy of
      // va_arg_tls somewhere in the function entry block.
      IRBuilder<> IRB(MSV.FnPrologueEnd);
      VAArgOverflowSize =
          IRB.CreateLoad(IRB.getInt64Ty(), MS.VAArgOverflowSizeTLS);
      Value *CopySize =
          IRB.CreateAdd(ConstantInt::get(MS.IntptrTy, SystemZOverflowOffset),
                        VAArgOverflowSize);
      VAArgTLSCopy = IRB.CreateAlloca(Type::getInt8Ty(*MS.C), CopySize);
      IRB.CreateMemCpy(VAArgTLSCopy, Align(8), MS.VAArgTLS, Align(8), CopySize);
      if (MS.TrackOrigins) {
        VAArgTLSOriginCopy = IRB.CreateAlloca(Type::getInt8Ty(*MS.C), CopySize);
        IRB.CreateMemCpy(VAArgTLSOriginCopy, Align(8), MS.VAArgOriginTLS,
                         Align(8), CopySize);
      }
    }

    // Instrument va_start.
    // Copy va_list shadow from the backup copy of the TLS contents.
    for (size_t VaStartNo = 0, VaStartNum = VAStartInstrumentationList.size();
         VaStartNo < VaStartNum; VaStartNo++) {
      CallInst *OrigInst = VAStartInstrumentationList[VaStartNo];
      IRBuilder<> IRB(OrigInst->getNextNode());
      Value *VAListTag = OrigInst->getArgOperand(0);
      copyRegSaveArea(IRB, VAListTag);
      copyOverflowArea(IRB, VAListTag);
    }
  }
};

/// A no-op implementation of VarArgHelper.
struct VarArgNoOpHelper : public VarArgHelper {
  VarArgNoOpHelper(Function &F, MemorySanitizer &MS,
                   MemorySanitizerVisitor &MSV) {}

  void visitCallBase(CallBase &CB, IRBuilder<> &IRB) override {}

  void visitVAStartInst(VAStartInst &I) override {}

  void visitVACopyInst(VACopyInst &I) override {}

  void finalizeInstrumentation() override {}
};

} // end anonymous namespace

static VarArgHelper *CreateVarArgHelper(Function &Func, MemorySanitizer &Msan,
                                        MemorySanitizerVisitor &Visitor) {
  // VarArg handling is only implemented on AMD64. False positives are possible
  // on other platforms.
  Triple TargetTriple(Func.getParent()->getTargetTriple());
  if (TargetTriple.getArch() == Triple::x86_64)
    return new VarArgAMD64Helper(Func, Msan, Visitor);
  else if (TargetTriple.isMIPS64())
    return new VarArgMIPS64Helper(Func, Msan, Visitor);
  else if (TargetTriple.getArch() == Triple::aarch64)
    return new VarArgAArch64Helper(Func, Msan, Visitor);
  else if (TargetTriple.getArch() == Triple::ppc64 ||
           TargetTriple.getArch() == Triple::ppc64le)
    return new VarArgPowerPC64Helper(Func, Msan, Visitor);
  else if (TargetTriple.getArch() == Triple::systemz)
    return new VarArgSystemZHelper(Func, Msan, Visitor);
  else
    return new VarArgNoOpHelper(Func, Msan, Visitor);
}

bool MemorySanitizer::sanitizeFunction(Function &F, TargetLibraryInfo &TLI) {
  if (!CompileKernel && F.getName() == kMsanModuleCtorName)
    return false;

  if (F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
    return false;

  MemorySanitizerVisitor Visitor(F, *this, TLI);

  // Clear out readonly/readnone attributes.
  AttributeMask B;
  B.addAttribute(Attribute::ReadOnly)
      .addAttribute(Attribute::ReadNone)
      .addAttribute(Attribute::WriteOnly)
      .addAttribute(Attribute::ArgMemOnly)
      .addAttribute(Attribute::Speculatable);
  F.removeFnAttrs(B);

  return Visitor.runOnFunction();
}
