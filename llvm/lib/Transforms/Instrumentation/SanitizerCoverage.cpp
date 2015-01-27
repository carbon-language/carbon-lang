//===-- SanitizerCoverage.cpp - coverage instrumentation for sanitizers ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Coverage instrumentation that works with AddressSanitizer
// and potentially with other Sanitizers.
//
// We create a Guard variable with the same linkage
// as the function and inject this code into the entry block (CoverageLevel=1)
// or all blocks (CoverageLevel>=2):
// if (Guard < 0) {
//    __sanitizer_cov(&Guard);
// }
// The accesses to Guard are atomic. The rest of the logic is
// in __sanitizer_cov (it's fine to call it more than once).
//
// With CoverageLevel>=3 we also split critical edges this effectively
// instrumenting all edges.
//
// CoverageLevel>=4 add indirect call profiling implented as a function call.
//
// This coverage implementation provides very limited data:
// it only tells if a given function (block) was ever executed. No counters.
// But for many use cases this is what we need and the added slowdown small.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "sancov"

static const char *const kSanCovModuleInitName = "__sanitizer_cov_module_init";
static const char *const kSanCovName = "__sanitizer_cov";
static const char *const kSanCovIndirCallName = "__sanitizer_cov_indir_call16";
static const char *const kSanCovTraceEnter = "__sanitizer_cov_trace_func_enter";
static const char *const kSanCovTraceBB = "__sanitizer_cov_trace_basic_block";
static const char *const kSanCovModuleCtorName = "sancov.module_ctor";
static const uint64_t    kSanCtorAndDtorPriority = 2;

static cl::opt<int> ClCoverageLevel("sanitizer-coverage-level",
       cl::desc("Sanitizer Coverage. 0: none, 1: entry block, 2: all blocks, "
                "3: all blocks and critical edges, "
                "4: above plus indirect calls"),
       cl::Hidden, cl::init(0));

static cl::opt<int> ClCoverageBlockThreshold(
    "sanitizer-coverage-block-threshold",
    cl::desc("Add coverage instrumentation only to the entry block if there "
             "are more than this number of blocks."),
    cl::Hidden, cl::init(1500));

static cl::opt<bool>
    ClExperimentalTracing("sanitizer-coverage-experimental-tracing",
                          cl::desc("Experimental basic-block tracing: insert "
                                   "callbacks at every basic block"),
                          cl::Hidden, cl::init(false));

namespace {

class SanitizerCoverageModule : public ModulePass {
 public:
   SanitizerCoverageModule(int CoverageLevel = 0)
       : ModulePass(ID),
         CoverageLevel(std::max(CoverageLevel, (int)ClCoverageLevel)) {}
  bool runOnModule(Module &M) override;
  bool runOnFunction(Function &F);
  static char ID;  // Pass identification, replacement for typeid
  const char *getPassName() const override {
    return "SanitizerCoverageModule";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DataLayoutPass>();
  }

 private:
  void InjectCoverageForIndirectCalls(Function &F,
                                      ArrayRef<Instruction *> IndirCalls);
  bool InjectCoverage(Function &F, ArrayRef<BasicBlock *> AllBlocks,
                      ArrayRef<Instruction *> IndirCalls);
  void InjectCoverageAtBlock(Function &F, BasicBlock &BB);
  Function *SanCovFunction;
  Function *SanCovIndirCallFunction;
  Function *SanCovModuleInit;
  Function *SanCovTraceEnter, *SanCovTraceBB;
  InlineAsm *EmptyAsm;
  Type *IntptrTy;
  LLVMContext *C;

  GlobalVariable *GuardArray;

  int CoverageLevel;
};

}  // namespace

static Function *checkInterfaceFunction(Constant *FuncOrBitcast) {
  if (Function *F = dyn_cast<Function>(FuncOrBitcast))
     return F;
  std::string Err;
  raw_string_ostream Stream(Err);
  Stream << "SanitizerCoverage interface function redefined: "
         << *FuncOrBitcast;
  report_fatal_error(Err);
}

bool SanitizerCoverageModule::runOnModule(Module &M) {
  if (!CoverageLevel) return false;
  C = &(M.getContext());
  DataLayoutPass *DLP = &getAnalysis<DataLayoutPass>();
  IntptrTy = Type::getIntNTy(*C, DLP->getDataLayout().getPointerSizeInBits());
  Type *VoidTy = Type::getVoidTy(*C);
  IRBuilder<> IRB(*C);
  Type *Int32PtrTy = PointerType::getUnqual(IRB.getInt32Ty());

  Function *CtorFunc =
      Function::Create(FunctionType::get(VoidTy, false),
                       GlobalValue::InternalLinkage, kSanCovModuleCtorName, &M);
  ReturnInst::Create(*C, BasicBlock::Create(*C, "", CtorFunc));
  appendToGlobalCtors(M, CtorFunc, kSanCtorAndDtorPriority);

  SanCovFunction = checkInterfaceFunction(
      M.getOrInsertFunction(kSanCovName, VoidTy, Int32PtrTy, nullptr));
  SanCovIndirCallFunction = checkInterfaceFunction(M.getOrInsertFunction(
      kSanCovIndirCallName, VoidTy, IntptrTy, IntptrTy, nullptr));
  SanCovModuleInit = checkInterfaceFunction(
      M.getOrInsertFunction(kSanCovModuleInitName, Type::getVoidTy(*C),
                            Int32PtrTy, IntptrTy, nullptr));
  SanCovModuleInit->setLinkage(Function::ExternalLinkage);
  // We insert an empty inline asm after cov callbacks to avoid callback merge.
  EmptyAsm = InlineAsm::get(FunctionType::get(IRB.getVoidTy(), false),
                            StringRef(""), StringRef(""),
                            /*hasSideEffects=*/true);

  if (ClExperimentalTracing) {
    SanCovTraceEnter = checkInterfaceFunction(
        M.getOrInsertFunction(kSanCovTraceEnter, VoidTy, Int32PtrTy, nullptr));
    SanCovTraceBB = checkInterfaceFunction(
        M.getOrInsertFunction(kSanCovTraceBB, VoidTy, Int32PtrTy, nullptr));
  }

  // At this point we create a dummy array of guards because we don't
  // know how many elements we will need.
  Type *Int32Ty = IRB.getInt32Ty();
  GuardArray =
      new GlobalVariable(M, Int32Ty, false, GlobalValue::ExternalLinkage,
                         nullptr, "__sancov_gen_cov_tmp");

  for (auto &F : M)
    runOnFunction(F);

  // Now we know how many elements we need. Create an array of guards
  // with one extra element at the beginning for the size.
  Type *Int32ArrayNTy =
      ArrayType::get(Int32Ty, SanCovFunction->getNumUses() + 1);
  GlobalVariable *RealGuardArray = new GlobalVariable(
      M, Int32ArrayNTy, false, GlobalValue::PrivateLinkage,
      Constant::getNullValue(Int32ArrayNTy), "__sancov_gen_cov");

  // Replace the dummy array with the real one.
  GuardArray->replaceAllUsesWith(
      IRB.CreatePointerCast(RealGuardArray, Int32PtrTy));
  GuardArray->eraseFromParent();

  // Call __sanitizer_cov_module_init
  IRB.SetInsertPoint(CtorFunc->getEntryBlock().getTerminator());
  IRB.CreateCall2(SanCovModuleInit,
                  IRB.CreatePointerCast(RealGuardArray, Int32PtrTy),
                  ConstantInt::get(IntptrTy, SanCovFunction->getNumUses()));
  return true;
}

bool SanitizerCoverageModule::runOnFunction(Function &F) {
  if (F.empty()) return false;
  if (F.getName().find(".module_ctor") != std::string::npos)
    return false;  // Should not instrument sanitizer init functions.
  if (CoverageLevel >= 3)
    SplitAllCriticalEdges(F);
  SmallVector<Instruction*, 8> IndirCalls;
  SmallVector<BasicBlock*, 16> AllBlocks;
  for (auto &BB : F) {
    AllBlocks.push_back(&BB);
    if (CoverageLevel >= 4)
      for (auto &Inst : BB) {
        CallSite CS(&Inst);
        if (CS && !CS.getCalledFunction())
          IndirCalls.push_back(&Inst);
      }
  }
  InjectCoverage(F, AllBlocks, IndirCalls);
  return true;
}

bool
SanitizerCoverageModule::InjectCoverage(Function &F,
                                        ArrayRef<BasicBlock *> AllBlocks,
                                        ArrayRef<Instruction *> IndirCalls) {
  if (!CoverageLevel) return false;

  if (CoverageLevel == 1 ||
      (unsigned)ClCoverageBlockThreshold < AllBlocks.size()) {
    InjectCoverageAtBlock(F, F.getEntryBlock());
  } else {
    for (auto BB : AllBlocks)
      InjectCoverageAtBlock(F, *BB);
  }
  InjectCoverageForIndirectCalls(F, IndirCalls);
  return true;
}

// On every indirect call we call a run-time function
// __sanitizer_cov_indir_call* with two parameters:
//   - callee address,
//   - global cache array that contains kCacheSize pointers (zero-initialized).
//     The cache is used to speed up recording the caller-callee pairs.
// The address of the caller is passed implicitly via caller PC.
// kCacheSize is encoded in the name of the run-time function.
void SanitizerCoverageModule::InjectCoverageForIndirectCalls(
    Function &F, ArrayRef<Instruction *> IndirCalls) {
  if (IndirCalls.empty()) return;
  const int kCacheSize = 16;
  const int kCacheAlignment = 64;  // Align for better performance.
  Type *Ty = ArrayType::get(IntptrTy, kCacheSize);
  for (auto I : IndirCalls) {
    IRBuilder<> IRB(I);
    CallSite CS(I);
    Value *Callee = CS.getCalledValue();
    if (dyn_cast<InlineAsm>(Callee)) continue;
    GlobalVariable *CalleeCache = new GlobalVariable(
        *F.getParent(), Ty, false, GlobalValue::PrivateLinkage,
        Constant::getNullValue(Ty), "__sancov_gen_callee_cache");
    CalleeCache->setAlignment(kCacheAlignment);
    IRB.CreateCall2(SanCovIndirCallFunction,
                    IRB.CreatePointerCast(Callee, IntptrTy),
                    IRB.CreatePointerCast(CalleeCache, IntptrTy));
  }
}

void SanitizerCoverageModule::InjectCoverageAtBlock(Function &F,
                                                    BasicBlock &BB) {
  BasicBlock::iterator IP = BB.getFirstInsertionPt(), BE = BB.end();
  // Skip static allocas at the top of the entry block so they don't become
  // dynamic when we split the block.  If we used our optimized stack layout,
  // then there will only be one alloca and it will come first.
  for (; IP != BE; ++IP) {
    AllocaInst *AI = dyn_cast<AllocaInst>(IP);
    if (!AI || !AI->isStaticAlloca())
      break;
  }

  bool IsEntryBB = &BB == &F.getEntryBlock();
  DebugLoc EntryLoc =
      IsEntryBB ? IP->getDebugLoc().getFnDebugLoc(*C) : IP->getDebugLoc();
  IRBuilder<> IRB(IP);
  IRB.SetCurrentDebugLocation(EntryLoc);
  SmallVector<Value *, 1> Indices;
  Value *GuardP = IRB.CreateAdd(
      IRB.CreatePointerCast(GuardArray, IntptrTy),
      ConstantInt::get(IntptrTy, (1 + SanCovFunction->getNumUses()) * 4));
  Type *Int32PtrTy = PointerType::getUnqual(IRB.getInt32Ty());
  GuardP = IRB.CreateIntToPtr(GuardP, Int32PtrTy);
  LoadInst *Load = IRB.CreateLoad(GuardP);
  Load->setAtomic(Monotonic);
  Load->setAlignment(4);
  Load->setMetadata(F.getParent()->getMDKindID("nosanitize"),
                    MDNode::get(*C, None));
  Value *Cmp = IRB.CreateICmpSGE(Constant::getNullValue(Load->getType()), Load);
  Instruction *Ins = SplitBlockAndInsertIfThen(
      Cmp, IP, false, MDBuilder(*C).createBranchWeights(1, 100000));
  IRB.SetInsertPoint(Ins);
  IRB.SetCurrentDebugLocation(EntryLoc);
  // __sanitizer_cov gets the PC of the instruction using GET_CALLER_PC.
  IRB.CreateCall(SanCovFunction, GuardP);
  IRB.CreateCall(EmptyAsm);  // Avoids callback merge.

  if (ClExperimentalTracing) {
    // Experimental support for tracing.
    // Insert a callback with the same guard variable as used for coverage.
    IRB.SetInsertPoint(IP);
    IRB.CreateCall(IsEntryBB ? SanCovTraceEnter : SanCovTraceBB, GuardP);
  }
}

char SanitizerCoverageModule::ID = 0;
INITIALIZE_PASS(SanitizerCoverageModule, "sancov",
    "SanitizerCoverage: TODO."
    "ModulePass", false, false)
ModulePass *llvm::createSanitizerCoverageModulePass(int CoverageLevel) {
  return new SanitizerCoverageModule(CoverageLevel);
}
