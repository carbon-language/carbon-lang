//===- LineProfiling.cpp - Insert counters for line profiling -------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass creates counters for the number of times that the original source
// lines of code were executed.
//
// The lines are found from existing debug info in the LLVM IR. Iterating
// through LLVM instructions, every time the debug location changes we insert a
// new counter and instructions to increment the counter there. A global
// destructor runs to dump the counters out to a file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "insert-line-profiling"

#include "ProfilingUtils.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLoc.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include <set>
#include <string>
using namespace llvm;

STATISTIC(NumUpdatesInserted, "The # of counter increments inserted.");

namespace {
  class LineProfiler : public ModulePass {
    bool runOnModule(Module &M);
  public:
    static char ID;
    LineProfiler() : ModulePass(ID) {
      initializeLineProfilerPass(*PassRegistry::getPassRegistry());
    }
    virtual const char *getPassName() const {
      return "Line Profiler";
    }

  private:
    // Get pointers to the functions in the runtime library.
    Constant *getStartFileFunc();
    Constant *getCounterFunc();
    Constant *getEndFileFunc();

    // Insert an increment of the counter before instruction I.
    void InsertCounterUpdateBefore(Instruction *I);

    // Add the function to write out all our counters to the global destructor
    // list.
    void InsertCounterWriteout();

    // Mapping from the source location to the counter tracking that location.
    DenseMap<DebugLoc, GlobalVariable *> counters;

    Module *Mod;
    LLVMContext *Ctx;
  };
}

char LineProfiler::ID = 0;
INITIALIZE_PASS(LineProfiler, "insert-line-profiling",
                "Insert instrumentation for line profiling", false, false)

ModulePass *llvm::createLineProfilerPass() { return new LineProfiler(); }

bool LineProfiler::runOnModule(Module &M) {
  Mod = &M;
  Ctx = &M.getContext();

  DebugLoc last_line;  // initializes to unknown
  bool Changed = false;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    for (inst_iterator II = inst_begin(F), IE = inst_end(F); II != IE; ++II) {
      const DebugLoc &loc = II->getDebugLoc();
      if (loc.isUnknown()) continue;
      if (loc == last_line) continue;
      last_line = loc;

      InsertCounterUpdateBefore(&*II);
      ++NumUpdatesInserted;
      Changed = true;
    }
  }

  if (Changed) {
    InsertCounterWriteout();
  }

  return Changed;
}

void LineProfiler::InsertCounterUpdateBefore(Instruction *I) {
  const DebugLoc &loc = I->getDebugLoc();
  GlobalVariable *&counter = counters[loc];
  const Type *Int64Ty = Type::getInt64Ty(*Ctx);
  if (!counter) {
    counter = new GlobalVariable(*Mod, Int64Ty, false,
                                 GlobalValue::InternalLinkage,
                                 Constant::getNullValue(Int64Ty),
                                 "__llvm_prof_linecov_ctr", 0, false, 0);
    counter->setVisibility(GlobalVariable::HiddenVisibility);
    counter->setUnnamedAddr(true);
  }

  if (isa<PHINode>(I)) {
    // We may not error out or crash in this case, because a module could put
    // changing line numbers on phi nodes and still pass the verifier.
    dbgs() << "Refusing to insert code before phi: " << *I << "\n";
    I = I->getParent()->getFirstNonPHI();
  }

  IRBuilder<> builder(I);
  Value *ctr = builder.CreateLoad(counter);
  ctr = builder.CreateAdd(ctr, ConstantInt::get(Int64Ty, 1));
  builder.CreateStore(ctr, counter);
}

static DISubprogram FindSubprogram(DIScope scope) {
  while (!scope.isSubprogram()) {
    assert(scope.isLexicalBlock() &&
           "Debug location not lexical block or subprogram");
    scope = DILexicalBlock(scope).getContext();
  }
  return DISubprogram(scope);
}

Constant *LineProfiler::getStartFileFunc() {
  const Type *Args[1] = { Type::getInt8PtrTy(*Ctx) };
  const FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx),
                                              Args, false);
  return Mod->getOrInsertFunction("llvm_prof_linectr_start_file", FTy);
}

Constant *LineProfiler::getCounterFunc() {
  const Type *Args[] = {
    Type::getInt8PtrTy(*Ctx),   // const char *dir
    Type::getInt8PtrTy(*Ctx),   // const char *file
    Type::getInt32Ty(*Ctx),     // uint32_t line
    Type::getInt32Ty(*Ctx),     // uint32_t column
    Type::getInt64PtrTy(*Ctx),  // int64_t *counter
  };
  const FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx),
                                              Args, false);
  return Mod->getOrInsertFunction("llvm_prof_linectr_emit_counter", FTy);
}

Constant *LineProfiler::getEndFileFunc() {
  const FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx), false);
  return Mod->getOrInsertFunction("llvm_prof_linectr_end_file", FTy);
}

void LineProfiler::InsertCounterWriteout() {
  std::set<std::string> compile_units;
  for (DenseMap<DebugLoc, GlobalVariable *>::iterator I = counters.begin(),
           E = counters.end(); I != E; ++I) {
    const DebugLoc &loc = I->first;
    DISubprogram subprogram(FindSubprogram(DIScope(loc.getScope(*Ctx))));
    compile_units.insert(subprogram.getCompileUnit().getFilename().str());
  }

  const FunctionType *WriteoutFTy =
      FunctionType::get(Type::getVoidTy(*Ctx), false);
  Function *WriteoutF = Function::Create(WriteoutFTy,
                                         GlobalValue::InternalLinkage,
                                         "__llvm_prof_linecov_dtor",
                                         Mod);
  WriteoutF->setUnnamedAddr(true);
  BasicBlock *BB = BasicBlock::Create(*Ctx, "", WriteoutF);
  IRBuilder<> builder(BB);

  Constant *StartFile = getStartFileFunc();
  Constant *EmitCounter = getCounterFunc();
  Constant *EndFile = getEndFileFunc();

  for (std::set<std::string>::const_iterator CUI = compile_units.begin(),
           CUE = compile_units.end(); CUI != CUE; ++CUI) {
    builder.CreateCall(StartFile,
                       builder.CreateGlobalStringPtr(*CUI));
    for (DenseMap<DebugLoc, GlobalVariable *>::iterator I = counters.begin(),
             E = counters.end(); I != E; ++I) {
      const DebugLoc &loc = I->first;
      DISubprogram subprogram(FindSubprogram(DIScope(loc.getScope(*Ctx))));
      DICompileUnit compileunit(subprogram.getCompileUnit());

      if (compileunit.getFilename() != *CUI)
        continue;

      Value *Args[] = {
        builder.CreateGlobalStringPtr(subprogram.getDirectory()),
        builder.CreateGlobalStringPtr(subprogram.getFilename()),
        ConstantInt::get(Type::getInt32Ty(*Ctx), loc.getLine()),
        ConstantInt::get(Type::getInt32Ty(*Ctx), loc.getCol()),
        I->second
      };
      builder.CreateCall(EmitCounter, Args);
    }
    builder.CreateCall(EndFile);
  }
  builder.CreateRetVoid();

  InsertProfilingShutdownCall(WriteoutF, Mod);
}
