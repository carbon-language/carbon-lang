//===- BlockProfiling.cpp - Insert counters for block profiling -----------===//
// 
//                      The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass instruments the specified program with counters for basic block or
// function profiling.  This is the most basic form of profiling, which can tell
// which blocks are hot, but cannot reliably detect hot paths through the CFG.
// Block profiling counts the number of times each basic block executes, and
// function profiling counts the number of times each function is called.
//
// Note that this implementation is very naive.  Control equivalent regions of
// the CFG should not require duplicate counters, but we do put duplicate
// counters in.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"

namespace llvm {

static void insertInitializationCall(Function *MainFn, const char *FnName,
                                     GlobalValue *Array) {
  const Type *ArgVTy = PointerType::get(PointerType::get(Type::SByteTy));
  const Type *UIntPtr = PointerType::get(Type::UIntTy);
  Module &M = *MainFn->getParent();
  Function *InitFn = M.getOrInsertFunction(FnName, Type::VoidTy, Type::IntTy,
                                           ArgVTy, UIntPtr, Type::UIntTy, 0);

  // This could force argc and argv into programs that wouldn't otherwise have
  // them, but instead we just pass null values in.
  std::vector<Value*> Args(4);
  Args[0] = Constant::getNullValue(Type::IntTy);
  Args[1] = Constant::getNullValue(ArgVTy);

  // Skip over any allocas in the entry block.
  BasicBlock *Entry = MainFn->begin();
  BasicBlock::iterator InsertPos = Entry->begin();
  while (isa<AllocaInst>(InsertPos)) ++InsertPos;

  Function::aiterator AI;
  switch (MainFn->asize()) {
  default:
  case 2:
    AI = MainFn->abegin(); ++AI;
    if (AI->getType() != ArgVTy) {
      Args[1] = new CastInst(AI, ArgVTy, "argv.cast", InsertPos);
    } else {
      Args[1] = AI;
    }

  case 1:
    AI = MainFn->abegin();
    if (AI->getType() != Type::IntTy) {
      Args[0] = new CastInst(AI, Type::IntTy, "argc.cast", InsertPos);
    } else {
      Args[0] = AI;
    }
    
  case 0:
    break;
  }

  ConstantPointerRef *ArrayCPR = ConstantPointerRef::get(Array);
  std::vector<Constant*> GEPIndices(2, Constant::getNullValue(Type::LongTy));
  Args[2] = ConstantExpr::getGetElementPtr(ArrayCPR, GEPIndices);
  
  unsigned NumElements =
    cast<ArrayType>(Array->getType()->getElementType())->getNumElements();
  Args[3] = ConstantUInt::get(Type::UIntTy, NumElements);
  
  new CallInst(InitFn, Args, "", InsertPos);
}

static void IncrementCounterInBlock(BasicBlock *BB, unsigned CounterNum,
                                    ConstantPointerRef *CounterArray) {
  // Insert the increment after any alloca or PHI instructions...
  BasicBlock::iterator InsertPos = BB->begin();
  while (isa<AllocaInst>(InsertPos) || isa<PHINode>(InsertPos))
    ++InsertPos;

  // Create the getelementptr constant expression
  std::vector<Constant*> Indices(2);
  Indices[0] = Constant::getNullValue(Type::LongTy);
  Indices[1] = ConstantSInt::get(Type::LongTy, CounterNum);
  Constant *ElementPtr = ConstantExpr::getGetElementPtr(CounterArray, Indices);

  // Load, increment and store the value back.
  Value *OldVal = new LoadInst(ElementPtr, "OldFuncCounter", InsertPos);
  Value *NewVal = BinaryOperator::create(Instruction::Add, OldVal,
                                         ConstantInt::get(Type::UIntTy, 1),
                                         "NewFuncCounter", InsertPos);
  new StoreInst(NewVal, ElementPtr, InsertPos);
}


namespace {
  class FunctionProfiler : public Pass {
    bool run(Module &M);
  };

  RegisterOpt<FunctionProfiler> X("insert-function-profiling",
                               "Insert instrumentation for function profiling");
}

bool FunctionProfiler::run(Module &M) {
  Function *Main = M.getMainFunction();
  if (Main == 0) {
    std::cerr << "WARNING: cannot insert function profiling into a module"
              << " with no main function!\n";
    return false;  // No main, no instrumentation!
  }

  unsigned NumFunctions = 0;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) 
    if (!I->isExternal())
      ++NumFunctions;

  const Type *ATy = ArrayType::get(Type::UIntTy, NumFunctions);
  GlobalVariable *Counters =
    new GlobalVariable(ATy, false, GlobalValue::InternalLinkage,
                       Constant::getNullValue(ATy), "FuncProfCounters", &M);

  ConstantPointerRef *CounterCPR = ConstantPointerRef::get(Counters);

  // Instrument all of the functions...
  unsigned i = 0;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) 
    if (!I->isExternal())
      // Insert counter at the start of the function
      IncrementCounterInBlock(I->begin(), i++, CounterCPR);

  // Add the initialization call to main.
  insertInitializationCall(Main, "llvm_start_func_profiling", Counters);
  return true;
}


namespace {
  class BlockProfiler : public Pass {
    bool run(Module &M);
  };

  RegisterOpt<BlockProfiler> Y("insert-block-profiling",
                               "Insert instrumentation for block profiling");
}

bool BlockProfiler::run(Module &M) {
  Function *Main = M.getMainFunction();
  if (Main == 0) {
    std::cerr << "WARNING: cannot insert block profiling into a module"
              << " with no main function!\n";
    return false;  // No main, no instrumentation!
  }

  unsigned NumBlocks = 0;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) 
    NumBlocks += I->size();

  const Type *ATy = ArrayType::get(Type::UIntTy, NumBlocks);
  GlobalVariable *Counters =
    new GlobalVariable(ATy, false, GlobalValue::InternalLinkage,
                       Constant::getNullValue(ATy), "BlockProfCounters", &M);

  ConstantPointerRef *CounterCPR = ConstantPointerRef::get(Counters);

  // Instrument all of the blocks...
  unsigned i = 0;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) 
    for (Function::iterator BB = I->begin(), E = I->end(); BB != E; ++BB)
      // Insert counter at the start of the block
      IncrementCounterInBlock(BB, i++, CounterCPR);

  // Add the initialization call to main.
  insertInitializationCall(Main, "llvm_start_block_profiling", Counters);
  return true;
}

} // End llvm namespace
