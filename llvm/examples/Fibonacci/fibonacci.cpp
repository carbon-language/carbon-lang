//===--- examples/Fibonacci/fibonacci.cpp - An example use of the JIT -----===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Valery A. Khamenya and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This small program provides an example of how to build quickly a small module
// with function Fibonacci and execute it with the JIT.
//
// The goal of this snippet is to create in the memory the LLVM module
// consisting of one function as follow:
//
//   int fib(int x) {
//     if(x<=2) return 1;
//     return fib(x-1)+fib(x-2);
//   }
// 
// Once we have this, we compile the module via JIT, then execute the `fib'
// function and return result to a driver, i.e. to a "host program".
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include <iostream>
using namespace llvm;

static Function *CreateFibFunction(Module *M) {
  // Create the fib function and insert it into module M.  This function is said
  // to return an int and take an int parameter.
  Function *FibF = M->getOrInsertFunction("fib", Type::IntTy, Type::IntTy, 0);
  
  // Add a basic block to the function.
  BasicBlock *BB = new BasicBlock("EntryBlock", FibF);
  
  // Get pointers to the constants.
  Value *One = ConstantSInt::get(Type::IntTy, 1);
  Value *Two = ConstantSInt::get(Type::IntTy, 2);

  // Get pointer to the integer argument of the add1 function...
  Argument *ArgX = FibF->arg_begin();   // Get the arg.
  ArgX->setName("AnArg");            // Give it a nice symbolic name for fun.

  // Create the true_block.
  BasicBlock *RetBB = new BasicBlock("return", FibF);
  // Create an exit block.
  BasicBlock* RecurseBB = new BasicBlock("recurse", FibF);

  // Create the "if (arg < 2) goto exitbb"
  Value *CondInst = BinaryOperator::createSetLE(ArgX, Two, "cond", BB);
  new BranchInst(RetBB, RecurseBB, CondInst, BB);

  // Create: ret int 1
  new ReturnInst(One, RetBB);
  
  // create fib(x-1)
  Value *Sub = BinaryOperator::createSub(ArgX, One, "arg", RecurseBB);
  Value *CallFibX1 = new CallInst(FibF, Sub, "fibx1", RecurseBB);
      
  // create fib(x-2)
  Sub = BinaryOperator::createSub(ArgX, Two, "arg", RecurseBB);
  Value *CallFibX2 = new CallInst(FibF, Sub, "fibx2", RecurseBB);

  // fib(x-1)+fib(x-2)
  Value *Sum = BinaryOperator::createAdd(CallFibX1, CallFibX2,
                                         "addresult", RecurseBB);
      
  // Create the return instruction and add it to the basic block
  new ReturnInst(Sum, RecurseBB);

  return FibF;
}


int main(int argc, char **argv) {
  int n = argc > 1 ? atol(argv[1]) : 24;

  // Create some module to put our function into it.
  Module *M = new Module("test");

  // We are about to create the "fib" function:
  Function *FibF = CreateFibFunction(M);

  // Now we going to create JIT 
  ExistingModuleProvider *MP = new ExistingModuleProvider(M);
  ExecutionEngine *EE = ExecutionEngine::create(MP, false);

  std::cerr << "verifying... ";
  if (verifyModule(*M)) {
    std::cerr << argv[0] << ": Error constructing function!\n";
    return 1;
  }

  std::cerr << "OK\n";
  std::cerr << "We just constructed this LLVM module:\n\n---------\n" << *M;
  std::cerr << "---------\nstarting fibonacci(" << n << ") with JIT...\n";

  // Call the Fibonacci function with argument n:
  std::vector<GenericValue> Args(1);
  Args[0].IntVal = n;
  GenericValue GV = EE->runFunction(FibF, Args);

  // import result of execution
  std::cout << "Result: " << GV.IntVal << "\n";
  return 0;
}
