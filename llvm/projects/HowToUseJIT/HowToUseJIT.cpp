//===--- HowToUseJIT.cpp - An example use of the JIT ----------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Valery A. Khamenya and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This small program provides an example of how to quickly build a small
//  module with two functions and execute it with the JIT. 
// 
//===------------------------------------------------------------------------===

// Goal: 
//  The goal of this snippet is to create in the memory
//  the LLVM module consisting of two functions as follow:
//
// int add1(int x) {
//   return x+1;
// }
// 
// int foo() {
//   return add1(10);
// }
// 
// then compile the module via JIT, then execute the `foo' 
// function and return result to a driver, i.e. to a "host program".
// 
// Some remarks and questions:
// 
// - could we invoke some code using noname functions too?
//   e.g. evaluate "foo()+foo()" without fears to introduce 
//   conflict of temporary function name with some real
//   existing function name?
// 

#include <iostream>

#include <llvm/Module.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Constants.h>
#include <llvm/Instructions.h>
#include <llvm/ModuleProvider.h>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"


using namespace llvm;

int main() {

  // Create some module to put our function into it.
  Module *M = new Module("test");


  // We are about to create the add1 function:
  Function *Add1F;

  {
    // first create type for the single argument of add1 function: 
    // the type is 'int ()'
    std::vector<const Type*> ArgT(1);
    ArgT[0] = Type::IntTy;

    // now create full type of the add1 function:
    FunctionType *Add1T = FunctionType::get(Type::IntTy, // type of result
                                            ArgT,
                                            /*not vararg*/false);
 
    // Now create the add1 function entry and 
    // insert this entry into module M
    // (By passing a module as the last parameter to the Function constructor,
    // it automatically gets appended to the Module.)
    Add1F = new Function(Add1T, 
                         Function::ExternalLinkage, // maybe too much
                         "add1", M);

    // Add a basic block to the function... (again, it automatically inserts
    // because of the last argument.)
    BasicBlock *BB = new BasicBlock("EntryBlock of add1 function", Add1F);
  
    // Get pointers to the constant `1'...
    Value *One = ConstantSInt::get(Type::IntTy, 1);

    // Get pointers to the integer argument of the add1 function...
    assert(Add1F->abegin() != Add1F->aend()); // Make sure there's an arg
    Argument &ArgX = Add1F->afront();  // Get the arg

    // Create the add instruction... does not insert...
    Instruction *Add = BinaryOperator::create(Instruction::Add, One, &ArgX,
                                              "addresult");
  
    // explicitly insert it into the basic block...
    BB->getInstList().push_back(Add);
  
    // Create the return instruction and add it to the basic block
    BB->getInstList().push_back(new ReturnInst(Add));

    // function add1 is ready
  }


  // now we going to create function `foo':
  Function *FooF;

  {  
    // Create the foo function type:
    FunctionType *FooT = 
      FunctionType::get(Type::IntTy, // result has type: 'int ()'
                        std::vector<const Type*>(), // no arguments
                        /*not vararg*/false);
    
    // create the entry for function `foo' and insert
    // this entry into module M:
    FooF = 
      new Function(FooT, 
                   Function::ExternalLinkage, // too wide?
                   "foo", M);
    
    // Add a basic block to the FooF function...
    BasicBlock *BB = new BasicBlock("EntryBlock of add1 function", FooF);

    // Get pointers to the constant `10'...
    Value *Ten = ConstantSInt::get(Type::IntTy, 10);

    // Put the argument Ten on stack and make call:
    // ...
    std::vector<Value*> Params;
    Params.push_back(Ten);
    CallInst * Add1CallRes = new CallInst(Add1F, Params, "add1", BB);
    
    // Create the return instruction and add it to the basic block
    BB->getInstList().push_back(new ReturnInst(Add1CallRes));
    
  }

  // Now we going to create JIT ??
  ExistingModuleProvider* MP = new ExistingModuleProvider(M);
  ExecutionEngine* EE = ExecutionEngine::create( MP, true );

  // Call the `foo' function with no arguments:
  std::vector<GenericValue> noargs;
  GenericValue gv = EE->runFunction(FooF, noargs);

  // import result of execution:
  std::cout << "Result: " << gv.IntVal << std:: endl;

  return 0;
}
