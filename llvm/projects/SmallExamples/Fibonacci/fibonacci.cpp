//===--- fibonacci.cpp - An example use of the JIT ----------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Valery A. Khamenya and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This small program provides an example of how to build quickly a small
// module with function Fibonacci and execute it with the JIT. 
//
// This simple example shows as well 30% speed up with LLVM 1.3
// in comparison to gcc 3.3.3 at AMD Athlon XP 1500+ .
//
// (Modified from HowToUseJIT.cpp and Stacker/lib/compiler/StackerCompiler.cpp)
// 
//===------------------------------------------------------------------------===
// Goal: 
//  The goal of this snippet is to create in the memory
//  the LLVM module consisting of one function as follow:
//
// int fib(int x) {
//   if(x<=2) return 1;
//   return fib(x-1)+fib(x-2);
// }
// 
// then compile the module via JIT, then execute the `fib' 
// function and return result to a driver, i.e. to a "host program".
//

#include <iostream>

#include <llvm/Module.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Constants.h>
#include <llvm/Instructions.h>
#include <llvm/ModuleProvider.h>
#include <llvm/Analysis/Verifier.h>
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"


using namespace llvm;

int main(int argc, char**argv) {

  int n = argc > 1 ? atol(argv[1]) : 44;

  // Create some module to put our function into it.
  Module *M = new Module("test");


  // We are about to create the "fib" function:
  Function *FibF;

  {
    // first create type for the single argument of fib function: 
    // the type is 'int ()'
    std::vector<const Type*> ArgT(1);
    ArgT[0] = Type::IntTy;

    // now create full type of the "fib" function:
    FunctionType *FibT = FunctionType::get(Type::IntTy, // type of result
					   ArgT,
					   /*not vararg*/false);
 
    // Now create the fib function entry and 
    // insert this entry into module M
    // (By passing a module as the last parameter to the Function constructor,
    // it automatically gets appended to the Module.)
    FibF = new Function(FibT, 
			Function::ExternalLinkage, // maybe too much
			"fib", M);

    // Add a basic block to the function... (again, it automatically inserts
    // because of the last argument.)
    BasicBlock *BB = new BasicBlock("EntryBlock of fib function", FibF);
  
    // Get pointers to the constants ...
    Value *One = ConstantSInt::get(Type::IntTy, 1);
    Value *Two = ConstantSInt::get(Type::IntTy, 2);

    // Get pointers to the integer argument of the add1 function...
    assert(FibF->abegin() != FibF->aend()); // Make sure there's an arg

    Argument &ArgX = FibF->afront();  // Get the arg
    ArgX.setName("AnArg");            // Give it a nice symbolic name for fun.

    SetCondInst* CondInst 
      = new SetCondInst( Instruction::SetLE, 
			 &ArgX, Two );

    BB->getInstList().push_back(CondInst);

    // Create the true_block
    BasicBlock* true_bb = new BasicBlock("arg<=2");


    // Create the return instruction and add it 
    // to the basic block for true case:
    true_bb->getInstList().push_back(new ReturnInst(One));
      
    // Create an exit block
    BasicBlock* exit_bb = new BasicBlock("arg>2");
    
    {

      // create fib(x-1)
      CallInst* CallFibX1;
      {
	// Create the sub instruction... does not insert...
	Instruction *Sub 
	  = BinaryOperator::create(Instruction::Sub, &ArgX, One,
						"arg");       
       
	exit_bb->getInstList().push_back(Sub);

	CallFibX1 = new CallInst(FibF, Sub, "fib(x-1)");
	exit_bb->getInstList().push_back(CallFibX1);
	 
      }

      // create fib(x-2)
      CallInst* CallFibX2;
      {
	// Create the sub instruction... does not insert...
	Instruction * Sub
	  = BinaryOperator::create(Instruction::Sub, &ArgX, Two,
						"arg");

	exit_bb->getInstList().push_back(Sub);
	CallFibX2 = new CallInst(FibF, Sub, "fib(x-2)");
	exit_bb->getInstList().push_back(CallFibX2);
	  
      }

      // Create the add instruction... does not insert...
      Instruction *Add = 
	BinaryOperator::create(Instruction::Add, 
			       CallFibX1, CallFibX2, "addresult");
      
      // explicitly insert it into the basic block...
      exit_bb->getInstList().push_back(Add);
      
      // Create the return instruction and add it to the basic block
      exit_bb->getInstList().push_back(new ReturnInst(Add));      
    }

    // Create a branch on the SetCond
    BranchInst* br_inst = 
      new BranchInst( true_bb, exit_bb, CondInst );

    BB->getInstList().push_back( br_inst );
    FibF->getBasicBlockList().push_back(true_bb);
    FibF->getBasicBlockList().push_back(exit_bb);
  }

  // Now we going to create JIT 
  ExistingModuleProvider* MP = new ExistingModuleProvider(M);
  ExecutionEngine* EE = ExecutionEngine::create( MP, false );

  // Call the `foo' function with argument n:
  std::vector<GenericValue> args(1);
  args[0].IntVal = n;


  std::clog << "verifying... ";
  if (verifyModule(*M)) {
    std::cerr << argv[0]
	      << ": assembly parsed, but does not verify as correct!\n";
    return 1;
  }
  else 
    std::clog << "OK\n";


  std::clog << "We just constructed this LLVM module:\n\n---------\n" << *M;
  std::clog << "---------\nstarting fibonacci(" 
	    << n << ") with JIT...\n" << std::flush;

  GenericValue gv = EE->runFunction(FibF, args);

  // import result of execution:
  std::cout << "Result: " << gv.IntVal << std:: endl;

  return 0;
}
