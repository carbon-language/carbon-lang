//===- ModuleMaker.cpp - Example project which creates modules --*- C++ -*-===//
//
// This programs is a simple example that creates an LLVM module "from scratch",
// emitting it as a bytecode file to standard out.  This is just to show how
// LLVM projects work and to demonstrate some of the LLVM APIs.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Bytecode/Writer.h"

using namespace llvm;

int main() {
  // Create the "module" or "program" or "translation unit" to hold the
  // function
  Module *M = new Module("test");
  
  // Create the main function: first create the type 'int ()'
  FunctionType *FT = FunctionType::get(Type::IntTy, std::vector<const Type*>(),
                                       /*not vararg*/false);
  
  // By passing a module as the last parameter to the Function constructor,
  // it automatically gets appended to the Module.
  Function *F = new Function(FT, Function::ExternalLinkage, "main", M);
  
  // Add a basic block to the function... again, it automatically inserts
  // because of the last argument.
  BasicBlock *BB = new BasicBlock("EntryBlock", F);
  
  // Get pointers to the constant integers...
  Value *Two = ConstantSInt::get(Type::IntTy, 2);
  Value *Three = ConstantSInt::get(Type::IntTy, 3);
  
  // Create the add instruction... does not insert...
  Instruction *Add = BinaryOperator::create(Instruction::Add, Two, Three,
                                            "addresult");
  
  // explicitly insert it into the basic block...
  BB->getInstList().push_back(Add);
  
  // Create the return instruction and add it to the basic block
  BB->getInstList().push_back(new ReturnInst(Add));
  
  // Output the bytecode file to stdout
  WriteBytecodeToFile(M, std::cout);
  
  // Delete the module and all of its contents.
  delete M;
  return 0;
}
