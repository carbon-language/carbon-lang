//===- CodeGeneratorBug.cpp - Debug code generation bugs ------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements program code generation debugging support.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "ListReducer.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Linker.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/StringExtras.h"
#include "Support/FileUtilities.h"
using namespace llvm;

namespace llvm {
  extern cl::list<std::string> InputArgv;

  class ReduceMisCodegenFunctions : public ListReducer<Function*> {
    BugDriver &BD;
  public:
    ReduceMisCodegenFunctions(BugDriver &bd) : BD(bd) {}
    
    virtual TestResult doTest(std::vector<Function*> &Prefix,
                              std::vector<Function*> &Suffix) {
      if (!Prefix.empty() && TestFuncs(Prefix))
        return KeepPrefix;
      if (!Suffix.empty() && TestFuncs(Suffix))
        return KeepSuffix;
      return NoFailure;
    }
    
    bool TestFuncs(const std::vector<Function*> &CodegenTest,
                   bool KeepFiles = false);
  };
}

bool ReduceMisCodegenFunctions::TestFuncs(const std::vector<Function*> &Funcs,
                                          bool KeepFiles) {
  std::cout << "Testing functions: ";
  PrintFunctionList(Funcs);
  std::cout << "\t";

  // Clone the module for the two halves of the program we want.
  Module *SafeModule = CloneModule(BD.getProgram());

  // The JIT must extract the 'main' function.
  std::vector<Function*> RealFuncs(Funcs);
  if (BD.isExecutingJIT()) {
    if (Function *F = BD.Program->getMainFunction())
      RealFuncs.push_back(F);
  }
  Module *TestModule = SplitFunctionsOutOfModule(SafeModule, RealFuncs);

  // This is only applicable if we are debugging the JIT:
  // Find all external functions in the Safe modules that are actually used
  // (called or taken address of), and make them call the JIT wrapper instead
  if (BD.isExecutingJIT()) {
    // Must delete `main' from Safe module if it has it
    Function *safeMain = SafeModule->getNamedFunction("main");
    assert(safeMain && "`main' function not found in safe module!");
    DeleteFunctionBody(safeMain);

    // Add an external function "getPointerToNamedFunction" that JIT provides
    // Prototype: void *getPointerToNamedFunction(const char* Name)
    std::vector<const Type*> Params;
    Params.push_back(PointerType::get(Type::SByteTy)); // std::string&
    FunctionType *resolverTy = FunctionType::get(PointerType::get(Type::VoidTy),
                                                 Params, false /* isVarArg */);
    Function *resolverFunc = new Function(resolverTy,
                                          GlobalValue::ExternalLinkage,
                                          "getPointerToNamedFunction",
                                          SafeModule);

    // Use the function we just added to get addresses of functions we need
    // Iterate over the global declarations in the Safe module
    for (Module::iterator F=SafeModule->begin(),E=SafeModule->end(); F!=E; ++F){
      if (F->isExternal() && !F->use_empty() && &*F != resolverFunc &&
          F->getIntrinsicID() == 0 /* ignore intrinsics */ &&
          // Don't forward functions which are external in the test module too.
          !TestModule->getNamedFunction(F->getName())->isExternal()) {
        // If it has a non-zero use list,
        // 1. Add a string constant with its name to the global file
        Constant *InitArray = ConstantArray::get(F->getName());
        GlobalVariable *funcName =
          new GlobalVariable(InitArray->getType(), true /* isConstant */,
                             GlobalValue::InternalLinkage, InitArray,    
                             F->getName() + "_name", SafeModule);

        // 2. Use `GetElementPtr *funcName, 0, 0' to convert the string to an
        // sbyte* so it matches the signature of the resolver function.
        std::vector<Constant*> GEPargs(2, Constant::getNullValue(Type::IntTy));

        // 3. Replace all uses of `func' with calls to resolver by:
        // (a) Iterating through the list of uses of this function
        // (b) Insert a cast instruction in front of each use
        // (c) Replace use of old call with new call

        // GetElementPtr *funcName, ulong 0, ulong 0
        Value *GEP =
          ConstantExpr::getGetElementPtr(ConstantPointerRef::get(funcName),
                                         GEPargs);
        std::vector<Value*> ResolverArgs;
        ResolverArgs.push_back(GEP);

        // Insert code at the beginning of the function
        while (!F->use_empty())
          if (Instruction *Inst = dyn_cast<Instruction>(F->use_back())) {
            // call resolver(GetElementPtr...)
            CallInst *resolve = new CallInst(resolverFunc, ResolverArgs, 
                                             "resolver", Inst);
            // cast the result from the resolver to correctly-typed function
            CastInst *castResolver =
              new CastInst(resolve, PointerType::get(F->getFunctionType()),
                           "resolverCast", Inst);
            // actually use the resolved function
            Inst->replaceUsesOfWith(F, castResolver);
          } else {
            // FIXME: need to take care of cases where a function is used by
            // something other than an instruction; e.g., global variable
            // initializers and constant expressions.
            std::cerr << "UNSUPPORTED: Non-instruction is using an external "
                      << "function, " << F->getName() << "().\n";
            abort();
          }
      }
    }
  }

  if (verifyModule(*SafeModule) || verifyModule(*TestModule)) {
    std::cerr << "Bugpoint has a bug, an corrupted a module!!\n";
    abort();
  }

  // Remove all functions from the Test module EXCEPT for the ones specified in
  // Funcs.  We know which ones these are because they are non-external in
  // ToOptimize, but external in ToNotOptimize.
  //
  for (Module::iterator I = TestModule->begin(), E = TestModule->end();I!=E;++I)
    if (!I->isExternal()) {
      Function *TNOF = SafeModule->getFunction(I->getName(),
                                               I->getFunctionType());
      assert(TNOF && "Function doesn't exist in ToNotOptimize module??");
      if (!TNOF->isExternal())
        DeleteFunctionBody(I);
    }

  // Clean up the modules, removing extra cruft that we don't need anymore...
  TestModule = BD.performFinalCleanups(TestModule);

  std::string TestModuleBC = getUniqueFilename("bugpoint.test.bc");
  if (BD.writeProgramToFile(TestModuleBC, TestModule)) {
    std::cerr << "Error writing bytecode to `" << TestModuleBC << "'\nExiting.";
    exit(1);
  }
  delete TestModule;

  // Make the shared library
  std::string SafeModuleBC = getUniqueFilename("bugpoint.safe.bc");

  if (BD.writeProgramToFile(SafeModuleBC, SafeModule)) {
    std::cerr << "Error writing bytecode to `" << SafeModuleBC << "'\nExiting.";
    exit(1);
  }
  std::string SharedObject = BD.compileSharedObject(SafeModuleBC);
  delete SafeModule;

  // Run the code generator on the `Test' code, loading the shared library.
  // The function returns whether or not the new output differs from reference.
  int Result = BD.diffProgram(TestModuleBC, SharedObject, false);

  if (Result)
    std::cerr << ": still failing!\n";
  else
    std::cerr << ": didn't fail.\n";
    
  if (KeepFiles) {
    std::cout << "You can reproduce the problem with the command line: \n";
    if (BD.isExecutingJIT()) {
      std::cout << "  lli -load " << SharedObject << " " << TestModuleBC;
    } else {
      std::cout << "  llc " << TestModuleBC << " -o " << TestModuleBC << ".s\n";
      std::cout << "  gcc " << SharedObject << " " << TestModuleBC
                << ".s -o " << TestModuleBC << ".exe -Wl,-R.\n";
      std::cout << "  " << TestModuleBC << ".exe";
    }
    for (unsigned i=0, e = InputArgv.size(); i != e; ++i)
      std::cout << " " << InputArgv[i];
    std::cout << "\n";
    std::cout << "The shared object was created with:\n  llc -march=c "
              << SafeModuleBC << " -o temporary.c\n"
              << "  gcc -xc temporary.c -O2 -o " << SharedObject
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)
              << " -G"            // Compile a shared library, `-G' for Sparc
#else
              << " -shared"       // `-shared' for Linux/X86, maybe others
#endif
              << " -fno-strict-aliasing\n";
  } else {
    removeFile(TestModuleBC);
    removeFile(SafeModuleBC);
    removeFile(SharedObject);
  }
  return Result;
}

static void DisambiguateGlobalSymbols(Module *M) {
  // Try not to cause collisions by minimizing chances of renaming an
  // already-external symbol, so take in external globals and functions as-is.
  // The code should work correctly without disambiguation (assuming the same
  // mangler is used by the two code generators), but having symbols with the
  // same name causes warnings to be emitted by the code generator.
  Mangler Mang(*M);
  DEBUG(std::cerr << "Disambiguating globals (external-only)\n");
  for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
    I->setName(Mang.getValueName(I));
  DEBUG(std::cerr << "Disambiguating functions (external-only)\n");
  for (Module::iterator  I = M->begin(),  E = M->end();  I != E; ++I)
    I->setName(Mang.getValueName(I));
}


bool BugDriver::debugCodeGenerator() {
  if ((void*)cbe == (void*)Interpreter) {
    std::string Result = executeProgramWithCBE("bugpoint.cbe.out");
    std::cout << "\n*** The C backend cannot match the reference diff, but it "
              << "is used as the 'known good'\n    code generator, so I can't"
              << " debug it.  Perhaps you have a front-end problem?\n    As a"
              << " sanity check, I left the result of executing the program "
              << "with the C backend\n    in this file for you: '"
              << Result << "'.\n";
    return true;
  }

  // See if we can pin down which functions are being miscompiled...
  // First, build a list of all of the non-external functions in the program.
  std::vector<Function*> MisCodegenFunctions;
  for (Module::iterator I = Program->begin(), E = Program->end(); I != E; ++I)
    if (!I->isExternal())
      MisCodegenFunctions.push_back(I);

  // If we are executing the JIT, we *must* keep the function `main' in the
  // module that is passed in, and not the shared library. However, we still
  // want to be able to debug the `main' function alone. Thus, we create a new
  // function `main' which just calls the old one.
  if (isExecutingJIT()) {
    // Get the `main' function
    Function *oldMain = Program->getNamedFunction("main");
    assert(oldMain && "`main' function not found in program!");
    // Rename it
    oldMain->setName("llvm_old_main");
    // Create a NEW `main' function with same type
    Function *newMain = new Function(oldMain->getFunctionType(), 
                                     GlobalValue::ExternalLinkage,
                                     "main", Program);
    // Call the old main function and return its result
    BasicBlock *BB = new BasicBlock("entry", newMain);
    std::vector<Value*> args;
    for (Function::aiterator I = newMain->abegin(), E = newMain->aend(),
           OI = oldMain->abegin(); I != E; ++I, ++OI) {
      I->setName(OI->getName());    // Copy argument names from oldMain
      args.push_back(I);
    }
    CallInst *call = new CallInst(oldMain, args);
    BB->getInstList().push_back(call);
    
    // if the type of old function wasn't void, return value of call
    if (oldMain->getReturnType() != Type::VoidTy) {
      new ReturnInst(call, BB);
    } else {
      new ReturnInst(0, BB);
    }
  }

  DisambiguateGlobalSymbols(Program);

  // Do the reduction...
  if (!ReduceMisCodegenFunctions(*this).reduceList(MisCodegenFunctions)) {
    std::cerr << "*** Execution matches reference output! "
	      << "bugpoint can't help you with your problem!\n";
    return false;
  }

  std::cout << "\n*** The following functions are being miscompiled: ";
  PrintFunctionList(MisCodegenFunctions);
  std::cout << "\n";

  // Output a bunch of bytecode files for the user...
  ReduceMisCodegenFunctions(*this).TestFuncs(MisCodegenFunctions, true);

  return false;
}
