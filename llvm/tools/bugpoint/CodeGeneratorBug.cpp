//===- CodeGeneratorBug.cpp - Debug code generation bugs ------------------===//
//
// This file implements program code generation debugging support.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "SystemUtils.h"
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
#include "Support/Statistic.h"
#include "Support/StringExtras.h"
#include <algorithm>
#include <set>

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


bool ReduceMisCodegenFunctions::TestFuncs(const std::vector<Function*> &Funcs,
                                          bool KeepFiles)
{
  std::cout << "Testing functions: ";
  BD.PrintFunctionList(Funcs);
  std::cout << "\t";

  // Clone the module for the two halves of the program we want.
  Module *SafeModule = CloneModule(BD.Program);

  // Make sure functions & globals are all external so that linkage
  // between the two modules will work.
  for (Module::iterator I = SafeModule->begin(), E = SafeModule->end();I!=E;++I)
    I->setLinkage(GlobalValue::ExternalLinkage);
  for (Module::giterator I=SafeModule->gbegin(),E = SafeModule->gend();I!=E;++I)
    I->setLinkage(GlobalValue::ExternalLinkage);

  Module *TestModule = CloneModule(SafeModule);

  // Make sure global initializers exist only in the safe module (CBE->.so)
  for (Module::giterator I=TestModule->gbegin(),E = TestModule->gend();I!=E;++I)
    I->setInitializer(0);  // Delete the initializer to make it external

  // Remove the Test functions from the Safe module
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i) {
    Function *TNOF = SafeModule->getFunction(Funcs[i]->getName(),
                                             Funcs[i]->getFunctionType());
    DEBUG(std::cerr << "Removing function " << Funcs[i]->getName() << "\n");
    assert(TNOF && "Function doesn't exist in module!");
    DeleteFunctionBody(TNOF);       // Function is now external in this module!
  }

  // Remove the Safe functions from the Test module
  for (Module::iterator I=TestModule->begin(),E=TestModule->end(); I!=E; ++I) {
    bool funcFound = false;
    for (std::vector<Function*>::const_iterator F=Funcs.begin(),Fe=Funcs.end();
         F != Fe; ++F)
      if (I->getName() == (*F)->getName()) funcFound = true;

    if (!funcFound && !(BD.isExecutingJIT() && I->getName() == "main"))
      DeleteFunctionBody(I);
  }

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
      if (F->isExternal() && !F->use_empty() && &(*F) != resolverFunc &&
          F->getIntrinsicID() == 0 /* ignore intrinsics */) {
        // If it has a non-zero use list,
        // 1. Add a string constant with its name to the global file
        // The correct type is `const [ NUM x sbyte ]' where NUM is length of
        // function name + 1
        const std::string &Name = F->getName();
        GlobalVariable *funcName =
          new GlobalVariable(ArrayType::get(Type::SByteTy, Name.length()+1),
                             true /* isConstant */,
                             GlobalValue::InternalLinkage,
                             ConstantArray::get(Name),
                             Name + "_name",
                             SafeModule);

        // 2. Use `GetElementPtr *funcName, 0, 0' to convert the string to an
        // sbyte* so it matches the signature of the resolver function.
        std::vector<Constant*> GEPargs(2, Constant::getNullValue(Type::LongTy));

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
        for (Value::use_iterator i=F->use_begin(), e=F->use_end(); i!=e; ++i) {
          if (Instruction* Inst = dyn_cast<Instruction>(*i)) {
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
            // FIXME: need to take care of cases where a function is used that
            // is not an instruction, e.g. global variable initializer...
            std::cerr << "Non-instruction is using an external function!\n";
            abort();
          }
        }
      }
    }
  }

  DEBUG(std::cerr << "Safe module:\n";
        typedef Module::iterator MI;
        typedef Module::giterator MGI;

        for (MI I = SafeModule->begin(), E = SafeModule->end(); I != E; ++I)
          if (!I->isExternal()) std::cerr << "\t" << I->getName() << "\n";
        for (MGI I = SafeModule->gbegin(), E = SafeModule->gend(); I!=E; ++I)
          if (!I->isExternal()) std::cerr << "\t" << I->getName() << "\n";

        std::cerr << "Test module:\n";
        for (MI I = TestModule->begin(), E = TestModule->end(); I != E; ++I)
          if (!I->isExternal()) std::cerr << "\t" << I->getName() << "\n";
        for (MGI I=TestModule->gbegin(),E = TestModule->gend(); I!= E; ++I)
          if (!I->isExternal()) std::cerr << "\t" << I->getName() << "\n";
        );

  // Write out the bytecode to be sent to CBE
  std::string SafeModuleBC = getUniqueFilename("bugpoint.safe.bc");
  if (verifyModule(*SafeModule)) {
    std::cerr << "Bytecode file corrupted!\n";
    exit(1);
  }
  if (BD.writeProgramToFile(SafeModuleBC, SafeModule)) {
    std::cerr << "Error writing bytecode to `" << SafeModuleBC << "'\nExiting.";
    exit(1);
  }

  // Make a shared library
  std::string SharedObject;
  BD.compileSharedObject(SafeModuleBC, SharedObject);

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

  std::string TestModuleBC = getUniqueFilename("bugpoint.test.bc");
  if (verifyModule(*TestModule)) {
    std::cerr << "Bytecode file corrupted!\n";
    exit(1);
  }
  if (BD.writeProgramToFile(TestModuleBC, TestModule)) {
    std::cerr << "Error writing bytecode to `" << SafeModuleBC << "'\nExiting.";
    exit(1);
  }

  delete SafeModule;
  delete TestModule;

  // Run the code generator on the `Test' code, loading the shared library.
  // The function returns whether or not the new output differs from reference.
  int Result =  BD.diffProgram(TestModuleBC, SharedObject, false);
  if (KeepFiles) {
    std::cout << "You can reproduce the problem with the command line: \n"
              << (BD.isExecutingJIT() ? "lli" : "llc")
              << " -load " << SharedObject << " " << TestModuleBC;
    for (unsigned i=0, e = InputArgv.size(); i != e; ++i)
      std::cout << " " << InputArgv[i];
    std::cout << "\n";
    std::cout << "The shared object " << SharedObject << " was created from "
              << SafeModuleBC << ", using `dis -c'.\n";
  } else {
    removeFile(TestModuleBC);
    removeFile(SafeModuleBC);
    removeFile(SharedObject);
  }
  return Result;
}

namespace {
  struct Disambiguator {
    std::set<std::string>  SymbolNames;
    std::set<GlobalValue*> Symbols;
    uint64_t uniqueCounter;
    bool externalOnly;
  public:
    Disambiguator() : uniqueCounter(0), externalOnly(true) {}
    void setExternalOnly(bool value) { externalOnly = value; }
    void add(GlobalValue &V) {
      // If we're only processing externals and this isn't external, bail
      if (externalOnly && !V.isExternal()) return;
      // If we're already processed this symbol, don't add it again
      if (Symbols.count(&V) != 0) return;
      // Ignore intrinsic functions
      if (Function *F = dyn_cast<Function>(&V))
        if (F->getIntrinsicID() != 0)
          return;

      std::string SymName = V.getName();

      // Use the Mangler facility to make symbol names that will be valid in
      // shared objects.
      SymName = Mangler::makeNameProper(SymName);
      V.setName(SymName);

      if (SymbolNames.count(SymName) == 0) {
        DEBUG(std::cerr << "Disambiguator: adding " << SymName
                        << ", no conflicts.\n");
        SymbolNames.insert(SymName);
      } else { 
        // Mangle name before adding
        std::string newName;
        do {
          newName = SymName + "_" + utostr(uniqueCounter);
          if (SymbolNames.count(newName) == 0) break;
          else ++uniqueCounter;
        } while (1);
        //while (SymbolNames.count(V->getName()+utostr(uniqueCounter++))==0);
        DEBUG(std::cerr << "Disambiguator: conflict: " << SymName
                        << ", adding: " << newName << "\n");
        V.setName(newName);
        SymbolNames.insert(newName);
      }
      Symbols.insert(&V);
    }
  };
}

void DisambiguateGlobalSymbols(Module *M) {
  // First, try not to cause collisions by minimizing chances of renaming an
  // already-external symbol, so take in external globals and functions as-is.
  Disambiguator D;
  DEBUG(std::cerr << "Disambiguating globals (external-only)\n");
  for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I) D.add(*I);
  DEBUG(std::cerr << "Disambiguating functions (external-only)\n");
  for (Module::iterator  I = M->begin(),  E = M->end();  I != E; ++I) D.add(*I);

  // Now just rename functions and globals as necessary, keeping what's already
  // in the set unique.
  D.setExternalOnly(false);
  DEBUG(std::cerr << "Disambiguating globals\n");
  for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I) D.add(*I);
  DEBUG(std::cerr << "Disambiguating globals\n");
  for (Module::iterator  I = M->begin(),  E = M->end();  I != E; ++I) D.add(*I);
}


bool BugDriver::debugCodeGenerator() {
  // See if we can pin down which functions are being miscompiled...
  //First, build a list of all of the non-external functions in the program.
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
    oldMain->setName("old_main");
    // Create a NEW `main' function with same type
    Function *newMain = new Function(oldMain->getFunctionType(), 
                                     GlobalValue::ExternalLinkage,
                                     "main", Program);
    // Call the old main function and return its result
    BasicBlock *BB = new BasicBlock("entry", newMain);
    std::vector<Value*> args;
    for (Function::aiterator I=newMain->abegin(), E=newMain->aend(); I!=E; ++I)
      args.push_back(I);
    CallInst *call = new CallInst(oldMain, args);
    BB->getInstList().push_back(call);
    
    // if the type of old function wasn't void, return value of call
    ReturnInst *ret;
    if (oldMain->getReturnType() != Type::VoidTy) {
      ret = new ReturnInst(call);
    } else {
      ret = new ReturnInst();
    }

    // Add the return instruction to the BasicBlock
    BB->getInstList().push_back(ret);
  }

  DisambiguateGlobalSymbols(Program);

  // Do the reduction...
  if (!ReduceMisCodegenFunctions(*this).reduceList(MisCodegenFunctions)) {
    std::cerr << "*** Execution matches reference output!  No problem "
	      << "detected...\nbugpoint can't help you with your problem!\n";
    return false;
  }

  std::cout << "\n*** The following functions are being miscompiled: ";
  PrintFunctionList(MisCodegenFunctions);
  std::cout << "\n";

  // Output a bunch of bytecode files for the user...
  ReduceMisCodegenFunctions(*this).TestFuncs(MisCodegenFunctions, true);

  return false;
}
