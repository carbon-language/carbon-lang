//===- CodeGeneratorBug.cpp - Debug code generation bugs ------------------===//
//
// This file implements program code generation debugging support.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "SystemUtils.h"
#include "ListReducer.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Linker.h"
#include "Support/CommandLine.h"
#include "Support/Statistic.h"
#include "Support/StringExtras.h"
#include <algorithm>
#include <set>

// Passed as a command-line argument to Bugpoint
extern cl::opt<std::string> Output;

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
  
  bool TestFuncs(const std::vector<Function*> &CodegenTest);

  void DisambiguateGlobalSymbols(Module *M);
};


bool ReduceMisCodegenFunctions::TestFuncs(const std::vector<Function*> &Funcs)
{
  // Clone the module for the two halves of the program we want.
  Module *SafeModule = CloneModule(BD.Program);

  // Make sure functions & globals are all external so that linkage
  // between the two modules will work.
  for (Module::iterator I = SafeModule->begin(), E = SafeModule->end();I!=E;++I)
    I->setLinkage(GlobalValue::ExternalLinkage);
  for (Module::giterator I=SafeModule->gbegin(),E = SafeModule->gend();I!=E;++I)
    I->setLinkage(GlobalValue::ExternalLinkage);

  DisambiguateGlobalSymbols(SafeModule);
  Module *TestModule = CloneModule(SafeModule);

  // Make sure global initializers exist only in the safe module (CBE->.so)
  for (Module::giterator I=TestModule->gbegin(),E = TestModule->gend();I!=E;++I)
    I->setInitializer(0);  // Delete the initializer to make it external

  // Remove the Test functions from the Safe module, and
  // all of the global variables.
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i) {
    Function *TNOF = SafeModule->getFunction(Funcs[i]->getName(),
                                             Funcs[i]->getFunctionType());
    assert(TNOF && "Function doesn't exist in module!");
    DeleteFunctionBody(TNOF);       // Function is now external in this module!
  }

  // Write out the bytecode to be sent to CBE
  std::string SafeModuleBC = "bugpoint.safe.bc";
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

  std::string TestModuleBC = "bugpoint.test.bc";
  if (BD.writeProgramToFile(TestModuleBC, TestModule)) {
    std::cerr << "Error writing bytecode to `" << SafeModuleBC << "'\nExiting.";
    exit(1);
  }

  // Run the code generator on the `Test' code, loading the shared library.
  // The function returns whether or not the new output differs from reference.
  return BD.diffProgram(TestModuleBC, SharedObject, false);
}

namespace {
  struct Disambiguator /*: public unary_function<GlobalValue&, void>*/ {
    std::set<std::string> SymbolNames;
    std::set<Value*> Symbols;
    uint64_t uniqueCounter;
    bool externalOnly;

    Disambiguator() : uniqueCounter(0), externalOnly(true) {}
    void setExternalOnly(bool value) { externalOnly = value; }
    void operator() (GlobalValue &V) {
      if (externalOnly && !V.isExternal()) return;

      if (SymbolNames.count(V.getName()) == 0) {
        DEBUG(std::cerr << "Disambiguator: adding " << V.getName() 
                        << ", no conflicts.\n");
        Symbols.insert(&V);
        SymbolNames.insert(V.getName());
      } else { 
        // Mangle name before adding
        std::string newName;
        do {
          newName = V.getName() + "_" + utostr(uniqueCounter);
          if (SymbolNames.count(newName) == 0) break;
          else ++uniqueCounter;
        } while (1);
        //while (SymbolNames.count(V->getName()+utostr(uniqueCounter++))==0);
        DEBUG(std::cerr << "Disambiguator: conflict: " << V.getName()
                        << ", adding: " << newName << "\n");
        V.setName(newName);
        SymbolNames.insert(newName);
        Symbols.insert(&V);
      }
    }
  };
}

void ReduceMisCodegenFunctions::DisambiguateGlobalSymbols(Module *M) {
  // First, try not to cause collisions by minimizing chances of renaming an
  // already-external symbol, so take in external globals and functions as-is.
  Disambiguator D = std::for_each(M->gbegin(), M->gend(), Disambiguator());
  std::for_each(M->begin(), M->end(), D);

  // Now just rename functions and globals as necessary, keeping what's already
  // in the set unique.
  D.setExternalOnly(false);
  std::for_each(M->gbegin(), M->gend(), D);
  std::for_each(M->begin(), M->end(), D);
}


bool BugDriver::debugCodeGenerator() {
  // See if we can pin down which functions are being miscompiled...
  //First, build a list of all of the non-external functions in the program.
  std::vector<Function*> MisCodegenFunctions;
  for (Module::iterator I = Program->begin(), E = Program->end(); I != E; ++I)
    if (!I->isExternal())
      MisCodegenFunctions.push_back(I);

  // Do the reduction...
  ReduceMisCodegenFunctions(*this).reduceList(MisCodegenFunctions);

  std::cout << "\n*** The following functions are being miscompiled: ";
  PrintFunctionList(MisCodegenFunctions);
  std::cout << "\n";

  // Output a bunch of bytecode files for the user...
  ReduceMisCodegenFunctions(*this).TestFuncs(MisCodegenFunctions);

  return false;
}

