//===-- Internalize.cpp - Mark functions internal -------------------------===//
//
// This pass loops over all of the functions in the input module, looking for a
// main function.  If a main function is found, all other functions and all
// global variables with initializers are marked as internal.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "Support/Statistic.h"
#include "Support/CommandLine.h"
#include <fstream>
#include <set>

namespace {
  Statistic<> NumFunctions("internalize", "Number of functions internalized");
  Statistic<> NumGlobals  ("internalize", "Number of global vars internalized");

  // APIFile - A file which contains a list of symbols that should not be marked
  // external.
  cl::opt<std::string>
  APIFile("internalize-public-api-file", cl::value_desc("filename"),
          cl::desc("A file containing list of globals to not internalize"));
  
  class InternalizePass : public Pass {
    std::set<std::string> ExternalNames;
  public:
    InternalizePass() {
      if (!APIFile.empty())
        LoadFile(APIFile.c_str());
      else
        ExternalNames.insert("main");
    }

    void LoadFile(const char *Filename) {
      // Load the APIFile...
      std::ifstream In(Filename);
      if (!In.good()) {
        std::cerr << "WARNING: Internalize couldn't load file '" << Filename
                  << "'!: Not internalizing.\n";
        return;   // Do not internalize anything...
      }
      while (In) {
        std::string Symbol;
        In >> Symbol;
        if (!Symbol.empty())
          ExternalNames.insert(Symbol);
      }
    }

    virtual bool run(Module &M) {
      if (ExternalNames.empty()) return false;  // Error loading file...
      bool Changed = false;
      
      // Found a main function, mark all functions not named main as internal.
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
        if (!I->isExternal() &&         // Function must be defined here
            !I->hasInternalLinkage() &&  // Can't already have internal linkage
            !ExternalNames.count(I->getName())) {// Not marked to keep external?
          I->setLinkage(GlobalValue::InternalLinkage);
          Changed = true;
          ++NumFunctions;
          DEBUG(std::cerr << "Internalizing func " << I->getName() << "\n");
        }

      // Mark all global variables with initializers as internal as well...
      for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
        if (!I->isExternal() && !I->hasInternalLinkage() &&
            !ExternalNames.count(I->getName())) {
          I->setLinkage(GlobalValue::InternalLinkage);
          Changed = true;
          ++NumGlobals;
          DEBUG(std::cerr << "Internalizing gvar " << I->getName() << "\n");
        }
      
      return Changed;
    }
  };

  RegisterOpt<InternalizePass> X("internalize", "Internalize Global Symbols");
} // end anonymous namespace

Pass *createInternalizePass() {
  return new InternalizePass();
}
