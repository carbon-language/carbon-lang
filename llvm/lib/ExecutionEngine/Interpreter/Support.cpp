//===-- Support.cpp - Support routines for interpreter --------------------===//
// 
//  This file contains support routines for the interpreter core.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/SymbolTable.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Module.h"

//===----------------------------------------------------------------------===//
//
// LookupMatchingNames helper - Search a symbol table for values matching Name.
//
static inline void LookupMatchingNames(const std::string &Name,
                                       SymbolTable &SymTab,
				       std::vector<Value*> &Results) {
  // Loop over all of the type planes in the symbol table...
  for (SymbolTable::iterator I = SymTab.begin(), E = SymTab.end(); I != E; ++I){
    SymbolTable::VarMap &Plane = I->second;
    
    // Search the symbol table plane for this name...
    SymbolTable::VarMap::iterator Val = Plane.find(Name);
    if (Val != Plane.end())
      Results.push_back(Val->second);                    // Found a name match!
  }
}

// LookupMatchingNames - Search the current function namespace, then the global
// namespace looking for values that match the specified name.  Return ALL
// matches to that name.  This is obviously slow, and should only be used for
// user interaction.
//
std::vector<Value*> Interpreter::LookupMatchingNames(const std::string &Name) {
  std::vector<Value*> Results;
  Function *CurFunc = getCurrentFunction();
  
  if (CurFunc) ::LookupMatchingNames(Name, CurFunc->getSymbolTable(), Results);
  ::LookupMatchingNames(Name, getModule().getSymbolTable(), Results);
  return Results;
}

// ChooseOneOption - Prompt the user to choose among the specified options to
// pick one value.  If no options are provided, emit an error.  If a single 
// option is provided, just return that option.
//
Value *Interpreter::ChooseOneOption(const std::string &Name,
				    const std::vector<Value*> &Opts) {
  switch (Opts.size()) {
  case 1: return Opts[0];
  case 0: 
    std::cout << "Error: no entities named '" << Name << "' found!\n";
    return 0;
  default: break;  // Must prompt user...
  }

  std::cout << "Multiple entities named '" << Name
            << "' found!  Please choose:\n";
  std::cout << "  0. Cancel operation\n";
  for (unsigned i = 0; i < Opts.size(); ++i) {
    std::cout << "  " << (i+1) << ".";
    WriteAsOperand(std::cout, Opts[i]) << "\n";
  }

  unsigned Option;
  do {
    std::cout << "lli> " << std::flush;
    std::cin >> Option;
    if (Option > Opts.size())
      std::cout << "Invalid selection: Please choose from 0 to " << Opts.size()
                << "\n";
  } while (Option > Opts.size());

  if (Option == 0) return 0;
  return Opts[Option-1];
}
