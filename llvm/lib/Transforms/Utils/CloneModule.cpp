//===- CloneModule.cpp - Clone an entire module ---------------------------===//
//
// This file implements the CloneModule interface which makes a copy of an
// entire module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/SymbolTable.h"
#include "llvm/Constant.h"
#include "ValueMapper.h"

/// CloneModule - Return an exact copy of the specified module.  This is not as
/// easy as it might seem because we have to worry about making copies of global
/// variables and functions, and making their (intializers and references,
/// respectively) refer to the right globals.
///
Module *CloneModule(const Module *M) {
  // First off, we need to create the new module...
  Module *New = new Module(M->getModuleIdentifier());
  New->setEndianness(M->getEndianness());
  New->setPointerSize(M->getPointerSize());

  // Copy all of the type symbol table entries over...
  const SymbolTable &SymTab = M->getSymbolTable();
  SymbolTable::const_iterator TypeI = SymTab.find(Type::TypeTy);
  if (TypeI != SymTab.end())
    for (SymbolTable::VarMap::const_iterator I = TypeI->second.begin(),
           E = TypeI->second.end(); I != E; ++I)
      New->addTypeName(I->first, cast<Type>(I->second));

  // Create the value map that maps things from the old module over to the new
  // module.
  std::map<const Value*, Value*> ValueMap;

  // Loop over all of the global variables, making corresponding globals in the
  // new module.  Here we add them to the ValueMap and to the new Module.  We
  // don't worry about attributes or initializers, they will come later.
  //
  for (Module::const_giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
    ValueMap[I] = new GlobalVariable(I->getType()->getElementType(), false,
                                     GlobalValue::ExternalLinkage, 0,
                                     I->getName(), New);

  // Loop over the functions in the module, making external functions as before
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    ValueMap[I]=new Function(cast<FunctionType>(I->getType()->getElementType()),
                             GlobalValue::ExternalLinkage, I->getName(), New);

  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  //
  for (Module::const_giterator I = M->gbegin(), E = M->gend(); I != E; ++I) {
    GlobalVariable *GV = cast<GlobalVariable>(ValueMap[I]);
    if (I->hasInitializer())
      GV->setInitializer(cast<Constant>(MapValue(I->getInitializer(),
                                                 ValueMap)));
    GV->setLinkage(I->getLinkage());
  }

  // Similarly, copy over function bodies now...
  //
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    Function *F = cast<Function>(ValueMap[I]);
    if (!I->isExternal()) {
      Function::aiterator DestI = F->abegin();
      for (Function::const_aiterator J = I->abegin(); J != I->aend(); ++J) {
        DestI->setName(J->getName());
        ValueMap[J] = DestI++;
      }

      std::vector<ReturnInst*> Returns;  // Ignore returns cloned...
      CloneFunctionInto(F, I, ValueMap, Returns);
    }

    F->setLinkage(I->getLinkage());
  }

  return New;
}
