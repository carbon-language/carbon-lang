//===- FindUsedTypes.h - Find all Types used by a module --------------------=//
//
// This pass is used to seek out all of the types in use by the program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Assembly/CachedWriter.h"
#include "llvm/SymbolTable.h"
#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"

// IncorporateType - Incorporate one type and all of its subtypes into the
// collection of used types.
//
void FindUsedTypes::IncorporateType(const Type *Ty) {
  if (UsedTypes.count(Ty)) return;  // Already contain Ty.
                             
  // If ty doesn't already exist in the used types map, add it now.
  //
  UsedTypes.insert(Ty);
  
  // Make sure to add any types this type references now.
  //
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I)
    IncorporateType(*I);
}

// IncorporateSymbolTable - Add all types referenced by the specified symtab
// into the collection of used types.
//
void FindUsedTypes::IncorporateSymbolTable(const SymbolTable *ST) {
  assert(0 && "Unimp");
}


// doPassInitialization - This loops over global constants defined in the
// module, converting them to their new type.
//
bool FindUsedTypes::doPassInitialization(Module *m) {
  const Module *M = m;
  if (IncludeSymbolTables && M->hasSymbolTable())
    IncorporateSymbolTable(M->getSymbolTable()); // Add symtab first...

  // Loop over global variables, incorporating their types
  for (Module::const_giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
    IncorporateType((*I)->getType());
  return false;
}

// doPerMethodWork - This incorporates all types used by the specified method
//
bool FindUsedTypes::doPerMethodWork(Method *m) {
  const Method *M = m;
  if (IncludeSymbolTables && M->hasSymbolTable())
  IncorporateSymbolTable(M->getSymbolTable()); // Add symtab first...
  
  // Loop over all of the instructions in the method, adding their return type
  // as well as the types of their operands.
  //
  for (Method::inst_const_iterator II = M->inst_begin(), IE = M->inst_end();
       II != IE; ++II) {
    const Instruction *I = *II;
    const Type *Ty = I->getType();
    
    IncorporateType(Ty);  // Incorporate the type of the instruction
    for (User::op_const_iterator OI = I->op_begin(), OE = I->op_end();
         OI != OE; ++OI)
      if ((*OI)->getType() != Ty)          // Avoid set lookup in common case
        IncorporateType((*OI)->getType()); // Insert inst operand types as well
  }
  
  return false;
}

// Print the types found in the module.  If the optional Module parameter is
// passed in, then the types are printed symbolically if possible, using the
// symbol table from the module.
//
void FindUsedTypes::printTypes(ostream &o, const Module *M = 0) const {
  o << "Types in use by this module:\n";
  if (M) {
    CachedWriter CW(M, o);
    for (set<const Type *>::const_iterator I = UsedTypes.begin(),
           E = UsedTypes.end(); I != E; ++I)
      CW << "  " << *I << endl;
  } else
    for (set<const Type *>::const_iterator I = UsedTypes.begin(),
           E = UsedTypes.end(); I != E; ++I)
      o << "  " << *I << endl;
}
