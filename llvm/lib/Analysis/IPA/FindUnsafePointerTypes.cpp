//===- SafePointerAccess.cpp - Check pointer usage safety -------------------=//
//
// This file defines a pass that can be used to determine, interprocedurally, 
// which pointer types are accessed unsafely in a program.  If there is an
// "unsafe" access to a specific pointer type, transformations that depend on
// type safety cannot be permitted.
//
// The result of running this analysis over a program is a set of unsafe pointer
// types that cannot be transformed.  Safe pointer types are not tracked.
//
// Additionally, this analysis exports a hidden command line argument that (when
// enabled) prints out the reasons a type was determined to be unsafe.
//
// Currently, the only allowed operations on pointer types are:
//   alloca, malloc, free, getelementptr, load, and store
// 
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FindUnsafePointerTypes.h"
#include "llvm/Assembly/CachedWriter.h"
#include "llvm/Type.h"
#include "Support/CommandLine.h"

// Provide a command line option to turn on printing of which instructions cause
// a type to become invalid
//
static cl::Flag 
PrintFailures("printunsafeptrinst", "Print Unsafe Pointer Access Instructions",
              cl::Hidden, false);

static inline bool isSafeInstruction(const Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Alloca:
  case Instruction::Malloc:
  case Instruction::Free:
  case Instruction::Load:
  case Instruction::Store:
  case Instruction::GetElementPtr:
  case Instruction::Call:
  case Instruction::Invoke:
  case Instruction::PHINode:
    return true;
  }
  return false;
}


// doPerMethodWork - Inspect the operations that the specified method does on
// values of various types.  If they are deemed to be 'unsafe' note that the
// type is not safe to transform.
//
bool FindUnsafePointerTypes::doPerMethodWork(Method *Meth) {
  const Method *M = Meth;  // We don't need/want write access
  for (Method::const_inst_iterator I = M->inst_begin(), E = M->inst_end();
       I != E; ++I) {
    const Instruction *Inst = *I;
    const Type *ITy = Inst->getType();
    if (ITy->isPointerType() && !UnsafeTypes.count((PointerType*)ITy))
      if (!isSafeInstruction(Inst)) {
        UnsafeTypes.insert((PointerType*)ITy);

        if (PrintFailures) {
          CachedWriter CW(M->getParent(), cerr);
          CW << "FindUnsafePointerTypes: Type '" << ITy
             << "' marked unsafe in '" << Meth->getName() << "' by:\n" << Inst;
        }
      }
  }

  return false;
}


// printResults - Loop over the results of the analysis, printing out unsafe
// types.
//
void FindUnsafePointerTypes::printResults(const Module *M, ostream &o) {
  if (UnsafeTypes.empty()) {
    o << "SafePointerAccess Analysis: No unsafe types found!\n";
    return;
  }

  CachedWriter CW(M, o);

  CW << "SafePointerAccess Analysis: Found these unsafe types:\n";
  unsigned Counter = 1;
  for (set<PointerType*>::const_iterator I = getUnsafeTypes().begin(), 
         E = getUnsafeTypes().end(); I != E; ++I, ++Counter) {
    
    CW << " #" << Counter << ". " << (Value*)*I << endl;
  }
}
