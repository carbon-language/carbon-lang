//===- FindUnsafePointerTypes.cpp - Check pointer usage safety ------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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
#include "llvm/Module.h"
#include "llvm/Support/InstIterator.h"
#include "Support/CommandLine.h"

namespace llvm {

static RegisterAnalysis<FindUnsafePointerTypes>
X("unsafepointertypes", "Find Unsafe Pointer Types");

// Provide a command line option to turn on printing of which instructions cause
// a type to become invalid
//
static cl::opt<bool> 
PrintFailures("printunsafeptrinst", cl::Hidden,
              cl::desc("Print Unsafe Pointer Access Instructions"));

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
  case Instruction::PHI:
    return true;
  }
  return false;
}


bool FindUnsafePointerTypes::run(Module &Mod) {
  for (Module::iterator FI = Mod.begin(), E = Mod.end();
       FI != E; ++FI) {
    const Function *F = FI;  // We don't need/want write access
    for (const_inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      const Type *ITy = I->getType();
      if (isa<PointerType>(ITy) && !UnsafeTypes.count((PointerType*)ITy))
        if (!isSafeInstruction(*I)) {
          UnsafeTypes.insert((PointerType*)ITy);

          if (PrintFailures) {
            CachedWriter CW(F->getParent(), std::cerr);
            CW << "FindUnsafePointerTypes: Type '" << ITy
               << "' marked unsafe in '" << F->getName() << "' by:\n" << **I;
          }
        }
    }
  }

  return false;
}


// printResults - Loop over the results of the analysis, printing out unsafe
// types.
//
void FindUnsafePointerTypes::print(std::ostream &o, const Module *M) const {
  if (UnsafeTypes.empty()) {
    o << "SafePointerAccess Analysis: No unsafe types found!\n";
    return;
  }

  CachedWriter CW(M, o);

  CW << "SafePointerAccess Analysis: Found these unsafe types:\n";
  unsigned Counter = 1;
  for (std::set<PointerType*>::const_iterator I = getUnsafeTypes().begin(), 
         E = getUnsafeTypes().end(); I != E; ++I, ++Counter) {
    
    CW << " #" << Counter << ". " << (Value*)*I << "\n";
  }
}

} // End llvm namespace
