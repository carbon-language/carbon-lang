//===- CleanupGCCOutput.cpp - Cleanup GCC Output ----------------------------=//
//
// This pass is used to cleanup the output of GCC.  GCC's output is
// unneccessarily gross for a couple of reasons. This pass does the following
// things to try to clean it up:
//
// * Eliminate names for GCC types that we know can't be needed by the user.
// - Eliminate names for types that are unused in the entire translation unit
//    but only if they do not name a structure type!
// - Replace calls to 'sbyte *%malloc(uint)' and 'void %free(sbyte *)' with
//   malloc and free instructions.
//
// Note:  This code produces dead declarations, it is a good idea to run DCE
//        after this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CleanupGCCOutput.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"

static const Type *PtrArrSByte = 0; // '[sbyte]*' type
static const Type *PtrSByte = 0;    // 'sbyte*' type


static inline bool ShouldNukeSymtabEntry(const pair<string, Value*> &E) {
  // Nuke all names for primitive types!
  if (cast<Type>(E.second)->isPrimitiveType()) return true;

  // The only types that could contain .'s in the program are things generated
  // by GCC itself, including "complex.float" and friends.  Nuke them too.
  if (E.first.find('.') != string::npos) return true;

  return false;
}


// doPassInitialization - For this pass, it removes global symbol table
// entries for primitive types.  These are never used for linking in GCC and
// they make the output uglier to look at, so we nuke them.
//
bool CleanupGCCOutput::doPassInitialization(Module *M) {
  bool Changed = false;

  if (PtrArrSByte == 0) {
    PtrArrSByte = PointerType::get(ArrayType::get(Type::SByteTy));
    PtrSByte    = PointerType::get(Type::SByteTy);
  }

  if (M->hasSymbolTable()) {
    SymbolTable *ST = M->getSymbolTable();

    // Lookup %malloc and %free in the symbol table, for later use.  If they
    // don't exist, or are not external, we do not worry about converting calls
    // to that function into the appropriate instruction.
    //
    const PointerType *MallocType =   // Get the type for malloc
      PointerType::get(MethodType::get(PointerType::get(Type::SByteTy),
                                  vector<const Type*>(1, Type::UIntTy), false));
    Malloc = cast_or_null<Method>(ST->lookup(MallocType, "malloc"));
    if (Malloc && !Malloc->isExternal())
      Malloc = 0;  // Don't mess with locally defined versions of the fn

    const PointerType *FreeType =     // Get the type for free
      PointerType::get(MethodType::get(Type::VoidTy,
               vector<const Type*>(1, PointerType::get(Type::SByteTy)), false));
    Free = cast_or_null<Method>(ST->lookup(FreeType, "free"));
    if (Free && !Free->isExternal())
      Free = 0;  // Don't mess with locally defined versions of the fn
    

    // Check the symbol table for superfluous type entries...
    //
    // Grab the 'type' plane of the module symbol...
    SymbolTable::iterator STI = ST->find(Type::TypeTy);
    if (STI != ST->end()) {
      // Loop over all entries in the type plane...
      SymbolTable::VarMap &Plane = STI->second;
      for (SymbolTable::VarMap::iterator PI = Plane.begin(); PI != Plane.end();)
        if (ShouldNukeSymtabEntry(*PI)) {    // Should we remove this entry?
#if MAP_IS_NOT_BRAINDEAD
          PI = Plane.erase(PI);     // STD C++ Map should support this!
#else
          Plane.erase(PI);          // Alas, GCC 2.95.3 doesn't  *SIGH*
          PI = Plane.begin();
#endif
          Changed = true;
        } else {
          ++PI;
        }
    }
  }

  return Changed;
}

// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
// with a value, then remove and delete the original instruction.
//
static void ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                                 BasicBlock::iterator &BI, Value *V) {
  Instruction *I = *BI;
  // Replaces all of the uses of the instruction with uses of the value
  I->replaceAllUsesWith(V);

  // Remove the unneccesary instruction now...
  BIL.remove(BI);

  // Make sure to propogate a name if there is one already...
  if (I->hasName() && !V->hasName())
    V->setName(I->getName(), BIL.getParent()->getSymbolTable());

  // Remove the dead instruction now...
  delete I;
}


// ReplaceInstWithInst - Replace the instruction specified by BI with the
// instruction specified by I.  The original instruction is deleted and BI is
// updated to point to the new instruction.
//
static void ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                                BasicBlock::iterator &BI, Instruction *I) {
  assert(I->getParent() == 0 &&
         "ReplaceInstWithInst: Instruction already inserted into basic block!");

  // Insert the new instruction into the basic block...
  BI = BIL.insert(BI, I)+1;

  // Replace all uses of the old instruction, and delete it.
  ReplaceInstWithValue(BIL, BI, I);

  // Reexamine the instruction just inserted next time around the cleanup pass
  // loop.
  --BI;
}


// doOneCleanupPass - Do one pass over the input method, fixing stuff up.
//
bool CleanupGCCOutput::doOneCleanupPass(Method *M) {
  bool Changed = false;
  for (Method::iterator MI = M->begin(), ME = M->end(); MI != ME; ++MI) {
    BasicBlock *BB = *MI;
    BasicBlock::InstListType &BIL = BB->getInstList();

    for (BasicBlock::iterator BI = BB->begin(); BI != BB->end();) {
      Instruction *I = *BI;

      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        if (CI->getCalledValue() == Malloc) {      // Replace call to malloc?
          MallocInst *MallocI = new MallocInst(PtrArrSByte, CI->getOperand(1));
          BI = BIL.insert(BI, MallocI)+1;
          ReplaceInstWithInst(BIL, BI, new CastInst(MallocI, PtrSByte));
          Changed = true;
          continue;  // Skip the ++BI
        } else if (CI->getCalledValue() == Free) { // Replace call to free?
          ReplaceInstWithInst(BIL, BI, new FreeInst(CI->getOperand(1)));
          Changed = true;
          continue;  // Skip the ++BI
        }
      }

      ++BI;
    }
  }

  return Changed;
}




// doPerMethodWork - This method simplifies the specified method hopefully.
//
bool CleanupGCCOutput::doPerMethodWork(Method *M) {
  bool Changed = false;
  while (doOneCleanupPass(M)) Changed = true;
  return Changed;
}
