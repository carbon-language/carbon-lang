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
#include <map>
#include <algorithm>

static const Type *PtrArrSByte = 0; // '[sbyte]*' type
static const Type *PtrSByte = 0;    // 'sbyte*' type


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



// ConvertCallTo - Convert a call to a varargs function with no arg types
// specified to a concrete nonvarargs method.
//
static void ConvertCallTo(CallInst *CI, Method *Dest) {
  const MethodType::ParamTypes &ParamTys =
    Dest->getMethodType()->getParamTypes();
  BasicBlock *BB = CI->getParent();

  // Get an iterator to where we want to insert cast instructions if the
  // argument types don't agree.
  //
  BasicBlock::iterator BBI = find(BB->begin(), BB->end(), CI);
  assert(BBI != BB->end() && "CallInst not in parent block?");

  assert(CI->getNumOperands()-1 == ParamTys.size()&&
         "Method calls resolved funny somehow, incompatible number of args");

  vector<Value*> Params;

  // Convert all of the call arguments over... inserting cast instructions if
  // the types are not compatible.
  for (unsigned i = 1; i < CI->getNumOperands(); ++i) {
    Value *V = CI->getOperand(i);

    if (V->getType() != ParamTys[i-1]) { // Must insert a cast...
      Instruction *Cast = new CastInst(V, ParamTys[i-1]);
      BBI = BB->getInstList().insert(BBI, Cast)+1;
      V = Cast;
    }

    Params.push_back(V);
  }

  // Replace the old call instruction with a new call instruction that calls
  // the real method.
  //
  ReplaceInstWithInst(BB->getInstList(), BBI, new CallInst(Dest, Params));
}


// PatchUpMethodReferences - Go over the methods that are in the module and
// look for methods that have the same name.  More often than not, there will
// be things like:
//    void "foo"(...)
//    void "foo"(int, int)
// because of the way things are declared in C.  If this is the case, patch
// things up.
//
static bool PatchUpMethodReferences(SymbolTable *ST) {
  map<string, vector<Method*> > Methods;

  // Loop over the entries in the symbol table. If an entry is a method pointer,
  // then add it to the Methods map.  We do a two pass algorithm here to avoid
  // problems with iterators getting invalidated if we did a one pass scheme.
  //
  for (SymbolTable::iterator I = ST->begin(), E = ST->end(); I != E; ++I)
    if (const PointerType *PT = dyn_cast<PointerType>(I->first))
      if (const MethodType *MT = dyn_cast<MethodType>(PT->getValueType())) {
        SymbolTable::VarMap &Plane = I->second;
        for (SymbolTable::type_iterator PI = Plane.begin(), PE = Plane.end();
             PI != PE; ++PI) {
          const string &Name = PI->first;
          Method *M = cast<Method>(PI->second);
          Methods[Name].push_back(M);          
        }
      }

  bool Changed = false;

  // Now we have a list of all methods with a particular name.  If there is more
  // than one entry in a list, merge the methods together.
  //
  for (map<string, vector<Method*> >::iterator I = Methods.begin(), 
         E = Methods.end(); I != E; ++I) {
    vector<Method*> &Methods = I->second;
    if (Methods.size() > 1) {         // Found a multiply defined method.
      Method *Implementation = 0;     // Find the implementation
      Method *Concrete = 0;
      for (unsigned i = 0; i < Methods.size(); ++i) {
        if (!Methods[i]->isExternal()) {  // Found an implementation
          assert(Implementation == 0 && "Multiple definitions of the same"
                 " method. Case not handled yet!");
          Implementation = Methods[i];
        }

        if (!Methods[i]->getMethodType()->isVarArg() ||
            Methods[i]->getMethodType()->getParamTypes().size()) {
          if (Concrete) {  // Found two different methods types.  Can't choose
            Concrete = 0;
            break;
          }
          Concrete = Methods[i];
        }
      }

      // We should find exactly one non-vararg method definition, which is
      // probably the implementation.  Change all of the method definitions
      // and uses to use it instead.
      //
      if (!Concrete) {
        cerr << "Warning: Found methods types that are not compatible:\n";
        for (unsigned i = 0; i < Methods.size(); ++i) {
          cerr << "\t" << Methods[i]->getType()->getDescription() << " %"
               << Methods[i]->getName() << endl;
        }
        cerr << "  No linkage of methods named '" << Methods[0]->getName()
             << "' performed!\n";
      } else {
        for (unsigned i = 0; i < Methods.size(); ++i)
          if (Methods[i] != Concrete) {
            Method *Old = Methods[i];
            assert(Old->getReturnType() == Concrete->getReturnType() &&
                   "Differing return types not handled yet!");
            assert(Old->getMethodType()->getParamTypes().size() == 0 &&
                   "Cannot handle varargs fn's with specified element types!");
            
            // Attempt to convert all of the uses of the old method to the
            // concrete form of the method.  If there is a use of the method
            // that we don't understand here we punt to avoid making a bad
            // transformation.
            //
            // At this point, we know that the return values are the same for
            // our two functions and that the Old method has no varargs methods
            // specified.  In otherwords it's just <retty> (...)
            //
            for (unsigned i = 0; i < Old->use_size(); ) {
              User *U = *(Old->use_begin()+i);
              if (CastInst *CI = dyn_cast<CastInst>(U)) {
                // Convert casts directly
                assert(CI->getOperand(0) == Old);
                CI->setOperand(0, Concrete);
                Changed = true;
              } else if (CallInst *CI = dyn_cast<CallInst>(U)) {
                // Can only fix up calls TO the argument, not args passed in.
                if (CI->getCalledValue() == Old) {
                  ConvertCallTo(CI, Concrete);
                  Changed = true;
                } else {
                  cerr << "Couldn't cleanup this function call, must be an"
                       << " argument or something!" << CI;
                  ++i;
                }
              } else {
                cerr << "Cannot convert use of method: " << U << endl;
                ++i;
              }
            }
          }
        }
    }
  }

  return Changed;
}


// ShouldNukSymtabEntry - Return true if this module level symbol table entry
// should be eliminated.
//
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

    // Go over the methods that are in the module and look for methods that have
    // the same name.  More often than not, there will be things like:
    // void "foo"(...)  and void "foo"(int, int) because of the way things are
    // declared in C.  If this is the case, patch things up.
    //
    Changed |= PatchUpMethodReferences(ST);


    // If the module has a symbol table, they might be referring to the malloc
    // and free functions.  If this is the case, grab the method pointers that 
    // the module is using.
    //
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
          MallocInst *MallocI = new MallocInst(PtrArrSByte, CI->getOperand(1),
                                               CI->getName());
          CI->setName("");
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
