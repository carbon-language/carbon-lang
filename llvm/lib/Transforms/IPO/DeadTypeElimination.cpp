//===- CleanupGCCOutput.cpp - Cleanup GCC Output ----------------------------=//
//
// This pass is used to cleanup the output of GCC.  GCC's output is
// unneccessarily gross for a couple of reasons. This pass does the following
// things to try to clean it up:
//
// * Eliminate names for GCC types that we know can't be needed by the user.
// * Eliminate names for types that are unused in the entire translation unit
//
// Note:  This code produces dead declarations, it is a good idea to run DCE
//        after this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CleanupGCCOutput.h"
#include "TransformInternals.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iPHINode.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include <algorithm>
#include <iostream>
using std::vector;
using std::string;
using std::cerr;

static const Type *PtrSByte = 0;    // 'sbyte*' type

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
bool CleanupGCCOutput::PatchUpMethodReferences(Module *M) {
  SymbolTable *ST = M->getSymbolTable();
  if (!ST) return false;

  std::map<string, vector<Method*> > Methods;

  // Loop over the entries in the symbol table. If an entry is a method pointer,
  // then add it to the Methods map.  We do a two pass algorithm here to avoid
  // problems with iterators getting invalidated if we did a one pass scheme.
  //
  for (SymbolTable::iterator I = ST->begin(), E = ST->end(); I != E; ++I)
    if (const PointerType *PT = dyn_cast<PointerType>(I->first))
      if (isa<MethodType>(PT->getElementType())) {
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
  for (std::map<string, vector<Method*> >::iterator I = Methods.begin(), 
         E = Methods.end(); I != E; ++I) {
    vector<Method*> &Methods = I->second;
    Method *Implementation = 0;     // Find the implementation
    Method *Concrete = 0;
    for (unsigned i = 0; i < Methods.size(); ) {
      if (!Methods[i]->isExternal()) {  // Found an implementation
        assert(Implementation == 0 && "Multiple definitions of the same"
               " method. Case not handled yet!");
        Implementation = Methods[i];
      } else {
        // Ignore methods that are never used so they don't cause spurious
        // warnings... here we will actually DCE the function so that it isn't
        // used later.
        //
        if (Methods[i]->use_size() == 0) {
          M->getMethodList().remove(Methods[i]);
          delete Methods[i];
          Methods.erase(Methods.begin()+i);
          Changed = true;
          continue;
        }
      }
      
      if (Methods[i] && (!Methods[i]->getMethodType()->isVarArg() ||
                         Methods[i]->getMethodType()->getParamTypes().size())) {
        if (Concrete) {  // Found two different methods types.  Can't choose
          Concrete = 0;
          break;
        }
        Concrete = Methods[i];
      }
      ++i;
    }

    if (Methods.size() > 1) {         // Found a multiply defined method.
      // We should find exactly one non-vararg method definition, which is
      // probably the implementation.  Change all of the method definitions
      // and uses to use it instead.
      //
      if (!Concrete) {
        cerr << "Warning: Found methods types that are not compatible:\n";
        for (unsigned i = 0; i < Methods.size(); ++i) {
          cerr << "\t" << Methods[i]->getType()->getDescription() << " %"
               << Methods[i]->getName() << "\n";
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
                cerr << "Cannot convert use of method: " << U << "\n";
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
static inline bool ShouldNukeSymtabEntry(const std::pair<string, Value*> &E) {
  // Nuke all names for primitive types!
  if (cast<Type>(E.second)->isPrimitiveType()) return true;

  // Nuke all pointers to primitive types as well...
  if (const PointerType *PT = dyn_cast<PointerType>(E.second))
    if (PT->getElementType()->isPrimitiveType()) return true;

  // The only types that could contain .'s in the program are things generated
  // by GCC itself, including "complex.float" and friends.  Nuke them too.
  if (E.first.find('.') != string::npos) return true;

  return false;
}

// doInitialization - For this pass, it removes global symbol table
// entries for primitive types.  These are never used for linking in GCC and
// they make the output uglier to look at, so we nuke them.
//
bool CleanupGCCOutput::doInitialization(Module *M) {
  bool Changed = false;

  FUT.doInitialization(M);

  if (PtrSByte == 0)
    PtrSByte = PointerType::get(Type::SByteTy);

  if (M->hasSymbolTable()) {
    SymbolTable *ST = M->getSymbolTable();

    // Go over the methods that are in the module and look for methods that have
    // the same name.  More often than not, there will be things like:
    // void "foo"(...)  and void "foo"(int, int) because of the way things are
    // declared in C.  If this is the case, patch things up.
    //
    Changed |= PatchUpMethodReferences(M);

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


// FixCastsAndPHIs - The LLVM GCC has a tendancy to intermix Cast instructions
// in with the PHI nodes.  These cast instructions are potentially there for two
// different reasons:
//
//   1. The cast could be for an early PHI, and be accidentally inserted before
//      another PHI node.  In this case, the PHI node should be moved to the end
//      of the PHI nodes in the basic block.  We know that it is this case if
//      the source for the cast is a PHI node in this basic block.
//
//   2. If not #1, the cast must be a source argument for one of the PHI nodes
//      in the current basic block.  If this is the case, the cast should be
//      lifted into the basic block for the appropriate predecessor. 
//
static inline bool FixCastsAndPHIs(BasicBlock *BB) {
  bool Changed = false;

  BasicBlock::iterator InsertPos = BB->begin();

  // Find the end of the interesting instructions...
  while (isa<PHINode>(*InsertPos) || isa<CastInst>(*InsertPos)) ++InsertPos;

  // Back the InsertPos up to right after the last PHI node.
  while (InsertPos != BB->begin() && isa<CastInst>(*(InsertPos-1))) --InsertPos;

  // No PHI nodes, quick exit.
  if (InsertPos == BB->begin()) return false;

  // Loop over all casts trapped between the PHI's...
  BasicBlock::iterator I = BB->begin();
  while (I != InsertPos) {
    if (CastInst *CI = dyn_cast<CastInst>(*I)) { // Fix all cast instructions
      Value *Src = CI->getOperand(0);

      // Move the cast instruction to the current insert position...
      --InsertPos;                 // New position for cast to go...
      std::swap(*InsertPos, *I);   // Cast goes down, PHI goes up

      if (isa<PHINode>(Src) &&                                // Handle case #1
          cast<PHINode>(Src)->getParent() == BB) {
        // We're done for case #1
      } else {                                                // Handle case #2
        // In case #2, we have to do a few things:
        //   1. Remove the cast from the current basic block.
        //   2. Identify the PHI node that the cast is for.
        //   3. Find out which predecessor the value is for.
        //   4. Move the cast to the end of the basic block that it SHOULD be
        //

        // Remove the cast instruction from the basic block.  The remove only
        // invalidates iterators in the basic block that are AFTER the removed
        // element.  Because we just moved the CastInst to the InsertPos, no
        // iterators get invalidated.
        //
        BB->getInstList().remove(InsertPos);

        // Find the PHI node.  Since this cast was generated specifically for a
        // PHI node, there can only be a single PHI node using it.
        //
        assert(CI->use_size() == 1 && "Exactly one PHI node should use cast!");
        PHINode *PN = cast<PHINode>(*CI->use_begin());

        // Find out which operand of the PHI it is...
        unsigned i;
        for (i = 0; i < PN->getNumIncomingValues(); ++i)
          if (PN->getIncomingValue(i) == CI)
            break;
        assert(i != PN->getNumIncomingValues() && "PHI doesn't use cast!");

        // Get the predecessor the value is for...
        BasicBlock *Pred = PN->getIncomingBlock(i);

        // Reinsert the cast right before the terminator in Pred.
        Pred->getInstList().insert(Pred->end()-1, CI);
      }
    } else {
      ++I;
    }
  }


  return Changed;
}

// RefactorPredecessor - When we find out that a basic block is a repeated
// predecessor in a PHI node, we have to refactor the method until there is at
// most a single instance of a basic block in any predecessor list.
//
static inline void RefactorPredecessor(BasicBlock *BB, BasicBlock *Pred) {
  Method *M = BB->getParent();
  assert(find(BB->pred_begin(), BB->pred_end(), Pred) != BB->pred_end() &&
         "Pred is not a predecessor of BB!");

  // Create a new basic block, adding it to the end of the method.
  BasicBlock *NewBB = new BasicBlock("", M);

  // Add an unconditional branch to BB to the new block.
  NewBB->getInstList().push_back(new BranchInst(BB));

  // Get the terminator that causes a branch to BB from Pred.
  TerminatorInst *TI = Pred->getTerminator();

  // Find the first use of BB in the terminator...
  User::op_iterator OI = find(TI->op_begin(), TI->op_end(), BB);
  assert(OI != TI->op_end() && "Pred does not branch to BB!!!");

  // Change the use of BB to point to the new stub basic block
  *OI = NewBB;

  // Now we need to loop through all of the PHI nodes in BB and convert their
  // first incoming value for Pred to reference the new basic block instead.
  //
  for (BasicBlock::iterator I = BB->begin(); 
       PHINode *PN = dyn_cast<PHINode>(*I); ++I) {
    int BBIdx = PN->getBasicBlockIndex(Pred);
    assert(BBIdx != -1 && "PHI node doesn't have an entry for Pred!");

    // The value that used to look like it came from Pred now comes from NewBB
    PN->setIncomingBlock((unsigned)BBIdx, NewBB);
  }
}


// CheckIncomingValueFor - Make sure that the specified PHI node has an entry
// for the provided basic block.  If it doesn't, add one and return true.
//
static inline void CheckIncomingValueFor(PHINode *PN, BasicBlock *BB) {
  if (PN->getBasicBlockIndex(BB) != -1) return;  // Already has value

  Value      *NewVal = 0;
  const Type *Ty = PN->getType();

  if (const PointerType *PT = dyn_cast<PointerType>(Ty))
    NewVal = ConstantPointerNull::get(PT);
  else if (Ty == Type::BoolTy)
    NewVal = ConstantBool::True;
  else if (Ty == Type::FloatTy || Ty == Type::DoubleTy)
    NewVal = ConstantFP::get(Ty, 42);
  else if (Ty->isIntegral())
    NewVal = ConstantInt::get(Ty, 42);

  assert(NewVal && "Unknown PHI node type!");
  PN->addIncoming(NewVal, BB);
} 

// fixLocalProblems - Loop through the method and fix problems with the PHI
// nodes in the current method.  The two problems that are handled are:
//
//  1. PHI nodes with multiple entries for the same predecessor.  GCC sometimes
//     generates code that looks like this:
//
//  bb7:  br bool %cond1004, label %bb8, label %bb8
//  bb8: %reg119 = phi uint [ 0, %bb7 ], [ 1, %bb7 ]
//     
//     which is completely illegal LLVM code.  To compensate for this, we insert
//     an extra basic block, and convert the code to look like this:
//
//  bb7: br bool %cond1004, label %bbX, label %bb8
//  bbX: br label bb8
//  bb8: %reg119 = phi uint [ 0, %bbX ], [ 1, %bb7 ]
//
//
//  2. PHI nodes with fewer arguments than predecessors.
//     These can be generated by GCC if a variable is uninitalized over a path
//     in the CFG.  We fix this by adding an entry for the missing predecessors
//     that is initialized to either 42 for a numeric/FP value, or null if it's
//     a pointer value. This problem can be generated by code that looks like
//     this:
//         int foo(int y) {
//           int X;
//           if (y) X = 1;
//           return X;
//         }
//
static bool fixLocalProblems(Method *M) {
  bool Changed = false;
  // Don't use iterators because invalidation gets messy...
  for (unsigned MI = 0; MI < M->size(); ++MI) {
    BasicBlock *BB = M->getBasicBlocks()[MI];

    Changed |= FixCastsAndPHIs(BB);

    if (isa<PHINode>(BB->front())) {
      const vector<BasicBlock*> Preds(BB->pred_begin(), BB->pred_end());

      // Handle Problem #1.  Sort the list of predecessors so that it is easy to
      // decide whether or not duplicate predecessors exist.
      vector<BasicBlock*> SortedPreds(Preds);
      sort(SortedPreds.begin(), SortedPreds.end());

      // Loop over the predecessors, looking for adjacent BB's that are equal.
      BasicBlock *LastOne = 0;
      for (unsigned i = 0; i < Preds.size(); ++i) {
        if (SortedPreds[i] == LastOne) {   // Found a duplicate.
          RefactorPredecessor(BB, SortedPreds[i]);
          Changed = true;
        }
        LastOne = SortedPreds[i];
      }

      // Loop over all of the PHI nodes in the current BB.  These PHI nodes are
      // guaranteed to be at the beginning of the basic block.
      //
      for (BasicBlock::iterator I = BB->begin(); 
           PHINode *PN = dyn_cast<PHINode>(*I); ++I) {
        
        // Handle problem #2.
        if (PN->getNumIncomingValues() != Preds.size()) {
          assert(PN->getNumIncomingValues() <= Preds.size() &&
                 "Can't handle extra arguments to PHI nodes!");
          for (unsigned i = 0; i < Preds.size(); ++i)
            CheckIncomingValueFor(PN, Preds[i]);
          Changed = true;
        }
      }
    }
  }
  return Changed;
}




// doPerMethodWork - This method simplifies the specified method hopefully.
//
bool CleanupGCCOutput::runOnMethod(Method *M) {
  bool Changed = fixLocalProblems(M);

  FUT.runOnMethod(M);
  return Changed;
}

bool CleanupGCCOutput::doFinalization(Module *M) {
  bool Changed = false;
  FUT.doFinalization(M);

  if (M->hasSymbolTable()) {
    SymbolTable *ST = M->getSymbolTable();
    const std::set<const Type *> &UsedTypes = FUT.getTypes();

    // Check the symbol table for superfluous type entries that aren't used in
    // the program
    //
    // Grab the 'type' plane of the module symbol...
    SymbolTable::iterator STI = ST->find(Type::TypeTy);
    if (STI != ST->end()) {
      // Loop over all entries in the type plane...
      SymbolTable::VarMap &Plane = STI->second;
      for (SymbolTable::VarMap::iterator PI = Plane.begin(); PI != Plane.end();)
        if (!UsedTypes.count(cast<Type>(PI->second))) {
#if MAP_IS_NOT_BRAINDEAD
          PI = Plane.erase(PI);     // STD C++ Map should support this!
#else
          Plane.erase(PI);          // Alas, GCC 2.95.3 doesn't  *SIGH*
          PI = Plane.begin();       // N^2 algorithms are fun.  :(
#endif
          Changed = true;
        } else {
          ++PI;
        }
    }
  }
  return Changed;
}
