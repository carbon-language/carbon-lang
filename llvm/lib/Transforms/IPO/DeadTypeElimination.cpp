//===- CleanupGCCOutput.cpp - Cleanup GCC Output --------------------------===//
//
// This pass is used to cleanup the output of GCC.  GCC's output is
// unneccessarily gross for a couple of reasons. This pass does the following
// things to try to clean it up:
//
// * Eliminate names for GCC types that we know can't be needed by the user.
// * Eliminate names for types that are unused in the entire translation unit
// * Fix various problems that we might have in PHI nodes and casts
//
// Note:  This code produces dead declarations, it is a good idea to run DCE
//        after this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CleanupGCCOutput.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iPHINode.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "Support/StatisticReporter.h"
#include <algorithm>
#include <iostream>

static Statistic<> NumTypeSymtabEntriesKilled("cleangcc\t- Number of unused typenames removed from symtab");
static Statistic<> NumCastsMoved("cleangcc\t- Number of casts removed from head of basic block");
static Statistic<> NumRefactoredPreds("cleangcc\t- Number of predecessor blocks refactored");

using std::vector;

namespace {
  struct CleanupGCCOutput : public FunctionPass {
    const char *getPassName() const { return "Cleanup GCC Output"; }

    // doPassInitialization - For this pass, it removes global symbol table
    // entries for primitive types.  These are never used for linking in GCC and
    // they make the output uglier to look at, so we nuke them.
    //
    // Also, initialize instance variables.
    //
    bool doInitialization(Module *M);
    
    // runOnFunction - This method simplifies the specified function hopefully.
    //
    bool runOnFunction(Function *F);
    
    // doPassFinalization - Strip out type names that are unused by the program
    bool doFinalization(Module *M);
    
    // getAnalysisUsage - This function needs FindUsedTypes to do its job...
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired(FindUsedTypes::ID);
    }
  };
}

Pass *createCleanupGCCOutputPass() {
  return new CleanupGCCOutput();
}



// ShouldNukSymtabEntry - Return true if this module level symbol table entry
// should be eliminated.
//
static inline bool ShouldNukeSymtabEntry(const std::pair<std::string,Value*>&E){
  // Nuke all names for primitive types!
  if (cast<Type>(E.second)->isPrimitiveType()) return true;

  // Nuke all pointers to primitive types as well...
  if (const PointerType *PT = dyn_cast<PointerType>(E.second))
    if (PT->getElementType()->isPrimitiveType()) return true;

  // The only types that could contain .'s in the program are things generated
  // by GCC itself, including "complex.float" and friends.  Nuke them too.
  if (E.first.find('.') != std::string::npos) return true;

  return false;
}

// doInitialization - For this pass, it removes global symbol table
// entries for primitive types.  These are never used for linking in GCC and
// they make the output uglier to look at, so we nuke them.
//
bool CleanupGCCOutput::doInitialization(Module *M) {
  bool Changed = false;

  if (M->hasSymbolTable()) {
    SymbolTable *ST = M->getSymbolTable();

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
          ++NumTypeSymtabEntriesKilled;
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
      Changed = true;

      ++NumCastsMoved;

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
        Changed = true;
      }
    } else {
      ++I;
    }
  }

  return Changed;
}

// RefactorPredecessor - When we find out that a basic block is a repeated
// predecessor in a PHI node, we have to refactor the function until there is at
// most a single instance of a basic block in any predecessor list.
//
static inline void RefactorPredecessor(BasicBlock *BB, BasicBlock *Pred) {
  Function *M = BB->getParent();
  assert(find(pred_begin(BB), pred_end(BB), Pred) != pred_end(BB) &&
         "Pred is not a predecessor of BB!");

  // Create a new basic block, adding it to the end of the function.
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


// runOnFunction - Loop through the function and fix problems with the PHI nodes
// in the current function.  The problem is that PHI nodes might exist with
// multiple entries for the same predecessor.  GCC sometimes generates code that
// looks like this:
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
bool CleanupGCCOutput::runOnFunction(Function *M) {
  bool Changed = false;
  // Don't use iterators because invalidation gets messy...
  for (unsigned MI = 0; MI < M->size(); ++MI) {
    BasicBlock *BB = M->getBasicBlocks()[MI];

    Changed |= FixCastsAndPHIs(BB);

    if (isa<PHINode>(BB->front())) {
      const vector<BasicBlock*> Preds(pred_begin(BB), pred_end(BB));

      // Handle the problem.  Sort the list of predecessors so that it is easy
      // to decide whether or not duplicate predecessors exist.
      vector<BasicBlock*> SortedPreds(Preds);
      sort(SortedPreds.begin(), SortedPreds.end());

      // Loop over the predecessors, looking for adjacent BB's that are equal.
      BasicBlock *LastOne = 0;
      for (unsigned i = 0; i < Preds.size(); ++i) {
        if (SortedPreds[i] == LastOne) {   // Found a duplicate.
          RefactorPredecessor(BB, SortedPreds[i]);
          ++NumRefactoredPreds;
          Changed = true;
        }
        LastOne = SortedPreds[i];
      }
    }
  }
  return Changed;
}

bool CleanupGCCOutput::doFinalization(Module *M) {
  bool Changed = false;

  if (M->hasSymbolTable()) {
    SymbolTable *ST = M->getSymbolTable();
    const std::set<const Type *> &UsedTypes =
      getAnalysis<FindUsedTypes>().getTypes();

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
