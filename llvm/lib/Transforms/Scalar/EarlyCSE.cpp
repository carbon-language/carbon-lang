//===- EarlyCSE.cpp - Simple and fast CSE pass ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs a simple dominator tree walk that eliminates trivially
// redundant instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "early-cse"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumSimplify, "Number of insts simplified or DCE'd");
STATISTIC(NumCSE,      "Number of insts CSE'd");
STATISTIC(NumCSEMem,   "Number of load and call insts CSE'd");

static unsigned getHash(const void *V) {
  return DenseMapInfo<const void*>::getHashValue(V);
}

//===----------------------------------------------------------------------===//
// SimpleValue 
//===----------------------------------------------------------------------===//

namespace {
  /// SimpleValue - Instances of this struct represent available values in the
  /// scoped hash table.
  struct SimpleValue {
    Instruction *Inst;
    
    bool isSentinel() const {
      return Inst == DenseMapInfo<Instruction*>::getEmptyKey() ||
             Inst == DenseMapInfo<Instruction*>::getTombstoneKey();
    }
    
    static bool canHandle(Instruction *Inst) {
      return isa<CastInst>(Inst) || isa<BinaryOperator>(Inst) ||
             isa<GetElementPtrInst>(Inst) || isa<CmpInst>(Inst) ||
             isa<SelectInst>(Inst) || isa<ExtractElementInst>(Inst) ||
             isa<InsertElementInst>(Inst) || isa<ShuffleVectorInst>(Inst) ||
             isa<ExtractValueInst>(Inst) || isa<InsertValueInst>(Inst);
    }
    
    static SimpleValue get(Instruction *I) {
      SimpleValue X; X.Inst = I;
      assert((X.isSentinel() || canHandle(I)) && "Inst can't be handled!");
      return X;
    }
  };
}

namespace llvm {
// SimpleValue is POD.
template<> struct isPodLike<SimpleValue> {
  static const bool value = true;
};

template<> struct DenseMapInfo<SimpleValue> {
  static inline SimpleValue getEmptyKey() {
    return SimpleValue::get(DenseMapInfo<Instruction*>::getEmptyKey());
  }
  static inline SimpleValue getTombstoneKey() {
    return SimpleValue::get(DenseMapInfo<Instruction*>::getTombstoneKey());
  }
  static unsigned getHashValue(SimpleValue Val);
  static bool isEqual(SimpleValue LHS, SimpleValue RHS);
};
}

unsigned DenseMapInfo<SimpleValue>::getHashValue(SimpleValue Val) {
  Instruction *Inst = Val.Inst;
  
  // Hash in all of the operands as pointers.
  unsigned Res = 0;
  for (unsigned i = 0, e = Inst->getNumOperands(); i != e; ++i)
    Res ^= getHash(Inst->getOperand(i)) << i;

  if (CastInst *CI = dyn_cast<CastInst>(Inst))
    Res ^= getHash(CI->getType());
  else if (CmpInst *CI = dyn_cast<CmpInst>(Inst))
    Res ^= CI->getPredicate();
  else if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(Inst)) {
    for (ExtractValueInst::idx_iterator I = EVI->idx_begin(),
         E = EVI->idx_end(); I != E; ++I)
      Res ^= *I;
  } else if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(Inst)) {
    for (InsertValueInst::idx_iterator I = IVI->idx_begin(),
         E = IVI->idx_end(); I != E; ++I)
      Res ^= *I;
  } else {
    // nothing extra to hash in.
    assert((isa<BinaryOperator>(Inst) || isa<GetElementPtrInst>(Inst) ||
            isa<SelectInst>(Inst) || isa<ExtractElementInst>(Inst) ||
            isa<InsertElementInst>(Inst) || isa<ShuffleVectorInst>(Inst)) &&
           "Invalid/unknown instruction");
  }

  // Mix in the opcode.
  return (Res << 1) ^ Inst->getOpcode();
}

bool DenseMapInfo<SimpleValue>::isEqual(SimpleValue LHS, SimpleValue RHS) {
  Instruction *LHSI = LHS.Inst, *RHSI = RHS.Inst;

  if (LHS.isSentinel() || RHS.isSentinel())
    return LHSI == RHSI;
  
  if (LHSI->getOpcode() != RHSI->getOpcode()) return false;
  return LHSI->isIdenticalTo(RHSI);
}

//===----------------------------------------------------------------------===//
// MemoryValue 
//===----------------------------------------------------------------------===//

namespace {
  /// MemoryValue - Instances of this struct represent available load and call
  /// values in the scoped hash table.
  struct MemoryValue {
    Instruction *Inst;
    
    bool isSentinel() const {
      return Inst == DenseMapInfo<Instruction*>::getEmptyKey() ||
             Inst == DenseMapInfo<Instruction*>::getTombstoneKey();
    }
    
    static bool canHandle(Instruction *Inst) {
      if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
        return !LI->isVolatile();
      if (CallInst *CI = dyn_cast<CallInst>(Inst))
        return CI->onlyReadsMemory();
      return false;
    }
    
    static MemoryValue get(Instruction *I) {
      MemoryValue X; X.Inst = I;
      assert((X.isSentinel() || canHandle(I)) && "Inst can't be handled!");
      return X;
    }
  };
}

namespace llvm {
  // MemoryValue is POD.
  template<> struct isPodLike<MemoryValue> {
    static const bool value = true;
  };
  
  template<> struct DenseMapInfo<MemoryValue> {
    static inline MemoryValue getEmptyKey() {
      return MemoryValue::get(DenseMapInfo<Instruction*>::getEmptyKey());
    }
    static inline MemoryValue getTombstoneKey() {
      return MemoryValue::get(DenseMapInfo<Instruction*>::getTombstoneKey());
    }
    static unsigned getHashValue(MemoryValue Val);
    static bool isEqual(MemoryValue LHS, MemoryValue RHS);
  };
}
unsigned DenseMapInfo<MemoryValue>::getHashValue(MemoryValue Val) {
  Instruction *Inst = Val.Inst;
  // Hash in all of the operands as pointers.
  unsigned Res = 0;
  for (unsigned i = 0, e = Inst->getNumOperands(); i != e; ++i)
    Res ^= getHash(Inst->getOperand(i)) << i;
  // Mix in the opcode.
  return (Res << 1) ^ Inst->getOpcode();
}

bool DenseMapInfo<MemoryValue>::isEqual(MemoryValue LHS, MemoryValue RHS) {
  Instruction *LHSI = LHS.Inst, *RHSI = RHS.Inst;
  
  if (LHS.isSentinel() || RHS.isSentinel())
    return LHSI == RHSI;
  
  if (LHSI->getOpcode() != RHSI->getOpcode()) return false;
  return LHSI->isIdenticalTo(RHSI);
}


//===----------------------------------------------------------------------===//
// EarlyCSE pass. 
//===----------------------------------------------------------------------===//

namespace {
  
/// EarlyCSE - This pass does a simple depth-first walk over the dominator
/// tree, eliminating trivially redundant instructions and using instsimplify
/// to canonicalize things as it goes.  It is intended to be fast and catch
/// obvious cases so that instcombine and other passes are more effective.  It
/// is expected that a later pass of GVN will catch the interesting/hard
/// cases.
class EarlyCSE : public FunctionPass {
public:
  const TargetData *TD;
  DominatorTree *DT;
  typedef RecyclingAllocator<BumpPtrAllocator,
                      ScopedHashTableVal<SimpleValue, Value*> > AllocatorTy;
  typedef ScopedHashTable<SimpleValue, Value*, DenseMapInfo<SimpleValue>,
                          AllocatorTy> ScopedHTType;
  
  /// AvailableValues - This scoped hash table contains the current values of
  /// all of our simple scalar expressions.  As we walk down the domtree, we
  /// look to see if instructions are in this: if so, we replace them with what
  /// we find, otherwise we insert them so that dominated values can succeed in
  /// their lookup.
  ScopedHTType *AvailableValues;
  
  typedef ScopedHashTable<MemoryValue, std::pair<Value*, unsigned> > MemHTType;
  /// AvailableMemValues - This scoped hash table contains the current values of
  /// loads and other read-only memory values.  This allows us to get efficient
  /// access to dominating loads we we find a fully redundant load.  In addition
  /// to the most recent load, we keep track of a generation count of the read,
  /// which is compared against the current generation count.  The current
  /// generation count is  incremented after every possibly writing memory
  /// operation, which ensures that we only CSE loads with other loads that have
  /// no intervening store.
  MemHTType *AvailableMemValues;
  
  /// CurrentGeneration - This is the current generation of the memory value.
  unsigned CurrentGeneration;
  
  static char ID;
  explicit EarlyCSE() : FunctionPass(ID) {
    initializeEarlyCSEPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F);

private:
  
  bool processNode(DomTreeNode *Node);
  
  // This transformation requires dominator postdominator info
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<DominatorTree>();
    AU.setPreservesCFG();
  }
};
}

char EarlyCSE::ID = 0;

// createEarlyCSEPass - The public interface to this file.
FunctionPass *llvm::createEarlyCSEPass() {
  return new EarlyCSE();
}

INITIALIZE_PASS_BEGIN(EarlyCSE, "early-cse", "Early CSE", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(EarlyCSE, "early-cse", "Early CSE", false, false)

bool EarlyCSE::processNode(DomTreeNode *Node) {
  // Define a scope in the scoped hash table.  When we are done processing this
  // domtree node and recurse back up to our parent domtree node, this will pop
  // off all the values we install.
  ScopedHTType::ScopeTy Scope(*AvailableValues);
  
  // Define a scope for the memory values so that anything we add will get
  // popped when we recurse back up to our parent domtree node.
  MemHTType::ScopeTy MemScope(*AvailableMemValues);
  
  BasicBlock *BB = Node->getBlock();
  
  // If this block has a single predecessor, then the predecessor is the parent
  // of the domtree node and all of the live out memory values are still current
  // in this block.  If this block has multiple predecessors, then they could
  // have invalidated the live-out memory values of our parent value.  For now,
  // just be conservative and invalidate memory if this block has multiple
  // predecessors.
  if (BB->getSinglePredecessor() == 0)
    ++CurrentGeneration;
  
  bool Changed = false;

  // See if any instructions in the block can be eliminated.  If so, do it.  If
  // not, add them to AvailableValues.
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    Instruction *Inst = I++;
    
    // Dead instructions should just be removed.
    if (isInstructionTriviallyDead(Inst)) {
      DEBUG(dbgs() << "EarlyCSE DCE: " << *Inst << '\n');
      Inst->eraseFromParent();
      Changed = true;
      ++NumSimplify;
      continue;
    }
    
    // If the instruction can be simplified (e.g. X+0 = X) then replace it with
    // its simpler value.
    if (Value *V = SimplifyInstruction(Inst, TD, DT)) {
      DEBUG(dbgs() << "EarlyCSE Simplify: " << *Inst << "  to: " << *V << '\n');
      Inst->replaceAllUsesWith(V);
      Inst->eraseFromParent();
      Changed = true;
      ++NumSimplify;
      continue;
    }
    
    // If this is a simple instruction that we can value number, process it.
    if (SimpleValue::canHandle(Inst)) {
      // See if the instruction has an available value.  If so, use it.
      if (Value *V = AvailableValues->lookup(SimpleValue::get(Inst))) {
        DEBUG(dbgs() << "EarlyCSE CSE: " << *Inst << "  to: " << *V << '\n');
        Inst->replaceAllUsesWith(V);
        Inst->eraseFromParent();
        Changed = true;
        ++NumCSE;
        continue;
      }
      
      // Otherwise, just remember that this value is available.
      AvailableValues->insert(SimpleValue::get(Inst), Inst);
      continue;
    }
    
    // If this is a read-only memory value, process it.
    if (MemoryValue::canHandle(Inst)) {
      // If we have an available version of this value, and if it is the right
      // generation, replace this instruction.
      std::pair<Value*, unsigned> InVal = 
        AvailableMemValues->lookup(MemoryValue::get(Inst));
      if (InVal.first != 0 && InVal.second == CurrentGeneration) {
        DEBUG(dbgs() << "EarlyCSE CSE MEM: " << *Inst << "  to: "
                     << *InVal.first << '\n');
        if (!Inst->use_empty()) Inst->replaceAllUsesWith(InVal.first);
        Inst->eraseFromParent();
        Changed = true;
        ++NumCSEMem;
        continue;
      }
      
      // Otherwise, remember that we have this instruction.
      AvailableMemValues->insert(MemoryValue::get(Inst),
                         std::pair<Value*, unsigned>(Inst, CurrentGeneration));
      continue;
    }
    
    // Okay, this isn't something we can CSE at all.  Check to see if it is
    // something that could modify memory.  If so, our available memory values
    // cannot be used so bump the generation count.
    if (Inst->mayWriteToMemory())
      ++CurrentGeneration;
  }
  
  unsigned LiveOutGeneration = CurrentGeneration;
  for (DomTreeNode::iterator I = Node->begin(), E = Node->end(); I != E; ++I) {
    Changed |= processNode(*I);
    // Pop any generation changes off the stack from the recursive walk.
    CurrentGeneration = LiveOutGeneration;
  }
  return Changed;
}


bool EarlyCSE::runOnFunction(Function &F) {
  TD = getAnalysisIfAvailable<TargetData>();
  DT = &getAnalysis<DominatorTree>();
  ScopedHTType AVTable;
  AvailableValues = &AVTable;

  MemHTType MemTable;
  AvailableMemValues = &MemTable;
  
  CurrentGeneration = 0;
  return processNode(DT->getRootNode());
}
