//===- CorrelatedExprs.cpp - Pass to detect and eliminated c.e.'s ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Correlated Expression Elimination propagates information from conditional
// branches to blocks dominated by destinations of the branch.  It propagates
// information from the condition check itself into the body of the branch,
// allowing transformations like these for example:
//
//  if (i == 7)
//    ... 4*i;  // constant propagation
//
//  M = i+1; N = j+1;
//  if (i == j)
//    X = M-N;  // = M-M == 0;
//
// This is called Correlated Expression Elimination because we eliminate or
// simplify expressions that are correlated with the direction of a branch.  In
// this way we use static information to give us some information about the
// dynamic value of a variable.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "cee"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumCmpRemoved, "Number of cmp instruction eliminated");
STATISTIC(NumOperandsCann, "Number of operands canonicalized");
STATISTIC(BranchRevectors, "Number of branches revectored");

namespace {
  class ValueInfo;
  class VISIBILITY_HIDDEN Relation {
    Value *Val;          // Relation to what value?
    unsigned Rel;        // SetCC or ICmp relation, or Add if no information
  public:
    Relation(Value *V) : Val(V), Rel(Instruction::Add) {}
    bool operator<(const Relation &R) const { return Val < R.Val; }
    Value *getValue() const { return Val; }
    unsigned getRelation() const { return Rel; }

    // contradicts - Return true if the relationship specified by the operand
    // contradicts already known information.
    //
    bool contradicts(unsigned Rel, const ValueInfo &VI) const;

    // incorporate - Incorporate information in the argument into this relation
    // entry.  This assumes that the information doesn't contradict itself.  If
    // any new information is gained, true is returned, otherwise false is
    // returned to indicate that nothing was updated.
    //
    bool incorporate(unsigned Rel, ValueInfo &VI);

    // KnownResult - Whether or not this condition determines the result of a
    // setcc or icmp in the program.  False & True are intentionally 0 & 1 
    // so we can convert to bool by casting after checking for unknown.
    //
    enum KnownResult { KnownFalse = 0, KnownTrue = 1, Unknown = 2 };

    // getImpliedResult - If this relationship between two values implies that
    // the specified relationship is true or false, return that.  If we cannot
    // determine the result required, return Unknown.
    //
    KnownResult getImpliedResult(unsigned Rel) const;

    // print - Output this relation to the specified stream
    void print(std::ostream &OS) const;
    void dump() const;
  };


  // ValueInfo - One instance of this record exists for every value with
  // relationships between other values.  It keeps track of all of the
  // relationships to other values in the program (specified with Relation) that
  // are known to be valid in a region.
  //
  class VISIBILITY_HIDDEN ValueInfo {
    // RelationShips - this value is know to have the specified relationships to
    // other values.  There can only be one entry per value, and this list is
    // kept sorted by the Val field.
    std::vector<Relation> Relationships;

    // If information about this value is known or propagated from constant
    // expressions, this range contains the possible values this value may hold.
    ConstantRange Bounds;

    // If we find that this value is equal to another value that has a lower
    // rank, this value is used as it's replacement.
    //
    Value *Replacement;
  public:
    ValueInfo(const Type *Ty)
      : Bounds(Ty->isInteger() ? cast<IntegerType>(Ty)->getBitWidth()  : 32), 
               Replacement(0) {}

    // getBounds() - Return the constant bounds of the value...
    const ConstantRange &getBounds() const { return Bounds; }
    ConstantRange &getBounds() { return Bounds; }

    const std::vector<Relation> &getRelationships() { return Relationships; }

    // getReplacement - Return the value this value is to be replaced with if it
    // exists, otherwise return null.
    //
    Value *getReplacement() const { return Replacement; }

    // setReplacement - Used by the replacement calculation pass to figure out
    // what to replace this value with, if anything.
    //
    void setReplacement(Value *Repl) { Replacement = Repl; }

    // getRelation - return the relationship entry for the specified value.
    // This can invalidate references to other Relations, so use it carefully.
    //
    Relation &getRelation(Value *V) {
      // Binary search for V's entry...
      std::vector<Relation>::iterator I =
        std::lower_bound(Relationships.begin(), Relationships.end(),
                         Relation(V));

      // If we found the entry, return it...
      if (I != Relationships.end() && I->getValue() == V)
        return *I;

      // Insert and return the new relationship...
      return *Relationships.insert(I, V);
    }

    const Relation *requestRelation(Value *V) const {
      // Binary search for V's entry...
      std::vector<Relation>::const_iterator I =
        std::lower_bound(Relationships.begin(), Relationships.end(),
                         Relation(V));
      if (I != Relationships.end() && I->getValue() == V)
        return &*I;
      return 0;
    }

    // print - Output information about this value relation...
    void print(std::ostream &OS, Value *V) const;
    void dump() const;
  };

  // RegionInfo - Keeps track of all of the value relationships for a region.  A
  // region is the are dominated by a basic block.  RegionInfo's keep track of
  // the RegionInfo for their dominator, because anything known in a dominator
  // is known to be true in a dominated block as well.
  //
  class VISIBILITY_HIDDEN RegionInfo {
    BasicBlock *BB;

    // ValueMap - Tracks the ValueInformation known for this region
    typedef std::map<Value*, ValueInfo> ValueMapTy;
    ValueMapTy ValueMap;
  public:
    RegionInfo(BasicBlock *bb) : BB(bb) {}

    // getEntryBlock - Return the block that dominates all of the members of
    // this region.
    BasicBlock *getEntryBlock() const { return BB; }

    // empty - return true if this region has no information known about it.
    bool empty() const { return ValueMap.empty(); }

    const RegionInfo &operator=(const RegionInfo &RI) {
      ValueMap = RI.ValueMap;
      return *this;
    }

    // print - Output information about this region...
    void print(std::ostream &OS) const;
    void dump() const;

    // Allow external access.
    typedef ValueMapTy::iterator iterator;
    iterator begin() { return ValueMap.begin(); }
    iterator end() { return ValueMap.end(); }

    ValueInfo &getValueInfo(Value *V) {
      ValueMapTy::iterator I = ValueMap.lower_bound(V);
      if (I != ValueMap.end() && I->first == V) return I->second;
      return ValueMap.insert(I, std::make_pair(V, V->getType()))->second;
    }

    const ValueInfo *requestValueInfo(Value *V) const {
      ValueMapTy::const_iterator I = ValueMap.find(V);
      if (I != ValueMap.end()) return &I->second;
      return 0;
    }

    /// removeValueInfo - Remove anything known about V from our records.  This
    /// works whether or not we know anything about V.
    ///
    void removeValueInfo(Value *V) {
      ValueMap.erase(V);
    }
  };

  /// CEE - Correlated Expression Elimination
  class VISIBILITY_HIDDEN CEE : public FunctionPass {
    std::map<Value*, unsigned> RankMap;
    std::map<BasicBlock*, RegionInfo> RegionInfoMap;
    ETForest *EF;
  public:
    static char ID; // Pass identification, replacement for typeid
    CEE() : FunctionPass((intptr_t)&ID) {}

    virtual bool runOnFunction(Function &F);

    // We don't modify the program, so we preserve all analyses
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<ETForest>();
      AU.addRequiredID(BreakCriticalEdgesID);
    };

    // print - Implement the standard print form to print out analysis
    // information.
    virtual void print(std::ostream &O, const Module *M) const;

  private:
    RegionInfo &getRegionInfo(BasicBlock *BB) {
      std::map<BasicBlock*, RegionInfo>::iterator I
        = RegionInfoMap.lower_bound(BB);
      if (I != RegionInfoMap.end() && I->first == BB) return I->second;
      return RegionInfoMap.insert(I, std::make_pair(BB, BB))->second;
    }

    void BuildRankMap(Function &F);
    unsigned getRank(Value *V) const {
      if (isa<Constant>(V)) return 0;
      std::map<Value*, unsigned>::const_iterator I = RankMap.find(V);
      if (I != RankMap.end()) return I->second;
      return 0; // Must be some other global thing
    }

    bool TransformRegion(BasicBlock *BB, std::set<BasicBlock*> &VisitedBlocks);

    bool ForwardCorrelatedEdgeDestination(TerminatorInst *TI, unsigned SuccNo,
                                          RegionInfo &RI);

    void ForwardSuccessorTo(TerminatorInst *TI, unsigned Succ, BasicBlock *D,
                            RegionInfo &RI);
    void ReplaceUsesOfValueInRegion(Value *Orig, Value *New,
                                    BasicBlock *RegionDominator);
    void CalculateRegionExitBlocks(BasicBlock *BB, BasicBlock *OldSucc,
                                   std::vector<BasicBlock*> &RegionExitBlocks);
    void InsertRegionExitMerges(PHINode *NewPHI, Instruction *OldVal,
                             const std::vector<BasicBlock*> &RegionExitBlocks);

    void PropagateBranchInfo(BranchInst *BI);
    void PropagateSwitchInfo(SwitchInst *SI);
    void PropagateEquality(Value *Op0, Value *Op1, RegionInfo &RI);
    void PropagateRelation(unsigned Opcode, Value *Op0,
                           Value *Op1, RegionInfo &RI);
    void UpdateUsersOfValue(Value *V, RegionInfo &RI);
    void IncorporateInstruction(Instruction *Inst, RegionInfo &RI);
    void ComputeReplacements(RegionInfo &RI);

    // getCmpResult - Given a icmp instruction, determine if the result is
    // determined by facts we already know about the region under analysis.
    // Return KnownTrue, KnownFalse, or UnKnown based on what we can determine.
    Relation::KnownResult getCmpResult(CmpInst *ICI, const RegionInfo &RI);

    bool SimplifyBasicBlock(BasicBlock &BB, const RegionInfo &RI);
    bool SimplifyInstruction(Instruction *Inst, const RegionInfo &RI);
  };
  
  char CEE::ID = 0;
  RegisterPass<CEE> X("cee", "Correlated Expression Elimination");
}

FunctionPass *llvm::createCorrelatedExpressionEliminationPass() {
  return new CEE();
}


bool CEE::runOnFunction(Function &F) {
  // Build a rank map for the function...
  BuildRankMap(F);

  // Traverse the dominator tree, computing information for each node in the
  // tree.  Note that our traversal will not even touch unreachable basic
  // blocks.
  EF = &getAnalysis<ETForest>();

  std::set<BasicBlock*> VisitedBlocks;
  bool Changed = TransformRegion(&F.getEntryBlock(), VisitedBlocks);

  RegionInfoMap.clear();
  RankMap.clear();
  return Changed;
}

// TransformRegion - Transform the region starting with BB according to the
// calculated region information for the block.  Transforming the region
// involves analyzing any information this block provides to successors,
// propagating the information to successors, and finally transforming
// successors.
//
// This method processes the function in depth first order, which guarantees
// that we process the immediate dominator of a block before the block itself.
// Because we are passing information from immediate dominators down to
// dominatees, we obviously have to process the information source before the
// information consumer.
//
bool CEE::TransformRegion(BasicBlock *BB, std::set<BasicBlock*> &VisitedBlocks){
  // Prevent infinite recursion...
  if (VisitedBlocks.count(BB)) return false;
  VisitedBlocks.insert(BB);

  // Get the computed region information for this block...
  RegionInfo &RI = getRegionInfo(BB);

  // Compute the replacement information for this block...
  ComputeReplacements(RI);

  // If debugging, print computed region information...
  DEBUG(RI.print(*cerr.stream()));

  // Simplify the contents of this block...
  bool Changed = SimplifyBasicBlock(*BB, RI);

  // Get the terminator of this basic block...
  TerminatorInst *TI = BB->getTerminator();

  // Loop over all of the blocks that this block is the immediate dominator for.
  // Because all information known in this region is also known in all of the
  // blocks that are dominated by this one, we can safely propagate the
  // information down now.
  //
  std::vector<BasicBlock*> children;
  EF->getChildren(BB, children);
  if (!RI.empty()) {     // Time opt: only propagate if we can change something
    for (std::vector<BasicBlock*>::iterator CI = children.begin(), 
         E = children.end(); CI != E; ++CI) {
      assert(RegionInfoMap.find(*CI) == RegionInfoMap.end() &&
             "RegionInfo should be calculated in dominanace order!");
      getRegionInfo(*CI) = RI;
    }
  }

  // Now that all of our successors have information if they deserve it,
  // propagate any information our terminator instruction finds to our
  // successors.
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isConditional())
      PropagateBranchInfo(BI);
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    PropagateSwitchInfo(SI);
  }

  // If this is a branch to a block outside our region that simply performs
  // another conditional branch, one whose outcome is known inside of this
  // region, then vector this outgoing edge directly to the known destination.
  //
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
    while (ForwardCorrelatedEdgeDestination(TI, i, RI)) {
      ++BranchRevectors;
      Changed = true;
    }

  // Now that all of our successors have information, recursively process them.
  for (std::vector<BasicBlock*>::iterator CI = children.begin(), 
       E = children.end(); CI != E; ++CI)
    Changed |= TransformRegion(*CI, VisitedBlocks);

  return Changed;
}

// isBlockSimpleEnoughForCheck to see if the block is simple enough for us to
// revector the conditional branch in the bottom of the block, do so now.
//
static bool isBlockSimpleEnough(BasicBlock *BB) {
  assert(isa<BranchInst>(BB->getTerminator()));
  BranchInst *BI = cast<BranchInst>(BB->getTerminator());
  assert(BI->isConditional());

  // Check the common case first: empty block, or block with just a setcc.
  if (BB->size() == 1 ||
      (BB->size() == 2 && &BB->front() == BI->getCondition() &&
       BI->getCondition()->hasOneUse()))
    return true;

  // Check the more complex case now...
  BasicBlock::iterator I = BB->begin();

  // FIXME: This should be reenabled once the regression with SIM is fixed!
#if 0
  // PHI Nodes are ok, just skip over them...
  while (isa<PHINode>(*I)) ++I;
#endif

  // Accept the setcc instruction...
  if (&*I == BI->getCondition())
    ++I;

  // Nothing else is acceptable here yet.  We must not revector... unless we are
  // at the terminator instruction.
  if (&*I == BI)
    return true;

  return false;
}


bool CEE::ForwardCorrelatedEdgeDestination(TerminatorInst *TI, unsigned SuccNo,
                                           RegionInfo &RI) {
  // If this successor is a simple block not in the current region, which
  // contains only a conditional branch, we decide if the outcome of the branch
  // can be determined from information inside of the region.  Instead of going
  // to this block, we can instead go to the destination we know is the right
  // target.
  //

  // Check to see if we dominate the block. If so, this block will get the
  // condition turned to a constant anyway.
  //
  //if (EF->dominates(RI.getEntryBlock(), BB))
  // return 0;

  BasicBlock *BB = TI->getParent();

  // Get the destination block of this edge...
  BasicBlock *OldSucc = TI->getSuccessor(SuccNo);

  // Make sure that the block ends with a conditional branch and is simple
  // enough for use to be able to revector over.
  BranchInst *BI = dyn_cast<BranchInst>(OldSucc->getTerminator());
  if (BI == 0 || !BI->isConditional() || !isBlockSimpleEnough(OldSucc))
    return false;

  // We can only forward the branch over the block if the block ends with a
  // cmp we can determine the outcome for.
  //
  // FIXME: we can make this more generic.  Code below already handles more
  // generic case.
  if (!isa<CmpInst>(BI->getCondition()))
    return false;

  // Make a new RegionInfo structure so that we can simulate the effect of the
  // PHI nodes in the block we are skipping over...
  //
  RegionInfo NewRI(RI);

  // Remove value information for all of the values we are simulating... to make
  // sure we don't have any stale information.
  for (BasicBlock::iterator I = OldSucc->begin(), E = OldSucc->end(); I!=E; ++I)
    if (I->getType() != Type::VoidTy)
      NewRI.removeValueInfo(I);

  // Put the newly discovered information into the RegionInfo...
  for (BasicBlock::iterator I = OldSucc->begin(), E = OldSucc->end(); I!=E; ++I)
    if (PHINode *PN = dyn_cast<PHINode>(I)) {
      int OpNum = PN->getBasicBlockIndex(BB);
      assert(OpNum != -1 && "PHI doesn't have incoming edge for predecessor!?");
      PropagateEquality(PN, PN->getIncomingValue(OpNum), NewRI);
    } else if (CmpInst *CI = dyn_cast<CmpInst>(I)) {
      Relation::KnownResult Res = getCmpResult(CI, NewRI);
      if (Res == Relation::Unknown) return false;
      PropagateEquality(CI, ConstantInt::get(Type::Int1Ty, Res), NewRI);
    } else {
      assert(isa<BranchInst>(*I) && "Unexpected instruction type!");
    }

  // Compute the facts implied by what we have discovered...
  ComputeReplacements(NewRI);

  ValueInfo &PredicateVI = NewRI.getValueInfo(BI->getCondition());
  if (PredicateVI.getReplacement() &&
      isa<Constant>(PredicateVI.getReplacement()) &&
      !isa<GlobalValue>(PredicateVI.getReplacement())) {
    ConstantInt *CB = cast<ConstantInt>(PredicateVI.getReplacement());

    // Forward to the successor that corresponds to the branch we will take.
    ForwardSuccessorTo(TI, SuccNo, 
                       BI->getSuccessor(!CB->getZExtValue()), NewRI);
    return true;
  }

  return false;
}

static Value *getReplacementOrValue(Value *V, RegionInfo &RI) {
  if (const ValueInfo *VI = RI.requestValueInfo(V))
    if (Value *Repl = VI->getReplacement())
      return Repl;
  return V;
}

/// ForwardSuccessorTo - We have found that we can forward successor # 'SuccNo'
/// of Terminator 'TI' to the 'Dest' BasicBlock.  This method performs the
/// mechanics of updating SSA information and revectoring the branch.
///
void CEE::ForwardSuccessorTo(TerminatorInst *TI, unsigned SuccNo,
                             BasicBlock *Dest, RegionInfo &RI) {
  // If there are any PHI nodes in the Dest BB, we must duplicate the entry
  // in the PHI node for the old successor to now include an entry from the
  // current basic block.
  //
  BasicBlock *OldSucc = TI->getSuccessor(SuccNo);
  BasicBlock *BB = TI->getParent();

  DOUT << "Forwarding branch in basic block %" << BB->getName()
       << " from block %" << OldSucc->getName() << " to block %"
       << Dest->getName() << "\n"
       << "Before forwarding: " << *BB->getParent();

  // Because we know that there cannot be critical edges in the flow graph, and
  // that OldSucc has multiple outgoing edges, this means that Dest cannot have
  // multiple incoming edges.
  //
#ifndef NDEBUG
  pred_iterator DPI = pred_begin(Dest); ++DPI;
  assert(DPI == pred_end(Dest) && "Critical edge found!!");
#endif

  // Loop over any PHI nodes in the destination, eliminating them, because they
  // may only have one input.
  //
  while (PHINode *PN = dyn_cast<PHINode>(&Dest->front())) {
    assert(PN->getNumIncomingValues() == 1 && "Crit edge found!");
    // Eliminate the PHI node
    PN->replaceAllUsesWith(PN->getIncomingValue(0));
    Dest->getInstList().erase(PN);
  }

  // If there are values defined in the "OldSucc" basic block, we need to insert
  // PHI nodes in the regions we are dealing with to emulate them.  This can
  // insert dead phi nodes, but it is more trouble to see if they are used than
  // to just blindly insert them.
  //
  if (EF->dominates(OldSucc, Dest)) {
    // RegionExitBlocks - Find all of the blocks that are not dominated by Dest,
    // but have predecessors that are.  Additionally, prune down the set to only
    // include blocks that are dominated by OldSucc as well.
    //
    std::vector<BasicBlock*> RegionExitBlocks;
    CalculateRegionExitBlocks(Dest, OldSucc, RegionExitBlocks);

    for (BasicBlock::iterator I = OldSucc->begin(), E = OldSucc->end();
         I != E; ++I)
      if (I->getType() != Type::VoidTy) {
        // Create and insert the PHI node into the top of Dest.
        PHINode *NewPN = new PHINode(I->getType(), I->getName()+".fw_merge",
                                     Dest->begin());
        // There is definitely an edge from OldSucc... add the edge now
        NewPN->addIncoming(I, OldSucc);

        // There is also an edge from BB now, add the edge with the calculated
        // value from the RI.
        NewPN->addIncoming(getReplacementOrValue(I, RI), BB);

        // Make everything in the Dest region use the new PHI node now...
        ReplaceUsesOfValueInRegion(I, NewPN, Dest);

        // Make sure that exits out of the region dominated by NewPN get PHI
        // nodes that merge the values as appropriate.
        InsertRegionExitMerges(NewPN, I, RegionExitBlocks);
      }
  }

  // If there were PHI nodes in OldSucc, we need to remove the entry for this
  // edge from the PHI node, and we need to replace any references to the PHI
  // node with a new value.
  //
  for (BasicBlock::iterator I = OldSucc->begin(); isa<PHINode>(I); ) {
    PHINode *PN = cast<PHINode>(I);

    // Get the value flowing across the old edge and remove the PHI node entry
    // for this edge: we are about to remove the edge!  Don't remove the PHI
    // node yet though if this is the last edge into it.
    Value *EdgeValue = PN->removeIncomingValue(BB, false);

    // Make sure that anything that used to use PN now refers to EdgeValue
    ReplaceUsesOfValueInRegion(PN, EdgeValue, Dest);

    // If there is only one value left coming into the PHI node, replace the PHI
    // node itself with the one incoming value left.
    //
    if (PN->getNumIncomingValues() == 1) {
      assert(PN->getNumIncomingValues() == 1);
      PN->replaceAllUsesWith(PN->getIncomingValue(0));
      PN->getParent()->getInstList().erase(PN);
      I = OldSucc->begin();
    } else if (PN->getNumIncomingValues() == 0) {  // Nuke the PHI
      // If we removed the last incoming value to this PHI, nuke the PHI node
      // now.
      PN->replaceAllUsesWith(Constant::getNullValue(PN->getType()));
      PN->getParent()->getInstList().erase(PN);
      I = OldSucc->begin();
    } else {
      ++I;  // Otherwise, move on to the next PHI node
    }
  }

  // Actually revector the branch now...
  TI->setSuccessor(SuccNo, Dest);

  // If we just introduced a critical edge in the flow graph, make sure to break
  // it right away...
  SplitCriticalEdge(TI, SuccNo, this);

  // Make sure that we don't introduce critical edges from oldsucc now!
  for (unsigned i = 0, e = OldSucc->getTerminator()->getNumSuccessors();
       i != e; ++i)
    SplitCriticalEdge(OldSucc->getTerminator(), i, this);

  // Since we invalidated the CFG, recalculate the dominator set so that it is
  // useful for later processing!
  // FIXME: This is much worse than it really should be!
  //EF->recalculate();

  DOUT << "After forwarding: " << *BB->getParent();
}

/// ReplaceUsesOfValueInRegion - This method replaces all uses of Orig with uses
/// of New.  It only affects instructions that are defined in basic blocks that
/// are dominated by Head.
///
void CEE::ReplaceUsesOfValueInRegion(Value *Orig, Value *New,
                                     BasicBlock *RegionDominator) {
  assert(Orig != New && "Cannot replace value with itself");
  std::vector<Instruction*> InstsToChange;
  std::vector<PHINode*>     PHIsToChange;
  InstsToChange.reserve(Orig->getNumUses());

  // Loop over instructions adding them to InstsToChange vector, this allows us
  // an easy way to avoid invalidating the use_iterator at a bad time.
  for (Value::use_iterator I = Orig->use_begin(), E = Orig->use_end();
       I != E; ++I)
    if (Instruction *User = dyn_cast<Instruction>(*I))
      if (EF->dominates(RegionDominator, User->getParent()))
        InstsToChange.push_back(User);
      else if (PHINode *PN = dyn_cast<PHINode>(User)) {
        PHIsToChange.push_back(PN);
      }

  // PHIsToChange contains PHI nodes that use Orig that do not live in blocks
  // dominated by orig.  If the block the value flows in from is dominated by
  // RegionDominator, then we rewrite the PHI
  for (unsigned i = 0, e = PHIsToChange.size(); i != e; ++i) {
    PHINode *PN = PHIsToChange[i];
    for (unsigned j = 0, e = PN->getNumIncomingValues(); j != e; ++j)
      if (PN->getIncomingValue(j) == Orig &&
          EF->dominates(RegionDominator, PN->getIncomingBlock(j)))
        PN->setIncomingValue(j, New);
  }

  // Loop over the InstsToChange list, replacing all uses of Orig with uses of
  // New.  This list contains all of the instructions in our region that use
  // Orig.
  for (unsigned i = 0, e = InstsToChange.size(); i != e; ++i)
    if (PHINode *PN = dyn_cast<PHINode>(InstsToChange[i])) {
      // PHINodes must be handled carefully.  If the PHI node itself is in the
      // region, we have to make sure to only do the replacement for incoming
      // values that correspond to basic blocks in the region.
      for (unsigned j = 0, e = PN->getNumIncomingValues(); j != e; ++j)
        if (PN->getIncomingValue(j) == Orig &&
            EF->dominates(RegionDominator, PN->getIncomingBlock(j)))
          PN->setIncomingValue(j, New);

    } else {
      InstsToChange[i]->replaceUsesOfWith(Orig, New);
    }
}

static void CalcRegionExitBlocks(BasicBlock *Header, BasicBlock *BB,
                                 std::set<BasicBlock*> &Visited,
                                 ETForest &EF,
                                 std::vector<BasicBlock*> &RegionExitBlocks) {
  if (Visited.count(BB)) return;
  Visited.insert(BB);

  if (EF.dominates(Header, BB)) {  // Block in the region, recursively traverse
    for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I)
      CalcRegionExitBlocks(Header, *I, Visited, EF, RegionExitBlocks);
  } else {
    // Header does not dominate this block, but we have a predecessor that does
    // dominate us.  Add ourself to the list.
    RegionExitBlocks.push_back(BB);
  }
}

/// CalculateRegionExitBlocks - Find all of the blocks that are not dominated by
/// BB, but have predecessors that are.  Additionally, prune down the set to
/// only include blocks that are dominated by OldSucc as well.
///
void CEE::CalculateRegionExitBlocks(BasicBlock *BB, BasicBlock *OldSucc,
                                    std::vector<BasicBlock*> &RegionExitBlocks){
  std::set<BasicBlock*> Visited;  // Don't infinite loop

  // Recursively calculate blocks we are interested in...
  CalcRegionExitBlocks(BB, BB, Visited, *EF, RegionExitBlocks);

  // Filter out blocks that are not dominated by OldSucc...
  for (unsigned i = 0; i != RegionExitBlocks.size(); ) {
    if (EF->dominates(OldSucc, RegionExitBlocks[i]))
      ++i;  // Block is ok, keep it.
    else {
      // Move to end of list...
      std::swap(RegionExitBlocks[i], RegionExitBlocks.back());
      RegionExitBlocks.pop_back();        // Nuke the end
    }
  }
}

void CEE::InsertRegionExitMerges(PHINode *BBVal, Instruction *OldVal,
                             const std::vector<BasicBlock*> &RegionExitBlocks) {
  assert(BBVal->getType() == OldVal->getType() && "Should be derived values!");
  BasicBlock *BB = BBVal->getParent();

  // Loop over all of the blocks we have to place PHIs in, doing it.
  for (unsigned i = 0, e = RegionExitBlocks.size(); i != e; ++i) {
    BasicBlock *FBlock = RegionExitBlocks[i];  // Block on the frontier

    // Create the new PHI node
    PHINode *NewPN = new PHINode(BBVal->getType(),
                                 OldVal->getName()+".fw_frontier",
                                 FBlock->begin());

    // Add an incoming value for every predecessor of the block...
    for (pred_iterator PI = pred_begin(FBlock), PE = pred_end(FBlock);
         PI != PE; ++PI) {
      // If the incoming edge is from the region dominated by BB, use BBVal,
      // otherwise use OldVal.
      NewPN->addIncoming(EF->dominates(BB, *PI) ? BBVal : OldVal, *PI);
    }

    // Now make everyone dominated by this block use this new value!
    ReplaceUsesOfValueInRegion(OldVal, NewPN, FBlock);
  }
}



// BuildRankMap - This method builds the rank map data structure which gives
// each instruction/value in the function a value based on how early it appears
// in the function.  We give constants and globals rank 0, arguments are
// numbered starting at one, and instructions are numbered in reverse post-order
// from where the arguments leave off.  This gives instructions in loops higher
// values than instructions not in loops.
//
void CEE::BuildRankMap(Function &F) {
  unsigned Rank = 1;  // Skip rank zero.

  // Number the arguments...
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
    RankMap[I] = Rank++;

  // Number the instructions in reverse post order...
  ReversePostOrderTraversal<Function*> RPOT(&F);
  for (ReversePostOrderTraversal<Function*>::rpo_iterator I = RPOT.begin(),
         E = RPOT.end(); I != E; ++I)
    for (BasicBlock::iterator BBI = (*I)->begin(), E = (*I)->end();
         BBI != E; ++BBI)
      if (BBI->getType() != Type::VoidTy)
        RankMap[BBI] = Rank++;
}


// PropagateBranchInfo - When this method is invoked, we need to propagate
// information derived from the branch condition into the true and false
// branches of BI.  Since we know that there aren't any critical edges in the
// flow graph, this can proceed unconditionally.
//
void CEE::PropagateBranchInfo(BranchInst *BI) {
  assert(BI->isConditional() && "Must be a conditional branch!");

  // Propagate information into the true block...
  //
  PropagateEquality(BI->getCondition(), ConstantInt::getTrue(),
                    getRegionInfo(BI->getSuccessor(0)));

  // Propagate information into the false block...
  //
  PropagateEquality(BI->getCondition(), ConstantInt::getFalse(),
                    getRegionInfo(BI->getSuccessor(1)));
}


// PropagateSwitchInfo - We need to propagate the value tested by the
// switch statement through each case block.
//
void CEE::PropagateSwitchInfo(SwitchInst *SI) {
  // Propagate information down each of our non-default case labels.  We
  // don't yet propagate information down the default label, because a
  // potentially large number of inequality constraints provide less
  // benefit per unit work than a single equality constraint.
  //
  Value *cond = SI->getCondition();
  for (unsigned i = 1; i < SI->getNumSuccessors(); ++i)
    PropagateEquality(cond, SI->getSuccessorValue(i),
                      getRegionInfo(SI->getSuccessor(i)));
}


// PropagateEquality - If we discover that two values are equal to each other in
// a specified region, propagate this knowledge recursively.
//
void CEE::PropagateEquality(Value *Op0, Value *Op1, RegionInfo &RI) {
  if (Op0 == Op1) return;  // Gee whiz. Are these really equal each other?

  if (isa<Constant>(Op0))  // Make sure the constant is always Op1
    std::swap(Op0, Op1);

  // Make sure we don't already know these are equal, to avoid infinite loops...
  ValueInfo &VI = RI.getValueInfo(Op0);

  // Get information about the known relationship between Op0 & Op1
  Relation &KnownRelation = VI.getRelation(Op1);

  // If we already know they're equal, don't reprocess...
  if (KnownRelation.getRelation() == FCmpInst::FCMP_OEQ ||
      KnownRelation.getRelation() == ICmpInst::ICMP_EQ)
    return;

  // If this is boolean, check to see if one of the operands is a constant.  If
  // it's a constant, then see if the other one is one of a setcc instruction,
  // an AND, OR, or XOR instruction.
  //
  ConstantInt *CB = dyn_cast<ConstantInt>(Op1);
  if (CB && Op1->getType() == Type::Int1Ty) {
    if (Instruction *Inst = dyn_cast<Instruction>(Op0)) {
      // If we know that this instruction is an AND instruction, and the 
      // result is true, this means that both operands to the OR are known 
      // to be true as well.
      //
      if (CB->getZExtValue() && Inst->getOpcode() == Instruction::And) {
        PropagateEquality(Inst->getOperand(0), CB, RI);
        PropagateEquality(Inst->getOperand(1), CB, RI);
      }

      // If we know that this instruction is an OR instruction, and the result
      // is false, this means that both operands to the OR are know to be 
      // false as well.
      //
      if (!CB->getZExtValue() && Inst->getOpcode() == Instruction::Or) {
        PropagateEquality(Inst->getOperand(0), CB, RI);
        PropagateEquality(Inst->getOperand(1), CB, RI);
      }

      // If we know that this instruction is a NOT instruction, we know that 
      // the operand is known to be the inverse of whatever the current 
      // value is.
      //
      if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(Inst))
        if (BinaryOperator::isNot(BOp))
          PropagateEquality(BinaryOperator::getNotArgument(BOp),
                            ConstantInt::get(Type::Int1Ty, 
                                             !CB->getZExtValue()), RI);

      // If we know the value of a FCmp instruction, propagate the information
      // about the relation into this region as well.
      //
      if (FCmpInst *FCI = dyn_cast<FCmpInst>(Inst)) {
        if (CB->getZExtValue()) {  // If we know the condition is true...
          // Propagate info about the LHS to the RHS & RHS to LHS
          PropagateRelation(FCI->getPredicate(), FCI->getOperand(0),
                            FCI->getOperand(1), RI);
          PropagateRelation(FCI->getSwappedPredicate(),
                            FCI->getOperand(1), FCI->getOperand(0), RI);

        } else {               // If we know the condition is false...
          // We know the opposite of the condition is true...
          FCmpInst::Predicate C = FCI->getInversePredicate();

          PropagateRelation(C, FCI->getOperand(0), FCI->getOperand(1), RI);
          PropagateRelation(FCmpInst::getSwappedPredicate(C),
                            FCI->getOperand(1), FCI->getOperand(0), RI);
        }
      }
    
      // If we know the value of a ICmp instruction, propagate the information
      // about the relation into this region as well.
      //
      if (ICmpInst *ICI = dyn_cast<ICmpInst>(Inst)) {
        if (CB->getZExtValue()) { // If we know the condition is true...
          // Propagate info about the LHS to the RHS & RHS to LHS
          PropagateRelation(ICI->getPredicate(), ICI->getOperand(0),
                            ICI->getOperand(1), RI);
          PropagateRelation(ICI->getSwappedPredicate(), ICI->getOperand(1),
                            ICI->getOperand(1), RI);

        } else {               // If we know the condition is false ...
          // We know the opposite of the condition is true...
          ICmpInst::Predicate C = ICI->getInversePredicate();

          PropagateRelation(C, ICI->getOperand(0), ICI->getOperand(1), RI);
          PropagateRelation(ICmpInst::getSwappedPredicate(C),
                            ICI->getOperand(1), ICI->getOperand(0), RI);
        }
      }
    }
  }

  // Propagate information about Op0 to Op1 & visa versa
  PropagateRelation(ICmpInst::ICMP_EQ, Op0, Op1, RI);
  PropagateRelation(ICmpInst::ICMP_EQ, Op1, Op0, RI);
  PropagateRelation(FCmpInst::FCMP_OEQ, Op0, Op1, RI);
  PropagateRelation(FCmpInst::FCMP_OEQ, Op1, Op0, RI);
}


// PropagateRelation - We know that the specified relation is true in all of the
// blocks in the specified region.  Propagate the information about Op0 and
// anything derived from it into this region.
//
void CEE::PropagateRelation(unsigned Opcode, Value *Op0,
                            Value *Op1, RegionInfo &RI) {
  assert(Op0->getType() == Op1->getType() && "Equal types expected!");

  // Constants are already pretty well understood.  We will apply information
  // about the constant to Op1 in another call to PropagateRelation.
  //
  if (isa<Constant>(Op0)) return;

  // Get the region information for this block to update...
  ValueInfo &VI = RI.getValueInfo(Op0);

  // Get information about the known relationship between Op0 & Op1
  Relation &Op1R = VI.getRelation(Op1);

  // Quick bailout for common case if we are reprocessing an instruction...
  if (Op1R.getRelation() == Opcode)
    return;

  // If we already have information that contradicts the current information we
  // are propagating, ignore this info.  Something bad must have happened!
  //
  if (Op1R.contradicts(Opcode, VI)) {
    Op1R.contradicts(Opcode, VI);
    cerr << "Contradiction found for opcode: "
         << ((isa<ICmpInst>(Op0)||isa<ICmpInst>(Op1)) ? 
                  Instruction::getOpcodeName(Instruction::ICmp) :
                  Instruction::getOpcodeName(Opcode))
         << "\n";
    Op1R.print(*cerr.stream());
    return;
  }

  // If the information propagated is new, then we want process the uses of this
  // instruction to propagate the information down to them.
  //
  if (Op1R.incorporate(Opcode, VI))
    UpdateUsersOfValue(Op0, RI);
}


// UpdateUsersOfValue - The information about V in this region has been updated.
// Propagate this to all consumers of the value.
//
void CEE::UpdateUsersOfValue(Value *V, RegionInfo &RI) {
  for (Value::use_iterator I = V->use_begin(), E = V->use_end();
       I != E; ++I)
    if (Instruction *Inst = dyn_cast<Instruction>(*I)) {
      // If this is an instruction using a value that we know something about,
      // try to propagate information to the value produced by the
      // instruction.  We can only do this if it is an instruction we can
      // propagate information for (a setcc for example), and we only WANT to
      // do this if the instruction dominates this region.
      //
      // If the instruction doesn't dominate this region, then it cannot be
      // used in this region and we don't care about it.  If the instruction
      // is IN this region, then we will simplify the instruction before we
      // get to uses of it anyway, so there is no reason to bother with it
      // here.  This check is also effectively checking to make sure that Inst
      // is in the same function as our region (in case V is a global f.e.).
      //
      if (EF->properlyDominates(Inst->getParent(), RI.getEntryBlock()))
        IncorporateInstruction(Inst, RI);
    }
}

// IncorporateInstruction - We just updated the information about one of the
// operands to the specified instruction.  Update the information about the
// value produced by this instruction
//
void CEE::IncorporateInstruction(Instruction *Inst, RegionInfo &RI) {
  if (CmpInst *CI = dyn_cast<CmpInst>(Inst)) {
    // See if we can figure out a result for this instruction...
    Relation::KnownResult Result = getCmpResult(CI, RI);
    if (Result != Relation::Unknown) {
      PropagateEquality(CI, ConstantInt::get(Type::Int1Ty, Result != 0), RI);
    }
  }
}


// ComputeReplacements - Some values are known to be equal to other values in a
// region.  For example if there is a comparison of equality between a variable
// X and a constant C, we can replace all uses of X with C in the region we are
// interested in.  We generalize this replacement to replace variables with
// other variables if they are equal and there is a variable with lower rank
// than the current one.  This offers a canonicalizing property that exposes
// more redundancies for later transformations to take advantage of.
//
void CEE::ComputeReplacements(RegionInfo &RI) {
  // Loop over all of the values in the region info map...
  for (RegionInfo::iterator I = RI.begin(), E = RI.end(); I != E; ++I) {
    ValueInfo &VI = I->second;

    // If we know that this value is a particular constant, set Replacement to
    // the constant...
    Value *Replacement = 0;
    const APInt * Rplcmnt = VI.getBounds().getSingleElement();
    if (Rplcmnt)
      Replacement = ConstantInt::get(*Rplcmnt);

    // If this value is not known to be some constant, figure out the lowest
    // rank value that it is known to be equal to (if anything).
    //
    if (Replacement == 0) {
      // Find out if there are any equality relationships with values of lower
      // rank than VI itself...
      unsigned MinRank = getRank(I->first);

      // Loop over the relationships known about Op0.
      const std::vector<Relation> &Relationships = VI.getRelationships();
      for (unsigned i = 0, e = Relationships.size(); i != e; ++i)
        if (Relationships[i].getRelation() == FCmpInst::FCMP_OEQ) {
          unsigned R = getRank(Relationships[i].getValue());
          if (R < MinRank) {
            MinRank = R;
            Replacement = Relationships[i].getValue();
          }
        }
        else if (Relationships[i].getRelation() == ICmpInst::ICMP_EQ) {
          unsigned R = getRank(Relationships[i].getValue());
          if (R < MinRank) {
            MinRank = R;
            Replacement = Relationships[i].getValue();
          }
        }
    }

    // If we found something to replace this value with, keep track of it.
    if (Replacement)
      VI.setReplacement(Replacement);
  }
}

// SimplifyBasicBlock - Given information about values in region RI, simplify
// the instructions in the specified basic block.
//
bool CEE::SimplifyBasicBlock(BasicBlock &BB, const RegionInfo &RI) {
  bool Changed = false;
  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ) {
    Instruction *Inst = I++;

    // Convert instruction arguments to canonical forms...
    Changed |= SimplifyInstruction(Inst, RI);

    if (CmpInst *CI = dyn_cast<CmpInst>(Inst)) {
      // Try to simplify a setcc instruction based on inherited information
      Relation::KnownResult Result = getCmpResult(CI, RI);
      if (Result != Relation::Unknown) {
        DEBUG(cerr << "Replacing icmp with " << Result
                   << " constant: " << *CI);

        CI->replaceAllUsesWith(ConstantInt::get(Type::Int1Ty, (bool)Result));
        // The instruction is now dead, remove it from the program.
        CI->getParent()->getInstList().erase(CI);
        ++NumCmpRemoved;
        Changed = true;
      }
    }
  }

  return Changed;
}

// SimplifyInstruction - Inspect the operands of the instruction, converting
// them to their canonical form if possible.  This takes care of, for example,
// replacing a value 'X' with a constant 'C' if the instruction in question is
// dominated by a true seteq 'X', 'C'.
//
bool CEE::SimplifyInstruction(Instruction *I, const RegionInfo &RI) {
  bool Changed = false;

  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (const ValueInfo *VI = RI.requestValueInfo(I->getOperand(i)))
      if (Value *Repl = VI->getReplacement()) {
        // If we know if a replacement with lower rank than Op0, make the
        // replacement now.
        DOUT << "In Inst: " << *I << "  Replacing operand #" << i
             << " with " << *Repl << "\n";
        I->setOperand(i, Repl);
        Changed = true;
        ++NumOperandsCann;
      }

  return Changed;
}

// getCmpResult - Try to simplify a cmp instruction based on information
// inherited from a dominating icmp instruction.  V is one of the operands to
// the icmp instruction, and VI is the set of information known about it.  We
// take two cases into consideration here.  If the comparison is against a
// constant value, we can use the constant range to see if the comparison is
// possible to succeed.  If it is not a comparison against a constant, we check
// to see if there is a known relationship between the two values.  If so, we
// may be able to eliminate the check.
//
Relation::KnownResult CEE::getCmpResult(CmpInst *CI,
                                        const RegionInfo &RI) {
  Value *Op0 = CI->getOperand(0), *Op1 = CI->getOperand(1);
  unsigned short predicate = CI->getPredicate();

  if (isa<Constant>(Op0)) {
    if (isa<Constant>(Op1)) {
      if (Constant *Result = ConstantFoldInstruction(CI)) {
        // Wow, this is easy, directly eliminate the ICmpInst.
        DEBUG(cerr << "Replacing cmp with constant fold: " << *CI);
        return cast<ConstantInt>(Result)->getZExtValue()
          ? Relation::KnownTrue : Relation::KnownFalse;
      }
    } else {
      // We want to swap this instruction so that operand #0 is the constant.
      std::swap(Op0, Op1);
      if (isa<ICmpInst>(CI))
        predicate = cast<ICmpInst>(CI)->getSwappedPredicate();
      else
        predicate = cast<FCmpInst>(CI)->getSwappedPredicate();
    }
  }

  // Try to figure out what the result of this comparison will be...
  Relation::KnownResult Result = Relation::Unknown;

  // We have to know something about the relationship to prove anything...
  if (const ValueInfo *Op0VI = RI.requestValueInfo(Op0)) {

    // At this point, we know that if we have a constant argument that it is in
    // Op1.  Check to see if we know anything about comparing value with a
    // constant, and if we can use this info to fold the icmp.
    //
    if (ConstantInt *C = dyn_cast<ConstantInt>(Op1)) {
      // Check to see if we already know the result of this comparison...
      ICmpInst::Predicate ipred = ICmpInst::Predicate(predicate);
      ConstantRange R = ICmpInst::makeConstantRange(ipred, C->getValue());
      ConstantRange Int = R.intersectWith(Op0VI->getBounds());

      // If the intersection of the two ranges is empty, then the condition
      // could never be true!
      //
      if (Int.isEmptySet()) {
        Result = Relation::KnownFalse;

      // Otherwise, if VI.getBounds() (the possible values) is a subset of R
      // (the allowed values) then we know that the condition must always be
      // true!
      //
      } else if (Int == Op0VI->getBounds()) {
        Result = Relation::KnownTrue;
      }
    } else {
      // If we are here, we know that the second argument is not a constant
      // integral.  See if we know anything about Op0 & Op1 that allows us to
      // fold this anyway.
      //
      // Do we have value information about Op0 and a relation to Op1?
      if (const Relation *Op2R = Op0VI->requestRelation(Op1))
        Result = Op2R->getImpliedResult(predicate);
    }
  }
  return Result;
}

//===----------------------------------------------------------------------===//
//  Relation Implementation
//===----------------------------------------------------------------------===//

// contradicts - Return true if the relationship specified by the operand
// contradicts already known information.
//
bool Relation::contradicts(unsigned Op,
                           const ValueInfo &VI) const {
  assert (Op != Instruction::Add && "Invalid relation argument!");

  // If this is a relationship with a constant, make sure that this relationship
  // does not contradict properties known about the bounds of the constant.
  //
  if (ConstantInt *C = dyn_cast<ConstantInt>(Val))
    if (Op >= ICmpInst::FIRST_ICMP_PREDICATE && 
        Op <= ICmpInst::LAST_ICMP_PREDICATE) {
      ICmpInst::Predicate ipred = ICmpInst::Predicate(Op);
      if (ICmpInst::makeConstantRange(ipred, C->getValue())
                    .intersectWith(VI.getBounds()).isEmptySet())
        return true;
    }

  switch (Rel) {
  default: assert(0 && "Unknown Relationship code!");
  case Instruction::Add: return false;  // Nothing known, nothing contradicts
  case ICmpInst::ICMP_EQ:
    return Op == ICmpInst::ICMP_ULT || Op == ICmpInst::ICMP_SLT ||
           Op == ICmpInst::ICMP_UGT || Op == ICmpInst::ICMP_SGT ||
           Op == ICmpInst::ICMP_NE;
  case ICmpInst::ICMP_NE:  return Op == ICmpInst::ICMP_EQ;
  case ICmpInst::ICMP_ULE:
  case ICmpInst::ICMP_SLE: return Op == ICmpInst::ICMP_UGT ||
                                  Op == ICmpInst::ICMP_SGT;
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_SGE: return Op == ICmpInst::ICMP_ULT ||
                                  Op == ICmpInst::ICMP_SLT;
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLT:
    return Op == ICmpInst::ICMP_EQ  || Op == ICmpInst::ICMP_UGT ||
           Op == ICmpInst::ICMP_SGT || Op == ICmpInst::ICMP_UGE ||
           Op == ICmpInst::ICMP_SGE;
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGT:
    return Op == ICmpInst::ICMP_EQ  || Op == ICmpInst::ICMP_ULT ||
           Op == ICmpInst::ICMP_SLT || Op == ICmpInst::ICMP_ULE ||
           Op == ICmpInst::ICMP_SLE;
  case FCmpInst::FCMP_OEQ:
    return Op == FCmpInst::FCMP_OLT || Op == FCmpInst::FCMP_OGT ||
           Op == FCmpInst::FCMP_ONE;
  case FCmpInst::FCMP_ONE: return Op == FCmpInst::FCMP_OEQ;
  case FCmpInst::FCMP_OLE: return Op == FCmpInst::FCMP_OGT;
  case FCmpInst::FCMP_OGE: return Op == FCmpInst::FCMP_OLT;
  case FCmpInst::FCMP_OLT:
    return Op == FCmpInst::FCMP_OEQ || Op == FCmpInst::FCMP_OGT ||
           Op == FCmpInst::FCMP_OGE;
  case FCmpInst::FCMP_OGT:
    return Op == FCmpInst::FCMP_OEQ || Op == FCmpInst::FCMP_OLT ||
           Op == FCmpInst::FCMP_OLE;
  }
}

// incorporate - Incorporate information in the argument into this relation
// entry.  This assumes that the information doesn't contradict itself.  If any
// new information is gained, true is returned, otherwise false is returned to
// indicate that nothing was updated.
//
bool Relation::incorporate(unsigned Op, ValueInfo &VI) {
  assert(!contradicts(Op, VI) &&
         "Cannot incorporate contradictory information!");

  // If this is a relationship with a constant, make sure that we update the
  // range that is possible for the value to have...
  //
  if (ConstantInt *C = dyn_cast<ConstantInt>(Val))
    if (Op >= ICmpInst::FIRST_ICMP_PREDICATE && 
        Op <= ICmpInst::LAST_ICMP_PREDICATE) {
      ICmpInst::Predicate ipred = ICmpInst::Predicate(Op);
      VI.getBounds() = 
        ICmpInst::makeConstantRange(ipred, C->getValue())
                  .intersectWith(VI.getBounds());
    }

  switch (Rel) {
  default: assert(0 && "Unknown prior value!");
  case Instruction::Add:   Rel = Op; return true;
  case ICmpInst::ICMP_EQ:
  case ICmpInst::ICMP_NE:
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGT: return false;  // Nothing is more precise
  case ICmpInst::ICMP_ULE:
  case ICmpInst::ICMP_SLE:
    if (Op == ICmpInst::ICMP_EQ  || Op == ICmpInst::ICMP_ULT ||
        Op == ICmpInst::ICMP_SLT) {
      Rel = Op;
      return true;
    } else if (Op == ICmpInst::ICMP_NE) {
      Rel = Rel == ICmpInst::ICMP_ULE ? ICmpInst::ICMP_ULT :
            ICmpInst::ICMP_SLT;
      return true;
    }
    return false;
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_SGE:
    if (Op == ICmpInst::ICMP_EQ  || ICmpInst::ICMP_UGT ||
        Op == ICmpInst::ICMP_SGT) {
      Rel = Op;
      return true;
    } else if (Op == ICmpInst::ICMP_NE) {
      Rel = Rel == ICmpInst::ICMP_UGE ? ICmpInst::ICMP_UGT :
            ICmpInst::ICMP_SGT;
      return true;
    }
    return false;
  case FCmpInst::FCMP_OEQ: return false;  // Nothing is more precise
  case FCmpInst::FCMP_ONE: return false;  // Nothing is more precise
  case FCmpInst::FCMP_OLT: return false;  // Nothing is more precise
  case FCmpInst::FCMP_OGT: return false;  // Nothing is more precise
  case FCmpInst::FCMP_OLE:
    if (Op == FCmpInst::FCMP_OEQ || Op == FCmpInst::FCMP_OLT) {
      Rel = Op;
      return true;
    } else if (Op == FCmpInst::FCMP_ONE) {
      Rel = FCmpInst::FCMP_OLT;
      return true;
    }
    return false;
  case FCmpInst::FCMP_OGE: 
    return Op == FCmpInst::FCMP_OLT;
    if (Op == FCmpInst::FCMP_OEQ || Op == FCmpInst::FCMP_OGT) {
      Rel = Op;
      return true;
    } else if (Op == FCmpInst::FCMP_ONE) {
      Rel = FCmpInst::FCMP_OGT;
      return true;
    }
    return false;
  }
}

// getImpliedResult - If this relationship between two values implies that
// the specified relationship is true or false, return that.  If we cannot
// determine the result required, return Unknown.
//
Relation::KnownResult
Relation::getImpliedResult(unsigned Op) const {
  if (Rel == Op) return KnownTrue;
  if (Op >= ICmpInst::FIRST_ICMP_PREDICATE && 
      Op <= ICmpInst::LAST_ICMP_PREDICATE) {
    if (Rel == unsigned(ICmpInst::getInversePredicate(ICmpInst::Predicate(Op))))
      return KnownFalse;
  } else if (Op <= FCmpInst::LAST_FCMP_PREDICATE) {
    if (Rel == unsigned(FCmpInst::getInversePredicate(FCmpInst::Predicate(Op))))
    return KnownFalse;
  }

  switch (Rel) {
  default: assert(0 && "Unknown prior value!");
  case ICmpInst::ICMP_EQ:
    if (Op == ICmpInst::ICMP_ULE || Op == ICmpInst::ICMP_SLE || 
        Op == ICmpInst::ICMP_UGE || Op == ICmpInst::ICMP_SGE) return KnownTrue;
    if (Op == ICmpInst::ICMP_ULT || Op == ICmpInst::ICMP_SLT || 
        Op == ICmpInst::ICMP_UGT || Op == ICmpInst::ICMP_SGT) return KnownFalse;
    break;
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLT:
    if (Op == ICmpInst::ICMP_ULE || Op == ICmpInst::ICMP_SLE ||
        Op == ICmpInst::ICMP_NE) return KnownTrue;
    if (Op == ICmpInst::ICMP_EQ) return KnownFalse;
    break;
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGT:
    if (Op == ICmpInst::ICMP_UGE || Op == ICmpInst::ICMP_SGE ||
        Op == ICmpInst::ICMP_NE) return KnownTrue;
    if (Op == ICmpInst::ICMP_EQ) return KnownFalse;
    break;
  case FCmpInst::FCMP_OEQ:
    if (Op == FCmpInst::FCMP_OLE || Op == FCmpInst::FCMP_OGE) return KnownTrue;
    if (Op == FCmpInst::FCMP_OLT || Op == FCmpInst::FCMP_OGT) return KnownFalse;
    break;
  case FCmpInst::FCMP_OLT:
    if (Op == FCmpInst::FCMP_ONE || Op == FCmpInst::FCMP_OLE) return KnownTrue;
    if (Op == FCmpInst::FCMP_OEQ) return KnownFalse;
    break;
  case FCmpInst::FCMP_OGT:
    if (Op == FCmpInst::FCMP_ONE || Op == FCmpInst::FCMP_OGE) return KnownTrue;
    if (Op == FCmpInst::FCMP_OEQ) return KnownFalse;
    break;
  case ICmpInst::ICMP_NE:
  case ICmpInst::ICMP_SLE:
  case ICmpInst::ICMP_ULE:
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_SGE:
  case FCmpInst::FCMP_ONE:
  case FCmpInst::FCMP_OLE:
  case FCmpInst::FCMP_OGE:
  case FCmpInst::FCMP_FALSE:
  case FCmpInst::FCMP_ORD:
  case FCmpInst::FCMP_UNO:
  case FCmpInst::FCMP_UEQ:
  case FCmpInst::FCMP_UGT:
  case FCmpInst::FCMP_UGE:
  case FCmpInst::FCMP_ULT:
  case FCmpInst::FCMP_ULE:
  case FCmpInst::FCMP_UNE:
  case FCmpInst::FCMP_TRUE:
    break;
  }
  return Unknown;
}


//===----------------------------------------------------------------------===//
// Printing Support...
//===----------------------------------------------------------------------===//

// print - Implement the standard print form to print out analysis information.
void CEE::print(std::ostream &O, const Module *M) const {
  O << "\nPrinting Correlated Expression Info:\n";
  for (std::map<BasicBlock*, RegionInfo>::const_iterator I =
         RegionInfoMap.begin(), E = RegionInfoMap.end(); I != E; ++I)
    I->second.print(O);
}

// print - Output information about this region...
void RegionInfo::print(std::ostream &OS) const {
  if (ValueMap.empty()) return;

  OS << " RegionInfo for basic block: " << BB->getName() << "\n";
  for (std::map<Value*, ValueInfo>::const_iterator
         I = ValueMap.begin(), E = ValueMap.end(); I != E; ++I)
    I->second.print(OS, I->first);
  OS << "\n";
}

// print - Output information about this value relation...
void ValueInfo::print(std::ostream &OS, Value *V) const {
  if (Relationships.empty()) return;

  if (V) {
    OS << "  ValueInfo for: ";
    WriteAsOperand(OS, V);
  }
  OS << "\n    Bounds = " << Bounds << "\n";
  if (Replacement) {
    OS << "    Replacement = ";
    WriteAsOperand(OS, Replacement);
    OS << "\n";
  }
  for (unsigned i = 0, e = Relationships.size(); i != e; ++i)
    Relationships[i].print(OS);
}

// print - Output this relation to the specified stream
void Relation::print(std::ostream &OS) const {
  OS << "    is ";
  switch (Rel) {
  default:           OS << "*UNKNOWN*"; break;
  case ICmpInst::ICMP_EQ:
  case FCmpInst::FCMP_ORD:
  case FCmpInst::FCMP_UEQ:
  case FCmpInst::FCMP_OEQ: OS << "== "; break;
  case ICmpInst::ICMP_NE:
  case FCmpInst::FCMP_UNO:
  case FCmpInst::FCMP_UNE:
  case FCmpInst::FCMP_ONE: OS << "!= "; break;
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLT:
  case FCmpInst::FCMP_ULT:
  case FCmpInst::FCMP_OLT: OS << "< "; break;
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGT:
  case FCmpInst::FCMP_UGT:
  case FCmpInst::FCMP_OGT: OS << "> "; break;
  case ICmpInst::ICMP_ULE:
  case ICmpInst::ICMP_SLE:
  case FCmpInst::FCMP_ULE:
  case FCmpInst::FCMP_OLE: OS << "<= "; break;
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_SGE:
  case FCmpInst::FCMP_UGE:
  case FCmpInst::FCMP_OGE: OS << ">= "; break;
  }

  WriteAsOperand(OS, Val);
  OS << "\n";
}

// Don't inline these methods or else we won't be able to call them from GDB!
void Relation::dump() const { print(*cerr.stream()); }
void ValueInfo::dump() const { print(*cerr.stream(), 0); }
void RegionInfo::dump() const { print(*cerr.stream()); }
