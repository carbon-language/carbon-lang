//===- CorrelatedExprs.cpp - Pass to detect and eliminated c.e.'s ---------===//
//
// Correlated Expression Elimination propogates information from conditional
// branches to blocks dominated by destinations of the branch.  It propogates
// information from the condition check itself into the body of the branch,
// allowing transformations like these for example:
//
//  if (i == 7)
//    ... 4*i;  // constant propogation
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

#include "llvm/Transforms/Scalar.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/iOperators.h"
#include "llvm/ConstantHandling.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/CFG.h"
#include "Support/PostOrderIterator.h"
#include "Support/StatisticReporter.h"
#include <algorithm>

namespace {
  Statistic<>NumSetCCRemoved("cee\t\t- Number of setcc instruction eliminated");
  Statistic<>NumOperandsCann("cee\t\t- Number of operands cannonicalized");
  Statistic<>BranchRevectors("cee\t\t- Number of branches revectored");

  class ValueInfo;
  class Relation {
    Value *Val;                 // Relation to what value?
    Instruction::BinaryOps Rel; // SetCC relation, or Add if no information
  public:
    Relation(Value *V) : Val(V), Rel(Instruction::Add) {}
    bool operator<(const Relation &R) const { return Val < R.Val; }
    Value *getValue() const { return Val; }
    Instruction::BinaryOps getRelation() const { return Rel; }

    // contradicts - Return true if the relationship specified by the operand
    // contradicts already known information.
    //
    bool contradicts(Instruction::BinaryOps Rel, const ValueInfo &VI) const;

    // incorporate - Incorporate information in the argument into this relation
    // entry.  This assumes that the information doesn't contradict itself.  If
    // any new information is gained, true is returned, otherwise false is
    // returned to indicate that nothing was updated.
    //
    bool incorporate(Instruction::BinaryOps Rel, ValueInfo &VI);

    // KnownResult - Whether or not this condition determines the result of a
    // setcc in the program.  False & True are intentionally 0 & 1 so we can
    // convert to bool by casting after checking for unknown.
    //
    enum KnownResult { KnownFalse = 0, KnownTrue = 1, Unknown = 2 };

    // getImpliedResult - If this relationship between two values implies that
    // the specified relationship is true or false, return that.  If we cannot
    // determine the result required, return Unknown.
    //
    KnownResult getImpliedResult(Instruction::BinaryOps Rel) const;

    // print - Output this relation to the specified stream
    void print(std::ostream &OS) const;
    void dump() const;
  };


  // ValueInfo - One instance of this record exists for every value with
  // relationships between other values.  It keeps track of all of the
  // relationships to other values in the program (specified with Relation) that
  // are known to be valid in a region.
  //
  class ValueInfo {
    // RelationShips - this value is know to have the specified relationships to
    // other values.  There can only be one entry per value, and this list is
    // kept sorted by the Val field.
    std::vector<Relation> Relationships;

    // If information about this value is known or propogated from constant
    // expressions, this range contains the possible values this value may hold.
    ConstantRange Bounds;

    // If we find that this value is equal to another value that has a lower
    // rank, this value is used as it's replacement.
    //
    Value *Replacement;
  public:
    ValueInfo(const Type *Ty)
      : Bounds(Ty->isIntegral() ? Ty : Type::IntTy), Replacement(0) {}

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
    // This can invalidate references to other Relation's, so use it carefully.
    //
    Relation &getRelation(Value *V) {
      // Binary search for V's entry...
      std::vector<Relation>::iterator I =
        std::lower_bound(Relationships.begin(), Relationships.end(), V);

      // If we found the entry, return it...
      if (I != Relationships.end() && I->getValue() == V)
        return *I;

      // Insert and return the new relationship...
      return *Relationships.insert(I, V);
    }

    const Relation *requestRelation(Value *V) const {
      // Binary search for V's entry...
      std::vector<Relation>::const_iterator I =
        std::lower_bound(Relationships.begin(), Relationships.end(), V);
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
  class RegionInfo {
    BasicBlock *BB;

    // ValueMap - Tracks the ValueInformation known for this region
    typedef std::map<Value*, ValueInfo> ValueMapTy;
    ValueMapTy ValueMap;
  public:
    RegionInfo(BasicBlock *bb) : BB(bb) {}

    // getEntryBlock - Return the block that dominates all of the members of
    // this region.
    BasicBlock *getEntryBlock() const { return BB; }

    const RegionInfo &operator=(const RegionInfo &RI) {
      ValueMap = RI.ValueMap;
      return *this;
    }

    // print - Output information about this region...
    void print(std::ostream &OS) const;

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
  };

  /// CEE - Correlated Expression Elimination
  class CEE : public FunctionPass {
    std::map<Value*, unsigned> RankMap;
    std::map<BasicBlock*, RegionInfo> RegionInfoMap;
    DominatorSet *DS;
    DominatorTree *DT;
  public:
    virtual bool runOnFunction(Function &F);

    // We don't modify the program, so we preserve all analyses
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorSet>();
      AU.addRequired<DominatorTree>();
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
      if (isa<Constant>(V) || isa<GlobalValue>(V)) return 0;
      std::map<Value*, unsigned>::const_iterator I = RankMap.find(V);
      if (I != RankMap.end()) return I->second;
      return 0; // Must be some other global thing
    }

    bool TransformRegion(BasicBlock *BB, std::set<BasicBlock*> &VisitedBlocks);

    BasicBlock *isCorrelatedBranchBlock(BasicBlock *BB, RegionInfo &RI);
    void PropogateBranchInfo(BranchInst *BI);
    void PropogateEquality(Value *Op0, Value *Op1, RegionInfo &RI);
    void PropogateRelation(Instruction::BinaryOps Opcode, Value *Op0,
                           Value *Op1, RegionInfo &RI);
    void UpdateUsersOfValue(Value *V, RegionInfo &RI);
    void IncorporateInstruction(Instruction *Inst, RegionInfo &RI);
    void ComputeReplacements(RegionInfo &RI);


    // getSetCCResult - Given a setcc instruction, determine if the result is
    // determined by facts we already know about the region under analysis.
    // Return KnownTrue, KnownFalse, or Unknown based on what we can determine.
    //
    Relation::KnownResult getSetCCResult(SetCondInst *SC, const RegionInfo &RI);


    bool SimplifyBasicBlock(BasicBlock &BB, const RegionInfo &RI);
    bool SimplifyInstruction(Instruction *Inst, const RegionInfo &RI);
  }; 
  RegisterOpt<CEE> X("cee", "Correlated Expression Elimination");
}

Pass *createCorrelatedExpressionEliminationPass() { return new CEE(); }


bool CEE::runOnFunction(Function &F) {
  // Build a rank map for the function...
  BuildRankMap(F);

  // Traverse the dominator tree, computing information for each node in the
  // tree.  Note that our traversal will not even touch unreachable basic
  // blocks.
  DS = &getAnalysis<DominatorSet>();
  DT = &getAnalysis<DominatorTree>();
  
  std::set<BasicBlock*> VisitedBlocks;
  bool Changed = TransformRegion(&F.getEntryNode(), VisitedBlocks);

  RegionInfoMap.clear();
  RankMap.clear();
  return Changed;
}

// TransformRegion - Transform the region starting with BB according to the
// calculated region information for the block.  Transforming the region
// involves analyzing any information this block provides to successors,
// propogating the information to successors, and finally transforming
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
  DEBUG(RI.print(std::cerr));

  // Simplify the contents of this block...
  bool Changed = SimplifyBasicBlock(*BB, RI);

  // Get the terminator of this basic block...
  TerminatorInst *TI = BB->getTerminator();

  // Loop over all of the blocks that this block is the immediate dominator for.
  // Because all information known in this region is also known in all of the
  // blocks that are dominated by this one, we can safely propogate the
  // information down now.
  //
  DominatorTree::Node *BBN = (*DT)[BB];
  for (unsigned i = 0, e = BBN->getChildren().size(); i != e; ++i) {
    BasicBlock *Dominated = BBN->getChildren()[i]->getNode();
    assert(RegionInfoMap.find(Dominated) == RegionInfoMap.end() &&
           "RegionInfo should be calculated in dominanace order!");
    getRegionInfo(Dominated) = RI;
  }

  // Now that all of our successors have information if they deserve it,
  // propogate any information our terminator instruction finds to our
  // successors.
  if (BranchInst *BI = dyn_cast<BranchInst>(TI))
    if (BI->isConditional())
      PropogateBranchInfo(BI);

  // If this is a branch to a block outside our region that simply performs
  // another conditional branch, one whose outcome is known inside of this
  // region, then vector this outgoing edge directly to the known destination.
  //
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
    while (BasicBlock *Dest = isCorrelatedBranchBlock(TI->getSuccessor(i), RI)){
      // If there are any PHI nodes in the Dest BB, we must duplicate the entry
      // in the PHI node for the old successor to now include an entry from the
      // current basic block.
      //
      BasicBlock *OldSucc = TI->getSuccessor(i);

      // Loop over all of the PHI nodes...
      for (BasicBlock::iterator I = Dest->begin();
           PHINode *PN = dyn_cast<PHINode>(&*I); ++I) {
        // Find the entry in the PHI node for OldSucc, create a duplicate entry
        // for BB now.
        int BlockIndex = PN->getBasicBlockIndex(OldSucc);
        assert(BlockIndex != -1 && "Block should have entry in PHI!");
        PN->addIncoming(PN->getIncomingValue(BlockIndex), BB);
      }

      // Actually revector the branch now...
      TI->setSuccessor(i, Dest);
      ++BranchRevectors;
      Changed = true;
    }

  // Now that all of our successors have information, recursively process them.
  for (unsigned i = 0, e = BBN->getChildren().size(); i != e; ++i)
    Changed |= TransformRegion(BBN->getChildren()[i]->getNode(), VisitedBlocks);

  return Changed;
}

// If this block is a simple block not in the current region, which contains
// only a conditional branch, we determine if the outcome of the branch can be
// determined from information inside of the region.  Instead of going to this
// block, we can instead go to the destination we know is the right target.
//
BasicBlock *CEE::isCorrelatedBranchBlock(BasicBlock *BB, RegionInfo &RI) {
  // Check to see if we dominate the block. If so, this block will get the
  // condition turned to a constant anyway.
  //
  //if (DS->dominates(RI.getEntryBlock(), BB))
  // return 0;

  // Check to see if this is a conditional branch...
  if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator()))
    if (BI->isConditional()) {
      // Make sure that the block is either empty, or only contains a setcc.
      if (BB->size() == 1 || 
          (BB->size() == 2 && &BB->front() == BI->getCondition() &&
           BI->getCondition()->use_size() == 1))
        if (SetCondInst *SCI = dyn_cast<SetCondInst>(BI->getCondition())) {
          Relation::KnownResult Result = getSetCCResult(SCI, RI);
        
          if (Result == Relation::KnownTrue)
            return BI->getSuccessor(0);
          else if (Result == Relation::KnownFalse)
            return BI->getSuccessor(1);
        }
    }
  return 0;
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
  for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
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


// PropogateBranchInfo - When this method is invoked, we need to propogate
// information derived from the branch condition into the true and false
// branches of BI.  Since we know that there aren't any critical edges in the
// flow graph, this can proceed unconditionally.
//
void CEE::PropogateBranchInfo(BranchInst *BI) {
  assert(BI->isConditional() && "Must be a conditional branch!");
  BasicBlock *BB = BI->getParent();
  BasicBlock *TrueBB  = BI->getSuccessor(0);
  BasicBlock *FalseBB = BI->getSuccessor(1);

  // Propogate information into the true block...
  //
  PropogateEquality(BI->getCondition(), ConstantBool::True,
                    getRegionInfo(TrueBB));
  
  // Propogate information into the false block...
  //
  PropogateEquality(BI->getCondition(), ConstantBool::False,
                    getRegionInfo(FalseBB));
}


// PropogateEquality - If we discover that two values are equal to each other in
// a specified region, propogate this knowledge recursively.
//
void CEE::PropogateEquality(Value *Op0, Value *Op1, RegionInfo &RI) {
  if (Op0 == Op1) return;  // Gee whiz. Are these really equal each other?

  if (isa<Constant>(Op0))  // Make sure the constant is always Op1
    std::swap(Op0, Op1);

  // Make sure we don't already know these are equal, to avoid infinite loops...
  ValueInfo &VI = RI.getValueInfo(Op0);

  // Get information about the known relationship between Op0 & Op1
  Relation &KnownRelation = VI.getRelation(Op1);

  // If we already know they're equal, don't reprocess...
  if (KnownRelation.getRelation() == Instruction::SetEQ)
    return;

  // If this is boolean, check to see if one of the operands is a constant.  If
  // it's a constant, then see if the other one is one of a setcc instruction,
  // an AND, OR, or XOR instruction.
  //
  if (ConstantBool *CB = dyn_cast<ConstantBool>(Op1)) {

    if (Instruction *Inst = dyn_cast<Instruction>(Op0)) {
      // If we know that this instruction is an AND instruction, and the result
      // is true, this means that both operands to the OR are known to be true
      // as well.
      //
      if (CB->getValue() && Inst->getOpcode() == Instruction::And) {
        PropogateEquality(Inst->getOperand(0), CB, RI);
        PropogateEquality(Inst->getOperand(1), CB, RI);
      }
      
      // If we know that this instruction is an OR instruction, and the result
      // is false, this means that both operands to the OR are know to be false
      // as well.
      //
      if (!CB->getValue() && Inst->getOpcode() == Instruction::Or) {
        PropogateEquality(Inst->getOperand(0), CB, RI);
        PropogateEquality(Inst->getOperand(1), CB, RI);
      }
      
      // If we know that this instruction is a NOT instruction, we know that the
      // operand is known to be the inverse of whatever the current value is.
      //
      if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(Inst))
        if (BinaryOperator::isNot(BOp))
          PropogateEquality(BinaryOperator::getNotArgument(BOp),
                            ConstantBool::get(!CB->getValue()), RI);

      // If we know the value of a SetCC instruction, propogate the information
      // about the relation into this region as well.
      //
      if (SetCondInst *SCI = dyn_cast<SetCondInst>(Inst)) {
        if (CB->getValue()) {  // If we know the condition is true...
          // Propogate info about the LHS to the RHS & RHS to LHS
          PropogateRelation(SCI->getOpcode(), SCI->getOperand(0),
                            SCI->getOperand(1), RI);
          PropogateRelation(SCI->getSwappedCondition(),
                            SCI->getOperand(1), SCI->getOperand(0), RI);

        } else {               // If we know the condition is false...
          // We know the opposite of the condition is true...
          Instruction::BinaryOps C = SCI->getInverseCondition();
          
          PropogateRelation(C, SCI->getOperand(0), SCI->getOperand(1), RI);
          PropogateRelation(SetCondInst::getSwappedCondition(C),
                            SCI->getOperand(1), SCI->getOperand(0), RI);
        }
      }
    }
  }

  // Propogate information about Op0 to Op1 & visa versa
  PropogateRelation(Instruction::SetEQ, Op0, Op1, RI);
  PropogateRelation(Instruction::SetEQ, Op1, Op0, RI);
}


// PropogateRelation - We know that the specified relation is true in all of the
// blocks in the specified region.  Propogate the information about Op0 and
// anything derived from it into this region.
//
void CEE::PropogateRelation(Instruction::BinaryOps Opcode, Value *Op0,
                            Value *Op1, RegionInfo &RI) {
  assert(Op0->getType() == Op1->getType() && "Equal types expected!");

  // Constants are already pretty well understood.  We will apply information
  // about the constant to Op1 in another call to PropogateRelation.
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
  // are propogating, ignore this info.  Something bad must have happened!
  //
  if (Op1R.contradicts(Opcode, VI)) {
    Op1R.contradicts(Opcode, VI);
    std::cerr << "Contradiction found for opcode: "
              << Instruction::getOpcodeName(Opcode) << "\n";
    Op1R.print(std::cerr);
    return;
  }

  // If the information propogted is new, then we want process the uses of this
  // instruction to propogate the information down to them.
  //
  if (Op1R.incorporate(Opcode, VI))
    UpdateUsersOfValue(Op0, RI);
}


// UpdateUsersOfValue - The information about V in this region has been updated.
// Propogate this to all consumers of the value.
//
void CEE::UpdateUsersOfValue(Value *V, RegionInfo &RI) {
  for (Value::use_iterator I = V->use_begin(), E = V->use_end();
       I != E; ++I)
    if (Instruction *Inst = dyn_cast<Instruction>(*I)) {
      // If this is an instruction using a value that we know something about,
      // try to propogate information to the value produced by the
      // instruction.  We can only do this if it is an instruction we can
      // propogate information for (a setcc for example), and we only WANT to
      // do this if the instruction dominates this region.
      //
      // If the instruction doesn't dominate this region, then it cannot be
      // used in this region and we don't care about it.  If the instruction
      // is IN this region, then we will simplify the instruction before we
      // get to uses of it anyway, so there is no reason to bother with it
      // here.  This check is also effectively checking to make sure that Inst
      // is in the same function as our region (in case V is a global f.e.).
      //
      if (DS->properlyDominates(Inst->getParent(), RI.getEntryBlock()))
        IncorporateInstruction(Inst, RI);
    }
}

// IncorporateInstruction - We just updated the information about one of the
// operands to the specified instruction.  Update the information about the
// value produced by this instruction
//
void CEE::IncorporateInstruction(Instruction *Inst, RegionInfo &RI) {
  if (SetCondInst *SCI = dyn_cast<SetCondInst>(Inst)) {
    // See if we can figure out a result for this instruction...
    Relation::KnownResult Result = getSetCCResult(SCI, RI);
    if (Result != Relation::Unknown) {
      PropogateEquality(SCI, Result ? ConstantBool::True : ConstantBool::False,
                        RI);
    }
  }
}


// ComputeReplacements - Some values are known to be equal to other values in a
// region.  For example if there is a comparison of equality between a variable
// X and a constant C, we can replace all uses of X with C in the region we are
// interested in.  We generalize this replacement to replace variables with
// other variables if they are equal and there is a variable with lower rank
// than the current one.  This offers a cannonicalizing property that exposes
// more redundancies for later transformations to take advantage of.
//
void CEE::ComputeReplacements(RegionInfo &RI) {
  // Loop over all of the values in the region info map...
  for (RegionInfo::iterator I = RI.begin(), E = RI.end(); I != E; ++I) {
    ValueInfo &VI = I->second;

    // If we know that this value is a particular constant, set Replacement to
    // the constant...
    Value *Replacement = VI.getBounds().getSingleElement();

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
        if (Relationships[i].getRelation() == Instruction::SetEQ) {
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
    Instruction *Inst = &*I++;

    // Convert instruction arguments to canonical forms...
    Changed |= SimplifyInstruction(Inst, RI);

    if (SetCondInst *SCI = dyn_cast<SetCondInst>(Inst)) {
      // Try to simplify a setcc instruction based on inherited information
      Relation::KnownResult Result = getSetCCResult(SCI, RI);
      if (Result != Relation::Unknown) {
        DEBUG(std::cerr << "Replacing setcc with " << Result
                        << " constant: " << SCI);

        SCI->replaceAllUsesWith(ConstantBool::get((bool)Result));
        // The instruction is now dead, remove it from the program.
        SCI->getParent()->getInstList().erase(SCI);
        ++NumSetCCRemoved;
        Changed = true;
      }
    }
  }

  return Changed;
}

// SimplifyInstruction - Inspect the operands of the instruction, converting
// them to their cannonical form if possible.  This takes care of, for example,
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
        DEBUG(std::cerr << "In Inst: " << I << "  Replacing operand #" << i
                        << " with " << Repl << "\n");
        I->setOperand(i, Repl);
        Changed = true;
        ++NumOperandsCann;
      }

  return Changed;
}


// SimplifySetCC - Try to simplify a setcc instruction based on information
// inherited from a dominating setcc instruction.  V is one of the operands to
// the setcc instruction, and VI is the set of information known about it.  We
// take two cases into consideration here.  If the comparison is against a
// constant value, we can use the constant range to see if the comparison is
// possible to succeed.  If it is not a comparison against a constant, we check
// to see if there is a known relationship between the two values.  If so, we
// may be able to eliminate the check.
//
Relation::KnownResult CEE::getSetCCResult(SetCondInst *SCI,
                                          const RegionInfo &RI) {
  Value *Op0 = SCI->getOperand(0), *Op1 = SCI->getOperand(1);
  Instruction::BinaryOps Opcode = SCI->getOpcode();
  
  if (isa<Constant>(Op0)) {
    if (isa<Constant>(Op1)) {
      if (Constant *Result = ConstantFoldInstruction(SCI)) {
        // Wow, this is easy, directly eliminate the SetCondInst.
        DEBUG(std::cerr << "Replacing setcc with constant fold: " << SCI);
        return cast<ConstantBool>(Result)->getValue()
          ? Relation::KnownTrue : Relation::KnownFalse;
      }
    } else {
      // We want to swap this instruction so that operand #0 is the constant.
      std::swap(Op0, Op1);
      Opcode = SCI->getSwappedCondition();
    }
  }

  // Try to figure out what the result of this comparison will be...
  Relation::KnownResult Result = Relation::Unknown;

  // We have to know something about the relationship to prove anything...
  if (const ValueInfo *Op0VI = RI.requestValueInfo(Op0)) {

    // At this point, we know that if we have a constant argument that it is in
    // Op1.  Check to see if we know anything about comparing value with a
    // constant, and if we can use this info to fold the setcc.
    //
    if (ConstantIntegral *C = dyn_cast<ConstantIntegral>(Op1)) {
      // Check to see if we already know the result of this comparison...
      ConstantRange R = ConstantRange(Opcode, C);
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
        Result = Op2R->getImpliedResult(Opcode);
    }
  }
  return Result;
}

//===----------------------------------------------------------------------===//
//  Relation Implementation
//===----------------------------------------------------------------------===//

// CheckCondition - Return true if the specified condition is false.  Bound may
// be null.
static bool CheckCondition(Constant *Bound, Constant *C,
                           Instruction::BinaryOps BO) {
  assert(C != 0 && "C is not specified!");
  if (Bound == 0) return false;

  ConstantBool *Val;
  switch (BO) {
  default: assert(0 && "Unknown Condition code!");
  case Instruction::SetEQ: Val = *Bound == *C; break;
  case Instruction::SetNE: Val = *Bound != *C; break;
  case Instruction::SetLT: Val = *Bound <  *C; break;
  case Instruction::SetGT: Val = *Bound >  *C; break;
  case Instruction::SetLE: Val = *Bound <= *C; break;
  case Instruction::SetGE: Val = *Bound >= *C; break;
  }

  // ConstantHandling code may not succeed in the comparison...
  if (Val == 0) return false;
  return !Val->getValue();  // Return true if the condition is false...
}

// contradicts - Return true if the relationship specified by the operand
// contradicts already known information.
//
bool Relation::contradicts(Instruction::BinaryOps Op,
                           const ValueInfo &VI) const {
  assert (Op != Instruction::Add && "Invalid relation argument!");

  // If this is a relationship with a constant, make sure that this relationship
  // does not contradict properties known about the bounds of the constant.
  //
  if (ConstantIntegral *C = dyn_cast<ConstantIntegral>(Val))
    if (ConstantRange(Op, C).intersectWith(VI.getBounds()).isEmptySet())
      return true;

  switch (Rel) {
  default: assert(0 && "Unknown Relationship code!");
  case Instruction::Add: return false;  // Nothing known, nothing contradicts
  case Instruction::SetEQ:
    return Op == Instruction::SetLT || Op == Instruction::SetGT ||
           Op == Instruction::SetNE;
  case Instruction::SetNE: return Op == Instruction::SetEQ;
  case Instruction::SetLE: return Op == Instruction::SetGT;
  case Instruction::SetGE: return Op == Instruction::SetLT;
  case Instruction::SetLT:
    return Op == Instruction::SetEQ || Op == Instruction::SetGT ||
           Op == Instruction::SetGE;
  case Instruction::SetGT:
    return Op == Instruction::SetEQ || Op == Instruction::SetLT ||
           Op == Instruction::SetLE;
  }
}

// incorporate - Incorporate information in the argument into this relation
// entry.  This assumes that the information doesn't contradict itself.  If any
// new information is gained, true is returned, otherwise false is returned to
// indicate that nothing was updated.
//
bool Relation::incorporate(Instruction::BinaryOps Op, ValueInfo &VI) {
  assert(!contradicts(Op, VI) &&
         "Cannot incorporate contradictory information!");

  // If this is a relationship with a constant, make sure that we update the
  // range that is possible for the value to have...
  //
  if (ConstantIntegral *C = dyn_cast<ConstantIntegral>(Val))
    VI.getBounds() = ConstantRange(Op, C).intersectWith(VI.getBounds());

  switch (Rel) {
  default: assert(0 && "Unknown prior value!");
  case Instruction::Add:   Rel = Op; return true;
  case Instruction::SetEQ: return false;  // Nothing is more precise
  case Instruction::SetNE: return false;  // Nothing is more precise
  case Instruction::SetLT: return false;  // Nothing is more precise
  case Instruction::SetGT: return false;  // Nothing is more precise
  case Instruction::SetLE:
    if (Op == Instruction::SetEQ || Op == Instruction::SetLT) {
      Rel = Op;
      return true;
    } else if (Op == Instruction::SetNE) {
      Rel = Instruction::SetLT;
      return true;
    }
    return false;
  case Instruction::SetGE: return Op == Instruction::SetLT;
    if (Op == Instruction::SetEQ || Op == Instruction::SetGT) {
      Rel = Op;
      return true;
    } else if (Op == Instruction::SetNE) {
      Rel = Instruction::SetGT;
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
Relation::getImpliedResult(Instruction::BinaryOps Op) const {
  if (Rel == Op) return KnownTrue;
  if (Rel == SetCondInst::getInverseCondition(Op)) return KnownFalse;

  switch (Rel) {
  default: assert(0 && "Unknown prior value!");
  case Instruction::SetEQ:
    if (Op == Instruction::SetLE || Op == Instruction::SetGE) return KnownTrue;
    if (Op == Instruction::SetLT || Op == Instruction::SetGT) return KnownFalse;
    break;
  case Instruction::SetLT:
    if (Op == Instruction::SetNE || Op == Instruction::SetLE) return KnownTrue;
    if (Op == Instruction::SetEQ) return KnownFalse;
    break;
  case Instruction::SetGT:
    if (Op == Instruction::SetNE || Op == Instruction::SetGE) return KnownTrue;
    if (Op == Instruction::SetEQ) return KnownFalse;
    break;
  case Instruction::SetNE:
  case Instruction::SetLE:
  case Instruction::SetGE:
  case Instruction::Add:
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
  case Instruction::SetEQ: OS << "== "; break;
  case Instruction::SetNE: OS << "!= "; break;
  case Instruction::SetLT: OS << "< "; break;
  case Instruction::SetGT: OS << "> "; break;
  case Instruction::SetLE: OS << "<= "; break;
  case Instruction::SetGE: OS << ">= "; break;
  }

  WriteAsOperand(OS, Val);
  OS << "\n";
}

void Relation::dump() const { print(std::cerr); }
void ValueInfo::dump() const { print(std::cerr, 0); }
