//===-- AMDGPUStructurizeCFG.cpp -  ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// The pass implemented in this file transforms the programs control flow
/// graph into a form that's suitable for code generation on hardware that
/// implements control flow by execution masking. This currently includes all
/// AMD GPUs but may as well be useful for other types of hardware.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/Module.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

using namespace llvm;

namespace {

// Definition of the complex types used in this pass.

typedef std::pair<BasicBlock *, Value *> BBValuePair;
typedef ArrayRef<BasicBlock*> BBVecRef;

typedef SmallVector<RegionNode*, 8> RNVector;
typedef SmallVector<BasicBlock*, 8> BBVector;
typedef SmallVector<BBValuePair, 2> BBValueVector;

typedef DenseMap<PHINode *, BBValueVector> PhiMap;
typedef DenseMap<BasicBlock *, PhiMap> BBPhiMap;
typedef DenseMap<BasicBlock *, Value *> BBPredicates;
typedef DenseMap<BasicBlock *, BBPredicates> PredMap;
typedef DenseMap<BasicBlock *, unsigned> VisitedMap;

// The name for newly created blocks.

static const char *FlowBlockName = "Flow";

/// @brief Transforms the control flow graph on one single entry/exit region
/// at a time.
///
/// After the transform all "If"/"Then"/"Else" style control flow looks like
/// this:
///
/// \verbatim
/// 1
/// ||
/// | |
/// 2 |
/// | /
/// |/   
/// 3
/// ||   Where:
/// | |  1 = "If" block, calculates the condition
/// 4 |  2 = "Then" subregion, runs if the condition is true
/// | /  3 = "Flow" blocks, newly inserted flow blocks, rejoins the flow
/// |/   4 = "Else" optional subregion, runs if the condition is false
/// 5    5 = "End" block, also rejoins the control flow
/// \endverbatim
///
/// Control flow is expressed as a branch where the true exit goes into the
/// "Then"/"Else" region, while the false exit skips the region
/// The condition for the optional "Else" region is expressed as a PHI node.
/// The incomming values of the PHI node are true for the "If" edge and false
/// for the "Then" edge.
///
/// Additionally to that even complicated loops look like this:
///
/// \verbatim
/// 1
/// ||
/// | |
/// 2 ^  Where:
/// | /  1 = "Entry" block
/// |/   2 = "Loop" optional subregion, with all exits at "Flow" block
/// 3    3 = "Flow" block, with back edge to entry block
/// |
/// \endverbatim
///
/// The back edge of the "Flow" block is always on the false side of the branch
/// while the true side continues the general flow. So the loop condition
/// consist of a network of PHI nodes where the true incoming values expresses
/// breaks and the false values expresses continue states.
class AMDGPUStructurizeCFG : public RegionPass {

  static char ID;

  Type *Boolean;
  ConstantInt *BoolTrue;
  ConstantInt *BoolFalse;
  UndefValue *BoolUndef;

  Function *Func;
  Region *ParentRegion;

  DominatorTree *DT;

  RNVector Order;
  VisitedMap Visited;
  PredMap Predicates;
  BBPhiMap DeletedPhis;
  BBVector FlowsInserted;

  BasicBlock *LoopStart;
  BasicBlock *LoopEnd;
  BBPredicates LoopPred;

  void orderNodes();

  void buildPredicate(BranchInst *Term, unsigned Idx,
                      BBPredicates &Pred, bool Invert);

  void analyzeBlock(BasicBlock *BB);

  void analyzeLoop(BasicBlock *BB, unsigned &LoopIdx);

  void collectInfos();

  bool dominatesPredicates(BasicBlock *A, BasicBlock *B);

  void killTerminator(BasicBlock *BB);

  RegionNode *skipChained(RegionNode *Node);

  void delPhiValues(BasicBlock *From, BasicBlock *To);

  void addPhiValues(BasicBlock *From, BasicBlock *To);

  BasicBlock *getNextFlow(BasicBlock *Prev);

  bool isPredictableTrue(BasicBlock *Prev, BasicBlock *Node);

  BasicBlock *wireFlowBlock(BasicBlock *Prev, RegionNode *Node);

  void createFlow();

  void insertConditions();

  void rebuildSSA();

public:
  AMDGPUStructurizeCFG():
    RegionPass(ID) {

    initializeRegionInfoPass(*PassRegistry::getPassRegistry());
  }

  virtual bool doInitialization(Region *R, RGPassManager &RGM);

  virtual bool runOnRegion(Region *R, RGPassManager &RGM);

  virtual const char *getPassName() const {
    return "AMDGPU simplify control flow";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const {

    AU.addRequired<DominatorTree>();
    AU.addPreserved<DominatorTree>();
    RegionPass::getAnalysisUsage(AU);
  }

};

} // end anonymous namespace

char AMDGPUStructurizeCFG::ID = 0;

/// \brief Initialize the types and constants used in the pass
bool AMDGPUStructurizeCFG::doInitialization(Region *R, RGPassManager &RGM) {
  LLVMContext &Context = R->getEntry()->getContext();

  Boolean = Type::getInt1Ty(Context);
  BoolTrue = ConstantInt::getTrue(Context);
  BoolFalse = ConstantInt::getFalse(Context);
  BoolUndef = UndefValue::get(Boolean);

  return false;
}

/// \brief Build up the general order of nodes
void AMDGPUStructurizeCFG::orderNodes() {
  scc_iterator<Region *> I = scc_begin(ParentRegion),
                         E = scc_end(ParentRegion);
  for (Order.clear(); I != E; ++I) {
    std::vector<RegionNode *> &Nodes = *I;
    Order.append(Nodes.begin(), Nodes.end());
  }
}

/// \brief Build blocks and loop predicates
void AMDGPUStructurizeCFG::buildPredicate(BranchInst *Term, unsigned Idx,
                                          BBPredicates &Pred, bool Invert) {
  Value *True = Invert ? BoolFalse : BoolTrue;
  Value *False = Invert ? BoolTrue : BoolFalse;

  RegionInfo *RI = ParentRegion->getRegionInfo();
  BasicBlock *BB = Term->getParent();

  // Handle the case where multiple regions start at the same block
  Region *R = BB != ParentRegion->getEntry() ?
              RI->getRegionFor(BB) : ParentRegion;

  if (R == ParentRegion) {
    // It's a top level block in our region
    Value *Cond = True;
    if (Term->isConditional()) {
      BasicBlock *Other = Term->getSuccessor(!Idx);

      if (Visited.count(Other)) {
        if (!Pred.count(Other))
          Pred[Other] = False;

        if (!Pred.count(BB))
          Pred[BB] = True;
        return;
      }
      Cond = Term->getCondition();

      if (Idx != Invert)
        Cond = BinaryOperator::CreateNot(Cond, "", Term);
    }

    Pred[BB] = Cond;

  } else if (ParentRegion->contains(R)) {
    // It's a block in a sub region
    while(R->getParent() != ParentRegion)
      R = R->getParent();

    Pred[R->getEntry()] = True;

  } else {
    // It's a branch from outside into our parent region
    Pred[BB] = True;
  }
}

/// \brief Analyze the successors of each block and build up predicates
void AMDGPUStructurizeCFG::analyzeBlock(BasicBlock *BB) {
  pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
  BBPredicates &Pred = Predicates[BB];

  for (; PI != PE; ++PI) {
    BranchInst *Term = cast<BranchInst>((*PI)->getTerminator());

    for (unsigned i = 0, e = Term->getNumSuccessors(); i != e; ++i) {
      BasicBlock *Succ = Term->getSuccessor(i);
      if (Succ != BB)
        continue;
      buildPredicate(Term, i, Pred, false);
    }
  }
}

/// \brief Analyze the conditions leading to loop to a previous block
void AMDGPUStructurizeCFG::analyzeLoop(BasicBlock *BB, unsigned &LoopIdx) {
  BranchInst *Term = cast<BranchInst>(BB->getTerminator());

  for (unsigned i = 0, e = Term->getNumSuccessors(); i != e; ++i) {
    BasicBlock *Succ = Term->getSuccessor(i);

    // Ignore it if it's not a back edge
    if (!Visited.count(Succ))
      continue;

    buildPredicate(Term, i, LoopPred, true);

    LoopEnd = BB;
    if (Visited[Succ] < LoopIdx) {
      LoopIdx = Visited[Succ];
      LoopStart = Succ;
    }
  }
}

/// \brief Collect various loop and predicate infos
void AMDGPUStructurizeCFG::collectInfos() {
  unsigned Number = 0, LoopIdx = ~0;

  // Reset predicate
  Predicates.clear();

  // and loop infos
  LoopStart = LoopEnd = 0;
  LoopPred.clear();

  RNVector::reverse_iterator OI = Order.rbegin(), OE = Order.rend();
  for (Visited.clear(); OI != OE; Visited[(*OI++)->getEntry()] = ++Number) {

    // Analyze all the conditions leading to a node
    analyzeBlock((*OI)->getEntry());

    if ((*OI)->isSubRegion())
      continue;

    // Find the first/last loop nodes and loop predicates
    analyzeLoop((*OI)->getNodeAs<BasicBlock>(), LoopIdx);
  }
}

/// \brief Does A dominate all the predicates of B ?
bool AMDGPUStructurizeCFG::dominatesPredicates(BasicBlock *A, BasicBlock *B) {
  BBPredicates &Preds = Predicates[B];
  for (BBPredicates::iterator PI = Preds.begin(), PE = Preds.end();
       PI != PE; ++PI) {

    if (!DT->dominates(A, PI->first))
      return false;
  }
  return true;
}

/// \brief Remove phi values from all successors and the remove the terminator.
void AMDGPUStructurizeCFG::killTerminator(BasicBlock *BB) {
  TerminatorInst *Term = BB->getTerminator();
  if (!Term)
    return;

  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB);
       SI != SE; ++SI) {

    delPhiValues(BB, *SI);
  }

  Term->eraseFromParent();
}

/// First: Skip forward to the first region node that either isn't a subregion or not
/// dominating it's exit, remove all the skipped nodes from the node order.
///
/// Second: Handle the first successor directly if the resulting nodes successor
/// predicates are still dominated by the original entry
RegionNode *AMDGPUStructurizeCFG::skipChained(RegionNode *Node) {
  BasicBlock *Entry = Node->getEntry();

  // Skip forward as long as it is just a linear flow
  while (true) {
    BasicBlock *Entry = Node->getEntry();
    BasicBlock *Exit;

    if (Node->isSubRegion()) {
      Exit = Node->getNodeAs<Region>()->getExit();
    } else {
      TerminatorInst *Term = Entry->getTerminator();
      if (Term->getNumSuccessors() != 1)
        break;
      Exit = Term->getSuccessor(0);
    }

    // It's a back edge, break here so we can insert a loop node
    if (!Visited.count(Exit))
      return Node;

    // More than node edges are pointing to exit
    if (!DT->dominates(Entry, Exit))
      return Node;

    RegionNode *Next = ParentRegion->getNode(Exit);
    RNVector::iterator I = std::find(Order.begin(), Order.end(), Next);
    assert(I != Order.end());

    Visited.erase(Next->getEntry());
    Order.erase(I);
    Node = Next;
  }

  BasicBlock *BB = Node->getEntry();
  TerminatorInst *Term = BB->getTerminator();
  if (Term->getNumSuccessors() != 2)
    return Node;

  // Our node has exactly two succesors, check if we can handle
  // any of them directly
  BasicBlock *Succ = Term->getSuccessor(0);
  if (!Visited.count(Succ) || !dominatesPredicates(Entry, Succ)) {
    Succ = Term->getSuccessor(1);
    if (!Visited.count(Succ) || !dominatesPredicates(Entry, Succ))
      return Node;
  } else {
    BasicBlock *Succ2 = Term->getSuccessor(1);
    if (Visited.count(Succ2) && Visited[Succ] > Visited[Succ2] &&
        dominatesPredicates(Entry, Succ2))
      Succ = Succ2;
  }

  RegionNode *Next = ParentRegion->getNode(Succ);
  RNVector::iterator E = Order.end();
  RNVector::iterator I = std::find(Order.begin(), E, Next);
  assert(I != E);

  killTerminator(BB);
  FlowsInserted.push_back(BB);
  Visited.erase(Succ);
  Order.erase(I);
  return ParentRegion->getNode(wireFlowBlock(BB, Next));
}

/// \brief Remove all PHI values coming from "From" into "To" and remember
/// them in DeletedPhis
void AMDGPUStructurizeCFG::delPhiValues(BasicBlock *From, BasicBlock *To) {
  PhiMap &Map = DeletedPhis[To];
  for (BasicBlock::iterator I = To->begin(), E = To->end();
       I != E && isa<PHINode>(*I);) {

    PHINode &Phi = cast<PHINode>(*I++);
    while (Phi.getBasicBlockIndex(From) != -1) {
      Value *Deleted = Phi.removeIncomingValue(From, false);
      Map[&Phi].push_back(std::make_pair(From, Deleted));
    }
  }
}

/// \brief Add the PHI values back once we knew the new predecessor
void AMDGPUStructurizeCFG::addPhiValues(BasicBlock *From, BasicBlock *To) {
  if (!DeletedPhis.count(To))
    return;

  PhiMap &Map = DeletedPhis[To];
  SSAUpdater Updater;

  for (PhiMap::iterator I = Map.begin(), E = Map.end(); I != E; ++I) {

    PHINode *Phi = I->first;
    Updater.Initialize(Phi->getType(), "");
    BasicBlock *Fallback = To;
    bool HaveFallback = false;

    for (BBValueVector::iterator VI = I->second.begin(), VE = I->second.end();
         VI != VE; ++VI) {

      Updater.AddAvailableValue(VI->first, VI->second);
      BasicBlock *Dom = DT->findNearestCommonDominator(Fallback, VI->first);
      if (Dom == VI->first)
        HaveFallback = true;
      else if (Dom != Fallback)
        HaveFallback = false;
      Fallback = Dom;
    }
    if (!HaveFallback) {
      Value *Undef = UndefValue::get(Phi->getType());
      Updater.AddAvailableValue(Fallback, Undef);
    }

    Phi->addIncoming(Updater.GetValueAtEndOfBlock(From), From);
  }
  DeletedPhis.erase(To);
}

/// \brief Create a new flow node and update dominator tree and region info
BasicBlock *AMDGPUStructurizeCFG::getNextFlow(BasicBlock *Prev) {
  LLVMContext &Context = Func->getContext();
  BasicBlock *Insert = Order.empty() ? ParentRegion->getExit() :
                       Order.back()->getEntry();
  BasicBlock *Flow = BasicBlock::Create(Context, FlowBlockName,
                                        Func, Insert);
  DT->addNewBlock(Flow, Prev);
  ParentRegion->getRegionInfo()->setRegionFor(Flow, ParentRegion);
  FlowsInserted.push_back(Flow);
  return Flow;
}

/// \brief Can we predict that this node will always be called?
bool AMDGPUStructurizeCFG::isPredictableTrue(BasicBlock *Prev,
                                             BasicBlock *Node) {
  BBPredicates &Preds = Predicates[Node];
  bool Dominated = false;

  for (BBPredicates::iterator I = Preds.begin(), E = Preds.end();
       I != E; ++I) {

    if (I->second != BoolTrue)
      return false;

    if (!Dominated && DT->dominates(I->first, Prev))
      Dominated = true;
  }
  return Dominated;
}

/// \brief Wire up the new control flow by inserting or updating the branch
/// instructions at node exits
BasicBlock *AMDGPUStructurizeCFG::wireFlowBlock(BasicBlock *Prev,
                                                RegionNode *Node) {
  BasicBlock *Entry = Node->getEntry();

  if (LoopStart == Entry) {
    LoopStart = Prev;
    LoopPred[Prev] = BoolTrue;
  }

  // Wire it up temporary, skipChained may recurse into us
  BranchInst::Create(Entry, Prev);
  DT->changeImmediateDominator(Entry, Prev);
  addPhiValues(Prev, Entry);

  Node = skipChained(Node);

  BasicBlock *Next = getNextFlow(Prev);
  if (!isPredictableTrue(Prev, Entry)) {
    // Let Prev point to entry and next block
    Prev->getTerminator()->eraseFromParent();
    BranchInst::Create(Entry, Next, BoolUndef, Prev);
  } else {
    DT->changeImmediateDominator(Next, Entry);
  }

  // Let node exit(s) point to next block
  if (Node->isSubRegion()) {
    Region *SubRegion = Node->getNodeAs<Region>();
    BasicBlock *Exit = SubRegion->getExit();

    // Find all the edges from the sub region to the exit
    BBVector ToDo;
    for (pred_iterator I = pred_begin(Exit), E = pred_end(Exit); I != E; ++I) {
      if (SubRegion->contains(*I))
        ToDo.push_back(*I);
    }

    // Modify the edges to point to the new flow block
    for (BBVector::iterator I = ToDo.begin(), E = ToDo.end(); I != E; ++I) {
      delPhiValues(*I, Exit);
      TerminatorInst *Term = (*I)->getTerminator();
      Term->replaceUsesOfWith(Exit, Next);
    }

    // Update the region info
    SubRegion->replaceExit(Next);

  } else {
    BasicBlock *BB = Node->getNodeAs<BasicBlock>();
    killTerminator(BB);
    BranchInst::Create(Next, BB);

    if (BB == LoopEnd)
      LoopEnd = 0;
  }

  return Next;
}

/// Destroy node order and visited map, build up flow order instead.
/// After this function control flow looks like it should be, but
/// branches only have undefined conditions.
void AMDGPUStructurizeCFG::createFlow() {
  DeletedPhis.clear();

  BasicBlock *Prev = Order.pop_back_val()->getEntry();
  assert(Prev == ParentRegion->getEntry() && "Incorrect node order!");
  Visited.erase(Prev);

  if (LoopStart == Prev) {
    // Loop starts at entry, split entry so that we can predicate it
    BasicBlock::iterator Insert = Prev->getFirstInsertionPt();
    BasicBlock *Split = Prev->splitBasicBlock(Insert, FlowBlockName);
    DT->addNewBlock(Split, Prev);
    ParentRegion->getRegionInfo()->setRegionFor(Split, ParentRegion);
    Predicates[Split] = Predicates[Prev];
    Order.push_back(ParentRegion->getBBNode(Split));
    LoopPred[Prev] = BoolTrue;

  } else if (LoopStart == Order.back()->getEntry()) {
    // Loop starts behind entry, split entry so that we can jump to it
    Instruction *Term = Prev->getTerminator();
    BasicBlock *Split = Prev->splitBasicBlock(Term, FlowBlockName);
    DT->addNewBlock(Split, Prev);
    ParentRegion->getRegionInfo()->setRegionFor(Split, ParentRegion);
    Prev = Split;
  }

  killTerminator(Prev);
  FlowsInserted.clear();
  FlowsInserted.push_back(Prev);

  while (!Order.empty()) {
    RegionNode *Node = Order.pop_back_val();
    Visited.erase(Node->getEntry());
    Prev = wireFlowBlock(Prev, Node);
    if (LoopStart && !LoopEnd) {
      // Create an extra loop end node
      LoopEnd = Prev;
      Prev = getNextFlow(LoopEnd);
      BranchInst::Create(Prev, LoopStart, BoolUndef, LoopEnd);
      addPhiValues(LoopEnd, LoopStart);
    }
  }

  BasicBlock *Exit = ParentRegion->getExit();
  BranchInst::Create(Exit, Prev);
  addPhiValues(Prev, Exit);
  if (DT->dominates(ParentRegion->getEntry(), Exit))
    DT->changeImmediateDominator(Exit, Prev);

  if (LoopStart && LoopEnd) {
    BBVector::iterator FI = std::find(FlowsInserted.begin(),
                                      FlowsInserted.end(),
                                      LoopStart);
    for (; *FI != LoopEnd; ++FI) {
      addPhiValues(*FI, (*FI)->getTerminator()->getSuccessor(0));
    }
  }

  assert(Order.empty());
  assert(Visited.empty());
  assert(DeletedPhis.empty());
}

/// \brief Insert the missing branch conditions
void AMDGPUStructurizeCFG::insertConditions() {
  SSAUpdater PhiInserter;

  for (BBVector::iterator FI = FlowsInserted.begin(), FE = FlowsInserted.end();
       FI != FE; ++FI) {

    BranchInst *Term = cast<BranchInst>((*FI)->getTerminator());
    if (Term->isUnconditional())
      continue;

    PhiInserter.Initialize(Boolean, "");
    PhiInserter.AddAvailableValue(&Func->getEntryBlock(), BoolFalse);

    BasicBlock *Succ = Term->getSuccessor(0);
    BBPredicates &Preds = (*FI == LoopEnd) ? LoopPred : Predicates[Succ];
    for (BBPredicates::iterator PI = Preds.begin(), PE = Preds.end();
         PI != PE; ++PI) {

      PhiInserter.AddAvailableValue(PI->first, PI->second);
    }

    Term->setCondition(PhiInserter.GetValueAtEndOfBlock(*FI));
  }
}

/// Handle a rare case where the disintegrated nodes instructions
/// no longer dominate all their uses. Not sure if this is really nessasary
void AMDGPUStructurizeCFG::rebuildSSA() {
  SSAUpdater Updater;
  for (Region::block_iterator I = ParentRegion->block_begin(),
                              E = ParentRegion->block_end();
       I != E; ++I) {

    BasicBlock *BB = *I;
    for (BasicBlock::iterator II = BB->begin(), IE = BB->end();
         II != IE; ++II) {

      bool Initialized = false;
      for (Use *I = &II->use_begin().getUse(), *Next; I; I = Next) {

        Next = I->getNext();

        Instruction *User = cast<Instruction>(I->getUser());
        if (User->getParent() == BB) {
          continue;

        } else if (PHINode *UserPN = dyn_cast<PHINode>(User)) {
          if (UserPN->getIncomingBlock(*I) == BB)
            continue;
        }

        if (DT->dominates(II, User))
          continue;

        if (!Initialized) {
          Value *Undef = UndefValue::get(II->getType());
          Updater.Initialize(II->getType(), "");
          Updater.AddAvailableValue(&Func->getEntryBlock(), Undef);
          Updater.AddAvailableValue(BB, II);
          Initialized = true;
        }
        Updater.RewriteUseAfterInsertions(*I);
      }
    }
  }
}

/// \brief Run the transformation for each region found
bool AMDGPUStructurizeCFG::runOnRegion(Region *R, RGPassManager &RGM) {
  if (R->isTopLevelRegion())
    return false;

  Func = R->getEntry()->getParent();
  ParentRegion = R;

  DT = &getAnalysis<DominatorTree>();

  orderNodes();
  collectInfos();
  createFlow();
  insertConditions();
  rebuildSSA();

  Order.clear();
  Visited.clear();
  Predicates.clear();
  DeletedPhis.clear();
  FlowsInserted.clear();

  return true;
}

/// \brief Create the pass
Pass *llvm::createAMDGPUStructurizeCFGPass() {
  return new AMDGPUStructurizeCFG();
}
