//===------------------- SSI.cpp - Creates SSI Representation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass converts a list of variables to the Static Single Information
// form. This is a program representation described by Scott Ananian in his
// Master Thesis: "The Static Single Information Form (1999)".
// We are building an on-demand representation, that is, we do not convert
// every single variable in the target function to SSI form. Rather, we receive
// a list of target variables that must be converted. We also do not
// completely convert a target variable to the SSI format. Instead, we only
// change the variable in the points where new information can be attached
// to its live range, that is, at branch points.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ssi"

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/SSI.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"

using namespace llvm;

static const std::string SSI_PHI = "SSI_phi";
static const std::string SSI_SIG = "SSI_sigma";

STATISTIC(NumSigmaInserted, "Number of sigma functions inserted");
STATISTIC(NumPhiInserted, "Number of phi functions inserted");

void SSI::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredTransitive<DominanceFrontier>();
  AU.addRequiredTransitive<DominatorTree>();
  AU.setPreservesAll();
}

bool SSI::runOnFunction(Function &F) {
  DT_ = &getAnalysis<DominatorTree>();
  return false;
}

/// This methods creates the SSI representation for the list of values
/// received. It will only create SSI representation if a value is used
/// to decide a branch. Repeated values are created only once.
///
void SSI::createSSI(SmallVectorImpl<Instruction *> &value) {
  init(value);

  SmallPtrSet<Instruction*, 4> needConstruction;
  for (SmallVectorImpl<Instruction*>::iterator I = value.begin(),
       E = value.end(); I != E; ++I)
    if (created.insert(*I))
      needConstruction.insert(*I);

  insertSigmaFunctions(needConstruction);

  // Test if there is a need to transform to SSI
  if (!needConstruction.empty()) {
    insertPhiFunctions(needConstruction);
    renameInit(needConstruction);
    rename(DT_->getRoot());
    fixPhis();
  }

  clean();
}

/// Insert sigma functions (a sigma function is a phi function with one
/// operator)
///
void SSI::insertSigmaFunctions(SmallPtrSet<Instruction*, 4> &value) {
  for (SmallPtrSet<Instruction*, 4>::iterator I = value.begin(),
       E = value.end(); I != E; ++I) {
    for (Value::use_iterator begin = (*I)->use_begin(),
         end = (*I)->use_end(); begin != end; ++begin) {
      // Test if the Use of the Value is in a comparator
      if (CmpInst *CI = dyn_cast<CmpInst>(begin)) {
        // Iterates through all uses of CmpInst
        for (Value::use_iterator begin_ci = CI->use_begin(),
             end_ci = CI->use_end(); begin_ci != end_ci; ++begin_ci) {
          // Test if any use of CmpInst is in a Terminator
          if (TerminatorInst *TI = dyn_cast<TerminatorInst>(begin_ci)) {
            insertSigma(TI, *I);
          }
        }
      }
    }
  }
}

/// Inserts Sigma Functions in every BasicBlock successor to Terminator
/// Instruction TI. All inserted Sigma Function are related to Instruction I.
///
void SSI::insertSigma(TerminatorInst *TI, Instruction *I) {
  // Basic Block of the Terminator Instruction
  BasicBlock *BB = TI->getParent();
  for (unsigned i = 0, e = TI->getNumSuccessors(); i < e; ++i) {
    // Next Basic Block
    BasicBlock *BB_next = TI->getSuccessor(i);
    if (BB_next != BB &&
        BB_next->getSinglePredecessor() != NULL &&
        dominateAny(BB_next, I)) {
      PHINode *PN = PHINode::Create(I->getType(), SSI_SIG, BB_next->begin());
      PN->addIncoming(I, BB);
      sigmas[PN] = I;
      created.insert(PN);
      defsites[I].push_back(BB_next);
      ++NumSigmaInserted;
    }
  }
}

/// Insert phi functions when necessary
///
void SSI::insertPhiFunctions(SmallPtrSet<Instruction*, 4> &value) {
  DominanceFrontier *DF = &getAnalysis<DominanceFrontier>();
  for (SmallPtrSet<Instruction*, 4>::iterator I = value.begin(),
       E = value.end(); I != E; ++I) {
    // Test if there were any sigmas for this variable
    SmallPtrSet<BasicBlock *, 16> BB_visited;

    // Insert phi functions if there is any sigma function
    while (!defsites[*I].empty()) {

      BasicBlock *BB = defsites[*I].back();

      defsites[*I].pop_back();
      DominanceFrontier::iterator DF_BB = DF->find(BB);

      // The BB is unreachable. Skip it.
      if (DF_BB == DF->end())
        continue; 

      // Iterates through all the dominance frontier of BB
      for (std::set<BasicBlock *>::iterator DF_BB_begin =
           DF_BB->second.begin(), DF_BB_end = DF_BB->second.end();
           DF_BB_begin != DF_BB_end; ++DF_BB_begin) {
        BasicBlock *BB_dominated = *DF_BB_begin;

        // Test if has not yet visited this node and if the
        // original definition dominates this node
        if (BB_visited.insert(BB_dominated) &&
            DT_->properlyDominates(value_original[*I], BB_dominated) &&
            dominateAny(BB_dominated, *I)) {
          PHINode *PN = PHINode::Create(
              (*I)->getType(), SSI_PHI, BB_dominated->begin());
          phis.insert(std::make_pair(PN, *I));
          created.insert(PN);

          defsites[*I].push_back(BB_dominated);
          ++NumPhiInserted;
        }
      }
    }
    BB_visited.clear();
  }
}

/// Some initialization for the rename part
///
void SSI::renameInit(SmallPtrSet<Instruction*, 4> &value) {
  for (SmallPtrSet<Instruction*, 4>::iterator I = value.begin(),
       E = value.end(); I != E; ++I)
    value_stack[*I].push_back(*I);
}

/// Renames all variables in the specified BasicBlock.
/// Only variables that need to be rename will be.
///
void SSI::rename(BasicBlock *BB) {
  SmallPtrSet<Instruction*, 8> defined;

  // Iterate through instructions and make appropriate renaming.
  // For SSI_PHI (b = PHI()), store b at value_stack as a new
  // definition of the variable it represents.
  // For SSI_SIG (b = PHI(a)), substitute a with the current
  // value of a, present in the value_stack.
  // Then store bin the value_stack as the new definition of a.
  // For all other instructions (b = OP(a, c, d, ...)), we need to substitute
  // all operands with its current value, present in value_stack.
  for (BasicBlock::iterator begin = BB->begin(), end = BB->end();
       begin != end; ++begin) {
    Instruction *I = begin;
    if (PHINode *PN = dyn_cast<PHINode>(I)) { // Treat PHI functions
      Instruction* position;

      // Treat SSI_PHI
      if ((position = getPositionPhi(PN))) {
        value_stack[position].push_back(PN);
        defined.insert(position);
      // Treat SSI_SIG
      } else if ((position = getPositionSigma(PN))) {
        substituteUse(I);
        value_stack[position].push_back(PN);
        defined.insert(position);
      }

      // Treat all other PHI functions
      else {
        substituteUse(I);
      }
    }

    // Treat all other functions
    else {
      substituteUse(I);
    }
  }

  // This loop iterates in all BasicBlocks that are successors of the current
  // BasicBlock. For each SSI_PHI instruction found, insert an operand.
  // This operand is the current operand in value_stack for the variable
  // in "position". And the BasicBlock this operand represents is the current
  // BasicBlock.
  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI) {
    BasicBlock *BB_succ = *SI;

    for (BasicBlock::iterator begin = BB_succ->begin(),
         notPhi = BB_succ->getFirstNonPHI(); begin != *notPhi; ++begin) {
      Instruction *I = begin;
      PHINode *PN = dyn_cast<PHINode>(I);
      Instruction* position;
      if (PN && ((position = getPositionPhi(PN)))) {
        PN->addIncoming(value_stack[position].back(), BB);
      }
    }
  }

  // This loop calls rename on all children from this block. This time children
  // refers to a successor block in the dominance tree.
  DomTreeNode *DTN = DT_->getNode(BB);
  for (DomTreeNode::iterator begin = DTN->begin(), end = DTN->end();
       begin != end; ++begin) {
    DomTreeNodeBase<BasicBlock> *DTN_children = *begin;
    BasicBlock *BB_children = DTN_children->getBlock();
    rename(BB_children);
  }

  // Now we remove all inserted definitions of a variable from the top of
  // the stack leaving the previous one as the top.
  for (SmallPtrSet<Instruction*, 8>::iterator DI = defined.begin(),
       DE = defined.end(); DI != DE; ++DI)
    value_stack[*DI].pop_back();
}

/// Substitute any use in this instruction for the last definition of
/// the variable
///
void SSI::substituteUse(Instruction *I) {
  for (unsigned i = 0, e = I->getNumOperands(); i < e; ++i) {
    Value *operand = I->getOperand(i);
    for (DenseMap<Instruction*, SmallVector<Instruction*, 1> >::iterator
         VI = value_stack.begin(), VE = value_stack.end(); VI != VE; ++VI) {
      if (operand == VI->second.front() &&
          I != VI->second.back()) {
        PHINode *PN_I = dyn_cast<PHINode>(I);
        PHINode *PN_vs = dyn_cast<PHINode>(VI->second.back());

        // If a phi created in a BasicBlock is used as an operand of another
        // created in the same BasicBlock, this step marks this second phi,
        // to fix this issue later. It cannot be fixed now, because the
        // operands of the first phi are not final yet.
        if (PN_I && PN_vs &&
            VI->second.back()->getParent() == I->getParent()) {

          phisToFix.insert(PN_I);
        }

        I->setOperand(i, VI->second.back());
        break;
      }
    }
  }
}

/// Test if the BasicBlock BB dominates any use or definition of value.
/// If it dominates a phi instruction that is on the same BasicBlock,
/// that does not count.
///
bool SSI::dominateAny(BasicBlock *BB, Instruction *value) {
  for (Value::use_iterator begin = value->use_begin(),
       end = value->use_end(); begin != end; ++begin) {
    Instruction *I = cast<Instruction>(*begin);
    BasicBlock *BB_father = I->getParent();
    if (BB == BB_father && isa<PHINode>(I))
      continue;
    if (DT_->dominates(BB, BB_father)) {
      return true;
    }
  }
  return false;
}

/// When there is a phi node that is created in a BasicBlock and it is used
/// as an operand of another phi function used in the same BasicBlock,
/// LLVM looks this as an error. So on the second phi, the first phi is called
/// P and the BasicBlock it incomes is B. This P will be replaced by the value
/// it has for BasicBlock B. It also includes undef values for predecessors
/// that were not included in the phi.
///
void SSI::fixPhis() {
  for (SmallPtrSet<PHINode *, 1>::iterator begin = phisToFix.begin(),
       end = phisToFix.end(); begin != end; ++begin) {
    PHINode *PN = *begin;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i) {
      PHINode *PN_father = dyn_cast<PHINode>(PN->getIncomingValue(i));
      if (PN_father && PN->getParent() == PN_father->getParent() &&
          !DT_->dominates(PN->getParent(), PN->getIncomingBlock(i))) {
        BasicBlock *BB = PN->getIncomingBlock(i);
        int pos = PN_father->getBasicBlockIndex(BB);
        PN->setIncomingValue(i, PN_father->getIncomingValue(pos));
      }
    }
  }

  for (DenseMapIterator<PHINode *, Instruction*> begin = phis.begin(),
       end = phis.end(); begin != end; ++begin) {
    PHINode *PN = begin->first;
    BasicBlock *BB = PN->getParent();
    pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
    SmallVector<BasicBlock*, 8> Preds(PI, PE);
    for (unsigned size = Preds.size();
         PI != PE && PN->getNumIncomingValues() != size; ++PI) {
      bool found = false;
      for (unsigned i = 0, pn_end = PN->getNumIncomingValues();
           i < pn_end; ++i) {
        if (PN->getIncomingBlock(i) == *PI) {
          found = true;
          break;
        }
      }
      if (!found) {
        PN->addIncoming(UndefValue::get(PN->getType()), *PI);
      }
    }
  }
}

/// Return which variable (position on the vector of variables) this phi
/// represents on the phis list.
///
Instruction* SSI::getPositionPhi(PHINode *PN) {
  DenseMap<PHINode *, Instruction*>::iterator val = phis.find(PN);
  if (val == phis.end())
    return 0;
  else
    return val->second;
}

/// Return which variable (position on the vector of variables) this phi
/// represents on the sigmas list.
///
Instruction* SSI::getPositionSigma(PHINode *PN) {
  DenseMap<PHINode *, Instruction*>::iterator val = sigmas.find(PN);
  if (val == sigmas.end())
    return 0;
  else
    return val->second;
}

/// Initializes
///
void SSI::init(SmallVectorImpl<Instruction *> &value) {
  for (SmallVectorImpl<Instruction *>::iterator I = value.begin(),
       E = value.end(); I != E; ++I) {
    value_original[*I] = (*I)->getParent();
    defsites[*I].push_back((*I)->getParent());
  }
}

/// Clean all used resources in this creation of SSI
///
void SSI::clean() {
  phis.clear();
  sigmas.clear();
  phisToFix.clear();

  defsites.clear();
  value_stack.clear();
  value_original.clear();
}

/// createSSIPass - The public interface to this file...
///
FunctionPass *llvm::createSSIPass() { return new SSI(); }

char SSI::ID = 0;
static RegisterPass<SSI> X("ssi", "Static Single Information Construction");

/// SSIEverything - A pass that runs createSSI on every non-void variable,
/// intended for debugging.
namespace {
  struct VISIBILITY_HIDDEN SSIEverything : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    SSIEverything() : FunctionPass(&ID) {}

    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<SSI>();
    }
  };
}

bool SSIEverything::runOnFunction(Function &F) {
  SmallVector<Instruction *, 16> Insts;
  SSI &ssi = getAnalysis<SSI>();

  if (F.isDeclaration() || F.isIntrinsic()) return false;

  for (Function::iterator B = F.begin(), BE = F.end(); B != BE; ++B)
    for (BasicBlock::iterator I = B->begin(), E = B->end(); I != E; ++I)
      if (I->getType() != Type::getVoidTy(F.getContext()))
        Insts.push_back(I);

  ssi.createSSI(Insts);
  return true;
}

/// createSSIEverythingPass - The public interface to this file...
///
FunctionPass *llvm::createSSIEverythingPass() { return new SSIEverything(); }

char SSIEverything::ID = 0;
static RegisterPass<SSIEverything>
Y("ssi-everything", "Static Single Information Construction");
