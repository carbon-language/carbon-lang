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

static const unsigned UNSIGNED_INFINITE = ~0U;

STATISTIC(NumSigmaInserted, "Number of sigma functions inserted");
STATISTIC(NumPhiInserted, "Number of phi functions inserted");

void SSI::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominanceFrontier>();
  AU.addRequired<DominatorTree>();
  AU.setPreservesAll();
}

bool SSI::runOnFunction(Function &F) {
  DT_ = &getAnalysis<DominatorTree>();
  return false;
}

/// This methods creates the SSI representation for the list of values
/// received. It will only create SSI representation if a value is used
/// in a to decide a branch. Repeated values are created only once.
///
void SSI::createSSI(SmallVectorImpl<Instruction *> &value) {
  init(value);

  for (unsigned i = 0; i < num_values; ++i) {
    if (created.insert(value[i])) {
      needConstruction[i] = true;
    }
  }
  insertSigmaFunctions(value);

  // Test if there is a need to transform to SSI
  if (needConstruction.any()) {
    insertPhiFunctions(value);
    renameInit(value);
    rename(DT_->getRoot());
    fixPhis();
  }

  clean();
}

/// Insert sigma functions (a sigma function is a phi function with one
/// operator)
///
void SSI::insertSigmaFunctions(SmallVectorImpl<Instruction *> &value) {
  for (unsigned i = 0; i < num_values; ++i) {
    if (!needConstruction[i])
      continue;

    bool need = false;
    for (Value::use_iterator begin = value[i]->use_begin(), end =
         value[i]->use_end(); begin != end; ++begin) {
      // Test if the Use of the Value is in a comparator
      CmpInst *CI = dyn_cast<CmpInst>(begin);
      if (CI && isUsedInTerminator(CI)) {
        // Basic Block of the Instruction
        BasicBlock *BB = CI->getParent();
        // Last Instruction of the Basic Block
        const TerminatorInst *TI = BB->getTerminator();

        for (unsigned j = 0, e = TI->getNumSuccessors(); j < e; ++j) {
          // Next Basic Block
          BasicBlock *BB_next = TI->getSuccessor(j);
          if (BB_next != BB &&
              BB_next->getUniquePredecessor() != NULL &&
              dominateAny(BB_next, value[i])) {
            PHINode *PN = PHINode::Create(
                value[i]->getType(), SSI_SIG, BB_next->begin());
            PN->addIncoming(value[i], BB);
            sigmas.insert(std::make_pair(PN, i));
            created.insert(PN);
            need = true;
            defsites[i].push_back(BB_next);
            ++NumSigmaInserted;
          }
        }
      }
    }
    needConstruction[i] = need;
  }
}

/// Insert phi functions when necessary
///
void SSI::insertPhiFunctions(SmallVectorImpl<Instruction *> &value) {
  DominanceFrontier *DF = &getAnalysis<DominanceFrontier>();
  for (unsigned i = 0; i < num_values; ++i) {
    // Test if there were any sigmas for this variable
    if (needConstruction[i]) {

      SmallPtrSet<BasicBlock *, 1> BB_visited;

      // Insert phi functions if there is any sigma function
      while (!defsites[i].empty()) {

        BasicBlock *BB = defsites[i].back();

        defsites[i].pop_back();
        DominanceFrontier::iterator DF_BB = DF->find(BB);

        // Iterates through all the dominance frontier of BB
        for (std::set<BasicBlock *>::iterator DF_BB_begin =
             DF_BB->second.begin(), DF_BB_end = DF_BB->second.end();
             DF_BB_begin != DF_BB_end; ++DF_BB_begin) {
          BasicBlock *BB_dominated = *DF_BB_begin;

          // Test if has not yet visited this node and if the
          // original definition dominates this node
          if (BB_visited.insert(BB_dominated) &&
              DT_->properlyDominates(value_original[i], BB_dominated) &&
              dominateAny(BB_dominated, value[i])) {
            PHINode *PN = PHINode::Create(
                value[i]->getType(), SSI_PHI, BB_dominated->begin());
            phis.insert(std::make_pair(PN, i));
            created.insert(PN);

            defsites[i].push_back(BB_dominated);
            ++NumPhiInserted;
          }
        }
      }
      BB_visited.clear();
    }
  }
}

/// Some initialization for the rename part
///
void SSI::renameInit(SmallVectorImpl<Instruction *> &value) {
  value_stack.resize(num_values);
  for (unsigned i = 0; i < num_values; ++i) {
    value_stack[i].push_back(value[i]);
  }
}

/// Renames all variables in the specified BasicBlock.
/// Only variables that need to be rename will be.
///
void SSI::rename(BasicBlock *BB) {
  BitVector *defined = new BitVector(num_values, false);

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
      int position;

      // Treat SSI_PHI
      if ((position = getPositionPhi(PN)) != -1) {
        value_stack[position].push_back(PN);
        (*defined)[position] = true;
      }

      // Treat SSI_SIG
      else if ((position = getPositionSigma(PN)) != -1) {
        substituteUse(I);
        value_stack[position].push_back(PN);
        (*defined)[position] = true;
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
      PHINode *PN;
      int position;
      if ((PN = dyn_cast<PHINode>(I)) && ((position
          = getPositionPhi(PN)) != -1)) {
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
  if (defined->any()) {
    for (unsigned i = 0; i < num_values; ++i) {
      if ((*defined)[i]) {
        value_stack[i].pop_back();
      }
    }
  }
}

/// Substitute any use in this instruction for the last definition of
/// the variable
///
void SSI::substituteUse(Instruction *I) {
  for (unsigned i = 0, e = I->getNumOperands(); i < e; ++i) {
    Value *operand = I->getOperand(i);
    for (unsigned j = 0; j < num_values; ++j) {
      if (operand == value_stack[j].front() &&
          I != value_stack[j].back()) {
        PHINode *PN_I = dyn_cast<PHINode>(I);
        PHINode *PN_vs = dyn_cast<PHINode>(value_stack[j].back());

        // If a phi created in a BasicBlock is used as an operand of another
        // created in the same BasicBlock, this step marks this second phi,
        // to fix this issue later. It cannot be fixed now, because the
        // operands of the first phi are not final yet.
        if (PN_I && PN_vs &&
            value_stack[j].back()->getParent() == I->getParent()) {

          phisToFix.insert(PN_I);
        }

        I->setOperand(i, value_stack[j].back());
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
/// it has for BasicBlock B.
///
void SSI::fixPhis() {
  for (SmallPtrSet<PHINode *, 1>::iterator begin = phisToFix.begin(),
       end = phisToFix.end(); begin != end; ++begin) {
    PHINode *PN = *begin;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i) {
      PHINode *PN_father;
      if ((PN_father = dyn_cast<PHINode>(PN->getIncomingValue(i))) &&
          PN->getParent() == PN_father->getParent()) {
        BasicBlock *BB = PN->getIncomingBlock(i);
        int pos = PN_father->getBasicBlockIndex(BB);
        PN->setIncomingValue(i, PN_father->getIncomingValue(pos));
      }
    }
  }
}

/// Return which variable (position on the vector of variables) this phi
/// represents on the phis list.
///
unsigned SSI::getPositionPhi(PHINode *PN) {
  DenseMap<PHINode *, unsigned>::iterator val = phis.find(PN);
  if (val == phis.end())
    return UNSIGNED_INFINITE;
  else
    return val->second;
}

/// Return which variable (position on the vector of variables) this phi
/// represents on the sigmas list.
///
unsigned SSI::getPositionSigma(PHINode *PN) {
  DenseMap<PHINode *, unsigned>::iterator val = sigmas.find(PN);
  if (val == sigmas.end())
    return UNSIGNED_INFINITE;
  else
    return val->second;
}

/// Return true if the the Comparison Instruction is an operator
/// of the Terminator instruction of its Basic Block.
///
unsigned SSI::isUsedInTerminator(CmpInst *CI) {
  TerminatorInst *TI = CI->getParent()->getTerminator();
  if (TI->getNumOperands() == 0) {
    return false;
  } else if (CI == TI->getOperand(0)) {
    return true;
  } else {
    return false;
  }
}

/// Initializes
///
void SSI::init(SmallVectorImpl<Instruction *> &value) {
  num_values = value.size();
  needConstruction.resize(num_values, false);

  value_original.resize(num_values);
  defsites.resize(num_values);

  for (unsigned i = 0; i < num_values; ++i) {
    value_original[i] = value[i]->getParent();
    defsites[i].push_back(value_original[i]);
  }
}

/// Clean all used resources in this creation of SSI
///
void SSI::clean() {
  for (unsigned i = 0; i < num_values; ++i) {
    defsites[i].clear();
    if (i < value_stack.size())
      value_stack[i].clear();
  }

  phis.clear();
  sigmas.clear();
  phisToFix.clear();

  defsites.clear();
  value_stack.clear();
  value_original.clear();
  needConstruction.clear();
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
      if (I->getType() != Type::VoidTy)
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
