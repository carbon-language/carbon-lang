//===- IndVarSimplify.cpp - Induction Variable Elimination ----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Guarantees that all loops with identifiable, linear, induction variables will
// be transformed to have a single, canonical, induction variable.  After this
// pass runs, it guarantees the the first PHI node of the header block in the
// loop is the canonical induction variable if there is one.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "indvar"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/InductionVariable.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CFG.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumRemoved ("indvars", "Number of aux indvars removed");
  Statistic<> NumInserted("indvars", "Number of canonical indvars added");

  class IndVarSimplify : public FunctionPass {
    LoopInfo *Loops;
    TargetData *TD;
    bool Changed;
  public:
    virtual bool runOnFunction(Function &) {
      Loops = &getAnalysis<LoopInfo>();
      TD = &getAnalysis<TargetData>();
      Changed = false;

      // Induction Variables live in the header nodes of loops
      for (LoopInfo::iterator I = Loops->begin(), E = Loops->end(); I != E; ++I)
        runOnLoop(*I);
      return Changed;
    }

    unsigned getTypeSize(const Type *Ty) {
      if (unsigned Size = Ty->getPrimitiveSize())
        return Size;
      return TD->getTypeSize(Ty);  // Must be a pointer
    }

    Value *ComputeAuxIndVarValue(InductionVariable &IV, Value *CIV);  
    void ReplaceIndVar(InductionVariable &IV, Value *Counter);

    void runOnLoop(Loop *L);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();   // Need pointer size
      AU.addRequired<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.setPreservesCFG();
    }
  };
  RegisterOpt<IndVarSimplify> X("indvars", "Canonicalize Induction Variables");
}

Pass *llvm::createIndVarSimplifyPass() {
  return new IndVarSimplify();
}


void IndVarSimplify::runOnLoop(Loop *Loop) {
  // Transform all subloops before this loop...
  for (LoopInfo::iterator I = Loop->begin(), E = Loop->end(); I != E; ++I)
    runOnLoop(*I);

  // Get the header node for this loop.  All of the phi nodes that could be
  // induction variables must live in this basic block.
  //
  BasicBlock *Header = Loop->getHeader();
  
  // Loop over all of the PHI nodes in the basic block, calculating the
  // induction variables that they represent... stuffing the induction variable
  // info into a vector...
  //
  std::vector<InductionVariable> IndVars;    // Induction variables for block
  BasicBlock::iterator AfterPHIIt = Header->begin();
  for (; PHINode *PN = dyn_cast<PHINode>(AfterPHIIt); ++AfterPHIIt)
    IndVars.push_back(InductionVariable(PN, Loops));
  // AfterPHIIt now points to first non-phi instruction...

  // If there are no phi nodes in this basic block, there can't be indvars...
  if (IndVars.empty()) return;
  
  // Loop over the induction variables, looking for a canonical induction
  // variable, and checking to make sure they are not all unknown induction
  // variables.  Keep track of the largest integer size of the induction
  // variable.
  //
  InductionVariable *Canonical = 0;
  unsigned MaxSize = 0;

  for (unsigned i = 0; i != IndVars.size(); ++i) {
    InductionVariable &IV = IndVars[i];

    if (IV.InductionType != InductionVariable::Unknown) {
      unsigned IVSize = getTypeSize(IV.Phi->getType());

      if (IV.InductionType == InductionVariable::Canonical &&
          !isa<PointerType>(IV.Phi->getType()) && IVSize >= MaxSize)
        Canonical = &IV;
      
      if (IVSize > MaxSize) MaxSize = IVSize;

      // If this variable is larger than the currently identified canonical
      // indvar, the canonical indvar is not usable.
      if (Canonical && IVSize > getTypeSize(Canonical->Phi->getType()))
        Canonical = 0;
    }
  }

  // No induction variables, bail early... don't add a canonical indvar
  if (MaxSize == 0) return;


  // Figure out what the exit condition of the loop is.  We can currently only
  // handle loops with a single exit.  If we cannot figure out what the
  // termination condition is, we leave this variable set to null.
  //
  SetCondInst *TermCond = 0;
  if (Loop->getExitBlocks().size() == 1) {
    // Get ExitingBlock - the basic block in the loop which contains the branch
    // out of the loop.
    BasicBlock *Exit = Loop->getExitBlocks()[0];
    pred_iterator PI = pred_begin(Exit);
    assert(PI != pred_end(Exit) && "Should have one predecessor in loop!");
    BasicBlock *ExitingBlock = *PI;
    assert(++PI == pred_end(Exit) && "Exit block should have one pred!");
    assert(Loop->isLoopExit(ExitingBlock) && "Exiting block is not loop exit!");

    // Since the block is in the loop, yet branches out of it, we know that the
    // block must end with multiple destination terminator.  Which means it is
    // either a conditional branch, a switch instruction, or an invoke.
    if (BranchInst *BI = dyn_cast<BranchInst>(ExitingBlock->getTerminator())) {
      assert(BI->isConditional() && "Unconditional branch has multiple succs?");
      TermCond = dyn_cast<SetCondInst>(BI->getCondition());
    } else {
      // NOTE: if people actually exit loops with switch instructions, we could
      // handle them, but I don't think this is important enough to spend time
      // thinking about.
      assert(isa<SwitchInst>(ExitingBlock->getTerminator()) ||
             isa<InvokeInst>(ExitingBlock->getTerminator()) &&
             "Unknown multi-successor terminator!");
    }
  }

  if (TermCond)
    DEBUG(std::cerr << "INDVAR: Found termination condition: " << *TermCond);

  // Okay, we want to convert other induction variables to use a canonical
  // indvar.  If we don't have one, add one now...
  if (!Canonical) {
    // Create the PHI node for the new induction variable, and insert the phi
    // node at the start of the PHI nodes...
    const Type *IVType;
    switch (MaxSize) {
    default: assert(0 && "Unknown integer type size!");
    case 1: IVType = Type::UByteTy; break;
    case 2: IVType = Type::UShortTy; break;
    case 4: IVType = Type::UIntTy; break;
    case 8: IVType = Type::ULongTy; break;
    }
    
    PHINode *PN = new PHINode(IVType, "cann-indvar", Header->begin());

    // Create the increment instruction to add one to the counter...
    Instruction *Add = BinaryOperator::create(Instruction::Add, PN,
                                              ConstantUInt::get(IVType, 1),
                                              "next-indvar", AfterPHIIt);

    // Figure out which block is incoming and which is the backedge for the loop
    BasicBlock *Incoming, *BackEdgeBlock;
    pred_iterator PI = pred_begin(Header);
    assert(PI != pred_end(Header) && "Loop headers should have 2 preds!");
    if (Loop->contains(*PI)) {  // First pred is back edge...
      BackEdgeBlock = *PI++;
      Incoming      = *PI++;
    } else {
      Incoming      = *PI++;
      BackEdgeBlock = *PI++;
    }
    assert(PI == pred_end(Header) && "Loop headers should have 2 preds!");
    
    // Add incoming values for the PHI node...
    PN->addIncoming(Constant::getNullValue(IVType), Incoming);
    PN->addIncoming(Add, BackEdgeBlock);

    // Analyze the new induction variable...
    IndVars.push_back(InductionVariable(PN, Loops));
    assert(IndVars.back().InductionType == InductionVariable::Canonical &&
           "Just inserted canonical indvar that is not canonical!");
    Canonical = &IndVars.back();
    ++NumInserted;
    Changed = true;
    DEBUG(std::cerr << "INDVAR: Inserted canonical iv: " << *PN);
  } else {
    // If we have a canonical induction variable, make sure that it is the first
    // one in the basic block.
    if (&Header->front() != Canonical->Phi)
      Header->getInstList().splice(Header->begin(), Header->getInstList(),
                                   Canonical->Phi);
    DEBUG(std::cerr << "IndVar: Existing canonical iv used: "
                    << *Canonical->Phi);
  }

  DEBUG(std::cerr << "INDVAR: Replacing Induction variables:\n");

  // Get the current loop iteration count, which is always the value of the
  // canonical phi node...
  //
  PHINode *IterCount = Canonical->Phi;

  // Loop through and replace all of the auxiliary induction variables with
  // references to the canonical induction variable...
  //
  for (unsigned i = 0; i != IndVars.size(); ++i) {
    InductionVariable *IV = &IndVars[i];

    DEBUG(IV->print(std::cerr));

    // Don't modify the canonical indvar or unrecognized indvars...
    if (IV != Canonical && IV->InductionType != InductionVariable::Unknown) {
      ReplaceIndVar(*IV, IterCount);
      Changed = true;
      ++NumRemoved;
    }
  }
}

/// ComputeAuxIndVarValue - Given an auxillary induction variable, compute and
/// return a value which will always be equal to the induction variable PHI, but
/// is based off of the canonical induction variable CIV.
///
Value *IndVarSimplify::ComputeAuxIndVarValue(InductionVariable &IV, Value *CIV){
  Instruction *Phi = IV.Phi;
  const Type *IVTy = Phi->getType();
  if (isa<PointerType>(IVTy))    // If indexing into a pointer, make the
    IVTy = TD->getIntPtrType();  // index the appropriate type.

  BasicBlock::iterator AfterPHIIt = Phi;
  while (isa<PHINode>(AfterPHIIt)) ++AfterPHIIt;
  
  Value *Val = CIV;
  if (Val->getType() != IVTy)
    Val = new CastInst(Val, IVTy, Val->getName(), AfterPHIIt);

  if (!isa<ConstantInt>(IV.Step) ||   // If the step != 1
      !cast<ConstantInt>(IV.Step)->equalsInt(1)) {
    
    // If the types are not compatible, insert a cast now...
    if (IV.Step->getType() != IVTy)
      IV.Step = new CastInst(IV.Step, IVTy, IV.Step->getName(), AfterPHIIt);
    
    Val = BinaryOperator::create(Instruction::Mul, Val, IV.Step,
                                 Phi->getName()+"-scale", AfterPHIIt);
  }
  
  // If this is a pointer indvar...
  if (isa<PointerType>(Phi->getType())) {
    std::vector<Value*> Idx;
    // FIXME: this should not be needed when we fix PR82!
    if (Val->getType() != Type::LongTy)
      Val = new CastInst(Val, Type::LongTy, Val->getName(), AfterPHIIt);
    Idx.push_back(Val);
    Val = new GetElementPtrInst(IV.Start, Idx,
                                Phi->getName()+"-offset",
                                AfterPHIIt);
    
  } else if (!isa<Constant>(IV.Start) ||   // If Start != 0...
             !cast<Constant>(IV.Start)->isNullValue()) {
    // If the types are not compatible, insert a cast now...
    if (IV.Start->getType() != IVTy)
      IV.Start = new CastInst(IV.Start, IVTy, IV.Start->getName(),
                               AfterPHIIt);
    
    // Insert the instruction after the phi nodes...
    Val = BinaryOperator::create(Instruction::Add, Val, IV.Start,
                                 Phi->getName()+"-offset", AfterPHIIt);
  }
  
  // If the PHI node has a different type than val is, insert a cast now...
  if (Val->getType() != Phi->getType())
    Val = new CastInst(Val, Phi->getType(), Val->getName(), AfterPHIIt);

  // Move the PHI name to it's new equivalent value...
  std::string OldName = Phi->getName();
  Phi->setName("");
  Val->setName(OldName);

  return Val;
}

// ReplaceIndVar - Replace all uses of the specified induction variable with
// expressions computed from the specified loop iteration counter variable.
// Return true if instructions were deleted.
void IndVarSimplify::ReplaceIndVar(InductionVariable &IV, Value *CIV) {
  Value *IndVarVal = 0;
  PHINode *Phi = IV.Phi;
  
  assert(Phi->getNumOperands() == 4 &&
         "Only expect induction variables in canonical loops!");

  // Remember the incoming values used by the PHI node
  std::vector<Value*> PHIOps;
  PHIOps.reserve(2);
  PHIOps.push_back(Phi->getIncomingValue(0));
  PHIOps.push_back(Phi->getIncomingValue(1));

  // Delete all of the operands of the PHI node... so that the to-be-deleted PHI
  // node does not cause any expressions to be computed that would not otherwise
  // be.
  Phi->dropAllReferences();

  // Now that we are rid of unneeded uses of the PHI node, replace any remaining
  // ones with the appropriate code using the canonical induction variable.
  while (!Phi->use_empty()) {
    Instruction *U = cast<Instruction>(Phi->use_back());

    // TODO: Perform LFTR here if possible
    if (0) {

    } else {
      // Replace all uses of the old PHI node with the new computed value...
      if (IndVarVal == 0)
        IndVarVal = ComputeAuxIndVarValue(IV, CIV);
      U->replaceUsesOfWith(Phi, IndVarVal);
    }
  }

  // If the PHI is the last user of any instructions for computing PHI nodes
  // that are irrelevant now, delete those instructions.
  while (!PHIOps.empty()) {
    Instruction *MaybeDead = dyn_cast<Instruction>(PHIOps.back());
    PHIOps.pop_back();
    
    if (MaybeDead && isInstructionTriviallyDead(MaybeDead) && 
        (!isa<PHINode>(MaybeDead) ||
         MaybeDead->getParent() != Phi->getParent())) {
      PHIOps.insert(PHIOps.end(), MaybeDead->op_begin(),
                    MaybeDead->op_end());
      MaybeDead->getParent()->getInstList().erase(MaybeDead);
      
      // Erase any duplicates entries in the PHIOps list.
      std::vector<Value*>::iterator It =
        std::find(PHIOps.begin(), PHIOps.end(), MaybeDead);
      while (It != PHIOps.end()) {
        PHIOps.erase(It);
        It = std::find(PHIOps.begin(), PHIOps.end(), MaybeDead);
      }
    }
  }

  // Delete the old, now unused, phi node...
  Phi->getParent()->getInstList().erase(Phi);
}

