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

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
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
  public:
    virtual bool runOnFunction(Function &) {
      Loops = &getAnalysis<LoopInfo>();
      TD = &getAnalysis<TargetData>();
      
      // Induction Variables live in the header nodes of loops
      bool Changed = false;
      for (unsigned i = 0, e = Loops->getTopLevelLoops().size(); i != e; ++i)
        Changed |= runOnLoop(Loops->getTopLevelLoops()[i]);
      return Changed;
    }

    unsigned getTypeSize(const Type *Ty) {
      if (unsigned Size = Ty->getPrimitiveSize())
        return Size;
      return TD->getTypeSize(Ty);  // Must be a pointer
    }

    bool runOnLoop(Loop *L);

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


bool IndVarSimplify::runOnLoop(Loop *Loop) {
  // Transform all subloops before this loop...
  bool Changed = false;
  for (unsigned i = 0, e = Loop->getSubLoops().size(); i != e; ++i)
    Changed |= runOnLoop(Loop->getSubLoops()[i]);

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
  if (IndVars.empty()) return Changed;
  
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
  if (MaxSize == 0) return Changed;

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
  } else {
    // If we have a canonical induction variable, make sure that it is the first
    // one in the basic block.
    if (&Header->front() != Canonical->Phi)
      Header->getInstList().splice(Header->begin(), Header->getInstList(),
                                   Canonical->Phi);
  }

  DEBUG(std::cerr << "Induction variables:\n");

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

    while (isa<PHINode>(AfterPHIIt)) ++AfterPHIIt;

    // Don't do math with pointers...
    const Type *IVTy = IV->Phi->getType();
    if (isa<PointerType>(IVTy)) IVTy = Type::ULongTy;

    // Don't modify the canonical indvar or unrecognized indvars...
    if (IV != Canonical && IV->InductionType != InductionVariable::Unknown) {
      Instruction *Val = IterCount;
      if (!isa<ConstantInt>(IV->Step) ||   // If the step != 1
          !cast<ConstantInt>(IV->Step)->equalsInt(1)) {

        // If the types are not compatible, insert a cast now...
        if (Val->getType() != IVTy)
          Val = new CastInst(Val, IVTy, Val->getName(), AfterPHIIt);
        if (IV->Step->getType() != IVTy)
          IV->Step = new CastInst(IV->Step, IVTy, IV->Step->getName(),
                                  AfterPHIIt);

        Val = BinaryOperator::create(Instruction::Mul, Val, IV->Step,
                                     IV->Phi->getName()+"-scale", AfterPHIIt);
      }

      // If the start != 0
      if (IV->Start != Constant::getNullValue(IV->Start->getType())) {
        // If the types are not compatible, insert a cast now...
        if (Val->getType() != IVTy)
          Val = new CastInst(Val, IVTy, Val->getName(), AfterPHIIt);
        if (IV->Start->getType() != IVTy)
          IV->Start = new CastInst(IV->Start, IVTy, IV->Start->getName(),
                                   AfterPHIIt);

        // Insert the instruction after the phi nodes...
        Val = BinaryOperator::create(Instruction::Add, Val, IV->Start,
                                     IV->Phi->getName()+"-offset", AfterPHIIt);
      }

      // If the PHI node has a different type than val is, insert a cast now...
      if (Val->getType() != IV->Phi->getType())
        Val = new CastInst(Val, IV->Phi->getType(), Val->getName(), AfterPHIIt);
      
      // Replace all uses of the old PHI node with the new computed value...
      IV->Phi->replaceAllUsesWith(Val);

      // Move the PHI name to it's new equivalent value...
      std::string OldName = IV->Phi->getName();
      IV->Phi->setName("");
      Val->setName(OldName);

      // Get the incoming values used by the PHI node
      std::vector<Value*> PHIOps;
      PHIOps.reserve(IV->Phi->getNumIncomingValues());
      for (unsigned i = 0, e = IV->Phi->getNumIncomingValues(); i != e; ++i)
        PHIOps.push_back(IV->Phi->getIncomingValue(i));

      // Delete the old, now unused, phi node...
      Header->getInstList().erase(IV->Phi);

      // If the PHI is the last user of any instructions for computing PHI nodes
      // that are irrelevant now, delete those instructions.
      while (!PHIOps.empty()) {
        Instruction *MaybeDead = dyn_cast<Instruction>(PHIOps.back());
        PHIOps.pop_back();

        if (MaybeDead && isInstructionTriviallyDead(MaybeDead)) {
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

          // Erasing the instruction could invalidate the AfterPHI iterator!
          AfterPHIIt = Header->begin();
        }
      }

      Changed = true;
      ++NumRemoved;
    }
  }

  return Changed;
}

