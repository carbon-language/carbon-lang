//===- LoopStrengthReduce.cpp - Strength Reduce GEPs in Loops -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs a strength reduction on array references inside loops that
// have as one or more of their components the loop induction variable.  This is
// accomplished by creating a new Value to hold the initial value of the array
// access for the first iteration, and then creating a new GEP instruction in
// the loop to increment the value by the appropriate amount.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-reduce"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumReduced ("loop-reduce", "Number of GEPs strength reduced");
  Statistic<> NumInserted("loop-reduce", "Number of PHIs inserted");
  Statistic<> NumVariable("loop-reduce","Number of PHIs with variable strides");

  /// IVStrideUse - Keep track of one use of a strided induction variable, where
  /// the stride is stored externally.  The Offset member keeps track of the 
  /// offset from the IV, User is the actual user of the operand, and 'Operand'
  /// is the operand # of the User that is the use.
  struct IVStrideUse {
    SCEVHandle Offset;
    Instruction *User;
    Value *OperandValToReplace;

    // isUseOfPostIncrementedValue - True if this should use the
    // post-incremented version of this IV, not the preincremented version.
    // This can only be set in special cases, such as the terminating setcc
    // instruction for a loop or uses dominated by the loop.
    bool isUseOfPostIncrementedValue;
    
    IVStrideUse(const SCEVHandle &Offs, Instruction *U, Value *O)
      : Offset(Offs), User(U), OperandValToReplace(O),
        isUseOfPostIncrementedValue(false) {}
  };
  
  /// IVUsersOfOneStride - This structure keeps track of all instructions that
  /// have an operand that is based on the trip count multiplied by some stride.
  /// The stride for all of these users is common and kept external to this
  /// structure.
  struct IVUsersOfOneStride {
    /// Users - Keep track of all of the users of this stride as well as the
    /// initial value and the operand that uses the IV.
    std::vector<IVStrideUse> Users;
    
    void addUser(const SCEVHandle &Offset,Instruction *User, Value *Operand) {
      Users.push_back(IVStrideUse(Offset, User, Operand));
    }
  };


  class LoopStrengthReduce : public FunctionPass {
    LoopInfo *LI;
    DominatorSet *DS;
    ScalarEvolution *SE;
    const TargetData *TD;
    const Type *UIntPtrTy;
    bool Changed;

    /// MaxTargetAMSize - This is the maximum power-of-two scale value that the
    /// target can handle for free with its addressing modes.
    unsigned MaxTargetAMSize;

    /// IVUsesByStride - Keep track of all uses of induction variables that we
    /// are interested in.  The key of the map is the stride of the access.
    std::map<SCEVHandle, IVUsersOfOneStride> IVUsesByStride;

    /// CastedValues - As we need to cast values to uintptr_t, this keeps track
    /// of the casted version of each value.  This is accessed by
    /// getCastedVersionOf.
    std::map<Value*, Value*> CastedPointers;

    /// DeadInsts - Keep track of instructions we may have made dead, so that
    /// we can remove them after we are done working.
    std::set<Instruction*> DeadInsts;
  public:
    LoopStrengthReduce(unsigned MTAMS = 1)
      : MaxTargetAMSize(MTAMS) {
    }

    virtual bool runOnFunction(Function &) {
      LI = &getAnalysis<LoopInfo>();
      DS = &getAnalysis<DominatorSet>();
      SE = &getAnalysis<ScalarEvolution>();
      TD = &getAnalysis<TargetData>();
      UIntPtrTy = TD->getIntPtrType();
      Changed = false;

      for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
        runOnLoop(*I);
      
      return Changed;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // We split critical edges, so we change the CFG.  However, we do update
      // many analyses if they are around.
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreserved<LoopInfo>();
      AU.addPreserved<DominatorSet>();
      AU.addPreserved<ImmediateDominators>();
      AU.addPreserved<DominanceFrontier>();
      AU.addPreserved<DominatorTree>();

      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorSet>();
      AU.addRequired<TargetData>();
      AU.addRequired<ScalarEvolution>();
    }
    
    /// getCastedVersionOf - Return the specified value casted to uintptr_t.
    ///
    Value *getCastedVersionOf(Value *V);
private:
    void runOnLoop(Loop *L);
    bool AddUsersIfInteresting(Instruction *I, Loop *L,
                               std::set<Instruction*> &Processed);
    SCEVHandle GetExpressionSCEV(Instruction *E, Loop *L);

    void OptimizeIndvars(Loop *L);

    void StrengthReduceStridedIVUsers(const SCEVHandle &Stride,
                                      IVUsersOfOneStride &Uses,
                                      Loop *L, bool isOnlyStride);
    void DeleteTriviallyDeadInstructions(std::set<Instruction*> &Insts);
  };
  RegisterOpt<LoopStrengthReduce> X("loop-reduce",
                                    "Strength Reduce GEP Uses of Ind. Vars");
}

FunctionPass *llvm::createLoopStrengthReducePass(unsigned MaxTargetAMSize) {
  return new LoopStrengthReduce(MaxTargetAMSize);
}

/// getCastedVersionOf - Return the specified value casted to uintptr_t.
///
Value *LoopStrengthReduce::getCastedVersionOf(Value *V) {
  if (V->getType() == UIntPtrTy) return V;
  if (Constant *CB = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(CB, UIntPtrTy);

  Value *&New = CastedPointers[V];
  if (New) return New;
  
  BasicBlock::iterator InsertPt;
  if (Argument *Arg = dyn_cast<Argument>(V)) {
    // Insert into the entry of the function, after any allocas.
    InsertPt = Arg->getParent()->begin()->begin();
    while (isa<AllocaInst>(InsertPt)) ++InsertPt;
  } else {
    if (InvokeInst *II = dyn_cast<InvokeInst>(V)) {
      InsertPt = II->getNormalDest()->begin();
    } else {
      InsertPt = cast<Instruction>(V);
      ++InsertPt;
    }

    // Do not insert casts into the middle of PHI node blocks.
    while (isa<PHINode>(InsertPt)) ++InsertPt;
  }
  
  New = new CastInst(V, UIntPtrTy, V->getName(), InsertPt);
  DeadInsts.insert(cast<Instruction>(New));
  return New;
}


/// DeleteTriviallyDeadInstructions - If any of the instructions is the
/// specified set are trivially dead, delete them and see if this makes any of
/// their operands subsequently dead.
void LoopStrengthReduce::
DeleteTriviallyDeadInstructions(std::set<Instruction*> &Insts) {
  while (!Insts.empty()) {
    Instruction *I = *Insts.begin();
    Insts.erase(Insts.begin());
    if (isInstructionTriviallyDead(I)) {
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (Instruction *U = dyn_cast<Instruction>(I->getOperand(i)))
          Insts.insert(U);
      SE->deleteInstructionFromRecords(I);
      I->eraseFromParent();
      Changed = true;
    }
  }
}


/// GetExpressionSCEV - Compute and return the SCEV for the specified
/// instruction.
SCEVHandle LoopStrengthReduce::GetExpressionSCEV(Instruction *Exp, Loop *L) {
  // Scalar Evolutions doesn't know how to compute SCEV's for GEP instructions.
  // If this is a GEP that SE doesn't know about, compute it now and insert it.
  // If this is not a GEP, or if we have already done this computation, just let
  // SE figure it out.
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Exp);
  if (!GEP || SE->hasSCEV(GEP))
    return SE->getSCEV(Exp);
    
  // Analyze all of the subscripts of this getelementptr instruction, looking
  // for uses that are determined by the trip count of L.  First, skip all
  // operands the are not dependent on the IV.

  // Build up the base expression.  Insert an LLVM cast of the pointer to
  // uintptr_t first.
  SCEVHandle GEPVal = SCEVUnknown::get(getCastedVersionOf(GEP->getOperand(0)));

  gep_type_iterator GTI = gep_type_begin(GEP);
  
  for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i, ++GTI) {
    // If this is a use of a recurrence that we can analyze, and it comes before
    // Op does in the GEP operand list, we will handle this when we process this
    // operand.
    if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
      const StructLayout *SL = TD->getStructLayout(STy);
      unsigned Idx = cast<ConstantUInt>(GEP->getOperand(i))->getValue();
      uint64_t Offset = SL->MemberOffsets[Idx];
      GEPVal = SCEVAddExpr::get(GEPVal,
                                SCEVUnknown::getIntegerSCEV(Offset, UIntPtrTy));
    } else {
      Value *OpVal = getCastedVersionOf(GEP->getOperand(i));
      SCEVHandle Idx = SE->getSCEV(OpVal);

      uint64_t TypeSize = TD->getTypeSize(GTI.getIndexedType());
      if (TypeSize != 1)
        Idx = SCEVMulExpr::get(Idx,
                               SCEVConstant::get(ConstantUInt::get(UIntPtrTy,
                                                                   TypeSize)));
      GEPVal = SCEVAddExpr::get(GEPVal, Idx);
    }
  }

  SE->setSCEV(GEP, GEPVal);
  return GEPVal;
}

/// getSCEVStartAndStride - Compute the start and stride of this expression,
/// returning false if the expression is not a start/stride pair, or true if it
/// is.  The stride must be a loop invariant expression, but the start may be
/// a mix of loop invariant and loop variant expressions.
static bool getSCEVStartAndStride(const SCEVHandle &SH, Loop *L,
                                  SCEVHandle &Start, SCEVHandle &Stride) {
  SCEVHandle TheAddRec = Start;   // Initialize to zero.

  // If the outer level is an AddExpr, the operands are all start values except
  // for a nested AddRecExpr.
  if (SCEVAddExpr *AE = dyn_cast<SCEVAddExpr>(SH)) {
    for (unsigned i = 0, e = AE->getNumOperands(); i != e; ++i)
      if (SCEVAddRecExpr *AddRec =
             dyn_cast<SCEVAddRecExpr>(AE->getOperand(i))) {
        if (AddRec->getLoop() == L)
          TheAddRec = SCEVAddExpr::get(AddRec, TheAddRec);
        else
          return false;  // Nested IV of some sort?
      } else {
        Start = SCEVAddExpr::get(Start, AE->getOperand(i));
      }
        
  } else if (SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(SH)) {
    TheAddRec = SH;
  } else {
    return false;  // not analyzable.
  }
  
  SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(TheAddRec);
  if (!AddRec || AddRec->getLoop() != L) return false;
  
  // FIXME: Generalize to non-affine IV's.
  if (!AddRec->isAffine()) return false;

  Start = SCEVAddExpr::get(Start, AddRec->getOperand(0));
  
  if (!isa<SCEVConstant>(AddRec->getOperand(1)))
    DEBUG(std::cerr << "[" << L->getHeader()->getName()
                    << "] Variable stride: " << *AddRec << "\n");

  Stride = AddRec->getOperand(1);
  // Check that all constant strides are the unsigned type, we don't want to
  // have two IV's one of signed stride 4 and one of unsigned stride 4 to not be
  // merged.
  assert((!isa<SCEVConstant>(Stride) || Stride->getType()->isUnsigned()) &&
         "Constants should be canonicalized to unsigned!");

  return true;
}

/// AddUsersIfInteresting - Inspect the specified instruction.  If it is a
/// reducible SCEV, recursively add its users to the IVUsesByStride set and
/// return true.  Otherwise, return false.
bool LoopStrengthReduce::AddUsersIfInteresting(Instruction *I, Loop *L,
                                            std::set<Instruction*> &Processed) {
  if (I->getType() == Type::VoidTy) return false;
  if (!Processed.insert(I).second)
    return true;    // Instruction already handled.
  
  // Get the symbolic expression for this instruction.
  SCEVHandle ISE = GetExpressionSCEV(I, L);
  if (isa<SCEVCouldNotCompute>(ISE)) return false;
  
  // Get the start and stride for this expression.
  SCEVHandle Start = SCEVUnknown::getIntegerSCEV(0, ISE->getType());
  SCEVHandle Stride = Start;
  if (!getSCEVStartAndStride(ISE, L, Start, Stride))
    return false;  // Non-reducible symbolic expression, bail out.
  
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;++UI){
    Instruction *User = cast<Instruction>(*UI);

    // Do not infinitely recurse on PHI nodes.
    if (isa<PHINode>(User) && User->getParent() == L->getHeader())
      continue;

    // If this is an instruction defined in a nested loop, or outside this loop,
    // don't recurse into it.
    bool AddUserToIVUsers = false;
    if (LI->getLoopFor(User->getParent()) != L) {
      DEBUG(std::cerr << "FOUND USER in nested loop: " << *User
            << "   OF SCEV: " << *ISE << "\n");
      AddUserToIVUsers = true;
    } else if (!AddUsersIfInteresting(User, L, Processed)) {
      DEBUG(std::cerr << "FOUND USER: " << *User
            << "   OF SCEV: " << *ISE << "\n");
      AddUserToIVUsers = true;
    }

    if (AddUserToIVUsers) {
      // Okay, we found a user that we cannot reduce.  Analyze the instruction
      // and decide what to do with it.  If we are a use inside of the loop, use
      // the value before incrementation, otherwise use it after incrementation.
      if (L->contains(User->getParent())) {
        IVUsesByStride[Stride].addUser(Start, User, I);
      } else {
        // The value used will be incremented by the stride more than we are
        // expecting, so subtract this off.
        SCEVHandle NewStart = SCEV::getMinusSCEV(Start, Stride);
        IVUsesByStride[Stride].addUser(NewStart, User, I);
        IVUsesByStride[Stride].Users.back().isUseOfPostIncrementedValue = true;
      }
    }
  }
  return true;
}

namespace {
  /// BasedUser - For a particular base value, keep information about how we've
  /// partitioned the expression so far.
  struct BasedUser {
    /// Base - The Base value for the PHI node that needs to be inserted for
    /// this use.  As the use is processed, information gets moved from this
    /// field to the Imm field (below).  BasedUser values are sorted by this
    /// field.
    SCEVHandle Base;
    
    /// Inst - The instruction using the induction variable.
    Instruction *Inst;

    /// OperandValToReplace - The operand value of Inst to replace with the
    /// EmittedBase.
    Value *OperandValToReplace;

    /// Imm - The immediate value that should be added to the base immediately
    /// before Inst, because it will be folded into the imm field of the
    /// instruction.
    SCEVHandle Imm;

    /// EmittedBase - The actual value* to use for the base value of this
    /// operation.  This is null if we should just use zero so far.
    Value *EmittedBase;

    // isUseOfPostIncrementedValue - True if this should use the
    // post-incremented version of this IV, not the preincremented version.
    // This can only be set in special cases, such as the terminating setcc
    // instruction for a loop and uses outside the loop that are dominated by
    // the loop.
    bool isUseOfPostIncrementedValue;
    
    BasedUser(IVStrideUse &IVSU)
      : Base(IVSU.Offset), Inst(IVSU.User), 
        OperandValToReplace(IVSU.OperandValToReplace), 
        Imm(SCEVUnknown::getIntegerSCEV(0, Base->getType())), EmittedBase(0),
        isUseOfPostIncrementedValue(IVSU.isUseOfPostIncrementedValue) {}

    // Once we rewrite the code to insert the new IVs we want, update the
    // operands of Inst to use the new expression 'NewBase', with 'Imm' added
    // to it.
    void RewriteInstructionToUseNewBase(const SCEVHandle &NewBase,
                                        SCEVExpander &Rewriter, Loop *L,
                                        Pass *P);

    // Sort by the Base field.
    bool operator<(const BasedUser &BU) const { return Base < BU.Base; }

    void dump() const;
  };
}

void BasedUser::dump() const {
  std::cerr << " Base=" << *Base;
  std::cerr << " Imm=" << *Imm;
  if (EmittedBase)
    std::cerr << "  EB=" << *EmittedBase;

  std::cerr << "   Inst: " << *Inst;
}

// Once we rewrite the code to insert the new IVs we want, update the
// operands of Inst to use the new expression 'NewBase', with 'Imm' added
// to it.
void BasedUser::RewriteInstructionToUseNewBase(const SCEVHandle &NewBase,
                                               SCEVExpander &Rewriter,
                                               Loop *L, Pass *P) {
  if (!isa<PHINode>(Inst)) {
    SCEVHandle NewValSCEV = SCEVAddExpr::get(NewBase, Imm);
    Value *NewVal = Rewriter.expandCodeFor(NewValSCEV, Inst,
                                           OperandValToReplace->getType());
    // Replace the use of the operand Value with the new Phi we just created.
    Inst->replaceUsesOfWith(OperandValToReplace, NewVal);
    DEBUG(std::cerr << "    CHANGED: IMM =" << *Imm << "  Inst = " << *Inst);
    return;
  }
  
  // PHI nodes are more complex.  We have to insert one copy of the NewBase+Imm
  // expression into each operand block that uses it.  Note that PHI nodes can
  // have multiple entries for the same predecessor.  We use a map to make sure
  // that a PHI node only has a single Value* for each predecessor (which also
  // prevents us from inserting duplicate code in some blocks).
  std::map<BasicBlock*, Value*> InsertedCode;
  PHINode *PN = cast<PHINode>(Inst);
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    if (PN->getIncomingValue(i) == OperandValToReplace) {
      // If this is a critical edge, split the edge so that we do not insert the
      // code on all predecessor/successor paths.
      if (e != 1 &&
          PN->getIncomingBlock(i)->getTerminator()->getNumSuccessors() > 1) {

        // First step, split the critical edge.
        SplitCriticalEdge(PN->getIncomingBlock(i), PN->getParent(), P);
            
        // Next step: move the basic block.  In particular, if the PHI node
        // is outside of the loop, and PredTI is in the loop, we want to
        // move the block to be immediately before the PHI block, not
        // immediately after PredTI.
        if (L->contains(PN->getIncomingBlock(i)) &&
            !L->contains(PN->getParent())) {
          BasicBlock *NewBB = PN->getIncomingBlock(i);
          NewBB->moveBefore(PN->getParent());
        }
        break;
      }

      Value *&Code = InsertedCode[PN->getIncomingBlock(i)];
      if (!Code) {
        // Insert the code into the end of the predecessor block.
        BasicBlock::iterator InsertPt =PN->getIncomingBlock(i)->getTerminator();
      
        SCEVHandle NewValSCEV = SCEVAddExpr::get(NewBase, Imm);
        Code = Rewriter.expandCodeFor(NewValSCEV, InsertPt,
                                      OperandValToReplace->getType());
      }
      
      // Replace the use of the operand Value with the new Phi we just created.
      PN->setIncomingValue(i, Code);
      Rewriter.clear();
    }
  }
  DEBUG(std::cerr << "    CHANGED: IMM =" << *Imm << "  Inst = " << *Inst);
}


/// isTargetConstant - Return true if the following can be referenced by the
/// immediate field of a target instruction.
static bool isTargetConstant(const SCEVHandle &V) {

  // FIXME: Look at the target to decide if &GV is a legal constant immediate.
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(V)) {
    // PPC allows a sign-extended 16-bit immediate field.
    if ((int64_t)SC->getValue()->getRawValue() > -(1 << 16) &&
        (int64_t)SC->getValue()->getRawValue() < (1 << 16)-1)
      return true;
    return false;
  }

  return false;     // ENABLE this for x86

  if (SCEVUnknown *SU = dyn_cast<SCEVUnknown>(V))
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(SU->getValue()))
      if (CE->getOpcode() == Instruction::Cast)
        if (isa<GlobalValue>(CE->getOperand(0)))
          // FIXME: should check to see that the dest is uintptr_t!
          return true;
  return false;
}

/// MoveLoopVariantsToImediateField - Move any subexpressions from Val that are
/// loop varying to the Imm operand.
static void MoveLoopVariantsToImediateField(SCEVHandle &Val, SCEVHandle &Imm,
                                            Loop *L) {
  if (Val->isLoopInvariant(L)) return;  // Nothing to do.
  
  if (SCEVAddExpr *SAE = dyn_cast<SCEVAddExpr>(Val)) {
    std::vector<SCEVHandle> NewOps;
    NewOps.reserve(SAE->getNumOperands());
    
    for (unsigned i = 0; i != SAE->getNumOperands(); ++i)
      if (!SAE->getOperand(i)->isLoopInvariant(L)) {
        // If this is a loop-variant expression, it must stay in the immediate
        // field of the expression.
        Imm = SCEVAddExpr::get(Imm, SAE->getOperand(i));
      } else {
        NewOps.push_back(SAE->getOperand(i));
      }

    if (NewOps.empty())
      Val = SCEVUnknown::getIntegerSCEV(0, Val->getType());
    else
      Val = SCEVAddExpr::get(NewOps);
  } else if (SCEVAddRecExpr *SARE = dyn_cast<SCEVAddRecExpr>(Val)) {
    // Try to pull immediates out of the start value of nested addrec's.
    SCEVHandle Start = SARE->getStart();
    MoveLoopVariantsToImediateField(Start, Imm, L);
    
    std::vector<SCEVHandle> Ops(SARE->op_begin(), SARE->op_end());
    Ops[0] = Start;
    Val = SCEVAddRecExpr::get(Ops, SARE->getLoop());
  } else {
    // Otherwise, all of Val is variant, move the whole thing over.
    Imm = SCEVAddExpr::get(Imm, Val);
    Val = SCEVUnknown::getIntegerSCEV(0, Val->getType());
  }
}


/// MoveImmediateValues - Look at Val, and pull out any additions of constants
/// that can fit into the immediate field of instructions in the target.
/// Accumulate these immediate values into the Imm value.
static void MoveImmediateValues(SCEVHandle &Val, SCEVHandle &Imm,
                                bool isAddress, Loop *L) {
  if (SCEVAddExpr *SAE = dyn_cast<SCEVAddExpr>(Val)) {
    std::vector<SCEVHandle> NewOps;
    NewOps.reserve(SAE->getNumOperands());
    
    for (unsigned i = 0; i != SAE->getNumOperands(); ++i)
      if (isAddress && isTargetConstant(SAE->getOperand(i))) {
        Imm = SCEVAddExpr::get(Imm, SAE->getOperand(i));
      } else if (!SAE->getOperand(i)->isLoopInvariant(L)) {
        // If this is a loop-variant expression, it must stay in the immediate
        // field of the expression.
        Imm = SCEVAddExpr::get(Imm, SAE->getOperand(i));
      } else {
        NewOps.push_back(SAE->getOperand(i));
      }

    if (NewOps.empty())
      Val = SCEVUnknown::getIntegerSCEV(0, Val->getType());
    else
      Val = SCEVAddExpr::get(NewOps);
    return;
  } else if (SCEVAddRecExpr *SARE = dyn_cast<SCEVAddRecExpr>(Val)) {
    // Try to pull immediates out of the start value of nested addrec's.
    SCEVHandle Start = SARE->getStart();
    MoveImmediateValues(Start, Imm, isAddress, L);
    
    if (Start != SARE->getStart()) {
      std::vector<SCEVHandle> Ops(SARE->op_begin(), SARE->op_end());
      Ops[0] = Start;
      Val = SCEVAddRecExpr::get(Ops, SARE->getLoop());
    }
    return;
  }

  // Loop-variant expressions must stay in the immediate field of the
  // expression.
  if ((isAddress && isTargetConstant(Val)) ||
      !Val->isLoopInvariant(L)) {
    Imm = SCEVAddExpr::get(Imm, Val);
    Val = SCEVUnknown::getIntegerSCEV(0, Val->getType());
    return;
  }

  // Otherwise, no immediates to move.
}


/// IncrementAddExprUses - Decompose the specified expression into its added
/// subexpressions, and increment SubExpressionUseCounts for each of these
/// decomposed parts.
static void SeparateSubExprs(std::vector<SCEVHandle> &SubExprs,
                             SCEVHandle Expr) {
  if (SCEVAddExpr *AE = dyn_cast<SCEVAddExpr>(Expr)) {
    for (unsigned j = 0, e = AE->getNumOperands(); j != e; ++j)
      SeparateSubExprs(SubExprs, AE->getOperand(j));
  } else if (SCEVAddRecExpr *SARE = dyn_cast<SCEVAddRecExpr>(Expr)) {
    SCEVHandle Zero = SCEVUnknown::getIntegerSCEV(0, Expr->getType());
    if (SARE->getOperand(0) == Zero) {
      SubExprs.push_back(Expr);
    } else {
      // Compute the addrec with zero as its base.
      std::vector<SCEVHandle> Ops(SARE->op_begin(), SARE->op_end());
      Ops[0] = Zero;   // Start with zero base.
      SubExprs.push_back(SCEVAddRecExpr::get(Ops, SARE->getLoop()));
      

      SeparateSubExprs(SubExprs, SARE->getOperand(0));
    }
  } else if (!isa<SCEVConstant>(Expr) ||
             !cast<SCEVConstant>(Expr)->getValue()->isNullValue()) {
    // Do not add zero.
    SubExprs.push_back(Expr);
  }
}


/// RemoveCommonExpressionsFromUseBases - Look through all of the uses in Bases,
/// removing any common subexpressions from it.  Anything truly common is
/// removed, accumulated, and returned.  This looks for things like (a+b+c) and
/// (a+c+d) -> (a+c).  The common expression is *removed* from the Bases.
static SCEVHandle 
RemoveCommonExpressionsFromUseBases(std::vector<BasedUser> &Uses) {
  unsigned NumUses = Uses.size();

  // Only one use?  Use its base, regardless of what it is!
  SCEVHandle Zero = SCEVUnknown::getIntegerSCEV(0, Uses[0].Base->getType());
  SCEVHandle Result = Zero;
  if (NumUses == 1) {
    std::swap(Result, Uses[0].Base);
    return Result;
  }

  // To find common subexpressions, count how many of Uses use each expression.
  // If any subexpressions are used Uses.size() times, they are common.
  std::map<SCEVHandle, unsigned> SubExpressionUseCounts;
  
  std::vector<SCEVHandle> SubExprs;
  for (unsigned i = 0; i != NumUses; ++i) {
    // If the base is zero (which is common), return zero now, there are no
    // CSEs we can find.
    if (Uses[i].Base == Zero) return Zero;

    // Split the expression into subexprs.
    SeparateSubExprs(SubExprs, Uses[i].Base);
    // Add one to SubExpressionUseCounts for each subexpr present.
    for (unsigned j = 0, e = SubExprs.size(); j != e; ++j)
      SubExpressionUseCounts[SubExprs[j]]++;
    SubExprs.clear();
  }


  // Now that we know how many times each is used, build Result.
  for (std::map<SCEVHandle, unsigned>::iterator I =
       SubExpressionUseCounts.begin(), E = SubExpressionUseCounts.end();
       I != E; )
    if (I->second == NumUses) {  // Found CSE!
      Result = SCEVAddExpr::get(Result, I->first);
      ++I;
    } else {
      // Remove non-cse's from SubExpressionUseCounts.
      SubExpressionUseCounts.erase(I++);
    }
  
  // If we found no CSE's, return now.
  if (Result == Zero) return Result;
  
  // Otherwise, remove all of the CSE's we found from each of the base values.
  for (unsigned i = 0; i != NumUses; ++i) {
    // Split the expression into subexprs.
    SeparateSubExprs(SubExprs, Uses[i].Base);

    // Remove any common subexpressions.
    for (unsigned j = 0, e = SubExprs.size(); j != e; ++j)
      if (SubExpressionUseCounts.count(SubExprs[j])) {
        SubExprs.erase(SubExprs.begin()+j);
        --j; --e;
      }
    
    // Finally, the non-shared expressions together.
    if (SubExprs.empty())
      Uses[i].Base = Zero;
    else
      Uses[i].Base = SCEVAddExpr::get(SubExprs);
    SubExprs.clear();
  }
 
  return Result;
}


/// StrengthReduceStridedIVUsers - Strength reduce all of the users of a single
/// stride of IV.  All of the users may have different starting values, and this
/// may not be the only stride (we know it is if isOnlyStride is true).
void LoopStrengthReduce::StrengthReduceStridedIVUsers(const SCEVHandle &Stride,
                                                      IVUsersOfOneStride &Uses,
                                                      Loop *L,
                                                      bool isOnlyStride) {
  // Transform our list of users and offsets to a bit more complex table.  In
  // this new vector, each 'BasedUser' contains 'Base' the base of the
  // strided accessas well as the old information from Uses.  We progressively
  // move information from the Base field to the Imm field, until we eventually
  // have the full access expression to rewrite the use.
  std::vector<BasedUser> UsersToProcess;
  UsersToProcess.reserve(Uses.Users.size());
  for (unsigned i = 0, e = Uses.Users.size(); i != e; ++i) {
    UsersToProcess.push_back(Uses.Users[i]);
    
    // Move any loop invariant operands from the offset field to the immediate
    // field of the use, so that we don't try to use something before it is
    // computed.
    MoveLoopVariantsToImediateField(UsersToProcess.back().Base,
                                    UsersToProcess.back().Imm, L);
    assert(UsersToProcess.back().Base->isLoopInvariant(L) &&
           "Base value is not loop invariant!");
  }
  
  // We now have a whole bunch of uses of like-strided induction variables, but
  // they might all have different bases.  We want to emit one PHI node for this
  // stride which we fold as many common expressions (between the IVs) into as
  // possible.  Start by identifying the common expressions in the base values 
  // for the strides (e.g. if we have "A+C+B" and "A+B+D" as our bases, find
  // "A+B"), emit it to the preheader, then remove the expression from the
  // UsersToProcess base values.
  SCEVHandle CommonExprs = RemoveCommonExpressionsFromUseBases(UsersToProcess);
  
  // Next, figure out what we can represent in the immediate fields of
  // instructions.  If we can represent anything there, move it to the imm
  // fields of the BasedUsers.  We do this so that it increases the commonality
  // of the remaining uses.
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ++i) {
    // If the user is not in the current loop, this means it is using the exit
    // value of the IV.  Do not put anything in the base, make sure it's all in
    // the immediate field to allow as much factoring as possible.
    if (!L->contains(UsersToProcess[i].Inst->getParent())) {
      UsersToProcess[i].Imm = SCEVAddExpr::get(UsersToProcess[i].Imm,
                                               UsersToProcess[i].Base);
      UsersToProcess[i].Base = 
        SCEVUnknown::getIntegerSCEV(0, UsersToProcess[i].Base->getType());
    } else {
      
      // Addressing modes can be folded into loads and stores.  Be careful that
      // the store is through the expression, not of the expression though.
      bool isAddress = isa<LoadInst>(UsersToProcess[i].Inst);
      if (StoreInst *SI = dyn_cast<StoreInst>(UsersToProcess[i].Inst))
        if (SI->getOperand(1) == UsersToProcess[i].OperandValToReplace)
          isAddress = true;
      
      MoveImmediateValues(UsersToProcess[i].Base, UsersToProcess[i].Imm,
                          isAddress, L);
    }
  }
 
  // Now that we know what we need to do, insert the PHI node itself.
  //
  DEBUG(std::cerr << "INSERTING IV of STRIDE " << *Stride << " and BASE "
        << *CommonExprs << " :\n");
    
  SCEVExpander Rewriter(*SE, *LI);
  SCEVExpander PreheaderRewriter(*SE, *LI);
  
  BasicBlock  *Preheader = L->getLoopPreheader();
  Instruction *PreInsertPt = Preheader->getTerminator();
  Instruction *PhiInsertBefore = L->getHeader()->begin();
  
  assert(isa<PHINode>(PhiInsertBefore) &&
         "How could this loop have IV's without any phis?");
  PHINode *SomeLoopPHI = cast<PHINode>(PhiInsertBefore);
  assert(SomeLoopPHI->getNumIncomingValues() == 2 &&
         "This loop isn't canonicalized right");
  BasicBlock *LatchBlock =
   SomeLoopPHI->getIncomingBlock(SomeLoopPHI->getIncomingBlock(0) == Preheader);
  
  // Create a new Phi for this base, and stick it in the loop header.
  const Type *ReplacedTy = CommonExprs->getType();
  PHINode *NewPHI = new PHINode(ReplacedTy, "iv.", PhiInsertBefore);
  ++NumInserted;
  
  // Insert the stride into the preheader.
  Value *StrideV = PreheaderRewriter.expandCodeFor(Stride, PreInsertPt,
                                                   ReplacedTy);
  if (!isa<ConstantInt>(StrideV)) ++NumVariable;


  // Emit the initial base value into the loop preheader, and add it to the
  // Phi node.
  Value *PHIBaseV = PreheaderRewriter.expandCodeFor(CommonExprs, PreInsertPt,
                                                    ReplacedTy);
  NewPHI->addIncoming(PHIBaseV, Preheader);
  
  // Emit the increment of the base value before the terminator of the loop
  // latch block, and add it to the Phi node.
  SCEVHandle IncExp = SCEVAddExpr::get(SCEVUnknown::get(NewPHI),
                                       SCEVUnknown::get(StrideV));
  
  Value *IncV = Rewriter.expandCodeFor(IncExp, LatchBlock->getTerminator(),
                                       ReplacedTy);
  IncV->setName(NewPHI->getName()+".inc");
  NewPHI->addIncoming(IncV, LatchBlock);

  // Sort by the base value, so that all IVs with identical bases are next to
  // each other.
  std::sort(UsersToProcess.begin(), UsersToProcess.end());
  while (!UsersToProcess.empty()) {
    SCEVHandle Base = UsersToProcess.front().Base;

    DEBUG(std::cerr << "  INSERTING code for BASE = " << *Base << ":\n");
   
    // Emit the code for Base into the preheader.
    Value *BaseV = PreheaderRewriter.expandCodeFor(Base, PreInsertPt,
                                                   ReplacedTy);
    
    // If BaseV is a constant other than 0, make sure that it gets inserted into
    // the preheader, instead of being forward substituted into the uses.  We do
    // this by forcing a noop cast to be inserted into the preheader in this
    // case.
    if (Constant *C = dyn_cast<Constant>(BaseV))
      if (!C->isNullValue() && !isTargetConstant(Base)) {
        // We want this constant emitted into the preheader!
        BaseV = new CastInst(BaseV, BaseV->getType(), "preheaderinsert",
                             PreInsertPt);       
      }
    
    // Emit the code to add the immediate offset to the Phi value, just before
    // the instructions that we identified as using this stride and base.
    while (!UsersToProcess.empty() && UsersToProcess.front().Base == Base) {
      BasedUser &User = UsersToProcess.front();

      // If this instruction wants to use the post-incremented value, move it
      // after the post-inc and use its value instead of the PHI.
      Value *RewriteOp = NewPHI;
      if (User.isUseOfPostIncrementedValue) {
        RewriteOp = IncV;

        // If this user is in the loop, make sure it is the last thing in the
        // loop to ensure it is dominated by the increment.
        if (L->contains(User.Inst->getParent()))
          User.Inst->moveBefore(LatchBlock->getTerminator());
      }
      SCEVHandle RewriteExpr = SCEVUnknown::get(RewriteOp);

      // Clear the SCEVExpander's expression map so that we are guaranteed
      // to have the code emitted where we expect it.
      Rewriter.clear();
     
      // Now that we know what we need to do, insert code before User for the
      // immediate and any loop-variant expressions.
      if (!isa<ConstantInt>(BaseV) || !cast<ConstantInt>(BaseV)->isNullValue())
        // Add BaseV to the PHI value if needed.
        RewriteExpr = SCEVAddExpr::get(RewriteExpr, SCEVUnknown::get(BaseV));
      
      User.RewriteInstructionToUseNewBase(RewriteExpr, Rewriter, L, this);

      // Mark old value we replaced as possibly dead, so that it is elminated
      // if we just replaced the last use of that value.
      DeadInsts.insert(cast<Instruction>(User.OperandValToReplace));

      UsersToProcess.erase(UsersToProcess.begin());
      ++NumReduced;
    }
    // TODO: Next, find out which base index is the most common, pull it out.
  }

  // IMPORTANT TODO: Figure out how to partition the IV's with this stride, but
  // different starting values, into different PHIs.
}

// OptimizeIndvars - Now that IVUsesByStride is set up with all of the indvar
// uses in the loop, look to see if we can eliminate some, in favor of using
// common indvars for the different uses.
void LoopStrengthReduce::OptimizeIndvars(Loop *L) {
  // TODO: implement optzns here.




  // Finally, get the terminating condition for the loop if possible.  If we
  // can, we want to change it to use a post-incremented version of its
  // induction variable, to allow coallescing the live ranges for the IV into
  // one register value.
  PHINode *SomePHI = cast<PHINode>(L->getHeader()->begin());
  BasicBlock  *Preheader = L->getLoopPreheader();
  BasicBlock *LatchBlock =
   SomePHI->getIncomingBlock(SomePHI->getIncomingBlock(0) == Preheader);
  BranchInst *TermBr = dyn_cast<BranchInst>(LatchBlock->getTerminator());
  if (!TermBr || TermBr->isUnconditional() ||
      !isa<SetCondInst>(TermBr->getCondition()))
    return;
  SetCondInst *Cond = cast<SetCondInst>(TermBr->getCondition());

  // Search IVUsesByStride to find Cond's IVUse if there is one.
  IVStrideUse *CondUse = 0;
  const SCEVHandle *CondStride = 0;

  for (std::map<SCEVHandle, IVUsersOfOneStride>::iterator 
         I = IVUsesByStride.begin(), E = IVUsesByStride.end();
       I != E && !CondUse; ++I)
    for (std::vector<IVStrideUse>::iterator UI = I->second.Users.begin(),
           E = I->second.Users.end(); UI != E; ++UI)
      if (UI->User == Cond) {
        CondUse = &*UI;
        CondStride = &I->first;
        // NOTE: we could handle setcc instructions with multiple uses here, but
        // InstCombine does it as well for simple uses, it's not clear that it
        // occurs enough in real life to handle.
        break;
      }
  if (!CondUse) return;  // setcc doesn't use the IV.

  // setcc stride is complex, don't mess with users.
  // FIXME: Evaluate whether this is a good idea or not.
  if (!isa<SCEVConstant>(*CondStride)) return;

  // It's possible for the setcc instruction to be anywhere in the loop, and
  // possible for it to have multiple users.  If it is not immediately before
  // the latch block branch, move it.
  if (&*++BasicBlock::iterator(Cond) != (Instruction*)TermBr) {
    if (Cond->hasOneUse()) {   // Condition has a single use, just move it.
      Cond->moveBefore(TermBr);
    } else {
      // Otherwise, clone the terminating condition and insert into the loopend.
      Cond = cast<SetCondInst>(Cond->clone());
      Cond->setName(L->getHeader()->getName() + ".termcond");
      LatchBlock->getInstList().insert(TermBr, Cond);
      
      // Clone the IVUse, as the old use still exists!
      IVUsesByStride[*CondStride].addUser(CondUse->Offset, Cond,
                                         CondUse->OperandValToReplace);
      CondUse = &IVUsesByStride[*CondStride].Users.back();
    }
  }

  // If we get to here, we know that we can transform the setcc instruction to
  // use the post-incremented version of the IV, allowing us to coallesce the
  // live ranges for the IV correctly.
  CondUse->Offset = SCEV::getMinusSCEV(CondUse->Offset, *CondStride);
  CondUse->isUseOfPostIncrementedValue = true;
}

void LoopStrengthReduce::runOnLoop(Loop *L) {
  // First step, transform all loops nesting inside of this loop.
  for (LoopInfo::iterator I = L->begin(), E = L->end(); I != E; ++I)
    runOnLoop(*I);

  // Next, find all uses of induction variables in this loop, and catagorize
  // them by stride.  Start by finding all of the PHI nodes in the header for
  // this loop.  If they are induction variables, inspect their uses.
  std::set<Instruction*> Processed;   // Don't reprocess instructions.
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I)
    AddUsersIfInteresting(I, L, Processed);

  // If we have nothing to do, return.
  if (IVUsesByStride.empty()) return;

  // Optimize induction variables.  Some indvar uses can be transformed to use
  // strides that will be needed for other purposes.  A common example of this
  // is the exit test for the loop, which can often be rewritten to use the
  // computation of some other indvar to decide when to terminate the loop.
  OptimizeIndvars(L);


  // FIXME: We can widen subreg IV's here for RISC targets.  e.g. instead of
  // doing computation in byte values, promote to 32-bit values if safe.

  // FIXME: Attempt to reuse values across multiple IV's.  In particular, we
  // could have something like "for(i) { foo(i*8); bar(i*16) }", which should be
  // codegened as "for (j = 0;; j+=8) { foo(j); bar(j+j); }" on X86/PPC.  Need
  // to be careful that IV's are all the same type.  Only works for intptr_t
  // indvars.

  // If we only have one stride, we can more aggressively eliminate some things.
  bool HasOneStride = IVUsesByStride.size() == 1;

  // Note: this processes each stride/type pair individually.  All users passed
  // into StrengthReduceStridedIVUsers have the same type AND stride.
  for (std::map<SCEVHandle, IVUsersOfOneStride>::iterator SI
        = IVUsesByStride.begin(), E = IVUsesByStride.end(); SI != E; ++SI)
    StrengthReduceStridedIVUsers(SI->first, SI->second, L, HasOneStride);

  // Clean up after ourselves
  if (!DeadInsts.empty()) {
    DeleteTriviallyDeadInstructions(DeadInsts);

    BasicBlock::iterator I = L->getHeader()->begin();
    PHINode *PN;
    while ((PN = dyn_cast<PHINode>(I))) {
      ++I;  // Preincrement iterator to avoid invalidating it when deleting PN.
      
      // At this point, we know that we have killed one or more GEP
      // instructions.  It is worth checking to see if the cann indvar is also
      // dead, so that we can remove it as well.  The requirements for the cann
      // indvar to be considered dead are:
      // 1. the cann indvar has one use
      // 2. the use is an add instruction
      // 3. the add has one use
      // 4. the add is used by the cann indvar
      // If all four cases above are true, then we can remove both the add and
      // the cann indvar.
      // FIXME: this needs to eliminate an induction variable even if it's being
      // compared against some value to decide loop termination.
      if (PN->hasOneUse()) {
        BinaryOperator *BO = dyn_cast<BinaryOperator>(*(PN->use_begin()));
        if (BO && BO->hasOneUse()) {
          if (PN == *(BO->use_begin())) {
            DeadInsts.insert(BO);
            // Break the cycle, then delete the PHI.
            PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
            SE->deleteInstructionFromRecords(PN);
            PN->eraseFromParent();
          }
        }
      }
    }
    DeleteTriviallyDeadInstructions(DeadInsts);
  }

  CastedPointers.clear();
  IVUsesByStride.clear();
  return;
}
