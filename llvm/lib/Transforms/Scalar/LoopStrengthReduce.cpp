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

  /// IVStrideUse - Keep track of one use of a strided induction variable, where
  /// the stride is stored externally.  The Offset member keeps track of the 
  /// offset from the IV, User is the actual user of the operand, and 'Operand'
  /// is the operand # of the User that is the use.
  struct IVStrideUse {
    SCEVHandle Offset;
    Instruction *User;
    Value *OperandValToReplace;
    
    IVStrideUse(const SCEVHandle &Offs, Instruction *U, Value *O)
      : Offset(Offs), User(U), OperandValToReplace(O) {}
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
    std::map<Value*, IVUsersOfOneStride> IVUsesByStride;

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
      AU.setPreservesCFG();
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


    void StrengthReduceStridedIVUsers(Value *Stride, IVUsersOfOneStride &Uses,
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
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Exp);
  if (!GEP)
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

  return GEPVal;
}

/// getSCEVStartAndStride - Compute the start and stride of this expression,
/// returning false if the expression is not a start/stride pair, or true if it
/// is.  The stride must be a loop invariant expression, but the start may be
/// a mix of loop invariant and loop variant expressions.
static bool getSCEVStartAndStride(const SCEVHandle &SH, Loop *L,
                                  SCEVHandle &Start, Value *&Stride) {
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
  
  // FIXME: generalize to IV's with more complex strides (must emit stride
  // expression outside of loop!)
  if (!isa<SCEVConstant>(AddRec->getOperand(1)))
    return false;
  
  SCEVConstant *StrideC = cast<SCEVConstant>(AddRec->getOperand(1));
  Stride = StrideC->getValue();

  assert(Stride->getType()->isUnsigned() &&
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
  Value *Stride = 0;
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
      // and decide what to do with it.
      IVUsesByStride[Stride].addUser(Start, User, I);
    }
  }
  return true;
}

namespace {
  /// BasedUser - For a particular base value, keep information about how we've
  /// partitioned the expression so far.
  struct BasedUser {
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

    BasedUser(Instruction *I, Value *Op, const SCEVHandle &IMM)
      : Inst(I), OperandValToReplace(Op), Imm(IMM), EmittedBase(0) {}

    // Once we rewrite the code to insert the new IVs we want, update the
    // operands of Inst to use the new expression 'NewBase', with 'Imm' added
    // to it.
    void RewriteInstructionToUseNewBase(Value *NewBase, SCEVExpander &Rewriter);

    // No need to compare these.
    bool operator<(const BasedUser &BU) const { return 0; }

    void dump() const;
  };
}

void BasedUser::dump() const {
  std::cerr << " Imm=" << *Imm;
  if (EmittedBase)
    std::cerr << "  EB=" << *EmittedBase;

  std::cerr << "   Inst: " << *Inst;
}

// Once we rewrite the code to insert the new IVs we want, update the
// operands of Inst to use the new expression 'NewBase', with 'Imm' added
// to it.
void BasedUser::RewriteInstructionToUseNewBase(Value *NewBase,
                                               SCEVExpander &Rewriter) {
  if (!isa<PHINode>(Inst)) {
    SCEVHandle NewValSCEV = SCEVAddExpr::get(SCEVUnknown::get(NewBase), Imm);
    Value *NewVal = Rewriter.expandCodeFor(NewValSCEV, Inst,
                                           OperandValToReplace->getType());
    
    // Replace the use of the operand Value with the new Phi we just created.
    Inst->replaceUsesOfWith(OperandValToReplace, NewVal);
    DEBUG(std::cerr << "    CHANGED: IMM =" << *Imm << "  Inst = " << *Inst);
    return;
  }
  
  // PHI nodes are more complex.  We have to insert one copy of the NewBase+Imm
  // expression into each operand block that uses it.
  PHINode *PN = cast<PHINode>(Inst);
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    if (PN->getIncomingValue(i) == OperandValToReplace) {
      // FIXME: this should split any critical edges.

      // Insert the code into the end of the predecessor block.
      BasicBlock::iterator InsertPt = PN->getIncomingBlock(i)->getTerminator();
      
      SCEVHandle NewValSCEV = SCEVAddExpr::get(SCEVUnknown::get(NewBase), Imm);
      Value *NewVal = Rewriter.expandCodeFor(NewValSCEV, InsertPt,
                                             OperandValToReplace->getType());
      
      // Replace the use of the operand Value with the new Phi we just created.
      PN->setIncomingValue(i, NewVal);
      Rewriter.clear();
    }
  }
  DEBUG(std::cerr << "    CHANGED: IMM =" << *Imm << "  Inst = " << *Inst);
}


/// isTargetConstant - Return true if the following can be referenced by the
/// immediate field of a target instruction.
static bool isTargetConstant(const SCEVHandle &V) {

  // FIXME: Look at the target to decide if &GV is a legal constant immediate.
  if (isa<SCEVConstant>(V)) return true;

  return false;     // ENABLE this for x86

  if (SCEVUnknown *SU = dyn_cast<SCEVUnknown>(V))
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(SU->getValue()))
      if (CE->getOpcode() == Instruction::Cast)
        if (isa<GlobalValue>(CE->getOperand(0)))
          // FIXME: should check to see that the dest is uintptr_t!
          return true;
  return false;
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

/// StrengthReduceStridedIVUsers - Strength reduce all of the users of a single
/// stride of IV.  All of the users may have different starting values, and this
/// may not be the only stride (we know it is if isOnlyStride is true).
void LoopStrengthReduce::StrengthReduceStridedIVUsers(Value *Stride,
                                                      IVUsersOfOneStride &Uses,
                                                      Loop *L,
                                                      bool isOnlyStride) {
  // Transform our list of users and offsets to a bit more complex table.  In
  // this new vector, the first entry for each element is the base of the
  // strided access, and the second is the BasedUser object for the use.  We
  // progressively move information from the first to the second entry, until we
  // eventually emit the object.
  std::vector<std::pair<SCEVHandle, BasedUser> > UsersToProcess;
  UsersToProcess.reserve(Uses.Users.size());

  SCEVHandle ZeroBase = SCEVUnknown::getIntegerSCEV(0,
                                              Uses.Users[0].Offset->getType());

  for (unsigned i = 0, e = Uses.Users.size(); i != e; ++i)
    UsersToProcess.push_back(std::make_pair(Uses.Users[i].Offset,
                                            BasedUser(Uses.Users[i].User,
                                             Uses.Users[i].OperandValToReplace,
                                                      ZeroBase)));

  // First pass, figure out what we can represent in the immediate fields of
  // instructions.  If we can represent anything there, move it to the imm
  // fields of the BasedUsers.
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ++i) {
    // Addressing modes can be folded into loads and stores.  Be careful that
    // the store is through the expression, not of the expression though.
    bool isAddress = isa<LoadInst>(UsersToProcess[i].second.Inst);
    if (StoreInst *SI = dyn_cast<StoreInst>(UsersToProcess[i].second.Inst))
      if (SI->getOperand(1) == UsersToProcess[i].second.OperandValToReplace)
        isAddress = true;
          
    MoveImmediateValues(UsersToProcess[i].first, UsersToProcess[i].second.Imm,
                        isAddress, L);

    assert(UsersToProcess[i].first->isLoopInvariant(L) &&
           "Base value is not loop invariant!");
  }

  SCEVExpander Rewriter(*SE, *LI);
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

  DEBUG(std::cerr << "INSERTING IVs of STRIDE " << *Stride << ":\n");
  
  // FIXME: This loop needs increasing levels of intelligence.
  // STAGE 0: just emit everything as its own base.
  // STAGE 1: factor out common vars from bases, and try and push resulting
  //          constants into Imm field.  <-- We are here
  // STAGE 2: factor out large constants to try and make more constants
  //          acceptable for target loads and stores.

  // Sort by the base value, so that all IVs with identical bases are next to
  // each other.  
  std::sort(UsersToProcess.begin(), UsersToProcess.end());
  while (!UsersToProcess.empty()) {
    SCEVHandle Base = UsersToProcess.front().first;

    DEBUG(std::cerr << "  INSERTING PHI with BASE = " << *Base << ":\n");
   
    // Create a new Phi for this base, and stick it in the loop header.
    const Type *ReplacedTy = Base->getType();
    PHINode *NewPHI = new PHINode(ReplacedTy, "iv.", PhiInsertBefore);
    ++NumInserted;

    // Emit the initial base value into the loop preheader, and add it to the
    // Phi node.
    Value *BaseV = Rewriter.expandCodeFor(Base, PreInsertPt, ReplacedTy);
    NewPHI->addIncoming(BaseV, Preheader);

    // Emit the increment of the base value before the terminator of the loop
    // latch block, and add it to the Phi node.
    SCEVHandle Inc = SCEVAddExpr::get(SCEVUnknown::get(NewPHI),
                                      SCEVUnknown::get(Stride));

    Value *IncV = Rewriter.expandCodeFor(Inc, LatchBlock->getTerminator(),
                                         ReplacedTy);
    IncV->setName(NewPHI->getName()+".inc");
    NewPHI->addIncoming(IncV, LatchBlock);

    // Emit the code to add the immediate offset to the Phi value, just before
    // the instructions that we identified as using this stride and base.
    while (!UsersToProcess.empty() && UsersToProcess.front().first == Base) {
      BasedUser &User = UsersToProcess.front().second;

      // Clear the SCEVExpander's expression map so that we are guaranteed
      // to have the code emitted where we expect it.
      Rewriter.clear();
      
      // Now that we know what we need to do, insert code before User for the
      // immediate and any loop-variant expressions.
      User.RewriteInstructionToUseNewBase(NewPHI, Rewriter);

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
  //if (IVUsesByStride.empty()) return;

  // FIXME: We can widen subreg IV's here for RISC targets.  e.g. instead of
  // doing computation in byte values, promote to 32-bit values if safe.

  // FIXME: Attempt to reuse values across multiple IV's.  In particular, we
  // could have something like "for(i) { foo(i*8); bar(i*16) }", which should be
  // codegened as "for (j = 0;; j+=8) { foo(j); bar(j+j); }" on X86/PPC.  Need
  // to be careful that IV's are all the same type.  Only works for intptr_t
  // indvars.

  // If we only have one stride, we can more aggressively eliminate some things.
  bool HasOneStride = IVUsesByStride.size() == 1;

  for (std::map<Value*, IVUsersOfOneStride>::iterator SI
        = IVUsesByStride.begin(), E = IVUsesByStride.end(); SI != E; ++SI)
    StrengthReduceStridedIVUsers(SI->first, SI->second, L, HasOneStride);

  // Clean up after ourselves
  if (!DeadInsts.empty()) {
    DeleteTriviallyDeadInstructions(DeadInsts);

    BasicBlock::iterator I = L->getHeader()->begin();
    PHINode *PN;
    while ((PN = dyn_cast<PHINode>(I))) {
      ++I;  // Preincrement iterator to avoid invalidating it when deleting PN.
      
      // At this point, we know that we have killed one or more GEP instructions.
      // It is worth checking to see if the cann indvar is also dead, so that we
      // can remove it as well.  The requirements for the cann indvar to be
      // considered dead are:
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
