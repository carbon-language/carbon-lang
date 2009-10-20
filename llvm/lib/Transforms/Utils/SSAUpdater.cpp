//===- SSAUpdater.cpp - Unstructured SSA Update Tool ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SSAUpdater class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

typedef DenseMap<BasicBlock*, TrackingVH<Value> > AvailableValsTy;
typedef std::vector<std::pair<BasicBlock*, TrackingVH<Value> > >
                IncomingPredInfoTy;

static AvailableValsTy &getAvailableVals(void *AV) {
  return *static_cast<AvailableValsTy*>(AV);
}

static IncomingPredInfoTy &getIncomingPredInfo(void *IPI) {
  return *static_cast<IncomingPredInfoTy*>(IPI);
}


SSAUpdater::SSAUpdater(SmallVectorImpl<PHINode*> *NewPHI)
  : AV(0), PrototypeValue(0), IPI(0), InsertedPHIs(NewPHI) {}

SSAUpdater::~SSAUpdater() {
  delete &getAvailableVals(AV);
  delete &getIncomingPredInfo(IPI);
}

/// Initialize - Reset this object to get ready for a new set of SSA
/// updates.  ProtoValue is the value used to name PHI nodes.
void SSAUpdater::Initialize(Value *ProtoValue) {
  if (AV == 0)
    AV = new AvailableValsTy();
  else
    getAvailableVals(AV).clear();

  if (IPI == 0)
    IPI = new IncomingPredInfoTy();
  else
    getIncomingPredInfo(IPI).clear();
  PrototypeValue = ProtoValue;
}

/// HasValueForBlock - Return true if the SSAUpdater already has a value for
/// the specified block.
bool SSAUpdater::HasValueForBlock(BasicBlock *BB) const {
  return getAvailableVals(AV).count(BB);
}

/// AddAvailableValue - Indicate that a rewritten value is available in the
/// specified block with the specified value.
void SSAUpdater::AddAvailableValue(BasicBlock *BB, Value *V) {
  assert(PrototypeValue != 0 && "Need to initialize SSAUpdater");
  assert(PrototypeValue->getType() == V->getType() &&
         "All rewritten values must have the same type");
  getAvailableVals(AV)[BB] = V;
}

/// GetValueAtEndOfBlock - Construct SSA form, materializing a value that is
/// live at the end of the specified block.
Value *SSAUpdater::GetValueAtEndOfBlock(BasicBlock *BB) {
  assert(getIncomingPredInfo(IPI).empty() && "Unexpected Internal State");
  Value *Res = GetValueAtEndOfBlockInternal(BB);
  assert(getIncomingPredInfo(IPI).empty() && "Unexpected Internal State");
  return Res;
}

/// GetValueInMiddleOfBlock - Construct SSA form, materializing a value that
/// is live in the middle of the specified block.
///
/// GetValueInMiddleOfBlock is the same as GetValueAtEndOfBlock except in one
/// important case: if there is a definition of the rewritten value after the
/// 'use' in BB.  Consider code like this:
///
///      X1 = ...
///   SomeBB:
///      use(X)
///      X2 = ...
///      br Cond, SomeBB, OutBB
///
/// In this case, there are two values (X1 and X2) added to the AvailableVals
/// set by the client of the rewriter, and those values are both live out of
/// their respective blocks.  However, the use of X happens in the *middle* of
/// a block.  Because of this, we need to insert a new PHI node in SomeBB to
/// merge the appropriate values, and this value isn't live out of the block.
///
Value *SSAUpdater::GetValueInMiddleOfBlock(BasicBlock *BB) {
  // If there is no definition of the renamed variable in this block, just use
  // GetValueAtEndOfBlock to do our work.
  if (!getAvailableVals(AV).count(BB))
    return GetValueAtEndOfBlock(BB);

  // Otherwise, we have the hard case.  Get the live-in values for each
  // predecessor.
  SmallVector<std::pair<BasicBlock*, Value*>, 8> PredValues;
  Value *SingularValue = 0;

  // We can get our predecessor info by walking the pred_iterator list, but it
  // is relatively slow.  If we already have PHI nodes in this block, walk one
  // of them to get the predecessor list instead.
  if (PHINode *SomePhi = dyn_cast<PHINode>(BB->begin())) {
    for (unsigned i = 0, e = SomePhi->getNumIncomingValues(); i != e; ++i) {
      BasicBlock *PredBB = SomePhi->getIncomingBlock(i);
      Value *PredVal = GetValueAtEndOfBlock(PredBB);
      PredValues.push_back(std::make_pair(PredBB, PredVal));

      // Compute SingularValue.
      if (i == 0)
        SingularValue = PredVal;
      else if (PredVal != SingularValue)
        SingularValue = 0;
    }
  } else {
    bool isFirstPred = true;
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *PredBB = *PI;
      Value *PredVal = GetValueAtEndOfBlock(PredBB);
      PredValues.push_back(std::make_pair(PredBB, PredVal));

      // Compute SingularValue.
      if (isFirstPred) {
        SingularValue = PredVal;
        isFirstPred = false;
      } else if (PredVal != SingularValue)
        SingularValue = 0;
    }
  }

  // If there are no predecessors, just return undef.
  if (PredValues.empty())
    return UndefValue::get(PrototypeValue->getType());

  // Otherwise, if all the merged values are the same, just use it.
  if (SingularValue != 0)
    return SingularValue;

  // Otherwise, we do need a PHI: insert one now.
  PHINode *InsertedPHI = PHINode::Create(PrototypeValue->getType(),
                                         PrototypeValue->getName(),
                                         &BB->front());
  InsertedPHI->reserveOperandSpace(PredValues.size());

  // Fill in all the predecessors of the PHI.
  for (unsigned i = 0, e = PredValues.size(); i != e; ++i)
    InsertedPHI->addIncoming(PredValues[i].second, PredValues[i].first);

  // See if the PHI node can be merged to a single value.  This can happen in
  // loop cases when we get a PHI of itself and one other value.
  if (Value *ConstVal = InsertedPHI->hasConstantValue()) {
    InsertedPHI->eraseFromParent();
    return ConstVal;
  }

  // If the client wants to know about all new instructions, tell it.
  if (InsertedPHIs) InsertedPHIs->push_back(InsertedPHI);

  DEBUG(errs() << "  Inserted PHI: " << *InsertedPHI << "\n");
  return InsertedPHI;
}

/// RewriteUse - Rewrite a use of the symbolic value.  This handles PHI nodes,
/// which use their value in the corresponding predecessor.
void SSAUpdater::RewriteUse(Use &U) {
  Instruction *User = cast<Instruction>(U.getUser());
  BasicBlock *UseBB = User->getParent();
  PHINode *UserPN = dyn_cast<PHINode>(User);
  if (UserPN)
    UseBB = UserPN->getIncomingBlock(U);

  Value *V = GetValueInMiddleOfBlock(UseBB);
  U.set(V);
  if (UserPN) {
    // Incoming value from the same BB must be consistent
    for (unsigned i=0;i<UserPN->getNumIncomingValues();i++)
      if (UserPN->getIncomingBlock(i) == UseBB)
        UserPN->setIncomingValue(i, V);
  }
}


/// GetValueAtEndOfBlockInternal - Check to see if AvailableVals has an entry
/// for the specified BB and if so, return it.  If not, construct SSA form by
/// walking predecessors inserting PHI nodes as needed until we get to a block
/// where the value is available.
///
Value *SSAUpdater::GetValueAtEndOfBlockInternal(BasicBlock *BB) {
  AvailableValsTy &AvailableVals = getAvailableVals(AV);

  // Query AvailableVals by doing an insertion of null.
  std::pair<AvailableValsTy::iterator, bool> InsertRes =
  AvailableVals.insert(std::make_pair(BB, WeakVH()));

  // Handle the case when the insertion fails because we have already seen BB.
  if (!InsertRes.second) {
    // If the insertion failed, there are two cases.  The first case is that the
    // value is already available for the specified block.  If we get this, just
    // return the value.
    if (InsertRes.first->second != 0)
      return InsertRes.first->second;

    // Otherwise, if the value we find is null, then this is the value is not
    // known but it is being computed elsewhere in our recursion.  This means
    // that we have a cycle.  Handle this by inserting a PHI node and returning
    // it.  When we get back to the first instance of the recursion we will fill
    // in the PHI node.
    return InsertRes.first->second =
    PHINode::Create(PrototypeValue->getType(), PrototypeValue->getName(),
                    &BB->front());
  }

  // Okay, the value isn't in the map and we just inserted a null in the entry
  // to indicate that we're processing the block.  Since we have no idea what
  // value is in this block, we have to recurse through our predecessors.
  //
  // While we're walking our predecessors, we keep track of them in a vector,
  // then insert a PHI node in the end if we actually need one.  We could use a
  // smallvector here, but that would take a lot of stack space for every level
  // of the recursion, just use IncomingPredInfo as an explicit stack.
  IncomingPredInfoTy &IncomingPredInfo = getIncomingPredInfo(IPI);
  unsigned FirstPredInfoEntry = IncomingPredInfo.size();

  // As we're walking the predecessors, keep track of whether they are all
  // producing the same value.  If so, this value will capture it, if not, it
  // will get reset to null.  We distinguish the no-predecessor case explicitly
  // below.
  TrackingVH<Value> SingularValue;

  // We can get our predecessor info by walking the pred_iterator list, but it
  // is relatively slow.  If we already have PHI nodes in this block, walk one
  // of them to get the predecessor list instead.
  if (PHINode *SomePhi = dyn_cast<PHINode>(BB->begin())) {
    for (unsigned i = 0, e = SomePhi->getNumIncomingValues(); i != e; ++i) {
      BasicBlock *PredBB = SomePhi->getIncomingBlock(i);
      Value *PredVal = GetValueAtEndOfBlockInternal(PredBB);
      IncomingPredInfo.push_back(std::make_pair(PredBB, PredVal));

      // Compute SingularValue.
      if (i == 0)
        SingularValue = PredVal;
      else if (PredVal != SingularValue)
        SingularValue = 0;
    }
  } else {
    bool isFirstPred = true;
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *PredBB = *PI;
      Value *PredVal = GetValueAtEndOfBlockInternal(PredBB);
      IncomingPredInfo.push_back(std::make_pair(PredBB, PredVal));

      // Compute SingularValue.
      if (isFirstPred) {
        SingularValue = PredVal;
        isFirstPred = false;
      } else if (PredVal != SingularValue)
        SingularValue = 0;
    }
  }

  // If there are no predecessors, then we must have found an unreachable block
  // just return 'undef'.  Since there are no predecessors, InsertRes must not
  // be invalidated.
  if (IncomingPredInfo.size() == FirstPredInfoEntry)
    return InsertRes.first->second = UndefValue::get(PrototypeValue->getType());

  /// Look up BB's entry in AvailableVals.  'InsertRes' may be invalidated.  If
  /// this block is involved in a loop, a no-entry PHI node will have been
  /// inserted as InsertedVal.  Otherwise, we'll still have the null we inserted
  /// above.
  TrackingVH<Value> &InsertedVal = AvailableVals[BB];

  // If all the predecessor values are the same then we don't need to insert a
  // PHI.  This is the simple and common case.
  if (SingularValue) {
    // If a PHI node got inserted, replace it with the singlar value and delete
    // it.
    if (InsertedVal) {
      PHINode *OldVal = cast<PHINode>(InsertedVal);
      // Be careful about dead loops.  These RAUW's also update InsertedVal.
      if (InsertedVal != SingularValue)
        OldVal->replaceAllUsesWith(SingularValue);
      else
        OldVal->replaceAllUsesWith(UndefValue::get(InsertedVal->getType()));
      OldVal->eraseFromParent();
    } else {
      InsertedVal = SingularValue;
    }

    // Drop the entries we added in IncomingPredInfo to restore the stack.
    IncomingPredInfo.erase(IncomingPredInfo.begin()+FirstPredInfoEntry,
                           IncomingPredInfo.end());
    return InsertedVal;
  }

  // Otherwise, we do need a PHI: insert one now if we don't already have one.
  if (InsertedVal == 0)
    InsertedVal = PHINode::Create(PrototypeValue->getType(),
                                  PrototypeValue->getName(), &BB->front());

  PHINode *InsertedPHI = cast<PHINode>(InsertedVal);
  InsertedPHI->reserveOperandSpace(IncomingPredInfo.size()-FirstPredInfoEntry);

  // Fill in all the predecessors of the PHI.
  for (IncomingPredInfoTy::iterator I =
         IncomingPredInfo.begin()+FirstPredInfoEntry,
       E = IncomingPredInfo.end(); I != E; ++I)
    InsertedPHI->addIncoming(I->second, I->first);

  // Drop the entries we added in IncomingPredInfo to restore the stack.
  IncomingPredInfo.erase(IncomingPredInfo.begin()+FirstPredInfoEntry,
                         IncomingPredInfo.end());

  // See if the PHI node can be merged to a single value.  This can happen in
  // loop cases when we get a PHI of itself and one other value.
  if (Value *ConstVal = InsertedPHI->hasConstantValue()) {
    InsertedPHI->replaceAllUsesWith(ConstVal);
    InsertedPHI->eraseFromParent();
    InsertedVal = ConstVal;
  } else {
    DEBUG(errs() << "  Inserted PHI: " << *InsertedPHI << "\n");

    // If the client wants to know about all new instructions, tell it.
    if (InsertedPHIs) InsertedPHIs->push_back(InsertedPHI);
  }

  return InsertedVal;
}
