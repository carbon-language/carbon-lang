//===- MachineSSAUpdater.cpp - Unstructured SSA Update Tool ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MachineSSAUpdater class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/ADT/DenseMap.h"
using namespace llvm;

typedef DenseMap<MachineBasicBlock*, unsigned > AvailableValsTy;
typedef std::vector<std::pair<MachineBasicBlock*, unsigned> >
                IncomingPredInfoTy;

static AvailableValsTy &getAvailableVals(void *AV) {
  return *static_cast<AvailableValsTy*>(AV);
}

static IncomingPredInfoTy &getIncomingPredInfo(void *IPI) {
  return *static_cast<IncomingPredInfoTy*>(IPI);
}


MachineSSAUpdater::MachineSSAUpdater(SmallVectorImpl<MachineInstr*> *NewPHI)
  : AV(0), IPI(0), InsertedPHIs(NewPHI) {}

MachineSSAUpdater::~MachineSSAUpdater() {
  delete &getAvailableVals(AV);
  delete &getIncomingPredInfo(IPI);
}

/// Initialize - Reset this object to get ready for a new set of SSA
/// updates.  ProtoValue is the value used to name PHI nodes.
void MachineSSAUpdater::Initialize() {
  if (AV == 0)
    AV = new AvailableValsTy();
  else
    getAvailableVals(AV).clear();

  if (IPI == 0)
    IPI = new IncomingPredInfoTy();
  else
    getIncomingPredInfo(IPI).clear();
}

/// HasValueForBlock - Return true if the MachineSSAUpdater already has a value for
/// the specified block.
bool MachineSSAUpdater::HasValueForBlock(MachineBasicBlock *BB) const {
  return getAvailableVals(AV).count(BB);
}

/// AddAvailableValue - Indicate that a rewritten value is available in the
/// specified block with the specified value.
void MachineSSAUpdater::AddAvailableValue(MachineBasicBlock *BB, unsigned V) {
  getAvailableVals(AV)[BB] = V;
}

/// GetValueAtEndOfBlock - Construct SSA form, materializing a value that is
/// live at the end of the specified block.
unsigned MachineSSAUpdater::GetValueAtEndOfBlock(MachineBasicBlock *BB) {
  return GetValueAtEndOfBlockInternal(BB);
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
unsigned MachineSSAUpdater::GetValueInMiddleOfBlock(MachineBasicBlock *BB) {
  return 0;
}

/// RewriteUse - Rewrite a use of the symbolic value.  This handles PHI nodes,
/// which use their value in the corresponding predecessor.
void MachineSSAUpdater::RewriteUse(unsigned &U) {
}


/// GetValueAtEndOfBlockInternal - Check to see if AvailableVals has an entry
/// for the specified BB and if so, return it.  If not, construct SSA form by
/// walking predecessors inserting PHI nodes as needed until we get to a block
/// where the value is available.
///
unsigned MachineSSAUpdater::GetValueAtEndOfBlockInternal(MachineBasicBlock *BB){
  return 0;
}
