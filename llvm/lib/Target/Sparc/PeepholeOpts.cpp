//===-- PeepholeOpts.cpp --------------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// Support for performing several peephole opts in one or a few passes over the
// machine code of a method.
//
//===----------------------------------------------------------------------===//

#include "SparcInternals.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/BasicBlock.h"
#include "llvm/Pass.h"

namespace llvm {

//************************* Internal Functions *****************************/

static inline void
DeleteInstruction(MachineBasicBlock& mvec,
                  MachineBasicBlock::iterator& BBI,
                  const TargetMachine& target) {
  // Check if this instruction is in a delay slot of its predecessor.
  if (BBI != mvec.begin()) {
      const TargetInstrInfo& mii = target.getInstrInfo();
      MachineInstr* predMI = *(BBI-1);
      if (unsigned ndelay = mii.getNumDelaySlots(predMI->getOpCode())) {
        // This instruction is in a delay slot of its predecessor, so
        // replace it with a nop. By replacing in place, we save having
        // to update the I-I maps.
        // 
        assert(ndelay == 1 && "Not yet handling multiple-delay-slot targets");
        (*BBI)->replace(mii.getNOPOpCode(), 0);
        return;
      }
  }
  
  // The instruction is not in a delay slot, so we can simply erase it.
  mvec.erase(BBI);
  BBI = mvec.end();
}

//******************* Individual Peephole Optimizations ********************/

//----------------------------------------------------------------------------
// Function: IsUselessCopy
// Decide whether a machine instruction is a redundant copy:
// -- ADD    with g0 and result and operand are identical, or
// -- OR     with g0 and result and operand are identical, or
// -- FMOVS or FMOVD and result and operand are identical.
// Other cases are possible but very rare that they would be useless copies,
// so it's not worth analyzing them.
//----------------------------------------------------------------------------

static bool IsUselessCopy(const TargetMachine &target, const MachineInstr* MI) {
  if (MI->getOpCode() == V9::FMOVS || MI->getOpCode() == V9::FMOVD) {
    return (/* both operands are allocated to the same register */
            MI->getOperand(0).getAllocatedRegNum() == 
            MI->getOperand(1).getAllocatedRegNum());
  } else if (MI->getOpCode() == V9::ADDr || MI->getOpCode() == V9::ORr ||
             MI->getOpCode() == V9::ADDi || MI->getOpCode() == V9::ORi) {
    unsigned srcWithDestReg;
    
    for (srcWithDestReg = 0; srcWithDestReg < 2; ++srcWithDestReg)
      if (MI->getOperand(srcWithDestReg).hasAllocatedReg() &&
          MI->getOperand(srcWithDestReg).getAllocatedRegNum()
          == MI->getOperand(2).getAllocatedRegNum())
        break;
    
    if (srcWithDestReg == 2)
      return false;
    else {
      /* else source and dest are allocated to the same register */
      unsigned otherOp = 1 - srcWithDestReg;
      return (/* either operand otherOp is register %g0 */
              (MI->getOperand(otherOp).hasAllocatedReg() &&
               MI->getOperand(otherOp).getAllocatedRegNum() ==
               target.getRegInfo().getZeroRegNum()) ||
              
              /* or operand otherOp == 0 */
              (MI->getOperand(otherOp).getType()
               == MachineOperand::MO_SignExtendedImmed &&
               MI->getOperand(otherOp).getImmedValue() == 0));
    }
  }
  else
    return false;
}

inline bool
RemoveUselessCopies(MachineBasicBlock& mvec,
                    MachineBasicBlock::iterator& BBI,
                    const TargetMachine& target) {
  if (IsUselessCopy(target, *BBI)) {
    DeleteInstruction(mvec, BBI, target);
    return true;
  }
  return false;
}


//************************ Class Implementations **************************/

class PeepholeOpts: public BasicBlockPass {
  const TargetMachine &target;
  bool visit(MachineBasicBlock& mvec,
             MachineBasicBlock::iterator BBI) const;
public:
  PeepholeOpts(const TargetMachine &TM): target(TM) { }
  bool runOnBasicBlock(BasicBlock &BB); // apply this pass to each BB
  virtual const char *getPassName() const { return "Peephole Optimization"; }
};

// Apply a list of peephole optimizations to this machine instruction
// within its local context.  They are allowed to delete MI or any
// instruction before MI, but not 
//
bool PeepholeOpts::visit(MachineBasicBlock& mvec,
                         MachineBasicBlock::iterator BBI) const {
  /* Remove redundant copy instructions */
  return RemoveUselessCopies(mvec, BBI, target);
}


bool PeepholeOpts::runOnBasicBlock(BasicBlock &BB) {
  // Get the machine instructions for this BB
  // FIXME: MachineBasicBlock::get() is deprecated, hence inlining the function
  const Function *F = BB.getParent();
  MachineFunction &MF = MachineFunction::get(F);
  MachineBasicBlock *MBB = NULL;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    if (I->getBasicBlock() == &BB)
      MBB = I;

  assert(MBB && "MachineBasicBlock object not found for specified block!");
  MachineBasicBlock &mvec = *MBB;

  // Iterate over all machine instructions in the BB
  // Use a reverse iterator to allow deletion of MI or any instruction after it.
  // Insertions or deletions *before* MI are not safe.
  // 
  for (MachineBasicBlock::reverse_iterator RI=mvec.rbegin(),
         RE=mvec.rend(); RI != RE; ) {
    MachineBasicBlock::iterator BBI = RI.base()-1; // save before incr
    ++RI;             // pre-increment to delete MI or after it
    visit(mvec, BBI);
  }

  return true;
}


//===----------------------------------------------------------------------===//
// createPeepholeOptsPass - Public entrypoint for peephole optimization
// and this file as a whole...
//
FunctionPass* createPeepholeOptsPass(const TargetMachine &TM) {
  return new PeepholeOpts(TM);
}

} // End llvm namespace
