//===-- PeepholeOpts.cpp --------------------------------------------------===//
// 
// Support for performing several peephole opts in one or a few passes over the
// machine code of a method.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/PeepholeOpts.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptInfo.h"
#include "llvm/BasicBlock.h"
#include "llvm/Pass.h"

//************************* Internal Functions *****************************/

inline void
DeleteInstruction(MachineBasicBlock& mvec,
                  MachineBasicBlock::iterator& BBI,
                  const TargetMachine& target)
{
  // Check if this instruction is in a delay slot of its predecessor.
  if (BBI != mvec.begin())
    {
      const TargetInstrInfo& mii = target.getInstrInfo();
      MachineInstr* predMI = *(BBI-1);
      if (unsigned ndelay = mii.getNumDelaySlots(predMI->getOpCode()))
        {
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


inline bool
RemoveUselessCopies(MachineBasicBlock& mvec,
                    MachineBasicBlock::iterator& BBI,
                    const TargetMachine& target)
{
  if (target.getOptInfo().IsUselessCopy(*BBI))
    {
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
  PeepholeOpts(const TargetMachine &T): target(T) { }
  bool runOnBasicBlock(BasicBlock &BB); // apply this pass to each BB
};


// Register the pass with llc only, and not opt...
static RegisterLLC<PeepholeOpts>
X("peephole", "Peephole Optimization", createPeepholeOptsPass);


/* Apply a list of peephole optimizations to this machine instruction
 * within its local context.  They are allowed to delete MI or any
 * instruction before MI, but not 
 */
bool
PeepholeOpts::visit(MachineBasicBlock& mvec,
                    MachineBasicBlock::iterator BBI) const
{
  bool changed = false;

  /* Remove redundant copy instructions */
  changed |= RemoveUselessCopies(mvec, BBI, target);
  if (BBI == mvec.end())                // nothing more to do!
    return changed;

  return changed;
}


bool
PeepholeOpts::runOnBasicBlock(BasicBlock &BB)
{
  // Get the machine instructions for this BB
  // FIXME: MachineBasicBlock::get() is deprecated, hence inlining the function
  const Function *F = BB.getParent();
  MachineFunction &MF = MachineFunction::get(F);
  MachineBasicBlock *MBB = NULL;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    if (I->getBasicBlock() == &BB)
      MBB = I;
  }
  assert(MBB && "MachineBasicBlock object not found for specified block!");
  MachineBasicBlock &mvec = *MBB;

  // Iterate over all machine instructions in the BB
  // Use a reverse iterator to allow deletion of MI or any instruction after it.
  // Insertions or deletions *before* MI are not safe.
  // 
  for (MachineBasicBlock::reverse_iterator RI=mvec.rbegin(),
         RE=mvec.rend(); RI != RE; )
    {
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
Pass*
createPeepholeOptsPass(TargetMachine &T)
{
  return new PeepholeOpts(T);
}

