// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	PeepholeOpts.h
// 
// Purpose:
//	Support for performing several peephole opts in one or a few passes
//      over the machine code of a method.
//**************************************************************************/


#include "llvm/CodeGen/PeepholeOpts.h"
#include "llvm/CodeGen/MachineCodeForBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/MachineOptInfo.h"
#include "llvm/BasicBlock.h"
#include "llvm/Pass.h"


//********************* Internal Class Declarations ************************/


//************************* Internal Functions *****************************/

inline void
DeleteInstruction(MachineCodeForBasicBlock& mvec,
                  MachineCodeForBasicBlock::iterator& BBI,
                  const TargetMachine& target)
{
  // Check if this instruction is in a delay slot of its predecessor.
  // If so, replace this instruction with a nop, else just delete it.
  // By replacing in place, we save having to update the I-I maps.
  if (BBI != mvec.begin())
    {
      const MachineInstrInfo& mii = target.getInstrInfo();
      MachineInstr* predMI = *(BBI-1);
      if (unsigned ndelay = mii.getNumDelaySlots(predMI->getOpCode()))
        {
          assert(ndelay == 1 && "Not yet handling multiple-delay-slot targets");
          (*BBI)->replace(mii.getNOPOpCode(), 0);
        }
    }
  else
    {
      mvec.erase(BBI);
      BBI = mvec.end();
    }
}

//******************* Individual Peephole Optimizations ********************/


inline bool
RemoveUselessCopies(MachineCodeForBasicBlock& mvec,
                    MachineCodeForBasicBlock::iterator& BBI,
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
  bool visit(MachineCodeForBasicBlock& mvec,
             MachineCodeForBasicBlock::iterator BBI) const;
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
PeepholeOpts::visit(MachineCodeForBasicBlock& mvec,
                    MachineCodeForBasicBlock::iterator BBI) const
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
  MachineCodeForBasicBlock& mvec = MachineCodeForBasicBlock::get(&BB);

  // Iterate over all machine instructions in the BB
  // Use a reverse iterator to allow deletion of MI or any instruction after it.
  // Insertions or deletions *before* MI are not safe.
  // 
  for (MachineCodeForBasicBlock::reverse_iterator RI=mvec.rbegin(),
         RE=mvec.rend(); RI != RE; )
    {
      MachineCodeForBasicBlock::iterator BBI = RI.base()-1; // save before incr
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

