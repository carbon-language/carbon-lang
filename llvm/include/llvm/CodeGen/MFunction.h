//===-- llvm/CodeGen/MFunction.h - Machine Specific Function ----*- C++ -*-===//
//
// This class provides a way to represent a function in a machine-specific form.
// A function is represented as a list of machine specific blocks along with a
// list of registers that are used to receive arguments for the function.
//
// In the machine specific representation for a function, the function may
// either be in SSA form or in a register based form.  When in SSA form, the
// register numbers are indexes into the RegDefMap that the MFunction contains.
// This allows accessing SSA use-def information by using the source register
// number for a use.
//
// After register allocation occurs, all of the register numbers in a function
// refer to real hardware registers and the RegDefMap is cleared.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_MFUNCTION_H
#define CODEGEN_MFUNCTION_H

#include "llvm/CodeGen/MBasicBlock.h"
#include <iosfwd>
class MInstructionInfo;

class MFunction {
  iplist<MBasicBlock> BasicBlocks;
  // FIXME: This should contain a pointer to the LLVM function
public:

  /// print - Provide a way to get a simple debugging dump.  This dumps the
  /// machine code in a simple "assembly" language that is not really suitable
  /// for an assembler, but is useful for debugging.  This is completely target
  /// independant.
  ///
  void print(std::ostream &OS, const MInstructionInfo &MII) const;
  void dump(const MInstructionInfo &MII) const;

  // Provide accessors for the MBasicBlock list...
  typedef iplist<MBasicBlock> BasicBlockListType;
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  // Provide accessors for basic blocks...
  const BasicBlockListType &getBasicBlockList() const { return BasicBlocks; }
        BasicBlockListType &getBasicBlockList()       { return BasicBlocks; }
 
  //===--------------------------------------------------------------------===//
  // BasicBlock iterator forwarding functions
  //
  iterator                begin()       { return BasicBlocks.begin(); }
  const_iterator          begin() const { return BasicBlocks.begin(); }
  iterator                end  ()       { return BasicBlocks.end();   }
  const_iterator          end  () const { return BasicBlocks.end();   }

  reverse_iterator       rbegin()       { return BasicBlocks.rbegin(); }
  const_reverse_iterator rbegin() const { return BasicBlocks.rbegin(); }
  reverse_iterator       rend  ()       { return BasicBlocks.rend();   }
  const_reverse_iterator rend  () const { return BasicBlocks.rend();   }

  unsigned                 size() const { return BasicBlocks.size(); }
  bool                    empty() const { return BasicBlocks.empty(); }
  const MBasicBlock      &front() const { return BasicBlocks.front(); }
        MBasicBlock      &front()       { return BasicBlocks.front(); }
  const MBasicBlock       &back() const { return BasicBlocks.back(); }
        MBasicBlock       &back()       { return BasicBlocks.back(); }
};

#endif
