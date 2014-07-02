//===-- MCFunction.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures to hold a CFG reconstructed from
// machine code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCANALYSIS_MCFUNCTION_H
#define LLVM_MC_MCANALYSIS_MCFUNCTION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include <memory>
#include <string>
#include <vector>

namespace llvm {

class MCFunction;
class MCModule;
class MCTextAtom;

/// \brief Basic block containing a sequence of disassembled instructions.
/// The basic block is backed by an MCTextAtom, which holds the instructions,
/// and the address range it covers.
/// Create a basic block using MCFunction::createBlock.
class MCBasicBlock {
  const MCTextAtom *Insts;

  // MCFunction owns the basic block.
  MCFunction *Parent;
  friend class MCFunction;
  MCBasicBlock(const MCTextAtom &Insts, MCFunction *Parent);

  /// \name Predecessors/Successors, to represent the CFG.
  /// @{
  typedef std::vector<const MCBasicBlock *> BasicBlockListTy;
  BasicBlockListTy Successors;
  BasicBlockListTy Predecessors;
  /// @}
public:

  /// \brief Get the backing MCTextAtom, containing the instruction sequence.
  const MCTextAtom *getInsts() const { return Insts; }

  /// \name Get the owning MCFunction.
  /// @{
  const MCFunction *getParent() const { return Parent; }
        MCFunction *getParent()       { return Parent; }
  /// @}

  /// MC CFG access: Predecessors/Successors.
  /// @{
  typedef BasicBlockListTy::const_iterator succ_const_iterator;
  succ_const_iterator succ_begin() const { return Successors.begin(); }
  succ_const_iterator succ_end()   const { return Successors.end(); }

  typedef BasicBlockListTy::const_iterator pred_const_iterator;
  pred_const_iterator pred_begin() const { return Predecessors.begin(); }
  pred_const_iterator pred_end()   const { return Predecessors.end(); }

  void addSuccessor(const MCBasicBlock *MCBB);
  bool isSuccessor(const MCBasicBlock *MCBB) const;

  void addPredecessor(const MCBasicBlock *MCBB);
  bool isPredecessor(const MCBasicBlock *MCBB) const;

  /// \brief Split block, mirrorring NewAtom = Insts->split(..).
  /// This moves all successors to \p SplitBB, and
  /// adds a fallthrough to it.
  /// \p SplitBB The result of splitting Insts, a basic block directly following
  /// this basic block.
  void splitBasicBlock(MCBasicBlock *SplitBB);
  /// @}
};

/// \brief Represents a function in machine code, containing MCBasicBlocks.
/// MCFunctions are created by MCModule.
class MCFunction {
  MCFunction           (const MCFunction&) LLVM_DELETED_FUNCTION;
  MCFunction& operator=(const MCFunction&) LLVM_DELETED_FUNCTION;

  std::string Name;
  MCModule *ParentModule;
  typedef std::vector<std::unique_ptr<MCBasicBlock>> BasicBlockListTy;
  BasicBlockListTy Blocks;

  // MCModule owns the function.
  friend class MCModule;
  MCFunction(StringRef Name, MCModule *Parent);

public:
  /// \brief Create an MCBasicBlock backed by Insts and add it to this function.
  /// \param Insts Sequence of straight-line code backing the basic block.
  /// \returns The newly created basic block.
  MCBasicBlock &createBlock(const MCTextAtom &Insts);

  StringRef getName() const { return Name; }

  /// \name Get the owning MC Module.
  /// @{
  const MCModule *getParent() const { return ParentModule; }
        MCModule *getParent()       { return ParentModule; }
  /// @}

  /// \name Access to the function's basic blocks. No ordering is enforced,
  /// except that the first block is the entry block.
  /// @{
  /// \brief Get the entry point basic block.
  const MCBasicBlock *getEntryBlock() const { return front(); }
        MCBasicBlock *getEntryBlock()       { return front(); }

  bool empty() const { return Blocks.empty(); }

  typedef BasicBlockListTy::const_iterator const_iterator;
  typedef BasicBlockListTy::      iterator       iterator;
  const_iterator begin() const { return Blocks.begin(); }
        iterator begin()       { return Blocks.begin(); }
  const_iterator   end() const { return Blocks.end(); }
        iterator   end()       { return Blocks.end(); }

  const MCBasicBlock* front() const { return Blocks.front().get(); }
        MCBasicBlock* front()       { return Blocks.front().get(); }
  const MCBasicBlock*  back() const { return Blocks.back().get(); }
        MCBasicBlock*  back()       { return Blocks.back().get(); }

  /// \brief Find the basic block, if any, that starts at \p StartAddr.
  const MCBasicBlock *find(uint64_t StartAddr) const;
        MCBasicBlock *find(uint64_t StartAddr);
  /// @}
};

}

#endif
