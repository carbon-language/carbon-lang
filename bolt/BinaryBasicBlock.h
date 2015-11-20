//===--- BinaryBasicBlock.h - Interface for assembly-level basic block ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TODO: memory management for instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_FLO_BINARY_BASIC_BLOCK_H
#define LLVM_TOOLS_LLVM_FLO_BINARY_BASIC_BLOCK_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ilist.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace llvm {
namespace flo {

class BinaryFunction;

/// The intention is to keep the structure similar to MachineBasicBlock as
/// we might switch to it at some point.
class BinaryBasicBlock {

  /// Label associated with the block.
  MCSymbol *Label{nullptr};

  /// Original offset in the function.
  uint64_t Offset{std::numeric_limits<uint64_t>::max()};

  /// Alignment requirements for the block.
  uint64_t Alignment{1};

  /// Number of pseudo instructions in this block.
  uint32_t NumPseudos{0};

  /// Number of times this basic block was executed.
  uint64_t ExecutionCount{COUNT_NO_PROFILE};

  /// In cases where the parent function has been split, IsCold == true means
  /// this BB will be allocated outside its parent function.
  bool IsCold{false};

  /// Vector of all instructions in the block.
  std::vector<MCInst> Instructions;

  /// CFG information.
  std::vector<BinaryBasicBlock *> Predecessors;
  std::vector<BinaryBasicBlock *> Successors;

  struct BinaryBranchInfo {
    uint64_t Count;
    uint64_t MispredictedCount; /// number of branches mispredicted
  };

  /// Each successor has a corresponding BranchInfo entry in the list.
  std::vector<BinaryBranchInfo> BranchInfo;
  typedef std::vector<BinaryBranchInfo>::iterator          branch_info_iterator;
  typedef std::vector<BinaryBranchInfo>::const_iterator
                                                     const_branch_info_iterator;

  BinaryBasicBlock() {}

  explicit BinaryBasicBlock(
      MCSymbol *Label,
      uint64_t Offset = std::numeric_limits<uint64_t>::max())
    : Label(Label), Offset(Offset) {}

  explicit BinaryBasicBlock(uint64_t Offset)
    : Offset(Offset) {}

  // Exclusively managed by BinaryFunction.
  friend class BinaryFunction;
  friend bool operator<(const BinaryBasicBlock &LHS,
                        const BinaryBasicBlock &RHS);

public:
  static constexpr uint64_t COUNT_FALLTHROUGH_EDGE =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint64_t COUNT_NO_PROFILE =
      std::numeric_limits<uint64_t>::max();

  // Instructions iterators.
  typedef std::vector<MCInst>::iterator                                iterator;
  typedef std::vector<MCInst>::const_iterator                    const_iterator;
  typedef std::reverse_iterator<const_iterator>          const_reverse_iterator;
  typedef std::reverse_iterator<iterator>                      reverse_iterator;

  bool         empty()            const { return Instructions.empty(); }
  unsigned     size()     const { return (unsigned)Instructions.size(); }
  MCInst       &front()                 { return Instructions.front();  }
  MCInst       &back()                  { return Instructions.back();   }
  const MCInst &front()           const { return Instructions.front();  }
  const MCInst &back()            const { return Instructions.back();   }

  iterator                begin()       { return Instructions.begin();  }
  const_iterator          begin() const { return Instructions.begin();  }
  iterator                end  ()       { return Instructions.end();    }
  const_iterator          end  () const { return Instructions.end();    }
  reverse_iterator       rbegin()       { return Instructions.rbegin(); }
  const_reverse_iterator rbegin() const { return Instructions.rbegin(); }
  reverse_iterator       rend  ()       { return Instructions.rend();   }
  const_reverse_iterator rend  () const { return Instructions.rend();   }

  // CFG iterators.
  typedef std::vector<BinaryBasicBlock *>::iterator       pred_iterator;
  typedef std::vector<BinaryBasicBlock *>::const_iterator const_pred_iterator;
  typedef std::vector<BinaryBasicBlock *>::iterator       succ_iterator;
  typedef std::vector<BinaryBasicBlock *>::const_iterator const_succ_iterator;
  typedef std::vector<BinaryBasicBlock *>::reverse_iterator
                                                         pred_reverse_iterator;
  typedef std::vector<BinaryBasicBlock *>::const_reverse_iterator
                                                   const_pred_reverse_iterator;
  typedef std::vector<BinaryBasicBlock *>::reverse_iterator
                                                         succ_reverse_iterator;
  typedef std::vector<BinaryBasicBlock *>::const_reverse_iterator
                                                   const_succ_reverse_iterator;
  pred_iterator        pred_begin()       { return Predecessors.begin(); }
  const_pred_iterator  pred_begin() const { return Predecessors.begin(); }
  pred_iterator        pred_end()         { return Predecessors.end();   }
  const_pred_iterator  pred_end()   const { return Predecessors.end();   }
  pred_reverse_iterator        pred_rbegin()
                                          { return Predecessors.rbegin();}
  const_pred_reverse_iterator  pred_rbegin() const
                                          { return Predecessors.rbegin();}
  pred_reverse_iterator        pred_rend()
                                          { return Predecessors.rend();  }
  const_pred_reverse_iterator  pred_rend()   const
                                          { return Predecessors.rend();  }
  unsigned             pred_size()  const {
    return (unsigned)Predecessors.size();
  }
  bool                 pred_empty() const { return Predecessors.empty(); }

  succ_iterator        succ_begin()       { return Successors.begin();   }
  const_succ_iterator  succ_begin() const { return Successors.begin();   }
  succ_iterator        succ_end()         { return Successors.end();     }
  const_succ_iterator  succ_end()   const { return Successors.end();     }
  succ_reverse_iterator        succ_rbegin()
                                          { return Successors.rbegin();  }
  const_succ_reverse_iterator  succ_rbegin() const
                                          { return Successors.rbegin();  }
  succ_reverse_iterator        succ_rend()
                                          { return Successors.rend();    }
  const_succ_reverse_iterator  succ_rend()   const
                                          { return Successors.rend();    }
  unsigned             succ_size()  const {
    return (unsigned)Successors.size();
  }
  bool                 succ_empty() const { return Successors.empty();   }

  inline iterator_range<pred_iterator> predecessors() {
    return iterator_range<pred_iterator>(pred_begin(), pred_end());
  }
  inline iterator_range<const_pred_iterator> predecessors() const {
    return iterator_range<const_pred_iterator>(pred_begin(), pred_end());
  }
  inline iterator_range<succ_iterator> successors() {
    return iterator_range<succ_iterator>(succ_begin(), succ_end());
  }
  inline iterator_range<const_succ_iterator> successors() const {
    return iterator_range<const_succ_iterator>(succ_begin(), succ_end());
  }

  /// Return symbol marking the start of this basic block.
  MCSymbol *getLabel() const {
    return Label;
  }

  /// Return local name for the block.
  StringRef getName() const {
    return Label->getName();
  }

  /// Add instruction at the end of this basic block.
  void addInstruction(MCInst &Inst) {
    Instructions.emplace_back(Inst);
  }

  /// Add instruction before Pos in this basic block.
  const_iterator insertPseudoInstr(const_iterator Pos, MCInst &Instr) {
    ++NumPseudos;
    return Instructions.emplace(Pos, Instr);
  }

  uint32_t getNumPseudos() const { return NumPseudos; }

  /// Set minimum alignment for the basic block.
  void setAlignment(uint64_t Align) {
    Alignment = Align;
  }

  /// Return required alignment for the block.
  uint64_t getAlignment() const {
    return Alignment;
  }

  /// Return offset of the basic block from the function start.
  uint64_t getOffset() const {
    return Offset;
  }

  /// Adds block to successor list, and also updates predecessor list for
  /// successor block.
  /// Set branch info for this path.
  void addSuccessor(BinaryBasicBlock *Succ,
                    uint64_t Count = 0,
                    uint64_t MispredictedCount = 0);

  /// Remove /p Succ basic block from the list of successors. Update the
  /// list of predecessors of /p Succ and update branch info.
  void removeSuccessor(BinaryBasicBlock *Succ);

  /// Return the information about the number of times this basic block was
  /// executed.
  ///
  /// Return COUNT_NO_PROFILE if there's no profile info.
  uint64_t getExecutionCount() const {
    return ExecutionCount;
  }

  bool isCold() const {
    return IsCold;
  }

  bool eraseInstruction(MCInst *Inst) {
    auto I = Instructions.end();
    auto B = Instructions.begin();
    while (I > B) {
      --I;
      if (&*I == Inst) {
        Instructions.erase(I);
        return true;
      }
    }
    return false;
  }

private:

  /// Adds predecessor to the BB. Most likely you don't need to call this.
  void addPredecessor(BinaryBasicBlock *Pred);

  /// Remove predecessor of the basic block. Don't use directly, instead
  /// use removeSuccessor() funciton.
  void removePredecessor(BinaryBasicBlock *Pred);
};

bool operator<(const BinaryBasicBlock &LHS, const BinaryBasicBlock &RHS);


} // namespace flo
} // namespace llvm

#endif
