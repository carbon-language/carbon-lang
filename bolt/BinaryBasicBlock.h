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

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_BASIC_BLOCK_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_BASIC_BLOCK_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <utility>
#include <set>

namespace llvm {
namespace bolt {

class BinaryFunction;

/// The intention is to keep the structure similar to MachineBasicBlock as
/// we might switch to it at some point.
class BinaryBasicBlock {
public:

  /// Profile execution information for a given edge in CFG.
  ///
  /// If MispredictedCount equals COUNT_INFERRED, then we have a profile
  /// data for a fall-through edge with a Count representing an inferred
  /// execution count, i.e. the count we calculated internally, not the one
  /// coming from profile data.
  ///
  /// For all other values of MispredictedCount, Count represents the number of
  /// branch executions from a profile, and MispredictedCount is the number
  /// of times the branch was mispredicted according to this profile.
  struct BinaryBranchInfo {
    uint64_t Count;
    uint64_t MispredictedCount; /// number of branches mispredicted
  };

  static constexpr uint32_t INVALID_OFFSET =
                                          std::numeric_limits<uint32_t>::max();

private:
  /// Vector of all instructions in the block.
  std::vector<MCInst> Instructions;

  /// CFG information.
  std::vector<BinaryBasicBlock *> Predecessors;
  std::vector<BinaryBasicBlock *> Successors;
  std::vector<BinaryBasicBlock *> Throwers;
  std::vector<BinaryBasicBlock *> LandingPads;

  /// Each successor has a corresponding BranchInfo entry in the list.
  std::vector<BinaryBranchInfo> BranchInfo;

  /// Function that owns this basic block.
  BinaryFunction *Function;

  /// Label associated with the block.
  MCSymbol *Label{nullptr};

  /// [Begin, End) address range for this block in the output binary.
  std::pair<uint64_t, uint64_t> OutputAddressRange{0, 0};

  /// Original offset range of the basic block in the function.
  std::pair<uint32_t, uint32_t> InputRange{INVALID_OFFSET, INVALID_OFFSET};

  /// Alignment requirements for the block.
  uint64_t Alignment{1};

  /// Number of times this basic block was executed.
  uint64_t ExecutionCount{COUNT_NO_PROFILE};

  static constexpr unsigned InvalidIndex = ~0u;

  /// Index to BasicBlocks vector in BinaryFunction.
  unsigned Index{InvalidIndex};

  /// Index in the current layout.
  unsigned LayoutIndex{InvalidIndex};

  /// Number of pseudo instructions in this block.
  uint32_t NumPseudos{0};

  /// CFI state at the entry to this basic block.
  int32_t CFIState{-1};

  /// True if this basic block is (potentially) an external entry point into
  /// the function.
  bool IsEntryPoint{false};

  /// In cases where the parent function has been split, IsCold == true means
  /// this BB will be allocated outside its parent function.
  bool IsCold{false};

  /// Indicates if the block could be outlined.
  bool CanOutline{true};

  /// Flag to indicate whether this block is valid or not.  Invalid
  /// blocks may contain out of date or incorrect information.
  bool IsValid{true};

private:
  BinaryBasicBlock() = delete;
  BinaryBasicBlock(const BinaryBasicBlock &) = delete;
  BinaryBasicBlock& operator=(const BinaryBasicBlock &) = delete;

  explicit BinaryBasicBlock(
      BinaryFunction *Function,
      MCSymbol *Label,
      uint32_t Offset = INVALID_OFFSET)
    : Function(Function), Label(Label) {
    assert(Function && "Function must be non-null");
    InputRange.first = Offset;
  }

  // Exclusively managed by BinaryFunction.
  friend class BinaryFunction;
  friend bool operator<(const BinaryBasicBlock &LHS,
                        const BinaryBasicBlock &RHS);

  /// Assign new label to the basic block.
  void setLabel(MCSymbol *Symbol) {
    Label = Symbol;
  }

public:
  static constexpr uint64_t COUNT_INFERRED =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint64_t COUNT_NO_PROFILE =
      std::numeric_limits<uint64_t>::max();

  // Instructions iterators.
  using iterator       = std::vector<MCInst>::iterator;
  using const_iterator = std::vector<MCInst>::const_iterator;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  bool         empty()            const { return Instructions.empty(); }
  size_t       size()             const { return Instructions.size(); }
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
  using pred_iterator        = std::vector<BinaryBasicBlock *>::iterator;
  using const_pred_iterator  = std::vector<BinaryBasicBlock *>::const_iterator;
  using succ_iterator        = std::vector<BinaryBasicBlock *>::iterator;
  using const_succ_iterator  = std::vector<BinaryBasicBlock *>::const_iterator;
  using throw_iterator       = decltype(Throwers)::iterator;
  using const_throw_iterator = decltype(Throwers)::const_iterator;
  using lp_iterator          = decltype(LandingPads)::iterator;
  using const_lp_iterator    = decltype(LandingPads)::const_iterator;

  using pred_reverse_iterator = std::reverse_iterator<pred_iterator>;
  using const_pred_reverse_iterator =
    std::reverse_iterator<const_pred_iterator>;
  using succ_reverse_iterator = std::reverse_iterator<succ_iterator>;
  using const_succ_reverse_iterator =
    std::reverse_iterator<const_succ_iterator>;

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
  size_t               pred_size()  const {
    return Predecessors.size();
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
  size_t               succ_size()  const {
    return Successors.size();
  }
  bool                 succ_empty() const { return Successors.empty();   }

  throw_iterator        throw_begin()       { return Throwers.begin(); }
  const_throw_iterator  throw_begin() const { return Throwers.begin(); }
  throw_iterator        throw_end()         { return Throwers.end();   }
  const_throw_iterator  throw_end()   const { return Throwers.end();   }
  size_t                throw_size()  const {
    return Throwers.size();
  }
  bool                  throw_empty() const { return Throwers.empty(); }
  bool                  isLandingPad() const { return !Throwers.empty(); }

  lp_iterator        lp_begin()       { return LandingPads.begin();   }
  const_lp_iterator  lp_begin() const { return LandingPads.begin();   }
  lp_iterator        lp_end()         { return LandingPads.end();     }
  const_lp_iterator  lp_end()   const { return LandingPads.end();     }
  size_t             lp_size()  const {
    return LandingPads.size();
  }
  bool               lp_empty() const { return LandingPads.empty();   }

  inline iterator_range<iterator> instructions() {
    return iterator_range<iterator>(begin(), end());
  }
  inline iterator_range<const_iterator> instructions() const {
    return iterator_range<const_iterator>(begin(), end());
  }
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
  inline iterator_range<throw_iterator> throwers() {
    return iterator_range<throw_iterator>(throw_begin(), throw_end());
  }
  inline iterator_range<const_throw_iterator> throwers() const {
    return iterator_range<const_throw_iterator>(throw_begin(), throw_end());
  }
  inline iterator_range<lp_iterator> landing_pads() {
    return iterator_range<lp_iterator>(lp_begin(), lp_end());
  }
  inline iterator_range<const_lp_iterator> landing_pads() const {
    return iterator_range<const_lp_iterator>(lp_begin(), lp_end());
  }

  // BranchInfo iterators.
  using branch_info_iterator = std::vector<BinaryBranchInfo>::iterator;
  using const_branch_info_iterator =
                       std::vector<BinaryBranchInfo>::const_iterator;
  using branch_info_reverse_iterator =
                       std::reverse_iterator<branch_info_iterator>;
  using const_branch_info_reverse_iterator =
                       std::reverse_iterator<const_branch_info_iterator>;

  branch_info_iterator branch_info_begin() { return BranchInfo.begin(); }
  branch_info_iterator branch_info_end()   { return BranchInfo.end(); }
  const_branch_info_iterator branch_info_begin() const {
    return BranchInfo.begin();
  }
  const_branch_info_iterator branch_info_end() const {
    return BranchInfo.end();
  }
  branch_info_reverse_iterator branch_info_rbegin() {
    return BranchInfo.rbegin();
  }
  branch_info_reverse_iterator branch_info_rend() {
    return BranchInfo.rend();
  }
  const_branch_info_reverse_iterator branch_info_rbegin() const {
    return BranchInfo.rbegin();
  }
  const_branch_info_reverse_iterator branch_info_rend() const {
    return BranchInfo.rend();
  }

  size_t branch_info_size()  const { return BranchInfo.size(); }
  bool branch_info_empty() const { return BranchInfo.empty(); }

  inline iterator_range<branch_info_iterator> branch_info() {
    return iterator_range<branch_info_iterator>(
        BranchInfo.begin(), BranchInfo.end());
  }
  inline iterator_range<const_branch_info_iterator> branch_info() const {
    return iterator_range<const_branch_info_iterator>(
        BranchInfo.begin(), BranchInfo.end());
  }

  /// Get instruction at given index.
  MCInst &getInstructionAtIndex(unsigned Index) {
    return Instructions.at(Index);
  }

  const MCInst &getInstructionAtIndex(unsigned Index) const {
    return Instructions.at(Index);
  }

  /// Return symbol marking the start of this basic block.
  MCSymbol *getLabel() {
    return Label;
  }

  /// Return symbol marking the start of this basic block (const version).
  const MCSymbol *getLabel() const {
    return Label;
  }

  /// Get successor with given \p Label if \p Label != nullptr.
  /// Returns nullptr if no such successor is found.
  /// If the \p Label == nullptr and the block has only one successor then
  /// return the successor.
  BinaryBasicBlock *getSuccessor(const MCSymbol *Label = nullptr) const;

  /// If the basic block ends with a conditional branch (possibly followed by
  /// an unconditional branch) and thus has 2 successors, return a successor
  /// corresponding to a jump condition which could be true or false.
  /// Return nullptr if the basic block does not have a conditional jump.
  const BinaryBasicBlock *getConditionalSuccessor(bool Condition) const {
    if (succ_size() != 2)
      return nullptr;
    return Successors[Condition == true ? 0 : 1];
  }

  /// Find the fallthrough successor for a block, or nullptr if there is
  /// none.
  const BinaryBasicBlock* getFallthrough() const {
    if (succ_size() == 2)
      return getConditionalSuccessor(false);
    else
      return getSuccessor();
  }

  const BinaryBranchInfo &getBranchInfo(bool Condition) const {
    assert(BranchInfo.size() == 2 &&
           "could only be called for blocks with 2 successors");
    return BranchInfo[Condition == true ? 0 : 1];
  };

  BinaryBranchInfo &getBranchInfo(const BinaryBasicBlock &Succ) {
    auto BI = branch_info_begin();
    for (auto BB : successors()) {
      if (&Succ == BB)
        return *BI;
      ++BI;
    }
    llvm_unreachable("Invalid successor");
    return *BI;
  }

  /// Try to compute the taken and misprediction frequencies for the given
  /// successor.  The result is an error if no information can be found.
  ErrorOr<std::pair<double, double>>
  getBranchStats(const BinaryBasicBlock *Succ) const;

  /// If the basic block ends with a conditional branch (possibly followed by
  /// an unconditional branch) and thus has 2 successor, reverse the order of
  /// its successors in CFG, update branch info, and return true. If the basic
  /// block does not have 2 successors return false.
  bool swapConditionalSuccessors();

  /// Add an instruction with unconditional control transfer to \p Successor
  /// basic block to the end of this basic block.
  void addBranchInstruction(const BinaryBasicBlock *Successor);

  /// Add an instruction with tail call control transfer to \p Target
  /// to the end of this basic block.
  void addTailCallInstruction(const MCSymbol *Target);

  /// Return the number of call instructions in this basic block.
  uint32_t getNumCalls() const;

  /// Get landing pad with given label. Returns nullptr if no such
  /// landing pad is found.
  BinaryBasicBlock *getLandingPad(const MCSymbol *Label) const;

  /// Return local name for the block.
  StringRef getName() const {
    return Label->getName();
  }

  /// Add instruction at the end of this basic block.
  /// Returns the index of the instruction in the Instructions vector of the BB.
  uint32_t addInstruction(MCInst &&Inst) {
    adjustNumPseudos(Inst, 1);
    Instructions.emplace_back(Inst);
    return Instructions.size() - 1;
  }

  /// Add instruction at the end of this basic block.
  /// Returns the index of the instruction in the Instructions vector of the BB.
  uint32_t addInstruction(const MCInst &Inst) {
    adjustNumPseudos(Inst, 1);
    Instructions.push_back(Inst);
    return Instructions.size() - 1;
  }

  /// Add a range of instructions to the end of this basic block.
  template <typename Itr>
  void addInstructions(Itr Begin, Itr End) {
    while (Begin != End) {
      addInstruction(*Begin++);
    }
  }

  /// Add instruction before Pos in this basic block.
  template <typename Itr>
  Itr insertPseudoInstr(Itr Pos, MCInst &Instr) {
    ++NumPseudos;
    return Instructions.emplace(Pos, Instr);
  }

  /// Return the number of pseudo instructions in the basic block.
  uint32_t getNumPseudos() const;

  /// Return the number of emitted instructions for this basic block.
  uint32_t getNumNonPseudos() const {
    return size() - getNumPseudos();
  }

  /// Return iterator to the first non-pseudo instruction or end()
  /// if no such instruction was found.
  iterator getFirstNonPseudo();

  /// Return a pointer to the first non-pseudo instruction in this basic
  /// block.  Returns nullptr if none exists.
  MCInst *getFirstNonPseudoInstr() {
    auto II = getFirstNonPseudo();
    return II == Instructions.end() ? nullptr : &*II;
  }

  /// Return reverse iterator to the last non-pseudo instruction or rend()
  /// if no such instruction was found.
  reverse_iterator getLastNonPseudo();

  /// Return a pointer to the last non-pseudo instruction in this basic
  /// block.  Returns nullptr if none exists.
  MCInst *getLastNonPseudoInstr() {
    auto RII = getLastNonPseudo();
    return RII == Instructions.rend() ? nullptr : &*RII;
  }

  /// Set CFI state at entry to this basic block.
  void setCFIState(int32_t NewCFIState) {
    assert((CFIState == -1 || NewCFIState == CFIState) &&
           "unexpected change of CFI state for basic block");
    CFIState = NewCFIState;
  }

  /// Return CFI state (expected) at entry of this basic block.
  int32_t getCFIState() const {
    assert(CFIState >= 0 && "unknown CFI state");
    return CFIState;
  }

  /// Calculate and return CFI state right before instruction \p Instr in
  /// this basic block. If \p Instr is nullptr then return the state at
  /// the end of the basic block.
  int32_t getCFIStateAtInstr(const MCInst *Instr) const;

  /// Calculate and return CFI state after execution of this basic block.
  /// The state depends on CFI state at entry and CFI instructions inside the
  /// basic block.
  int32_t getCFIStateAtExit() const {
    return getCFIStateAtInstr(nullptr);
  }

  /// Set minimum alignment for the basic block.
  void setAlignment(uint64_t Align) {
    Alignment = Align;
  }

  /// Return required alignment for the block.
  uint64_t getAlignment() const {
    return Alignment;
  }

  /// Adds block to successor list, and also updates predecessor list for
  /// successor block.
  /// Set branch info for this path.
  void addSuccessor(BinaryBasicBlock *Succ,
                    uint64_t Count = 0,
                    uint64_t MispredictedCount = 0);

  void addSuccessor(BinaryBasicBlock *Succ, const BinaryBranchInfo &BI) {
    addSuccessor(Succ, BI.Count, BI.MispredictedCount);
  }

  /// Add a range of successors.
  template <typename Itr>
  void addSuccessors(Itr Begin, Itr End) {
    while (Begin != End) {
      addSuccessor(*Begin++);
    }
  }

  /// Add a range of successors with branch info.
  template <typename Itr, typename BrItr>
  void addSuccessors(Itr Begin, Itr End, BrItr BrBegin, BrItr BrEnd) {
    assert(std::distance(Begin, End) == std::distance(BrBegin, BrEnd));
    while (Begin != End) {
      addSuccessor(*Begin++, *BrBegin++);
    }
  }

  /// Replace Succ with NewSucc.  This routine is helpful for preserving
  /// the order of conditional successors when editing the CFG.
  void replaceSuccessor(BinaryBasicBlock *Succ,
                        BinaryBasicBlock *NewSucc,
                        uint64_t Count = 0,
                        uint64_t MispredictedCount = 0);

  /// Remove /p Succ basic block from the list of successors. Update the
  /// list of predecessors of /p Succ and update branch info.
  void removeSuccessor(BinaryBasicBlock *Succ);

  /// Remove a range of successor blocks.
  template <typename Itr>
  void removeSuccessors(Itr Begin, Itr End) {
    while (Begin != End) {
      removeSuccessor(*Begin++);
    }
  }

  /// Remove useless duplicate successors.  When the conditional
  /// successor is the same as the unconditional successor, we can
  /// remove the conditional successor and branch instruction.
  void removeDuplicateConditionalSuccessor(MCInst *CondBranch);

  /// Test if BB is a predecessor of this block.
  bool isPredecessor(const BinaryBasicBlock *BB) const {
    auto Itr = std::find(Predecessors.begin(), Predecessors.end(), BB);
    return Itr != Predecessors.end();
  }

  /// Test if BB is a successor of this block.
  bool isSuccessor(const BinaryBasicBlock *BB) const {
    auto Itr = std::find(Successors.begin(), Successors.end(), BB);
    return Itr != Successors.end();
  }

  /// Test if this BB has a valid execution count.
  bool hasProfile() const {
    return ExecutionCount != COUNT_NO_PROFILE;
  }

  /// Return the information about the number of times this basic block was
  /// executed.
  ///
  /// Return COUNT_NO_PROFILE if there's no profile info.
  uint64_t getExecutionCount() const {
    return ExecutionCount;
  }

  /// Return the execution count for blocks with known profile.
  /// Return 0 if the block has no profile.
  uint64_t getKnownExecutionCount() const {
    return !hasProfile() ? 0 : ExecutionCount;
  }

  /// Set the execution count for this block.
  void setExecutionCount(uint64_t Count) {
    ExecutionCount = Count;
  }

  bool isEntryPoint() const {
    return IsEntryPoint;
  }

  void setEntryPoint(bool Value = true) {
    IsEntryPoint = Value;
  }

  bool isValid() const {
    return IsValid;
  }

  void markValid(const bool Valid) {
    IsValid = Valid;
  }

  bool isCold() const {
    return IsCold;
  }

  void setIsCold(const bool Flag) {
    IsCold = Flag;
  }

  /// Return true if the block can be outlined. At the moment we disallow
  /// outlining of blocks that can potentially throw exceptions or are
  /// the beginning of a landing pad. The entry basic block also can
  /// never be outlined.
  bool canOutline() const {
    return CanOutline;
  }

  void setCanOutline(const bool Flag) {
    CanOutline = Flag;
  }

  /// Erase pseudo instruction at a given iterator.
  iterator erasePseudoInstruction(iterator II) {
    --NumPseudos;
    return Instructions.erase(II);
  }

  /// Erase given (non-pseudo) instruction if found.
  /// Warning: this will invalidate succeeding instruction pointers.
  bool eraseInstruction(MCInst *Inst) {
    return replaceInstruction(Inst, std::vector<MCInst>());
  }

  /// Erase non-pseudo instruction at a given iterator \p II.
  iterator eraseInstruction(iterator II) {
    return Instructions.erase(II);
  }

  /// Retrieve iterator for \p Inst or return end iterator if instruction is not
  /// from this basic block.
  decltype(Instructions)::iterator findInstruction(const MCInst *Inst) {
    if (Instructions.empty())
      return Instructions.end();
    size_t Index = Inst - &Instructions[0];
    return Index >= Instructions.size() ? Instructions.end()
                                        : Instructions.begin() + Index;
  }

  /// Replace an instruction with a sequence of instructions. Returns true
  /// if the instruction to be replaced was found and replaced.
  template <typename Itr>
  bool replaceInstruction(const MCInst *Inst, Itr Begin, Itr End) {
    auto I = findInstruction(Inst);
    if (I != Instructions.end()) {
      adjustNumPseudos(*Inst, -1);
      Instructions.insert(Instructions.erase(I), Begin, End);
      adjustNumPseudos(Begin, End, 1);
      return true;
    }
    return false;
  }

  bool replaceInstruction(const MCInst *Inst,
                          const std::vector<MCInst> &Replacement) {
    return replaceInstruction(Inst, Replacement.begin(), Replacement.end());
  }

  /// Insert \p NewInst before \p At, which must be an existing instruction in
  /// this BB. Return a pointer to the newly inserted instruction.
  iterator insertInstruction(iterator At, MCInst &&NewInst) {
    adjustNumPseudos(NewInst, 1);
    return Instructions.emplace(At, std::move(NewInst));
  }

  /// Helper to retrieve any terminators in \p BB before \p Pos. This is used
  /// to skip CFI instructions and to retrieve the first terminator instruction
  /// in basic blocks with two terminators (conditional jump and unconditional
  /// jump).
  MCInst *getTerminatorBefore(MCInst *Pos);

  /// Used to identify whether an instruction is before a terminator and whether
  /// moving it to the end of the BB would render it dead code.
  bool hasTerminatorAfter(MCInst *Pos);

  /// Split apart the instructions in this basic block starting at Inst.
  /// The instructions following Inst are removed and returned in a vector.
  std::vector<MCInst> splitInstructions(const MCInst *Inst) {
    std::vector<MCInst> SplitInst;

    assert(!Instructions.empty());
    while(&Instructions.back() != Inst) {
      SplitInst.push_back(Instructions.back());
      Instructions.pop_back();
    }
    std::reverse(SplitInst.begin(), SplitInst.end());
    NumPseudos = 0;
    adjustNumPseudos(Instructions.begin(), Instructions.end(), 1);
    return SplitInst;
  }

  /// Sets address of the basic block in the output.
  void setOutputStartAddress(uint64_t Address) {
    OutputAddressRange.first = Address;
  }

  /// Sets address past the end of the basic block in the output.
  void setOutputEndAddress(uint64_t Address) {
    OutputAddressRange.second = Address;
  }

  /// Gets the memory address range of this BB in the input binary.
  std::pair<uint64_t, uint64_t> getInputAddressRange() const {
    return InputRange;
  }

  /// Gets the memory address range of this BB in the output binary.
  std::pair<uint64_t, uint64_t> getOutputAddressRange() const {
    return OutputAddressRange;
  }

  /// Return size of the basic block in the output binary.
  uint64_t getOutputSize() const {
    return OutputAddressRange.second - OutputAddressRange.first;
  }

  BinaryFunction *getFunction() const {
    return Function;
  }

  /// Analyze and interpret the terminators of this basic block. TBB must be
  /// initialized with the original fall-through for this BB.
  bool analyzeBranch(const MCSymbol *&TBB,
                     const MCSymbol *&FBB,
                     MCInst *&CondBranch,
                     MCInst *&UncondBranch);

  /// Printer required for printing dominator trees.
  void printAsOperand(raw_ostream &OS, bool PrintType = true) {
    if (PrintType) {
      OS << "basic block ";
    }
    OS << getName();
  }

  /// A simple dump function for debugging.
  void dump() const;

  /// Validate successor invariants for this BB.
  bool validateSuccessorInvariants();

  /// Return offset of the basic block from the function start on input.
  uint32_t getInputOffset() const {
    return InputRange.first;
  }

  /// Return offset from the function start to location immediately past
  /// the end of the basic block.
  uint32_t getEndOffset() const {
    return InputRange.second;
  }

  /// Return size of the basic block on input.
  uint32_t getOriginalSize() const {
    return InputRange.second - InputRange.first;
  }

  /// Returns an estimate of size of basic block during run time.
  uint64_t estimateSize() const;

private:
  void adjustNumPseudos(const MCInst &Inst, int Sign);

  template <typename Itr>
  void adjustNumPseudos(Itr Begin, Itr End, int Sign) {
    while (Begin != End) {
      adjustNumPseudos(*Begin++, Sign);
    }
  }

  /// Adds predecessor to the BB. Most likely you don't need to call this.
  void addPredecessor(BinaryBasicBlock *Pred);

  /// Remove predecessor of the basic block. Don't use directly, instead
  /// use removeSuccessor() function.
  void removePredecessor(BinaryBasicBlock *Pred);

  /// Return offset of the basic block from the function start.
  uint32_t getOffset() const {
    return InputRange.first;
  }

  /// Set end offset of this basic block.
  void setEndOffset(uint32_t Offset) {
    InputRange.second = Offset;
  }

  /// Get the index of this basic block.
  unsigned getIndex() const {
    assert(isValid());
    return Index;
  }

  /// Set the index of this basic block.
  void setIndex(unsigned I) {
    Index = I;
  }

  /// Return index in the current layout. The user is responsible for
  /// making sure the indices are up to date,
  /// e.g. by calling BinaryFunction::updateLayoutIndices();
  unsigned getLayoutIndex() const {
    assert(isValid());
    return LayoutIndex;
  }

  /// Set layout index. To be used by BinaryFunction.
  void setLayoutIndex(unsigned Index) {
    LayoutIndex = Index;
  }
};

bool operator<(const BinaryBasicBlock &LHS, const BinaryBasicBlock &RHS);

} // namespace bolt


// GraphTraits specializations for basic block graphs (CFGs)
template <> struct GraphTraits<bolt::BinaryBasicBlock *> {
  using NodeType = bolt::BinaryBasicBlock;
  using ChildIteratorType = bolt::BinaryBasicBlock::succ_iterator;

  static NodeType *getEntryNode(bolt::BinaryBasicBlock *BB) { return BB; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->succ_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->succ_end();
  }
};

template <> struct GraphTraits<const bolt::BinaryBasicBlock *> {
  using NodeType = const bolt::BinaryBasicBlock;
  using ChildIteratorType = bolt::BinaryBasicBlock::const_succ_iterator;

  static NodeType *getEntryNode(const bolt::BinaryBasicBlock *BB) {
    return BB;
  }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->succ_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->succ_end();
  }
};

template <> struct GraphTraits<Inverse<bolt::BinaryBasicBlock *>> {
  using NodeType = bolt::BinaryBasicBlock;
  using ChildIteratorType = bolt::BinaryBasicBlock::pred_iterator;
  static NodeType *getEntryNode(Inverse<bolt::BinaryBasicBlock *> G) {
    return G.Graph;
  }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->pred_end();
  }
};

template <> struct GraphTraits<Inverse<const bolt::BinaryBasicBlock *>> {
  using NodeType = const bolt::BinaryBasicBlock;
  using ChildIteratorType = bolt::BinaryBasicBlock::const_pred_iterator;
  static NodeType *getEntryNode(Inverse<const bolt::BinaryBasicBlock *> G) {
    return G.Graph;
  }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->pred_end();
  }
};

} // namespace llvm

#endif
