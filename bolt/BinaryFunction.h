//===--- BinaryFunction.h - Interface for machine-level function ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to function in binary (machine) form. This is assembly-level
// code representation with the control flow.
//
// TODO: memory management for instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_FUNCTION_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_FUNCTION_H

#include "BinaryBasicBlock.h"
#include "BinaryContext.h"
#include "BinaryLoop.h"
#include "DataReader.h"
#include "DebugData.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ilist.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <unordered_map>
#include <vector>

using namespace llvm::object;

namespace llvm {

class DWARFCompileUnit;
class DWARFDebugInfoEntryMinimal;

namespace bolt {

using DWARFUnitLineTable = std::pair<DWARFCompileUnit *,
                                     const DWARFDebugLine::LineTable *>;

/// Class encapsulating runtime statistics about an execution unit.
class DynoStats {

#define DYNO_STATS\
  D(FIRST_DYNO_STAT,              "<reserved>", Fn)\
  D(FORWARD_COND_BRANCHES,        "executed forward branches", Fn)\
  D(FORWARD_COND_BRANCHES_TAKEN,  "taken forward branches", Fn)\
  D(BACKWARD_COND_BRANCHES,       "executed backward branches", Fn)\
  D(BACKWARD_COND_BRANCHES_TAKEN, "taken backward branches", Fn)\
  D(UNCOND_BRANCHES,              "executed unconditional branches", Fn)\
  D(FUNCTION_CALLS,               "all function calls", Fn)\
  D(INDIRECT_CALLS,               "indirect calls", Fn)\
  D(PLT_CALLS,                    "PLT calls", Fn)\
  D(INSTRUCTIONS,                 "executed instructions", Fn)\
  D(JUMP_TABLE_BRANCHES,          "taken jump table branches", Fn)\
  D(ALL_BRANCHES,                 "total branches",\
      Fadd(ALL_CONDITIONAL, UNCOND_BRANCHES))\
  D(ALL_TAKEN,                    "taken branches",\
      Fadd(TAKEN_CONDITIONAL, UNCOND_BRANCHES))\
  D(NONTAKEN_CONDITIONAL,         "non-taken conditional branches",\
      Fsub(ALL_CONDITIONAL, TAKEN_CONDITIONAL))\
  D(TAKEN_CONDITIONAL,            "taken conditional branches",\
      Fadd(FORWARD_COND_BRANCHES_TAKEN, BACKWARD_COND_BRANCHES_TAKEN))\
  D(ALL_CONDITIONAL,              "all conditional branches",\
      Fadd(FORWARD_COND_BRANCHES, BACKWARD_COND_BRANCHES))\
  D(LAST_DYNO_STAT,               "<reserved>", Fn)

public:
#define D(name, ...) name,
  enum Category : uint8_t { DYNO_STATS };
#undef D


private:
  uint64_t Stats[LAST_DYNO_STAT];

#define D(name, desc, ...) desc,
  static constexpr const char *Desc[] = { DYNO_STATS };
#undef D

public:
  DynoStats() {
    for (auto Stat = FIRST_DYNO_STAT + 0; Stat < LAST_DYNO_STAT; ++Stat)
      Stats[Stat] = 0;
  }

  uint64_t &operator[](size_t I) {
    assert(I > FIRST_DYNO_STAT && I < LAST_DYNO_STAT &&
           "index out of bounds");
    return Stats[I];
  }

  uint64_t operator[](size_t I) const {
    switch (I) {
#define D(name, desc, func) \
    case name: \
      return func;
#define Fn Stats[I]
#define Fadd(a, b) operator[](a) + operator[](b)
#define Fsub(a, b) operator[](a) - operator[](b)
#define F(a) operator[](a)
#define Radd(a, b) (a + b)
#define Rsub(a, b) (a - b)
    DYNO_STATS
#undef Fn
#undef D
    default:
      llvm_unreachable("index out of bounds");
    }
    return 0;
  }

  void print(raw_ostream &OS, const DynoStats *Other = nullptr) const;

  void operator+=(const DynoStats &Other);
  bool operator<(const DynoStats &Other) const;
  bool lessThan(const DynoStats &Other, ArrayRef<Category> Keys) const;

  static const char* Description(const Category C) {
    return Desc[C];
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const DynoStats &Stats) {
  Stats.print(OS, nullptr);
  return OS;
}

DynoStats operator+(const DynoStats &A, const DynoStats &B);

/// BinaryFunction is a representation of machine-level function.
///
/// We use the term "Binary" as "Machine" was already taken.
class BinaryFunction : public AddressRangesOwner {
public:
  enum class State : char {
    Empty = 0,        /// Function body is empty
    Disassembled,     /// Function have been disassembled
    CFG,              /// Control flow graph have been built
    Assembled,        /// Function has been assembled in memory
  };

  /// Settings for splitting function bodies into hot/cold partitions.
  enum SplittingType : char {
    ST_NONE = 0,      /// Do not split functions
    ST_EH,            /// Split blocks comprising landing pads
    ST_LARGE,         /// Split functions that exceed maximum size in addition
                      /// to landing pads.
    ST_ALL,           /// Split all functions
  };

  /// Choose which strategy should the block layout heuristic prioritize when
  /// facing conflicting goals.
  enum LayoutType : char {
    /// LT_NONE - do not change layout of basic blocks
    LT_NONE = 0, /// no reordering
    /// LT_REVERSE - reverse the order of basic blocks, meant for testing
    /// purposes. The first basic block is left intact and the rest are
    /// put in the reverse order.
    LT_REVERSE,
    /// LT_OPTIMIZE - optimize layout of basic blocks based on profile.
    LT_OPTIMIZE,
    /// LT_OPTIMIZE_BRANCH is an implementation of what is suggested in Pettis'
    /// paper (PLDI '90) about block reordering, trying to minimize branch
    /// mispredictions.
    LT_OPTIMIZE_BRANCH,
    /// LT_OPTIMIZE_CACHE piggybacks on the idea from Ispike paper (CGO '04)
    /// that suggests putting frequently executed chains first in the layout.
    LT_OPTIMIZE_CACHE,
    /// Create clusters and use random order for them.
    LT_OPTIMIZE_SHUFFLE,
  };

  enum JumpTableSupportLevel : char {
    JTS_NONE = 0,       /// Disable jump tables support
    JTS_BASIC = 1,      /// Enable basic jump tables support
    JTS_SPLIT = 2,      /// Enable hot/cold splitting of jump tables
    JTS_AGGRESSIVE = 3, /// Aggressive splitting of jump tables
  };

  static constexpr uint64_t COUNT_NO_PROFILE =
    std::numeric_limits<uint64_t>::max();
  // Function size, in number of BBs, above which we fallback to a heuristic
  // solution to the layout problem instead of seeking the optimal one.
  static constexpr uint64_t FUNC_SIZE_THRESHOLD = 10;

  using BasicBlockOrderType = std::vector<BinaryBasicBlock *>;

private:

  /// Current state of the function.
  State CurrentState{State::Empty};

  /// A list of function names.
  std::vector<std::string> Names;

  /// Containing section
  SectionRef Section;

  /// Address of the function in memory. Also could be an offset from
  /// base address for position independent binaries.
  uint64_t Address;

  /// Address of an identical function that can replace this one. By default
  /// this is the same as the address of this functions, and the icf pass can
  /// potentially set it to some other function's address.
  ///
  /// In case multiple functions are identical to each other, one of the
  /// functions (the representative) will point to its own address, while the
  /// rest of the functions will point to the representative through one or
  /// more steps.
  uint64_t IdenticalFunctionAddress;

  /// Original size of the function.
  uint64_t Size;

  /// Offset in the file.
  uint64_t FileOffset{0};

  /// Maximum size this function is allowed to have.
  uint64_t MaxSize{std::numeric_limits<uint64_t>::max()};

  /// Alignment requirements for the function.
  uint64_t Alignment{1};

  const MCSymbol *PersonalityFunction{nullptr};
  uint8_t PersonalityEncoding{dwarf::DW_EH_PE_sdata4 | dwarf::DW_EH_PE_pcrel};

  BinaryContext &BC;

  std::unique_ptr<BinaryLoopInfo> BLI;

  /// False if the function is too complex to reconstruct its control
  /// flow graph and re-assemble.
  bool IsSimple{true};

  /// True if this function needs to be emitted in two separate parts, one for
  /// the hot basic blocks and another for the cold basic blocks.
  bool IsSplit{false};

  /// Indicate if this function has associated exception handling metadata.
  bool HasEHRanges{false};

  /// True if the function uses DW_CFA_GNU_args_size CFIs.
  bool UsesGnuArgsSize{false};

  /// The address for the code for this function in codegen memory.
  uint64_t ImageAddress{0};

  /// The size of the code in memory.
  uint64_t ImageSize{0};

  /// Name for the section this function code should reside in.
  std::string CodeSectionName;

  /// The profile data for the number of times the function was executed.
  uint64_t ExecutionCount{COUNT_NO_PROFILE};

  /// Profile match ration.
  float ProfileMatchRatio{0.0};

  /// Score of the function (estimated number of instructions executed,
  /// according to profile data). -1 if the score has not been calculated yet.
  int64_t FunctionScore{-1};

  /// Original LSDA address for the function.
  uint64_t LSDAAddress{0};

  /// Landing pads for the function.
  std::set<const MCSymbol *> LandingPads;

  /// Associated DIEs in the .debug_info section with their respective CUs.
  /// There can be multiple because of identical code folding.
  std::vector<std::pair<const DWARFDebugInfoEntryMinimal *,
                        DWARFCompileUnit *>> SubprogramDIEs;

  /// Offset of this function's address ranges in the .debug_ranges section of
  /// the output binary.
  uint32_t AddressRangesOffset{-1U};

  /// Get basic block index assuming it belongs to this function.
  unsigned getIndex(const BinaryBasicBlock *BB) const {
    assert(BB->getIndex() < BasicBlocks.size());
    return BB->getIndex();
  }

  /// Return basic block that originally contained offset \p Offset
  /// from the function start.
  BinaryBasicBlock *getBasicBlockContainingOffset(uint64_t Offset);

  /// Return basic block that started at offset \p Offset.
  BinaryBasicBlock *getBasicBlockAtOffset(uint64_t Offset) {
    BinaryBasicBlock *BB = getBasicBlockContainingOffset(Offset);
    return BB && BB->getOffset() == Offset ? BB : nullptr;
  }

  /// Release memory taken by the list.
  template<typename T> BinaryFunction &clearList(T& List) {
    T TempList;
    TempList.swap(List);
    return *this;
  }

  BinaryFunction &updateState(BinaryFunction::State State) {
    CurrentState = State;
    return *this;
  }

  /// Helper function that compares an instruction of this function to the
  /// given instruction of the given function. The functions should have
  /// identical CFG.
  bool isInstrEquivalentWith(
      const MCInst &Inst, const BinaryBasicBlock &BB, const MCInst &InstOther,
      const BinaryBasicBlock &BBOther, const BinaryFunction &BF) const;

  /// Helper function that compares the callees of two call instructions.
  /// Callees are considered equivalent if both refer to the same function
  /// or if both calls are recursive. Instructions should have same opcodes
  /// and same number of operands. Returns true and the callee operand index
  /// when callees are quivalent, and false, 0 otherwise.
  std::pair<bool, unsigned> isCalleeEquivalentWith(
      const MCInst &Inst, const BinaryBasicBlock &BB, const MCInst &InstOther,
      const BinaryBasicBlock &BBOther, const BinaryFunction &BF) const;

  /// Helper function that compares the targets two jump or invoke instructions.
  /// A target of an invoke we consider its landing pad basic block. The
  /// corresponding functions should have identical CFG. Instructions should
  /// have same opcodes and same number of operands. Returns true and the target
  /// operand index when targets are equivalent,  and false, 0 otherwise.
  std::pair<bool, unsigned> isTargetEquivalentWith(
      const MCInst &Inst, const BinaryBasicBlock &BB, const MCInst &InstOther,
      const BinaryBasicBlock &BBOther, const BinaryFunction &BF,
      bool AreInvokes) const;

  /// Clear the landing pads for all blocks contained in the range of
  /// [StartIndex, StartIndex + NumBlocks).  This also has the effect of
  /// removing throws that point to any of these blocks.
  void clearLandingPads(const unsigned StartIndex, const unsigned NumBlocks);

  /// Add landing pads for all blocks in the range
  /// [StartIndex, StartIndex + NumBlocks) using LPToBBIndex.
  void addLandingPads(const unsigned StartIndex, const unsigned NumBlocks);

  /// Recompute the landing pad information for all the basic blocks in the
  /// range of [StartIndex to StartIndex + NumBlocks).
  void recomputeLandingPads(const unsigned StartIndex,
                            const unsigned NumBlocks);

  using BranchListType = std::vector<std::pair<uint32_t, uint32_t>>;
  BranchListType TakenBranches; /// All local taken branches.
  BranchListType FTBranches;    /// All fall-through branches.

  /// Storage for all landing pads and their corresponding invokes.
  using LandingPadsMapType = std::map<const MCSymbol *, std::vector<unsigned> >;
  LandingPadsMapType LPToBBIndex;

  /// Map offset in the function to a local label.
  using LabelsMapType = std::map<uint32_t, MCSymbol *>;
  LabelsMapType Labels;

  /// Temporary holder of instructions before CFG is constructed.
  /// Map offset in the function to MCInst.
  using InstrMapType = std::map<uint32_t, MCInst>;
  InstrMapType Instructions;

  /// Temporary holder of offsets of tail call instructions before CFG is
  /// constructed. Map from offset to the corresponding target address of the
  /// tail call.
  using TailCallOffsetMapType = std::map<uint32_t, uint64_t>;
  TailCallOffsetMapType TailCallOffsets;

  /// Temporary holder of tail call terminated basic blocks used during CFG
  /// construction. Map from tail call terminated basic block to a struct with
  /// information about the tail call.
  struct TailCallInfo {
    uint32_t Offset;            // offset of the tail call from the function
                                // start
    uint32_t Index;             // index of the tail call in the basic block
    uint64_t TargetAddress;     // address of the callee
    uint64_t Count{0};          // taken count from profile data
    uint64_t Mispreds{0};       // mispredicted count from progile data
    uint32_t CFIStateBefore{0}; // CFI state before the tail call instruction

    TailCallInfo(uint32_t Offset, uint32_t Index, uint64_t TargetAddress) :
      Offset(Offset), Index(Index), TargetAddress(TargetAddress) { }
  };
  using TailCallBasicBlockMapType = std::map<BinaryBasicBlock *, TailCallInfo>;
  TailCallBasicBlockMapType TailCallTerminatedBlocks;

  /// List of DWARF CFI instructions. Original CFI from the binary must be
  /// sorted w.r.t. offset that it appears. We rely on this to replay CFIs
  /// if needed (to fix state after reordering BBs).
  using CFIInstrMapType = std::vector<MCCFIInstruction>;
  using cfi_iterator       = CFIInstrMapType::iterator;
  using const_cfi_iterator = CFIInstrMapType::const_iterator;
  CFIInstrMapType FrameInstructions;

  /// Exception handling ranges.
  struct CallSite {
    const MCSymbol *Start;
    const MCSymbol *End;
    const MCSymbol *LP;
    uint64_t Action;
  };
  std::vector<CallSite> CallSites;

  /// Binary blobs reprsenting action, type, and type index tables for this
  /// function' LSDA (exception handling).
  ArrayRef<uint8_t> LSDAActionAndTypeTables;
  ArrayRef<uint8_t> LSDATypeIndexTable;

  /// Marking for the beginning of language-specific data area for the function.
  MCSymbol *LSDASymbol{nullptr};

  /// Map to discover which CFIs are attached to a given instruction offset.
  /// Maps an instruction offset into a FrameInstructions offset.
  /// This is only relevant to the buildCFG phase and is discarded afterwards.
  std::multimap<uint32_t, uint32_t> OffsetToCFI;

  /// List of CFI instructions associated with the CIE (common to more than one
  /// function and that apply before the entry basic block).
  CFIInstrMapType CIEFrameInstructions;

  /// Representation of a jump table.
  ///
  /// The jump table may include other jump tables that are referenced by
  /// a different label at a different offset in this jump table.
  struct JumpTable {
    /// Original address.
    uint64_t Address;

    /// Size of the entry used for storage.
    std::size_t EntrySize;

    /// All the entries as labels.
    std::vector<MCSymbol *> Entries;

    /// All the entries as offsets into a function. Invalid after CFG is built.
    std::vector<uint64_t> OffsetEntries;

    /// Map <Offset> -> <Label> used for embedded jump tables. Label at 0 offset
    /// is the main label for the jump table.
    std::map<unsigned, MCSymbol *> Labels;

    /// Return the size of the jump table.
    uint64_t getSize() const {
      return Entries.size() * EntrySize;
    }

    /// Constructor.
    JumpTable(uint64_t Address,
              std::size_t EntrySize,
              decltype(Entries) &&Entries,
              decltype(OffsetEntries) &&OffsetEntries,
              decltype(Labels) &&Labels)
      : Address(Address), EntrySize(EntrySize), Entries(Entries),
        OffsetEntries(OffsetEntries), Labels(Labels)
    {}

    /// Dynamic number of times each entry in the table was referenced.
    /// Identical entries will have a shared count (identical for every
    /// entry in the set).
    std::vector<uint64_t> Counts;

    /// Total number of times this jump table was used.
    uint64_t Count{0};

    /// Emit jump table data. Callee supplies sections for the data.
    /// Return the number of total bytes emitted.
    uint64_t emit(MCStreamer *Streamer, MCSection *HotSection,
                  MCSection *ColdSection);

    /// Print for debugging purposes.
    void print(raw_ostream &OS) const;
  };

  /// All compound jump tables for this function.
  /// <OriginalAddress> -> <JumpTable>
  std::map<uint64_t, JumpTable> JumpTables;

  /// Return jump table that covers a given \p Address in memory.
  JumpTable *getJumpTableContainingAddress(uint64_t Address) {
    auto JTI = JumpTables.upper_bound(Address);
    if (JTI == JumpTables.begin())
      return nullptr;
    --JTI;
    if (JTI->first + JTI->second.getSize() > Address) {
      return &JTI->second;
    }
    return nullptr;
  }

  /// All jump table sites in the function.
  std::vector<std::pair<uint64_t, uint64_t>> JTSites;

  // Blocks are kept sorted in the layout order. If we need to change the
  // layout (if BasicBlocksLayout stores a different order than BasicBlocks),
  // the terminating instructions need to be modified.
  using BasicBlockListType = std::vector<BinaryBasicBlock*>;
  BasicBlockListType BasicBlocks;
  BasicBlockOrderType BasicBlocksLayout;

  // At each basic block entry we attach a CFI state to detect if reordering
  // corrupts the CFI state for a block. The CFI state is simply the index in
  // FrameInstructions for the CFI responsible for creating this state.
  // This vector is indexed by BB index.
  std::vector<uint32_t> BBCFIState;

  /// Symbol in the output.
  MCSymbol *OutputSymbol;

  /// Symbol at the end of the function.
  MCSymbol *FunctionEndLabel{nullptr};

  /// Unique number associated with the function.
  uint64_t  FunctionNumber;

  /// Count the number of functions created.
  static uint64_t Count;

  template <typename Itr, typename T>
  class Iterator : public std::iterator<std::bidirectional_iterator_tag, T> {
   public:
    Iterator &operator++() { ++itr; return *this; }
    Iterator &operator--() { --itr; return *this; }
    Iterator operator++(int) { auto tmp(itr); itr++; return tmp; }
    Iterator operator--(int) { auto tmp(itr); itr--; return tmp; }
    bool operator==(const Iterator& other) const { return itr == other.itr; }
    bool operator!=(const Iterator& other) const { return itr != other.itr; }
    T& operator*() { return **itr; }
    Iterator(Itr itr) : itr(itr) { }
   private:
    Itr itr;
  };

  BinaryFunction& operator=(const BinaryFunction &) = delete;
  BinaryFunction(const BinaryFunction &) = delete;

  friend class RewriteInstance;

  /// Creation should be handled by RewriteInstance::createBinaryFunction().
  BinaryFunction(const std::string &Name, SectionRef Section, uint64_t Address,
                 uint64_t Size, BinaryContext &BC, bool IsSimple) :
      Names({Name}), Section(Section), Address(Address),
      IdenticalFunctionAddress(Address), Size(Size), BC(BC), IsSimple(IsSimple),
      CodeSectionName(".text." + Name), FunctionNumber(++Count) {
    OutputSymbol = BC.Ctx->getOrCreateSymbol(Name);
  }

public:

  BinaryFunction(BinaryFunction &&) = default;

  typedef Iterator<BasicBlockListType::iterator, BinaryBasicBlock> iterator;
  typedef Iterator<BasicBlockListType::const_iterator,
                   const BinaryBasicBlock> const_iterator;
  typedef Iterator<BasicBlockListType::reverse_iterator,
                   BinaryBasicBlock> reverse_iterator;
  typedef Iterator<BasicBlockListType::const_reverse_iterator,
                   const BinaryBasicBlock> const_reverse_iterator;

  typedef BasicBlockOrderType::iterator order_iterator;
  typedef BasicBlockOrderType::const_iterator const_order_iterator;
  typedef BasicBlockOrderType::reverse_iterator reverse_order_iterator;
  typedef BasicBlockOrderType::const_reverse_iterator
    const_reverse_order_iterator;

  // CFG iterators.
  iterator                 begin()       { return BasicBlocks.begin(); }
  const_iterator           begin() const { return BasicBlocks.begin(); }
  iterator                 end  ()       { return BasicBlocks.end();   }
  const_iterator           end  () const { return BasicBlocks.end();   }

  reverse_iterator        rbegin()       { return BasicBlocks.rbegin(); }
  const_reverse_iterator  rbegin() const { return BasicBlocks.rbegin(); }
  reverse_iterator        rend  ()       { return BasicBlocks.rend();   }
  const_reverse_iterator  rend  () const { return BasicBlocks.rend();   }

  unsigned                  size() const { return (unsigned)BasicBlocks.size();}
  bool                     empty() const { return BasicBlocks.empty(); }
  const BinaryBasicBlock &front() const  { return *BasicBlocks.front(); }
        BinaryBasicBlock &front()        { return *BasicBlocks.front(); }
  const BinaryBasicBlock & back() const  { return *BasicBlocks.back(); }
        BinaryBasicBlock & back()        { return *BasicBlocks.back(); }

  order_iterator       layout_begin()    { return BasicBlocksLayout.begin(); }
  const_order_iterator layout_begin()    const
                                         { return BasicBlocksLayout.begin(); }
  order_iterator       layout_end()      { return BasicBlocksLayout.end(); }
  const_order_iterator layout_end()      const
                                         { return BasicBlocksLayout.end(); }
  reverse_order_iterator       layout_rbegin()
                                         { return BasicBlocksLayout.rbegin(); }
  const_reverse_order_iterator layout_rbegin() const
                                         { return BasicBlocksLayout.rbegin(); }
  reverse_order_iterator       layout_rend()
                                         { return BasicBlocksLayout.rend(); }
  const_reverse_order_iterator layout_rend()   const
                                         { return BasicBlocksLayout.rend(); }
  unsigned layout_size()  const { return (unsigned)BasicBlocksLayout.size(); }
  bool     layout_empty() const { return BasicBlocksLayout.empty(); }
  const BinaryBasicBlock *layout_front() const
                                         { return BasicBlocksLayout.front(); }
        BinaryBasicBlock *layout_front() { return BasicBlocksLayout.front(); }
  const BinaryBasicBlock *layout_back()  const
                                         { return BasicBlocksLayout.back(); }
        BinaryBasicBlock *layout_back()  { return BasicBlocksLayout.back(); }

  inline iterator_range<order_iterator> layout() {
    return iterator_range<order_iterator>(BasicBlocksLayout.begin(),
                                          BasicBlocksLayout.end());
  }

  inline iterator_range<const_order_iterator> layout() const {
    return iterator_range<const_order_iterator>(BasicBlocksLayout.begin(),
                                                BasicBlocksLayout.end());
  }

  cfi_iterator        cie_begin()       { return CIEFrameInstructions.begin(); }
  const_cfi_iterator  cie_begin() const { return CIEFrameInstructions.begin(); }
  cfi_iterator        cie_end()         { return CIEFrameInstructions.end(); }
  const_cfi_iterator  cie_end()   const { return CIEFrameInstructions.end(); }
  bool                cie_empty() const { return CIEFrameInstructions.empty(); }

  inline iterator_range<cfi_iterator> cie() {
    return iterator_range<cfi_iterator>(cie_begin(), cie_end());
  }
  inline iterator_range<const_cfi_iterator> cie() const {
    return iterator_range<const_cfi_iterator>(cie_begin(), cie_end());
  }

  /// Modify code layout making necessary adjustments to instructions at the
  /// end of basic blocks.
  void modifyLayout(LayoutType Type, bool MinBranchClusters, bool Split);

  /// Find the loops in the CFG of the function and store information about
  /// them.
  void calculateLoopInfo();

  /// Returns if loop detection has been run for this function.
  bool hasLoopInfo() const {
    return BLI != nullptr;
  }

  /// Print loop information about the function.
  void printLoopInfo(raw_ostream &OS) const;

  /// View CFG in graphviz program
  void viewGraph() const;

  /// Dump CFG in graphviz format
  void dumpGraph(raw_ostream& OS) const;

  /// Dump CFG in graphviz format to file.
  void dumpGraphToFile(std::string Filename) const;

  /// Dump CFG in graphviz format to a file with a filename that is derived
  /// from the function name and Annotation strings.  Useful for dumping the
  /// CFG after an optimization pass.
  void dumpGraphForPass(std::string Annotation = "") const;

  /// Return BinaryContext for the function.
  const BinaryContext &getBinaryContext() const {
    return BC;
  }

  /// Return dynostats for the function.
  ///
  /// The function relies on branch instructions being in-sync with CFG for
  /// branch instructions stats. Thus it is better to call it after
  /// fixBranches().
  DynoStats getDynoStats() const;

  /// Returns the basic block after the given basic block in the layout or
  /// nullptr the last basic block is given.
  const BinaryBasicBlock *getBasicBlockAfter(const BinaryBasicBlock *BB) const {
    for (auto I = layout_begin(), E = layout_end(); I != E; ++I) {
      if (*I == BB && std::next(I) != E)
        return *std::next(I);
    }
    return nullptr;
  }

  BinaryBasicBlock *getBasicBlockAfter(const BinaryBasicBlock *BB) {
    for (auto I = layout_begin(), E = layout_end(); I != E; ++I) {
      if (*I == BB && std::next(I) != E)
        return *std::next(I);
    }
    return nullptr;
  }

  /// Return the name of the function as extracted from the binary file.
  /// If the function has multiple names - return the last one
  /// followed by "(*#<numnames>)".
  /// We should preferably only use getName() for diagnostics and use
  /// hasName() to match function name against a given string.
  ///
  /// We pick the last name from the list to match the name of the function
  /// in profile data for easier manual analysis.
  std::string getPrintName() const {
    return Names.size() == 1 ?
              Names.back() :
              (Names.back() + "(*" + std::to_string(Names.size()) + ")");
  }

  /// Check if (possibly one out of many) function name matches the given
  /// string. Use this member function instead of direct name comparison.
  bool hasName(std::string &FunctionName) const {
    for (auto &Name : Names)
      if (Name == FunctionName)
        return true;
    return false;
  }

  /// Return a vector of all possible names for the function.
  const std::vector<std::string> &getNames() const {
    return Names;
  }

  State getCurrentState() const {
    return CurrentState;
  }

  /// Return containing file section.
  SectionRef getSection() const {
    return Section;
  }

  /// Return original address of the function (or offset from base for PIC).
  uint64_t getAddress() const {
    return Address;
  }

  /// Get the original address for the given basic block within this function.
  uint64_t getBasicBlockOriginalAddress(const BinaryBasicBlock *BB) const {
    return Address + BB->getOffset();
  }

  /// Return offset of the function body in the binary file.
  uint64_t getFileOffset() const {
    return FileOffset;
  }

  /// Return (original) size of the function.
  uint64_t getSize() const {
    return Size;
  }

  /// Return the maximum size the body of the function could have.
  uint64_t getMaxSize() const {
    return MaxSize;
  }

  /// Return MC symbol associated with the function.
  /// All references to the function should use this symbol.
  MCSymbol *getSymbol() {
    return OutputSymbol;
  }

  /// Return MC symbol associated with the function (const version).
  /// All references to the function should use this symbol.
  const MCSymbol *getSymbol() const {
    return OutputSymbol;
  }

  /// Return MC symbol associated with the end of the function.
  MCSymbol *getFunctionEndLabel() {
    assert(BC.Ctx && "cannot be called with empty context");
    if (!FunctionEndLabel) {
      FunctionEndLabel = BC.Ctx->createTempSymbol("func_end", true);
    }
    return FunctionEndLabel;
  }

  /// Return internal section name for this function.
  StringRef getCodeSectionName() const {
    assert(!CodeSectionName.empty() && "no section name for function");
    return StringRef(CodeSectionName);
  }

  /// Return true if the function could be correctly processed.
  bool isSimple() const {
    return IsSimple;
  }

  /// Return true if the function body is non-contiguous.
  bool isSplit() const {
    return IsSplit;
  }

  /// Return true if the function has exception handling tables.
  bool hasEHRanges() const {
    return HasEHRanges;
  }

  /// Return true if the function uses DW_CFA_GNU_args_size CFIs.
  bool usesGnuArgsSize() const {
    return UsesGnuArgsSize;
  }

  const MCSymbol *getPersonalityFunction() const {
    return PersonalityFunction;
  }

  uint8_t getPersonalityEncoding() const {
    return PersonalityEncoding;
  }

  /// Return true if the function has CFI instructions
  bool hasCFI() const {
    return !FrameInstructions.empty() || !CIEFrameInstructions.empty();
  }

  /// Return unique number associated with the function.
  uint64_t getFunctionNumber() const {
    return FunctionNumber;
  }

  /// Return true if the given address \p PC is inside the function body.
  bool containsAddress(uint64_t PC) const {
    return Address <= PC && PC < Address + Size;
  }

  /// Register alternative function name.
  void addAlternativeName(std::string NewName) {
    Names.emplace_back(NewName);
  }

  /// Create a basic block at a given \p Offset in the
  /// function.
  /// If \p DeriveAlignment is true, set the alignment of the block based
  /// on the alignment of the existing offset.
  /// The new block is not inserted into the CFG.  The client must
  /// use insertBasicBlocks to add any new blocks to the CFG.
  ///
  std::unique_ptr<BinaryBasicBlock>
  createBasicBlock(uint64_t Offset,
                   MCSymbol *Label = nullptr,
                   bool DeriveAlignment = false) {
    assert(BC.Ctx && "cannot be called with empty context");
    if (!Label) {
      Label = BC.Ctx->createTempSymbol("BB", true);
    }
    auto BB = std::unique_ptr<BinaryBasicBlock>(
      new BinaryBasicBlock(this, Label, Offset));

    if (DeriveAlignment) {
      uint64_t DerivedAlignment = Offset & (1 + ~Offset);
      BB->setAlignment(std::min(DerivedAlignment, uint64_t(32)));
    }

    return BB;
  }

  /// Create a basic block at a given \p Offset in the
  /// function and append it to the end of list of blocks.
  /// If \p DeriveAlignment is true, set the alignment of the block based
  /// on the alignment of the existing offset.
  ///
  /// Returns NULL if basic block already exists at the \p Offset.
  BinaryBasicBlock *addBasicBlock(uint64_t Offset, MCSymbol *Label,
                                  bool DeriveAlignment = false) {
    assert((CurrentState == State::CFG || !getBasicBlockAtOffset(Offset)) &&
           "basic block already exists in pre-CFG state");
    auto BBPtr = createBasicBlock(Offset, Label, DeriveAlignment);
    BasicBlocks.emplace_back(BBPtr.release());

    auto BB = BasicBlocks.back();
    BB->setIndex(BasicBlocks.size() - 1);

    assert(CurrentState == State::CFG || std::is_sorted(begin(), end()));

    return BB;
  }

  /// Rebuilds BBs layout, ignoring dead BBs. Returns the number of removed
  /// BBs.
  unsigned eraseDeadBBs(std::map<BinaryBasicBlock *, bool> &ToPreserve);

  /// Get the relative order between two basic blocks in the original
  /// layout.  The result is > 0 if B occurs before A and < 0 if B
  /// occurs after A.  If A and B are the same block, the result is 0.
  signed getOriginalLayoutRelativeOrder(const BinaryBasicBlock *A,
                                        const BinaryBasicBlock *B) const {
    return getIndex(A) - getIndex(B);
  }

  /// Return basic block range that originally contained offset \p Offset
  /// from the function start to the function end.
  iterator_range<iterator> getBasicBlockRangeFromOffsetToEnd(uint64_t Offset) {
    auto *BB = getBasicBlockContainingOffset(Offset);
    return BB ? iterator_range<iterator>(BasicBlocks.begin() + getIndex(BB), end())
      : iterator_range<iterator>(end(), end());
  }

  /// Insert the BBs contained in NewBBs into the basic blocks for this
  /// function. Update the associated state of all blocks as needed, i.e.
  /// BB offsets, BB indices, and optionally CFI state. The new BBs are
  /// inserted after Start. This operation could affect fallthrough branches
  /// for Start.
  ///
  void insertBasicBlocks(
    BinaryBasicBlock *Start,
    std::vector<std::unique_ptr<BinaryBasicBlock>> &&NewBBs,
    bool UpdateCFIState = true);

  /// Update the basic block layout for this function.  The BBs from
  /// [Start->Index, Start->Index + NumNewBlocks) are inserted into the
  /// layout after the BB indicated by Start.
  void updateLayout(BinaryBasicBlock* Start, const unsigned NumNewBlocks);

  /// Update the basic block layout for this function.  The layout is
  /// computed from scratch using modifyLayout.
  void updateLayout(LayoutType Type, bool MinBranchClusters, bool Split);

  /// Make sure basic blocks' indices match the current layout.
  void updateLayoutIndices() const {
    unsigned Index = 0;
    for (auto *BB : layout()) {
      BB->setLayoutIndex(Index++);
    }
  }

  /// Determine direction of the branch based on the current layout.
  /// Callee is responsible of updating basic block indices prior to using
  /// this function (e.g. by calling BinaryFunction::updateLayoutIndices()).
  static bool isForwardBranch(const BinaryBasicBlock *From,
                       const BinaryBasicBlock *To) {
    assert(From->getFunction() == To->getFunction() &&
           "basic blocks should be in the same function");
    return To->getLayoutIndex() > From->getLayoutIndex();
  }

  /// Dump function information to debug output. If \p PrintInstructions
  /// is true - include instruction disassembly.
  void dump(std::string Annotation = "", bool PrintInstructions = true) const;

  /// Print function information to the \p OS stream.
  void print(raw_ostream &OS, std::string Annotation = "",
             bool PrintInstructions = true) const;

  void addInstruction(uint64_t Offset, MCInst &&Instruction) {
    Instructions.emplace(Offset, std::forward<MCInst>(Instruction));
  }

  /// Return instruction at a given offset in the function. Valid before
  /// CFG is constructed.
  MCInst *getInstructionAtOffset(uint64_t Offset) {
    assert(CurrentState == State::Disassembled &&
           "can only call function in Disassembled state");
    auto II = Instructions.find(Offset);
    return (II == Instructions.end()) ? nullptr : &II->second;
  }

  /// Return true if function profile is present and accurate.
  bool hasValidProfile() const {
    return ExecutionCount != COUNT_NO_PROFILE &&
           ProfileMatchRatio == 1.0f;
  }

  void addCFIInstruction(uint64_t Offset, MCCFIInstruction &&Inst) {
    assert(!Instructions.empty());

    // Fix CFI instructions skipping NOPs. We need to fix this because changing
    // CFI state after a NOP, besides being wrong and innacurate,  makes it
    // harder for us to recover this information, since we can create empty BBs
    // with NOPs and then reorder it away.
    // We fix this by moving the CFI instruction just before any NOPs.
    auto I = Instructions.lower_bound(Offset);
    if (Offset == getSize()) {
      assert(I == Instructions.end() && "unexpected iterator value");
      // Sometimes compiler issues restore_state after all instructions
      // in the function (even after nop).
      --I;
      Offset = I->first;
    }
    assert(I->first == Offset && "CFI pointing to unknown instruction");
    if (I == Instructions.begin()) {
      CIEFrameInstructions.emplace_back(std::forward<MCCFIInstruction>(Inst));
      return;
    }

    --I;
    while (I != Instructions.begin() && BC.MIA->isNoop(I->second)) {
      Offset = I->first;
      --I;
    }
    OffsetToCFI.emplace(Offset, FrameInstructions.size());
    FrameInstructions.emplace_back(std::forward<MCCFIInstruction>(Inst));
    return;
  }

  /// Insert a CFI pseudo instruction in a basic block. This pseudo instruction
  /// is a placeholder that refers to a real MCCFIInstruction object kept by
  /// this function that will be emitted at that position.
  BinaryBasicBlock::const_iterator
  addCFIPseudo(BinaryBasicBlock *BB, BinaryBasicBlock::const_iterator Pos,
               uint32_t Offset) {
    MCInst CFIPseudo;
    BC.MIA->createCFI(CFIPseudo, Offset);
    return BB->insertPseudoInstr(Pos, CFIPseudo);
  }

  /// Retrieve the MCCFIInstruction object associated with a CFI pseudo.
  const MCCFIInstruction* getCFIFor(const MCInst &Instr) const {
    if (!BC.MIA->isCFI(Instr))
      return nullptr;
    uint32_t Offset = Instr.getOperand(0).getImm();
    assert(Offset < FrameInstructions.size() && "Invalid CFI offset");
    return &FrameInstructions[Offset];
  }

  BinaryFunction &setFileOffset(uint64_t Offset) {
    FileOffset = Offset;
    return *this;
  }

  BinaryFunction &setSize(uint64_t S) {
    Size = S;
    return *this;
  }

  BinaryFunction &setMaxSize(uint64_t Size) {
    MaxSize = Size;
    return *this;
  }

  BinaryFunction &setSimple(bool Simple) {
    IsSimple = Simple;
    return *this;
  }

  BinaryFunction &setUsesGnuArgsSize(bool Uses = true) {
    UsesGnuArgsSize = Uses;
    return *this;
  }

  BinaryFunction &setPersonalityFunction(uint64_t Addr) {
    PersonalityFunction = BC.getOrCreateGlobalSymbol(Addr, "FUNCat");
    return *this;
  }

  BinaryFunction &setPersonalityEncoding(uint8_t Encoding) {
    PersonalityEncoding = Encoding;
    return *this;
  }

  BinaryFunction &setAlignment(uint64_t Align) {
    Alignment = Align;
    return *this;
  }

  uint64_t getAlignment() const {
    return Alignment;
  }

  BinaryFunction &setImageAddress(uint64_t Address) {
    ImageAddress = Address;
    return *this;
  }

  /// Return the address of this function' image in memory.
  uint64_t getImageAddress() const {
    return ImageAddress;
  }

  BinaryFunction &setImageSize(uint64_t Size) {
    ImageSize = Size;
    return *this;
  }

  /// Return the size of this function' image in memory.
  uint64_t getImageSize() const {
    return ImageSize;
  }

  /// Set the profile data for the number of times the function was called.
  BinaryFunction &setExecutionCount(uint64_t Count) {
    ExecutionCount = Count;
    return *this;
  }

  /// Set LSDA address for the function.
  BinaryFunction &setLSDAAddress(uint64_t Address) {
    LSDAAddress = Address;
    return *this;
  }

  /// Set LSDA symbol for the function.
  BinaryFunction &setLSDASymbol(MCSymbol *Symbol) {
    LSDASymbol = Symbol;
    return *this;
  }

  /// Sets the function's address ranges list offset in .debug_ranges.
  void setAddressRangesOffset(uint32_t Offset) {
    AddressRangesOffset = Offset;
  }

  /// Returns the offset of the function's address ranges in .debug_ranges.
  uint32_t getAddressRangesOffset() const { return AddressRangesOffset; }

  /// Return the profile information about the number of times
  /// the function was executed.
  ///
  /// Return COUNT_NO_PROFILE if there's no profile info.
  uint64_t getExecutionCount() const {
    return ExecutionCount;
  }

  /// Return original LSDA address for the function or NULL.
  uint64_t getLSDAAddress() const {
    return LSDAAddress;
  }

  /// Return the address of an identical function. If none is found this will
  /// return this function's address.
  uint64_t getIdenticalFunctionAddress() const {
    return IdenticalFunctionAddress;
  }

  /// Set the address of an identical function.
  void setIdenticalFunctionAddress(uint64_t Address) {
    IdenticalFunctionAddress = Address;
  }

  /// Return symbol pointing to function's LSDA.
  MCSymbol *getLSDASymbol() {
    if (LSDASymbol)
      return LSDASymbol;
    if (CallSites.empty())
      return nullptr;

    LSDASymbol =
      BC.Ctx->getOrCreateSymbol(Twine("GCC_except_table") +
                                Twine::utohexstr(getFunctionNumber()));

    return LSDASymbol;
  }

  /// Disassemble function from raw data \p FunctionData.
  /// If successful, this function will populate the list of instructions
  /// for this function together with offsets from the function start
  /// in the input. It will also populate Labels with destinations for
  /// local branches, and TakenBranches with [from, to] info.
  ///
  /// \p FunctionData is the set bytes representing the function body.
  ///
  /// The Function should be properly initialized before this function
  /// is called. I.e. function address and size should be set.
  ///
  /// Returns true on successful disassembly, and updates the current
  /// state to State:Disassembled.
  ///
  /// Returns false if disassembly failed.
  bool disassemble(ArrayRef<uint8_t> FunctionData);

  /// Builds a list of basic blocks with successor and predecessor info.
  ///
  /// The function should in Disassembled state prior to call.
  ///
  /// Returns true on success and update the current function state to
  /// State::CFG. Returns false if CFG cannot be built.
  bool buildCFG();

  /// Verify that any assumptions we've made about indirect branches were
  /// correct and also make any necessary changes to unknown indirect branches.
  ///
  /// Catch-22: we need to know indirect branch targets to build CFG, and
  /// in order to determine the value for indirect branches we need to know CFG.
  ///
  /// As such, the process of decoding indirect branches is broken into 2 steps:
  /// first we make our best guess about a branch without knowing the CFG,
  /// and later after we have the CFG for the function, we verify our earlier
  /// assumptions and also do our best at processing unknown indirect branches.
  ///
  /// Return true upon successful processing, or false if the control flow
  /// cannot be statically evaluated for any given indirect branch.
  bool postProcessIndirectBranches();

  /// Check how closely the profile data matches the function and set
  /// ProfileMatchRatio to reflect the accuracy.
  void evaluateProfileData(const FuncBranchData &BranchData);

  /// Walks the list of basic blocks filling in missing information about
  /// edge frequency for fall-throughs.
  ///
  /// Assumes the CFG has been built and edge frequency for taken branches
  /// has been filled with LBR data.
  void inferFallThroughCounts();

  /// Converts conditional tail calls to unconditional tail calls. We do this to
  /// handle conditional tail calls correctly and to give a chance to the
  /// simplify conditional tail call pass to decide whether to re-optimize them
  /// using profile information.
  void removeConditionalTailCalls();

  /// Computes a function hotness score: the sum of the products of BB frequency
  /// and size.
  uint64_t getFunctionScore();

  /// Annotate each basic block entry with its current CFI state. This is used
  /// to detect when reordering changes the CFI state seen by a basic block and
  /// fix this.
  /// The CFI state is simply the index in FrameInstructions for the
  /// MCCFIInstruction object responsible for this state.
  void annotateCFIState();

  /// After reordering, this function checks the state of CFI and fixes it if it
  /// is corrupted. If it is unable to fix it, it returns false.
  bool fixCFIState();

  /// Associate DW_CFA_GNU_args_size info with invoke instructions
  /// (call instructions with non-empty landing pad).
  void propagateGnuArgsSizeInfo();

  /// Adjust branch instructions to match the CFG.
  ///
  /// As it comes to internal branches, the CFG represents "the ultimate source
  /// of truth". Transformations on functions and blocks have to update the CFG
  /// and fixBranches() would make sure the correct branch instructions are
  /// inserted at the end of basic blocks.
  ///
  /// We do require a conditional branch at the end of the basic block if
  /// the block has 2 successors as CFG currently lacks the conditional
  /// code support (it will probably stay that way). We only use this
  /// branch instruction for its conditional code, the destination is
  /// determined by CFG - first successor representing true/taken branch,
  /// while the second successor - false/fall-through branch.
  ///
  /// When we reverse the branch condition, the CFG is updated accordingly.
  void fixBranches();

  /// Split function in two: a part with warm or hot BBs and a part with never
  /// executed BBs. The cold part is moved to a new BinaryFunction.
  void splitFunction();

  /// Process LSDA information for the function.
  void parseLSDA(ArrayRef<uint8_t> LSDAData, uint64_t LSDAAddress);

  /// Update exception handling ranges for the function.
  void updateEHRanges();

  /// Emit exception handling ranges for the function.
  void emitLSDA(MCStreamer *Streamer);

  /// Emit jump tables for the function.
  void emitJumpTables(MCStreamer *Streamer);

  /// Merge profile data of this function into those of the given
  /// function. The functions should have been proven identical with
  /// isIdenticalWith.
  void mergeProfileDataInto(BinaryFunction &BF) const;

  /// Returns true if this function has identical code and
  /// CFG with the given function.
  bool isIdenticalWith(const BinaryFunction &BF) const;

  /// Returns a hash value for the function. To be used for ICF.
  std::size_t hash() const;

  /// Sets the associated .debug_info entry.
  void addSubprogramDIE(DWARFCompileUnit *Unit,
                          const DWARFDebugInfoEntryMinimal *DIE) {
    SubprogramDIEs.emplace_back(DIE, Unit);
  }

  const decltype(SubprogramDIEs) &getSubprogramDIEs() const {
    return SubprogramDIEs;
  }

  /// Return DWARF compile unit with line info.
  DWARFUnitLineTable getDWARFUnitLineTable() const {
    for (auto &DIEUnitPair : SubprogramDIEs) {
      if (auto *LT = BC.DwCtx->getLineTableForUnit(DIEUnitPair.second)) {
        return std::make_pair(DIEUnitPair.second, LT);
      }
    }
    return std::make_pair(nullptr, nullptr);
  }

  /// Returns the size of the basic block in the original binary.
  size_t getBasicBlockOriginalSize(const BinaryBasicBlock *BB) const;

  /// Returns an estimate of the function's hot part after splitting.
  /// This is a very rough estimate, as with C++ exceptions there are
  /// blocks we don't move, and it makes no attempt at estimating the size
  /// of the added/removed branch instructions.
  /// Note that this size is optimistic and the actual size may increase
  /// after relaxation.
  size_t estimateHotSize() const {
    size_t Estimate = 0;
    for (const auto *BB : BasicBlocksLayout) {
      if (BB->getExecutionCount() != 0) {
        Estimate += BC.computeCodeSize(BB->begin(), BB->end());
      }
    }
    return Estimate;
  }

  virtual ~BinaryFunction();

  /// Info for fragmented functions.
  class FragmentInfo {
  private:
    uint64_t Address{0};
    uint64_t ImageAddress{0};
    uint64_t ImageSize{0};
    uint64_t FileOffset{0};
  public:
    uint64_t getAddress() const { return Address; }
    uint64_t getImageAddress() const { return ImageAddress; }
    uint64_t getImageSize() const { return ImageSize; }
    uint64_t getFileOffset() const { return FileOffset; }

    void setAddress(uint64_t VAddress) { Address = VAddress; }
    void setImageAddress(uint64_t Address) { ImageAddress = Address; }
    void setImageSize(uint64_t Size) { ImageSize = Size; }
    void setFileOffset(uint64_t Offset) { FileOffset = Offset; }
  };

  /// Cold fragment of the function.
  FragmentInfo ColdFragment;

  FragmentInfo &cold() { return ColdFragment; }

  const FragmentInfo &cold() const { return ColdFragment; }
};

/// Return program-wide dynostats.
template <typename FuncsType>
inline DynoStats getDynoStats(const FuncsType &Funcs) {
  DynoStats dynoStats;
  for (auto &BFI : Funcs) {
    auto &BF = BFI.second;
    if (BF.isSimple()) {
      dynoStats += BF.getDynoStats();
    }
  }
  return dynoStats;
}

/// Call a function with optional before and after dynostats printing.
template <typename FnType, typename FuncsType>
inline void
callWithDynoStats(FnType &&Func,
                  const FuncsType &Funcs,
                  StringRef Phase,
                  const bool Flag) {
  DynoStats dynoStatsBefore;
  if (Flag) {
    dynoStatsBefore = getDynoStats(Funcs);
    outs() << "BOLT-INFO: program-wide dynostats before running "
           << Phase << ":\n\n" << dynoStatsBefore << '\n';
  }

  Func();

  if (Flag) {
    auto dynoStatsAfter = getDynoStats(Funcs);
    outs() << "BOLT-INFO: program-wide dynostats after running "
           << Phase << ":\n\n" << dynoStatsBefore << '\n';
    dynoStatsAfter.print(outs(), &dynoStatsBefore);
    outs() << '\n';
  }
}

inline raw_ostream &operator<<(raw_ostream &OS,
                               const BinaryFunction &Function) {
  OS << Function.getPrintName();
  return OS;
}

inline raw_ostream &operator<<(raw_ostream &OS,
                               const BinaryFunction::State State) {
  switch (State) {
  default:                                  OS << "<unknown>"; break;
  case BinaryFunction::State::Empty:        OS << "empty";  break;
  case BinaryFunction::State::Disassembled: OS << "disassembled";  break;
  case BinaryFunction::State::CFG:          OS << "CFG constructed";  break;
  case BinaryFunction::State::Assembled:    OS << "assembled";  break;
  }

  return OS;
}

} // namespace bolt


// GraphTraits specializations for function basic block graphs (CFGs)
template <> struct GraphTraits<bolt::BinaryFunction *> :
  public GraphTraits<bolt::BinaryBasicBlock *> {
  static NodeType *getEntryNode(bolt::BinaryFunction *F) {
    return *F->layout_begin();
  }

  typedef bolt::BinaryBasicBlock * nodes_iterator;
  static nodes_iterator nodes_begin(bolt::BinaryFunction *F) {
    return &(*F->begin());
  }
  static nodes_iterator nodes_end(bolt::BinaryFunction *F) {
    return &(*F->end());
  }
  static size_t size(bolt::BinaryFunction *F) {
    return F->size();
  }
};

template <> struct GraphTraits<const bolt::BinaryFunction *> :
  public GraphTraits<const bolt::BinaryBasicBlock *> {
  static NodeType *getEntryNode(const bolt::BinaryFunction *F) {
    return *F->layout_begin();
  }

  typedef const bolt::BinaryBasicBlock * nodes_iterator;
  static nodes_iterator nodes_begin(const bolt::BinaryFunction *F) {
    return &(*F->begin());
  }
  static nodes_iterator nodes_end(const bolt::BinaryFunction *F) {
    return &(*F->end());
  }
  static size_t size(const bolt::BinaryFunction *F) {
    return F->size();
  }
};

template <> struct GraphTraits<Inverse<bolt::BinaryFunction *>> :
  public GraphTraits<Inverse<bolt::BinaryBasicBlock *>> {
  static NodeType *getEntryNode(Inverse<bolt::BinaryFunction *> G) {
    return *G.Graph->layout_begin();
  }
};

template <> struct GraphTraits<Inverse<const bolt::BinaryFunction *>> :
  public GraphTraits<Inverse<const bolt::BinaryBasicBlock *>> {
  static NodeType *getEntryNode(Inverse<const bolt::BinaryFunction *> G) {
    return *G.Graph->layout_begin();
  }
};


} // namespace llvm

#endif
