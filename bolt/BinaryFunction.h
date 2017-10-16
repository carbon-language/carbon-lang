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
#include <unordered_set>
#include <vector>

using namespace llvm::object;

namespace llvm {

class DWARFCompileUnit;
class DWARFDebugInfoEntryMinimal;

namespace bolt {

struct SectionInfo;

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
  D(LOADS,                        "executed load instructions", Fn)\
  D(STORES,                       "executed store instructions", Fn)\
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
  D(LAST_DYNO_STAT,               "<reserved>", 0)

public:
#define D(name, ...) name,
  enum Category : uint8_t { DYNO_STATS };
#undef D


private:
  uint64_t Stats[LAST_DYNO_STAT+1];

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
  bool operator==(const DynoStats &Other) const;
  bool operator!=(const DynoStats &Other) const { return !operator==(Other); }
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

enum JumpTableSupportLevel : char {
  JTS_NONE = 0,       /// Disable jump tables support.
  JTS_BASIC = 1,      /// Enable basic jump tables support (in-place).
  JTS_MOVE = 2,       /// Move jump tables to a separate section.
  JTS_SPLIT = 3,      /// Enable hot/cold splitting of jump tables.
  JTS_AGGRESSIVE = 4, /// Aggressive splitting of jump tables.
};

enum IndirectCallPromotionType : char {
  ICP_NONE,        /// Don't perform ICP.
  ICP_CALLS,       /// Perform ICP on indirect calls.
  ICP_JUMP_TABLES, /// Perform ICP on jump tables.
  ICP_ALL          /// Perform ICP on calls and jump tables.
};

/// BinaryFunction is a representation of machine-level function.
///
/// We use the term "Binary" as "Machine" was already taken.
class BinaryFunction {
public:
  enum class State : char {
    Empty = 0,        /// Function body is empty.
    Disassembled,     /// Function have been disassembled.
    CFG,              /// Control flow graph have been built.
    CFG_Finalized,    /// CFG is finalized. No optimizations allowed.
    Emitted,          /// Instructions have been emitted to output.
  };

  /// Settings for splitting function bodies into hot/cold partitions.
  enum SplittingType : char {
    ST_NONE = 0,      /// Do not split functions
    ST_EH,            /// Split blocks comprising landing pads
    ST_LARGE,         /// Split functions that exceed maximum size in addition
                      /// to landing pads.
    ST_ALL,           /// Split all functions
  };

  enum ReorderType : char {
    RT_NONE = 0,
    RT_EXEC_COUNT,
    RT_HFSORT,
    RT_HFSORT_PLUS,
    RT_PETTIS_HANSEN,
    RT_RANDOM,
    RT_USER
  };

  /// Branch statistics for jump table entries.
  struct JumpInfo {
    uint64_t Mispreds{0};
    uint64_t Count{0};
  };

  static constexpr uint64_t COUNT_NO_PROFILE =
    BinaryBasicBlock::COUNT_NO_PROFILE;

  /// We have to use at least 2-byte alignment for functions because of C++ ABI.
  static constexpr unsigned MinAlign = 2;

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

  /// Original size of the function.
  uint64_t Size;

  /// Address of the function in output.
  uint64_t OutputAddress{0};

  /// Size of the function in the output file.
  uint64_t OutputSize{0};

  /// Offset in the file.
  uint64_t FileOffset{0};

  /// Maximum size this function is allowed to have.
  uint64_t MaxSize{std::numeric_limits<uint64_t>::max()};

  /// Alignment requirements for the function.
  uint64_t Alignment{2};

  const MCSymbol *PersonalityFunction{nullptr};
  uint8_t PersonalityEncoding{dwarf::DW_EH_PE_sdata4 | dwarf::DW_EH_PE_pcrel};

  BinaryContext &BC;

  std::unique_ptr<BinaryLoopInfo> BLI;

  /// False if the function is too complex to reconstruct its control
  /// flow graph.
  /// In relocation mode we still disassemble and re-assemble such functions.
  bool IsSimple{true};

  /// In AArch64, preserve nops to maintain code equal to input (assuming no
  /// optimizations are done).
  bool PreserveNops{false};

  /// Indicate if this function has associated exception handling metadata.
  bool HasEHRanges{false};

  /// True if the function uses DW_CFA_GNU_args_size CFIs.
  bool UsesGnuArgsSize{false};

  /// True if the function has more than one entry point.
  bool IsMultiEntry{false};

  /// Indicate if the function body was folded into another function. Used
  /// for ICF optimization without relocations.
  bool IsFolded{false};

  /// The address for the code for this function in codegen memory.
  uint64_t ImageAddress{0};

  /// The size of the code in memory.
  uint64_t ImageSize{0};

  /// Name for the section this function code should reside in.
  std::string CodeSectionName;

  /// Name for the corresponding cold code section.
  std::string ColdCodeSectionName;

  /// The profile data for the number of times the function was executed.
  uint64_t ExecutionCount{COUNT_NO_PROFILE};

  /// Profile data for branches.
  FuncBranchData *BranchData{nullptr};

  /// Profile match ratio for BranchData.
  float ProfileMatchRatio{0.0f};

  /// Score of the function (estimated number of instructions executed,
  /// according to profile data). -1 if the score has not been calculated yet.
  int64_t FunctionScore{-1};

  /// Original LSDA address for the function.
  uint64_t LSDAAddress{0};

  /// Associated DIEs in the .debug_info section with their respective CUs.
  /// There can be multiple because of identical code folding.
  std::vector<std::pair<const DWARFDebugInfoEntryMinimal *,
                        DWARFCompileUnit *>> SubprogramDIEs;

  /// Line table for the function with containing compilation unit.
  /// Because of identical code folding the function could have multiple
  /// associated compilation units. The first of them with line number info
  /// is referenced by UnitLineTable.
  DWARFUnitLineTable UnitLineTable{nullptr, nullptr};

  /// Last computed hash value.
  mutable uint64_t Hash{0};

  /// For PLT functions it contains a symbol associated with a function
  /// reference. It is nullptr for non-PLT functions.
  const MCSymbol *PLTSymbol{nullptr};

  /// Function order for streaming into the destination binary.
  uint32_t Index{-1U};

  /// Get basic block index assuming it belongs to this function.
  unsigned getIndex(const BinaryBasicBlock *BB) const {
    assert(BB->getIndex() < BasicBlocks.size());
    return BB->getIndex();
  }

  /// Return basic block that originally contained offset \p Offset
  /// from the function start.
  BinaryBasicBlock *getBasicBlockContainingOffset(uint64_t Offset);

  const BinaryBasicBlock *getBasicBlockContainingOffset(uint64_t Offset) const {
    return const_cast<BinaryFunction *>(this)
      ->getBasicBlockContainingOffset(Offset);
  }

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

  /// Update the indices of all the basic blocks starting at StartIndex.
  void updateBBIndices(const unsigned StartIndex);

  /// Annotate each basic block entry with its current CFI state. This is
  /// run right after the construction of CFG while basic blocks are in their
  /// original order.
  void annotateCFIState();

  /// Associate DW_CFA_GNU_args_size info with invoke instructions
  /// (call instructions with non-empty landing pad).
  void propagateGnuArgsSizeInfo();

  /// Synchronize branch instructions with CFG.
  void postProcessBranches();

  /// Helper function that compares an instruction of this function to the
  /// given instruction of the given function. The functions should have
  /// identical CFG.
  template <class Compare>
  bool isInstrEquivalentWith(
      const MCInst &InstA, const BinaryBasicBlock &BBA,
      const MCInst &InstB, const BinaryBasicBlock &BBB,
      const BinaryFunction &BFB, Compare Comp) const {
    if (InstA.getOpcode() != InstB.getOpcode()) {
      return false;
    }

    // In this function we check for special conditions:
    //
    //    * instructions with landing pads
    //
    // Most of the common cases should be handled by MCInst::equals()
    // that compares regular instruction operands.
    //
    // NB: there's no need to compare jump table indirect jump instructions
    //     separately as jump tables are handled by comparing corresponding
    //     symbols.
    const auto EHInfoA = BC.MIA->getEHInfo(InstA);
    const auto EHInfoB = BC.MIA->getEHInfo(InstB);

    // Action indices should match.
    if (EHInfoA.second != EHInfoB.second)
      return false;

    if (!EHInfoA.first != !EHInfoB.first)
      return false;

    if (EHInfoA.first && EHInfoB.first) {
      const auto *LPA = BBA.getLandingPad(EHInfoA.first);
      const auto *LPB = BBB.getLandingPad(EHInfoB.first);
      assert(LPA && LPB && "cannot locate landing pad(s)");

      if (LPA->getLayoutIndex() != LPB->getLayoutIndex())
        return false;
    }

    return InstA.equals(InstB, Comp);
  }

  /// Recompute landing pad information for the function and all its blocks.
  void recomputeLandingPads();

  /// Temporary holder of offsets that are potentially entry points.
  std::unordered_set<uint64_t> EntryOffsets;

  /// Temporary holder of offsets that are data markers (used in AArch)
  /// It is possible to have data in code sections. To ease the identification
  /// of data in code sections, the ABI requires the symbol table to have
  /// symbols named "$d" identifying the start of data inside code and "$x"
  /// identifying the end of a chunk of data inside code. DataOffsets contain
  /// all offsets of $d symbols and CodeOffsets all offsets of $x symbols.
  std::set<uint64_t> DataOffsets;
  std::set<uint64_t> CodeOffsets;
  /// The address offset where we emitted the constant island, that is, the
  /// chunk of data in the function code area (AArch only)
  int64_t OutputDataOffset;

  /// Map labels to corresponding basic blocks.
  std::unordered_map<const MCSymbol *, BinaryBasicBlock *> LabelToBB;

  using BranchListType = std::vector<std::pair<uint32_t, uint32_t>>;
  BranchListType TakenBranches;       /// All local taken branches.
  BranchListType FTBranches;          /// All fall-through branches.
  BranchListType IgnoredBranches;     /// Branches ignored by CFG purposes.

  /// Map offset in the function to a label.
  /// Labels are used for building CFG for simple functions. For non-simple
  /// function in relocation mode we need to emit them for relocations
  /// referencing function internals to work (e.g. jump tables).
  using LabelsMapType = std::map<uint32_t, MCSymbol *>;
  LabelsMapType Labels;

  /// Temporary holder of instructions before CFG is constructed.
  /// Map offset in the function to MCInst.
  using InstrMapType = std::map<uint32_t, size_t>;
  InstrMapType InstructionOffsets;
  std::vector<MCInst> Instructions;

  /// List of DWARF CFI instructions. Original CFI from the binary must be
  /// sorted w.r.t. offset that it appears. We rely on this to replay CFIs
  /// if needed (to fix state after reordering BBs).
  using CFIInstrMapType = std::vector<MCCFIInstruction>;
  using cfi_iterator       = CFIInstrMapType::iterator;
  using const_cfi_iterator = CFIInstrMapType::const_iterator;

  /// We don't decode Call Frame Info encoded in DWARF program state
  /// machine. Instead we define a "CFI State" - a frame information that
  /// is a result of executing FDE CFI program up to a given point. The
  /// program consists of opaque Call Frame Instructions:
  ///
  ///   CFI #0
  ///   CFI #1
  ///   ....
  ///   CFI #N
  ///
  /// When we refer to "CFI State K" - it corresponds to a row in an abstract
  /// Call Frame Info table. This row is reached right before executing CFI #K.
  ///
  /// At any point of execution in a function we are in any one of (N + 2)
  /// states described in the original FDE program. We can't have more states
  /// without intelligent processing of CFIs.
  ///
  /// When the final layout of basic blocks is known, and we finalize CFG,
  /// we modify the original program to make sure the same state could be
  /// reached even when basic blocks containing CFI instructions are executed
  /// in a different order.
  CFIInstrMapType FrameInstructions;

  /// Exception handling ranges.
  struct CallSite {
    const MCSymbol *Start;
    const MCSymbol *End;
    const MCSymbol *LP;
    uint64_t Action;
  };
  std::vector<CallSite> CallSites;
  std::vector<CallSite> ColdCallSites;

  /// Binary blobs reprsenting action, type, and type index tables for this
  /// function' LSDA (exception handling).
  ArrayRef<uint8_t> LSDAActionTable;
  std::vector<uint64_t> LSDATypeTable;
  ArrayRef<uint8_t> LSDATypeIndexTable;

  /// Marking for the beginning of language-specific data area for the function.
  MCSymbol *LSDASymbol{nullptr};
  MCSymbol *ColdLSDASymbol{nullptr};

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
public:
  struct JumpTable {
    enum JumpTableType : char {
      JTT_NORMAL,
      JTT_PIC,
    };

    /// Original address.
    uint64_t Address;

    /// Size of the entry used for storage.
    std::size_t EntrySize;

    /// The type of this jump table.
    JumpTableType Type;

    /// All the entries as labels.
    std::vector<MCSymbol *> Entries;

    /// All the entries as offsets into a function. Invalid after CFG is built.
    std::vector<uint64_t> OffsetEntries;

    /// Map <Offset> -> <Label> used for embedded jump tables. Label at 0 offset
    /// is the main label for the jump table.
    std::map<unsigned, MCSymbol *> Labels;

    /// Corresponding section if any.
    SectionInfo *SecInfo{nullptr};

    /// Corresponding section name if any.
    std::string SectionName;

    /// Return the size of the jump table.
    uint64_t getSize() const {
      return std::max(OffsetEntries.size(), Entries.size()) * EntrySize;
    }

    /// Get the indexes for symbol entries that correspond to the jump table
    /// starting at (or containing) 'Addr'.
    std::pair<size_t, size_t> getEntriesForAddress(const uint64_t Addr) const;

    /// Constructor.
    JumpTable(uint64_t Address,
              std::size_t EntrySize,
              JumpTableType Type,
              decltype(OffsetEntries) &&OffsetEntries,
              decltype(Labels) &&Labels)
      : Address(Address), EntrySize(EntrySize), Type(Type),
        OffsetEntries(OffsetEntries), Labels(Labels)
    {}

    /// Dynamic number of times each entry in the table was referenced.
    /// Identical entries will have a shared count (identical for every
    /// entry in the set).
    std::vector<JumpInfo> Counts;

    /// Total number of times this jump table was used.
    uint64_t Count{0};

    /// Change all entries of the jump table in \p JTAddress pointing to
    /// \p OldDest to \p NewDest. Return false if unsuccessful.
    bool replaceDestination(uint64_t JTAddress, const MCSymbol *OldDest,
                            MCSymbol *NewDest);

    /// Update jump table at its original location.
    void updateOriginal(BinaryContext &BC);

    /// Emit jump table data. Callee supplies sections for the data.
    /// Return the number of total bytes emitted.
    uint64_t emit(MCStreamer *Streamer, MCSection *HotSection,
                  MCSection *ColdSection);

    /// Print for debugging purposes.
    void print(raw_ostream &OS) const;
  };
private:

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

  const JumpTable *getJumpTableContainingAddress(uint64_t Address) const {
    auto JTI = JumpTables.upper_bound(Address);
    if (JTI == JumpTables.begin())
      return nullptr;
    --JTI;
    if (JTI->first + JTI->second.getSize() > Address) {
      return &JTI->second;
    }
    return nullptr;
  }

  /// Compare two jump tables in 2 functions. The function relies on consistent
  /// ordering of basic blocks in both binary functions (e.g. DFS).
  bool equalJumpTables(const JumpTable *JumpTableA,
                       const JumpTable *JumpTableB,
                       const BinaryFunction &BFB) const;

  /// All jump table sites in the function.
  std::vector<std::pair<uint64_t, uint64_t>> JTSites;

  /// List of relocations in this function.
  std::map<uint64_t, Relocation> Relocations;

  /// Map of relocations used for moving the function body as it is.
  std::map<uint64_t, Relocation> MoveRelocations;

  /// Offsets in function that should have PC-relative relocation.
  std::set<uint64_t> PCRelativeRelocationOffsets;

  /// Offsets in function that are data values in a constant island identified
  /// after disassembling
  std::map<uint64_t, MCSymbol *> IslandSymbols;

  // Blocks are kept sorted in the layout order. If we need to change the
  // layout (if BasicBlocksLayout stores a different order than BasicBlocks),
  // the terminating instructions need to be modified.
  using BasicBlockListType = std::vector<BinaryBasicBlock *>;
  BasicBlockListType BasicBlocks;
  BasicBlockListType DeletedBasicBlocks;
  BasicBlockOrderType BasicBlocksLayout;
  /// Previous layout replaced by modifyLayout
  BasicBlockOrderType BasicBlocksPreviousLayout;

  /// BasicBlockOffsets are used during CFG construction to map from code
  /// offsets to BinaryBasicBlocks.  Any modifications made to the CFG
  /// after initial construction are not reflected in this data structure.
  using BasicBlockOffset = std::pair<uint64_t, BinaryBasicBlock *>;
  struct CompareBasicBlockOffsets {
    bool operator()(const BasicBlockOffset &A,
                    const BasicBlockOffset &B) const {
      return A.first < B.first;
    }
  };
  std::vector<BasicBlockOffset> BasicBlockOffsets;

  /// Symbol in the output.
  ///
  /// NB: function can have multiple symbols associated with it. We will emit
  ///     all symbols for the function
  MCSymbol *OutputSymbol;

  MCSymbol *ColdSymbol{nullptr};

  /// Symbol at the end of the function.
  mutable MCSymbol *FunctionEndLabel{nullptr};

  /// Symbol at the end of the cold part of split function.
  mutable MCSymbol *FunctionColdEndLabel{nullptr};

  mutable MCSymbol *FunctionConstantIslandLabel{nullptr};

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

  /// Register alternative function name.
  void addAlternativeName(std::string NewName) {
    Names.emplace_back(NewName);
  }

  /// Return label at a given \p Address in the function. If the label does
  /// not exist - create it. Assert if the \p Address does not belong to
  /// the function. If \p CreatePastEnd is true, then return the function
  /// end label when the \p Address points immediately past the last byte
  /// of the function.
  MCSymbol *getOrCreateLocalLabel(uint64_t Address, bool CreatePastEnd = false);

  /// Register an entry point at a given \p Offset into the function.
  void markDataAtOffset(uint64_t Offset) {
    DataOffsets.emplace(Offset);
  }

  /// Register an entry point at a given \p Offset into the function.
  void markCodeAtOffset(uint64_t Offset) {
    CodeOffsets.emplace(Offset);
  }

  /// Register an entry point at a given \p Offset into the function.
  MCSymbol *addEntryPointAtOffset(uint64_t Offset) {
    EntryOffsets.emplace(Offset);
    IsMultiEntry = (Offset == 0 ? IsMultiEntry : true);
    return getOrCreateLocalLabel(getAddress() + Offset);
  }

  /// This is called in disassembled state.
  void addEntryPoint(uint64_t Address);

  /// Return true if there is a registered entry point at a given offset
  /// into the function.
  bool hasEntryPointAtOffset(uint64_t Offset) {
    assert(!EntryOffsets.empty() && "entry points uninitialized or destroyed");
    return EntryOffsets.count(Offset);
  }

  void addInstruction(uint64_t Offset, MCInst &&Instruction) {
    assert(InstructionOffsets.size() == Instructions.size() &&
           "There must be one instruction at every offset.");
    Instructions.emplace_back(std::forward<MCInst>(Instruction));
    InstructionOffsets[Offset] = Instructions.size() - 1;
  }

  /// Return instruction at a given offset in the function. Valid before
  /// CFG is constructed.
  MCInst *getInstructionAtOffset(uint64_t Offset) {
    assert(CurrentState == State::Disassembled &&
           "can only call function in Disassembled state");
    auto II = InstructionOffsets.find(Offset);
    return (II == InstructionOffsets.end())
       ? nullptr : &Instructions[II->second];
  }

  /// Analyze and process indirect branch \p Instruction before it is
  /// added to Instructions list.
  IndirectBranchType processIndirectBranch(MCInst &Instruction,
                                           unsigned Size,
                                           uint64_t Offset);

  DenseMap<const MCInst *, SmallVector<MCInst *, 4>>
  computeLocalUDChain(const MCInst *CurInstr);

  /// Emit line number information corresponding to \p NewLoc. \p PrevLoc
  /// provides a context for de-duplication of line number info.
  ///
  /// Return new current location which is either \p NewLoc or \p PrevLoc.
  SMLoc emitLineInfo(SMLoc NewLoc, SMLoc PrevLoc) const;

  BinaryFunction& operator=(const BinaryFunction &) = delete;
  BinaryFunction(const BinaryFunction &) = delete;

  friend class RewriteInstance;
  friend class BinaryContext;

  /// Creation should be handled by RewriteInstance::createBinaryFunction().
  BinaryFunction(const std::string &Name, SectionRef Section, uint64_t Address,
                 uint64_t Size, BinaryContext &BC, bool IsSimple) :
      Names({Name}), Section(Section), Address(Address),
      Size(Size), BC(BC), IsSimple(IsSimple),
      CodeSectionName(".local.text." + Name),
      ColdCodeSectionName(".local.cold.text." + Name),
      FunctionNumber(++Count) {
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

  size_t                    size() const { return BasicBlocks.size();}
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
  size_t   layout_size()  const { return BasicBlocksLayout.size(); }
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

  /// Update layout of basic blocks used for output.
  void updateBasicBlockLayout(BasicBlockOrderType &NewLayout,
                              bool SavePrevLayout) {
    if (SavePrevLayout)
      BasicBlocksPreviousLayout = BasicBlocksLayout;

    BasicBlocksLayout.clear();
    BasicBlocksLayout.swap(NewLayout);
  }

  /// Return a list of basic blocks sorted using DFS and update layout indices
  /// using the same order. Does not modify the current layout.
  BasicBlockOrderType dfs() const;

  /// Find the loops in the CFG of the function and store information about
  /// them.
  void calculateLoopInfo();

  /// Returns if loop detection has been run for this function.
  bool hasLoopInfo() const {
    return BLI != nullptr;
  }

  const BinaryLoopInfo &getLoopInfo() {
    return *BLI.get();
  }

  bool isLoopFree() {
    if (!hasLoopInfo()) {
      calculateLoopInfo();
    }
    return BLI->empty();
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

  /// Return BinaryContext for the function.
  BinaryContext &getBinaryContext() {
    return BC;
  }

  /// Attempt to validate CFG invariants.
  bool validateCFG() const;

  /// Return dynostats for the function.
  ///
  /// The function relies on branch instructions being in-sync with CFG for
  /// branch instructions stats. Thus it is better to call it after
  /// fixBranches().
  DynoStats getDynoStats() const;

  BinaryBasicBlock *getBasicBlockForLabel(const MCSymbol *Label) {
    auto I = LabelToBB.find(Label);
    return I == LabelToBB.end() ? nullptr : I->second;
  }

  const BinaryBasicBlock *getBasicBlockForLabel(const MCSymbol *Label) const {
    auto I = LabelToBB.find(Label);
    return I == LabelToBB.end() ? nullptr : I->second;
  }

  /// Returns the basic block after the given basic block in the layout or
  /// nullptr the last basic block is given.
  const BinaryBasicBlock *getBasicBlockAfter(const BinaryBasicBlock *BB,
                                             bool IgnoreSplits = true) const {
    return
      const_cast<BinaryFunction *>(this)->getBasicBlockAfter(BB, IgnoreSplits);
  }

  BinaryBasicBlock *getBasicBlockAfter(const BinaryBasicBlock *BB,
                                       bool IgnoreSplits = true) {
    for (auto I = layout_begin(), E = layout_end(); I != E; ++I) {
      auto Next = std::next(I);
      if (*I == BB && Next != E) {
        return (IgnoreSplits || (*I)->isCold() == (*Next)->isCold())
          ? *Next : nullptr;
      }
    }
    return nullptr;
  }

  /// Retrieve the landing pad BB associated with invoke instruction \p Invoke
  /// that is in \p BB. Return nullptr if none exists
  BinaryBasicBlock *getLandingPadBBFor(const BinaryBasicBlock &BB,
                                       const MCInst &InvokeInst) {
    assert(BC.MIA->isInvoke(InvokeInst) && "must be invoke instruction");
    MCLandingPad LP = BC.MIA->getEHInfo(InvokeInst);
    if (LP.first) {
      auto *LBB = BB.getLandingPad(LP.first);
      assert (LBB && "Landing pad should be defined");
      return LBB;
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
  bool hasName(const std::string &FunctionName) const {
    for (auto &Name : Names)
      if (Name == FunctionName)
        return true;
    return false;
  }

  /// Return a vector of all possible names for the function.
  const std::vector<std::string> &getNames() const {
    return Names;
  }

  /// Return a state the function is in (see BinaryFunction::State definition
  /// for description).
  State getState() const {
    return CurrentState;
  }

  /// Return true if function has a control flow graph available.
  bool hasCFG() const {
    return getState() == State::CFG ||
           getState() == State::CFG_Finalized ||
           getState() == State::Emitted;
  }

  bool isEmitted() const {
    return getState() == State::Emitted;
  }

  /// Return containing file section.
  SectionRef getSection() const {
    return Section;
  }

  /// Return original address of the function (or offset from base for PIC).
  uint64_t getAddress() const {
    return Address;
  }

  uint64_t getOutputAddress() const {
    return OutputAddress;
  }

  uint64_t getOutputSize() const {
    return OutputSize;
  }

  /// Does this function have a valid streaming order index?
  bool hasValidIndex() const {
    return Index != -1U;
  }

  /// Get the streaming order index for this function.
  uint32_t getIndex() const {
    return Index;
  }

  /// Set the streaming order index for this function.
  void setIndex(uint32_t Idx) {
    assert(!hasValidIndex());
    Index = Idx;
  }

  /// Get the original address for the given basic block within this function.
  uint64_t getBasicBlockOriginalAddress(const BinaryBasicBlock *BB) const {
    return Address + BB->getOffset();
  }

  /// Return offset of the function body in the binary file.
  uint64_t getFileOffset() const {
    return FileOffset;
  }

  /// Return (original) byte size of the function.
  uint64_t getSize() const {
    return Size;
  }

  /// Return the maximum size the body of the function could have.
  uint64_t getMaxSize() const {
    return MaxSize;
  }

  /// Return the number of emitted instructions for this function.
  uint32_t getNumNonPseudos() const {
    uint32_t N = 0;
    for (auto &BB : layout()) {
      N += BB->getNumNonPseudos();
    }
    return N;
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

  MCSymbol *getColdSymbol() {
    if (ColdSymbol)
      return ColdSymbol;

    ColdSymbol = BC.Ctx->getOrCreateSymbol(
        Twine(getSymbol()->getName()).concat(".cold"));

    return ColdSymbol;
  }

  /// Return MC symbol associated with the end of the function.
  MCSymbol *getFunctionEndLabel() const {
    assert(BC.Ctx && "cannot be called with empty context");
    if (!FunctionEndLabel) {
      FunctionEndLabel = BC.Ctx->createTempSymbol("func_end", true);
    }
    return FunctionEndLabel;
  }

  /// Return MC symbol associated with the end of the cold part of the function.
  MCSymbol *getFunctionColdEndLabel() const {
    if (!FunctionColdEndLabel) {
      FunctionColdEndLabel = BC.Ctx->createTempSymbol("func_cold_end", true);
    }
    return FunctionColdEndLabel;
  }

  /// Return a label used to identify where the constant island was emitted
  /// (AArch only). This is used to update the symbol table accordingly,
  /// emitting data marker symbols as required by the ABI.
  MCSymbol *getFunctionConstantIslandLabel() const {
    if (!FunctionConstantIslandLabel) {
      FunctionConstantIslandLabel =
          BC.Ctx->createTempSymbol("func_const_island", true);
    }
    return FunctionConstantIslandLabel;
  }

  /// Return true if this is a function representing a PLT entry.
  bool isPLTFunction() const {
    return PLTSymbol != nullptr;
  }

  /// Return PLT function reference symbol for PLT functions and nullptr for
  /// non-PLT functions.
  const MCSymbol *getPLTSymbol() const {
    return PLTSymbol;
  }

  /// Set function PLT reference symbol for PLT functions.
  void setPLTSymbol(const MCSymbol *Symbol) {
    assert(Size == 0 && "function size should be 0 for PLT functions");
    PLTSymbol = Symbol;
  }

  /// Register relocation type \p RelType at a given \p Address in the function
  /// against \p Symbol.
  /// Assert if the \p Address is not inside this function.
  void addRelocation(uint64_t Address, MCSymbol *Symbol, uint64_t RelType,
                     uint64_t Addend, uint64_t Value) {
    assert(Address >= getAddress() && Address < getAddress() + getMaxSize() &&
           "address is outside of the function");
    auto Offset = Address - getAddress();
    switch (RelType) {
    case ELF::R_X86_64_32:
    case ELF::R_X86_64_32S:
    case ELF::R_X86_64_64:
    case ELF::R_AARCH64_ABS64:
    case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
    case ELF::R_AARCH64_LD64_GOT_LO12_NC:
    case ELF::R_AARCH64_ADD_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
    case ELF::R_AARCH64_ADR_GOT_PAGE:
    case ELF::R_AARCH64_ADR_PREL_PG_HI21:
      Relocations.emplace(Offset,
                          Relocation{Offset, Symbol, RelType, Addend, Value});
      break;
    case ELF::R_X86_64_PC32:
    case ELF::R_X86_64_PC8:
    case ELF::R_X86_64_PLT32:
    case ELF::R_X86_64_GOTPCRELX:
    case ELF::R_X86_64_REX_GOTPCRELX:
    case ELF::R_AARCH64_JUMP26:
    case ELF::R_AARCH64_CALL26:
      break;

    // The following relocations are ignored.
    case ELF::R_X86_64_GOTPCREL:
    case ELF::R_X86_64_TPOFF32:
    case ELF::R_X86_64_GOTTPOFF:
      return;
    default:
      llvm_unreachable("unexpected relocation type in code");
    }
    MoveRelocations[Offset] =
      Relocation{Offset, Symbol, RelType, Addend, Value};
  }

  /// Register a fact that we should have a PC-relative relocation at a given
  /// address in a function. During disassembly we have to make sure we create
  /// relocation at that location.
  void addPCRelativeRelocationAddress(uint64_t Address) {
    assert(Address >= getAddress() && Address < getAddress() + getSize() &&
           "address is outside of the function");
    PCRelativeRelocationOffsets.emplace(Address - getAddress());
  }

  /// Return internal section name for this function.
  StringRef getCodeSectionName() const {
    return StringRef(CodeSectionName);
  }

  /// Return cold code section name for the function.
  StringRef getColdCodeSectionName() const {
    return StringRef(ColdCodeSectionName);
  }

  /// Return true if the function could be correctly processed.
  bool isSimple() const {
    return IsSimple;
  }

  /// Return true if the function body is non-contiguous.
  bool isSplit() const {
    return size() > 1 &&
           layout_front()->isCold() != layout_back()->isCold();
  }

  /// Return true if the function has exception handling tables.
  bool hasEHRanges() const {
    return HasEHRanges;
  }

  /// Return true if the function uses DW_CFA_GNU_args_size CFIs.
  bool usesGnuArgsSize() const {
    return UsesGnuArgsSize;
  }

  /// Return true if the function has more than one entry point.
  bool isMultiEntry() const {
    return IsMultiEntry;
  }

  bool isFolded() const {
    return IsFolded;
  }

  /// Return true if the function uses jump tables.
  bool hasJumpTables() const {
    return JumpTables.size();
  }

  const JumpTable *getJumpTable(const MCInst &Inst) const {
    const auto Address = BC.MIA->getJumpTable(Inst);
    return getJumpTableContainingAddress(Address);
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
  bool containsAddress(uint64_t PC, bool UseMaxSize=false) const {
    if (UseMaxSize)
      return Address <= PC && PC < Address + MaxSize;
    return Address <= PC && PC < Address + Size;
  }

  /// Add new names this function is known under.
  template <class ContainterTy>
  void addNewNames(const ContainterTy &NewNames) {
    Names.insert(Names.begin(),  NewNames.begin(), NewNames.end());
  }

  /// Create a basic block at a given \p Offset in the
  /// function.
  /// If \p DeriveAlignment is true, set the alignment of the block based
  /// on the alignment of the existing offset.
  /// The new block is not inserted into the CFG.  The client must
  /// use insertBasicBlocks to add any new blocks to the CFG.
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

    LabelToBB.emplace(Label, BB.get());

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

    if (CurrentState == State::Disassembled) {
      BasicBlockOffsets.emplace_back(std::make_pair(Offset, BB));
    }

    assert(CurrentState == State::CFG ||
           (std::is_sorted(BasicBlockOffsets.begin(),
                           BasicBlockOffsets.end(),
                           CompareBasicBlockOffsets()) &&
            std::is_sorted(begin(), end())));

    return BB;
  }

  /// Mark all blocks that are unreachable from a root (entry point
  /// or landing pad) as invalid.
  void markUnreachable();

  /// Rebuilds BBs layout, ignoring dead BBs. Returns the number of removed
  /// BBs and the removed number of bytes of code.
  std::pair<unsigned, uint64_t> eraseInvalidBBs();

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
    return BB
      ? iterator_range<iterator>(BasicBlocks.begin() + getIndex(BB), end())
      : iterator_range<iterator>(end(), end());
  }

  /// Insert the BBs contained in NewBBs into the basic blocks for this
  /// function. Update the associated state of all blocks as needed, i.e.
  /// BB offsets and BB indices. The new BBs are inserted after Start.
  /// This operation could affect fallthrough branches for Start.
  ///
  void insertBasicBlocks(
    BinaryBasicBlock *Start,
    std::vector<std::unique_ptr<BinaryBasicBlock>> &&NewBBs,
    const bool UpdateLayout = true,
    const bool UpdateCFIState = true);

  iterator insertBasicBlocks(
    iterator StartBB,
    std::vector<std::unique_ptr<BinaryBasicBlock>> &&NewBBs,
    const bool UpdateLayout = true,
    const bool UpdateCFIState = true);

  /// Update the basic block layout for this function.  The BBs from
  /// [Start->Index, Start->Index + NumNewBlocks) are inserted into the
  /// layout after the BB indicated by Start.
  void updateLayout(BinaryBasicBlock* Start, const unsigned NumNewBlocks);

  /// Make sure basic blocks' indices match the current layout.
  void updateLayoutIndices() const {
    unsigned Index = 0;
    for (auto *BB : layout()) {
      BB->setLayoutIndex(Index++);
    }
  }

  /// Recompute the CFI state for NumNewBlocks following Start after inserting
  /// new blocks into the CFG.  This must be called after updateLayout.
  void updateCFIState(BinaryBasicBlock *Start, const unsigned NumNewBlocks);

  /// Change \p OrigDest to \p NewDest in the jump table used at the end of
  /// \p BB. Returns false if \p OrigDest couldn't be find as a valid target
  /// and no replacement took place.
  bool replaceJumpTableEntryIn(BinaryBasicBlock *BB,
                               BinaryBasicBlock *OldDest,
                               BinaryBasicBlock *NewDest);

  /// Split the CFG edge <From, To> by inserting an intermediate basic block.
  /// Returns a pointer to this new intermediate basic block. BB "From" will be
  /// updated to jump to the intermediate block, which in turn will have an
  /// unconditional branch to BB "To".
  /// User needs to manually call fixBranches(). This function only creates the
  /// correct CFG edges.
  BinaryBasicBlock *splitEdge(BinaryBasicBlock *From, BinaryBasicBlock *To);

  /// Determine direction of the branch based on the current layout.
  /// Callee is responsible of updating basic block indices prior to using
  /// this function (e.g. by calling BinaryFunction::updateLayoutIndices()).
  static bool isForwardBranch(const BinaryBasicBlock *From,
                              const BinaryBasicBlock *To) {
    assert(From->getFunction() == To->getFunction() &&
           "basic blocks should be in the same function");
    return To->getLayoutIndex() > From->getLayoutIndex();
  }

  /// Determine direction of the call to callee symbol relative to the start
  /// of this function.
  /// Note: this doesn't take function splitting into account.
  bool isForwardCall(const MCSymbol *CalleeSymbol) const;

  /// Dump function information to debug output. If \p PrintInstructions
  /// is true - include instruction disassembly.
  void dump(bool PrintInstructions = true) const;

  /// Print function information to the \p OS stream.
  void print(raw_ostream &OS, std::string Annotation = "",
             bool PrintInstructions = true) const;

  /// Return true if function has a profile, even if the profile does not
  /// match CFG 100%.
  bool hasProfile() const {
    return ExecutionCount != COUNT_NO_PROFILE;
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
    auto I = InstructionOffsets.lower_bound(Offset);
    if (Offset == getSize()) {
      assert(I == InstructionOffsets.end() && "unexpected iterator value");
      // Sometimes compiler issues restore_state after all instructions
      // in the function (even after nop).
      --I;
      Offset = I->first;
    }
    assert(I->first == Offset && "CFI pointing to unknown instruction");
    if (I == InstructionOffsets.begin()) {
      CIEFrameInstructions.emplace_back(std::forward<MCCFIInstruction>(Inst));
      return;
    }

    --I;
    while (I != InstructionOffsets.begin() &&
           BC.MIA->isNoop(Instructions[I->second])) {
      Offset = I->first;
      --I;
    }
    OffsetToCFI.emplace(Offset, FrameInstructions.size());
    FrameInstructions.emplace_back(std::forward<MCCFIInstruction>(Inst));
    return;
  }

  BinaryBasicBlock::iterator addCFIInstruction(BinaryBasicBlock *BB,
                                               BinaryBasicBlock::iterator Pos,
                                               MCCFIInstruction &&Inst) {
    auto Idx = FrameInstructions.size();
    FrameInstructions.emplace_back(std::forward<MCCFIInstruction>(Inst));
    return addCFIPseudo(BB, Pos, Idx);
  }

  /// Insert a CFI pseudo instruction in a basic block. This pseudo instruction
  /// is a placeholder that refers to a real MCCFIInstruction object kept by
  /// this function that will be emitted at that position.
  BinaryBasicBlock::iterator addCFIPseudo(BinaryBasicBlock *BB,
                                          BinaryBasicBlock::iterator Pos,
                                          uint32_t Offset) {
    MCInst CFIPseudo;
    BC.MIA->createCFI(CFIPseudo, Offset);
    return BB->insertPseudoInstr(Pos, CFIPseudo);
  }

  /// Retrieve the MCCFIInstruction object associated with a CFI pseudo.
  MCCFIInstruction* getCFIFor(const MCInst &Instr) {
    if (!BC.MIA->isCFI(Instr))
      return nullptr;
    uint32_t Offset = Instr.getOperand(0).getImm();
    assert(Offset < FrameInstructions.size() && "Invalid CFI offset");
    return &FrameInstructions[Offset];
  }

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

  BinaryFunction &setOutputAddress(uint64_t Address) {
    OutputAddress = Address;
    return *this;
  }

  BinaryFunction &setOutputSize(uint64_t Size) {
    OutputSize = Size;
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

  BinaryFunction &setFolded(bool Folded = true) {
    IsFolded = Folded;
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

  /// Return the profile information about the number of times
  /// the function was executed.
  ///
  /// Return COUNT_NO_PROFILE if there's no profile info.
  uint64_t getExecutionCount() const {
    return ExecutionCount;
  }

  /// Return the execution count for functions with known profile.
  /// Return 0 if the function has no profile.
  uint64_t getKnownExecutionCount() const {
    return ExecutionCount == COUNT_NO_PROFILE ? 0 : ExecutionCount;
  }

  /// Return original LSDA address for the function or NULL.
  uint64_t getLSDAAddress() const {
    return LSDAAddress;
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

  /// Return symbol pointing to function's LSDA for the cold part.
  MCSymbol *getColdLSDASymbol() {
    if (ColdLSDASymbol)
      return ColdLSDASymbol;
    if (ColdCallSites.empty())
      return nullptr;

    ColdLSDASymbol =
      BC.Ctx->getOrCreateSymbol(Twine("GCC_cold_except_table") +
                                Twine::utohexstr(getFunctionNumber()));

    return ColdLSDASymbol;
  }

  /// True if the symbol is a mapping symbol used in AArch64 to delimit
  /// data inside code section.
  bool isDataMarker(const SymbolRef &Symbol, uint64_t SymbolSize) const;
  bool isCodeMarker(const SymbolRef &Symbol, uint64_t SymbolSize) const;

  void setOutputDataAddress(uint64_t Address) {
    OutputDataOffset = Address;
  }

  uint64_t getOutputDataAddress() const {
    return OutputDataOffset;
  }

  /// Detects whether \p Address is inside a data region in this function
  /// (constant islands).
  bool isInConstantIsland(uint64_t Address) const {
    if (Address <= getAddress())
      return false;

    auto Offset = Address - getAddress();

    if (Offset >= getMaxSize())
      return false;

    auto DataIter = DataOffsets.upper_bound(Offset);
    if (DataIter == DataOffsets.begin())
      return false;
    DataIter = std::prev(DataIter);

    auto CodeIter = CodeOffsets.upper_bound(Offset);
    if (CodeIter == CodeOffsets.begin())
      return true;

    return *std::prev(CodeIter) <= *DataIter;
  }

  uint64_t estimateConstantIslandSize() const {
    uint64_t Size = 0;
    for (auto DataIter = DataOffsets.begin(); DataIter != DataOffsets.end();
         ++DataIter) {
      auto NextData = std::next(DataIter);
      auto CodeIter = CodeOffsets.lower_bound(*DataIter);
      if (CodeIter == CodeOffsets.end() &&
          NextData == DataOffsets.end()) {
        Size += getMaxSize() - *DataIter;
        continue;
      }

      uint64_t NextMarker;
      if (CodeIter == CodeOffsets.end())
        NextMarker = *NextData;
      else if (NextData == DataOffsets.end())
        NextMarker = *CodeIter;
      else
        NextMarker = (*CodeIter > *NextData) ? *NextData : *CodeIter;

      Size += NextMarker - *DataIter;
    }
    return Size;
  }

  bool hasConstantIsland() const {
    return !DataOffsets.empty();
  }

  /// Return true iff the symbol could be seen inside this function otherwise
  /// it is probably another function.
  bool isSymbolValidInScope(const SymbolRef &Symbol, uint64_t SymbolSize) const;

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
  void disassemble(ArrayRef<uint8_t> FunctionData);

  /// Post-processing for jump tables after disassembly. Since their
  /// boundaries are not known until all call sites are seen, we need this
  /// extra pass to perform any final adjustments.
  void postProcessJumpTables();

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

  /// In functions with multiple entry points, the profile collection records
  /// data for other entry points in a different function entry. This function
  /// attempts to fetch extra profile data for each secondary entry point.
  bool fetchProfileForOtherEntryPoints();

  /// Find the best matching profile for a function after the creation of basic
  /// blocks.
  void matchProfileData();

  /// Check how closely the profile data matches the function and set
  /// Return accuracy (ranging from 0.0 to 1.0) of matching.
  float evaluateProfileData(const FuncBranchData &BranchData);

  /// Return profile data associated with this function, or nullptr if the
  /// function has no associated profile.
  const FuncBranchData *getBranchData() const {
    return BranchData;
  }

  FuncBranchData *getBranchData() {
    return BranchData;
  }

  /// Updates profile data associated with this function
  void setBranchData(FuncBranchData *Data) {
    BranchData = Data;
  }

  /// Walks the list of basic blocks filling in missing information about
  /// edge frequency for fall-throughs.
  ///
  /// Assumes the CFG has been built and edge frequency for taken branches
  /// has been filled with LBR data.
  void inferFallThroughCounts();

  /// Clear execution profile of the function.
  void clearProfile();

  /// Converts conditional tail calls to unconditional tail calls. We do this to
  /// handle conditional tail calls correctly and to give a chance to the
  /// simplify conditional tail call pass to decide whether to re-optimize them
  /// using profile information.
  void removeConditionalTailCalls();

  // Convert COUNT_NO_PROFILE to 0
  void removeTagsFromProfile();

  /// If our profile data comes from sample addresses instead of LBR entries,
  /// collect sample count for all addresses in this function address space,
  /// aggregating them per basic block and assigning an execution count to each
  /// basic block based on the number of samples recorded at those addresses.
  /// The last step is to infer edge counts based on BB execution count. Note
  /// this is the opposite of the LBR way, where we infer BB execution count
  /// based on edge counts.
  void readSampleData();

  /// Computes a function hotness score: the sum of the products of BB frequency
  /// and size.
  uint64_t getFunctionScore();

  /// Return true if the layout has been changed by basic block reordering,
  /// false otherwise.
  bool hasLayoutChanged() const;

  /// Get the edit distance of the new layout with respect to the previous
  /// layout after basic block reordering.
  uint64_t getEditDistance() const;

  /// Get the number of instructions within this function.
  uint64_t getInstructionCount() const;

  const CFIInstrMapType &getFDEProgram() const {
    return FrameInstructions;
  }

  /// After reordering, this function checks the state of CFI and fixes it if it
  /// is corrupted. If it is unable to fix it, it returns false.
  bool fixCFIState();

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

  /// Mark function as finalized. No further optimizations are permitted.
  void setFinalized() {
    CurrentState = State::CFG_Finalized;
  }

  void setEmitted() {
    CurrentState = State::Emitted;
  }

  /// Process LSDA information for the function.
  void parseLSDA(ArrayRef<uint8_t> LSDAData, uint64_t LSDAAddress);

  /// Update exception handling ranges for the function.
  void updateEHRanges();

  /// Emit exception handling ranges for the function.
  void emitLSDA(MCStreamer *Streamer, bool EmitColdPart);

  /// Emit jump tables for the function.
  void emitJumpTables(MCStreamer *Streamer);

  /// Emit function code. The caller is responsible for emitting function
  /// symbol(s) and setting the section to emit the code to.
  void emitBody(MCStreamer &Streamer, bool EmitColdPart);

  /// Emit function as a blob with relocations and labels for relocations.
  void emitBodyRaw(MCStreamer *Streamer);

  /// Helper for emitBody to write data inside a function (used for AArch64)
  void emitConstantIslands(MCStreamer &Streamer);

  /// Merge profile data of this function into those of the given
  /// function. The functions should have been proven identical with
  /// isIdenticalWith.
  void mergeProfileDataInto(BinaryFunction &BF) const;

  /// Returns true if this function has identical code and CFG with
  /// the given function \p BF.
  ///
  /// If \p IgnoreSymbols is set to true, then symbolic operands are ignored
  /// during comparison.
  ///
  /// If \p UseDFS is set to true, then compute DFS of each function and use
  /// is for CFG equivalency. Potentially it will help to catch more cases,
  /// but is slower.
  bool isIdenticalWith(const BinaryFunction &BF,
                       bool IgnoreSymbols = false,
                       bool UseDFS = false) const;

  /// Returns a hash value for the function. To be used for ICF. Two congruent
  /// functions (functions with different symbolic references but identical
  /// otherwise) are required to have identical hashes.
  ///
  /// If \p UseDFS is set, then process blocks in DFS order that we recompute.
  /// Otherwise use the existing layout order.
  std::size_t hash(bool Recompute = true, bool UseDFS = false) const;

  /// Sets the associated .debug_info entry.
  void addSubprogramDIE(DWARFCompileUnit *Unit,
                        const DWARFDebugInfoEntryMinimal *DIE) {
    SubprogramDIEs.emplace_back(DIE, Unit);
    if (!UnitLineTable.first) {
      if (const auto *LineTable = BC.DwCtx->getLineTableForUnit(Unit)) {
        UnitLineTable = std::make_pair(Unit, LineTable);
      }
    }
  }

  /// Return all compilation units with entry for this function.
  /// Because of identical code folding there could be multiple of these.
  const decltype(SubprogramDIEs) &getSubprogramDIEs() const {
    return SubprogramDIEs;
  }

  /// Return DWARF compile unit with line info for this function.
  DWARFUnitLineTable getDWARFUnitLineTable() const {
    return UnitLineTable;
  }

  /// Scan from - to offsets for conditional jumps
  Optional<SmallVector<std::pair<uint64_t, uint64_t>, 16>>
  getFallthroughsInTrace(uint64_t From, uint64_t To) const;

  /// Returns an estimate of the function's hot part after splitting.
  /// This is a very rough estimate, as with C++ exceptions there are
  /// blocks we don't move, and it makes no attempt at estimating the size
  /// of the added/removed branch instructions.
  /// Note that this size is optimistic and the actual size may increase
  /// after relaxation.
  size_t estimateHotSize(const bool UseSplitSize = true) const {
    size_t Estimate = 0;
    if (UseSplitSize && isSplit()) {
      for (const auto *BB : BasicBlocksLayout) {
        if (!BB->isCold()) {
          Estimate += BC.computeCodeSize(BB->begin(), BB->end());
        }
      }
    } else {
      for (const auto *BB : BasicBlocksLayout) {
        if (BB->getKnownExecutionCount() != 0) {
          Estimate += BC.computeCodeSize(BB->begin(), BB->end());
        }
      }
    }
    return Estimate;
  }

  size_t estimateColdSize() const {
    if (!isSplit())
      return estimateSize();
    size_t Estimate = 0;
    for (const auto *BB : BasicBlocksLayout) {
      if (BB->isCold()) {
        Estimate += BC.computeCodeSize(BB->begin(), BB->end());
      }
    }
    return Estimate;
  }

  size_t estimateSize() const {
    size_t Estimate = 0;
    for (const auto *BB : BasicBlocksLayout) {
      Estimate += BC.computeCodeSize(BB->begin(), BB->end());
    }
    return Estimate;
  }

  /// Return output address ranges for a function.
  DWARFAddressRangesVector getOutputAddressRanges() const;

  /// Given an address corresponding to an instruction in the input binary,
  /// return an address of this instruction in output binary.
  ///
  /// Return 0 if no matching address could be found or the instruction was
  /// removed.
  uint64_t translateInputToOutputAddress(uint64_t Address) const;

  /// Take address ranges corresponding to the input binary and translate
  /// them to address ranges in the output binary.
  DWARFAddressRangesVector translateInputToOutputRanges(
      const DWARFAddressRangesVector &InputRanges) const;

  /// Similar to translateInputToOutputRanges() but operates on location lists
  /// and moves associated data to output location lists.
  ///
  /// \p BaseAddress is applied to all addresses in \pInputLL.
  DWARFDebugLoc::LocationList translateInputToOutputLocationList(
      const DWARFDebugLoc::LocationList &InputLL,
      uint64_t BaseAddress) const;

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
  DynoStats DynoStatsBefore;
  if (Flag) {
    DynoStatsBefore = getDynoStats(Funcs);
  }

  Func();

  if (Flag) {
    const auto DynoStatsAfter = getDynoStats(Funcs);
    const auto Changed = (DynoStatsAfter != DynoStatsBefore);
    outs() << "BOLT-INFO: program-wide dynostats after running "
           << Phase << (Changed ? "" : " (no change)") << ":\n\n"
           << DynoStatsBefore << '\n';
    if (Changed) {
      DynoStatsAfter.print(outs(), &DynoStatsBefore);
    }
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
  case BinaryFunction::State::Empty:        OS << "empty";  break;
  case BinaryFunction::State::Disassembled: OS << "disassembled";  break;
  case BinaryFunction::State::CFG:          OS << "CFG constructed";  break;
  case BinaryFunction::State::CFG_Finalized:OS << "CFG finalized";  break;
  case BinaryFunction::State::Emitted:      OS << "emitted";  break;
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
    llvm_unreachable("Not implemented");
    return &(*F->begin());
  }
  static nodes_iterator nodes_end(bolt::BinaryFunction *F) {
    llvm_unreachable("Not implemented");
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
    llvm_unreachable("Not implemented");
    return &(*F->begin());
  }
  static nodes_iterator nodes_end(const bolt::BinaryFunction *F) {
    llvm_unreachable("Not implemented");
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
