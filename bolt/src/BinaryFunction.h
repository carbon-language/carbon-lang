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
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_FUNCTION_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_FUNCTION_H

#include "BinaryBasicBlock.h"
#include "BinaryContext.h"
#include "BinaryLoop.h"
#include "BinarySection.h"
#include "DebugData.h"
#include "JumpTable.h"
#include "MCPlus.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/iterator.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace llvm::object;

namespace llvm {

class DWARFUnit;
class DWARFDebugInfoEntryMinimal;

namespace bolt {

using DWARFUnitLineTable = std::pair<DWARFUnit *,
                                     const DWARFDebugLine::LineTable *>;

/// Types of macro-fusion alignment corrections.
enum MacroFusionType {
  MFT_NONE,
  MFT_HOT,
  MFT_ALL
};

enum IndirectCallPromotionType : char {
  ICP_NONE,        /// Don't perform ICP.
  ICP_CALLS,       /// Perform ICP on indirect calls.
  ICP_JUMP_TABLES, /// Perform ICP on jump tables.
  ICP_ALL          /// Perform ICP on calls and jump tables.
};

/// Information on a single indirect call to a particular callee.
struct IndirectCallProfile {
  MCSymbol *Symbol;
  uint32_t Offset;
  uint64_t Count;
  uint64_t Mispreds;

  IndirectCallProfile(MCSymbol *Symbol, uint64_t Count,
                      uint64_t Mispreds, uint32_t Offset = 0)
    : Symbol(Symbol), Offset(Offset), Count(Count), Mispreds(Mispreds) {}

  bool operator==(const IndirectCallProfile &Other) const {
    return Symbol == Other.Symbol &&
           Offset == Other.Offset;
  }
};

/// Aggregated information for an indirect call site.
using IndirectCallSiteProfile = SmallVector<IndirectCallProfile, 4>;

inline raw_ostream &operator<<(raw_ostream &OS,
                               const bolt::IndirectCallSiteProfile &ICSP) {
  std::string TempString;
  raw_string_ostream SS(TempString);

  const char *Sep = "\n        ";
  uint64_t TotalCount = 0;
  uint64_t TotalMispreds = 0;
  for (auto &CSP : ICSP) {
    SS << Sep << "{ " << (CSP.Symbol ? CSP.Symbol->getName() : "<unknown>")
       << ": " << CSP.Count << " (" << CSP.Mispreds << " misses) }";
    Sep = ",\n        ";
    TotalCount += CSP.Count;
    TotalMispreds += CSP.Mispreds;
  }
  SS.flush();

  OS << TotalCount << " (" << TotalMispreds << " misses) :" << TempString;
  return OS;
}

/// BinaryFunction is a representation of machine-level function.
///
/// We use the term "Binary" as "Machine" was already taken.
class BinaryFunction {
public:
  enum class State : char {
    Empty = 0,        /// Function body is empty.
    Disassembled,     /// Function have been disassembled.
    CFG,              /// Control flow graph has been built.
    CFG_Finalized,    /// CFG is finalized. No optimizations allowed.
    EmittedCFG,       /// Instructions have been emitted to output.
    Emitted,          /// Same as above plus CFG is destroyed.
  };

  /// Types of profile the function can use. Could be a combination.
  enum {
    PF_NONE = 0,         /// No profile.
    PF_LBR = 1,          /// Profile is based on last branch records.
    PF_SAMPLE = 2,       /// Non-LBR sample-based profile.
    PF_MEMEVENT = 4,     /// Profile has mem events.
  };

  /// Struct for tracking exception handling ranges.
  struct CallSite {
    const MCSymbol *Start;
    const MCSymbol *End;
    const MCSymbol *LP;
    uint64_t Action;
  };

  using IslandProxiesType =
    std::map<BinaryFunction *, std::map<const MCSymbol *, MCSymbol *>>;

  struct IslandInfo {
    /// Temporary holder of offsets that are data markers (used in AArch)
    /// It is possible to have data in code sections. To ease the identification
    /// of data in code sections, the ABI requires the symbol table to have
    /// symbols named "$d" identifying the start of data inside code and "$x"
    /// identifying the end of a chunk of data inside code. DataOffsets contain
    /// all offsets of $d symbols and CodeOffsets all offsets of $x symbols.
    std::set<uint64_t> DataOffsets;
    std::set<uint64_t> CodeOffsets;

    /// Offsets in function that are data values in a constant island identified
    /// after disassembling
    std::map<uint64_t, MCSymbol *> Offsets;
    SmallPtrSet<MCSymbol *, 4> Symbols;
    std::map<const MCSymbol *, BinaryFunction *> ProxySymbols;
    std::map<const MCSymbol *, MCSymbol *> ColdSymbols;
    /// Keeps track of other functions we depend on because there is a reference
    /// to the constant islands in them.
    IslandProxiesType Proxies, ColdProxies;
    std::set<BinaryFunction *> Dependency; // The other way around
  };

  static constexpr uint64_t COUNT_NO_PROFILE =
    BinaryBasicBlock::COUNT_NO_PROFILE;

  /// We have to use at least 2-byte alignment for functions because of C++ ABI.
  static constexpr unsigned MinAlign = 2;

  static const char TimerGroupName[];
  static const char TimerGroupDesc[];

  using BasicBlockOrderType = std::vector<BinaryBasicBlock *>;

  /// Mark injected functions
  bool IsInjected = false;

private:
  /// Current state of the function.
  State CurrentState{State::Empty};

  /// A list of symbols associated with the function entry point.
  ///
  /// Multiple symbols would typically result from identical code-folding
  /// optimization.
  typedef SmallVector<MCSymbol *, 1> SymbolListTy;
  SymbolListTy Symbols;

  /// The list of names this function is known under. Used for fuzzy-matching
  /// the function to its name in a profile, command line, etc.
  std::vector<std::string> Aliases;

  /// Containing section
  BinarySection *InputSection = nullptr;

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
  uint16_t Alignment{2};

  /// Maximum number of bytes used for alignment of hot part of the function.
  uint16_t MaxAlignmentBytes{0};

  /// Maximum number of bytes used for alignment of cold part of the function.
  uint16_t MaxColdAlignmentBytes{0};

  const MCSymbol *PersonalityFunction{nullptr};
  uint8_t PersonalityEncoding{dwarf::DW_EH_PE_sdata4 | dwarf::DW_EH_PE_pcrel};

  BinaryContext &BC;

  std::unique_ptr<BinaryLoopInfo> BLI;

  /// Set of external addresses in the code that are not a function start
  /// and are referenced from this function.
  std::set<uint64_t> InterproceduralReferences;

  /// All labels in the function that are referenced via relocations from
  /// data objects. Typically these are jump table destinations and computed
  /// goto labels.
  std::set<uint64_t> ExternallyReferencedOffsets;

  /// Offsets of indirect branches with unknown destinations.
  std::set<uint64_t> UnknownIndirectBranchOffsets;

  /// A set of local and global symbols corresponding to secondary entry points.
  /// Each additional function entry point has a corresponding entry in the map.
  /// The key is a local symbol corresponding to a basic block and the value
  /// is a global symbol corresponding to an external entry point.
  std::unordered_map<const MCSymbol *, MCSymbol *> SecondaryEntryPoints;

  /// False if the function is too complex to reconstruct its control
  /// flow graph.
  /// In relocation mode we still disassemble and re-assemble such functions.
  bool IsSimple{true};

  /// Indication that the function should be ignored for optimization purposes.
  /// If we can skip emission of some functions, then ignored functions could
  /// be not fully disassembled and will not be emitted.
  bool IsIgnored{false};

  /// Pseudo functions should not be disassembled or emitted.
  bool IsPseudo{false};

  /// True if the original function code has all necessary relocations to track
  /// addresses of functions emitted to new locations. Typically set for
  /// functions that we are not going to emit.
  bool HasExternalRefRelocations{false};

  /// True if the function has an indirect branch with unknown destination.
  bool HasUnknownControlFlow{false};

  /// The code from inside the function references one of the code locations
  /// from the same function as a data, i.e. it's possible the label is used
  /// inside an address calculation or could be referenced from outside.
  bool HasInternalLabelReference{false};

  /// In AArch64, preserve nops to maintain code equal to input (assuming no
  /// optimizations are done).
  bool PreserveNops{false};

  /// Indicate if this function has associated exception handling metadata.
  bool HasEHRanges{false};

  /// True if the function uses DW_CFA_GNU_args_size CFIs.
  bool UsesGnuArgsSize{false};

  /// True if the function might have a profile available externally.
  /// Used to check if processing of the function is required under certain
  /// conditions.
  bool HasProfileAvailable{false};

  bool HasMemoryProfile{false};

  /// Execution halts whenever this function is entered.
  bool TrapsOnEntry{false};

  /// True if the function had an indirect branch with a fixed internal
  /// destination.
  bool HasFixedIndirectBranch{false};

  /// True if the function is a fragment of another function. This means that
  /// this function could only be entered via its parent or one of its sibling
  /// fragments. It could be entered at any basic block. It can also return
  /// the control to any basic block of its parent or its sibling.
  bool IsFragment{false};

  /// Indicate that the function body has SDT marker
  bool HasSDTMarker{false};

  /// True if the original entry point was patched.
  bool IsPatched{false};

  /// The address for the code for this function in codegen memory.
  uint64_t ImageAddress{0};

  /// The size of the code in memory.
  uint64_t ImageSize{0};

  /// Name for the section this function code should reside in.
  std::string CodeSectionName;

  /// Name for the corresponding cold code section.
  std::string ColdCodeSectionName;

  /// Parent function for split function fragments.
  BinaryFunction *ParentFunction{nullptr};

  /// Indicate if the function body was folded into another function.
  /// Used by ICF optimization.
  BinaryFunction *FoldedIntoFunction{nullptr};

  /// All fragments for a parent function.
  std::unordered_set<BinaryFunction *> Fragments;

  /// The profile data for the number of times the function was executed.
  uint64_t ExecutionCount{COUNT_NO_PROFILE};

  /// Profile match ratio.
  float ProfileMatchRatio{0.0f};

  /// Indicates the type of profile the function is using.
  uint16_t ProfileFlags{PF_NONE};

  /// For functions with mismatched profile we store all call profile
  /// information at a function level (as opposed to tying it to
  /// specific call sites).
  IndirectCallSiteProfile AllCallSites;

  /// Score of the function (estimated number of instructions executed,
  /// according to profile data). -1 if the score has not been calculated yet.
  mutable int64_t FunctionScore{-1};

  /// Original LSDA address for the function.
  uint64_t LSDAAddress{0};

  /// Associated DIEs in the .debug_info section with their respective CUs.
  /// There can be multiple because of identical code folding.
  std::vector<DWARFDie> SubprogramDIEs;

  /// Line table for the function with containing compilation unit.
  /// Because of identical code folding the function could have multiple
  /// associated compilation units. The first of them with line number info
  /// is referenced by UnitLineTable.
  DWARFUnitLineTable UnitLineTable{nullptr, nullptr};

  /// Last computed hash value. Note that the value could be recomputed using
  /// different parameters by every pass.
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

  /// Update the indices of all the basic blocks starting at StartIndex.
  void updateBBIndices(const unsigned StartIndex);

  /// Annotate each basic block entry with its current CFI state. This is
  /// run right after the construction of CFG while basic blocks are in their
  /// original order.
  void annotateCFIState();

  /// Associate DW_CFA_GNU_args_size info with invoke instructions
  /// (call instructions with non-empty landing pad).
  void propagateGnuArgsSizeInfo(MCPlusBuilder::AllocatorIdTy AllocId);

  /// Synchronize branch instructions with CFG.
  void postProcessBranches();

  /// The address offset where we emitted the constant island, that is, the
  /// chunk of data in the function code area (AArch only)
  int64_t OutputDataOffset{0};
  int64_t OutputColdDataOffset{0};

  /// Map labels to corresponding basic blocks.
  std::unordered_map<const MCSymbol *, BinaryBasicBlock *> LabelToBB;

  using BranchListType = std::vector<std::pair<uint32_t, uint32_t>>;
  BranchListType TakenBranches;       /// All local taken branches.
  BranchListType IgnoredBranches;     /// Branches ignored by CFG purposes.

  /// Map offset in the function to a label.
  /// Labels are used for building CFG for simple functions. For non-simple
  /// function in relocation mode we need to emit them for relocations
  /// referencing function internals to work (e.g. jump tables).
  using LabelsMapType = std::map<uint32_t, MCSymbol *>;
  LabelsMapType Labels;

  /// Temporary holder of instructions before CFG is constructed.
  /// Map offset in the function to MCInst.
  using InstrMapType = std::map<uint32_t, MCInst>;
  InstrMapType Instructions;

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

  /// A map of restore state CFI instructions to their equivalent CFI
  /// instructions that produce the same state, in order to eliminate
  /// remember-restore CFI instructions when rewriting CFI.
  DenseMap<int32_t , SmallVector<int32_t, 4>> FrameRestoreEquivalents;

  // For tracking exception handling ranges.
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

  /// All compound jump tables for this function. This duplicates what's stored
  /// in the BinaryContext, but additionally it gives quick access for all
  /// jump tables used by this function.
  ///
  /// <OriginalAddress> -> <JumpTable *>
  std::map<uint64_t, JumpTable *> JumpTables;

  /// All jump table sites in the function before CFG is built.
  std::vector<std::pair<uint64_t, uint64_t>> JTSites;

  /// List of relocations in this function.
  std::map<uint64_t, Relocation> Relocations;

  /// Map of relocations used for moving the function body as it is.
  using MoveRelocationsTy = std::map<uint64_t, Relocation>;
  MoveRelocationsTy MoveRelocations;

  /// Information on function constant islands.
  IslandInfo Islands;

  // Blocks are kept sorted in the layout order. If we need to change the
  // layout (if BasicBlocksLayout stores a different order than BasicBlocks),
  // the terminating instructions need to be modified.
  using BasicBlockListType = std::vector<BinaryBasicBlock *>;
  BasicBlockListType BasicBlocks;
  BasicBlockListType DeletedBasicBlocks;
  BasicBlockOrderType BasicBlocksLayout;
  /// Previous layout replaced by modifyLayout
  BasicBlockOrderType BasicBlocksPreviousLayout;
  bool ModifiedLayout{false};

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

  MCSymbol *ColdSymbol{nullptr};

  /// Symbol at the end of the function.
  mutable MCSymbol *FunctionEndLabel{nullptr};

  /// Symbol at the end of the cold part of split function.
  mutable MCSymbol *FunctionColdEndLabel{nullptr};

  mutable MCSymbol *FunctionConstantIslandLabel{nullptr};
  mutable MCSymbol *FunctionColdConstantIslandLabel{nullptr};

  /// Unique number associated with the function.
  uint64_t  FunctionNumber;

  /// Count the number of functions created.
  static uint64_t Count;

  /// Map offsets of special instructions to addresses in the output.
  using InputOffsetToAddressMapTy = std::unordered_map<uint64_t, uint64_t>;
  InputOffsetToAddressMapTy InputOffsetToAddressMap;

  /// Register alternative function name.
  void addAlternativeName(std::string NewName) {
    Aliases.push_back(std::move(NewName));
  }

  /// Return a label at a given \p Address in the function. If the label does
  /// not exist - create it. Assert if the \p Address does not belong to
  /// the function. If \p CreatePastEnd is true, then return the function
  /// end label when the \p Address points immediately past the last byte
  /// of the function.
  /// NOTE: the function always returns a local (temp) symbol, even if there's
  ///       a global symbol that corresponds to an entry at this address.
  MCSymbol *getOrCreateLocalLabel(uint64_t Address, bool CreatePastEnd = false);

  /// Register an entry point at a given \p Offset into the function.
  void markDataAtOffset(uint64_t Offset) {
    Islands.DataOffsets.emplace(Offset);
  }

  /// Register an entry point at a given \p Offset into the function.
  void markCodeAtOffset(uint64_t Offset) {
    Islands.CodeOffsets.emplace(Offset);
  }

  /// Register secondary entry point at a given \p Offset into the function.
  /// Return global symbol for use by extern function references.
  MCSymbol *addEntryPointAtOffset(uint64_t Offset);

  /// Register an internal offset in a function referenced from outside.
  void registerReferencedOffset(uint64_t Offset) {
    ExternallyReferencedOffsets.emplace(Offset);
  }

  /// True if there are references to internals of this function from data,
  /// e.g. from jump tables.
  bool hasInternalReference() const {
    return !ExternallyReferencedOffsets.empty();
  }

  /// Return an entry ID corresponding to a symbol known to belong to
  /// the function.
  ///
  /// Prefer to use BinaryContext::getFunctionForSymbol(EntrySymbol, &ID)
  /// instead of calling this function directly.
  uint64_t getEntryIDForSymbol(const MCSymbol *EntrySymbol) const;

  void setParentFunction(BinaryFunction *BF) {
    assert((!ParentFunction || ParentFunction == BF) &&
           "cannot have more than one parent function");
    ParentFunction = BF;
  }

  void addFragment(BinaryFunction *BF) {
    Fragments.insert(BF);
  }

  void addInstruction(uint64_t Offset, MCInst &&Instruction) {
    Instructions.emplace(Offset, std::forward<MCInst>(Instruction));
  }

  /// Convert CFI instructions to a standard form (remove remember/restore).
  void normalizeCFIState();

  /// Analyze and process indirect branch \p Instruction before it is
  /// added to Instructions list.
  IndirectBranchType processIndirectBranch(MCInst &Instruction,
                                           unsigned Size,
                                           uint64_t Offset,
                                           uint64_t &TargetAddress);

  DenseMap<const MCInst *, SmallVector<MCInst *, 4>>
  computeLocalUDChain(const MCInst *CurInstr);

  BinaryFunction &operator=(const BinaryFunction &) = delete;
  BinaryFunction(const BinaryFunction &) = delete;

  friend class MachORewriteInstance;
  friend class RewriteInstance;
  friend class BinaryContext;
  friend class DataReader;
  friend class DataAggregator;

  static std::string buildCodeSectionName(StringRef Name,
                                          const BinaryContext &BC);
  static std::string buildColdCodeSectionName(StringRef Name,
                                              const BinaryContext &BC);

  /// Creation should be handled by RewriteInstance or BinaryContext
  BinaryFunction(const std::string &Name, BinarySection &Section,
                 uint64_t Address, uint64_t Size, BinaryContext &BC)
      : InputSection(&Section), Address(Address), Size(Size), BC(BC),
        CodeSectionName(buildCodeSectionName(Name, BC)),
        ColdCodeSectionName(buildColdCodeSectionName(Name, BC)),
        FunctionNumber(++Count) {
    Symbols.push_back(BC.Ctx->getOrCreateSymbol(Name));
  }

  /// This constructor is used to create an injected function
  BinaryFunction(const std::string &Name, BinaryContext &BC, bool IsSimple)
      : Address(0), Size(0), BC(BC), IsSimple(IsSimple),
        CodeSectionName(buildCodeSectionName(Name, BC)),
        ColdCodeSectionName(buildColdCodeSectionName(Name, BC)),
        FunctionNumber(++Count) {
    Symbols.push_back(BC.Ctx->getOrCreateSymbol(Name));
    IsInjected = true;
  }

  /// Clear state of the function that could not be disassembled or if its
  /// disassembled state was later invalidated.
  void clearDisasmState();

  /// Release memory allocated for CFG and instructions.
  /// We still keep basic blocks for address translation/mapping purposes.
  void releaseCFG() {
    for (auto *BB : BasicBlocks) {
      BB->releaseCFG();
    }
    for (auto BB : DeletedBasicBlocks) {
      BB->releaseCFG();
    }

    clearList(CallSites);
    clearList(ColdCallSites);
    clearList(LSDATypeTable);

    clearList(MoveRelocations);

    clearList(LabelToBB);

    if (!isMultiEntry())
      clearList(Labels);

    clearList(FrameInstructions);
    clearList(FrameRestoreEquivalents);
  }

public:
  BinaryFunction(BinaryFunction &&) = default;

  using iterator = pointee_iterator<BasicBlockListType::iterator>;
  using const_iterator = pointee_iterator<BasicBlockListType::const_iterator>;
  using reverse_iterator =
      pointee_iterator<BasicBlockListType::reverse_iterator>;
  using const_reverse_iterator =
      pointee_iterator<BasicBlockListType::const_reverse_iterator>;

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
  inline iterator_range<iterator> blocks() {
    return iterator_range<iterator>(begin(), end());
  }
  inline iterator_range<const_iterator> blocks() const {
    return iterator_range<const_iterator>(begin(), end());
  }

  // Iterators by pointer.
  BasicBlockListType::iterator pbegin()  { return BasicBlocks.begin(); }
  BasicBlockListType::iterator pend()    { return BasicBlocks.end(); }

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

  inline iterator_range<reverse_order_iterator> rlayout() {
    return
      iterator_range<reverse_order_iterator>(BasicBlocksLayout.rbegin(),
                                             BasicBlocksLayout.rend());
  }

  inline iterator_range<const_reverse_order_iterator> rlayout() const {
    return
      iterator_range<const_reverse_order_iterator>(BasicBlocksLayout.rbegin(),
                                                   BasicBlocksLayout.rend());
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

  /// Iterate over all jump tables associated with this function.
  iterator_range<std::map<uint64_t, JumpTable *>::const_iterator>
  jumpTables() const {
    return make_range(JumpTables.begin(), JumpTables.end());
  }

  /// Returns the raw binary encoding of this function.
  ErrorOr<ArrayRef<uint8_t>> getData() const;

  BinaryFunction &updateState(BinaryFunction::State State) {
    CurrentState = State;
    return *this;
  }

  /// Update layout of basic blocks used for output.
  void updateBasicBlockLayout(BasicBlockOrderType &NewLayout) {
    BasicBlocksPreviousLayout = BasicBlocksLayout;

    if (NewLayout != BasicBlocksLayout) {
      ModifiedLayout = true;
      BasicBlocksLayout.clear();
      BasicBlocksLayout.swap(NewLayout);
    }
  }

  /// Recompute landing pad information for the function and all its blocks.
  void recomputeLandingPads();

  /// Return current basic block layout.
  const BasicBlockOrderType &getLayout() const {
    return BasicBlocksLayout;
  }

  /// Return a list of basic blocks sorted using DFS and update layout indices
  /// using the same order. Does not modify the current layout.
  BasicBlockOrderType dfs() const;

  /// Find the loops in the CFG of the function and store information about
  /// them.
  void calculateLoopInfo();

  /// Calculate missed macro-fusion opportunities and update BinaryContext
  /// stats.
  void calculateMacroOpFusionStats();

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
                                       const MCInst &InvokeInst) const {
    assert(BC.MIB->isInvoke(InvokeInst) && "must be invoke instruction");
    const auto LP = BC.MIB->getEHInfo(InvokeInst);
    if (LP && LP->first) {
      auto *LBB = BB.getLandingPad(LP->first);
      assert (LBB && "Landing pad should be defined");
      return LBB;
    }
    return nullptr;
  }

  /// Return instruction at a given offset in the function. Valid before
  /// CFG is constructed or while instruction offsets are available in CFG.
  MCInst *getInstructionAtOffset(uint64_t Offset);

  const MCInst *getInstructionAtOffset(uint64_t Offset) const {
    return const_cast<BinaryFunction *>(this)->getInstructionAtOffset(Offset);
  }

  /// Return jump table that covers a given \p Address in memory.
  JumpTable *getJumpTableContainingAddress(uint64_t Address) {
    auto JTI = JumpTables.upper_bound(Address);
    if (JTI == JumpTables.begin())
      return nullptr;
    --JTI;
    if (JTI->first + JTI->second->getSize() > Address)
      return JTI->second;
    if (JTI->second->getSize() == 0 && JTI->first == Address)
      return JTI->second;
    return nullptr;
  }

  const JumpTable *getJumpTableContainingAddress(uint64_t Address) const {
    return const_cast<BinaryFunction *>(this)->
      getJumpTableContainingAddress(Address);
  }

  /// Return the name of the function if the function has just one name.
  /// If the function has multiple names - return one followed
  /// by "(*#<numnames>)".
  ///
  /// We should use getPrintName() for diagnostics and use
  /// hasName() to match function name against a given string.
  ///
  /// NOTE: for disambiguating names of local symbols we use the following
  ///       naming schemes:
  ///           primary:     <function>/<id>
  ///           alternative: <function>/<file>/<id2>
  std::string getPrintName() const {
    const auto NumNames = Symbols.size() + Aliases.size();
    return NumNames == 1
        ? getOneName().str()
        : (getOneName().str() + "(*" + std::to_string(NumNames) + ")");
  }

  /// The function may have many names. For that reason, we avoid having
  /// getName() method as most of the time the user needs a different
  /// interface, such as forEachName(), hasName(), hasNameRegex(), etc.
  /// In some cases though, we need just a name uniquely identifying
  /// the function, and that's what this method is for.
  StringRef getOneName() const {
    return Symbols[0]->getName();
  }

  /// Return the name of the function as getPrintName(), but also trying
  /// to demangle it.
  std::string getDemangledName() const;

  /// Call \p Callback for every name of this function as long as the Callback
  /// returns false. Stop if Callback returns true or all names have been used.
  /// Return the name for which the Callback returned true if any.
  template <typename FType>
  Optional<StringRef> forEachName(FType Callback) const {
    for (auto *Symbol : Symbols)
      if (Callback(Symbol->getName()))
        return Symbol->getName();

    for (auto &Name : Aliases)
      if (Callback(StringRef(Name)))
        return StringRef(Name);

    return NoneType();
  }

  /// Check if (possibly one out of many) function name matches the given
  /// string. Use this member function instead of direct name comparison.
  bool hasName(const std::string &FunctionName) const {
    auto Res = forEachName([&](StringRef Name) {
      return Name == FunctionName;
    });
    return Res.hasValue();
  }

  /// Check if of function names matches the given regex.
  Optional<StringRef> hasNameRegex(const StringRef NameRegex) const;

  /// Return a vector of all possible names for the function.
  const std::vector<StringRef> getNames() const {
    std::vector<StringRef> AllNames;
    forEachName([&AllNames] (StringRef Name) {
        AllNames.push_back(Name);
        return false;
    });

    return AllNames;
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
           getState() == State::EmittedCFG;
  }

  /// Return true if the function state implies that it includes instructions.
  bool hasInstructions() const {
    return getState() == State::Disassembled ||
           hasCFG();
  }

  bool isEmitted() const {
    return getState() == State::EmittedCFG ||
           getState() == State::Emitted;
  }

  BinarySection &getSection() const {
    assert(InputSection);
    return *InputSection;
  }

  bool isInjected() const {
    return IsInjected;
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
    return Symbols[0];
  }

  /// Return MC symbol associated with the function (const version).
  /// All references to the function should use this symbol.
  const MCSymbol *getSymbol() const {
    return Symbols[0];
  }

  /// Return a list of symbols associated with the main entry of the function.
  SymbolListTy &getSymbols() { return Symbols; }
  const SymbolListTy &getSymbols() const { return Symbols; }

  /// If a local symbol \p BBLabel corresponds to a basic block that is a
  /// secondary entry point into the function, then return a global symbol
  /// that represents the secondary entry point. Otherwise return nullptr.
  MCSymbol *getSecondaryEntryPointSymbol(const MCSymbol *BBLabel) const {
    auto I = SecondaryEntryPoints.find(BBLabel);
    if (I == SecondaryEntryPoints.end())
      return nullptr;

    return I->second;
  }

  /// If the basic block serves as a secondary entry point to the function,
  /// return a global symbol representing the entry. Otherwise return nullptr.
  MCSymbol *getSecondaryEntryPointSymbol(const BinaryBasicBlock &BB) const {
    return getSecondaryEntryPointSymbol(BB.getLabel());
  }

  /// Return true if the basic block is an entry point into the function
  /// (either primary or secondary).
  bool isEntryPoint(const BinaryBasicBlock &BB) const {
    if (&BB == BasicBlocks.front())
      return true;
    return getSecondaryEntryPointSymbol(BB);
  }


  /// Return MC symbol corresponding to an enumerated entry for multiple-entry
  /// functions.
  MCSymbol *getSymbolForEntryID(uint64_t EntryNum);
  const MCSymbol *getSymbolForEntryID(uint64_t EntryNum) const {
    return const_cast<BinaryFunction *>(this)->getSymbolForEntryID(EntryNum);
  }

  using EntryPointCallbackTy = function_ref<bool(uint64_t, const MCSymbol*)>;

  /// Invoke \p Callback function for every entry point in the function starting
  /// with the main entry and using entries in the ascending address order.
  /// Stop calling the function after false is returned by the callback.
  ///
  /// Pass an offset of the entry point in the input binary and a corresponding
  /// global symbol to the callback function.
  ///
  /// Return true of all callbacks returned true, false otherwise.
  bool forEachEntryPoint(EntryPointCallbackTy Callback) const;

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
      std::unique_lock<std::shared_timed_mutex> Lock(BC.CtxMutex);
      FunctionEndLabel = BC.Ctx->createTempSymbol("func_end", true);
    }
    return FunctionEndLabel;
  }

  /// Return MC symbol associated with the end of the cold part of the function.
  MCSymbol *getFunctionColdEndLabel() const {
    if (!FunctionColdEndLabel) {
      std::unique_lock<std::shared_timed_mutex> Lock(BC.CtxMutex);
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

  MCSymbol *getFunctionColdConstantIslandLabel() const {
    if (!FunctionColdConstantIslandLabel) {
      FunctionColdConstantIslandLabel =
          BC.Ctx->createTempSymbol("func_cold_const_island", true);
    }
    return FunctionColdConstantIslandLabel;
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
    IsPseudo = true;
  }

  /// Update output values of the function based on the final \p Layout.
  void updateOutputValues(const MCAsmLayout &Layout);

  /// Return mapping of input to output addresses. Most users should call
  /// translateInputToOutputAddress() for address translation.
  InputOffsetToAddressMapTy &getInputOffsetToAddressMap() {
    assert(isEmitted() && "cannot use address mapping before code emission");
    return InputOffsetToAddressMap;
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
    case ELF::R_X86_64_8:
    case ELF::R_X86_64_16:
    case ELF::R_X86_64_32:
    case ELF::R_X86_64_32S:
    case ELF::R_X86_64_64:
    case ELF::R_AARCH64_ABS64:
    case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
    case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
    case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
    case ELF::R_AARCH64_TLSDESC_LD64_LO12:
    case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
    case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
    case ELF::R_AARCH64_LD64_GOT_LO12_NC:
    case ELF::R_AARCH64_TLSDESC_ADD_LO12:
    case ELF::R_AARCH64_ADD_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
    case ELF::R_AARCH64_ADR_GOT_PAGE:
    case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
    case ELF::R_AARCH64_ADR_PREL_PG_HI21:
      Relocations[Offset] = Relocation{Offset, Symbol, RelType, Addend, Value};
      break;
    case ELF::R_X86_64_PC32:
    case ELF::R_X86_64_PC8:
    case ELF::R_X86_64_PLT32:
    case ELF::R_X86_64_GOTPCRELX:
    case ELF::R_X86_64_REX_GOTPCRELX:
    case ELF::R_AARCH64_JUMP26:
    case ELF::R_AARCH64_CALL26:
    case ELF::R_AARCH64_TLSDESC_CALL:
      break;

    // The following relocations are ignored.
    case ELF::R_X86_64_GOTPCREL:
    case ELF::R_X86_64_TPOFF32:
    case ELF::R_X86_64_GOTTPOFF:
      return;
    default:
      llvm_unreachable("unexpected relocation type in code");
    }

    // FIXME: if we ever find a use for MoveRelocations, this is the place to
    // initialize those:
    //    MoveRelocations[Offset] =
    //      Relocation{Offset, Symbol, RelType, Addend, Value};
  }

  /// Return then name of the section this function originated from.
  StringRef getOriginSectionName() const {
    return getSection().getName();
  }

  /// Return internal section name for this function.
  StringRef getCodeSectionName() const {
    return StringRef(CodeSectionName);
  }

  /// Assign a code section name to the function.
  void setCodeSectionName(StringRef Name) {
    CodeSectionName = Name;
  }

  /// Get output code section.
  ErrorOr<BinarySection &> getCodeSection() const {
    return BC.getUniqueSectionByName(getCodeSectionName());
  }

  /// Return cold code section name for the function.
  StringRef getColdCodeSectionName() const {
    return StringRef(ColdCodeSectionName);
  }

  /// Assign a section name for the cold part of the function.
  void setColdCodeSectionName(StringRef Name) {
    ColdCodeSectionName = Name;
  }

  /// Get output code section for cold code of this function.
  ErrorOr<BinarySection &> getColdCodeSection() const {
    return BC.getUniqueSectionByName(getColdCodeSectionName());
  }

  /// Return true iif the function will halt execution on entry.
  bool trapsOnEntry() const {
    return TrapsOnEntry;
  }

  /// Make the function always trap on entry. Other than the trap instruction,
  /// the function body will be empty.
  void setTrapOnEntry();

  /// Return true if the function could be correctly processed.
  bool isSimple() const {
    return IsSimple;
  }

  /// Return true if the function should be ignored for optimization purposes.
  bool isIgnored() const {
    return IsIgnored;
  }

  /// Return true if the function should not be disassembled, emitted, or
  /// otherwise processed.
  bool isPseudo() const {
    return IsPseudo;
  }

  /// Return true if the original function code has all necessary relocations
  /// to track addresses of functions emitted to new locations.
  bool hasExternalRefRelocations() const {
    return HasExternalRefRelocations;
  }

  /// Return true if the function has instruction(s) with unknown control flow.
  bool hasUnknownControlFlow() const {
    return HasUnknownControlFlow;
  }

  /// Return true if the function body is non-contiguous.
  bool isSplit() const {
    return isSimple() &&
           layout_size() &&
           layout_front()->isCold() != layout_back()->isCold();
  }

  /// Return true if the function has exception handling tables.
  bool hasEHRanges() const {
    return HasEHRanges;
  }

  /// Return true if the function has associated debug info.
  bool hasDebugInfo() const {
    return !SubprogramDIEs.empty();
  }

  /// Return true if the function uses DW_CFA_GNU_args_size CFIs.
  bool usesGnuArgsSize() const {
    return UsesGnuArgsSize;
  }

  /// Return true if the function has more than one entry point.
  bool isMultiEntry() const {
    return !SecondaryEntryPoints.empty();
  }

  /// Return true if the function might have a profile available externally,
  /// but not yet populated into the function.
  bool hasProfileAvailable() const {
    return HasProfileAvailable;
  }

  bool hasMemoryProfile() const {
    return HasMemoryProfile;
  }

  /// Return true if the body of the function was merged into another function.
  bool isFolded() const {
    return FoldedIntoFunction != nullptr;
  }

  /// If this function was folded, return the function it was folded into.
  BinaryFunction *getFoldedIntoFunction() const {
    return FoldedIntoFunction;
  }

  /// Return true if the function uses jump tables.
  bool hasJumpTables() const {
    return !JumpTables.empty();
  }

  /// Return true if the function has SDT marker
  bool hasSDTMarker() const { return HasSDTMarker; }

  /// Return true if the original entry point was patched.
  bool isPatched() const {
    return IsPatched;
  }

  const JumpTable *getJumpTable(const MCInst &Inst) const {
    const auto Address = BC.MIB->getJumpTable(Inst);
    return getJumpTableContainingAddress(Address);
  }

  JumpTable *getJumpTable(const MCInst &Inst) {
    const auto Address = BC.MIB->getJumpTable(Inst);
    return getJumpTableContainingAddress(Address);
  }

  const MCSymbol *getPersonalityFunction() const {
    return PersonalityFunction;
  }

  uint8_t getPersonalityEncoding() const {
    return PersonalityEncoding;
  }

  const std::vector<CallSite> &getCallSites() const {
    return CallSites;
  }

  const std::vector<CallSite> &getColdCallSites() const {
    return ColdCallSites;
  }

  const ArrayRef<uint8_t> getLSDAActionTable() const {
    return LSDAActionTable;
  }

  const std::vector<uint64_t> &getLSDATypeTable() const {
    return LSDATypeTable;
  }

  const ArrayRef<uint8_t> getLSDATypeIndexTable() const {
    return LSDATypeIndexTable;
  }

  const MoveRelocationsTy &getMoveRelocations() const {
    return MoveRelocations;
  }

  const LabelsMapType &getLabels() const {
    return Labels;
  }

  IslandInfo &getIslandInfo() {
    return Islands;
  }

  const IslandInfo &getIslandInfo() const {
    return Islands;
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
  bool containsAddress(uint64_t PC, bool UseMaxSize = false) const {
    if (UseMaxSize)
      return Address <= PC && PC < Address + MaxSize;
    return Address <= PC && PC < Address + Size;
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
      std::unique_lock<std::shared_timed_mutex> Lock(BC.CtxMutex);
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
  BinaryBasicBlock *addBasicBlock(uint64_t Offset, MCSymbol *Label = nullptr,
                                  bool DeriveAlignment = false) {
    assert((CurrentState == State::CFG || !getBasicBlockAtOffset(Offset)) &&
           "basic block already exists in pre-CFG state");

    if (!Label) {
      std::unique_lock<std::shared_timed_mutex> Lock(BC.CtxMutex);
      Label = BC.Ctx->createTempSymbol("BB", true);
    }
    auto BBPtr = createBasicBlock(Offset, Label, DeriveAlignment);
    BasicBlocks.emplace_back(BBPtr.release());

    auto *BB = BasicBlocks.back();
    BB->setIndex(BasicBlocks.size() - 1);

    if (CurrentState == State::Disassembled) {
      BasicBlockOffsets.emplace_back(std::make_pair(Offset, BB));
    } else if (CurrentState == State::CFG) {
      BB->setLayoutIndex(layout_size());
      BasicBlocksLayout.emplace_back(BB);
    }

    assert(CurrentState == State::CFG ||
           (std::is_sorted(BasicBlockOffsets.begin(),
                           BasicBlockOffsets.end(),
                           CompareBasicBlockOffsets()) &&
            std::is_sorted(begin(), end())));

    return BB;
  }

  /// Add basic block \BB as an entry point to the function. Return global
  /// symbol associated with the entry.
  MCSymbol *addEntryPoint(const BinaryBasicBlock &BB);

  /// Mark all blocks that are unreachable from a root (entry point
  /// or landing pad) as invalid.
  void markUnreachableBlocks();

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
    const bool UpdateCFIState = true,
    const bool RecomputeLandingPads = true);

  iterator insertBasicBlocks(
    iterator StartBB,
    std::vector<std::unique_ptr<BinaryBasicBlock>> &&NewBBs,
    const bool UpdateLayout = true,
    const bool UpdateCFIState = true,
    const bool RecomputeLandingPads = true);

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

  /// Return true if we detected ambiguous jump tables in this function, which
  /// happen when one JT is used in more than one indirect jumps. This precludes
  /// us from splitting edges for this JT unless we duplicate the JT (see
  /// disambiguateJumpTables).
  bool checkForAmbiguousJumpTables();

  /// Detect when two distinct indirect jumps are using the same jump table and
  /// duplicate it, allocating a separate JT for each indirect branch. This is
  /// necessary for code transformations on the CFG that change an edge induced
  /// by an indirect branch, e.g.: instrumentation or shrink wrapping. However,
  /// this is only possible if we are not updating jump tables in place, but are
  /// writing it to a new location (moving them).
  void disambiguateJumpTables(MCPlusBuilder::AllocatorIdTy AllocId);

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

  /// We may have built an overly conservative CFG for functions with calls
  /// to functions that the compiler knows will never return. In this case,
  /// clear all successors from these blocks.
  void deleteConservativeEdges();

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

  /// Print all relocations between \p Offset and \p Offset + \p Size in
  /// this function.
  void printRelocations(raw_ostream &OS, uint64_t Offset, uint64_t Size) const;

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

  /// Mark this function as having a valid profile.
  void markProfiled(uint16_t Flags) {
    if (ExecutionCount == COUNT_NO_PROFILE)
      ExecutionCount = 0;
    ProfileFlags = Flags;
    ProfileMatchRatio = 1.0f;
  }

  /// Return flags describing a profile for this function.
  uint16_t getProfileFlags() const {
    return ProfileFlags;
  }

  void addCFIInstruction(uint64_t Offset, MCCFIInstruction &&Inst) {
    assert(!Instructions.empty());

    // Fix CFI instructions skipping NOPs. We need to fix this because changing
    // CFI state after a NOP, besides being wrong and inaccurate,  makes it
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
    while (I != Instructions.begin() && BC.MIB->isNoop(I->second)) {
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
    BC.MIB->createCFI(CFIPseudo, Offset);
    return BB->insertPseudoInstr(Pos, CFIPseudo);
  }

  /// Retrieve the MCCFIInstruction object associated with a CFI pseudo.
  MCCFIInstruction* getCFIFor(const MCInst &Instr) {
    if (!BC.MIB->isCFI(Instr))
      return nullptr;
    uint32_t Offset = Instr.getOperand(0).getImm();
    assert(Offset < FrameInstructions.size() && "Invalid CFI offset");
    return &FrameInstructions[Offset];
  }

  const MCCFIInstruction* getCFIFor(const MCInst &Instr) const {
    if (!BC.MIB->isCFI(Instr))
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

  void setPseudo(bool Pseudo) {
    IsPseudo = Pseudo;
  }

  BinaryFunction &setUsesGnuArgsSize(bool Uses = true) {
    UsesGnuArgsSize = Uses;
    return *this;
  }

  BinaryFunction &setHasProfileAvailable(bool V = true) {
    HasProfileAvailable = V;
    return *this;
  }

  /// Mark function that should not be emitted.
  void setIgnored();

  void setIsPatched(bool V) {
    IsPatched = V;
  }

  void setFolded(BinaryFunction *BF) {
    FoldedIntoFunction = BF;
  }

  BinaryFunction &setPersonalityFunction(uint64_t Addr) {
    assert(!PersonalityFunction && "can't set personality function twice");
    PersonalityFunction = BC.getOrCreateGlobalSymbol(Addr, "FUNCat");
    return *this;
  }

  BinaryFunction &setPersonalityEncoding(uint8_t Encoding) {
    PersonalityEncoding = Encoding;
    return *this;
  }

  BinaryFunction &setAlignment(uint16_t Align) {
    Alignment = Align;
    return *this;
  }

  uint16_t getAlignment() const {
    return Alignment;
  }

  BinaryFunction &setMaxAlignmentBytes(uint16_t MaxAlignBytes) {
    MaxAlignmentBytes = MaxAlignBytes;
    return *this;
  }

  uint16_t getMaxAlignmentBytes() const {
    return MaxAlignmentBytes;
  }

  BinaryFunction &setMaxColdAlignmentBytes(uint16_t MaxAlignBytes) {
    MaxColdAlignmentBytes = MaxAlignBytes;
    return *this;
  }

  uint16_t getMaxColdAlignmentBytes() const {
    return MaxColdAlignmentBytes;
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

  BinaryFunction *getParentFunction() const {
    return ParentFunction;
  }

  /// Set the profile data for the number of times the function was called.
  BinaryFunction &setExecutionCount(uint64_t Count) {
    ExecutionCount = Count;
    return *this;
  }

  /// Adjust execution count for the function by a given \p Count. The value
  /// \p Count will be subtracted from the current function count.
  ///
  /// The function will proportionally adjust execution count for all
  /// basic blocks and edges in the control flow graph.
  void adjustExecutionCount(uint64_t Count);

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

  void setOutputColdDataAddress(uint64_t Address) {
    OutputColdDataOffset = Address;
  }

  uint64_t getOutputColdDataAddress() const {
    return OutputColdDataOffset;
  }

  /// If \p Address represents an access to a constant island managed by this
  /// function, return a symbol so code can safely refer to it. Otherwise,
  /// return nullptr. First return value is the symbol for reference in the
  /// hot code area while the second return value is the symbol for reference
  /// in the cold code area, as when the function is split the islands are
  /// duplicated.
  MCSymbol *getOrCreateIslandAccess(uint64_t Address) {
    MCSymbol *Symbol;
    if (!isInConstantIsland(Address))
      return nullptr;

    // Register our island at global namespace
    Symbol = BC.getOrCreateGlobalSymbol(Address, "ISLANDat");

    // Internal bookkeeping
    const auto Offset = Address - getAddress();
    assert((!Islands.Offsets.count(Offset) ||
            Islands.Offsets[Offset] == Symbol) &&
           "Inconsistent island symbol management");
    if (!Islands.Offsets.count(Offset)) {
      Islands.Offsets[Offset] = Symbol;
      Islands.Symbols.insert(Symbol);
    }
    return Symbol;
  }

  /// Called by an external function which wishes to emit references to constant
  /// island symbols of this function. We create a proxy for it, so we emit
  /// separate symbols when emitting our constant island on behalf of this other
  /// function.
  MCSymbol *
  getOrCreateProxyIslandAccess(uint64_t Address, BinaryFunction &Referrer) {
    auto Symbol = getOrCreateIslandAccess(Address);
    if (!Symbol)
      return nullptr;

    MCSymbol *Proxy;
    if (!Islands.Proxies[&Referrer].count(Symbol)) {
      Proxy =
          BC.Ctx->getOrCreateSymbol(Symbol->getName() +
                                    ".proxy.for." + Referrer.getPrintName());
      Islands.Proxies[&Referrer][Symbol] = Proxy;
      Islands.Proxies[&Referrer][Proxy] = Symbol;
    }
    Proxy = Islands.Proxies[&Referrer][Symbol];
    return Proxy;
  }

  /// Detects whether \p Address is inside a data region in this function
  /// (constant islands).
  bool isInConstantIsland(uint64_t Address) const {
    if (Address < getAddress())
      return false;

    auto Offset = Address - getAddress();

    if (Offset >= getMaxSize())
      return false;

    auto DataIter = Islands.DataOffsets.upper_bound(Offset);
    if (DataIter == Islands.DataOffsets.begin())
      return false;
    DataIter = std::prev(DataIter);

    auto CodeIter = Islands.CodeOffsets.upper_bound(Offset);
    if (CodeIter == Islands.CodeOffsets.begin())
      return true;

    return *std::prev(CodeIter) <= *DataIter;
  }

  uint64_t
  estimateConstantIslandSize(const BinaryFunction *OnBehalfOf = nullptr) const {
    uint64_t Size = 0;
    for (auto DataIter = Islands.DataOffsets.begin();
         DataIter != Islands.DataOffsets.end();
         ++DataIter) {
      auto NextData = std::next(DataIter);
      auto CodeIter = Islands.CodeOffsets.lower_bound(*DataIter);
      if (CodeIter == Islands.CodeOffsets.end() &&
          NextData == Islands.DataOffsets.end()) {
        Size += getMaxSize() - *DataIter;
        continue;
      }

      uint64_t NextMarker;
      if (CodeIter == Islands.CodeOffsets.end())
        NextMarker = *NextData;
      else if (NextData == Islands.DataOffsets.end())
        NextMarker = *CodeIter;
      else
        NextMarker = (*CodeIter > *NextData) ? *NextData : *CodeIter;

      Size += NextMarker - *DataIter;
    }

    if (!OnBehalfOf) {
      for (auto *ExternalFunc : Islands.Dependency)
        Size += ExternalFunc->estimateConstantIslandSize(this);
    }
    return Size;
  }

  bool hasConstantIsland() const {
    return !Islands.DataOffsets.empty();
  }

  /// Return true iff the symbol could be seen inside this function otherwise
  /// it is probably another function.
  bool isSymbolValidInScope(const SymbolRef &Symbol, uint64_t SymbolSize) const;

  /// Disassemble function from raw data.
  /// If successful, this function will populate the list of instructions
  /// for this function together with offsets from the function start
  /// in the input. It will also populate Labels with destinations for
  /// local branches, and TakenBranches with [from, to] info.
  ///
  /// The Function should be properly initialized before this function
  /// is called. I.e. function address and size should be set.
  ///
  /// Returns true on successful disassembly, and updates the current
  /// state to State:Disassembled.
  ///
  /// Returns false if disassembly failed.
  bool disassemble();

  /// Scan function for references to other functions. In relocation mode,
  /// add relocations for external references.
  ///
  /// Return true on success.
  bool scanExternalRefs();

  /// Return the size of a data object located at \p Offset in the function.
  /// Return 0 if there is no data object at the \p Offset.
  size_t getSizeOfDataInCodeAt(uint64_t Offset) const;

  /// Verify that starting at \p Offset function contents are filled with
  /// zero-value bytes.
  bool isZeroPaddingAt(uint64_t Offset) const;

  /// Check that entry points have an associated instruction at their
  /// offsets after disassembly.
  void postProcessEntryPoints();

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
  bool buildCFG(MCPlusBuilder::AllocatorIdTy);

  /// Perform post-processing of the CFG.
  void postProcessCFG();

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
  bool postProcessIndirectBranches(MCPlusBuilder::AllocatorIdTy AllocId);

  /// Return all call site profile info for this function.
  IndirectCallSiteProfile &getAllCallSites() {
    return AllCallSites;
  }

  const IndirectCallSiteProfile &getAllCallSites() const {
    return AllCallSites;
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

  /// Computes a function hotness score: the sum of the products of BB frequency
  /// and size.
  uint64_t getFunctionScore() const;

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

  void moveRememberRestorePair(BinaryBasicBlock *BB);

  bool replayCFIInstrs(int32_t FromState, int32_t ToState,
                       BinaryBasicBlock *InBB,
                       BinaryBasicBlock::iterator InsertIt);

  /// unwindCFIState is used to unwind from a higher to a lower state number
  /// without using remember-restore instructions. We do that by keeping track
  /// of what values have been changed from state A to B and emitting
  /// instructions that undo this change.
  SmallVector<int32_t, 4> unwindCFIState(int32_t FromState, int32_t ToState,
                                         BinaryBasicBlock *InBB,
                                         BinaryBasicBlock::iterator &InsertIt);

  /// After reordering, this function checks the state of CFI and fixes it if it
  /// is corrupted. If it is unable to fix it, it returns false.
  bool finalizeCFIState();

  /// Return true if this function needs an address-transaltion table after
  /// its code emission.
  bool requiresAddressTranslation() const;

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

  void setEmitted(bool KeepCFG = false) {
    CurrentState = State::EmittedCFG;
    if (!KeepCFG) {
      releaseCFG();
      CurrentState = State::Emitted;
    }
  }

  /// Process LSDA information for the function.
  void parseLSDA(ArrayRef<uint8_t> LSDAData, uint64_t LSDAAddress);

  /// Update exception handling ranges for the function.
  void updateEHRanges();

  /// Traverse cold basic blocks and replace references to constants in islands
  /// with a proxy symbol for the duplicated constant island that is going to be
  /// emitted in the cold region.
  void duplicateConstantIslands();

  /// Merge profile data of this function into those of the given
  /// function. The functions should have been proven identical with
  /// isIdenticalWith.
  void mergeProfileDataInto(BinaryFunction &BF) const;

  /// Returns the last computed hash value of the function.
  size_t getHash() const {
    return Hash;
  }

  using OperandHashFuncTy =
    function_ref<typename std::string(const MCOperand&)>;

  /// Compute the hash value of the function based on its contents.
  ///
  /// If \p UseDFS is set, process basic blocks in DFS order. Otherwise, use
  /// the existing layout order.
  ///
  /// By default, instruction operands are ignored while calculating the hash.
  /// The caller can change this via passing \p OperandHashFunc function.
  /// The return result of this function will be mixed with internal hash.
  size_t computeHash(bool UseDFS = false,
                     OperandHashFuncTy OperandHashFunc =
                       [](const MCOperand&) { return std::string(); }) const;

  /// Sets the associated .debug_info entry.
  void addSubprogramDIE(const DWARFDie DIE) {
    static std::mutex CriticalSectionMutex;
    std::lock_guard<std::mutex> Lock(CriticalSectionMutex);
    SubprogramDIEs.emplace_back(DIE);
  }

  void setDWARFUnitLineTable(DWARFUnit *Unit,
                             const DWARFDebugLine::LineTable *Table) {
    UnitLineTable = std::make_pair(Unit, Table);
  }

  /// Return all compilation units with entry for this function.
  /// Because of identical code folding there could be multiple of these.
  decltype(SubprogramDIEs) &getSubprogramDIEs() {
    return SubprogramDIEs;
  }

  const decltype(SubprogramDIEs) &getSubprogramDIEs() const {
    return SubprogramDIEs;
  }

  /// Return DWARF compile unit with line info for this function.
  DWARFUnitLineTable &getDWARFUnitLineTable() {
    return UnitLineTable;
  }

  const DWARFUnitLineTable &getDWARFUnitLineTable() const {
    return UnitLineTable;
  }

  /// Finalize profile for the function.
  void postProcessProfile();

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
  DebugAddressRangesVector getOutputAddressRanges() const;

  /// Given an address corresponding to an instruction in the input binary,
  /// return an address of this instruction in output binary.
  ///
  /// Return 0 if no matching address could be found or the instruction was
  /// removed.
  uint64_t translateInputToOutputAddress(uint64_t Address) const;

  /// Take address ranges corresponding to the input binary and translate
  /// them to address ranges in the output binary.
  DebugAddressRangesVector translateInputToOutputRanges(
      const DWARFAddressRangesVector &InputRanges) const;

  /// Similar to translateInputToOutputRanges() but operates on location lists
  /// and moves associated data to output location lists.
  ///
  /// \p BaseAddress is applied to all addresses in \pInputLL.
  DWARFDebugLoc::LocationList translateInputToOutputLocationList(
      DWARFDebugLoc::LocationList InputLL) const;

  /// Return true if the function is an AArch64 linker inserted veneer
  bool isAArch64Veneer() const;

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
  case BinaryFunction::State::EmittedCFG:   OS << "emitted with CFG";  break;
  case BinaryFunction::State::Emitted:      OS << "emitted";  break;
  }

  return OS;
}

} // namespace bolt


// GraphTraits specializations for function basic block graphs (CFGs)
template <> struct GraphTraits<bolt::BinaryFunction *> :
  public GraphTraits<bolt::BinaryBasicBlock *> {
  static NodeRef getEntryNode(bolt::BinaryFunction *F) {
    return *F->layout_begin();
  }

  using nodes_iterator = pointer_iterator<bolt::BinaryFunction::iterator>;

  static nodes_iterator nodes_begin(bolt::BinaryFunction *F) {
    llvm_unreachable("Not implemented");
    return nodes_iterator(F->begin());
  }
  static nodes_iterator nodes_end(bolt::BinaryFunction *F) {
    llvm_unreachable("Not implemented");
    return nodes_iterator(F->end());
  }
  static size_t size(bolt::BinaryFunction *F) {
    return F->size();
  }
};

template <> struct GraphTraits<const bolt::BinaryFunction *> :
  public GraphTraits<const bolt::BinaryBasicBlock *> {
  static NodeRef getEntryNode(const bolt::BinaryFunction *F) {
    return *F->layout_begin();
  }

  using nodes_iterator = pointer_iterator<bolt::BinaryFunction::const_iterator>;

  static nodes_iterator nodes_begin(const bolt::BinaryFunction *F) {
    llvm_unreachable("Not implemented");
    return nodes_iterator(F->begin());
  }
  static nodes_iterator nodes_end(const bolt::BinaryFunction *F) {
    llvm_unreachable("Not implemented");
    return nodes_iterator(F->end());
  }
  static size_t size(const bolt::BinaryFunction *F) {
    return F->size();
  }
};

template <> struct GraphTraits<Inverse<bolt::BinaryFunction *>> :
  public GraphTraits<Inverse<bolt::BinaryBasicBlock *>> {
  static NodeRef getEntryNode(Inverse<bolt::BinaryFunction *> G) {
    return *G.Graph->layout_begin();
  }
};

template <> struct GraphTraits<Inverse<const bolt::BinaryFunction *>> :
  public GraphTraits<Inverse<const bolt::BinaryBasicBlock *>> {
  static NodeRef getEntryNode(Inverse<const bolt::BinaryFunction *> G) {
    return *G.Graph->layout_begin();
  }
};

} // namespace llvm

#endif
