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

#ifndef LLVM_TOOLS_LLVM_FLO_BINARY_FUNCTION_H
#define LLVM_TOOLS_LLVM_FLO_BINARY_FUNCTION_H

#include "BinaryBasicBlock.h"
#include "BinaryContext.h"
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
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <map>

using namespace llvm::object;

namespace llvm {
namespace flo {

/// BinaryFunction is a representation of machine-level function.
//
/// We use the term "Binary" as "Machine" was already taken.
class BinaryFunction {
public:
  enum class State : char {
    Empty = 0,        /// Function body is empty
    Disassembled,     /// Function have been disassembled
    CFG,              /// Control flow graph have been built
    Assembled,        /// Function has been assembled in memory
  };

  // Choose which strategy should the block layout heuristic prioritize when
  // facing conflicting goals.
  enum HeuristicPriority : char {
    HP_NONE = 0,
    // HP_BRANCH_PREDICTOR is an implementation of what is suggested in Pettis'
    // paper (PLDI '90) about block reordering, trying to minimize branch
    // mispredictions.
    HP_BRANCH_PREDICTOR,
    // HP_CACHE_UTILIZATION pigbacks on the idea from Ispike paper (CGO '04)
    // that suggests putting frequently executed chains first in the layout.
    HP_CACHE_UTILIZATION,
  };

  static constexpr uint64_t COUNT_NO_PROFILE =
    std::numeric_limits<uint64_t>::max();
  // Function size, in number of BBs, above which we fallback to a heuristic
  // solution to the layout problem instead of seeking the optimal one.
  static constexpr uint64_t FUNC_SIZE_THRESHOLD = 10;

private:

  /// Current state of the function.
  State CurrentState{State::Empty};

  /// Name of the function as we know it.
  std::string Name;

  /// Symbol associated with this function.
  SymbolRef Symbol;

  /// Containing section
  SectionRef Section;

  /// Address of the function in memory. Also could be an offset from
  /// base address for position independent binaries.
  uint64_t Address;

  /// Original size of the function.
  uint64_t Size;

  /// Offset in the file.
  uint64_t FileOffset{0};

  /// Maximum size this function is allowed to have.
  uint64_t MaxSize{std::numeric_limits<uint64_t>::max()};

  /// Alignment requirements for the function.
  uint64_t Alignment{1};

  /// False if the function is too complex to reconstruct its control
  /// flow graph and re-assemble.
  bool IsSimple{true};

  BinaryContext &BC;

  /// The address for the code for this function in codegen memory.
  uint64_t ImageAddress{0};

  /// The size of the code in memory.
  uint64_t ImageSize{0};

  /// Name for the section this function code should reside in.
  std::string CodeSectionName;

  /// The profile data for the number of times the function was executed.
  uint64_t ExecutionCount{COUNT_NO_PROFILE};

  /// Binary blob reprsenting action, type, and type index tables for this
  /// function' LSDA (exception handling).
  ArrayRef<uint8_t> LSDATables;

  /// Original LSDA address for the function.
  uint64_t LSDAAddress{0};

  /// Landing pads for the function.
  std::set<MCSymbol *> LandingPads;

  /// Release storage used by instructions.
  BinaryFunction &clearInstructions() {
    InstrMapType TempMap;
    Instructions.swap(TempMap);
    return *this;
  }

  /// Release storage used by instructions.
  BinaryFunction &clearLabels() {
    LabelsMapType TempMap;
    Labels.swap(TempMap);
    return *this;
  }

  /// Release memory taken by local branch info.
  BinaryFunction &clearLocalBranches() {
    LocalBranchesListType TempList;
    LocalBranches.swap(TempList);
    return *this;
  }

  BinaryFunction &clearFTBranches() {
    LocalBranchesListType TempList;
    FTBranches.swap(TempList);
    return *this;
  }

  BinaryFunction &updateState(BinaryFunction::State State) {
    CurrentState = State;
    return *this;
  }

  const BinaryBasicBlock *
  getOriginalLayoutSuccessor(const BinaryBasicBlock *BB) const;

  /// Storage for all local branches in the function (non-fall-throughs).
  using LocalBranchesListType = std::vector<std::pair<uint32_t, uint32_t>>;
  LocalBranchesListType LocalBranches;
  LocalBranchesListType FTBranches;

  /// Map offset in the function to a local label.
  using LabelsMapType = std::map<uint32_t, MCSymbol *>;
  LabelsMapType Labels;

  /// Temporary holder of instructions before CFG is constructed.
  /// Map offset in the function to MCInst.
  using InstrMapType = std::map<uint32_t, MCInst>;
  InstrMapType Instructions;

  /// List of DWARF CFI instructions
  using CFIInstrMapType = std::multimap<uint32_t, MCCFIInstruction>;
  CFIInstrMapType FrameInstructions;

  // Blocks are kept sorted in the layout order. If we need to change the
  // layout (if BasicBlocksLayout stores a different order than BasicBlocks),
  // the terminating instructions need to be modified.
  using BasicBlockListType = std::vector<BinaryBasicBlock>;
  using BasicBlockOrderType = std::vector<BinaryBasicBlock*>;
  BasicBlockListType BasicBlocks;
  BasicBlockOrderType BasicBlocksLayout;

public:

  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;
  typedef BasicBlockOrderType::iterator order_iterator;
  typedef BasicBlockOrderType::const_iterator const_order_iterator;

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
  const BinaryBasicBlock &front() const  { return BasicBlocks.front(); }
        BinaryBasicBlock &front()        { return BasicBlocks.front(); }
  const BinaryBasicBlock & back() const  { return BasicBlocks.back(); }
        BinaryBasicBlock & back()        { return BasicBlocks.back(); }

  unsigned layout_size() const {
    return (unsigned)BasicBlocksLayout.size();
  }
  const_order_iterator layout_begin() const {
    return BasicBlocksLayout.begin();
  }
  order_iterator layout_begin() { return BasicBlocksLayout.begin(); }

  inline iterator_range<order_iterator> layout() {
    return iterator_range<order_iterator>(BasicBlocksLayout.begin(),
                                          BasicBlocksLayout.end());
  }

  BinaryFunction(std::string Name, SymbolRef Symbol, SectionRef Section,
                 uint64_t Address, uint64_t Size, BinaryContext &BC) :
      Name(Name), Symbol(Symbol), Section(Section), Address(Address),
      Size(Size), BC(BC), CodeSectionName(".text." + Name) {}

  /// Perform optimal code layout based on edge frequencies making necessary
  /// adjustments to instructions at the end of basic blocks.
  void optimizeLayout(HeuristicPriority Priority);

  /// Dynamic programming implementation for the TSP, applied to BB layout. Find
  /// the optimal way to maximize weight during a path traversing all BBs. In
  /// this way, we will convert the hottest branches into fall-throughs.
  ///
  /// Uses exponential amount of memory on the number of basic blocks and should
  /// only be used for small functions.
  void solveOptimalLayout();

  /// View CFG in graphviz program
  void viewGraph();

  /// Basic block iterator

  /// Return the name of the function as extracted from the binary file.
  StringRef getName() const {
    return Name;
  }

  /// Return symbol associated with the function start.
  SymbolRef getSymbol() const {
    return Symbol;
  }

  /// Return containing file section.
  SectionRef getSection() const {
    return Section;
  }

  /// Return original address of the function (or offset from base for PIC).
  uint64_t getAddress() const {
    return Address;
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

  /// Return internal section name for this function.
  StringRef getCodeSectionName() const {
    assert(!CodeSectionName.empty() && "no section name for function");
    return StringRef(CodeSectionName);
  }

  /// Return true if the function could be correctly processed.
  bool isSimple() const {
    return IsSimple;
  }

  /// Return true if the given address \p PC is inside the function body.
  bool containsAddress(uint64_t PC) const {
    return Address <= PC && PC < Address + Size;
  }

  /// Create a basic block at a given \p Offset in the
  /// function and append it to the end of list of blocks.
  /// If \p DeriveAlignment is true, set the alignment of the block based
  /// on the alignment of the existing offset.
  /// 
  /// Returns NULL if basic block already exists at the \p Offset.
  BinaryBasicBlock *addBasicBlock(uint64_t Offset, MCSymbol *Label,
                                  bool DeriveAlignment = false) {
    assert(!getBasicBlockAtOffset(Offset) && "basic block already exists");
    if (!Label)
      Label = BC.Ctx->createTempSymbol("BB", true);
    BasicBlocks.emplace_back(BinaryBasicBlock(Label, Offset));

    auto BB = &BasicBlocks.back();

    if (DeriveAlignment) {
      uint64_t DerivedAlignment = Offset & (1 + ~Offset);
      BB->setAlignment(std::min(DerivedAlignment, uint64_t(16)));
    }

    return BB;
  }

  /// Rebuilds BBs layout, ignoring dead BBs. Returns the number of removed
  /// BBs.
  unsigned eraseDeadBBs(std::map<BinaryBasicBlock *, bool> &ToPreserve);

  /// Return basic block that started at offset \p Offset.
  BinaryBasicBlock *getBasicBlockAtOffset(uint64_t Offset) {
    BinaryBasicBlock *BB = getBasicBlockContainingOffset(Offset);
    if (BB && BB->Offset == Offset)
      return BB;

    return nullptr;
  }

  /// Return basic block that originally contained offset \p Offset
  /// from the function start.
  BinaryBasicBlock *getBasicBlockContainingOffset(uint64_t Offset);

  /// Dump function information to debug output. If \p PrintInstructions
  /// is true - include instruction disassembly.
  void dump(std::string Annotation = "", bool PrintInstructions = true) const {
    print(dbgs(), Annotation, PrintInstructions);
  }

  /// Print function information to the \p OS stream.
  void print(raw_ostream &OS, std::string Annotation = "",
             bool PrintInstructions = true) const;

  void addInstruction(uint64_t Offset, MCInst &&Instruction) {
    Instructions.emplace(Offset, std::forward<MCInst>(Instruction));
  }

  void addCFIInstruction(uint64_t Offset, MCCFIInstruction &&Inst) {
    FrameInstructions.emplace(Offset, std::forward<MCCFIInstruction>(Inst));
  }

  BinaryFunction &setFileOffset(uint64_t Offset) {
    FileOffset = Offset;
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

  /// Disassemble function from raw data \p FunctionData.
  /// If successful, this function will populate the list of instructions
  /// for this function together with offsets from the function start
  /// in the input. It will also populate Labels with destinations for
  /// local branches, and LocalBranches with [from, to] info.
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

  /// Walks the list of basic blocks filling in missing information about
  /// edge frequency for fall-throughs.
  ///
  /// Assumes the CFG has been built and edge frequency for taken branches
  /// has been filled with LBR data.
  void inferFallThroughCounts();

  /// Traverse the CFG checking branches, inverting their condition, removing or
  /// adding jumps based on a new layout order.
  void fixBranches();

  /// Process LSDA information for the function.
  void parseLSDA(ArrayRef<uint8_t> LSDAData, uint64_t LSDAAddress);

  virtual ~BinaryFunction() {}
};

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

} // namespace flo
} // namespace llvm

#endif
