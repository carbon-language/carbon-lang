//===--- BinaryFunction.cpp - Interface for machine-level function --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//


#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "DataReader.h"
#include "Passes/MCF.h"
#include "Passes/ReorderAlgorithm.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <queue>
#include <string>
#include <functional>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltCategory;
extern cl::OptionCategory BoltOptCategory;
extern cl::OptionCategory BoltRelocCategory;

extern bool shouldProcess(const BinaryFunction &);

extern cl::opt<bool> Relocs;
extern cl::opt<bool> UpdateDebugSections;
extern cl::opt<IndirectCallPromotionType> IndirectCallPromotion;
extern cl::opt<unsigned> Verbosity;
extern cl::opt<unsigned> PrintFuncStat;

static cl::opt<bool>
AggressiveSplitting("split-all-cold",
  cl::desc("outline as many cold basic blocks as possible"),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
AlignBlocks("align-blocks",
  cl::desc("try to align BBs inserting nops"),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
DotToolTipCode("dot-tooltip-code",
  cl::desc("add basic block instructions as tool tips on nodes"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<uint32_t>
DynoStatsScale("dyno-stats-scale",
  cl::desc("scale to be applied while reporting dyno stats"),
  cl::Optional,
  cl::init(1),
  cl::Hidden,
  cl::cat(BoltCategory));

cl::opt<JumpTableSupportLevel>
JumpTables("jump-tables",
  cl::desc("jump tables support (default=basic)"),
  cl::init(JTS_BASIC),
  cl::values(
      clEnumValN(JTS_NONE, "none",
                 "do not optimize functions with jump tables"),
      clEnumValN(JTS_BASIC, "basic",
                 "optimize functions with jump tables"),
      clEnumValN(JTS_MOVE, "move",
                 "move jump tables to a separate section"),
      clEnumValN(JTS_SPLIT, "split",
                 "split jump tables section into hot and cold based on "
                 "function execution frequency"),
      clEnumValN(JTS_AGGRESSIVE, "aggressive",
                 "aggressively split jump tables section based on usage "
                 "of the tables"),
      clEnumValEnd),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<bool>
PrintDynoStats("dyno-stats",
  cl::desc("print execution info based on profile"),
  cl::cat(BoltCategory));

static cl::opt<bool>
PrintJumpTables("print-jump-tables",
  cl::desc("print jump tables"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::list<std::string>
PrintOnly("print-only",
  cl::CommaSeparated,
  cl::desc("list of functions to print"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<bool>
SplitEH("split-eh",
  cl::desc("split C++ exception handling code (experimental)"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

cl::opt<MCFCostFunction>
DoMCF("mcf",
  cl::desc("solve a min cost flow problem on the CFG to fix edge counts "
           "(default=disable)"),
  cl::init(MCF_DISABLE),
  cl::values(
    clEnumValN(MCF_DISABLE, "none",
               "disable MCF"),
    clEnumValN(MCF_LINEAR, "linear",
               "cost function is inversely proportional to edge count"),
    clEnumValN(MCF_QUADRATIC, "quadratic",
               "cost function is inversely proportional to edge count squared"),
    clEnumValN(MCF_LOG, "log",
               "cost function is inversely proportional to log of edge count"),
    clEnumValN(MCF_BLAMEFTS, "blamefts",
               "tune cost to blame fall-through edges for surplus flow"),
    clEnumValEnd),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

bool shouldPrint(const BinaryFunction &Function) {
  if (PrintOnly.empty())
    return true;

  for (auto &Name : opts::PrintOnly) {
    if (Function.hasName(Name)) {
      return true;
    }
  }

  return false;
}

} // namespace opts

namespace llvm {
namespace bolt {

// Temporary constant.
//
// TODO: move to architecture-specific file together with the code that is
// using it.
constexpr unsigned NoRegister = 0;

constexpr const char *DynoStats::Desc[];
constexpr unsigned BinaryFunction::MinAlign;

namespace {

/// Gets debug line information for the instruction located at the given
/// address in the original binary. The SMLoc's pointer is used
/// to point to this information, which is represented by a
/// DebugLineTableRowRef. The returned pointer is null if no debug line
/// information for this instruction was found.
SMLoc findDebugLineInformationForInstructionAt(
    uint64_t Address,
    DWARFUnitLineTable &ULT) {
  // We use the pointer in SMLoc to store an instance of DebugLineTableRowRef,
  // which occupies 64 bits. Thus, we can only proceed if the struct fits into
  // the pointer itself.
  assert(
      sizeof(decltype(SMLoc().getPointer())) >= sizeof(DebugLineTableRowRef) &&
      "Cannot fit instruction debug line information into SMLoc's pointer");

  SMLoc NullResult = DebugLineTableRowRef::NULL_ROW.toSMLoc();

  auto &LineTable = ULT.second;
  if (!LineTable)
    return NullResult;

  uint32_t RowIndex = LineTable->lookupAddress(Address);
  if (RowIndex == LineTable->UnknownRowIndex)
    return NullResult;

  assert(RowIndex < LineTable->Rows.size() &&
         "Line Table lookup returned invalid index.");

  decltype(SMLoc().getPointer()) Ptr;
  DebugLineTableRowRef *InstructionLocation =
    reinterpret_cast<DebugLineTableRowRef *>(&Ptr);

  InstructionLocation->DwCompileUnitIndex = ULT.first->getOffset();
  InstructionLocation->RowIndex = RowIndex + 1;

  return SMLoc::getFromPointer(Ptr);
}

} // namespace

bool DynoStats::operator<(const DynoStats &Other) const {
  return std::lexicographical_compare(
    &Stats[FIRST_DYNO_STAT], &Stats[LAST_DYNO_STAT],
    &Other.Stats[FIRST_DYNO_STAT], &Other.Stats[LAST_DYNO_STAT]
  );
}

bool DynoStats::operator==(const DynoStats &Other) const {
  return std::equal(
    &Stats[FIRST_DYNO_STAT], &Stats[LAST_DYNO_STAT],
    &Other.Stats[FIRST_DYNO_STAT]
  );
}

bool DynoStats::lessThan(const DynoStats &Other,
                         ArrayRef<Category> Keys) const {
  return std::lexicographical_compare(
    Keys.begin(), Keys.end(),
    Keys.begin(), Keys.end(),
    [this,&Other](const Category A, const Category) {
      return Stats[A] < Other.Stats[A];
    }
  );
}

uint64_t BinaryFunction::Count = 0;

BinaryBasicBlock *
BinaryFunction::getBasicBlockContainingOffset(uint64_t Offset) {
  if (Offset > Size)
    return nullptr;

  if (BasicBlockOffsets.empty())
    return nullptr;

  /*
   * This is commented out because it makes BOLT too slow.
   * assert(std::is_sorted(BasicBlockOffsets.begin(),
   *                       BasicBlockOffsets.end(),
   *                       CompareBasicBlockOffsets())));
   */
  auto I = std::upper_bound(BasicBlockOffsets.begin(),
                            BasicBlockOffsets.end(),
                            BasicBlockOffset(Offset, nullptr),
                            CompareBasicBlockOffsets());
  assert(I != BasicBlockOffsets.begin() && "first basic block not at offset 0");
  --I;
  auto *BB = I->second;
  return (Offset < BB->getOffset() + BB->getOriginalSize()) ? BB : nullptr;
}

void BinaryFunction::markUnreachable() {
  std::stack<BinaryBasicBlock *> Stack;

  for (auto *BB : layout()) {
    BB->markValid(false);
  }

  // Add all entries and landing pads as roots.
  for (auto *BB : BasicBlocks) {
    if (BB->isEntryPoint() || BB->isLandingPad()) {
      Stack.push(BB);
      BB->markValid(true);
    }
  }

  // Determine reachable BBs from the entry point
  while (!Stack.empty()) {
    auto BB = Stack.top();
    Stack.pop();
    for (auto Succ : BB->successors()) {
      if (Succ->isValid())
        continue;
      Succ->markValid(true);
      Stack.push(Succ);
    }
  }
}

// Any unnecessary fallthrough jumps revealed after calling eraseInvalidBBs
// will be cleaned up by fixBranches().
std::pair<unsigned, uint64_t> BinaryFunction::eraseInvalidBBs() {
  BasicBlockOrderType NewLayout;
  unsigned Count = 0;
  uint64_t Bytes = 0;
  for (auto *BB : layout()) {
    assert((!BB->isEntryPoint() || BB->isValid()) &&
           "all entry blocks must be valid");
    if (BB->isValid()) {
      NewLayout.push_back(BB);
    } else {
      ++Count;
      Bytes += BC.computeCodeSize(BB->begin(), BB->end());
    }
  }
  BasicBlocksLayout = std::move(NewLayout);

  BasicBlockListType NewBasicBlocks;
  for (auto I = BasicBlocks.begin(), E = BasicBlocks.end(); I != E; ++I) {
    if ((*I)->isValid()) {
      NewBasicBlocks.push_back(*I);
    } else {
      DeletedBasicBlocks.push_back(*I);
    }
  }
  BasicBlocks = std::move(NewBasicBlocks);

  assert(BasicBlocks.size() == BasicBlocksLayout.size());

  // Update CFG state if needed
  if (Count > 0) {
    updateBBIndices(0);
    recomputeLandingPads(0, BasicBlocks.size());
  }

  return std::make_pair(Count, Bytes);
}

bool BinaryFunction::isForwardCall(const MCSymbol *CalleeSymbol) const {
  // This function should work properly before and after function reordering.
  // In order to accomplish this, we use the function index (if it is valid).
  // If the function indices are not valid, we fall back to the original
  // addresses.  This should be ok because the functions without valid indices
  // should have been ordered with a stable sort.
  const auto *CalleeBF = BC.getFunctionForSymbol(CalleeSymbol);
  if (CalleeBF) {
    if (hasValidIndex() && CalleeBF->hasValidIndex()) {
      return getIndex() < CalleeBF->getIndex();
    } else if (hasValidIndex() && !CalleeBF->hasValidIndex()) {
      return true;
    } else if (!hasValidIndex() && CalleeBF->hasValidIndex()) {
      return false;
    } else {
      return getAddress() < CalleeBF->getAddress();
    }
  } else {
    // Absolute symbol.
    auto const CalleeSI = BC.GlobalSymbols.find(CalleeSymbol->getName());
    assert(CalleeSI != BC.GlobalSymbols.end() && "unregistered symbol found");
    return CalleeSI->second > getAddress();
  }
}

void BinaryFunction::dump(bool PrintInstructions) const {
  print(dbgs(), "", PrintInstructions);
}

void BinaryFunction::print(raw_ostream &OS, std::string Annotation,
                           bool PrintInstructions) const {
  // FIXME: remove after #15075512 is done.
  if (!opts::shouldProcess(*this) || !opts::shouldPrint(*this))
    return;

  StringRef SectionName;
  Section.getName(SectionName);
  OS << "Binary Function \"" << *this << "\" " << Annotation << " {";
  if (Names.size() > 1) {
    OS << "\n  Other names : ";
    auto Sep = "";
    for (unsigned i = 0; i < Names.size() - 1; ++i) {
      OS << Sep << Names[i];
      Sep = "\n                ";
    }
  }
  OS << "\n  Number      : "   << FunctionNumber
     << "\n  State       : "   << CurrentState
     << "\n  Address     : 0x" << Twine::utohexstr(Address)
     << "\n  Size        : 0x" << Twine::utohexstr(Size)
     << "\n  MaxSize     : 0x" << Twine::utohexstr(MaxSize)
     << "\n  Offset      : 0x" << Twine::utohexstr(FileOffset)
     << "\n  Section     : "   << SectionName
     << "\n  Orc Section : "   << getCodeSectionName()
     << "\n  LSDA        : 0x" << Twine::utohexstr(getLSDAAddress())
     << "\n  IsSimple    : "   << IsSimple
     << "\n  IsSplit     : "   << IsSplit
     << "\n  BB Count    : "   << BasicBlocksLayout.size();

  if (hasCFG()) {
    OS << "\n  Hash        : "   << Twine::utohexstr(hash());
  }
  if (FrameInstructions.size()) {
    OS << "\n  CFI Instrs  : "   << FrameInstructions.size();
  }
  if (BasicBlocksLayout.size()) {
    OS << "\n  BB Layout   : ";
    auto Sep = "";
    for (auto BB : BasicBlocksLayout) {
      OS << Sep << BB->getName();
      Sep = ", ";
    }
  }
  if (ImageAddress)
    OS << "\n  Image       : 0x" << Twine::utohexstr(ImageAddress);
  if (ExecutionCount != COUNT_NO_PROFILE) {
    OS << "\n  Exec Count  : " << ExecutionCount;
    OS << "\n  Profile Acc : " << format("%.1f%%", ProfileMatchRatio * 100.0f);
  }

  if (opts::PrintDynoStats && !BasicBlocksLayout.empty()) {
    OS << '\n';
    DynoStats dynoStats = getDynoStats();
    OS << dynoStats;
  }

  OS << "\n}\n";

  if (!PrintInstructions || !BC.InstPrinter)
    return;

  // Offset of the instruction in function.
  uint64_t Offset{0};

  if (BasicBlocks.empty() && !Instructions.empty()) {
    // Print before CFG was built.
    for (const auto &II : Instructions) {
      Offset = II.first;

      // Print label if exists at this offset.
      auto LI = Labels.find(Offset);
      if (LI != Labels.end())
        OS << LI->second->getName() << ":\n";

      BC.printInstruction(OS, II.second, Offset, this);
    }
  }

  for (uint32_t I = 0, E = BasicBlocksLayout.size(); I != E; ++I) {
    auto BB = BasicBlocksLayout[I];
    if (I != 0 &&
        BB->isCold() != BasicBlocksLayout[I - 1]->isCold())
      OS << "-------   HOT-COLD SPLIT POINT   -------\n\n";

    OS << BB->getName() << " ("
       << BB->size() << " instructions, align : " << BB->getAlignment()
       << ")\n";

    if (BB->isEntryPoint())
      OS << "  Entry Point\n";

    if (BB->isLandingPad())
      OS << "  Landing Pad\n";

    uint64_t BBExecCount = BB->getExecutionCount();
    if (hasValidProfile()) {
      OS << "  Exec Count : " << BBExecCount << "\n";
    }
    if (BB->getCFIState() >= 0) {
      OS << "  CFI State : " << BB->getCFIState() << '\n';
    }
    if (!BB->pred_empty()) {
      OS << "  Predecessors: ";
      auto Sep = "";
      for (auto Pred : BB->predecessors()) {
        OS << Sep << Pred->getName();
        Sep = ", ";
      }
      OS << '\n';
    }
    if (!BB->throw_empty()) {
      OS << "  Throwers: ";
      auto Sep = "";
      for (auto Throw : BB->throwers()) {
        OS << Sep << Throw->getName();
        Sep = ", ";
      }
      OS << '\n';
    }

    Offset = RoundUpToAlignment(Offset, BB->getAlignment());

    // Note: offsets are imprecise since this is happening prior to relaxation.
    Offset = BC.printInstructions(OS, BB->begin(), BB->end(), Offset, this);

    if (!BB->succ_empty()) {
      OS << "  Successors: ";
      auto BI = BB->branch_info_begin();
      auto Sep = "";
      for (auto Succ : BB->successors()) {
        assert(BI != BB->branch_info_end() && "missing BranchInfo entry");
        OS << Sep << Succ->getName();
        if (ExecutionCount != COUNT_NO_PROFILE &&
            BI->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED) {
          OS << " (mispreds: " << BI->MispredictedCount
             << ", count: " << BI->Count << ")";
        } else if (ExecutionCount != COUNT_NO_PROFILE &&
                   BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE) {
          OS << " (inferred count: " << BI->Count << ")";
        }
        Sep = ", ";
        ++BI;
      }
      OS << '\n';
    }

    if (!BB->lp_empty()) {
      OS << "  Landing Pads: ";
      auto Sep = "";
      for (auto LP : BB->landing_pads()) {
        OS << Sep << LP->getName();
        if (ExecutionCount != COUNT_NO_PROFILE) {
          OS << " (count: " << LP->getExecutionCount() << ")";
        }
        Sep = ", ";
      }
      OS << '\n';
    }

    // In CFG_Finalized state we can miscalculate CFI state at exit.
    if (CurrentState == State::CFG) {
      const auto CFIStateAtExit = BB->getCFIStateAtExit();
      if (CFIStateAtExit >= 0)
        OS << "  CFI State: " << CFIStateAtExit << '\n';
    }

    OS << '\n';
  }

  // Dump new exception ranges for the function.
  if (!CallSites.empty()) {
    OS << "EH table:\n";
    for (auto &CSI : CallSites) {
      OS << "  [" << *CSI.Start << ", " << *CSI.End << ") landing pad : ";
      if (CSI.LP)
        OS << *CSI.LP;
      else
        OS << "0";
      OS << ", action : " << CSI.Action << '\n';
    }
    OS << '\n';
  }

  // Print all jump tables.
  for (auto &JTI : JumpTables) {
    JTI.second.print(OS);
  }

  OS << "DWARF CFI Instructions:\n";
  if (OffsetToCFI.size()) {
    // Pre-buildCFG information
    for (auto &Elmt : OffsetToCFI) {
      OS << format("    %08x:\t", Elmt.first);
      assert(Elmt.second < FrameInstructions.size() && "Incorrect CFI offset");
      BinaryContext::printCFI(OS, FrameInstructions[Elmt.second]);
      OS << "\n";
    }
  } else {
    // Post-buildCFG information
    for (uint32_t I = 0, E = FrameInstructions.size(); I != E; ++I) {
      const MCCFIInstruction &CFI = FrameInstructions[I];
      OS << format("    %d:\t", I);
      BinaryContext::printCFI(OS, CFI);
      OS << "\n";
    }
  }
  if (FrameInstructions.empty())
    OS << "    <empty>\n";

  OS << "End of Function \"" << *this << "\"\n\n";
}

BinaryFunction::IndirectBranchType
BinaryFunction::analyzeIndirectBranch(MCInst &Instruction,
                                      unsigned Size,
                                      uint64_t Offset) {
  auto &MIA = BC.MIA;

  IndirectBranchType Type = IndirectBranchType::UNKNOWN;

  // An instruction referencing memory used by jump instruction (directly or
  // via register). This location could be an array of function pointers
  // in case of indirect tail call, or a jump table.
  MCInst *MemLocInstr = nullptr;

  // Address of the table referenced by MemLocInstr. Could be either an
  // array of function pointers, or a jump table.
  uint64_t ArrayStart = 0;

  auto analyzePICJumpTable =
      [&](InstrMapType::reverse_iterator II,
          InstrMapType::reverse_iterator IE,
          unsigned R1,
          unsigned R2) {
    // Analyze PIC-style jump table code template:
    //
    //    lea PIC_JUMP_TABLE(%rip), {%r1|%r2}     <- MemLocInstr
    //    mov ({%r1|%r2}, %index, 4), {%r2|%r1}
    //    add %r2, %r1
    //    jmp *%r1
    //
    // (with any irrelevant instructions in-between)
    //
    // When we call this helper we've already determined %r1 and %r2, and
    // reverse instruction iterator \p II is pointing to the ADD instruction.
    //
    // PIC jump table looks like following:
    //
    //   JT:  ----------
    //    E1:| L1 - JT  |
    //       |----------|
    //    E2:| L2 - JT  |
    //       |----------|
    //       |          |
    //          ......
    //    En:| Ln - JT  |
    //        ----------
    //
    // Where L1, L2, ..., Ln represent labels in the function.
    //
    // The actual relocations in the table will be of the form:
    //
    //   Ln - JT
    //    = (Ln - En) + (En - JT)
    //    = R_X86_64_PC32(Ln) + En - JT
    //    = R_X86_64_PC32(Ln + offsetof(En))
    //
    DEBUG(dbgs() << "BOLT-DEBUG: checking for PIC jump table\n");
    MCInst *MovInstr = nullptr;
    while (++II != IE) {
      auto &Instr = II->second;
      const auto &InstrDesc = BC.MII->get(Instr.getOpcode());
      if (!InstrDesc.hasDefOfPhysReg(Instr, R1, *BC.MRI) &&
          !InstrDesc.hasDefOfPhysReg(Instr, R2, *BC.MRI)) {
        // Ignore instructions that don't affect R1, R2 registers.
        continue;
      } else if (!MovInstr) {
        // Expect to see MOV instruction.
        if (!MIA->isMOVSX64rm32(Instr)) {
          DEBUG(dbgs() << "BOLT-DEBUG: MOV instruction expected.\n");
          break;
        }

        // Check if it's setting %r1 or %r2. In canonical form it sets %r2.
        // If it sets %r1 - rename the registers so we have to only check
        // a single form.
        auto MovDestReg = Instr.getOperand(0).getReg();
        if (MovDestReg != R2)
          std::swap(R1, R2);
        if (MovDestReg != R2) {
          DEBUG(dbgs() << "BOLT-DEBUG: MOV instruction expected to set %r2\n");
          break;
        }

        // Verify operands for MOV.
        unsigned  BaseRegNum;
        int64_t   ScaleValue;
        unsigned  IndexRegNum;
        int64_t   DispValue;
        unsigned  SegRegNum;
        if (!MIA->evaluateX86MemoryOperand(Instr, &BaseRegNum,
                                           &ScaleValue, &IndexRegNum,
                                           &DispValue, &SegRegNum))
          break;
        if (BaseRegNum != R1 ||
            ScaleValue != 4 ||
            IndexRegNum == bolt::NoRegister ||
            DispValue != 0 ||
            SegRegNum != bolt::NoRegister)
          break;
        MovInstr = &Instr;
      } else {
        assert(MovInstr && "MOV instruction expected to be set");
        if (!InstrDesc.hasDefOfPhysReg(Instr, R1, *BC.MRI))
          continue;
        if (!MIA->isLEA64r(Instr)) {
          DEBUG(dbgs() << "BOLT-DEBUG: LEA instruction expected\n");
          break;
        }
        if (Instr.getOperand(0).getReg() != R1) {
          DEBUG(dbgs() << "BOLT-DEBUG: LEA instruction expected to set %r1\n");
          break;
        }

        // Verify operands for LEA.
        unsigned      BaseRegNum;
        int64_t       ScaleValue;
        unsigned      IndexRegNum;
        const MCExpr *DispExpr = nullptr;
        unsigned      SegRegNum;
        if (!MIA->evaluateX86MemoryOperand(Instr, &BaseRegNum,
                                           &ScaleValue, &IndexRegNum,
                                           nullptr, &SegRegNum, &DispExpr))
          break;
        if (BaseRegNum != BC.MRI->getProgramCounter() ||
            IndexRegNum != bolt::NoRegister ||
            SegRegNum != bolt::NoRegister ||
            DispExpr == nullptr)
          break;
        MemLocInstr = &Instr;
        break;
      }
    }

    if (!MemLocInstr)
      return IndirectBranchType::UNKNOWN;

    DEBUG(dbgs() << "BOLT-DEBUG: checking potential PIC jump table\n");
    return IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE;
  };

  // Try to find a (base) memory location from where the address for
  // the indirect branch is loaded. For X86-64 the memory will be specified
  // in the following format:
  //
  //   {%rip}/{%basereg} + Imm + IndexReg * Scale
  //
  // We are interested in the cases where Scale == sizeof(uintptr_t) and
  // the contents of the memory are presumably a function array.
  //
  // Normal jump table:
  //
  //    jmp *(JUMP_TABLE, %index, Scale)
  //
  //    or
  //
  //    mov (JUMP_TABLE, %index, Scale), %r1
  //    ...
  //    jmp %r1
  //
  // We handle PIC-style jump tables separately.
  //
  if (Instruction.getNumPrimeOperands() == 1) {
    // If the indirect jump is on register - try to detect if the
    // register value is loaded from a memory location.
    assert(Instruction.getOperand(0).isReg() && "register operand expected");
    const auto R1 = Instruction.getOperand(0).getReg();
    // Check if one of the previous instructions defines the jump-on register.
    // We will check that this instruction belongs to the same basic block
    // in postProcessIndirectBranches().
    for (auto PrevII = Instructions.rbegin(); PrevII != Instructions.rend();
         ++PrevII) {
      auto &PrevInstr = PrevII->second;
      const auto &PrevInstrDesc = BC.MII->get(PrevInstr.getOpcode());

      if (!PrevInstrDesc.hasDefOfPhysReg(PrevInstr, R1, *BC.MRI))
        continue;

      if (MIA->isMoveMem2Reg(PrevInstr)) {
        MemLocInstr = &PrevInstr;
        break;
      } else if (MIA->isADD64rr(PrevInstr)) {
        auto R2 = PrevInstr.getOperand(2).getReg();
        if (R1 == R2)
          return IndirectBranchType::UNKNOWN;
        Type = analyzePICJumpTable(PrevII, Instructions.rend(), R1, R2);
        break;
      }  else {
        return IndirectBranchType::UNKNOWN;
      }
    }
    if (!MemLocInstr) {
      // No definition seen for the register in this function so far. Could be
      // an input parameter - which means it is an external code reference.
      // It also could be that the definition happens to be in the code that
      // we haven't processed yet. Since we have to be conservative, return
      // as UNKNOWN case.
      return IndirectBranchType::UNKNOWN;
    }
  } else {
    MemLocInstr = &Instruction;
  }

  const auto RIPRegister = BC.MRI->getProgramCounter();
  auto PtrSize = BC.AsmInfo->getPointerSize();

  // Analyze the memory location.
  unsigned      BaseRegNum;
  int64_t       ScaleValue;
  unsigned      IndexRegNum;
  int64_t       DispValue;
  unsigned      SegRegNum;
  const MCExpr *DispExpr;
  if (!MIA->evaluateX86MemoryOperand(*MemLocInstr, &BaseRegNum,
                                     &ScaleValue, &IndexRegNum,
                                     &DispValue, &SegRegNum,
                                     &DispExpr))
    return IndirectBranchType::UNKNOWN;

  // Do not set annotate with index reg if address was precomputed earlier
  // and reg may not be live at the jump site.
  if (MemLocInstr != &Instruction)
    IndexRegNum = 0;

  if ((BaseRegNum != bolt::NoRegister && BaseRegNum != RIPRegister) ||
      SegRegNum != bolt::NoRegister)
    return IndirectBranchType::UNKNOWN;

  if (Type == IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE &&
      (ScaleValue != 1 || BaseRegNum != RIPRegister))
    return IndirectBranchType::UNKNOWN;

  if (Type != IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE &&
      ScaleValue != PtrSize)
    return IndirectBranchType::UNKNOWN;

  // RIP-relative addressing should be converted to symbol form by now
  // in processed instructions (but not in jump).
  if (DispExpr) {
    auto SI = BC.GlobalSymbols.find(DispExpr->getSymbol().getName());
    assert(SI != BC.GlobalSymbols.end() && "global symbol needs a value");
    ArrayStart = SI->second;
  } else {
    ArrayStart = static_cast<uint64_t>(DispValue);
    if (BaseRegNum == RIPRegister)
      ArrayStart += getAddress() + Offset + Size;
  }

  DEBUG(dbgs() << "BOLT-DEBUG: addressed memory is 0x"
               << Twine::utohexstr(ArrayStart) << '\n');

  // Check if there's already a jump table registered at this address.
  if (auto *JT = getJumpTableContainingAddress(ArrayStart)) {
    auto JTOffset = ArrayStart - JT->Address;
    if (Type == IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE && JTOffset != 0) {
        // Adjust the size of this jump table and create a new one if necessary.
        // We cannot re-use the entries since the offsets are relative to the
        // table start.
        DEBUG(dbgs() << "BOLT-DEBUG: adjusting size of jump table at 0x"
                     << Twine::utohexstr(JT->Address) << '\n');
        JT->OffsetEntries.resize(JTOffset / JT->EntrySize);
    } else {
      // Re-use an existing jump table. Perhaps parts of it.
      if (Type != IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE) {
        assert(JT->Type == JumpTable::JTT_NORMAL &&
               "normal jump table expected");
        Type = IndirectBranchType::POSSIBLE_JUMP_TABLE;
      } else {
        assert(JT->Type == JumpTable::JTT_PIC && "PIC jump table expected");
      }

      // Get or create a new label for the table.
      auto LI = JT->Labels.find(JTOffset);
      if (LI == JT->Labels.end()) {
        auto *JTStartLabel = BC.getOrCreateGlobalSymbol(ArrayStart,
                                                        "JUMP_TABLEat");
        auto Result = JT->Labels.emplace(JTOffset, JTStartLabel);
        assert(Result.second && "error adding jump table label");
        LI = Result.first;
      }

      BC.MIA->replaceMemOperandDisp(*MemLocInstr, LI->second, BC.Ctx.get());
      BC.MIA->setJumpTable(BC.Ctx.get(), Instruction, ArrayStart, IndexRegNum);

      JTSites.emplace_back(Offset, ArrayStart);

      return Type;
    }
  }

  auto SectionOrError = BC.getSectionForAddress(ArrayStart);
  if (!SectionOrError) {
    // No section - possibly an absolute address. Since we don't allow
    // internal function addresses to escape the function scope - we
    // consider it a tail call.
    if (opts::Verbosity >= 1) {
      errs() << "BOLT-WARNING: no section for address 0x"
             << Twine::utohexstr(ArrayStart) << " referenced from function "
             << *this << '\n';
    }
    return IndirectBranchType::POSSIBLE_TAIL_CALL;
  }
  auto &Section = *SectionOrError;
  if (Section.isVirtual()) {
    // The contents are filled at runtime.
    return IndirectBranchType::POSSIBLE_TAIL_CALL;
  }
  // Extract the value at the start of the array.
  StringRef SectionContents;
  Section.getContents(SectionContents);
  auto EntrySize =
    Type == IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE ? 4 : PtrSize;
  DataExtractor DE(SectionContents, BC.AsmInfo->isLittleEndian(), EntrySize);
  auto ValueOffset = static_cast<uint32_t>(ArrayStart - Section.getAddress());
  uint64_t Value = 0;
  std::vector<uint64_t> JTOffsetCandidates;
  while (ValueOffset <= Section.getSize() - EntrySize) {
    DEBUG(dbgs() << "BOLT-DEBUG: indirect jmp at 0x"
                 << Twine::utohexstr(getAddress() + Offset)
                 << " is referencing address 0x"
                 << Twine::utohexstr(Section.getAddress() + ValueOffset));
    // Extract the value and increment the offset.
    if (Type == IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE) {
      Value = ArrayStart + DE.getSigned(&ValueOffset, 4);
    } else {
      Value = DE.getAddress(&ValueOffset);
    }
    DEBUG(dbgs() << ", which contains value "
                 << Twine::utohexstr(Value) << '\n');
    if (containsAddress(Value) && Value != getAddress()) {
      // Is it possible to have a jump table with function start as an entry?
      JTOffsetCandidates.push_back(Value - getAddress());
      if (Type == IndirectBranchType::UNKNOWN)
        Type = IndirectBranchType::POSSIBLE_JUMP_TABLE;
      continue;
    }
    // Potentially a switch table can contain  __builtin_unreachable() entry
    // pointing just right after the function. In this case we have to check
    // another entry. Otherwise the entry is outside of this function scope
    // and it's not a switch table.
    if (Value == getAddress() + getSize()) {
      JTOffsetCandidates.push_back(Value - getAddress());
    } else {
      break;
    }
  }
  if (Type == IndirectBranchType::POSSIBLE_JUMP_TABLE ||
      Type == IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE) {
    assert(JTOffsetCandidates.size() > 2 &&
           "expected more than 2 jump table entries");
    auto *JTStartLabel = BC.getOrCreateGlobalSymbol(ArrayStart, "JUMP_TABLEat");
    DEBUG(dbgs() << "BOLT-DEBUG: creating jump table "
                 << JTStartLabel->getName()
                 << " in function " << *this << " with "
                 << JTOffsetCandidates.size() << " entries.\n");
    auto JumpTableType =
      Type == IndirectBranchType::POSSIBLE_JUMP_TABLE
        ? JumpTable::JTT_NORMAL
        : JumpTable::JTT_PIC;
    JumpTables.emplace(ArrayStart, JumpTable{ArrayStart,
                                             EntrySize,
                                             JumpTableType,
                                             std::move(JTOffsetCandidates),
                                             {{0, JTStartLabel}}});
    BC.MIA->replaceMemOperandDisp(*MemLocInstr, JTStartLabel, BC.Ctx.get());
    BC.MIA->setJumpTable(BC.Ctx.get(), Instruction, ArrayStart, IndexRegNum);

    JTSites.emplace_back(Offset, ArrayStart);

    return Type;
  }
  BC.InterproceduralReferences.insert(Value);
  return IndirectBranchType::POSSIBLE_TAIL_CALL;
}

MCSymbol *BinaryFunction::getOrCreateLocalLabel(uint64_t Address,
                                                bool CreatePastEnd) {
  MCSymbol *Result;
  // Check if there's already a registered label.
  auto Offset = Address - getAddress();

  if ((Offset == getSize()) && CreatePastEnd)
    return getFunctionEndLabel();

  // Check if there's a global symbol registered at given address.
  // If so - reuse it since we want to keep the symbol value updated.
  if (Offset != 0) {
    if (auto *Symbol = BC.getGlobalSymbolAtAddress(Address)) {
      Labels[Offset] = Symbol;
      return Symbol;
    }
  }

  auto LI = Labels.find(Offset);
  if (LI == Labels.end()) {
    Result = BC.Ctx->createTempSymbol();
    Labels[Offset] = Result;
  } else {
    Result = LI->second;
  }
  return Result;
}

void BinaryFunction::disassemble(ArrayRef<uint8_t> FunctionData) {
  assert(FunctionData.size() == getSize() &&
         "function size does not match raw data size");

  auto &Ctx = BC.Ctx;
  auto &MIA = BC.MIA;

  DWARFUnitLineTable ULT = getDWARFUnitLineTable();

  // Insert a label at the beginning of the function. This will be our first
  // basic block.
  Labels[0] = Ctx->createTempSymbol("BB0", false);
  addEntryPointAtOffset(0);

  auto handleRIPOperand =
      [&](MCInst &Instruction, uint64_t Address, uint64_t Size) {
    uint64_t TargetAddress{0};
    MCSymbol *TargetSymbol{nullptr};
    if (!MIA->evaluateMemOperandTarget(Instruction, TargetAddress, Address,
                                       Size)) {
      errs() << "BOLT-ERROR: rip-relative operand can't be evaluated:\n";
      BC.InstPrinter->printInst(&Instruction, errs(), "", *BC.STI);
      errs() << '\n';
      Instruction.dump_pretty(errs(), BC.InstPrinter.get());
      errs() << '\n';;
      return false;
    }
    if (TargetAddress == 0) {
      if (opts::Verbosity >= 1) {
        outs() << "BOLT-INFO: rip-relative operand is zero in function "
               << *this << ".\n";
      }
    }

    // Note that the address does not necessarily have to reside inside
    // a section, it could be an absolute address too.
    auto Section = BC.getSectionForAddress(TargetAddress);
    if (Section && Section->isText()) {
      if (containsAddress(TargetAddress)) {
        if (TargetAddress != getAddress()) {
          // The address could potentially escape. Mark it as another entry
          // point into the function.
          DEBUG(dbgs() << "BOLT-DEBUG: potentially escaped address 0x"
                       << Twine::utohexstr(TargetAddress) << " in function "
                       << *this << '\n');
          TargetSymbol = getOrCreateLocalLabel(TargetAddress);
          addEntryPointAtOffset(TargetAddress - getAddress());
        }
      } else {
        BC.InterproceduralReferences.insert(TargetAddress);
      }
    }
    if (!TargetSymbol)
      TargetSymbol = BC.getOrCreateGlobalSymbol(TargetAddress, "DATAat");
    MIA->replaceMemOperandDisp(
        Instruction, MCOperand::createExpr(MCSymbolRefExpr::create(
                         TargetSymbol, MCSymbolRefExpr::VK_None, *BC.Ctx)));
    return true;
  };

  uint64_t Size = 0;  // instruction size
  for (uint64_t Offset = 0; Offset < getSize(); Offset += Size) {
    MCInst Instruction;
    const uint64_t AbsoluteInstrAddr = getAddress() + Offset;

    if (!BC.DisAsm->getInstruction(Instruction,
                                   Size,
                                   FunctionData.slice(Offset),
                                   AbsoluteInstrAddr,
                                   nulls(),
                                   nulls())) {
      // Functions with "soft" boundaries, e.g. coming from assembly source,
      // can have 0-byte padding at the end.
      bool IsZeroPadding = true;
      for (auto I = Offset; I < getSize(); ++I) {
        if (FunctionData[I] != 0) {
          IsZeroPadding = false;
          break;
        }
      }

      if (!IsZeroPadding) {
        // Ignore this function. Skip to the next one in non-relocs mode.
        errs() << "BOLT-WARNING: unable to disassemble instruction at offset 0x"
               << Twine::utohexstr(Offset) << " (address 0x"
               << Twine::utohexstr(AbsoluteInstrAddr) << ") in function "
               << *this << '\n';
        IsSimple = false;
      }
      break;
    }

    // Cannot process functions with AVX-512 instructions.
    if (MIA->hasEVEXEncoding(Instruction)) {
      if (opts::Verbosity >= 1) {
        errs() << "BOLT-WARNING: function " << *this << " uses instruction"
               " encoded with EVEX (AVX-512) at offset 0x"
               << Twine::utohexstr(Offset) << ". Disassembly could be wrong."
               " Skipping further processing.\n";
      }
      IsSimple = false;
      break;
    }

    // Check if there's a relocation associated with this instruction.
    if (!Relocations.empty()) {
      auto RI = Relocations.lower_bound(Offset);
      if (RI != Relocations.end() && RI->first < Offset + Size) {
        const auto &Relocation = RI->second;
        DEBUG(dbgs() << "BOLT-DEBUG: replacing immediate with relocation"
                     " against " << Relocation.Symbol->getName()
                     << " in function " << *this
                     << " for instruction at offset 0x"
                     << Twine::utohexstr(Offset) << '\n');
        int64_t Value;
        const auto Result =
          BC.MIA->replaceImmWithSymbol(Instruction, Relocation.Symbol,
                                       Relocation.Addend, Ctx.get(), Value);
        (void)Result;
        assert(Result && "cannot replace immediate with relocation");

        // Make sure we replaced the correct immediate (instruction
        // can have multiple immediate operands).
        assert(static_cast<uint64_t>(Value) == Relocation.Value &&
               "immediate value mismatch in function");
      }
    }

    // Convert instruction to a shorter version that could be relaxed if needed.
    MIA->shortenInstruction(Instruction);

    if (MIA->isBranch(Instruction) || MIA->isCall(Instruction)) {
      uint64_t TargetAddress = 0;
      if (MIA->evaluateBranch(Instruction,
                              AbsoluteInstrAddr,
                              Size,
                              TargetAddress)) {
        // Check if the target is within the same function. Otherwise it's
        // a call, possibly a tail call.
        //
        // If the target *is* the function address it could be either a branch
        // or a recursive call.
        bool IsCall = MIA->isCall(Instruction);
        const bool IsCondBranch = MIA->isConditionalBranch(Instruction);
        MCSymbol *TargetSymbol = nullptr;

        if (IsCall && containsAddress(TargetAddress)) {
          if (TargetAddress == getAddress()) {
            // Recursive call.
            TargetSymbol = getSymbol();
          } else {
            // Possibly an old-style PIC code
            errs() << "BOLT-WARNING: internal call detected at 0x"
                   << Twine::utohexstr(AbsoluteInstrAddr)
                   << " in function " << *this << ". Skipping.\n";
            IsSimple = false;
          }
        }

        if (!TargetSymbol) {
          // Create either local label or external symbol.
          if (containsAddress(TargetAddress)) {
            TargetSymbol = getOrCreateLocalLabel(TargetAddress);
          } else {
            if (TargetAddress == getAddress() + getSize() &&
                TargetAddress < getAddress() + getMaxSize()) {
              // Result of __builtin_unreachable().
              DEBUG(dbgs() << "BOLT-DEBUG: jump past end detected at 0x"
                           << Twine::utohexstr(AbsoluteInstrAddr)
                           << " in function " << *this
                           << " : replacing with nop.\n");
              BC.MIA->createNoop(Instruction);
              if (IsCondBranch) {
                // Register branch offset for profile validation.
                IgnoredBranches.emplace_back(Offset, Offset + Size);
              }
              goto add_instruction;
            }
            BC.InterproceduralReferences.insert(TargetAddress);
            if (opts::Verbosity >= 2 && !IsCall && Size == 2 && !opts::Relocs) {
              errs() << "BOLT-WARNING: relaxed tail call detected at 0x"
                     << Twine::utohexstr(AbsoluteInstrAddr)
                     << " in function " << *this
                     << ". Code size will be increased.\n";
            }

            assert(!MIA->isTailCall(Instruction) &&
                   "synthetic tail call instruction found");

            // This is a call regardless of the opcode.
            // Assign proper opcode for tail calls, so that they could be
            // treated as calls.
            if (!IsCall) {
              if (!MIA->convertJmpToTailCall(Instruction)) {
                assert(IsCondBranch && "unknown tail call instruction");
                if (opts::Verbosity >= 2) {
                  errs() << "BOLT-WARNING: conditional tail call detected in "
                         << "function " << *this << " at 0x"
                         << Twine::utohexstr(AbsoluteInstrAddr) << ".\n";
                }
              }
              IsCall = true;
            }

            TargetSymbol = BC.getOrCreateGlobalSymbol(TargetAddress,
                                                      "FUNCat");
            if (TargetAddress == 0) {
              // We actually see calls to address 0 in presence of weak symbols
              // originating from libraries. This code is never meant to be
              // executed.
              if (opts::Verbosity >= 2) {
                outs() << "BOLT-INFO: Function " << *this
                       << " has a call to address zero.\n";
              }
            }

            if (opts::Relocs) {
              // Check if we need to create relocation to move this function's
              // code without re-assembly.
              size_t RelSize = (Size < 5) ? 1 : 4;
              auto RelOffset = Offset + Size - RelSize;
              auto RI = MoveRelocations.find(RelOffset);
              if (RI == MoveRelocations.end()) {
                uint64_t RelType = (RelSize == 1) ? ELF::R_X86_64_PC8
                                                  : ELF::R_X86_64_PC32;
                DEBUG(dbgs() << "BOLT-DEBUG: creating relocation for static"
                             << " function call to " << TargetSymbol->getName()
                             << " at offset 0x"
                             << Twine::utohexstr(RelOffset)
                             << " with size " << RelSize
                             << " for function " << *this << '\n');
                addRelocation(getAddress() + RelOffset, TargetSymbol, RelType,
                              -RelSize, 0);
              }
              auto OI = PCRelativeRelocationOffsets.find(RelOffset);
              if (OI != PCRelativeRelocationOffsets.end()) {
                PCRelativeRelocationOffsets.erase(OI);
              }
            }
          }
        }

        if (!IsCall) {
          // Add taken branch info.
          TakenBranches.emplace_back(Offset, TargetAddress - getAddress());
        }
        Instruction.clear();
        Instruction.addOperand(
            MCOperand::createExpr(
              MCSymbolRefExpr::create(TargetSymbol,
                                      MCSymbolRefExpr::VK_None,
                                      *Ctx)));

        // Record call offset for profile matching.
        if (IsCall) {
          MIA->addAnnotation(Ctx.get(), Instruction, "Offset", Offset);
        }
        if (IsCondBranch) {
          // Add fallthrough branch info.
          FTBranches.emplace_back(Offset, Offset + Size);
          if (IsCall) {
            MIA->setConditionalTailCall(Instruction, TargetAddress);
          }
        }
      } else {
        // Could not evaluate branch. Should be an indirect call or an
        // indirect branch. Bail out on the latter case.
        MIA->addAnnotation(Ctx.get(), Instruction, "Offset", Offset);
        if (MIA->isIndirectBranch(Instruction)) {
          auto Result = analyzeIndirectBranch(Instruction, Size, Offset);
          switch (Result) {
          default:
            llvm_unreachable("unexpected result");
          case IndirectBranchType::POSSIBLE_TAIL_CALL:
            {
              auto Result = MIA->convertJmpToTailCall(Instruction);
              (void)Result;
              assert(Result);
            }
            break;
          case IndirectBranchType::POSSIBLE_JUMP_TABLE:
          case IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE:
            if (opts::JumpTables == JTS_NONE)
              IsSimple = false;
            break;
          case IndirectBranchType::UNKNOWN:
            // Keep processing. We'll do more checks and fixes in
            // postProcessIndirectBranches().
            break;
          };
        }
        // Indirect call. We only need to fix it if the operand is RIP-relative
        if (IsSimple && MIA->hasRIPOperand(Instruction)) {
          if (!handleRIPOperand(Instruction, AbsoluteInstrAddr, Size)) {
            errs() << "BOLT-ERROR: cannot handle RIP operand at 0x"
                   << Twine::utohexstr(AbsoluteInstrAddr)
                   << ". Skipping function " << *this << ".\n";
            if (opts::Relocs)
              exit(1);
            IsSimple = false;
          }
        }
      }
    } else {
      if (MIA->hasRIPOperand(Instruction)) {
        if (!handleRIPOperand(Instruction, AbsoluteInstrAddr, Size)) {
          errs() << "BOLT-ERROR: cannot handle RIP operand at 0x"
                 << Twine::utohexstr(AbsoluteInstrAddr)
                 << ". Skipping function " << *this << ".\n";
          if (opts::Relocs)
            exit(1);
          IsSimple = false;
        }
      }
    }

add_instruction:
    if (ULT.first && ULT.second) {
      Instruction.setLoc(
          findDebugLineInformationForInstructionAt(AbsoluteInstrAddr, ULT));
    }

    addInstruction(Offset, std::move(Instruction));
  }

  postProcessJumpTables();

  updateState(State::Disassembled);
}

void BinaryFunction::postProcessJumpTables() {
  // Create labels for all entries.
  for (auto &JTI : JumpTables) {
    auto &JT = JTI.second;
    for (auto Offset : JT.OffsetEntries) {
      auto *Label = getOrCreateLocalLabel(getAddress() + Offset,
                                          /*CreatePastEnd*/ true);
      JT.Entries.push_back(Label);
    }
  }

  // Add TakenBranches from JumpTables.
  //
  // We want to do it after initial processing since we don't know jump tables'
  // boundaries until we process them all.
  for (auto &JTSite : JTSites) {
    const auto JTSiteOffset = JTSite.first;
    const auto JTAddress = JTSite.second;
    const auto *JT = getJumpTableContainingAddress(JTAddress);
    assert(JT && "cannot find jump table for address");
    auto EntryOffset = JTAddress - JT->Address;
    while (EntryOffset < JT->getSize()) {
      auto TargetOffset = JT->OffsetEntries[EntryOffset / JT->EntrySize];
      if (TargetOffset < getSize())
        TakenBranches.emplace_back(JTSiteOffset, TargetOffset);

      // Take ownership of jump table relocations.
      if (opts::Relocs)
        BC.removeRelocationAt(JT->Address + EntryOffset);

      EntryOffset += JT->EntrySize;

      // A label at the next entry means the end of this jump table.
      if (JT->Labels.count(EntryOffset))
        break;
    }
  }

  // Free memory used by jump table offsets.
  for (auto &JTI : JumpTables) {
    auto &JT = JTI.second;
    clearList(JT.OffsetEntries);
  }

  // Remove duplicates branches. We can get a bunch of them from jump tables.
  // Without doing jump table value profiling we don't have use for extra
  // (duplicate) branches.
  std::sort(TakenBranches.begin(), TakenBranches.end());
  auto NewEnd = std::unique(TakenBranches.begin(), TakenBranches.end());
  TakenBranches.erase(NewEnd, TakenBranches.end());
}

bool BinaryFunction::postProcessIndirectBranches() {
  for (auto *BB : layout()) {
    for (auto &Instr : *BB) {
      if (!BC.MIA->isIndirectBranch(Instr))
        continue;

      // If there's an indirect branch in a single-block function -
      // it must be a tail call.
      if (layout_size() == 1) {
        BC.MIA->convertJmpToTailCall(Instr);
        return true;
      }

      // Validate the tail call or jump table assumptions.
      if (BC.MIA->isTailCall(Instr) || BC.MIA->getJumpTable(Instr)) {
        if (BC.MIA->getMemoryOperandNo(Instr) != -1) {
          // We have validated memory contents addressed by the jump
          // instruction already.
          continue;
        }
        // This is jump on register. Just make sure the register is defined
        // in the containing basic block. Other assumptions were checked
        // earlier.
        assert(Instr.getOperand(0).isReg() && "register operand expected");
        const auto R1 = Instr.getOperand(0).getReg();
        auto PrevInstr = BB->rbegin();
        while (PrevInstr != BB->rend()) {
          const auto &PrevInstrDesc = BC.MII->get(PrevInstr->getOpcode());
          if (PrevInstrDesc.hasDefOfPhysReg(*PrevInstr, R1, *BC.MRI)) {
            break;
          }
          ++PrevInstr;
        }
        if (PrevInstr == BB->rend()) {
          if (opts::Verbosity >= 2) {
            outs() << "BOLT-INFO: rejected potential "
                       << (BC.MIA->isTailCall(Instr) ? "indirect tail call"
                                                     : "jump table")
                       << " in function " << *this
                       << " because the jump-on register was not defined in "
                       << " basic block " << BB->getName() << ".\n";
            DEBUG(dbgs() << BC.printInstructions(dbgs(), BB->begin(), BB->end(),
                                                 BB->getOffset(), this, true));
          }
          return false;
        }
        // In case of PIC jump table we need to do more checks.
        if (BC.MIA->isMoveMem2Reg(*PrevInstr))
          continue;
        assert(BC.MIA->isADD64rr(*PrevInstr) && "add instruction expected");
        auto R2 = PrevInstr->getOperand(2).getReg();
        // Make sure both regs are set in the same basic block prior to ADD.
        bool IsR1Set = false;
        bool IsR2Set = false;
        while ((++PrevInstr != BB->rend()) && !(IsR1Set && IsR2Set)) {
          const auto &PrevInstrDesc = BC.MII->get(PrevInstr->getOpcode());
          if (PrevInstrDesc.hasDefOfPhysReg(*PrevInstr, R1, *BC.MRI))
            IsR1Set = true;
          else if (PrevInstrDesc.hasDefOfPhysReg(*PrevInstr, R2, *BC.MRI))
            IsR2Set = true;
        }

        if (!IsR1Set || !IsR2Set)
          return false;

        continue;
      }

      // If this block contains an epilogue code and has an indirect branch,
      // then most likely it's a tail call. Otherwise, we cannot tell for sure
      // what it is and conservatively reject the function's CFG.
      bool IsEpilogue = false;
      for (const auto &Instr : *BB) {
        if (BC.MIA->isLeave(Instr) || BC.MIA->isPop(Instr)) {
          IsEpilogue = true;
          break;
        }
      }
      if (!IsEpilogue) {
        if (opts::Verbosity >= 2) {
          outs() << "BOLT-INFO: rejected potential indirect tail call in "
                 << "function " << *this << " in basic block "
                 << BB->getName() << ".\n";
          DEBUG(BC.printInstructions(dbgs(), BB->begin(), BB->end(),
                                     BB->getOffset(), this, true));
        }
        return false;
      }
      BC.MIA->convertJmpToTailCall(Instr);
    }
  }
  return true;
}

void BinaryFunction::clearLandingPads(const unsigned StartIndex,
                                      const unsigned NumBlocks) {
  // remove all landing pads/throws for the given collection of blocks
  for (auto I = StartIndex; I < StartIndex + NumBlocks; ++I) {
    BasicBlocks[I]->clearLandingPads();
  }
}

void BinaryFunction::addLandingPads(const unsigned StartIndex,
                                    const unsigned NumBlocks) {
  for (auto *BB : BasicBlocks) {
    if (LandingPads.find(BB->getLabel()) != LandingPads.end()) {
      const MCSymbol *LP = BB->getLabel();
      for (unsigned I : LPToBBIndex[LP]) {
        assert(I < BasicBlocks.size());
        BinaryBasicBlock *ThrowBB = BasicBlocks[I];
        const unsigned ThrowBBIndex = getIndex(ThrowBB);
        if (ThrowBBIndex >= StartIndex && ThrowBBIndex < StartIndex + NumBlocks)
          ThrowBB->addLandingPad(BB);
      }
    }
  }
}

void BinaryFunction::recomputeLandingPads(const unsigned StartIndex,
                                          const unsigned NumBlocks) {
  assert(LPToBBIndex.empty());

  clearLandingPads(StartIndex, NumBlocks);

  for (auto I = StartIndex; I < StartIndex + NumBlocks; ++I) {
    auto *BB = BasicBlocks[I];
    for (auto &Instr : BB->instructions()) {
      // Store info about associated landing pad.
      if (BC.MIA->isInvoke(Instr)) {
        const MCSymbol *LP;
        uint64_t Action;
        std::tie(LP, Action) = BC.MIA->getEHInfo(Instr);
        if (LP) {
          LPToBBIndex[LP].push_back(getIndex(BB));
        }
      }
    }
  }

  addLandingPads(StartIndex, NumBlocks);

  clearList(LPToBBIndex);
}

bool BinaryFunction::buildCFG() {
  auto &MIA = BC.MIA;

  if (!isSimple()) {
    assert(!opts::Relocs &&
           "cannot process file with non-simple function in relocs mode");
    return false;
  }

  if (!(CurrentState == State::Disassembled))
    return false;

  assert(BasicBlocks.empty() && "basic block list should be empty");
  assert((Labels.find(0) != Labels.end()) &&
         "first instruction should always have a label");

  // Create basic blocks in the original layout order:
  //
  //  * Every instruction with associated label marks
  //    the beginning of a basic block.
  //  * Conditional instruction marks the end of a basic block,
  //    except when the following instruction is an
  //    unconditional branch, and the unconditional branch is not
  //    a destination of another branch. In the latter case, the
  //    basic block will consist of a single unconditional branch
  //    (missed "double-jump" optimization).
  //
  // Created basic blocks are sorted in layout order since they are
  // created in the same order as instructions, and instructions are
  // sorted by offsets.
  BinaryBasicBlock *InsertBB{nullptr};
  BinaryBasicBlock *PrevBB{nullptr};
  bool IsLastInstrNop{false};
  const MCInst *PrevInstr{nullptr};

  auto addCFIPlaceholders =
      [this](uint64_t CFIOffset, BinaryBasicBlock *InsertBB) {
        for (auto FI = OffsetToCFI.lower_bound(CFIOffset),
                  FE = OffsetToCFI.upper_bound(CFIOffset);
             FI != FE; ++FI) {
          addCFIPseudo(InsertBB, InsertBB->end(), FI->second);
        }
      };

  for (auto I = Instructions.begin(), E = Instructions.end(); I != E; ++I) {
    const uint32_t Offset = I->first;
    const auto &Instr = I->second;

    auto LI = Labels.find(Offset);
    if (LI != Labels.end()) {
      // Always create new BB at branch destination.
      PrevBB = InsertBB;
      InsertBB = addBasicBlock(LI->first, LI->second,
                               /* DeriveAlignment = */ IsLastInstrNop);
      if (hasEntryPointAtOffset(Offset))
        InsertBB->setEntryPoint();
    }
    // Ignore nops. We use nops to derive alignment of the next basic block.
    // It will not always work, as some blocks are naturally aligned, but
    // it's just part of heuristic for block alignment.
    if (MIA->isNoop(Instr)) {
      IsLastInstrNop = true;
      continue;
    }
    if (!InsertBB) {
      // It must be a fallthrough or unreachable code. Create a new block unless
      // we see an unconditional branch following a conditional one. The latter
      // should not be a conditional tail call.
      assert(PrevBB && "no previous basic block for a fall through");
      assert(PrevInstr && "no previous instruction for a fall through");
      if (MIA->isUnconditionalBranch(Instr) &&
          !MIA->isUnconditionalBranch(*PrevInstr) &&
          !MIA->getConditionalTailCall(*PrevInstr)) {
        // Temporarily restore inserter basic block.
        InsertBB = PrevBB;
      } else {
        InsertBB = addBasicBlock(Offset,
                                 BC.Ctx->createTempSymbol("FT", true),
                                 /* DeriveAlignment = */ IsLastInstrNop);
      }
    }
    if (Offset == 0) {
      // Add associated CFI pseudos in the first offset (0)
      addCFIPlaceholders(0, InsertBB);
    }

    IsLastInstrNop = false;
    uint32_t InsertIndex = InsertBB->addInstruction(Instr);
    PrevInstr = &Instr;

    // Record conditional tail call info.
    if (const auto CTCDest = MIA->getConditionalTailCall(Instr)) {
      TailCallTerminatedBlocks.emplace(
        std::make_pair(InsertBB, TailCallInfo(Offset, InsertIndex, *CTCDest)));
    }

    // Add associated CFI instrs. We always add the CFI instruction that is
    // located immediately after this instruction, since the next CFI
    // instruction reflects the change in state caused by this instruction.
    auto NextInstr = std::next(I);
    uint64_t CFIOffset;
    if (NextInstr != E)
      CFIOffset = NextInstr->first;
    else
      CFIOffset = getSize();
    addCFIPlaceholders(CFIOffset, InsertBB);

    // Store info about associated landing pad.
    if (MIA->isInvoke(Instr)) {
      const MCSymbol *LP;
      uint64_t Action;
      std::tie(LP, Action) = MIA->getEHInfo(Instr);
      if (LP) {
        LPToBBIndex[LP].push_back(getIndex(InsertBB));
      }
    }

    // How well do we detect tail calls here?
    if (MIA->isTerminator(Instr)) {
      PrevBB = InsertBB;
      InsertBB = nullptr;
    }
  }

  if (BasicBlocks.empty()) {
    setSimple(false);
    return false;
  }

  // Intermediate dump.
  DEBUG(print(dbgs(), "after creating basic blocks"));

  // TODO: handle properly calls to no-return functions,
  // e.g. exit(3), etc. Otherwise we'll see a false fall-through
  // blocks.

  // Possibly assign/re-assign branch profile data.
  matchProfileData();

  for (auto &Branch : TakenBranches) {
    DEBUG(dbgs() << "registering branch [0x" << Twine::utohexstr(Branch.first)
                 << "] -> [0x" << Twine::utohexstr(Branch.second) << "]\n");
    auto *FromBB = getBasicBlockContainingOffset(Branch.first);
    assert(FromBB && "cannot find BB containing FROM branch");
    auto *ToBB = getBasicBlockAtOffset(Branch.second);
    assert(ToBB && "cannot find BB containing TO branch");

    if (!BranchData) {
      FromBB->addSuccessor(ToBB);
      continue;
    }

    auto BranchInfoOrErr = BranchData->getBranch(Branch.first, Branch.second);
    if (!BranchInfoOrErr) {
      FromBB->addSuccessor(ToBB);
      continue;
    }

    const BranchInfo &BInfo = BranchInfoOrErr.get();
    FromBB->addSuccessor(ToBB, BInfo.Branches, BInfo.Mispreds);
    // Populate profile counts for the jump table.
    auto *LastInstr = FromBB->getLastNonPseudoInstr();
    if (!LastInstr)
      continue;
    auto JTAddress = BC.MIA->getJumpTable(*LastInstr);
    if (!JTAddress)
      continue;
    auto *JT = getJumpTableContainingAddress(JTAddress);
    if (!JT)
      continue;
    JT->Count += BInfo.Branches;
    if (opts::IndirectCallPromotion < ICP_JUMP_TABLES &&
        opts::JumpTables < JTS_AGGRESSIVE)
      continue;
    if (JT->Counts.empty())
      JT->Counts.resize(JT->Entries.size());
    auto EI = JT->Entries.begin();
    auto Delta = (JTAddress - JT->Address) / JT->EntrySize;
    EI += Delta;
    while (EI != JT->Entries.end()) {
      if (ToBB->getLabel() == *EI) {
        assert(Delta < JT->Counts.size());
        JT->Counts[Delta].Mispreds += BInfo.Mispreds;
        JT->Counts[Delta].Count += BInfo.Branches;
      }
      ++Delta;
      ++EI;
      // A label marks the start of another jump table.
      if (JT->Labels.count(Delta * JT->EntrySize))
        break;
    }
  }

  for (auto &Branch : FTBranches) {
    DEBUG(dbgs() << "registering fallthrough [0x"
                 << Twine::utohexstr(Branch.first) << "] -> [0x"
                 << Twine::utohexstr(Branch.second) << "]\n");
    auto *FromBB = getBasicBlockContainingOffset(Branch.first);
    assert(FromBB && "cannot find BB containing FROM branch");
    // Try to find the destination basic block. If the jump instruction was
    // followed by a no-op then the destination offset recorded in FTBranches
    // will point to that no-op but the destination basic block will start
    // after the no-op due to ignoring no-ops when creating basic blocks.
    // So we have to skip any no-ops when trying to find the destination
    // basic block.
    auto *ToBB = getBasicBlockAtOffset(Branch.second);
    if (ToBB == nullptr) {
      auto I = Instructions.find(Branch.second), E = Instructions.end();
      while (ToBB == nullptr && I != E && MIA->isNoop(I->second)) {
        ++I;
        if (I == E)
          break;
        ToBB = getBasicBlockAtOffset(I->first);
      }
      if (ToBB == nullptr) {
        // We have a fall-through that does not point to another BB, ignore it
        // as it may happen in cases where we have a BB finished by two
        // branches.
        // This can also happen when we delete a branch past the end of a
        // function in case of a call to __builtin_unreachable().
        continue;
      }
    }

    // Does not add a successor if we can't find profile data, leave it to the
    // inference pass to guess its frequency
    if (BranchData) {
      auto BranchInfoOrErr = BranchData->getBranch(Branch.first, Branch.second);
      if (BranchInfoOrErr) {
        const BranchInfo &BInfo = BranchInfoOrErr.get();
        FromBB->addSuccessor(ToBB, BInfo.Branches, BInfo.Mispreds);
      }
    }
  }

  for (auto &I : TailCallTerminatedBlocks) {
    TailCallInfo &TCInfo = I.second;
    if (BranchData) {
      auto BranchInfoOrErr = BranchData->getDirectCallBranch(TCInfo.Offset);
      if (BranchInfoOrErr) {
        const BranchInfo &BInfo = BranchInfoOrErr.get();
        TCInfo.Count = BInfo.Branches;
        TCInfo.Mispreds = BInfo.Mispreds;
      }
    }
  }

  // Add fall-through branches (except for non-taken conditional branches with
  // profile data, which were already accounted for in TakenBranches).
  PrevBB = nullptr;
  bool IsPrevFT = false; // Is previous block a fall-through.
  for (auto BB : BasicBlocks) {
    if (IsPrevFT) {
      PrevBB->addSuccessor(BB, BinaryBasicBlock::COUNT_NO_PROFILE,
                           BinaryBasicBlock::COUNT_INFERRED);
    }
    if (BB->empty()) {
      IsPrevFT = true;
      PrevBB = BB;
      continue;
    }

    auto LastInstIter = --BB->end();
    while (MIA->isCFI(*LastInstIter) && LastInstIter != BB->begin())
      --LastInstIter;

    // Check if the last instruction is a conditional jump that serves as a tail
    // call.
    const auto IsCondTailCall = MIA->getConditionalTailCall(*LastInstIter);
    if (BB->succ_size() == 0) {
      if (IsCondTailCall) {
        // Conditional tail call without profile data for non-taken branch.
        IsPrevFT = true;
      } else {
        // Unless the last instruction is a terminator, control will fall
        // through to the next basic block.
        IsPrevFT = MIA->isTerminator(*LastInstIter) ? false : true;
      }
    } else if (BB->succ_size() == 1) {
      if (IsCondTailCall) {
        // Conditional tail call with data for non-taken branch. A fall-through
        // edge has already ben added in the CFG.
        IsPrevFT = false;
      } else {
        // Fall-through should be added if the last instruction is a conditional
        // jump, since there was no profile data for the non-taken branch.
        IsPrevFT = MIA->isConditionalBranch(*LastInstIter) ? true : false;
      }
    } else {
      // Ends with 2 branches, with an indirect jump or it is a conditional
      // branch whose frequency has been inferred from LBR.
      IsPrevFT = false;
    }

    PrevBB = BB;
  }

  if (!IsPrevFT) {
    // Possibly a call that does not return.
    DEBUG(dbgs() << "last block was marked as a fall-through\n");
  }

  // Add associated landing pad blocks to each basic block.
  addLandingPads(0, BasicBlocks.size());

  // Infer frequency for non-taken branches
  if (hasValidProfile() && opts::DoMCF != MCF_DISABLE) {
    // Convert COUNT_NO_PROFILE to 0
    removeTagsFromProfile();
    solveMCF(*this, opts::DoMCF);
  } else if (hasValidProfile()) {
    inferFallThroughCounts();
  } else {
    clearProfile();
  }

  // Assign CFI information to each BB entry.
  annotateCFIState();

  // Convert conditional tail call branches to conditional branches that jump
  // to a tail call.
  removeConditionalTailCalls();

  // Set the basic block layout to the original order.
  PrevBB = nullptr;
  for (auto BB : BasicBlocks) {
    BasicBlocksLayout.emplace_back(BB);
    if (PrevBB)
      PrevBB->setEndOffset(BB->getOffset());
    PrevBB = BB;
  }
  PrevBB->setEndOffset(getSize());

  // Make any necessary adjustments for indirect branches.
  if (!postProcessIndirectBranches()) {
    if (opts::Verbosity) {
      errs() << "BOLT-WARNING: failed to post-process indirect branches for "
             << *this << '\n';
    }
    // In relocation mode we want to keep processing the function but avoid
    // optimizing it.
    setSimple(false);
  }

  // Eliminate inconsistencies between branch instructions and CFG.
  postProcessBranches();

  // If our profiling data comes from samples instead of LBR entries,
  // now is the time to read this data and attach it to BBs. At this point,
  // conditional tail calls are converted into a branch and a new basic block,
  // making it slightly different than the original binary where profiled data
  // was collected. However, this shouldn't matter for plain sampling events.
  if (!BC.DR.hasLBR())
    readSampleData();

  // Clean-up memory taken by instructions and labels.
  //
  // NB: don't clear Labels list as we may need them if we mark the function
  //     as non-simple later in the process of discovering extra entry points.
  clearList(Instructions);
  clearList(TailCallTerminatedBlocks);
  clearList(OffsetToCFI);
  clearList(TakenBranches);
  clearList(FTBranches);
  clearList(IgnoredBranches);
  clearList(LPToBBIndex);
  clearList(EntryOffsets);

  // Update the state.
  CurrentState = State::CFG;

  // Annotate invoke instructions with GNU_args_size data.
  propagateGnuArgsSizeInfo();

  assert(validateCFG() && "Invalid CFG detected after disassembly");

  return true;
}

void BinaryFunction::removeTagsFromProfile() {
  for (auto *BB : BasicBlocks) {
    if (BB->ExecutionCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      BB->ExecutionCount = 0;
    for (auto &BI : BB->branch_info()) {
      if (BI.Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
          BI.MispredictedCount != BinaryBasicBlock::COUNT_NO_PROFILE)
        continue;
      BI.Count = 0;
      BI.MispredictedCount = 0;
    }
  }
}

void BinaryFunction::readSampleData() {
  auto SampleDataOrErr = BC.DR.getFuncSampleData(getNames());

  if (!SampleDataOrErr)
    return;

  // Non-LBR mode territory
  // First step is to assign BB execution count based on samples from perf
  ProfileMatchRatio = 1.0f;
  removeTagsFromProfile();
  bool NormalizeByInsnCount =
      BC.DR.usesEvent("cycles") || BC.DR.usesEvent("instructions");
  bool NormalizeByCalls = BC.DR.usesEvent("branches");
  static bool NagUser{true};
  if (NagUser) {
    outs() << "BOLT-INFO: operating with non-LBR profiling data.\n";
    if (NormalizeByInsnCount) {
      outs() << "BOLT-INFO: normalizing samples by instruction count.\n";
    } else if (NormalizeByCalls) {
      outs() << "BOLT-INFO: normalizing samples by branches.\n";
    }
    NagUser = false;
  }
  uint64_t LastOffset = getSize();
  uint64_t TotalEntryCount{0};
  for (auto I = BasicBlockOffsets.rbegin(), E = BasicBlockOffsets.rend();
       I != E; ++I) {
    uint64_t CurOffset = I->first;
    // Always work with samples multiplied by 1000 to avoid losing them if we
    // later need to normalize numbers
    uint64_t NumSamples =
        SampleDataOrErr->getSamples(CurOffset, LastOffset) * 1000;
    if (NormalizeByInsnCount && I->second->getNumNonPseudos())
      NumSamples /= I->second->getNumNonPseudos();
    else if (NormalizeByCalls) {
      uint32_t NumCalls = I->second->getNumCalls();
      NumSamples /= NumCalls + 1;
    }
    I->second->setExecutionCount(NumSamples);
    if (I->second->isEntryPoint())
      TotalEntryCount += NumSamples;
    LastOffset = CurOffset;
  }
  ExecutionCount = TotalEntryCount;

  estimateEdgeCounts(BC, *this);

  if (opts::DoMCF != MCF_DISABLE)
    solveMCF(*this, opts::DoMCF);
}

void BinaryFunction::addEntryPoint(uint64_t Address) {
  assert(containsAddress(Address) && "address does not belong to the function");

  auto Offset = Address - getAddress();

  DEBUG(dbgs() << "BOLT-INFO: adding external entry point to function " << *this
               << " at offset 0x" << Twine::utohexstr(Address - getAddress())
               << '\n');

  auto *EntrySymbol = BC.getGlobalSymbolAtAddress(Address);

  // If we haven't disassembled the function yet we can add a new entry point
  // even if it doesn't have an associated entry in the symbol table.
  if (CurrentState == State::Empty) {
    if (!EntrySymbol) {
      DEBUG(dbgs() << "creating local label\n");
      EntrySymbol = getOrCreateLocalLabel(Address);
    } else {
      DEBUG(dbgs() << "using global symbol " << EntrySymbol->getName() << '\n');
    }
    addEntryPointAtOffset(Address - getAddress());
    Labels.emplace(Offset, EntrySymbol);
    return;
  }

  assert(EntrySymbol && "expected symbol at address");

  if (isSimple()) {
    // Find basic block corresponding to the address and substitute label.
    auto *BB = getBasicBlockAtOffset(Offset);
    if (!BB) {
      // TODO #14762450: split basic block and process function.
      if (opts::Verbosity || opts::Relocs) {
        errs() << "BOLT-WARNING: no basic block at offset 0x"
               << Twine::utohexstr(Offset) << " in function " << *this
               << ". Marking non-simple.\n";
      }
      setSimple(false);
    } else {
      BB->setLabel(EntrySymbol);
      BB->setEntryPoint(true);
    }
  }

  // Fix/append labels list.
  auto LI = Labels.find(Offset);
  if (LI != Labels.end()) {
    LI->second = EntrySymbol;
  } else {
    Labels.emplace(Offset, EntrySymbol);
  }
}

bool BinaryFunction::fetchProfileForOtherEntryPoints() {
  if (!BranchData)
    return false;

  // Check if we are missing profiling data for secondary entry points
  bool First{true};
  bool Updated{false};
  for (auto BB : BasicBlocks) {
    if (First) {
      First = false;
      continue;
    }
    if (BB->isEntryPoint()) {
      uint64_t EntryAddress = BB->getOffset() + getAddress();
      // Look for branch data associated with this entry point
      std::vector<std::string> Names;
      std::multimap<uint64_t, std::string>::iterator I, E;
      for (std::tie(I, E) = BC.GlobalAddresses.equal_range(EntryAddress);
           I != E; ++I) {
        Names.push_back(I->second);
      }
      if (!Names.empty()) {
        if (FuncBranchData *Data = BC.DR.getFuncBranchData(Names)) {
          BranchData->appendFrom(*Data, BB->getOffset());
          Data->Used = true;
          Updated = true;
        }
      }
    }
  }
  return Updated;
}

void BinaryFunction::matchProfileData() {
  // This functionality is available for LBR-mode only
  // TODO: Implement evaluateProfileData() for samples, checking whether
  // sample addresses match instruction addresses in the function
  if (!BC.DR.hasLBR())
    return;

  if (BranchData) {
    ProfileMatchRatio = evaluateProfileData(*BranchData);
    if (ProfileMatchRatio == 1.0f) {
      if (fetchProfileForOtherEntryPoints()) {
        ProfileMatchRatio = evaluateProfileData(*BranchData);
        ExecutionCount = BranchData->ExecutionCount;
      }
      return;
    }
  }

  // Check if the function name can fluctuate between several compilations
  // possibly triggered by minor unrelated code changes in the source code
  // of the input binary.
  const auto HasVolatileName = [this]() {
    for (const auto Name : getNames()) {
      if (getLTOCommonName(Name))
        return true;
    }
    return false;
  }();
  if (!HasVolatileName)
    return;

  // Check for a profile that matches with 100% confidence.
  const auto AllBranchData = BC.DR.getFuncBranchDataRegex(getNames());
  for (auto *NewBranchData : AllBranchData) {
    // Prevent functions from sharing the same profile.
    if (NewBranchData->Used)
      continue;

    if (evaluateProfileData(*NewBranchData) != 1.0f)
      continue;

    if (BranchData)
      BranchData->Used = false;

    // Update function profile data with the new set.
    BranchData = NewBranchData;
    ExecutionCount = NewBranchData->ExecutionCount;
    ProfileMatchRatio = 1.0f;
    BranchData->Used = true;
    break;
  }
}

float BinaryFunction::evaluateProfileData(const FuncBranchData &BranchData) {
  // Until we define a minimal profile, we consider an empty branch data to be
  // a valid profile. It could happen to a function without branches when we
  // still have an EntryData for execution count.
  if (BranchData.Data.empty()) {
    return 1.0f;
  }

  BranchListType ProfileBranches(BranchData.Data.size());
  std::transform(BranchData.Data.begin(),
                 BranchData.Data.end(),
                 ProfileBranches.begin(),
                 [](const BranchInfo &BI) {
                   return std::make_pair(BI.From.Offset,
                                         BI.To.Name == BI.From.Name ?
                                         BI.To.Offset : -1U);
                 });
  BranchListType LocalProfileBranches;
  std::copy_if(ProfileBranches.begin(),
               ProfileBranches.end(),
               std::back_inserter(LocalProfileBranches),
               [](const std::pair<uint32_t, uint32_t> &Branch) {
                 return Branch.second != -1U;
               });

  // Profile referencing external functions.
  BranchListType ExternProfileBranches;
  std::copy_if(ProfileBranches.begin(),
               ProfileBranches.end(),
               std::back_inserter(ExternProfileBranches),
               [](const std::pair<uint32_t, uint32_t> &Branch) {
                 return Branch.second == -1U;
               });

  std::sort(LocalProfileBranches.begin(), LocalProfileBranches.end());

  BranchListType FunctionBranches = TakenBranches;
  FunctionBranches.insert(FunctionBranches.end(),
                          FTBranches.begin(),
                          FTBranches.end());
  FunctionBranches.insert(FunctionBranches.end(),
                          IgnoredBranches.begin(),
                          IgnoredBranches.end());
  std::sort(FunctionBranches.begin(), FunctionBranches.end());

  BranchListType DiffBranches; // Branches in profile without a match.
  std::set_difference(LocalProfileBranches.begin(),
                      LocalProfileBranches.end(),
                      FunctionBranches.begin(),
                      FunctionBranches.end(),
                      std::back_inserter(DiffBranches));

  // Branches without a match in CFG.
  BranchListType OrphanBranches;

  // Eliminate recursive calls and returns from recursive calls from the list
  // of branches that have no match. They are not considered local branches.
  auto isRecursiveBranch = [&](std::pair<uint32_t, uint32_t> &Branch) {
    auto SrcInstrI = Instructions.find(Branch.first);
    if (SrcInstrI == Instructions.end())
      return false;

    // Check if it is a recursive call.
    const auto &SrcInstr = SrcInstrI->second;
    if ((BC.MIA->isCall(SrcInstr) || BC.MIA->isIndirectBranch(SrcInstr)) &&
        Branch.second == 0)
      return true;

    auto DstInstrI = Instructions.find(Branch.second);
    if (DstInstrI == Instructions.end())
      return false;

    // Check if it is a return from a recursive call.
    bool IsSrcReturn = BC.MIA->isReturn(SrcInstr);
    // "rep ret" is considered to be 2 different instructions.
    if (!IsSrcReturn && BC.MIA->isPrefix(SrcInstr)) {
      auto SrcInstrSuccessorI = SrcInstrI;
      ++SrcInstrSuccessorI;
      assert(SrcInstrSuccessorI != Instructions.end() &&
             "unexpected prefix instruction at the end of function");
      IsSrcReturn = BC.MIA->isReturn(SrcInstrSuccessorI->second);
    }
    if (IsSrcReturn && Branch.second != 0) {
      // Make sure the destination follows the call instruction.
      auto DstInstrPredecessorI = DstInstrI;
      --DstInstrPredecessorI;
      assert(DstInstrPredecessorI != Instructions.end() && "invalid iterator");
      if (BC.MIA->isCall(DstInstrPredecessorI->second))
        return true;
    }
    return false;
  };
  std::remove_copy_if(DiffBranches.begin(),
                      DiffBranches.end(),
                      std::back_inserter(OrphanBranches),
                      isRecursiveBranch);

  // Check all external branches.
  std::copy_if(ExternProfileBranches.begin(),
               ExternProfileBranches.end(),
               std::back_inserter(OrphanBranches),
               [&](const std::pair<uint32_t, uint32_t> &Branch) {
                 auto II = Instructions.find(Branch.first);
                 if (II == Instructions.end())
                   return true;
                 const auto &Instr = II->second;
                 // Check for calls, tail calls, rets and indirect branches.
                 // When matching profiling info, we did not reach the stage
                 // when we identify tail calls, so they are still represented
                 // by regular branch instructions and we need isBranch() here.
                 if (BC.MIA->isCall(Instr) ||
                     BC.MIA->isBranch(Instr) ||
                     BC.MIA->isReturn(Instr))
                   return false;
                 // Check for "rep ret"
                 if (BC.MIA->isPrefix(Instr)) {
                   ++II;
                   if (II != Instructions.end() && BC.MIA->isReturn(II->second))
                     return false;
                 }
                 return true;
               });

  float MatchRatio =
    (float) (ProfileBranches.size() - OrphanBranches.size()) /
    (float) ProfileBranches.size();

  if (opts::Verbosity >= 2 && !OrphanBranches.empty()) {
    errs() << "BOLT-WARNING: profile branches match only "
           << format("%.1f%%", MatchRatio * 100.0f) << " ("
           << (ProfileBranches.size() - OrphanBranches.size()) << '/'
           << ProfileBranches.size() << ") for function "
           << *this << '\n';
    DEBUG(
      for (auto &OBranch : OrphanBranches)
        errs() << "\t0x" << Twine::utohexstr(OBranch.first) << " -> 0x"
               << Twine::utohexstr(OBranch.second) << " (0x"
               << Twine::utohexstr(OBranch.first + getAddress()) << " -> 0x"
               << Twine::utohexstr(OBranch.second + getAddress()) << ")\n";
        );
  }

  return MatchRatio;
}

void BinaryFunction::clearProfile() {
  // Keep function execution profile the same. Only clear basic block and edge
  // counts.
  for (auto *BB : BasicBlocks) {
    BB->ExecutionCount = 0;
    for (auto &BI : BB->branch_info()) {
      BI.Count = 0;
      BI.MispredictedCount = 0;
    }
  }
}


void BinaryFunction::inferFallThroughCounts() {
  assert(!BasicBlocks.empty() && "basic block list should not be empty");
  assert(BranchData && "cannot infer counts without branch data");

  // Compute preliminary execution count for each basic block
  for (auto CurBB : BasicBlocks) {
    CurBB->ExecutionCount = 0;
  }

  for (auto CurBB : BasicBlocks) {
    auto SuccCount = CurBB->branch_info_begin();
    for (auto Succ : CurBB->successors()) {
      if (SuccCount->Count != BinaryBasicBlock::COUNT_NO_PROFILE)
        Succ->setExecutionCount(Succ->getExecutionCount() + SuccCount->Count);
      ++SuccCount;
    }
  }

  // Set entry BBs to zero, we'll update their execution count next with entry
  // data (we maintain a separate data structure for branches to function entry
  // points)
  for (auto BB : BasicBlocks) {
    if (BB->isEntryPoint())
      BB->ExecutionCount = 0;
  }

  // Update execution counts of landing pad blocks and entry BBs
  // There is a slight skew introduced here as branches originated from RETs
  // may be accounted for in the execution count of an entry block if the last
  // instruction in a predecessor fall-through block is a call. This situation
  // should rarely happen because there are few multiple-entry functions.
  for (const auto &I : BranchData->EntryData) {
    BinaryBasicBlock *BB = getBasicBlockAtOffset(I.To.Offset);
    if (BB && (BB->isEntryPoint() ||
               LandingPads.find(BB->getLabel()) != LandingPads.end())) {
      BB->setExecutionCount(BB->getExecutionCount() + I.Branches);
    }
  }

  // Work on a basic block at a time, propagating frequency information
  // forwards.
  // It is important to walk in the layout order.
  for (auto CurBB : BasicBlocks) {
    uint64_t BBExecCount = CurBB->getExecutionCount();

    // Propagate this information to successors, filling in fall-through edges
    // with frequency information
    if (CurBB->succ_size() == 0)
      continue;

    // Calculate frequency of outgoing branches from this node according to
    // LBR data.
    uint64_t ReportedBranches = 0;
    for (const auto &SuccCount : CurBB->branch_info()) {
      if (SuccCount.Count != BinaryBasicBlock::COUNT_NO_PROFILE)
        ReportedBranches += SuccCount.Count;
    }

    // Calculate frequency of outgoing tail calls from this node according to
    // LBR data.
    uint64_t ReportedTailCalls = 0;
    auto TCI = TailCallTerminatedBlocks.find(CurBB);
    if (TCI != TailCallTerminatedBlocks.end()) {
      ReportedTailCalls = TCI->second.Count;
    }

    // Calculate frequency of throws from this node according to LBR data
    // for branching into associated landing pads. Since it is possible
    // for a landing pad to be associated with more than one basic blocks,
    // we may overestimate the frequency of throws for such blocks.
    uint64_t ReportedThrows = 0;
    for (BinaryBasicBlock *LP: CurBB->landing_pads()) {
      ReportedThrows += LP->getExecutionCount();
    }

    uint64_t TotalReportedJumps =
      ReportedBranches + ReportedTailCalls + ReportedThrows;

    // Infer the frequency of the fall-through edge, representing not taking the
    // branch.
    uint64_t Inferred = 0;
    if (BBExecCount > TotalReportedJumps)
      Inferred = BBExecCount - TotalReportedJumps;

    DEBUG({
      if (opts::Verbosity >= 1 && BBExecCount < TotalReportedJumps)
        errs()
            << "BOLT-WARNING: Fall-through inference is slightly inconsistent. "
               "exec frequency is less than the outgoing edges frequency ("
            << BBExecCount << " < " << ReportedBranches
            << ") for  BB at offset 0x"
            << Twine::utohexstr(getAddress() + CurBB->getOffset()) << '\n';
    });

    if (CurBB->succ_size() <= 2) {
      // If there is an FT it will be the last successor.
      auto &SuccCount = *CurBB->branch_info_rbegin();
      auto &Succ = *CurBB->succ_rbegin();
      if (SuccCount.Count == BinaryBasicBlock::COUNT_NO_PROFILE) {
        SuccCount.Count = Inferred;
        Succ->ExecutionCount += Inferred;
      }
    }

  } // end for (CurBB : BasicBlocks)

  return;
}

void BinaryFunction::removeConditionalTailCalls() {
  for (auto &I : TailCallTerminatedBlocks) {
    BinaryBasicBlock *BB = I.first;
    TailCallInfo &TCInfo = I.second;

    // Get the conditional tail call instruction.
    MCInst &CondTailCallInst = BB->getInstructionAtIndex(TCInfo.Index);
    if (!BC.MIA->isConditionalBranch(CondTailCallInst)) {
      // The block is not terminated with a conditional tail call.
      continue;
    }

    // Assert that the tail call does not throw.
    const MCSymbol *LP;
    uint64_t Action;
    std::tie(LP, Action) = BC.MIA->getEHInfo(CondTailCallInst);
    assert(!LP && "found tail call with associated landing pad");

    // Create the unconditional tail call instruction.
    const auto *TailCallTargetLabel = BC.MIA->getTargetSymbol(CondTailCallInst);
    assert(TailCallTargetLabel && "symbol expected for direct tail call");
    MCInst TailCallInst;
    BC.MIA->createTailCall(TailCallInst, TailCallTargetLabel, BC.Ctx.get());

    // The way we will remove this conditional tail call depends on the
    // direction of the jump when it is taken. We want to preserve this
    // direction.
    BinaryBasicBlock *TailCallBB = nullptr;
    MCSymbol *TCLabel = BC.Ctx->createTempSymbol("TC", true);
    if (getAddress() >= TCInfo.TargetAddress) {
      // Backward jump: We will reverse the condition of the tail call, change
      // its target to the following (currently fall-through) block, and insert
      // a new block between them that will contain the unconditional tail call.

      // Reverse the condition of the tail call and update its target.
      unsigned InsertIdx = getIndex(BB) + 1;
      assert(InsertIdx < size() && "no fall-through for conditional tail call");
      BinaryBasicBlock *NextBB = BasicBlocks[InsertIdx];

      BC.MIA->reverseBranchCondition(
          CondTailCallInst, NextBB->getLabel(), BC.Ctx.get());

      // Create a basic block containing the unconditional tail call instruction
      // and place it between BB and NextBB.
      std::vector<std::unique_ptr<BinaryBasicBlock>> TailCallBBs;
      TailCallBBs.emplace_back(createBasicBlock(NextBB->getOffset(), TCLabel));
      TailCallBBs[0]->addInstruction(TailCallInst);
      insertBasicBlocks(BB, std::move(TailCallBBs),
                        /* UpdateLayout */ false,
                        /* UpdateCFIState */ false);
      TailCallBB = BasicBlocks[InsertIdx];

      // Add the correct CFI state for the new block.
      TailCallBB->setCFIState(TCInfo.CFIStateBefore);
    } else {
      // Forward jump: we will create a new basic block at the end of the
      // function containing the unconditional tail call and change the target
      // of the conditional tail call to this basic block.

      // Create a basic block containing the unconditional tail call
      // instruction and place it at the end of the function.
      // We have to add 1 byte as there's potentially an existing branch past
      // the end of the code as a result of __builtin_unreachable().
      const BinaryBasicBlock *LastBB = BasicBlocks.back();
      uint64_t NewBlockOffset =
        LastBB->getOffset()
          + BC.computeCodeSize(LastBB->begin(), LastBB->end()) + 1;
      TailCallBB = addBasicBlock(NewBlockOffset, TCLabel);
      TailCallBB->addInstruction(TailCallInst);

      // Add the correct CFI state for the new block. It has to be inserted in
      // the one before last position (the last position holds the CFI state
      // after the last block).
      TailCallBB->setCFIState(TCInfo.CFIStateBefore);

      // Replace the target of the conditional tail call with the label of the
      // new basic block.
      BC.MIA->replaceBranchTarget(CondTailCallInst, TCLabel, BC.Ctx.get());
    }

    // Add CFG edge with profile info from BB to TailCallBB info and swap
    // edges if the TailCallBB corresponds to the taken branch.
    BB->addSuccessor(TailCallBB, TCInfo.Count, TCInfo.Mispreds);
    if (getAddress() < TCInfo.TargetAddress)
      BB->swapConditionalSuccessors();

    // Add execution count for the block.
    if (hasValidProfile())
      TailCallBB->setExecutionCount(TCInfo.Count);
  }
}

uint64_t BinaryFunction::getFunctionScore() {
  if (FunctionScore != -1)
    return FunctionScore;

  uint64_t TotalScore = 0ULL;
  for (auto BB : layout()) {
    uint64_t BBExecCount = BB->getExecutionCount();
    if (BBExecCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      continue;
    BBExecCount *= BB->getNumNonPseudos();
    TotalScore += BBExecCount;
  }
  FunctionScore = TotalScore;
  return FunctionScore;
}

void BinaryFunction::annotateCFIState() {
  assert(CurrentState == State::Disassembled && "unexpected function state");
  assert(!BasicBlocks.empty() && "basic block list should not be empty");

  // This is an index of the last processed CFI in FDE CFI program.
  int32_t State = 0;

  // This is an index of RememberState CFI reflecting effective state right
  // after execution of RestoreState CFI.
  //
  // It differs from State iff the CFI at (State-1)
  // was RestoreState (modulo GNU_args_size CFIs, which are ignored).
  //
  // This allows us to generate shorter replay sequences when producing new
  // CFI programs.
  int32_t EffectiveState = 0;

  // For tracking RememberState/RestoreState sequences.
  std::stack<int32_t> StateStack;

  for (auto *BB : BasicBlocks) {
    BB->setCFIState(EffectiveState);

    // While building the CFG, we want to save the CFI state before a tail call
    // instruction, so that we can correctly remove conditional tail calls.
    auto TCI = TailCallTerminatedBlocks.find(BB);
    bool SaveState = TCI != TailCallTerminatedBlocks.end();

    uint32_t Idx = 0; // instruction index in a current basic block
    for (const auto &Instr : *BB) {
      ++Idx;
      if (SaveState && Idx == TCI->second.Index) {
        TCI->second.CFIStateBefore = EffectiveState;
        SaveState = false;
      }

      const auto *CFI = getCFIFor(Instr);
      if (!CFI)
        continue;

      ++State;

      if (CFI->getOperation() == MCCFIInstruction::OpRememberState) {
        StateStack.push(EffectiveState);
      } else if (CFI->getOperation() == MCCFIInstruction::OpRestoreState) {
        assert(!StateStack.empty() && "corrupt CFI stack");
        EffectiveState = StateStack.top();
        StateStack.pop();
      } else if (CFI->getOperation() != MCCFIInstruction::OpGnuArgsSize) {
        // OpGnuArgsSize CFIs do not affect the CFI state.
        EffectiveState = State;
      }
    }
  }

  assert(StateStack.empty() && "corrupt CFI stack");
}

bool BinaryFunction::fixCFIState() {
  DEBUG(dbgs() << "Trying to fix CFI states for each BB after reordering.\n");
  DEBUG(dbgs() << "This is the list of CFI states for each BB of " << *this
               << ": ");

  auto replayCFIInstrs =
      [this](int32_t FromState, int32_t ToState, BinaryBasicBlock *InBB,
         BinaryBasicBlock::iterator InsertIt) -> bool {
    if (FromState == ToState)
      return true;
    assert(FromState < ToState && "can only replay CFIs forward");

    std::vector<uint32_t> NewCFIs;
    uint32_t NestedLevel = 0;
    for (auto CurState = FromState; CurState < ToState; ++CurState) {
      MCCFIInstruction *Instr = &FrameInstructions[CurState];
      if (Instr->getOperation() == MCCFIInstruction::OpRememberState)
        ++NestedLevel;
      if (!NestedLevel)
        NewCFIs.push_back(CurState);
      if (Instr->getOperation() == MCCFIInstruction::OpRestoreState)
        --NestedLevel;
    }

    // TODO: If in replaying the CFI instructions to reach this state we
    // have state stack instructions, we could still work out the logic
    // to extract only the necessary instructions to reach this state
    // without using the state stack. Not sure if it is worth the effort
    // because this happens rarely.
    if (NestedLevel != 0) {
      errs() << "BOLT-WARNING: CFI rewriter detected nested CFI state"
             << " while replaying CFI instructions for BB "
             << InBB->getName() << " in function " << *this << '\n';
      return false;
    }

    for (auto CFI : NewCFIs) {
      // Ignore GNU_args_size instructions.
      if (FrameInstructions[CFI].getOperation() !=
          MCCFIInstruction::OpGnuArgsSize) {
        InsertIt = addCFIPseudo(InBB, InsertIt, CFI);
        ++InsertIt;
      }
    }

    return true;
  };

  int32_t State = 0;
  auto *FDEStartBB = BasicBlocksLayout[0];
  bool SeenCold = false;
  auto Sep = "";
  (void)Sep;
  for (auto *BB : BasicBlocksLayout) {
    const auto CFIStateAtExit = BB->getCFIStateAtExit();

    // Hot-cold border: check if this is the first BB to be allocated in a cold
    // region (with a different FDE). If yes, we need to reset the CFI state and
    // the FDEStartBB that is used to insert remember_state CFIs.
    if (!SeenCold && BB->isCold()) {
      State = 0;
      FDEStartBB = BB;
      SeenCold = true;
    }

    // We need to recover the correct state if it doesn't match expected
    // state at BB entry point.
    if (BB->getCFIState() < State) {
      // In this case, State is currently higher than what this BB expect it
      // to be. To solve this, we need to insert a CFI instruction to remember
      // the old state at function entry, then another CFI instruction to
      // restore it at the entry of this BB and replay CFI instructions to
      // reach the desired state.
      int32_t OldState = BB->getCFIState();
      // Remember state at function entry point (our reference state).
      auto InsertIt = FDEStartBB->begin();
      while (InsertIt != FDEStartBB->end() && BC.MIA->isCFI(*InsertIt))
        ++InsertIt;
      addCFIPseudo(FDEStartBB, InsertIt, FrameInstructions.size());
      FrameInstructions.emplace_back(
          MCCFIInstruction::createRememberState(nullptr));
      // Restore state
      InsertIt = addCFIPseudo(BB, BB->begin(), FrameInstructions.size());
      ++InsertIt;
      FrameInstructions.emplace_back(
          MCCFIInstruction::createRestoreState(nullptr));
      if (!replayCFIInstrs(0, OldState, BB, InsertIt))
        return false;
      // Check if we messed up the stack in this process
      int StackOffset = 0;
      for (BinaryBasicBlock *CurBB : BasicBlocksLayout) {
        if (CurBB == BB)
          break;
        for (auto &Instr : *CurBB) {
          if (auto *CFI = getCFIFor(Instr)) {
            if (CFI->getOperation() == MCCFIInstruction::OpRememberState)
              ++StackOffset;
            if (CFI->getOperation() == MCCFIInstruction::OpRestoreState)
              --StackOffset;
          }
        }
      }
      auto Pos = BB->begin();
      while (Pos != BB->end() && BC.MIA->isCFI(*Pos)) {
        auto CFI = getCFIFor(*Pos);
        if (CFI->getOperation() == MCCFIInstruction::OpRememberState)
          ++StackOffset;
        if (CFI->getOperation() == MCCFIInstruction::OpRestoreState)
          --StackOffset;
        ++Pos;
      }

      if (StackOffset != 0) {
        errs() << "BOLT-WARNING: not possible to remember/recover state"
               << " without corrupting CFI state stack in function "
               << *this << " @ " << BB->getName() << "\n";
        return false;
      }
    } else if (BB->getCFIState() > State) {
      // If BB's CFI state is greater than State, it means we are behind in the
      // state. Just emit all instructions to reach this state at the
      // beginning of this BB. If this sequence of instructions involve
      // remember state or restore state, bail out.
      if (!replayCFIInstrs(State, BB->getCFIState(), BB, BB->begin()))
        return false;
    }

    State = CFIStateAtExit;
    DEBUG(dbgs() << Sep << State; Sep = ", ");
  }
  DEBUG(dbgs() << "\n");
  return true;
}

void BinaryFunction::modifyLayout(LayoutType Type, bool MinBranchClusters,
                                  bool Split) {
  if (BasicBlocksLayout.empty() || Type == LT_NONE)
    return;

  BasicBlockOrderType NewLayout;
  std::unique_ptr<ReorderAlgorithm> Algo;

  // Cannot do optimal layout without profile.
  if (Type != LT_REVERSE && !hasValidProfile())
    return;

  if (Type == LT_REVERSE) {
    Algo.reset(new ReverseReorderAlgorithm());
  }
  else if (BasicBlocksLayout.size() <= FUNC_SIZE_THRESHOLD &&
           Type != LT_OPTIMIZE_SHUFFLE) {
    // Work on optimal solution if problem is small enough
    DEBUG(dbgs() << "finding optimal block layout for " << *this << "\n");
    Algo.reset(new OptimalReorderAlgorithm());
  }
  else {
    DEBUG(dbgs() << "running block layout heuristics on " << *this << "\n");

    std::unique_ptr<ClusterAlgorithm> CAlgo;
    if (MinBranchClusters)
      CAlgo.reset(new MinBranchGreedyClusterAlgorithm());
    else
      CAlgo.reset(new PHGreedyClusterAlgorithm());

    switch(Type) {
    case LT_OPTIMIZE:
      Algo.reset(new OptimizeReorderAlgorithm(std::move(CAlgo)));
      break;

    case LT_OPTIMIZE_BRANCH:
      Algo.reset(new OptimizeBranchReorderAlgorithm(std::move(CAlgo)));
      break;

    case LT_OPTIMIZE_CACHE:
      Algo.reset(new OptimizeCacheReorderAlgorithm(std::move(CAlgo)));
      break;

    case LT_OPTIMIZE_SHUFFLE:
      Algo.reset(new RandomClusterReorderAlgorithm(std::move(CAlgo)));
      break;

    default:
      llvm_unreachable("unexpected layout type");
    }
  }

  Algo->reorderBasicBlocks(*this, NewLayout);
  if (opts::PrintFuncStat > 0)
    BasicBlocksPreviousLayout = BasicBlocksLayout;
  BasicBlocksLayout.clear();
  BasicBlocksLayout.swap(NewLayout);

  if (Split)
    splitFunction();
}

uint64_t BinaryFunction::getInstructionCount() const {
  uint64_t Count = 0;
  for (auto &Block : BasicBlocksLayout) {
    Count += Block->getNumNonPseudos();
  }
  return Count;
}

bool BinaryFunction::hasLayoutChanged() const {
  assert(opts::PrintFuncStat > 0 && "PrintFuncStat flag is not on");
  return BasicBlocksPreviousLayout != BasicBlocksLayout;
}

uint64_t BinaryFunction::getEditDistance() const {
  assert(opts::PrintFuncStat > 0 && "PrintFuncStat flag is not on");
  const auto LayoutSize = BasicBlocksPreviousLayout.size();
  if (LayoutSize < 2) {
    return 0;
  }

  std::vector<std::vector<uint64_t>> ChangeMatrix(
      LayoutSize + 1, std::vector<uint64_t>(LayoutSize + 1));

  for (uint64_t I = 0; I <= LayoutSize; ++I) {
    ChangeMatrix[I][0] = I;
    ChangeMatrix[0][I] = I;
  }

  for (uint64_t I = 1; I <= LayoutSize; ++I) {
    for (uint64_t J = 1; J <= LayoutSize; ++J) {
      if (BasicBlocksPreviousLayout[I] != BasicBlocksLayout[J]) {
        ChangeMatrix[I][J] =
            std::min(std::min(ChangeMatrix[I - 1][J], ChangeMatrix[I][J - 1]),
                     ChangeMatrix[I - 1][J - 1]) + 1;
      } else {
        ChangeMatrix[I][J] = ChangeMatrix[I - 1][J - 1];
      }
    }
  }

  return ChangeMatrix[LayoutSize][LayoutSize];
}

void BinaryFunction::emitBody(MCStreamer &Streamer, bool EmitColdPart) {
  int64_t CurrentGnuArgsSize = 0;
  for (auto BB : layout()) {
    if (EmitColdPart != BB->isCold())
      continue;

    if (opts::AlignBlocks && BB->getAlignment() > 1)
      Streamer.EmitCodeAlignment(BB->getAlignment());
    Streamer.EmitLabel(BB->getLabel());

    // Remember if last instruction emitted was a prefix
    bool LastIsPrefix = false;
    SMLoc LastLocSeen;
    for (auto I = BB->begin(), E = BB->end(); I != E; ++I) {
      auto &Instr = *I;
      // Handle pseudo instructions.
      if (BC.MIA->isEHLabel(Instr)) {
        const auto *Label = BC.MIA->getTargetSymbol(Instr);
        assert(Instr.getNumOperands() == 1 && Label &&
               "bad EH_LABEL instruction");
        Streamer.EmitLabel(const_cast<MCSymbol *>(Label));
        continue;
      }
      if (BC.MIA->isCFI(Instr)) {
        Streamer.EmitCFIInstruction(*getCFIFor(Instr));
        continue;
      }
      if (opts::UpdateDebugSections && UnitLineTable.first) {
        LastLocSeen = emitLineInfo(Instr.getLoc(), LastLocSeen);
      }

      // Emit GNU_args_size CFIs as necessary.
      if (usesGnuArgsSize() && BC.MIA->isInvoke(Instr)) {
        auto NewGnuArgsSize = BC.MIA->getGnuArgsSize(Instr);
        assert(NewGnuArgsSize >= 0 && "expected non-negative GNU_args_size");
        if (NewGnuArgsSize != CurrentGnuArgsSize) {
          CurrentGnuArgsSize = NewGnuArgsSize;
          Streamer.EmitCFIGnuArgsSize(CurrentGnuArgsSize);
        }
      }

      Streamer.EmitInstruction(Instr, *BC.STI);
      LastIsPrefix = BC.MIA->isPrefix(Instr);
    }
  }
}

void BinaryFunction::emitBodyRaw(MCStreamer *Streamer) {

  // #14998851: Fix gold linker's '--emit-relocs'.
  assert(false &&
         "cannot emit raw body unless relocation accuracy is guaranteed");

  // Raw contents of the function.
  StringRef SectionContents;
  Section.getContents(SectionContents);

  // Raw contents of the function.
  StringRef FunctionContents =
      SectionContents.substr(getAddress() - Section.getAddress(),
                             getSize());

  if (opts::Verbosity)
    outs() << "BOLT-INFO: emitting function " << *this << " in raw ("
           << getSize() << " bytes).\n";

  // We split the function blob into smaller blocks and output relocations
  // and/or labels between them.
  uint64_t FunctionOffset = 0;
  auto LI = Labels.begin();
  auto RI = MoveRelocations.begin();
  while (LI != Labels.end() ||
         RI != MoveRelocations.end()) {
    uint64_t NextLabelOffset = (LI == Labels.end() ? getSize() : LI->first);
    uint64_t NextRelocationOffset =
      (RI == MoveRelocations.end() ? getSize() : RI->first);
    auto NextStop = std::min(NextLabelOffset, NextRelocationOffset);
    assert(NextStop <= getSize() && "internal overflow error");
    if (FunctionOffset < NextStop) {
      Streamer->EmitBytes(
          FunctionContents.slice(FunctionOffset, NextStop));
      FunctionOffset = NextStop;
    }
    if (LI != Labels.end() && FunctionOffset == LI->first) {
      Streamer->EmitLabel(LI->second);
      DEBUG(dbgs() << "BOLT-DEBUG: emitted label " << LI->second->getName()
                   << " at offset 0x" << Twine::utohexstr(LI->first) << '\n');
      ++LI;
    }
    if (RI != MoveRelocations.end() && FunctionOffset == RI->first) {
      auto RelocationSize = RI->second.emit(Streamer);
      DEBUG(dbgs() << "BOLT-DEBUG: emitted relocation for symbol "
                   << RI->second.Symbol->getName() << " at offset 0x"
                   << Twine::utohexstr(RI->first)
                   << " with size " << RelocationSize << '\n');
      FunctionOffset += RelocationSize;
      ++RI;
    }
  }
  assert(FunctionOffset <= getSize() && "overflow error");
  if (FunctionOffset < getSize()) {
    Streamer->EmitBytes(FunctionContents.substr(FunctionOffset));
  }
}

namespace {

#ifndef MAX_PATH
#define MAX_PATH 255
#endif

std::string constructFilename(std::string Filename,
                              std::string Annotation,
                              std::string Suffix) {
  std::replace(Filename.begin(), Filename.end(), '/', '-');
  if (!Annotation.empty()) {
    Annotation.insert(0, "-");
  }
  if (Filename.size() + Annotation.size() + Suffix.size() > MAX_PATH) {
    assert(Suffix.size() + Annotation.size() <= MAX_PATH);
    if (opts::Verbosity >= 1) {
      errs() << "BOLT-WARNING: Filename \"" << Filename << Annotation << Suffix
             << "\" exceeds the " << MAX_PATH << " size limit, truncating.\n";
    }
    Filename.resize(MAX_PATH - (Suffix.size() + Annotation.size()));
  }
  Filename += Annotation;
  Filename += Suffix;
  return Filename;
}

std::string formatEscapes(const std::string& Str) {
  std::string Result;
  for (unsigned I = 0; I < Str.size(); ++I) {
    auto C = Str[I];
    switch (C) {
    case '\n':
      Result += "&#13;";
      break;
    case '"':
      break;
    default:
      Result += C;
      break;
    }
  }
  return Result;
}

}

void BinaryFunction::dumpGraph(raw_ostream& OS) const {
  OS << "strict digraph \"" << getPrintName() << "\" {\n";
  uint64_t Offset = Address;
  for (auto *BB : BasicBlocks) {
    auto LayoutPos = std::find(BasicBlocksLayout.begin(),
                               BasicBlocksLayout.end(),
                               BB);
    unsigned Layout = LayoutPos - BasicBlocksLayout.begin();
    const char* ColdStr = BB->isCold() ? " (cold)" : "";
    OS << format("\"%s\" [label=\"%s%s\\n(C:%lu,O:%lu,I:%u,L:%u:CFI:%u)\"]\n",
                 BB->getName().data(),
                 BB->getName().data(),
                 ColdStr,
                 (BB->ExecutionCount != BinaryBasicBlock::COUNT_NO_PROFILE
                  ? BB->ExecutionCount
                  : 0),
                 BB->getOffset(),
                 getIndex(BB),
                 Layout,
                 BB->getCFIState());
    OS << format("\"%s\" [shape=box]\n", BB->getName().data());
    if (opts::DotToolTipCode) {
      std::string Str;
      raw_string_ostream CS(Str);
      Offset = BC.printInstructions(CS, BB->begin(), BB->end(), Offset, this);
      const auto Code = formatEscapes(CS.str());
      OS << format("\"%s\" [tooltip=\"%s\"]\n",
                   BB->getName().data(),
                   Code.c_str());
    }

    // analyzeBranch is just used to get the names of the branch
    // opcodes.
    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    const bool Success = BB->analyzeBranch(TBB,
                                           FBB,
                                           CondBranch,
                                           UncondBranch);

    const auto *LastInstr = BB->getLastNonPseudoInstr();
    const bool IsJumpTable = LastInstr && BC.MIA->getJumpTable(*LastInstr);

    auto BI = BB->branch_info_begin();
    for (auto *Succ : BB->successors()) {
      std::string Branch;
      if (Success) {
        if (Succ == BB->getConditionalSuccessor(true)) {
          Branch = CondBranch
            ? BC.InstPrinter->getOpcodeName(CondBranch->getOpcode())
            : "TB";
        } else if (Succ == BB->getConditionalSuccessor(false)) {
          Branch = UncondBranch
            ? BC.InstPrinter->getOpcodeName(UncondBranch->getOpcode())
            : "FB";
        } else {
          Branch = "FT";
        }
      }
      if (IsJumpTable) {
        Branch = "JT";
      }
      OS << format("\"%s\" -> \"%s\" [label=\"%s",
                   BB->getName().data(),
                   Succ->getName().data(),
                   Branch.c_str());

      if (BB->getExecutionCount() != COUNT_NO_PROFILE &&
          BI->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED) {
        OS << "\\n(C:" << BI->Count << ",M:" << BI->MispredictedCount << ")";
      } else if (ExecutionCount != COUNT_NO_PROFILE &&
                 BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE) {
        OS << "\\n(IC:" << BI->Count << ")";
      }
      OS << "\"]\n";

      ++BI;
    }
    for (auto *LP : BB->landing_pads()) {
      OS << format("\"%s\" -> \"%s\" [constraint=false style=dashed]\n",
                   BB->getName().data(),
                   LP->getName().data());
    }
  }
  OS << "}\n";
}

void BinaryFunction::viewGraph() const {
  SmallString<MAX_PATH> Filename;
  if (auto EC = sys::fs::createTemporaryFile("bolt-cfg", "dot", Filename)) {
    errs() << "BOLT-ERROR: " << EC.message() << ", unable to create "
           << " bolt-cfg-XXXXX.dot temporary file.\n";
    return;
  }
  dumpGraphToFile(Filename.str());
  if (DisplayGraph(Filename)) {
    errs() << "BOLT-ERROR: Can't display " << Filename << " with graphviz.\n";
  }
  if (auto EC = sys::fs::remove(Filename)) {
    errs() << "BOLT-WARNING: " << EC.message() << ", failed to remove "
           << Filename << "\n";
  }
}

void BinaryFunction::dumpGraphForPass(std::string Annotation) const {
  auto Filename = constructFilename(getPrintName(), Annotation, ".dot");
  outs() << "BOLT-DEBUG: Dumping CFG to " << Filename << "\n";
  dumpGraphToFile(Filename);
}

void BinaryFunction::dumpGraphToFile(std::string Filename) const {
  std::error_code EC;
  raw_fd_ostream of(Filename, EC, sys::fs::F_None);
  if (EC) {
    if (opts::Verbosity >= 1) {
      errs() << "BOLT-WARNING: " << EC.message() << ", unable to open "
             << Filename << " for output.\n";
    }
    return;
  }
  dumpGraph(of);
}

bool BinaryFunction::validateCFG() const {
  bool Valid = true;
  for (auto *BB : BasicBlocks) {
    Valid &= BB->validateSuccessorInvariants();
  }

  if (!Valid)
    return Valid;

  for (auto *BB : BasicBlocks) {
    std::set<BinaryBasicBlock *> Seen;
    for (auto *LPBlock : BB->LandingPads) {
      Valid &= Seen.count(LPBlock) == 0;
      if (!Valid) {
        errs() << "BOLT-WARNING: Duplicate LP seen " << LPBlock->getName()
               << "in " << *this << "\n";
        break;
      }
      Seen.insert(LPBlock);
      auto count = LPBlock->Throwers.count(BB);
      Valid &= (count == 1);
      if (!Valid) {
        errs() << "BOLT-WARNING: Inconsistent landing pad detected in "
               << *this << ": " << LPBlock->getName()
               << " is in LandingPads but not in " << BB->getName()
               << "->Throwers\n";
        break;
      }
    }
  }

  return Valid;
}

void BinaryFunction::fixBranches() {
  auto &MIA = BC.MIA;
  auto *Ctx = BC.Ctx.get();

  for (unsigned I = 0, E = BasicBlocksLayout.size(); I != E; ++I) {
    BinaryBasicBlock *BB = BasicBlocksLayout[I];
    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    if (!BB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch))
      continue;

    // We will create unconditional branch with correct destination if needed.
    if (UncondBranch)
      BB->eraseInstruction(UncondBranch);

    // Basic block that follows the current one in the final layout.
    const BinaryBasicBlock *NextBB = nullptr;
    if (I + 1 != E && BB->isCold() == BasicBlocksLayout[I + 1]->isCold())
      NextBB = BasicBlocksLayout[I + 1];

    if (BB->succ_size() == 1) {
      // __builtin_unreachable() could create a conditional branch that
      // falls-through into the next function - hence the block will have only
      // one valid successor. Since behaviour is undefined - we replace
      // the conditional branch with an unconditional if required.
      if (CondBranch)
        BB->eraseInstruction(CondBranch);
      if (BB->getSuccessor() == NextBB)
        continue;
      BB->addBranchInstruction(BB->getSuccessor());
    } else if (BB->succ_size() == 2) {
      assert(CondBranch && "conditional branch expected");
      const auto *TSuccessor = BB->getConditionalSuccessor(true);
      const auto *FSuccessor = BB->getConditionalSuccessor(false);
      if (NextBB && NextBB == TSuccessor) {
        std::swap(TSuccessor, FSuccessor);
        MIA->reverseBranchCondition(*CondBranch, TSuccessor->getLabel(), Ctx);
        BB->swapConditionalSuccessors();
      } else {
        MIA->replaceBranchTarget(*CondBranch, TSuccessor->getLabel(), Ctx);
      }
      if (TSuccessor == FSuccessor) {
        BB->removeDuplicateConditionalSuccessor(CondBranch);
      }
      if (!NextBB || (NextBB != TSuccessor && NextBB != FSuccessor)) {
        BB->addBranchInstruction(FSuccessor);
      }
    }
    // Cases where the number of successors is 0 (block ends with a
    // terminator) or more than 2 (switch table) don't require branch
    // instruction adjustments.
  }
  assert(validateCFG() && "Invalid CFG detected after fixing branches");
}

void BinaryFunction::splitFunction() {
  bool AllCold = true;
  for (BinaryBasicBlock *BB : BasicBlocksLayout) {
    auto ExecCount = BB->getExecutionCount();
    if (ExecCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      return;
    if (ExecCount != 0)
      AllCold = false;
  }

  if (AllCold)
    return;

  assert(BasicBlocksLayout.size() > 0);

  // Never outline the first basic block.
  BasicBlocks.front()->setCanOutline(false);
  for (auto BB : BasicBlocks) {
    if (!BB->canOutline())
      continue;
    if (BB->getExecutionCount() != 0) {
      BB->setCanOutline(false);
      continue;
    }
    if (hasEHRanges() && !opts::SplitEH) {
      // We cannot move landing pads (or rather entry points for landing
      // pads).
      if (BB->isLandingPad()) {
        BB->setCanOutline(false);
        continue;
      }
      // We cannot move a block that can throw since exception-handling
      // runtime cannot deal with split functions. However, if we can guarantee
      // that the block never throws, it is safe to move the block to
      // decrease the size of the function.
      for (auto &Instr : *BB) {
        if (BC.MIA->isInvoke(Instr)) {
          BB->setCanOutline(false);
          break;
        }
      }
    }
  }

  if (opts::AggressiveSplitting) {
    // All blocks with 0 count that we can move go to the end of the function.
    // Even if they were natural to cluster formation and were seen in-between
    // hot basic blocks.
    std::stable_sort(BasicBlocksLayout.begin(), BasicBlocksLayout.end(),
        [&] (BinaryBasicBlock *A, BinaryBasicBlock *B) {
          return A->canOutline() < B->canOutline();
        });
  } else if (hasEHRanges() && !opts::SplitEH) {
    // Typically functions with exception handling have landing pads at the end.
    // We cannot move beginning of landing pads, but we can move 0-count blocks
    // comprising landing pads to the end and thus facilitate splitting.
    auto FirstLP = BasicBlocksLayout.begin();
    while ((*FirstLP)->isLandingPad())
      ++FirstLP;

    std::stable_sort(FirstLP, BasicBlocksLayout.end(),
        [&] (BinaryBasicBlock *A, BinaryBasicBlock *B) {
          return A->canOutline() < B->canOutline();
        });
  }

  // Separate hot from cold starting from the bottom.
  for (auto I = BasicBlocksLayout.rbegin(), E = BasicBlocksLayout.rend();
       I != E; ++I) {
    BinaryBasicBlock *BB = *I;
    if (!BB->canOutline())
      break;
    BB->setIsCold(true);
    IsSplit = true;
  }
}

void BinaryFunction::propagateGnuArgsSizeInfo() {
  assert(CurrentState == State::CFG && "unexpected function state");

  if (!hasEHRanges() || !usesGnuArgsSize())
    return;

  // The current value of DW_CFA_GNU_args_size affects all following
  // invoke instructions until the next CFI overrides it.
  // It is important to iterate basic blocks in the original order when
  // assigning the value.
  uint64_t CurrentGnuArgsSize = 0;
  for (auto BB : BasicBlocks) {
    for (auto II = BB->begin(); II != BB->end(); ) {
      auto &Instr = *II;
      if (BC.MIA->isCFI(Instr)) {
        auto CFI = getCFIFor(Instr);
        if (CFI->getOperation() == MCCFIInstruction::OpGnuArgsSize) {
          CurrentGnuArgsSize = CFI->getOffset();
          // Delete DW_CFA_GNU_args_size instructions and only regenerate
          // during the final code emission. The information is embedded
          // inside call instructions.
          II = BB->erasePseudoInstruction(II);
          continue;
        }
      } else if (BC.MIA->isInvoke(Instr)) {
        // Add the value of GNU_args_size as an extra operand to invokes.
        BC.MIA->addGnuArgsSize(Instr, CurrentGnuArgsSize);
      }
      ++II;
    }
  }
}

void BinaryFunction::postProcessBranches() {
  if (!isSimple())
    return;
  for (auto *BB : BasicBlocksLayout) {
    auto LastInstrRI = BB->getLastNonPseudo();
    if (BB->succ_size() == 1) {
      if (LastInstrRI != BB->rend() &&
          BC.MIA->isConditionalBranch(*LastInstrRI)) {
        // __builtin_unreachable() could create a conditional branch that
        // falls-through into the next function - hence the block will have only
        // one valid successor. Such behaviour is undefined and thus we remove
        // the conditional branch while leaving a valid successor.
        BB->eraseInstruction(std::next(LastInstrRI.base()));
        DEBUG(dbgs() << "BOLT-DEBUG: erasing conditional branch in "
                     << BB->getName() << " in function " << *this << '\n');
      }
    } else if (BB->succ_size() == 0) {
      // Ignore unreachable basic blocks.
      if (BB->pred_size() == 0 || BB->isLandingPad())
        continue;

      // If it's the basic block that does not end up with a terminator - we
      // insert a return instruction unless it's a call instruction.
      if (LastInstrRI == BB->rend()) {
        DEBUG(dbgs() << "BOLT-DEBUG: at least one instruction expected in BB "
                     << BB->getName() << " in function " << *this << '\n');
        continue;
      }
      if (!BC.MIA->isTerminator(*LastInstrRI) &&
          !BC.MIA->isCall(*LastInstrRI)) {
        DEBUG(dbgs() << "BOLT-DEBUG: adding return to basic block "
                     << BB->getName() << " in function " << *this << '\n');
        MCInst ReturnInstr;
        BC.MIA->createReturn(ReturnInstr);
        BB->addInstruction(ReturnInstr);
      }
    }
  }
  assert(validateCFG() && "invalid CFG");
}

void BinaryFunction::mergeProfileDataInto(BinaryFunction &BF) const {
  // No reason to merge invalid or empty profiles into BF.
  if (!hasValidProfile())
    return;

  // Update function execution count.
  if (getExecutionCount() != BinaryFunction::COUNT_NO_PROFILE) {
    BF.setExecutionCount(BF.getKnownExecutionCount() + getExecutionCount());
  }

  // Since we are merging a valid profile, the new profile should be valid too.
  // It has either already been valid, or it has been cleaned up.
  BF.ProfileMatchRatio = 1.0f;

  // Update basic block and edge counts.
  auto BBMergeI = BF.begin();
  for (BinaryBasicBlock *BB : BasicBlocks) {
    BinaryBasicBlock *BBMerge = &*BBMergeI;
    assert(getIndex(BB) == BF.getIndex(BBMerge));

    // Update basic block count.
    if (BB->getExecutionCount() != BinaryBasicBlock::COUNT_NO_PROFILE) {
      BBMerge->setExecutionCount(
          BBMerge->getKnownExecutionCount() + BB->getExecutionCount());
    }

    // Update edge count for successors of this basic block.
    auto BBMergeSI = BBMerge->succ_begin();
    auto BIMergeI = BBMerge->branch_info_begin();
    auto BII = BB->branch_info_begin();
    for (const auto *BBSucc : BB->successors()) {
      (void)BBSucc;
      assert(getIndex(BBSucc) == BF.getIndex(*BBMergeSI));

      // At this point no branch count should be set to COUNT_NO_PROFILE.
      assert(BII->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
             "unexpected unknown branch profile");
      assert(BIMergeI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
             "unexpected unknown branch profile");

      BIMergeI->Count += BII->Count;

      // When we merge inferred and real fall-through branch data, the merged
      // data is considered inferred.
      if (BII->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED &&
          BIMergeI->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED) {
        BIMergeI->MispredictedCount += BII->MispredictedCount;
      } else {
        BIMergeI->MispredictedCount = BinaryBasicBlock::COUNT_INFERRED;
      }

      ++BBMergeSI;
      ++BII;
      ++BIMergeI;
    }
    assert(BBMergeSI == BBMerge->succ_end());

    ++BBMergeI;
  }
  assert(BBMergeI == BF.end());
}

BinaryFunction::BasicBlockOrderType BinaryFunction::dfs() const {
  BasicBlockOrderType DFS;
  unsigned Index = 0;
  std::stack<BinaryBasicBlock *> Stack;

  // Push entry points to the stack in reverse order.
  //
  // NB: we rely on the original order of entries to match.
  for (auto BBI = layout_rbegin(); BBI != layout_rend(); ++BBI) {
    auto *BB = *BBI;
    if (BB->isEntryPoint())
      Stack.push(BB);
    BB->setLayoutIndex(BinaryBasicBlock::InvalidIndex);
  }

  while (!Stack.empty()) {
    auto *BB = Stack.top();
    Stack.pop();

    if (BB->getLayoutIndex() != BinaryBasicBlock::InvalidIndex)
      continue;

    BB->setLayoutIndex(Index++);
    DFS.push_back(BB);

    for (auto *SuccBB : BB->landing_pads()) {
      Stack.push(SuccBB);
    }

    for (auto *SuccBB : BB->successors()) {
      Stack.push(SuccBB);
    }
  }

  return DFS;
}

bool BinaryFunction::isIdenticalWith(const BinaryFunction &OtherBF,
                                     bool IgnoreSymbols,
                                     bool UseDFS) const {
  assert(hasCFG() && OtherBF.hasCFG() && "both functions should have CFG");

  // Compare the two functions, one basic block at a time.
  // Currently we require two identical basic blocks to have identical
  // instruction sequences and the same index in their corresponding
  // functions. The latter is important for CFG equality.

  if (layout_size() != OtherBF.layout_size())
    return false;

  // Comparing multi-entry functions could be non-trivial.
  if (isMultiEntry() || OtherBF.isMultiEntry())
    return false;

  // Process both functions in either DFS or existing order.
  const auto &Order = UseDFS ? dfs() : BasicBlocksLayout;
  const auto &OtherOrder = UseDFS ? OtherBF.dfs() : OtherBF.BasicBlocksLayout;

  auto BBI = OtherOrder.begin();
  for (const auto *BB : Order) {
    const auto *OtherBB = *BBI;

    if (BB->getLayoutIndex() != OtherBB->getLayoutIndex())
      return false;

    // Compare successor basic blocks.
    // NOTE: the comparison for jump tables is only partially verified here.
    if (BB->succ_size() != OtherBB->succ_size())
      return false;

    auto SuccBBI = OtherBB->succ_begin();
    for (const auto *SuccBB : BB->successors()) {
      const auto *SuccOtherBB = *SuccBBI;
      if (SuccBB->getLayoutIndex() != SuccOtherBB->getLayoutIndex())
        return false;
      ++SuccBBI;
    }

    // Compare all instructions including pseudos.
    auto I = BB->begin(), E = BB->end();
    auto OtherI = OtherBB->begin(), OtherE = OtherBB->end();
    while (I != E && OtherI != OtherE) {

      bool Identical;
      if (IgnoreSymbols) {
        Identical =
          isInstrEquivalentWith(*I, *BB, *OtherI, *OtherBB, OtherBF,
                                [](const MCSymbol *A, const MCSymbol *B) {
                                  return true;
                                });
      } else {
        // Compare symbols.
        auto AreSymbolsIdentical = [&] (const MCSymbol *A, const MCSymbol *B) {
          if (A == B)
            return true;

          // All local symbols are considered identical since they affect a
          // control flow and we check the control flow separately.
          // If a local symbol is escaped, then the function (potentially) has
          // multiple entry points and we exclude such functions from
          // comparison.
          if (A->isTemporary() && B->isTemporary())
            return true;

          // Compare symbols as functions.
          const auto *FunctionA = BC.getFunctionForSymbol(A);
          const auto *FunctionB = BC.getFunctionForSymbol(B);
          if (FunctionA && FunctionB) {
            // Self-referencing functions and recursive calls.
            if (FunctionA == this && FunctionB == &OtherBF)
              return true;
            return FunctionA == FunctionB;
          }

          // Check if symbols are jump tables.
          auto SIA = BC.GlobalSymbols.find(A->getName());
          if (SIA == BC.GlobalSymbols.end())
            return false;
          auto SIB = BC.GlobalSymbols.find(B->getName());
          if (SIB == BC.GlobalSymbols.end())
            return false;

          assert((SIA->second != SIB->second) &&
                 "different symbols should not have the same value");

          const auto *JumpTableA = getJumpTableContainingAddress(SIA->second);
          if (!JumpTableA)
            return false;
          const auto *JumpTableB =
            OtherBF.getJumpTableContainingAddress(SIB->second);
          if (!JumpTableB)
            return false;

          if ((SIA->second - JumpTableA->Address) !=
              (SIB->second - JumpTableB->Address))
            return false;

          return equalJumpTables(JumpTableA, JumpTableB, OtherBF);
        };

        Identical =
          isInstrEquivalentWith(*I, *BB, *OtherI, *OtherBB, OtherBF,
                                AreSymbolsIdentical);
      }

      if (!Identical)
        return false;

      ++I; ++OtherI;
    }

    // One of the identical blocks may have a trailing unconditional jump that
    // is ignored for CFG purposes.
    auto *TrailingInstr = (I != E ? &(*I)
                                  : (OtherI != OtherE ? &(*OtherI) : 0));
    if (TrailingInstr && !BC.MIA->isUnconditionalBranch(*TrailingInstr)) {
      return false;
    }

    ++BBI;
  }

  return true;
}

bool BinaryFunction::equalJumpTables(const JumpTable *JumpTableA,
                                     const JumpTable *JumpTableB,
                                     const BinaryFunction &BFB) const {
  if (JumpTableA->EntrySize != JumpTableB->EntrySize)
    return false;

  if (JumpTableA->Type != JumpTableB->Type)
    return false;

  if (JumpTableA->getSize() != JumpTableB->getSize())
    return false;

  for (uint64_t Index = 0; Index < JumpTableA->Entries.size(); ++Index) {
    const auto *LabelA = JumpTableA->Entries[Index];
    const auto *LabelB = JumpTableB->Entries[Index];

    const auto *TargetA = getBasicBlockForLabel(LabelA);
    const auto *TargetB = BFB.getBasicBlockForLabel(LabelB);

    if (!TargetA || !TargetB) {
      assert((TargetA || LabelA == getFunctionEndLabel()) &&
             "no target basic block found");
      assert((TargetB || LabelB == BFB.getFunctionEndLabel()) &&
             "no target basic block found");

      if (TargetA != TargetB)
        return false;

      continue;
    }

    assert(TargetA && TargetB && "cannot locate target block(s)");

    if (TargetA->getLayoutIndex() != TargetB->getLayoutIndex())
      return false;
  }

  return true;
}

std::size_t BinaryFunction::hash(bool Recompute, bool UseDFS) const {
  assert(hasCFG() && "function is expected to have CFG");

  if (!Recompute)
    return Hash;

  const auto &Order = UseDFS ? dfs() : BasicBlocksLayout;

  // The hash is computed by creating a string of all the opcodes
  // in the function and hashing that string with std::hash.
  std::string Opcodes;
  for (const auto *BB : Order) {
    for (const auto &Inst : *BB) {
      unsigned Opcode = Inst.getOpcode();

      if (BC.MII->get(Opcode).isPseudo())
        continue;

      // Ignore unconditional jumps since we check CFG consistency by processing
      // basic blocks in order and do not rely on branches to be in-sync with
      // CFG. Note that we still use condition code of conditional jumps.
      if (BC.MIA->isUnconditionalBranch(Inst))
        continue;

      if (Opcode == 0) {
        Opcodes.push_back(0);
        continue;
      }

      while (Opcode) {
        uint8_t LSB = Opcode & 0xff;
        Opcodes.push_back(LSB);
        Opcode = Opcode >> 8;
      }
    }
  }

  return Hash = std::hash<std::string>{}(Opcodes);
}

void BinaryFunction::insertBasicBlocks(
  BinaryBasicBlock *Start,
  std::vector<std::unique_ptr<BinaryBasicBlock>> &&NewBBs,
  const bool UpdateLayout,
  const bool UpdateCFIState) {
  const auto StartIndex = getIndex(Start);
  const auto NumNewBlocks = NewBBs.size();

  BasicBlocks.insert(BasicBlocks.begin() + StartIndex + 1,
                     NumNewBlocks,
                     nullptr);

  auto I = StartIndex + 1;
  for (auto &BB :  NewBBs) {
    assert(!BasicBlocks[I]);
    BasicBlocks[I++] = BB.release();
  }

  updateBBIndices(StartIndex);

  recomputeLandingPads(StartIndex, NumNewBlocks + 1);

  // Make sure the basic blocks are sorted properly.
  assert(std::is_sorted(begin(), end()));

  if (UpdateLayout) {
    updateLayout(Start, NumNewBlocks);
  }

  if (UpdateCFIState) {
    updateCFIState(Start, NumNewBlocks);
  }
}

void BinaryFunction::updateBBIndices(const unsigned StartIndex) {
  for (auto I = StartIndex; I < BasicBlocks.size(); ++I) {
    BasicBlocks[I]->Index = I;
  }
}

void BinaryFunction::updateCFIState(BinaryBasicBlock *Start,
                                    const unsigned NumNewBlocks) {
  assert(TailCallTerminatedBlocks.empty());
  const auto CFIState = Start->getCFIStateAtExit();
  const auto StartIndex = getIndex(Start) + 1;
  for (unsigned I = 0; I < NumNewBlocks; ++I) {
    BasicBlocks[StartIndex + I]->setCFIState(CFIState);
  }
}

void BinaryFunction::updateLayout(BinaryBasicBlock* Start,
                                  const unsigned NumNewBlocks) {
  // Insert new blocks in the layout immediately after Start.
  auto Pos = std::find(layout_begin(), layout_end(), Start);
  assert(Pos != layout_end());
  auto Begin = &BasicBlocks[getIndex(Start) + 1];
  auto End = &BasicBlocks[getIndex(Start) + NumNewBlocks + 1];
  BasicBlocksLayout.insert(Pos + 1, Begin, End);
  updateLayoutIndices();
}

void BinaryFunction::updateLayout(LayoutType Type,
                                  bool MinBranchClusters,
                                  bool Split) {
  // Recompute layout with original parameters.
  BasicBlocksLayout = BasicBlocks;
  modifyLayout(Type, MinBranchClusters, Split);
  updateLayoutIndices();
}

bool BinaryFunction::replaceJumpTableEntryIn(BinaryBasicBlock *BB,
                                             BinaryBasicBlock *OldDest,
                                             BinaryBasicBlock *NewDest) {
  auto *Instr = BB->getLastNonPseudoInstr();
  if (!Instr || !BC.MIA->isIndirectBranch(*Instr))
    return false;
  auto JTAddress = BC.MIA->getJumpTable(*Instr);
  assert(JTAddress && "Invalid jump table address");
  auto *JT = getJumpTableContainingAddress(JTAddress);
  assert(JT && "No jump table structure for this indirect branch");
  bool Patched = JT->replaceDestination(JTAddress, OldDest->getLabel(),
                                        NewDest->getLabel());
  assert(Patched && "Invalid entry to be replaced in jump table");
  return true;
}

BinaryBasicBlock *BinaryFunction::splitEdge(BinaryBasicBlock *From,
                                            BinaryBasicBlock *To) {
  // Create intermediate BB
  MCSymbol *Tmp = BC.Ctx->createTempSymbol("SplitEdge", true);
  auto NewBB = createBasicBlock(0, Tmp);
  auto NewBBPtr = NewBB.get();

  // Update "From" BB
  auto I = From->succ_begin();
  auto BI = From->branch_info_begin();
  for (; I != From->succ_end(); ++I) {
    if (*I == To)
      break;
    ++BI;
  }
  assert(I != From->succ_end() && "Invalid CFG edge in splitEdge!");
  uint64_t OrigCount{BI->Count};
  uint64_t OrigMispreds{BI->MispredictedCount};
  replaceJumpTableEntryIn(From, To, NewBBPtr);
  From->replaceSuccessor(To, NewBBPtr, OrigCount, OrigMispreds);

  NewBB->addSuccessor(To, OrigCount, OrigMispreds);
  NewBB->setExecutionCount(OrigCount);
  NewBB->setIsCold(From->isCold());

  // Update CFI and BB layout with new intermediate BB
  std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
  NewBBs.emplace_back(std::move(NewBB));
  insertBasicBlocks(From, std::move(NewBBs), true, true);
  return NewBBPtr;
}

bool BinaryFunction::isSymbolValidInScope(const SymbolRef &Symbol,
                                          uint64_t SymbolSize) const {
  // Some symbols are tolerated inside function bodies, others are not.
  // The real function boundaries may not be known at this point.

  // It's okay to have a zero-sized symbol in the middle of non-zero-sized
  // function.
  if (SymbolSize == 0 && containsAddress(*Symbol.getAddress()))
    return true;

  if (Symbol.getType() != SymbolRef::ST_Unknown)
    return false;

  if (Symbol.getFlags() & SymbolRef::SF_Global)
    return false;

  return true;
}

SMLoc BinaryFunction::emitLineInfo(SMLoc NewLoc, SMLoc PrevLoc) const {
  auto *FunctionCU = UnitLineTable.first;
  const auto *FunctionLineTable = UnitLineTable.second;
  assert(FunctionCU && "cannot emit line info for function without CU");

  auto RowReference = DebugLineTableRowRef::fromSMLoc(NewLoc);

  // Check if no new line info needs to be emitted.
  if (RowReference == DebugLineTableRowRef::NULL_ROW ||
      NewLoc.getPointer() == PrevLoc.getPointer())
    return PrevLoc;

  unsigned CurrentFilenum = 0;
  const auto *CurrentLineTable = FunctionLineTable;

  // If the CU id from the current instruction location does not
  // match the CU id from the current function, it means that we
  // have come across some inlined code.  We must look up the CU
  // for the instruction's original function and get the line table
  // from that.
  const auto FunctionUnitIndex = FunctionCU->getOffset();
  const auto CurrentUnitIndex = RowReference.DwCompileUnitIndex;
  if (CurrentUnitIndex != FunctionUnitIndex) {
    CurrentLineTable = BC.DwCtx->getLineTableForUnit(
        BC.DwCtx->getCompileUnitForOffset(CurrentUnitIndex));
    // Add filename from the inlined function to the current CU.
    CurrentFilenum =
      BC.addDebugFilenameToUnit(FunctionUnitIndex, CurrentUnitIndex,
        CurrentLineTable->Rows[RowReference.RowIndex - 1].File);
  }

  const auto &CurrentRow = CurrentLineTable->Rows[RowReference.RowIndex - 1];
  if (!CurrentFilenum)
    CurrentFilenum = CurrentRow.File;

  BC.Ctx->setCurrentDwarfLoc(
    CurrentFilenum,
    CurrentRow.Line,
    CurrentRow.Column,
    (DWARF2_FLAG_IS_STMT * CurrentRow.IsStmt) |
    (DWARF2_FLAG_BASIC_BLOCK * CurrentRow.BasicBlock) |
    (DWARF2_FLAG_PROLOGUE_END * CurrentRow.PrologueEnd) |
    (DWARF2_FLAG_EPILOGUE_BEGIN * CurrentRow.EpilogueBegin),
    CurrentRow.Isa,
    CurrentRow.Discriminator);
  BC.Ctx->setDwarfCompileUnitID(FunctionUnitIndex);

  return NewLoc;
}

BinaryFunction::~BinaryFunction() {
  for (auto BB : BasicBlocks) {
    delete BB;
  }
  for (auto BB : DeletedBasicBlocks) {
    delete BB;
  }
}

void BinaryFunction::emitJumpTables(MCStreamer *Streamer) {
  if (JumpTables.empty())
    return;
  if (opts::PrintJumpTables) {
    outs() << "BOLT-INFO: jump tables for function " << *this << ":\n";
  }
  for (auto &JTI : JumpTables) {
    auto &JT = JTI.second;
    if (opts::PrintJumpTables)
      JT.print(outs());
    if (opts::JumpTables == JTS_BASIC && opts::Relocs) {
      JT.updateOriginal(BC);
    } else {
      MCSection *HotSection, *ColdSection;
      if (opts::JumpTables == JTS_BASIC) {
        JT.SectionName =
                  ".local.JUMP_TABLEat0x" + Twine::utohexstr(JT.Address).str();
        HotSection = BC.Ctx->getELFSection(JT.SectionName,
                                           ELF::SHT_PROGBITS,
                                           ELF::SHF_ALLOC);
        ColdSection = HotSection;
      } else {
        HotSection = BC.MOFI->getReadOnlySection();
        ColdSection = BC.MOFI->getReadOnlyColdSection();
      }
      JT.emit(Streamer, HotSection, ColdSection);
    }
  }
}

std::pair<size_t, size_t>
BinaryFunction::JumpTable::getEntriesForAddress(const uint64_t Addr) const {
  const uint64_t InstOffset = Addr - Address;
  size_t StartIndex = 0, EndIndex = 0;
  uint64_t Offset = 0;

  for (size_t I = 0; I < Entries.size(); ++I) {
    auto LI = Labels.find(Offset);
    if (LI != Labels.end()) {
      const auto NextLI = std::next(LI);
      const auto NextOffset =
        NextLI == Labels.end() ? getSize() : NextLI->first;
      if (InstOffset >= LI->first && InstOffset < NextOffset) {
        StartIndex = I;
        EndIndex = I;
        while (Offset < NextOffset) {
          ++EndIndex;
          Offset += EntrySize;
        }
        break;
      }
    }
    Offset += EntrySize;
  }

  return std::make_pair(StartIndex, EndIndex);
}

bool BinaryFunction::JumpTable::replaceDestination(uint64_t JTAddress,
                                                   const MCSymbol *OldDest,
                                                   MCSymbol *NewDest) {
  bool Patched{false};
  const auto Range = getEntriesForAddress(JTAddress);
  for (auto I = &Entries[Range.first], E = &Entries[Range.second];
       I != E; ++I) {
    auto &Entry = *I;
    if (Entry == OldDest) {
      Patched = true;
      Entry = NewDest;
    }
  }
  return Patched;
}

void BinaryFunction::JumpTable::updateOriginal(BinaryContext &BC) {
  // In non-relocation mode we have to emit jump tables in local sections.
  // This way we only overwrite them when a corresponding function is
  // overwritten.
  assert(opts::Relocs && "relocation mode expected");
  auto SectionOrError = BC.getSectionForAddress(Address);
  assert(SectionOrError && "section not found for jump table");
  auto Section = SectionOrError.get();
  uint64_t Offset = Address - Section.getAddress();
  StringRef SectionName;
  Section.getName(SectionName);
  for (auto *Entry : Entries) {
    const auto RelType = (Type == JTT_NORMAL) ? ELF::R_X86_64_64
                                              : ELF::R_X86_64_PC32;
    const uint64_t RelAddend = (Type == JTT_NORMAL)
        ? 0 : Offset - (Address - Section.getAddress());
    DEBUG(dbgs() << "adding relocation to section " << SectionName
                 << " at offset " << Twine::utohexstr(Offset) << " for symbol "
                 << Entry->getName() << " with addend "
                 << Twine::utohexstr(RelAddend) << '\n');
    BC.addSectionRelocation(Section, Offset, Entry, RelType, RelAddend);
    Offset += EntrySize;
  }
}

uint64_t BinaryFunction::JumpTable::emit(MCStreamer *Streamer,
                                         MCSection *HotSection,
                                         MCSection *ColdSection) {
  // Pre-process entries for aggressive splitting.
  // Each label represents a separate switch table and gets its own count
  // determining its destination.
  std::map<MCSymbol *, uint64_t> LabelCounts;
  if (opts::JumpTables > JTS_SPLIT && !Counts.empty()) {
    MCSymbol *CurrentLabel = Labels[0];
    uint64_t CurrentLabelCount = 0;
    for (unsigned Index = 0; Index < Entries.size(); ++Index) {
      auto LI = Labels.find(Index * EntrySize);
      if (LI != Labels.end()) {
        LabelCounts[CurrentLabel] = CurrentLabelCount;
        CurrentLabel = LI->second;
        CurrentLabelCount = 0;
      }
      CurrentLabelCount += Counts[Index].Count;
    }
    LabelCounts[CurrentLabel] = CurrentLabelCount;
  } else {
    Streamer->SwitchSection(Count > 0 ? HotSection : ColdSection);
    Streamer->EmitValueToAlignment(EntrySize);
  }
  MCSymbol *LastLabel = nullptr;
  uint64_t Offset = 0;
  for (auto *Entry : Entries) {
    auto LI = Labels.find(Offset);
    if (LI != Labels.end()) {
      DEBUG(dbgs() << "BOLT-DEBUG: emitting jump table "
                   << LI->second->getName() << " (originally was at address 0x"
                   << Twine::utohexstr(Address + Offset)
                   << (Offset ? "as part of larger jump table\n" : "\n"));
      if (!LabelCounts.empty()) {
        DEBUG(dbgs() << "BOLT-DEBUG: jump table count: "
                     << LabelCounts[LI->second] << '\n');
        if (LabelCounts[LI->second] > 0) {
          Streamer->SwitchSection(HotSection);
        } else {
          Streamer->SwitchSection(ColdSection);
        }
        Streamer->EmitValueToAlignment(EntrySize);
      }
      Streamer->EmitLabel(LI->second);
      LastLabel = LI->second;
    }
    if (Type == JTT_NORMAL) {
      Streamer->EmitSymbolValue(Entry, EntrySize);
    } else { // JTT_PIC
      auto JT = MCSymbolRefExpr::create(LastLabel, Streamer->getContext());
      auto E = MCSymbolRefExpr::create(Entry, Streamer->getContext());
      auto Value = MCBinaryExpr::createSub(E, JT, Streamer->getContext());
      Streamer->EmitValue(Value, EntrySize);
    }
    Offset += EntrySize;
  }

  return Offset;
}

void BinaryFunction::JumpTable::print(raw_ostream &OS) const {
  uint64_t Offset = 0;
  for (const auto *Entry : Entries) {
    auto LI = Labels.find(Offset);
    if (LI != Labels.end()) {
      OS << "Jump Table " << LI->second->getName() << " at @0x"
         << Twine::utohexstr(Address+Offset);
      if (Offset) {
        OS << " (possibly part of larger jump table):\n";
      } else {
        OS << " with total count of " << Count << ":\n";
      }
    }
    OS << format("  0x%04" PRIx64 " : ", Offset) << Entry->getName();
    if (!Counts.empty()) {
      OS << " : " << Counts[Offset / EntrySize].Mispreds
         << "/" << Counts[Offset / EntrySize].Count;
    }
    OS << '\n';
    Offset += EntrySize;
  }
  OS << "\n\n";
}

void BinaryFunction::calculateLoopInfo() {
  // Discover loops.
  BinaryDominatorTree DomTree(false);
  DomTree.recalculate<BinaryFunction>(*this);
  BLI.reset(new BinaryLoopInfo());
  BLI->analyze(DomTree);

  // Traverse discovered loops and add depth and profile information.
  std::stack<BinaryLoop *> St;
  for (auto I = BLI->begin(), E = BLI->end(); I != E; ++I) {
    St.push(*I);
    ++BLI->OuterLoops;
  }

  while (!St.empty()) {
    BinaryLoop *L = St.top();
    St.pop();
    ++BLI->TotalLoops;
    BLI->MaximumDepth = std::max(L->getLoopDepth(), BLI->MaximumDepth);

    // Add nested loops in the stack.
    for (BinaryLoop::iterator I = L->begin(), E = L->end(); I != E; ++I) {
      St.push(*I);
    }

    // Skip if no valid profile is found.
    if (!hasValidProfile()) {
      L->EntryCount = COUNT_NO_PROFILE;
      L->ExitCount = COUNT_NO_PROFILE;
      L->TotalBackEdgeCount = COUNT_NO_PROFILE;
      continue;
    }

    // Compute back edge count.
    SmallVector<BinaryBasicBlock *, 1> Latches;
    L->getLoopLatches(Latches);

    for (BinaryBasicBlock *Latch : Latches) {
      auto BI = Latch->branch_info_begin();
      for (BinaryBasicBlock *Succ : Latch->successors()) {
        if (Succ == L->getHeader()) {
          assert(BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
                 "profile data not found");
          L->TotalBackEdgeCount += BI->Count;
        }
        ++BI;
      }
    }

    // Compute entry count.
    L->EntryCount = L->getHeader()->getExecutionCount() - L->TotalBackEdgeCount;

    // Compute exit count.
    SmallVector<BinaryLoop::Edge, 1> ExitEdges;
    L->getExitEdges(ExitEdges);
    for (BinaryLoop::Edge &Exit : ExitEdges) {
      const BinaryBasicBlock *Exiting = Exit.first;
      const BinaryBasicBlock *ExitTarget = Exit.second;
      auto BI = Exiting->branch_info_begin();
      for (BinaryBasicBlock *Succ : Exiting->successors()) {
        if (Succ == ExitTarget) {
          assert(BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
                 "profile data not found");
          L->ExitCount += BI->Count;
        }
        ++BI;
      }
    }
  }
}

DWARFAddressRangesVector BinaryFunction::getOutputAddressRanges() const {
  DWARFAddressRangesVector OutputRanges;

  OutputRanges.emplace_back(getOutputAddress(),
                            getOutputAddress() + getOutputSize());
  if (isSplit()) {
    assert(isEmitted() && "split function should be emitted");
    OutputRanges.emplace_back(cold().getAddress(),
                              cold().getAddress() + cold().getImageSize());
  }

  return OutputRanges;
}

uint64_t BinaryFunction::translateInputToOutputAddress(uint64_t Address) const {
  // If the function hasn't changed return the same address.
  if (!isEmitted() && !opts::Relocs)
    return Address;

  if (Address < getAddress())
    return 0;

  // FIXME: #18950828 - we rely on relative offsets inside basic blocks to stay
  //        intact. Instead we can use pseudo instructions and/or annotations.
  const auto Offset = Address - getAddress();
  const auto *BB = getBasicBlockContainingOffset(Offset);
  if (!BB) {
    // Special case for address immediately past the end of the function.
    if (Offset == getSize())
      return getOutputAddress() + getOutputSize();

    return 0;
  }

  return std::min(BB->getOutputAddressRange().first + Offset - BB->getOffset(),
                  BB->getOutputAddressRange().second);
}

DWARFAddressRangesVector BinaryFunction::translateInputToOutputRanges(
    const DWARFAddressRangesVector &InputRanges) const {
  // If the function hasn't changed return the same ranges.
  if (!isEmitted() && !opts::Relocs)
    return InputRanges;

  // Even though we will merge ranges in a post-processing pass, we attempt to
  // merge them in a main processing loop as it improves the processing time.
  uint64_t PrevEndAddress = 0;
  DWARFAddressRangesVector OutputRanges;
  for (const auto &Range : InputRanges) {
    if (!containsAddress(Range.first)) {
      DEBUG(dbgs() << "BOLT-DEBUG: invalid debug address range detected for "
                   << *this << " : [0x" << Twine::utohexstr(Range.first)
                   << ", 0x" << Twine::utohexstr(Range.second) << "]\n");
      PrevEndAddress = 0;
      continue;
    }
    auto InputOffset = Range.first - getAddress();
    const auto InputEndOffset = std::min(Range.second - getAddress(), getSize());

    auto BBI = std::upper_bound(BasicBlockOffsets.begin(),
                                BasicBlockOffsets.end(),
                                BasicBlockOffset(InputOffset, nullptr),
                                CompareBasicBlockOffsets());
    --BBI;
    do {
      const auto *BB = BBI->second;
      if (InputOffset < BB->getOffset() || InputOffset >= BB->getEndOffset()) {
        DEBUG(dbgs() << "BOLT-DEBUG: invalid debug address range detected for "
                     << *this << " : [0x" << Twine::utohexstr(Range.first)
                     << ", 0x" << Twine::utohexstr(Range.second) << "]\n");
        PrevEndAddress = 0;
        break;
      }

      // Skip the range if the block was deleted.
      if (const auto OutputStart = BB->getOutputAddressRange().first) {
        const auto StartAddress = OutputStart + InputOffset - BB->getOffset();
        auto EndAddress = BB->getOutputAddressRange().second;
        if (InputEndOffset < BB->getEndOffset())
          EndAddress = StartAddress + InputEndOffset - InputOffset;

        if (StartAddress == PrevEndAddress) {
          OutputRanges.back().second = std::max(OutputRanges.back().second,
                                                EndAddress);
        } else {
          OutputRanges.emplace_back(StartAddress,
                                    std::max(StartAddress, EndAddress));
        }
        PrevEndAddress = OutputRanges.back().second;
      }

      InputOffset = BB->getEndOffset();
      ++BBI;
    } while (InputOffset < InputEndOffset);
  }

  // Post-processing pass to sort and merge ranges.
  std::sort(OutputRanges.begin(), OutputRanges.end());
  DWARFAddressRangesVector MergedRanges;
  PrevEndAddress = 0;
  for(const auto &Range : OutputRanges) {
    if (Range.first <= PrevEndAddress) {
      MergedRanges.back().second = std::max(MergedRanges.back().second,
                                            Range.second);
    } else {
      MergedRanges.emplace_back(Range.first, Range.second);
    }
    PrevEndAddress = MergedRanges.back().second;
  }

  return MergedRanges;
}

DWARFDebugLoc::LocationList BinaryFunction::translateInputToOutputLocationList(
      const DWARFDebugLoc::LocationList &InputLL,
      uint64_t BaseAddress) const {
  // If the function wasn't changed - there's nothing to update.
  if (!isEmitted() && !opts::Relocs) {
    if (!BaseAddress) {
      return InputLL;
    } else {
      auto OutputLL = std::move(InputLL);
      for (auto &Entry : OutputLL.Entries) {
        Entry.Begin += BaseAddress;
        Entry.End += BaseAddress;
      }
      return OutputLL;
    }
  }

  uint64_t PrevEndAddress = 0;
  SmallVectorImpl<unsigned char> *PrevLoc = nullptr;
  DWARFDebugLoc::LocationList OutputLL;
  for (auto &Entry : InputLL.Entries) {
    const auto Start = Entry.Begin + BaseAddress;
    const auto End = Entry.End + BaseAddress;
    if (!containsAddress(Start)) {
      DEBUG(dbgs() << "BOLT-DEBUG: invalid debug address range detected for "
                   << *this << " : [0x" << Twine::utohexstr(Start)
                   << ", 0x" << Twine::utohexstr(End) << "]\n");
      continue;
    }
    auto InputOffset = Start - getAddress();
    const auto InputEndOffset = std::min(End - getAddress(), getSize());
    auto BBI = std::upper_bound(BasicBlockOffsets.begin(),
                                BasicBlockOffsets.end(),
                                BasicBlockOffset(InputOffset, nullptr),
                                CompareBasicBlockOffsets());
    --BBI;
    do {
      const auto *BB = BBI->second;
      if (InputOffset < BB->getOffset() || InputOffset >= BB->getEndOffset()) {
        DEBUG(dbgs() << "BOLT-DEBUG: invalid debug address range detected for "
                     << *this << " : [0x" << Twine::utohexstr(Start)
                     << ", 0x" << Twine::utohexstr(End) << "]\n");
        PrevEndAddress = 0;
        break;
      }

      // Skip the range if the block was deleted.
      if (const auto OutputStart = BB->getOutputAddressRange().first) {
        const auto StartAddress = OutputStart + InputOffset - BB->getOffset();
        auto EndAddress = BB->getOutputAddressRange().second;
        if (InputEndOffset < BB->getEndOffset())
          EndAddress = StartAddress + InputEndOffset - InputOffset;

        if (StartAddress == PrevEndAddress && Entry.Loc == *PrevLoc) {
          OutputLL.Entries.back().End = std::max(OutputLL.Entries.back().End,
                                                 EndAddress);
        } else {
          OutputLL.Entries.emplace_back(
              DWARFDebugLoc::Entry{StartAddress,
                                   std::max(StartAddress, EndAddress),
                                   Entry.Loc});
        }
        PrevEndAddress = OutputLL.Entries.back().End;
        PrevLoc = &OutputLL.Entries.back().Loc;
      }

      ++BBI;
      InputOffset = BB->getEndOffset();
    } while (InputOffset < InputEndOffset);
  }

  // Sort and merge adjacent entries with identical location.
  std::stable_sort(OutputLL.Entries.begin(), OutputLL.Entries.end(),
      [] (const DWARFDebugLoc::Entry &A, const DWARFDebugLoc::Entry &B) {
        return A.Begin < B.Begin;
      });
  DWARFDebugLoc::LocationList MergedLL;
  PrevEndAddress = 0;
  PrevLoc = nullptr;
  for(const auto &Entry : OutputLL.Entries) {
    if (Entry.Begin <= PrevEndAddress && *PrevLoc == Entry.Loc) {
      MergedLL.Entries.back().End = std::max(Entry.End,
                                             MergedLL.Entries.back().End);;
    } else {
      const auto Begin = std::max(Entry.Begin, PrevEndAddress);
      const auto End = std::max(Begin, Entry.End);
      MergedLL.Entries.emplace_back(DWARFDebugLoc::Entry{Begin,
                                                         End,
                                                         Entry.Loc});
    }
    PrevEndAddress = MergedLL.Entries.back().End;
    PrevLoc = &MergedLL.Entries.back().Loc;
  }

  return MergedLL;
}

void BinaryFunction::printLoopInfo(raw_ostream &OS) const {
  OS << "Loop Info for Function \"" << *this << "\"";
  if (hasValidProfile()) {
    OS << " (count: " << getExecutionCount() << ")";
  }
  OS << "\n";

  std::stack<BinaryLoop *> St;
  for (auto I = BLI->begin(), E = BLI->end(); I != E; ++I) {
    St.push(*I);
  }
  while (!St.empty()) {
    BinaryLoop *L = St.top();
    St.pop();

    for (BinaryLoop::iterator I = L->begin(), E = L->end(); I != E; ++I) {
      St.push(*I);
    }

    if (!hasValidProfile())
      continue;

    OS << (L->getLoopDepth() > 1 ? "Nested" : "Outer") << " loop header: "
       << L->getHeader()->getName();
    OS << "\n";
    OS << "Loop basic blocks: ";
    auto Sep = "";
    for (auto BI = L->block_begin(), BE = L->block_end(); BI != BE; ++BI) {
      OS << Sep << (*BI)->getName();
      Sep = ", ";
    }
    OS << "\n";
    if (hasValidProfile()) {
      OS << "Total back edge count: " << L->TotalBackEdgeCount << "\n";
      OS << "Loop entry count: " << L->EntryCount << "\n";
      OS << "Loop exit count: " << L->ExitCount << "\n";
      if (L->EntryCount > 0) {
        OS << "Average iters per entry: "
           << format("%.4lf", (double)L->TotalBackEdgeCount / L->EntryCount)
           << "\n";
      }
    }
    OS << "----\n";
  }

  OS << "Total number of loops: "<< BLI->TotalLoops << "\n";
  OS << "Number of outer loops: " << BLI->OuterLoops << "\n";
  OS << "Maximum nested loop depth: " << BLI->MaximumDepth << "\n\n";
}

DynoStats BinaryFunction::getDynoStats() const {
  DynoStats Stats;

  // Return empty-stats about the function we don't completely understand.
  if (!isSimple() || !hasValidProfile())
    return Stats;

  // If the function was folded in non-relocation mode we keep its profile
  // for optimization. However, it should be excluded from the dyno stats.
  if (isFolded())
    return Stats;

  // Update enumeration of basic blocks for correct detection of branch'
  // direction.
  updateLayoutIndices();

  for (const auto &BB : layout()) {
    // The basic block execution count equals to the sum of incoming branch
    // frequencies. This may deviate from the sum of outgoing branches of the
    // basic block especially since the block may contain a function that
    // does not return or a function that throws an exception.
    const uint64_t BBExecutionCount =  BB->getKnownExecutionCount();

    // Ignore empty blocks and blocks that were not executed.
    if (BB->getNumNonPseudos() == 0 || BBExecutionCount == 0)
      continue;

    // Count the number of calls by iterating through all instructions.
    for (const auto &Instr : *BB) {
      if (BC.MIA->isStore(Instr)) {
        Stats[DynoStats::STORES] += BBExecutionCount;
      }
      if (BC.MIA->isLoad(Instr)) {
        Stats[DynoStats::LOADS] += BBExecutionCount;
      }
      if (!BC.MIA->isCall(Instr))
        continue;
      uint64_t CallFreq = BBExecutionCount;
      if (BC.MIA->getConditionalTailCall(Instr)) {
        CallFreq = 0;
        if (auto FreqOrErr =
                BC.MIA->tryGetAnnotationAs<uint64_t>(Instr, "CTCTakenFreq")) {
          CallFreq = *FreqOrErr;
        }
      }
      Stats[DynoStats::FUNCTION_CALLS] += CallFreq;
      if (BC.MIA->getMemoryOperandNo(Instr) != -1) {
        Stats[DynoStats::INDIRECT_CALLS] += CallFreq;
      } else if (const auto *CallSymbol = BC.MIA->getTargetSymbol(Instr)) {
        const auto *BF = BC.getFunctionForSymbol(CallSymbol);
        if (BF && BF->isPLTFunction())
          Stats[DynoStats::PLT_CALLS] += CallFreq;

          // We don't process PLT functions and hence have to adjust
          // relevant dynostats here.
          Stats[DynoStats::LOADS] += CallFreq;
          Stats[DynoStats::INDIRECT_CALLS] += CallFreq;
      }
    }

    Stats[DynoStats::INSTRUCTIONS] += BB->getNumNonPseudos() * BBExecutionCount;

    // Jump tables.
    const auto *LastInstr = BB->getLastNonPseudoInstr();
    if (BC.MIA->getJumpTable(*LastInstr)) {
      Stats[DynoStats::JUMP_TABLE_BRANCHES] += BBExecutionCount;
      DEBUG(
        static uint64_t MostFrequentJT;
        if (BBExecutionCount > MostFrequentJT) {
          MostFrequentJT = BBExecutionCount;
          dbgs() << "BOLT-INFO: most frequently executed jump table is in "
                 << "function " << *this << " in basic block " << BB->getName()
                 << " executed totally " << BBExecutionCount << " times.\n";
        }
      );
      continue;
    }

    // Update stats for branches.
    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    if (!BB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch)) {
      continue;
    }

    if (!CondBranch && !UncondBranch) {
      continue;
    }

    // Simple unconditional branch.
    if (!CondBranch) {
      Stats[DynoStats::UNCOND_BRANCHES] += BBExecutionCount;
      continue;
    }

    // CTCs
    if (BC.MIA->getConditionalTailCall(*CondBranch)) {
      if (BB->branch_info_begin() != BB->branch_info_end())
        Stats[DynoStats::UNCOND_BRANCHES] += BB->branch_info_begin()->Count;
      continue;
    }

    // Conditional branch that could be followed by an unconditional branch.
    uint64_t TakenCount = BB->getBranchInfo(true).Count;
    if (TakenCount == COUNT_NO_PROFILE)
      TakenCount = 0;

    uint64_t NonTakenCount = BB->getBranchInfo(false).Count;
    if (NonTakenCount == COUNT_NO_PROFILE)
      NonTakenCount = 0;

    if (isForwardBranch(BB, BB->getConditionalSuccessor(true))) {
      Stats[DynoStats::FORWARD_COND_BRANCHES] += BBExecutionCount;
      Stats[DynoStats::FORWARD_COND_BRANCHES_TAKEN] += TakenCount;
    } else {
      Stats[DynoStats::BACKWARD_COND_BRANCHES] += BBExecutionCount;
      Stats[DynoStats::BACKWARD_COND_BRANCHES_TAKEN] += TakenCount;
    }

    if (UncondBranch) {
      Stats[DynoStats::UNCOND_BRANCHES] += NonTakenCount;
    }
  }

  return Stats;
}

Optional<SmallVector<std::pair<uint64_t, uint64_t>, 16>>
BinaryFunction::getFallthroughsInTrace(uint64_t From, uint64_t To) const {
  SmallVector<std::pair<uint64_t, uint64_t>, 16> Res;

  if (CurrentState != State::Disassembled)
    return NoneType();

  // Get iterators and validate trace start/end
  auto FromIter = Instructions.find(From);
  if (FromIter == Instructions.end())
    return NoneType();

  auto ToIter = Instructions.find(To);
  if (ToIter == Instructions.end())
    return NoneType();

  // Trace needs to go forward
  if (FromIter->first > ToIter->first)
    return NoneType();

  // Trace needs to finish in a branch
  if (!BC.MIA->isBranch(ToIter->second) && !BC.MIA->isCall(ToIter->second) &&
      !BC.MIA->isReturn(ToIter->second))
    return NoneType();

  // Analyze intermediate instructions
  for (; FromIter != ToIter; ++FromIter) {
    // This operates under an assumption that we collect all branches in LBR
    // No unconditional branches in the middle of the trace
    if (BC.MIA->isUnconditionalBranch(FromIter->second) ||
        BC.MIA->isReturn(FromIter->second) ||
        BC.MIA->isCall(FromIter->second))
      return NoneType();

    if (!BC.MIA->isConditionalBranch(FromIter->second))
      continue;

    const uint64_t Src = FromIter->first;
    auto Next = std::next(FromIter);
    const uint64_t Dst = Next->first;
    Res.push_back(std::make_pair(Src, Dst));
  }

  return Res;
}

void DynoStats::print(raw_ostream &OS, const DynoStats *Other) const {
  auto printStatWithDelta = [&](const std::string &Name, uint64_t Stat,
                                uint64_t OtherStat) {
    OS << format("%'20lld : ", Stat * opts::DynoStatsScale) << Name;
    if (Other) {
      if (Stat != OtherStat) {
       OtherStat = std::max(OtherStat, uint64_t(1)); // to prevent divide by 0
       OS << format(" (%+.1f%%)",
                    ( (float) Stat - (float) OtherStat ) * 100.0 /
                      (float) (OtherStat) );
      } else {
        OS << " (=)";
      }
    }
    OS << '\n';
  };

  for (auto Stat = DynoStats::FIRST_DYNO_STAT + 1;
       Stat < DynoStats::LAST_DYNO_STAT;
       ++Stat) {
    printStatWithDelta(Desc[Stat], Stats[Stat], Other ? (*Other)[Stat] : 0);
  }
}

void DynoStats::operator+=(const DynoStats &Other) {
  for (auto Stat = DynoStats::FIRST_DYNO_STAT + 1;
       Stat < DynoStats::LAST_DYNO_STAT;
       ++Stat) {
    Stats[Stat] += Other[Stat];
  }
}

} // namespace bolt
} // namespace llvm
