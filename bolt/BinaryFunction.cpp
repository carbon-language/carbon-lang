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
#include "ReorderAlgorithm.h"
#include "DataReader.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
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

namespace opts {

extern cl::opt<unsigned> Verbosity;
extern cl::opt<bool> PrintDynoStats;

static cl::opt<bool>
JumpTables("jump-tables",
           cl::desc("enable jump table support (experimental)"),
           cl::ZeroOrMore);

static cl::opt<bool>
AgressiveSplitting("split-all-cold",
                   cl::desc("outline as many cold basic blocks as possible"),
                   cl::ZeroOrMore);

static cl::opt<bool>
DotToolTipCode("dot-tooltip-code",
               cl::desc("add basic block instructions as tool tips on nodes"),
               cl::ZeroOrMore,
               cl::Hidden);

static cl::opt<uint32_t>
DynoStatsScale("dyno-stats-scale",
               cl::desc("scale to be applied while reporting dyno stats"),
               cl::Optional,
               cl::init(1));

} // namespace opts

namespace llvm {
namespace bolt {

// Temporary constant.
//
// TODO: move to architecture-specific file together with the code that is
// using it.
constexpr unsigned NoRegister = 0;

constexpr const char *DynoStats::Desc[];

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

uint64_t BinaryFunction::Count = 0;

BinaryBasicBlock *
BinaryFunction::getBasicBlockContainingOffset(uint64_t Offset) {
  if (Offset > Size)
    return nullptr;

  if (BasicBlocks.empty())
    return nullptr;

  // This is commented out because it makes BOLT too slow.
  // assert(std::is_sorted(begin(), end()));
  auto I = std::upper_bound(begin(),
                            end(),
                            BinaryBasicBlock(Offset));
  assert(I != begin() && "first basic block not at offset 0");

  return &*--I;
}

size_t
BinaryFunction::getBasicBlockOriginalSize(const BinaryBasicBlock *BB) const {
  auto Index = getIndex(BB);
  if (Index + 1 == BasicBlocks.size()) {
    return Size - BB->getOffset();
  } else {
    return BasicBlocks[Index + 1]->getOffset() - BB->getOffset();
  }
}

unsigned BinaryFunction::eraseDeadBBs(
    std::map<BinaryBasicBlock *, bool> &ToPreserve) {
  BasicBlockOrderType NewLayout;
  unsigned Count = 0;
  assert(ToPreserve[BasicBlocksLayout.front()] == true &&
         "unable to remove an entry basic block");
  for (auto I = BasicBlocksLayout.begin(), E = BasicBlocksLayout.end(); I != E;
       ++I) {
    if (ToPreserve[*I])
      NewLayout.push_back(*I);
    else
      ++Count;
  }
  BasicBlocksLayout = std::move(NewLayout);
  return Count;
}

void BinaryFunction::dump(std::string Annotation,
                          bool PrintInstructions) const {
  print(dbgs(), Annotation, PrintInstructions);
}

void BinaryFunction::print(raw_ostream &OS, std::string Annotation,
                           bool PrintInstructions) const {
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
  if (IdenticalFunctionAddress != Address)
    OS << "\n  Id Fun Addr : 0x" << Twine::utohexstr(IdenticalFunctionAddress);

  if (opts::PrintDynoStats && !BasicBlocksLayout.empty()) {
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
       << BB->size() << " instructions, align : "
       << BB->getAlignment() << ")\n";

    if (BB->isLandingPad()) {
      OS << "  Landing Pad\n";
    }

    uint64_t BBExecCount = BB->getExecutionCount();
    if (BBExecCount != BinaryBasicBlock::COUNT_NO_PROFILE) {
      OS << "  Exec Count : " << BBExecCount << "\n";
    }
    if (!BBCFIState.empty()) {
      OS << "  CFI State : " << BBCFIState[getIndex(BB)] << '\n';
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
            BI->MispredictedCount != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
          OS << " (mispreds: " << BI->MispredictedCount
             << ", count: " << BI->Count << ")";
        } else if (ExecutionCount != COUNT_NO_PROFILE &&
                   BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
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

  for(unsigned Index = 0; Index < JumpTables.size(); ++Index) {
    const auto &JumpTable = JumpTables[Index];
    OS << "Jump Table #" << (Index + 1) << '\n';
    for (unsigned EIndex = 0; EIndex < JumpTable.Entries.size(); ++EIndex) {
      const auto *Entry = JumpTable.Entries[EIndex];
      OS << "  entry " << EIndex << ": " << Entry->getName() << '\n';
    }
    OS << '\n';
  }

  OS << "DWARF CFI Instructions:\n";
  if (OffsetToCFI.size()) {
    // Pre-buildCFG information
    for (auto &Elmt : OffsetToCFI) {
      OS << format("    %08x:\t", Elmt.first);
      assert(Elmt.second < FrameInstructions.size() && "Incorrect CFI offset");
      BinaryContext::printCFI(OS, FrameInstructions[Elmt.second].getOperation());
      OS << "\n";
    }
  } else {
    // Post-buildCFG information
    for (uint32_t I = 0, E = FrameInstructions.size(); I != E; ++I) {
      const MCCFIInstruction &CFI = FrameInstructions[I];
      OS << format("    %d:\t", I);
      BinaryContext::printCFI(OS, CFI.getOperation());
      OS << "\n";
    }
  }
  if (FrameInstructions.empty())
    OS << "    <empty>\n";

  OS << "End of Function \"" << *this << "\"\n\n";
}

bool BinaryFunction::disassemble(ArrayRef<uint8_t> FunctionData) {
  assert(FunctionData.size() == getSize() &&
         "function size does not match raw data size");

  auto &Ctx = BC.Ctx;
  auto &MIA = BC.MIA;

  DWARFUnitLineTable ULT = getDWARFUnitLineTable();

  // Insert a label at the beginning of the function. This will be our first
  // basic block.
  Labels[0] = Ctx->createTempSymbol("BB0", false);

  auto getOrCreateLocalLabel = [&](uint64_t Address) {
    MCSymbol *Result;
    // Check if there's already a registered label.
    auto Offset = Address - getAddress();
    assert(Offset < getSize() && "address outside of function bounds");
    auto LI = Labels.find(Offset);
    if (LI == Labels.end()) {
      Result = Ctx->createTempSymbol();
      Labels[Offset] = Result;
    } else {
      Result = LI->second;
    }
    return Result;
  };

  auto handleRIPOperand =
      [&](MCInst &Instruction, uint64_t Address, uint64_t Size) {
    uint64_t TargetAddress{0};
    MCSymbol *TargetSymbol{nullptr};
    if (!BC.MIA->evaluateMemOperandTarget(Instruction, TargetAddress, Address,
                                          Size)) {
      DEBUG(dbgs() << "BOLT: rip-relative operand can't be evaluated:\n";
            BC.InstPrinter->printInst(&Instruction, dbgs(), "", *BC.STI);
            dbgs() << '\n';
            Instruction.dump_pretty(dbgs(), BC.InstPrinter.get());
            dbgs() << '\n';);
      return false;
    }
    if (TargetAddress == 0) {
      if (opts::Verbosity >= 1) {
        errs() << "BOLT-WARNING: rip-relative operand is zero in function "
               << *this << ". Ignoring function.\n";
      }
      return false;
    }

    // Note that the address does not necessarily have to reside inside
    // a section, it could be an absolute address too.
    auto Section = BC.getSectionForAddress(TargetAddress);
    if (Section && Section->isText()) {
      if (containsAddress(TargetAddress)) {
        TargetSymbol = getOrCreateLocalLabel(TargetAddress);
      } else {
        BC.InterproceduralReferences.insert(TargetAddress);
      }
    }
    if (!TargetSymbol)
      TargetSymbol = BC.getOrCreateGlobalSymbol(TargetAddress, "DATAat");
    BC.MIA->replaceMemOperandDisp(
        Instruction, MCOperand::createExpr(MCSymbolRefExpr::create(
                         TargetSymbol, MCSymbolRefExpr::VK_None, *BC.Ctx)));
    return true;
  };

  enum class IndirectBranchType : char {
    UNKNOWN = 0,            /// Unable to determine type.
    POSSIBLE_TAIL_CALL,     /// Possibly a tail call.
    POSSIBLE_JUMP_TABLE,    /// Possibly a switch/jump table
    POSSIBLE_GOTO           /// Possibly a gcc's computed goto.
  };

  auto analyzeIndirectBranch =
      [&](MCInst &Instruction, unsigned Size, uint64_t Offset) {
    // Try to find a (base) memory location from where the address for
    // the indirect branch is loaded. For X86-64 the memory will be specified
    // in the following format:
    //
    //   {%rip}/{%basereg} + Imm + IndexReg * Scale
    //
    // We are interested in the cases where Scale == sizeof(uintptr_t) and
    // the contents of the memory are presumably a function array.
    auto *MemLocInstr = &Instruction;
    if (Instruction.getNumOperands() == 1) {
      // If the indirect jump is on register - try to detect if the
      // register value is loaded from a memory location.
      assert(Instruction.getOperand(0).isReg() && "register operand expected");
      const auto JmpRegNum = Instruction.getOperand(0).getReg();
      // Check if one of the previous instructions defines the jump-on register.
      // We will check that this instruction belongs to the same basic block
      // in postProcessIndirectBranches().
      for (auto PrevII = Instructions.rbegin(); PrevII != Instructions.rend();
           ++PrevII) {
        auto &PrevInstr = PrevII->second;
        const auto &PrevInstrDesc = BC.MII->get(PrevInstr.getOpcode());
        if (!PrevInstrDesc.hasDefOfPhysReg(PrevInstr, JmpRegNum, *BC.MRI))
          continue;
        if (!MIA->isMoveMem2Reg(PrevInstr))
          return IndirectBranchType::UNKNOWN;
        MemLocInstr = &PrevInstr;
        break;
      }
      if (MemLocInstr == &Instruction) {
        // No definition seen for the register in this function so far. Could be
        // an input parameter - which means it is an external code reference.
        // It also could be that the definition happens to be in the code that
        // we haven't processed yet. Since we have to be conservative, return
        // as UNKNOWN case.
        return IndirectBranchType::UNKNOWN;
      }
    }

    const auto RIPRegister = BC.MRI->getProgramCounter();
    auto PtrSize = BC.AsmInfo->getPointerSize();

    // Analyze contents of the memory if possible.
    unsigned  BaseRegNum;
    int64_t   ScaleValue;
    unsigned  IndexRegNum;
    int64_t   DispValue;
    unsigned  SegRegNum;
    if (!MIA->evaluateX86MemoryOperand(*MemLocInstr, BaseRegNum,
                                       ScaleValue, IndexRegNum,
                                       DispValue, SegRegNum))
      return IndirectBranchType::UNKNOWN;

    if ((BaseRegNum != bolt::NoRegister && BaseRegNum != RIPRegister) ||
        SegRegNum != bolt::NoRegister || ScaleValue != PtrSize)
      return IndirectBranchType::UNKNOWN;

    auto ArrayStart = DispValue;
    if (BaseRegNum == RIPRegister)
      ArrayStart += getAddress() + Offset + Size;

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
    DataExtractor DE(SectionContents, BC.AsmInfo->isLittleEndian(), PtrSize);
    auto ValueOffset = static_cast<uint32_t>(ArrayStart - Section.getAddress());
    uint64_t Value = 0;
    auto Result = IndirectBranchType::UNKNOWN;
    std::vector<MCSymbol *> JTLabelCandidates;
    while (ValueOffset <= Section.getSize() - PtrSize) {
      DEBUG(dbgs() << "BOLT-DEBUG: indirect jmp at 0x"
                   << Twine::utohexstr(getAddress() + Offset)
                   << " is referencing address 0x"
                   << Twine::utohexstr(Section.getAddress() + ValueOffset));
      // Extract the value and increment the offset.
      Value = DE.getAddress(&ValueOffset);
      DEBUG(dbgs() << ", which contains value "
                   << Twine::utohexstr(Value) << '\n');
      if (containsAddress(Value) && Value != getAddress()) {
        // Is it possible to have a jump table with function start as an entry?
        auto *JTEntry = getOrCreateLocalLabel(Value);
        JTLabelCandidates.push_back(JTEntry);
        TakenBranches.emplace_back(Offset, Value - getAddress());
        Result = IndirectBranchType::POSSIBLE_JUMP_TABLE;
        continue;
      }
      // Potentially a switch table can contain  __builtin_unreachable() entry
      // pointing just right after the function. In this case we have to check
      // another entry. Otherwise the entry is outside of this function scope
      // and it's not a switch table.
      if (Value != getAddress() + getSize()) {
        break;
      }
      JTLabelCandidates.push_back(getFunctionEndLabel());
    }
    if (Result == IndirectBranchType::POSSIBLE_JUMP_TABLE) {
      assert(JTLabelCandidates.size() > 2 &&
             "expected more than 2 jump table entries");
      auto *JTStartLabel = BC.Ctx->createTempSymbol("JUMP_TABLE", true);
      JumpTables.emplace_back(JumpTable{JTStartLabel,
                              std::move(JTLabelCandidates)});
      BC.MIA->replaceMemOperandDisp(*MemLocInstr, JTStartLabel, BC.Ctx.get());
      BC.MIA->setJumpTableIndex(Instruction, JumpTables.size());
      DEBUG(dbgs() << "BOLT-DEBUG: creating jump table "
                   << JTStartLabel->getName()
                   << " in function " << *this << " with "
                   << JTLabelCandidates.size() << " entries.\n");
      return Result;
    }
    BC.InterproceduralReferences.insert(Value);
    return IndirectBranchType::POSSIBLE_TAIL_CALL;
  };

  for (uint64_t Offset = 0; Offset < getSize(); ) {
    MCInst Instruction;
    uint64_t Size;
    uint64_t AbsoluteInstrAddr = getAddress() + Offset;

    if (!BC.DisAsm->getInstruction(Instruction,
                                   Size,
                                   FunctionData.slice(Offset),
                                   AbsoluteInstrAddr,
                                   nulls(),
                                   nulls())) {
      // Ignore this function. Skip to the next one.
      if (opts::Verbosity >= 1) {
        errs() << "BOLT-WARNING: unable to disassemble instruction at offset 0x"
               << Twine::utohexstr(Offset) << " (address 0x"
               << Twine::utohexstr(AbsoluteInstrAddr) << ") in function "
               << *this << '\n';
      }
      IsSimple = false;
      break;
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
        bool IsCondBranch = MIA->isConditionalBranch(Instruction);
        MCSymbol *TargetSymbol{nullptr};

        if (IsCall && containsAddress(TargetAddress)) {
          if (TargetAddress == getAddress()) {
            // Recursive call.
            TargetSymbol = getSymbol();
          } else {
            // Possibly an old-style PIC code
            if (opts::Verbosity >= 1) {
              errs() << "BOLT-WARNING: internal call detected at 0x"
                     << Twine::utohexstr(AbsoluteInstrAddr)
                     << " in function " << *this << ". Skipping.\n";
            }
            IsSimple = false;
          }
        }

        if (!TargetSymbol) {
          // Create either local label or external symbol.
          if (containsAddress(TargetAddress)) {
            TargetSymbol = getOrCreateLocalLabel(TargetAddress);
          } else {
            BC.InterproceduralReferences.insert(TargetAddress);
            if (opts::Verbosity >= 2 && !IsCall && Size == 2) {
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
              if (!MIA->convertJmpToTailCall(Instruction) &&
                  opts::Verbosity >= 2) {
                assert(IsCondBranch && "unknown tail call instruction");
                errs() << "BOLT-WARNING: conditional tail call detected in "
                       << "function " << *this << " at 0x"
                       << Twine::utohexstr(AbsoluteInstrAddr) << ".\n";
              }
              // TODO: A better way to do this would be using annotations for
              // MCInst objects.
              TailCallOffsets.emplace(std::make_pair(Offset,
                                                     TargetAddress));
              IsCall = true;
            }

            TargetSymbol = BC.getOrCreateGlobalSymbol(TargetAddress,
                                                      "FUNCat");
            if (TargetAddress == 0) {
              // We actually see calls to address 0 because of the weak symbols
              // from the libraries. In reality more often than not it is
              // unreachable code, but we don't know it and have to emit calls
              // to 0 which make LLVM JIT unhappy.
              if (opts::Verbosity >= 1) {
                errs() << "BOLT-WARNING: Function " << *this
                       << " has a call to address zero. Ignoring function.\n";
              }
              IsSimple = false;
            }
          }
        }

        Instruction.clear();
        Instruction.addOperand(
            MCOperand::createExpr(
              MCSymbolRefExpr::create(TargetSymbol,
                                      MCSymbolRefExpr::VK_None,
                                      *Ctx)));
        if (!IsCall) {
          // Add taken branch info.
          TakenBranches.emplace_back(Offset, TargetAddress - getAddress());
        }
        if (IsCondBranch) {
          // Add fallthrough branch info.
          FTBranches.emplace_back(Offset, Offset + Size);
        }
      } else {
        // Could not evaluate branch. Should be an indirect call or an
        // indirect branch. Bail out on the latter case.
        if (MIA->isIndirectBranch(Instruction)) {
          auto Result = analyzeIndirectBranch(Instruction, Size, Offset);
          switch (Result) {
          default:
            llvm_unreachable("unexpected result");
          case IndirectBranchType::POSSIBLE_TAIL_CALL:
            MIA->convertJmpToTailCall(Instruction);
            break;
          case IndirectBranchType::POSSIBLE_JUMP_TABLE:
            if (!opts::JumpTables)
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
            if (opts::Verbosity >= 1) {
              errs() << "BOLT-WARNING: cannot handle RIP operand at 0x"
                     << Twine::utohexstr(AbsoluteInstrAddr)
                     << ". Skipping function " << *this << ".\n";
            }
            IsSimple = false;
          }
        }
      }
    } else {
      if (MIA->hasRIPOperand(Instruction)) {
        if (!handleRIPOperand(Instruction, AbsoluteInstrAddr, Size)) {
          if (opts::Verbosity >= 1) {
            errs() << "BOLT-WARNING: cannot handle RIP operand at 0x"
                   << Twine::utohexstr(AbsoluteInstrAddr)
                   << ". Skipping function " << *this << ".\n";
          }
          IsSimple = false;
        }
      }
    }

    if (ULT.first && ULT.second) {
      Instruction.setLoc(
          findDebugLineInformationForInstructionAt(AbsoluteInstrAddr, ULT));
    }

    addInstruction(Offset, std::move(Instruction));

    Offset += Size;
  }

  // TODO: clear memory if not simple function?

  // Update state.
  updateState(State::Disassembled);

  return true;
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

      // Validate the tail call assumptions.
      if (BC.MIA->isTailCall(Instr) || (BC.MIA->getJumpTableIndex(Instr) > 0)) {
        if (BC.MIA->getMemoryOperandNo(Instr) != -1) {
          // We have validated memory contents addressed by the jump
          // instruction already.
          continue;
        }
        // This is jump on register. Just make sure the register is defined
        // in the containing basic block. Other assumptions were checked
        // earlier.
        assert(Instr.getOperand(0).isReg() && "register operand expected");
        const auto JmpRegNum = Instr.getOperand(0).getReg();
        bool IsJmpRegSetInBB = false;
        for (const auto &OtherInstr : *BB) {
          const auto &OtherInstrDesc = BC.MII->get(OtherInstr.getOpcode());
          if (OtherInstrDesc.hasDefOfPhysReg(OtherInstr, JmpRegNum, *BC.MRI)) {
            IsJmpRegSetInBB = true;
            break;
          }
        }
        if (IsJmpRegSetInBB)
          continue;
        if (opts::Verbosity >= 2) {
          outs() << "BOLT-INFO: rejected potential "
                     << (BC.MIA->isTailCall(Instr) ? "indirect tail call"
                                                   : "jump table")
                     << " in function " << *this
                     << " because the jump-on register was not defined in "
                     << " basic block " << BB->getName() << ".\n";
          DEBUG(dbgs() << BC.printInstructions(dbgs(), BB->begin(), BB->end(),
                                               BB->getOffset(), this));
        }
        return false;
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
                                     BB->getOffset(), this));
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

  auto BranchDataOrErr = BC.DR.getFuncBranchData(getNames());
  if (!BranchDataOrErr) {
    DEBUG(dbgs() << "no branch data found for \"" << *this << "\"\n");
  } else {
    ExecutionCount = BranchDataOrErr->ExecutionCount;
  }

  if (!isSimple())
    return false;

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
  //    (missed optimization opportunity?).
  //
  // Created basic blocks are sorted in layout order since they are
  // created in the same order as instructions, and instructions are
  // sorted by offsets.
  BinaryBasicBlock *InsertBB{nullptr};
  BinaryBasicBlock *PrevBB{nullptr};
  bool IsLastInstrNop{false};
  bool IsPreviousInstrTailCall{false};
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
      // we see an unconditional branch following a conditional one.
      assert(PrevBB && "no previous basic block for a fall through");
      assert(PrevInstr && "no previous instruction for a fall through");
      if (MIA->isUnconditionalBranch(Instr) &&
          !MIA->isUnconditionalBranch(*PrevInstr) && !IsPreviousInstrTailCall) {
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

    // Record whether this basic block is terminated with a tail call.
    auto TCI = TailCallOffsets.find(Offset);
    if (TCI != TailCallOffsets.end()) {
      uint64_t TargetAddr = TCI->second;
      TailCallTerminatedBlocks.emplace(
          std::make_pair(InsertBB,
                         TailCallInfo(Offset, InsertIndex, TargetAddr)));
      IsPreviousInstrTailCall = true;
    } else {
      IsPreviousInstrTailCall = false;
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

  // Intermediate dump.
  DEBUG(print(dbgs(), "after creating basic blocks"));

  // TODO: handle properly calls to no-return functions,
  // e.g. exit(3), etc. Otherwise we'll see a false fall-through
  // blocks.

  // Make sure we can use profile data for this function.
  if (BranchDataOrErr)
    evaluateProfileData(BranchDataOrErr.get());

  for (auto &Branch : TakenBranches) {
    DEBUG(dbgs() << "registering branch [0x" << Twine::utohexstr(Branch.first)
                 << "] -> [0x" << Twine::utohexstr(Branch.second) << "]\n");
    BinaryBasicBlock *FromBB = getBasicBlockContainingOffset(Branch.first);
    assert(FromBB && "cannot find BB containing FROM branch");
    BinaryBasicBlock *ToBB = getBasicBlockAtOffset(Branch.second);
    assert(ToBB && "cannot find BB containing TO branch");

    if (BranchDataOrErr.getError()) {
      FromBB->addSuccessor(ToBB);
    } else {
      const FuncBranchData &BranchData = BranchDataOrErr.get();
      auto BranchInfoOrErr = BranchData.getBranch(Branch.first, Branch.second);
      if (BranchInfoOrErr.getError()) {
        FromBB->addSuccessor(ToBB);
      } else {
        const BranchInfo &BInfo = BranchInfoOrErr.get();
        FromBB->addSuccessor(ToBB, BInfo.Branches, BInfo.Mispreds);
      }
    }
  }

  for (auto &Branch : FTBranches) {
    DEBUG(dbgs() << "registering fallthrough [0x"
                 << Twine::utohexstr(Branch.first) << "] -> [0x"
                 << Twine::utohexstr(Branch.second) << "]\n");
    BinaryBasicBlock *FromBB = getBasicBlockContainingOffset(Branch.first);
    assert(FromBB && "cannot find BB containing FROM branch");
    // Try to find the destination basic block. If the jump instruction was
    // followed by a no-op then the destination offset recorded in FTBranches
    // will point to that no-op but the destination basic block will start
    // after the no-op due to ingoring no-ops when creating basic blocks.
    // So we have to skip any no-ops when trying to find the destination
    // basic block.
    BinaryBasicBlock *ToBB = getBasicBlockAtOffset(Branch.second);
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
        continue;
      }
    }

    // Does not add a successor if we can't find profile data, leave it to the
    // inference pass to guess its frequency
    if (BranchDataOrErr) {
      const FuncBranchData &BranchData = BranchDataOrErr.get();
      auto BranchInfoOrErr = BranchData.getBranch(Branch.first, Branch.second);
      if (BranchInfoOrErr) {
        const BranchInfo &BInfo = BranchInfoOrErr.get();
        FromBB->addSuccessor(ToBB, BInfo.Branches, BInfo.Mispreds);
      }
    }
  }

  for (auto &I : TailCallTerminatedBlocks) {
    TailCallInfo &TCInfo = I.second;
    if (BranchDataOrErr) {
      const FuncBranchData &BranchData = BranchDataOrErr.get();
      auto BranchInfoOrErr = BranchData.getDirectCallBranch(TCInfo.Offset);
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
      PrevBB->addSuccessor(BB, BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE,
                           BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE);
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
    bool IsCondTailCall = MIA->isConditionalBranch(*LastInstIter) &&
                          TailCallTerminatedBlocks.count(BB);

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
  if (hasValidProfile())
    inferFallThroughCounts();

  // Update CFI information for each BB
  annotateCFIState();

  // Convert conditional tail call branches to conditional branches that jump
  // to a tail call.
  removeConditionalTailCalls();

  // Set the basic block layout to the original order.
  for (auto BB : BasicBlocks) {
    BasicBlocksLayout.emplace_back(BB);
  }

  // Make any necessary adjustments for indirect branches.
  if (!postProcessIndirectBranches())
    setSimple(false);

  // Fix the possibly corrupted CFI state. CFI state may have been corrupted
  // because of the CFG modifications while removing conditional tail calls.
  fixCFIState();

  // Clean-up memory taken by instructions and labels.
  clearList(Instructions);
  clearList(TailCallOffsets);
  clearList(TailCallTerminatedBlocks);
  clearList(OffsetToCFI);
  clearList(Labels);
  clearList(TakenBranches);
  clearList(FTBranches);
  clearList(LPToBBIndex);

  // Update the state.
  CurrentState = State::CFG;

  // Annotate invoke instructions with GNU_args_size data.
  propagateGnuArgsSizeInfo();

  return true;
}

void BinaryFunction::evaluateProfileData(const FuncBranchData &BranchData) {
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

  // Until we define a minimal profile, we consider no branch data to be a valid
  // profile. It could happen to a function without branches.
  if (LocalProfileBranches.empty()) {
    ProfileMatchRatio = 1.0f;
    return;
  }

  std::sort(LocalProfileBranches.begin(), LocalProfileBranches.end());

  BranchListType FunctionBranches = TakenBranches;
  FunctionBranches.insert(FunctionBranches.end(),
                          FTBranches.begin(),
                          FTBranches.end());
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
    if (BC.MIA->isCall(SrcInstrI->second) && Branch.second == 0)
      return true;

    auto DstInstrI = Instructions.find(Branch.second);
    if (DstInstrI == Instructions.end())
      return false;

    // Check if it is a return from a recursive call.
    bool IsSrcReturn = BC.MIA->isReturn(SrcInstrI->second);
    // "rep ret" is considered to be 2 different instructions.
    if (!IsSrcReturn && BC.MIA->isPrefix(SrcInstrI->second)) {
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

  ProfileMatchRatio =
    (float) (LocalProfileBranches.size() - OrphanBranches.size()) /
    (float) LocalProfileBranches.size();

  if (opts::Verbosity >= 2 && !OrphanBranches.empty()) {
    errs() << "BOLT-WARNING: profile branches match only "
           << format("%.1f%%", ProfileMatchRatio * 100.0f) << " ("
           << (LocalProfileBranches.size() - OrphanBranches.size()) << '/'
           << LocalProfileBranches.size() << ") for function "
           << *this << '\n';
    DEBUG(
      for (auto &OBranch : OrphanBranches)
        errs() << "\t0x" << Twine::utohexstr(OBranch.first) << " -> 0x"
               << Twine::utohexstr(OBranch.second) << " (0x"
               << Twine::utohexstr(OBranch.first + getAddress()) << " -> 0x"
    );
  }
}

void BinaryFunction::inferFallThroughCounts() {
  assert(!BasicBlocks.empty() && "basic block list should not be empty");

  auto BranchDataOrErr = BC.DR.getFuncBranchData(getNames());

  // Compute preliminary execution time for each basic block
  for (auto CurBB : BasicBlocks) {
    if (CurBB == *BasicBlocks.begin()) {
      CurBB->setExecutionCount(ExecutionCount);
      continue;
    }
    CurBB->ExecutionCount = 0;
  }

  for (auto CurBB : BasicBlocks) {
    auto SuccCount = CurBB->branch_info_begin();
    for (auto Succ : CurBB->successors()) {
      // Do not update execution count of the entry block (when we have tail
      // calls). We already accounted for those when computing the func count.
      if (Succ == *BasicBlocks.begin()) {
        ++SuccCount;
        continue;
      }
      if (SuccCount->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        Succ->setExecutionCount(Succ->getExecutionCount() + SuccCount->Count);
      ++SuccCount;
    }
  }

  // Update execution counts of landing pad blocks.
  if (!BranchDataOrErr.getError()) {
    const FuncBranchData &BranchData = BranchDataOrErr.get();
    for (const auto &I : BranchData.EntryData) {
      BinaryBasicBlock *BB = getBasicBlockAtOffset(I.To.Offset);
      if (BB && LandingPads.find(BB->getLabel()) != LandingPads.end()) {
        BB->setExecutionCount(BB->getExecutionCount() + I.Branches);
      }
    }
  }

  // Work on a basic block at a time, propagating frequency information forwards
  // It is important to walk in the layout order
  for (auto CurBB : BasicBlocks) {
    uint64_t BBExecCount = CurBB->getExecutionCount();

    // Propagate this information to successors, filling in fall-through edges
    // with frequency information
    if (CurBB->succ_size() == 0)
      continue;

    // Calculate frequency of outgoing branches from this node according to
    // LBR data
    uint64_t ReportedBranches = 0;
    for (const auto &SuccCount : CurBB->branch_info()) {
      if (SuccCount.Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        ReportedBranches += SuccCount.Count;
    }

    // Calculate frequency of outgoing tail calls from this node according to
    // LBR data
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
    // branch
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
      if (SuccCount.Count == BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
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
      BinaryBasicBlock *NextBB = getBasicBlockAtIndex(InsertIdx);

      BC.MIA->reverseBranchCondition(
          CondTailCallInst, NextBB->getLabel(), BC.Ctx.get());

      // Create a basic block containing the unconditional tail call instruction
      // and place it between BB and NextBB.
      std::vector<std::unique_ptr<BinaryBasicBlock>> TailCallBBs;
      TailCallBBs.emplace_back(createBasicBlock(NextBB->getOffset(), TCLabel));
      TailCallBBs[0]->addInstruction(TailCallInst);
      insertBasicBlocks(BB, std::move(TailCallBBs), /* UpdateCFIState */ false);
      TailCallBB = getBasicBlockAtIndex(InsertIdx);

      // Add the correct CFI state for the new block.
      BBCFIState.insert(BBCFIState.begin() + InsertIdx, TCInfo.CFIStateBefore);
    } else {
      // Forward jump: we will create a new basic block at the end of the
      // function containing the unconditional tail call and change the target
      // of the conditional tail call to this basic block.

      // Create a basic block containing the unconditional tail call
      // instruction and place it at the end of the function.
      // We have to add 1 byte as there's potentially an existing branch past
      // the end of the code as a result of __builtin_unreachable().
      const BinaryBasicBlock *LastBB = BasicBlocks.back();
      uint64_t NewBlockOffset = LastBB->getOffset() +
                         BC.computeCodeSize(LastBB->begin(), LastBB->end()) + 1;
      TailCallBB = addBasicBlock(NewBlockOffset, TCLabel);
      TailCallBB->addInstruction(TailCallInst);

      // Add the correct CFI state for the new block. It has to be inserted in
      // the one before last position (the last position holds the CFI state
      // after the last block).
      BBCFIState.insert(BBCFIState.begin() + BBCFIState.size() - 1,
                        TCInfo.CFIStateBefore);

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
  assert(!BasicBlocks.empty() && "basic block list should not be empty");

  uint32_t State = 0;
  uint32_t HighestState = 0;
  std::stack<uint32_t> StateStack;

  for (auto CI = BasicBlocks.begin(), CE = BasicBlocks.end(); CI != CE; ++CI) {
    BinaryBasicBlock *CurBB = *CI;
    // Annotate this BB entry
    BBCFIState.emplace_back(State);

    // While building the CFG, we want to save the CFI state before a tail call
    // instruction, so that we can correctly remove condtional tail calls
    auto TCI = TailCallTerminatedBlocks.find(CurBB);
    bool SaveState = TCI != TailCallTerminatedBlocks.end();

    // Advance state
    uint32_t Idx = 0;
    for (const auto &Instr : *CurBB) {
      auto *CFI = getCFIFor(Instr);
      if (CFI == nullptr) {
        if (SaveState && Idx == TCI->second.Index)
          TCI->second.CFIStateBefore = State;
        ++Idx;
        continue;
      }
      ++HighestState;
      if (CFI->getOperation() == MCCFIInstruction::OpRememberState) {
        StateStack.push(State);
      } else if (CFI->getOperation() == MCCFIInstruction::OpRestoreState) {
        assert(!StateStack.empty() && "Corrupt CFI stack");
        State = StateStack.top();
        StateStack.pop();
      } else if (CFI->getOperation() != MCCFIInstruction::OpGnuArgsSize) {
        State = HighestState;
      }
      ++Idx;
    }
  }

  // Store the state after the last BB
  BBCFIState.emplace_back(State);

  assert(StateStack.empty() && "Corrupt CFI stack");
}

bool BinaryFunction::fixCFIState() {
  auto Sep = "";
  DEBUG(dbgs() << "Trying to fix CFI states for each BB after reordering.\n");
  DEBUG(dbgs() << "This is the list of CFI states for each BB of " << *this
               << ": ");

  auto replayCFIInstrs =
      [this](uint32_t FromState, uint32_t ToState, BinaryBasicBlock *InBB,
             BinaryBasicBlock::const_iterator InsertIt) -> bool {
        if (FromState == ToState)
          return true;
        assert(FromState < ToState);

        std::vector<uint32_t> NewCFIs;
        uint32_t NestedLevel = 0;
        for (uint32_t CurState = FromState; CurState < ToState; ++CurState) {
          assert(CurState < FrameInstructions.size());
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
          if (opts::Verbosity >= 1) {
            errs() << "BOLT-WARNING: CFI rewriter detected nested CFI state"
                   << " while replaying CFI instructions for BB "
                   << InBB->getName() << " in function " << *this << '\n';
          }
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

  uint32_t State = 0;
  auto *FDEStartBB = BasicBlocksLayout[0];
  for (uint32_t I = 0, E = BasicBlocksLayout.size(); I != E; ++I) {
    auto *BB = BasicBlocksLayout[I];
    uint32_t BBIndex = getIndex(BB);

    // Hot-cold border: check if this is the first BB to be allocated in a cold
    // region (a different FDE). If yes, we need to reset the CFI state and
    // the FDEStartBB that is used to insert remember_state CFIs (t12863876).
    if (I != 0 && BB->isCold() != BasicBlocksLayout[I - 1]->isCold()) {
      State = 0;
      FDEStartBB = BB;
    }

    // We need to recover the correct state if it doesn't match expected
    // state at BB entry point.
    if (BBCFIState[BBIndex] < State) {
      // In this case, State is currently higher than what this BB expect it
      // to be. To solve this, we need to insert a CFI instruction to remember
      // the old state at function entry, then another CFI instruction to
      // restore it at the entry of this BB and replay CFI instructions to
      // reach the desired state.
      uint32_t OldState = BBCFIState[BBIndex];
      // Remember state at function entry point (our reference state).
      BinaryBasicBlock::const_iterator InsertIt = FDEStartBB->begin();
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
        if (opts::Verbosity >= 1) {
          errs() << " BOLT-WARNING: not possible to remember/recover state"
                 << " without corrupting CFI state stack in function "
                 << *this << "\n";
        }
        return false;
      }
    } else if (BBCFIState[BBIndex] > State) {
      // If BBCFIState[BBIndex] > State, it means we are behind in the
      // state. Just emit all instructions to reach this state at the
      // beginning of this BB. If this sequence of instructions involve
      // remember state or restore state, bail out.
      if (!replayCFIInstrs(State, BBCFIState[BBIndex], BB, BB->begin()))
        return false;
    }

    State = BBCFIState[BBIndex + 1];
    DEBUG(dbgs() << Sep << State);
    DEBUG(Sep = ", ");
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
  BasicBlocksLayout.clear();
  BasicBlocksLayout.swap(NewLayout);

  if (Split)
    splitFunction();
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
    OS << format("\"%s\" [label=\"%s\\n(O:%lu,I:%u,L%u)\"]\n",
                 BB->getName().data(),
                 BB->getName().data(),
                 BB->getOffset(),
                 getIndex(BB),
                 Layout);
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

    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    const bool Success = BB->analyzeBranch(TBB,
                                           FBB,
                                           CondBranch,
                                           UncondBranch);

    auto BI = BB->branch_info_begin();
    for (auto *Succ : BB->successors()) {
      std::string Branch;
      if (Success) {
        if (CondBranch && Succ->getLabel() == TBB) {
          Branch = BC.InstPrinter->getOpcodeName(CondBranch->getOpcode());
        } else if(UncondBranch && Succ->getLabel() == TBB) {
          Branch = BC.InstPrinter->getOpcodeName(UncondBranch->getOpcode());
        } else {
          Branch = "FT";
        }
      }
      OS << format("\"%s\" -> \"%s\" [label=\"%s",
                   BB->getName().data(),
                   Succ->getName().data(),
                   Branch.c_str());

      if (BB->getExecutionCount() != COUNT_NO_PROFILE &&
          BI->MispredictedCount != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
        OS << "\\n(M:" << BI->MispredictedCount << ",C:" << BI->Count << ")";
      } else if (ExecutionCount != COUNT_NO_PROFILE &&
                 BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
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
      if (!NextBB || (NextBB != TSuccessor && NextBB != FSuccessor)) {
        BB->addBranchInstruction(FSuccessor);
      }
    }
    // Cases where the number of successors is 0 (block ends with a
    // terminator) or more than 2 (switch table) don't require branch
    // instruction adjustments.
  }
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
    if (hasEHRanges()) {
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

  if (opts::AgressiveSplitting) {
    // All blocks with 0 count that we can move go to the end of the function.
    std::stable_sort(BasicBlocksLayout.begin(), BasicBlocksLayout.end(),
        [&] (BinaryBasicBlock *A, BinaryBasicBlock *B) {
          return A->canOutline() < B->canOutline();
        });
  } else if (hasEHRanges()) {
    // Typically functions with exception handling have landing pads at the end.
    // We cannot move beginning of landing pads, but we can move 0-count blocks
    // comprising landing pads to the end and thus facilitating splitting.
    auto FirstLP = BasicBlocksLayout.begin();
    while ((*FirstLP)->isLandingPad())
      ++FirstLP;

    std::stable_sort(FirstLP, BasicBlocksLayout.end(),
        [&] (BinaryBasicBlock *A, BinaryBasicBlock *B) {
          return A->canOutline() < B->canOutline();
        });
  }

  // Separate hot from cold
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
        } else {
          ++II;
        }
        continue;
      }

      // Add the value of GNU_args_size as an extra operand if landing pad
      // is non-empty.
      BC.MIA->addGnuArgsSize(Instr, CurrentGnuArgsSize);
      ++II;
    }
  }
}

void BinaryFunction::mergeProfileDataInto(BinaryFunction &BF) const {
  if (!hasValidProfile() || !BF.hasValidProfile())
    return;

  // Update BF's execution count.
  uint64_t MyExecutionCount = getExecutionCount();
  if (MyExecutionCount != BinaryFunction::COUNT_NO_PROFILE) {
    uint64_t OldExecCount = BF.getExecutionCount();
    uint64_t NewExecCount =
      OldExecCount == BinaryFunction::COUNT_NO_PROFILE ?
        MyExecutionCount :
        MyExecutionCount + OldExecCount;
    BF.setExecutionCount(NewExecCount);
  }

  // Update BF's basic block and edge counts.
  auto BBMergeI = BF.begin();
  for (BinaryBasicBlock *BB : BasicBlocks) {
    BinaryBasicBlock *BBMerge = &*BBMergeI;
    assert(getIndex(BB) == BF.getIndex(BBMerge));

    // Update BF's basic block count.
    uint64_t MyBBExecutionCount = BB->getExecutionCount();
    if (MyBBExecutionCount != BinaryBasicBlock::COUNT_NO_PROFILE) {
      uint64_t OldExecCount = BBMerge->getExecutionCount();
      uint64_t NewExecCount =
        OldExecCount == BinaryBasicBlock::COUNT_NO_PROFILE ?
          MyBBExecutionCount :
          MyBBExecutionCount + OldExecCount;
      BBMerge->setExecutionCount(NewExecCount);
    }

    // Update BF's edge count for successors of this basic block.
    auto BBMergeSI = BBMerge->succ_begin();
    auto BII = BB->branch_info_begin();
    auto BIMergeI = BBMerge->branch_info_begin();
    for (BinaryBasicBlock *BBSucc : BB->successors()) {
      BinaryBasicBlock *BBMergeSucc = *BBMergeSI;
      assert(getIndex(BBSucc) == BF.getIndex(BBMergeSucc));

      if (BII->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
        uint64_t OldBranchCount = BIMergeI->Count;
        uint64_t NewBranchCount =
          OldBranchCount == BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE ?
            BII->Count :
            BII->Count + OldBranchCount;
        BIMergeI->Count = NewBranchCount;
      }

      if (BII->MispredictedCount != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
        uint64_t OldMispredictedCount = BIMergeI->MispredictedCount;
        uint64_t NewMispredictedCount =
          OldMispredictedCount == BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE ?
            BII->MispredictedCount :
            BII->MispredictedCount + OldMispredictedCount;
        BIMergeI->MispredictedCount = NewMispredictedCount;
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

std::pair<bool, unsigned> BinaryFunction::isCalleeEquivalentWith(
    const MCInst &Inst, const BinaryBasicBlock &BB, const MCInst &InstOther,
    const BinaryBasicBlock &BBOther, const BinaryFunction &BF) const {
  // The callee operand in a direct call is the first operand. This
  // operand should be a symbol corresponding to the callee function.
  constexpr unsigned CalleeOpIndex = 0;

  // Helper function.
  auto getGlobalAddress = [this] (const MCSymbol &Symbol) -> uint64_t {
    auto AI = BC.GlobalSymbols.find(Symbol.getName());
    assert(AI != BC.GlobalSymbols.end());
    return AI->second;
  };

  const MCOperand &CalleeOp = Inst.getOperand(CalleeOpIndex);
  const MCOperand &CalleeOpOther = InstOther.getOperand(CalleeOpIndex);
  if (!CalleeOp.isExpr() || !CalleeOpOther.isExpr()) {
    // At least one of these is actually an indirect call.
    return std::make_pair(false, 0);
  }

  const MCSymbol &CalleeSymbol = CalleeOp.getExpr()->getSymbol();
  uint64_t CalleeAddress = getGlobalAddress(CalleeSymbol);

  const MCSymbol &CalleeSymbolOther = CalleeOpOther.getExpr()->getSymbol();
  uint64_t CalleeAddressOther = getGlobalAddress(CalleeSymbolOther);

  bool BothRecursiveCalls =
    CalleeAddress == getAddress() &&
    CalleeAddressOther == BF.getAddress();

  bool SameCallee = CalleeAddress == CalleeAddressOther;

  return std::make_pair(BothRecursiveCalls || SameCallee, CalleeOpIndex);
}

std::pair<bool, unsigned> BinaryFunction::isTargetEquivalentWith(
    const MCInst &Inst, const BinaryBasicBlock &BB, const MCInst &InstOther,
    const BinaryBasicBlock &BBOther, const BinaryFunction &BF,
    bool AreInvokes) const {
  // The target operand in a (non-indirect) jump instruction is the
  // first operand.
  unsigned TargetOpIndex = 0;
  if (AreInvokes) {
    // The landing pad operand in an invoke is either the second or the
    // sixth operand, depending on the number of operands of the invoke.
    TargetOpIndex = 1;
    if (Inst.getNumOperands() == 7 || Inst.getNumOperands() == 8)
      TargetOpIndex = 5;
  }

  const MCOperand &TargetOp = Inst.getOperand(TargetOpIndex);
  const MCOperand &TargetOpOther = InstOther.getOperand(TargetOpIndex);
  if (!TargetOp.isExpr() || !TargetOpOther.isExpr()) {
    assert(AreInvokes);
    // An invoke without a landing pad operand has no catch handler. As long
    // as both invokes have no catch target, we can consider they have the
    // same catch target.
    return std::make_pair(!TargetOp.isExpr() && !TargetOpOther.isExpr(),
                          TargetOpIndex);
  }

  const MCSymbol &TargetSymbol = TargetOp.getExpr()->getSymbol();
  BinaryBasicBlock *TargetBB =
    AreInvokes ?
      BB.getLandingPad(&TargetSymbol) :
      BB.getSuccessor(&TargetSymbol);

  const MCSymbol &TargetSymbolOther = TargetOpOther.getExpr()->getSymbol();
  BinaryBasicBlock *TargetBBOther =
    AreInvokes ?
      BBOther.getLandingPad(&TargetSymbolOther) :
      BBOther.getSuccessor(&TargetSymbolOther);

  if (TargetBB == nullptr || TargetBBOther == nullptr) {
    assert(!AreInvokes);
    // This is a tail call implemented with a jump that was not
    // converted to a call (e.g. conditional jump). Since the
    // instructions were not identical, the functions canot be
    // proven identical either.
    return std::make_pair(false, 0);
  }

  return std::make_pair(getIndex(TargetBB) == BF.getIndex(TargetBBOther),
                        TargetOpIndex);
}

bool BinaryFunction::isInstrEquivalentWith(
    const MCInst &Inst, const BinaryBasicBlock &BB, const MCInst &InstOther,
    const BinaryBasicBlock &BBOther, const BinaryFunction &BF) const {
  // First check their opcodes.
  if (Inst.getOpcode() != InstOther.getOpcode()) {
    return false;
  }

  // Then check if they have the same number of operands.
  unsigned NumOperands = Inst.getNumOperands();
  unsigned NumOperandsOther = InstOther.getNumOperands();
  if (NumOperands != NumOperandsOther) {
    return false;
  }

  // We are interested in 3 special cases:
  //
  // a) both instructions are recursive calls.
  // b) both instructions are local jumps to basic blocks with same indices.
  // c) both instructions are invokes with landing pad blocks with same indices.
  //
  // In any of these cases the instructions will differ in some operands, but
  // given identical CFG of the functions, they can still be considered
  // equivalent.
  bool BothCalls =
    BC.MIA->isCall(Inst) &&
    BC.MIA->isCall(InstOther);
  bool BothInvokes =
    BC.MIA->isInvoke(Inst) &&
    BC.MIA->isInvoke(InstOther);
  bool BothBranches =
    BC.MIA->isBranch(Inst) &&
    !BC.MIA->isIndirectBranch(Inst) &&
    BC.MIA->isBranch(InstOther) &&
    !BC.MIA->isIndirectBranch(InstOther);

  if (!BothCalls && !BothInvokes && !BothBranches) {
    return Inst.equals(InstOther);
  }

  // We figure out if both instructions are recursive calls (case a) or else
  // if they are calls to the same function.
  bool EquivCallees = false;
  unsigned CalleeOpIndex = 0;
  if (BothCalls) {
    std::tie(EquivCallees, CalleeOpIndex) =
      isCalleeEquivalentWith(Inst, BB, InstOther, BBOther, BF);
  }

  // We figure out if both instructions are jumps (case b) or invokes (case c)
  // with equivalent jump targets or landing pads respectively.
  assert(!(BothInvokes && BothBranches));
  bool SameTarget = false;
  unsigned TargetOpIndex = 0;
  if (BothInvokes || BothBranches) {
    std::tie(SameTarget, TargetOpIndex) =
      isTargetEquivalentWith(Inst, BB, InstOther, BBOther, BF, BothInvokes);
  }

  // Compare all operands.
  for (unsigned i = 0; i < NumOperands; ++i) {
    if (i == CalleeOpIndex && BothCalls && EquivCallees)
      continue;

    if (i == TargetOpIndex && (BothInvokes || BothBranches) && SameTarget)
      continue;

    if (!Inst.getOperand(i).equals(InstOther.getOperand(i)))
      return false;
  }

  // The instructions are equal although (some of) their operands
  // may differ.
  return true;
}

bool BinaryFunction::isIdenticalWith(const BinaryFunction &BF) const {

  assert(CurrentState == State::CFG && BF.CurrentState == State::CFG);

  // Compare the two functions, one basic block at a time.
  // Currently we require two identical basic blocks to have identical
  // instruction sequences and the same index in their corresponding
  // functions. The latter is important for CFG equality.

  // We do not consider functions with just different pseudo instruction
  // sequences non-identical by default. However we print a wanring
  // in case two instructions that are identical have different pseudo
  // instruction sequences.
  bool PseudosDiffer = false;

  if (size() != BF.size())
    return false;

  auto BBI = BF.begin();
  for (const BinaryBasicBlock *BB : BasicBlocks) {
    const BinaryBasicBlock *BBOther = &*BBI;
    if (getIndex(BB) != BF.getIndex(BBOther))
      return false;

    // Compare successor basic blocks.
    if (BB->succ_size() != BBOther->succ_size())
      return false;

    auto SuccBBI = BBOther->succ_begin();
    for (const BinaryBasicBlock *SuccBB : BB->successors()) {
      const BinaryBasicBlock *SuccBBOther = *SuccBBI;
      if (getIndex(SuccBB) != BF.getIndex(SuccBBOther))
        return false;
      ++SuccBBI;
    }

    // Compare landing pads.
    if (BB->lp_size() != BBOther->lp_size())
      return false;

    auto LPI = BBOther->lp_begin();
    for (const BinaryBasicBlock *LP : BB->landing_pads()) {
      const BinaryBasicBlock *LPOther = *LPI;
      if (getIndex(LP) != BF.getIndex(LPOther))
        return false;
      ++LPI;
    }

    // Compare instructions.
    auto I = BB->begin(), E = BB->end();
    auto OtherI = BBOther->begin(), OtherE = BBOther->end();
    while (I != E && OtherI != OtherE) {
      const MCInst &Inst = *I;
      const MCInst &InstOther = *OtherI;

      bool IsInstPseudo = BC.MII->get(Inst.getOpcode()).isPseudo();
      bool IsInstOtherPseudo = BC.MII->get(InstOther.getOpcode()).isPseudo();

      if (IsInstPseudo == IsInstOtherPseudo) {
        // Either both are pseudos or none is.
        bool areEqual =
          isInstrEquivalentWith(Inst, *BB, InstOther, *BBOther, BF);

        if (!areEqual && IsInstPseudo) {
          // Different pseudo instructions.
          PseudosDiffer = true;
        }
        else if (!areEqual) {
          // Different non-pseudo instructions.
          return false;
        }

        ++I; ++OtherI;
      }
      else {
        // One instruction is a pseudo while the other is not.
        PseudosDiffer = true;
        IsInstPseudo ? ++I : ++OtherI;
      }
    }

    // Check for trailing instructions or pseudos in one of the basic blocks.
    auto TrailI = I == E ? OtherI : I;
    auto TrailE = I == E ? OtherE : E;
    while (TrailI != TrailE) {
      const MCInst &InstTrail = *TrailI;
      if (!BC.MII->get(InstTrail.getOpcode()).isPseudo()) {
        // One of the functions has more instructions in this basic block
        // than the other, hence not identical.
        return false;
      }

      // There are trailing pseudos only in one of the basic blocks.
      PseudosDiffer = true;
      ++TrailI;
    }

    ++BBI;
  }

  if (opts::Verbosity >= 1 && PseudosDiffer) {
    errs() << "BOLT-WARNING: functions " << *this << " and "
           << BF << " are identical, but have different"
           << " pseudo instruction sequences.\n";
  }

  return true;
}

std::size_t BinaryFunction::hash() const {
  assert(CurrentState == State::CFG);

  // The hash is computed by creating a string of all the opcodes
  // in the function and hashing that string with std::hash.
  std::string Opcodes;
  for (const BinaryBasicBlock *BB : BasicBlocks) {
    for (const MCInst &Inst : *BB) {
      unsigned Opcode = Inst.getOpcode();

      if (BC.MII->get(Opcode).isPseudo())
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

  return std::hash<std::string>{}(Opcodes);
}

void BinaryFunction::insertBasicBlocks(
  BinaryBasicBlock *Start,
  std::vector<std::unique_ptr<BinaryBasicBlock>> &&NewBBs,
  bool UpdateCFIState) {
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

  // Recompute indices and offsets for all basic blocks after Start.
  uint64_t Offset = Start->getOffset();
  for (auto I = StartIndex; I < BasicBlocks.size(); ++I) {
    auto *BB = BasicBlocks[I];
    BB->setOffset(Offset);
    Offset += BC.computeCodeSize(BB->begin(), BB->end());
    BB->setIndex(I);
  }

  if (UpdateCFIState) {
    // Recompute CFI state for all BBs.
    BBCFIState.clear();
    annotateCFIState();
  }

  recomputeLandingPads(StartIndex, NumNewBlocks + 1);

  // Make sure the basic blocks are sorted properly.
  assert(std::is_sorted(begin(), end()));
}

// TODO: Which of these methods is better?
void BinaryFunction::updateLayout(BinaryBasicBlock* Start,
                                  const unsigned NumNewBlocks) {
  // Insert new blocks in the layout immediately after Start.
  auto Pos = std::find(layout_begin(), layout_end(), Start);
  assert(Pos != layout_end());
  auto Begin = &BasicBlocks[getIndex(Start) + 1];
  auto End = &BasicBlocks[getIndex(Start) + NumNewBlocks + 1];
  BasicBlocksLayout.insert(Pos + 1, Begin, End);
}

void BinaryFunction::updateLayout(LayoutType Type,
                                  bool MinBranchClusters,
                                  bool Split) {
  // Recompute layout with original parameters.
  BasicBlocksLayout = BasicBlocks;
  modifyLayout(Type, MinBranchClusters, Split);
}

BinaryFunction::~BinaryFunction() {
  for (auto BB : BasicBlocks) {
    delete BB;
  }
}

void BinaryFunction::emitJumpTables(MCStreamer *Streamer) {
  if (JumpTables.empty())
    return;

  Streamer->SwitchSection(BC.MOFI->getReadOnlySection());
  for (auto &JumpTable : JumpTables) {
    DEBUG(dbgs() << "BOLT-DEBUG: emitting jump table "
                 << JumpTable.StartLabel->getName() << '\n');
    Streamer->EmitLabel(JumpTable.StartLabel);
    // TODO (#9806207): based on jump table type (PIC vs non-PIC etc.)
    // we would need to emit different references.
    for (auto *Entry : JumpTable.Entries) {
      Streamer->EmitSymbolValue(Entry, BC.AsmInfo->getPointerSize());
    }
  }
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
          assert(BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE &&
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
          assert(BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE &&
                 "profile data not found");
          L->ExitCount += BI->Count;
        }
        ++BI;
      }
    }
  }
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

  // Update enumeration of basic blocks for correct detection of branch'
  // direction.
  updateLayoutIndices();

  for (const auto &BB : layout()) {
    // The basic block execution count equals to the sum of incoming branch
    // frequencies. This may deviate from the sum of outgoing branches of the
    // basic block especially since the block may contain a function that
    // does not return or a function that throws an exception.
    uint64_t BBExecutionCount = 0;
    for (const auto &BI : BB->branch_info())
      if (BI.Count != BinaryBasicBlock::COUNT_NO_PROFILE)
        BBExecutionCount += BI.Count;

    // Ignore empty blocks and blocks that were not executed.
    if (BB->getNumNonPseudos() == 0 || BBExecutionCount == 0)
      continue;

    // Count the number of calls by iterating through all instructions.
    for (const auto &Instr : *BB) {
      if (!BC.MIA->isCall(Instr))
        continue;
      Stats[DynoStats::FUNCTION_CALLS] += BBExecutionCount;
      if (BC.MIA->getMemoryOperandNo(Instr) != -1) {
        Stats[DynoStats::INDIRECT_CALLS] += BBExecutionCount;
      } else if (const auto *CallSymbol = BC.MIA->getTargetSymbol(Instr)) {
        if (BC.getFunctionForSymbol(CallSymbol))
          continue;
        auto GSI = BC.GlobalSymbols.find(CallSymbol->getName());
        if (GSI == BC.GlobalSymbols.end())
          continue;
        auto Section = BC.getSectionForAddress(GSI->second);
        if (!Section)
          continue;
        StringRef SectionName;
        Section->getName(SectionName);
        if (SectionName == ".plt") {
          Stats[DynoStats::PLT_CALLS] += BBExecutionCount;
        }
      }
    }

    Stats[DynoStats::INSTRUCTIONS] += BB->getNumNonPseudos() * BBExecutionCount;

    // Jump tables.
    const auto *LastInstr = BB->findLastNonPseudoInstruction();
    if (BC.MIA->getJumpTableIndex(*LastInstr) > 0) {
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

    // Conditional branch that could be followed by an unconditional branch.
    uint64_t TakenCount = BB->getBranchInfo(true).Count;
    if (TakenCount == COUNT_NO_PROFILE)
      TakenCount = 0;
    uint64_t NonTakenCount = BB->getBranchInfo(false).Count;
    if (NonTakenCount == COUNT_NO_PROFILE)
      NonTakenCount = 0;

    assert(TakenCount + NonTakenCount == BBExecutionCount &&
           "internal calculation error");

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

void DynoStats::print(raw_ostream &OS, const DynoStats *Other) const {
  auto printStatWithDelta = [&](const std::string &Name, uint64_t Stat,
                                uint64_t OtherStat) {
    OS << format("%'20lld : ", Stat * opts::DynoStatsScale) << Name;
    if (Other) {
       OS << format(" (%+.1f%%)",
                    ( (float) Stat - (float) OtherStat ) * 100.0 /
                      (float) (OtherStat + 1) );
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
