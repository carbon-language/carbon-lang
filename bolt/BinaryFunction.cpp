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
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <queue>
#include <string>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

namespace llvm {
namespace bolt {

namespace opts {

static cl::opt<bool>
AgressiveSplitting("split-all-cold",
                   cl::desc("outline as many cold basic blocks as possible"),
                   cl::Optional);

static cl::opt<bool>
PrintClusters("print-clusters", cl::desc("print clusters"), cl::Optional);

static cl::opt<bool>
PrintDebugInfo("print-debug-info",
               cl::desc("print debug info when printing functions"),
               cl::Hidden);

} // namespace opts

namespace {

// Finds which DWARF compile unit owns an address in the executable by
// querying .debug_aranges.
DWARFCompileUnit *FindCompileUnitForAddress(uint64_t Address,
                                            const BinaryContext &BC) {
  auto DebugAranges = BC.DwCtx->getDebugAranges();
  if (!DebugAranges)
    return nullptr;

  uint32_t CompileUnitIndex = DebugAranges->findAddress(Address);

  auto It = BC.OffsetToDwarfCU.find(CompileUnitIndex);
  if (It == BC.OffsetToDwarfCU.end()) {
    return nullptr;
  } else {
    return It->second;
  }
}

} // namespace

uint64_t BinaryFunction::Count = 0;

BinaryBasicBlock *
BinaryFunction::getBasicBlockContainingOffset(uint64_t Offset) {
  if (Offset > Size)
    return nullptr;

  if (BasicBlocks.empty())
    return nullptr;

  auto I = std::upper_bound(BasicBlocks.begin(),
                            BasicBlocks.end(),
                            BinaryBasicBlock(Offset));
  assert(I != BasicBlocks.begin() && "first basic block not at offset 0");

  return &(*--I);
}

size_t
BinaryFunction::getBasicBlockOriginalSize(const BinaryBasicBlock *BB) const {
  auto Index = getIndex(BB);
  if (Index + 1 == BasicBlocks.size()) {
    return Size - BB->getOffset();
  } else {
    return BasicBlocks[Index + 1].getOffset() - BB->getOffset();
  }
}

unsigned BinaryFunction::eraseDeadBBs(
    std::map<BinaryBasicBlock *, bool> &ToPreserve) {
  BasicBlockOrderType NewLayout;
  unsigned Count = 0;
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

void BinaryFunction::print(raw_ostream &OS, std::string Annotation,
                           bool PrintInstructions) const {
  StringRef SectionName;
  Section.getName(SectionName);
  OS << "Binary Function \"" << getName() << "\" " << Annotation << " {"
     << "\n  Number      : "   << FunctionNumber
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
  if (ExecutionCount != COUNT_NO_PROFILE)
    OS << "\n  Exec Count  : " << ExecutionCount;

  OS << "\n}\n";

  if (!PrintInstructions || !BC.InstPrinter)
    return;

  // Offset of the instruction in function.
  uint64_t Offset{0};

  auto printCFI = [&OS] (uint32_t Operation) {
    switch(Operation) {
    case MCCFIInstruction::OpSameValue:        OS << "OpSameValue";       break;
    case MCCFIInstruction::OpRememberState:    OS << "OpRememberState";   break;
    case MCCFIInstruction::OpRestoreState:     OS << "OpRestoreState";    break;
    case MCCFIInstruction::OpOffset:           OS << "OpOffset";          break;
    case MCCFIInstruction::OpDefCfaRegister:   OS << "OpDefCfaRegister";  break;
    case MCCFIInstruction::OpDefCfaOffset:     OS << "OpDefCfaOffset";    break;
    case MCCFIInstruction::OpDefCfa:           OS << "OpDefCfa";          break;
    case MCCFIInstruction::OpRelOffset:        OS << "OpRelOffset";       break;
    case MCCFIInstruction::OpAdjustCfaOffset:  OS << "OfAdjustCfaOffset"; break;
    case MCCFIInstruction::OpEscape:           OS << "OpEscape";          break;
    case MCCFIInstruction::OpRestore:          OS << "OpRestore";         break;
    case MCCFIInstruction::OpUndefined:        OS << "OpUndefined";       break;
    case MCCFIInstruction::OpRegister:         OS << "OpRegister";        break;
    case MCCFIInstruction::OpWindowSave:       OS << "OpWindowSave";      break;
    case MCCFIInstruction::OpGnuArgsSize:      OS << "OpGnuArgsSize";     break;
    default:                                   OS << "Op#" << Operation; break;
    }
  };

  // Used in printInstruction below to print debug line information.
  DWARFCompileUnit *Unit = nullptr;
  const DWARFDebugLine::LineTable *LineTable = nullptr;

  if (opts::PrintDebugInfo) {
    Unit = FindCompileUnitForAddress(getAddress(), BC);
    LineTable = Unit ? BC.DwCtx->getLineTableForUnit(Unit) : nullptr;
  }

  auto printInstruction = [&](const MCInst &Instruction) {
    if (BC.MIA->isEHLabel(Instruction)) {
      OS << "  EH_LABEL: "
         << cast<MCSymbolRefExpr>(Instruction.getOperand(0).getExpr())->
                                                                    getSymbol()
         << '\n';
      return;
    }
    OS << format("    %08" PRIx64 ": ", Offset);
    if (BC.MIA->isCFI(Instruction)) {
      uint32_t Offset = Instruction.getOperand(0).getImm();
      OS << "\t!CFI\t$" << Offset << "\t; ";
      assert(Offset < FrameInstructions.size() && "Invalid CFI offset");
      printCFI(FrameInstructions[Offset].getOperation());
      OS << "\n";
      return;
    }
    BC.InstPrinter->printInst(&Instruction, OS, "", *BC.STI);
    if (BC.MIA->isCall(Instruction)) {
      if (BC.MIA->isTailCall(Instruction))
        OS << " # TAILCALL ";
      if (BC.MIA->isInvoke(Instruction)) {
        const MCSymbol *LP;
        uint64_t Action;
        std::tie(LP, Action) = BC.MIA->getEHInfo(Instruction);
        OS << " # handler: ";
        if (LP)
          OS << *LP;
        else
          OS << '0';
        OS << "; action: " << Action;
        auto GnuArgsSize = BC.MIA->getGnuArgsSize(Instruction);
        if (GnuArgsSize >= 0)
          OS << "; GNU_args_size = " << GnuArgsSize;
      }
    }
    if (opts::PrintDebugInfo && LineTable) {
      auto RowRef = DebugLineTableRowRef::fromSMLoc(Instruction.getLoc());

      if (RowRef != DebugLineTableRowRef::NULL_ROW) {
        const auto &Row = LineTable->Rows[RowRef.RowIndex - 1];
        OS << " # debug line "
          << LineTable->Prologue.FileNames[Row.File - 1].Name
          << ":" << Row.Line;

        if (Row.Column) {
          OS << ":" << Row.Column;
        }
      }
    }

    OS << "\n";
    // In case we need MCInst printer:
    // Instr.dump_pretty(OS, InstructionPrinter.get());
  };

  if (BasicBlocks.empty() && !Instructions.empty()) {
    // Print before CFG was built.
    for (const auto &II : Instructions) {
      auto Offset = II.first;

      // Print label if exists at this offset.
      auto LI = Labels.find(Offset);
      if (LI != Labels.end())
        OS << LI->second->getName() << ":\n";

      printInstruction(II.second);
    }
  }

  for (uint32_t I = 0, E = BasicBlocksLayout.size(); I != E; ++I) {
    auto BB = BasicBlocksLayout[I];
    if (I != 0 &&
        BB->IsCold != BasicBlocksLayout[I - 1]->IsCold)
      OS << "-------   HOT-COLD SPLIT POINT   -------\n\n";

    OS << BB->getName() << " ("
       << BB->Instructions.size() << " instructions, align : "
       << BB->getAlignment() << ")\n";

    if (LandingPads.find(BB->getLabel()) != LandingPads.end()) {
      OS << "  Landing Pad\n";
    }

    uint64_t BBExecCount = BB->getExecutionCount();
    if (BBExecCount != BinaryBasicBlock::COUNT_NO_PROFILE) {
      OS << "  Exec Count : " << BBExecCount << "\n";
    }
    if (!BBCFIState.empty()) {
      OS << "  CFI State : " << BBCFIState[getIndex(BB)] << '\n';
    }
    if (!BB->Predecessors.empty()) {
      OS << "  Predecessors: ";
      auto Sep = "";
      for (auto Pred : BB->Predecessors) {
        OS << Sep << Pred->getName();
        Sep = ", ";
      }
      OS << '\n';
    }

    Offset = RoundUpToAlignment(Offset, BB->getAlignment());

    for (auto &Instr : *BB) {
      printInstruction(Instr);

      // Calculate the size of the instruction.
      // Note: this is imprecise since happening prior to relaxation.
      SmallString<256> Code;
      SmallVector<MCFixup, 4> Fixups;
      raw_svector_ostream VecOS(Code);
      BC.MCE->encodeInstruction(Instr, VecOS, Fixups, *BC.STI);
      Offset += Code.size();
    }

    if (!BB->Successors.empty()) {
      OS << "  Successors: ";
      auto BI = BB->BranchInfo.begin();
      auto Sep = "";
      for (auto Succ : BB->Successors) {
        assert(BI != BB->BranchInfo.end() && "missing BranchInfo entry");
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

  OS << "DWARF CFI Instructions:\n";
  if (OffsetToCFI.size()) {
    // Pre-buildCFG information
    for (auto &Elmt : OffsetToCFI) {
      OS << format("    %08x:\t", Elmt.first);
      assert(Elmt.second < FrameInstructions.size() && "Incorrect CFI offset");
      printCFI(FrameInstructions[Elmt.second].getOperation());
      OS << "\n";
    }
  } else {
    // Post-buildCFG information
    for (uint32_t I = 0, E = FrameInstructions.size(); I != E; ++I) {
      const MCCFIInstruction &CFI = FrameInstructions[I];
      OS << format("    %d:\t", I);
      printCFI(CFI.getOperation());
      OS << "\n";
    }
  }
  if (FrameInstructions.empty())
    OS << "    <empty>\n";

  OS << "End of Function \"" << getName() << "\"\n\n";
}

bool BinaryFunction::disassemble(ArrayRef<uint8_t> FunctionData,
                                 bool ExtractDebugLineData) {
  assert(FunctionData.size() == getSize() &&
         "function size does not match raw data size");

  auto &Ctx = BC.Ctx;
  auto &MIA = BC.MIA;
  DWARFCompileUnit *CompileUnit = nullptr;

  if (ExtractDebugLineData) {
    CompileUnit = FindCompileUnitForAddress(getAddress(), BC);
  }

  // Insert a label at the beginning of the function. This will be our first
  // basic block.
  Labels[0] = Ctx->createTempSymbol("BB0", false);

  auto handleRIPOperand =
      [&](MCInst &Instruction, uint64_t Address, uint64_t Size) -> bool {
        uint64_t TargetAddress{0};
        MCSymbol *TargetSymbol{nullptr};
        if (!BC.MIA->evaluateRIPOperand(Instruction, Address, Size,
                                        TargetAddress)) {
          DEBUG(dbgs() << "BOLT: rip-relative operand could not be evaluated:\n";
                BC.InstPrinter->printInst(&Instruction, dbgs(), "", *BC.STI);
                dbgs() << '\n';
                Instruction.dump_pretty(dbgs(), BC.InstPrinter.get());
                dbgs() << '\n';);
          return false;
        }
        // FIXME: check that the address is in data, not in code.
        if (TargetAddress == 0) {
          errs() << "BOLT-WARNING: rip-relative operand is zero in function "
                 << getName() << ". Ignoring function.\n";
          return false;
        }
        TargetSymbol = BC.getOrCreateGlobalSymbol(TargetAddress, "DATAat");
        BC.MIA->replaceRIPOperandDisp(
            Instruction, MCOperand::createExpr(MCSymbolRefExpr::create(
                             TargetSymbol, MCSymbolRefExpr::VK_None, *BC.Ctx)));
        return true;
      };

  bool IsSimple = true;
  for (uint64_t Offset = 0; IsSimple && (Offset < getSize()); ) {
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
      errs() << "BOLT-WARNING: unable to disassemble instruction at offset 0x"
             << Twine::utohexstr(Offset) << " (address 0x"
             << Twine::utohexstr(AbsoluteInstrAddr) << ") in function "
             << getName() << '\n';
      IsSimple = false;
      break;
    }

    if (MIA->isUnsupported(Instruction)) {
      errs() << "BOLT-WARNING: unsupported instruction seen at offset 0x"
             << Twine::utohexstr(Offset) << " (address 0x"
             << Twine::utohexstr(AbsoluteInstrAddr) << ") in function "
             << getName() << '\n';
      IsSimple = false;
      break;
    }

    if (MIA->isBranch(Instruction) || MIA->isCall(Instruction)) {
      uint64_t InstructionTarget = 0;
      if (MIA->evaluateBranch(Instruction,
                              AbsoluteInstrAddr,
                              Size,
                              InstructionTarget)) {
        // Check if the target is within the same function. Otherwise it's
        // a call, possibly a tail call.
        //
        // If the target *is* the function address it could be either a branch
        // or a recursive call.
        bool IsCall = MIA->isCall(Instruction);
        bool IsCondBranch = MIA->isConditionalBranch(Instruction);
        MCSymbol *TargetSymbol{nullptr};
        uint64_t TargetOffset{0};

        if (IsCall && containsAddress(InstructionTarget)) {
          if (InstructionTarget == getAddress()) {
            // Recursive call.
            TargetSymbol = Ctx->getOrCreateSymbol(getName());
          } else {
            // Possibly an old-style PIC code
            errs() << "BOLT: internal call detected at 0x"
                   << Twine::utohexstr(AbsoluteInstrAddr)
                   << " in function " << getName() << ". Skipping.\n";
            IsSimple = false;
          }
        }

        if (!TargetSymbol) {
          // Create either local label or external symbol.
          if (containsAddress(InstructionTarget)) {
            // Check if there's already a registered label.
            TargetOffset = InstructionTarget - getAddress();
            auto LI = Labels.find(TargetOffset);
            if (LI == Labels.end()) {
              TargetSymbol = Ctx->createTempSymbol();
              Labels[TargetOffset] = TargetSymbol;
            } else {
              TargetSymbol = LI->second;
            }
          } else {
            BC.InterproceduralBranchTargets.insert(InstructionTarget);
            if (!IsCall && Size == 2) {
              errs() << "BOLT-WARNING: relaxed tail call detected at 0x"
                     << Twine::utohexstr(AbsoluteInstrAddr)
                     << ". Code size will be increased.\n";
            }

            // This is a call regardless of the opcode.
            // Assign proper opcode for tail calls, so that they could be
            // treated as calls.
            if (!IsCall) {
              MIA->convertJmpToTailCall(Instruction);
              IsCall = true;
            }

            TargetSymbol = BC.getOrCreateGlobalSymbol(InstructionTarget,
                                                      "FUNCat");
            if (InstructionTarget == 0) {
              // We actually see calls to address 0 because of the weak symbols
              // from the libraries. In reality more often than not it is
              // unreachable code, but we don't know it and have to emit calls
              // to 0 which make LLVM JIT unhappy.
              errs() << "BOLT-WARNING: Function " << getName()
                     << " has a call to address zero. Ignoring function.\n";
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
          // Add local branch info.
          LocalBranches.push_back({Offset, TargetOffset});
        }
        if (IsCondBranch) {
          // Add fallthrough branch info.
          FTBranches.push_back({Offset, Offset + Size});
        }
      } else {
        // Should be an indirect call or an indirect branch. Bail out on the
        // latter case.
        if (MIA->isIndirectBranch(Instruction)) {
          DEBUG(dbgs() << "BOLT-WARNING: indirect branch detected at 0x"
                 << Twine::utohexstr(AbsoluteInstrAddr)
                 << ". Skipping function " << getName() << ".\n");
          IsSimple = false;
        }
        // Indirect call. We only need to fix it if the operand is RIP-relative
        if (MIA->hasRIPOperand(Instruction)) {
          if (!handleRIPOperand(Instruction, AbsoluteInstrAddr, Size)) {
            errs() << "BOLT-WARNING: cannot handle RIP operand at 0x"
                   << Twine::utohexstr(AbsoluteInstrAddr)
                   << ". Skipping function " << getName() << ".\n";
            IsSimple = false;
          }
        }
      }
    } else {
      if (MIA->hasRIPOperand(Instruction)) {
        if (!handleRIPOperand(Instruction, AbsoluteInstrAddr, Size)) {
          errs() << "BOLT-WARNING: cannot handle RIP operand at 0x"
                 << Twine::utohexstr(AbsoluteInstrAddr)
                 << ". Skipping function " << getName() << ".\n";
          IsSimple = false;
        }
      }
    }

    if (CompileUnit) {
      Instruction.setLoc(
          findDebugLineInformationForInstructionAt(AbsoluteInstrAddr,
                                                   CompileUnit));
    }

    addInstruction(Offset, std::move(Instruction));

    Offset += Size;
  }

  setSimple(IsSimple);

  // TODO: clear memory if not simple function?

  // Update state.
  updateState(State::Disassembled);

  return true;
}

SMLoc
BinaryFunction::findDebugLineInformationForInstructionAt(
    uint64_t Address,
    DWARFCompileUnit *Unit) {
  // We use the pointer in SMLoc to store an instance of DebugLineTableRowRef,
  // which occupies 64 bits. Thus, we can only proceed if the struct fits into
  // the pointer itself.
  assert(
      sizeof(decltype(SMLoc().getPointer())) >= sizeof(DebugLineTableRowRef) &&
      "Cannot fit instruction debug line information into SMLoc's pointer");

  const DWARFDebugLine::LineTable *LineTable =
      BC.DwCtx->getLineTableForUnit(Unit);

  SMLoc NullResult = DebugLineTableRowRef::NULL_ROW.toSMLoc();

  if (!LineTable) {
    return NullResult;
  }

  uint32_t RowIndex = LineTable->lookupAddress(Address);

  if (RowIndex == LineTable->UnknownRowIndex) {
    return NullResult;
  }

  assert(RowIndex < LineTable->Rows.size() &&
         "Line Table lookup returned invalid index.");

  decltype(SMLoc().getPointer()) Ptr;
  DebugLineTableRowRef *InstructionLocation =
    reinterpret_cast<DebugLineTableRowRef *>(&Ptr);

  InstructionLocation->DwCompileUnitIndex = Unit->getOffset();
  InstructionLocation->RowIndex = RowIndex + 1;

  return SMLoc::getFromPointer(Ptr);
}

bool BinaryFunction::buildCFG() {

  auto &MIA = BC.MIA;

  auto BranchDataOrErr = BC.DR.getFuncBranchData(getName());
  if (std::error_code EC = BranchDataOrErr.getError()) {
    DEBUG(dbgs() << "no branch data found for \"" << getName() << "\"\n");
  } else {
    if (!BranchDataOrErr.get().Data.empty())
      ExecutionCount = BranchDataOrErr.get().ExecutionCount;
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
  bool IsLastInstrNop = false;
  MCInst *PrevInstr{nullptr};

  auto addCFIPlaceholders =
      [this](uint64_t CFIOffset, BinaryBasicBlock *InsertBB) {
        for (auto FI = OffsetToCFI.lower_bound(CFIOffset),
                  FE = OffsetToCFI.upper_bound(CFIOffset);
             FI != FE; ++FI) {
          addCFIPseudo(InsertBB, InsertBB->end(), FI->second);
        }
      };

  for (auto I = Instructions.begin(), E = Instructions.end(); I != E; ++I) {
    auto &InstrInfo = *I;
    auto LI = Labels.find(InstrInfo.first);
    if (LI != Labels.end()) {
      // Always create new BB at branch destination.
      PrevBB = InsertBB;
      InsertBB = addBasicBlock(LI->first, LI->second,
                               /* DeriveAlignment = */ IsLastInstrNop);
    }
    // Ignore nops. We use nops to derive alignment of the next basic block.
    // It will not always work, as some blocks are naturally aligned, but
    // it's just part of heuristic for block alignment.
    if (MIA->isNoop(InstrInfo.second)) {
      IsLastInstrNop = true;
      continue;
    }
    if (!InsertBB) {
      // It must be a fallthrough or unreachable code. Create a new block unless
      // we see an unconditional branch following a conditional one.
      assert(PrevBB && "no previous basic block for a fall through");
      assert(PrevInstr && "no previous instruction for a fall through");
      if (MIA->isUnconditionalBranch(InstrInfo.second) &&
          !MIA->isUnconditionalBranch(*PrevInstr)) {
        // Temporarily restore inserter basic block.
        InsertBB = PrevBB;
      } else {
        InsertBB = addBasicBlock(InstrInfo.first,
                                 BC.Ctx->createTempSymbol("FT", true),
                                 /* DeriveAlignment = */ IsLastInstrNop);
      }
    }
    if (InstrInfo.first == 0) {
      // Add associated CFI pseudos in the first offset (0)
      addCFIPlaceholders(0, InsertBB);
    }

    IsLastInstrNop = false;
    InsertBB->addInstruction(InstrInfo.second);
    PrevInstr = &InstrInfo.second;
    // Add associated CFI instrs. We always add the CFI instruction that is
    // located immediately after this instruction, since the next CFI
    // instruction reflects the change in state caused by this instruction.
    auto NextInstr = I;
    ++NextInstr;
    uint64_t CFIOffset;
    if (NextInstr != E)
      CFIOffset = NextInstr->first;
    else
      CFIOffset = getSize();
    addCFIPlaceholders(CFIOffset, InsertBB);

    // Store info about associated landing pad.
    if (MIA->isInvoke(InstrInfo.second)) {
      const MCSymbol *LP;
      uint64_t Action;
      std::tie(LP, Action) = MIA->getEHInfo(InstrInfo.second);
      if (LP) {
        LPToBBIndex[LP].push_back(getIndex(InsertBB));
      }
    }

    // How well do we detect tail calls here?
    if (MIA->isTerminator(InstrInfo.second)) {
      PrevBB = InsertBB;
      InsertBB = nullptr;
    }
  }

  // Set the basic block layout to the original order.
  for (auto &BB : BasicBlocks) {
    BasicBlocksLayout.emplace_back(&BB);
  }

  // Intermediate dump.
  DEBUG(print(dbgs(), "after creating basic blocks"));

  // TODO: handle properly calls to no-return functions,
  // e.g. exit(3), etc. Otherwise we'll see a false fall-through
  // blocks.

  for (auto &Branch : LocalBranches) {
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
    BinaryBasicBlock *ToBB = getBasicBlockAtOffset(Branch.second);
    // We have a fall-through that does not point to another BB, ignore it as
    // it may happen in cases where we have a BB finished by two branches.
    if (ToBB == nullptr)
      continue;

    // Does not add a successor if we can't find profile data, leave it to the
    // inference pass to guess its frequency
    if (!BranchDataOrErr.getError()) {
      const FuncBranchData &BranchData = BranchDataOrErr.get();
      auto BranchInfoOrErr = BranchData.getBranch(Branch.first, Branch.second);
      if (!BranchInfoOrErr.getError()) {
        const BranchInfo &BInfo = BranchInfoOrErr.get();
        FromBB->addSuccessor(ToBB, BInfo.Branches, BInfo.Mispreds);
      }
    }
  }

  // Add fall-through branches (except for non-taken conditional branches with
  // profile data, which were already accounted for in LocalBranches).
  PrevBB = nullptr;
  bool IsPrevFT = false; // Is previous block a fall-through.
  for (auto &BB : BasicBlocks) {
    if (IsPrevFT) {
      PrevBB->addSuccessor(&BB, BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE,
                           BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE);
    }
    if (BB.empty()) {
      IsPrevFT = true;
      PrevBB = &BB;
      continue;
    }

    auto LastInstIter = --BB.end();
    while (MIA->isCFI(*LastInstIter) && LastInstIter != BB.begin())
      --LastInstIter;
    if (BB.succ_size() == 0) {
      IsPrevFT = MIA->isTerminator(*LastInstIter) ? false : true;
    } else if (BB.succ_size() == 1) {
      IsPrevFT =  MIA->isConditionalBranch(*LastInstIter) ? true : false;
    } else {
      // Ends with 2 branches, with an indirect jump or it is a conditional
      // branch whose frequency has been inferred from LBR
      IsPrevFT = false;
    }

    PrevBB = &BB;
  }

  if (!IsPrevFT) {
    // Possibly a call that does not return.
    DEBUG(dbgs() << "last block was marked as a fall-through\n");
  }

  // Add associated landing pad blocks to each basic block.
  for (auto &BB : BasicBlocks) {
    if (LandingPads.find(BB.getLabel()) != LandingPads.end()) {
      MCSymbol *LP = BB.getLabel();
      for (unsigned I : LPToBBIndex.at(LP)) {
        BinaryBasicBlock *ThrowBB = getBasicBlockAtIndex(I);
        ThrowBB->addLandingPad(&BB);
      }
    }
  }

  // Infer frequency for non-taken branches
  if (ExecutionCount != COUNT_NO_PROFILE && !BranchDataOrErr.getError()) {
    inferFallThroughCounts();
  }

  // Update CFI information for each BB
  annotateCFIState();

  // Clean-up memory taken by instructions and labels.
  clearInstructions();
  clearCFIOffsets();
  clearLabels();
  clearLocalBranches();
  clearFTBranches();
  clearLPToBBIndex();

  // Update the state.
  CurrentState = State::CFG;

  // Annotate invoke instructions with GNU_args_size data.
  propagateGnuArgsSizeInfo();

  return true;
}

void BinaryFunction::inferFallThroughCounts() {
  assert(!BasicBlocks.empty() && "basic block list should not be empty");

  auto BranchDataOrErr = BC.DR.getFuncBranchData(getName());

  // Compute preliminary execution time for each basic block
  for (auto &CurBB : BasicBlocks) {
    if (&CurBB == &*BasicBlocks.begin()) {
      CurBB.ExecutionCount = ExecutionCount;
      continue;
    }
    CurBB.ExecutionCount = 0;
  }

  for (auto &CurBB : BasicBlocks) {
    auto SuccCount = CurBB.BranchInfo.begin();
    for (auto Succ : CurBB.successors()) {
      // Do not update execution count of the entry block (when we have tail
      // calls). We already accounted for those when computing the func count.
      if (Succ == &*BasicBlocks.begin())
        continue;
      if (SuccCount->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        Succ->ExecutionCount += SuccCount->Count;
      ++SuccCount;
    }
  }

  // Udate execution counts of landing pad blocks.
  if (!BranchDataOrErr.getError()) {
    const FuncBranchData &BranchData = BranchDataOrErr.get();
    for (const auto &I : BranchData.EntryData) {
      BinaryBasicBlock *BB = getBasicBlockAtOffset(I.To.Offset);
      if (BB && LandingPads.find(BB->getLabel()) != LandingPads.end()) {
        BB->ExecutionCount += I.Branches;
      }
    }
  }

  // Work on a basic block at a time, propagating frequency information forwards
  // It is important to walk in the layour order
  for (auto &CurBB : BasicBlocks) {
    uint64_t BBExecCount = CurBB.getExecutionCount();

    // Propagate this information to successors, filling in fall-through edges
    // with frequency information
    if (CurBB.succ_size() == 0)
      continue;

    // Calculate frequency of outgoing branches from this node according to
    // LBR data
    uint64_t ReportedBranches = 0;
    for (auto &SuccCount : CurBB.BranchInfo) {
      if (SuccCount.Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        ReportedBranches += SuccCount.Count;
    }

    // Calculate frequency of throws from this node according to LBR data
    // for branching into associated landing pads. Since it is possible
    // for a landing pad to be associated with more than one basic blocks,
    // we may overestimate the frequency of throws for such blocks.
    uint64_t ReportedThrows = 0;
    for (BinaryBasicBlock *LP: CurBB.LandingPads) {
      ReportedThrows += LP->ExecutionCount;
    }

    uint64_t TotalReportedJumps = ReportedBranches + ReportedThrows;

    // Infer the frequency of the fall-through edge, representing not taking the
    // branch
    uint64_t Inferred = 0;
    if (BBExecCount > TotalReportedJumps)
      Inferred = BBExecCount - TotalReportedJumps;

    DEBUG({
      if (BBExecCount < TotalReportedJumps)
        dbgs()
            << "BOLT-WARNING: Fall-through inference is slightly inconsistent. "
               "exec frequency is less than the outgoing edges frequency ("
            << BBExecCount << " < " << ReportedBranches
            << ") for  BB at offset 0x"
            << Twine::utohexstr(getAddress() + CurBB.getOffset()) << '\n';
    });

    // Put this information into the fall-through edge
    if (CurBB.succ_size() == 0)
      continue;
    // If there is a FT, the last successor will be it.
    auto &SuccCount = CurBB.BranchInfo.back();
    auto &Succ = CurBB.Successors.back();
    if (SuccCount.Count == BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE) {
      SuccCount.Count = Inferred;
      Succ->ExecutionCount += Inferred;
    }

  } // end for (CurBB : BasicBlocks)

  return;
}

uint64_t BinaryFunction::getFunctionScore() {
  if (FunctionScore != -1)
    return FunctionScore;

  uint64_t TotalScore = 0ULL;
  for (auto BB : layout()) {
    uint64_t BBExecCount = BB->getExecutionCount();
    if (BBExecCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      continue;
    BBExecCount *= (BB->Instructions.size() - BB->getNumPseudos());
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
    BinaryBasicBlock &CurBB = *CI;
    // Annotate this BB entry
    BBCFIState.emplace_back(State);

    // Advance state
    for (const auto &Instr : CurBB) {
      MCCFIInstruction *CFI = getCFIFor(Instr);
      if (CFI == nullptr)
        continue;
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
    }
  }

  // Store the state after the last BB
  BBCFIState.emplace_back(State);

  assert(StateStack.empty() && "Corrupt CFI stack");
}

bool BinaryFunction::fixCFIState() {
  auto Sep = "";
  DEBUG(dbgs() << "Trying to fix CFI states for each BB after reordering.\n");
  DEBUG(dbgs() << "This is the list of CFI states for each BB of " << getName()
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
          errs() << "BOLT-WARNING: CFI rewriter detected nested CFI state while "
                 << " replaying CFI instructions for BB " << InBB->getName()
                 << " in function " << getName() << '\n';
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
  BinaryBasicBlock *EntryBB = *BasicBlocksLayout.begin();
  for (uint32_t I = 0, E = BasicBlocksLayout.size(); I != E; ++I) {
    BinaryBasicBlock *BB = BasicBlocksLayout[I];
    uint32_t BBIndex = getIndex(BB);

    // Hot-cold border: check if this is the first BB to be allocated in a cold
    // region (a different function). If yes, we need to reset the CFI state.
    if (I != 0 &&
        BB->IsCold != BasicBlocksLayout[I - 1]->IsCold)
      State = 0;

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
      BinaryBasicBlock::const_iterator InsertIt = EntryBB->begin();
      while (InsertIt != EntryBB->end() && BC.MIA->isCFI(*InsertIt))
        ++InsertIt;
      addCFIPseudo(EntryBB, InsertIt, FrameInstructions.size());
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
          if (MCCFIInstruction *CFI = getCFIFor(Instr)) {
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
        errs() << " BOLT-WARNING: not possible to remember/recover state"
               << " without corrupting CFI state stack in function "
               << getName() << "\n";
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

void BinaryFunction::modifyLayout(LayoutType Type, bool Split) {
  if (BasicBlocksLayout.empty() || Type == LT_NONE)
    return;

  if (Type == LT_REVERSE) {
    BasicBlockOrderType ReverseOrder;
    auto FirstBB = BasicBlocksLayout.front();
    ReverseOrder.push_back(FirstBB);
    for (auto RBBI = BasicBlocksLayout.rbegin(); *RBBI != FirstBB; ++RBBI)
      ReverseOrder.push_back(*RBBI);
    BasicBlocksLayout.swap(ReverseOrder);

    if (Split)
      splitFunction();

    fixBranches();

    return;
  }

  // Cannot do optimal layout without profile.
  if (getExecutionCount() == BinaryFunction::COUNT_NO_PROFILE)
    return;

  // Work on optimal solution if problem is small enough
  if (BasicBlocksLayout.size() <= FUNC_SIZE_THRESHOLD)
    return solveOptimalLayout(Split);

  DEBUG(dbgs() << "running block layout heuristics on " << getName() << "\n");

  // Greedy heuristic implementation for the TSP, applied to BB layout. Try to
  // maximize weight during a path traversing all BBs. In this way, we will
  // convert the hottest branches into fall-throughs.

  // Encode an edge between two basic blocks, source and destination
  typedef std::pair<BinaryBasicBlock *, BinaryBasicBlock *> EdgeTy;
  std::map<EdgeTy, uint64_t> Weight;

  // Define a comparison function to establish SWO between edges
  auto Comp = [&] (EdgeTy A, EdgeTy B) {
    // With equal weights, prioritize branches with lower index
    // source/destination. This helps to keep original block order for blocks
    // when optimal order cannot be deducted from a profile.
    if (Weight[A] == Weight[B]) {
      uint32_t ASrcBBIndex = getIndex(A.first);
      uint32_t BSrcBBIndex = getIndex(B.first);
      if (ASrcBBIndex != BSrcBBIndex)
        return ASrcBBIndex > BSrcBBIndex;
      return getIndex(A.second) > getIndex(B.second);
    }
    return Weight[A] < Weight[B];
  };
  std::priority_queue<EdgeTy, std::vector<EdgeTy>, decltype(Comp)> Queue(Comp);

  typedef std::vector<BinaryBasicBlock *> ClusterTy;
  typedef std::map<BinaryBasicBlock *, int> BBToClusterMapTy;
  std::vector<ClusterTy> Clusters;
  BBToClusterMapTy BBToClusterMap;

  // Encode relative weights between two clusters
  std::vector<std::map<uint32_t, uint64_t>> ClusterEdges;
  ClusterEdges.resize(BasicBlocksLayout.size());

  for (auto BB : BasicBlocksLayout) {
    // Create a cluster for this BB
    uint32_t I = Clusters.size();
    Clusters.emplace_back();
    auto &Cluster = Clusters.back();
    Cluster.push_back(BB);
    BBToClusterMap[BB] = I;
    // Populate priority queue with edges
    auto BI = BB->BranchInfo.begin();
    for (auto &I : BB->successors()) {
      if (BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        Weight[std::make_pair(BB, I)] = BI->Count;
      Queue.push(std::make_pair(BB, I));
      ++BI;
    }
  }

  // Grow clusters in a greedy fashion
  while (!Queue.empty()) {
    auto elmt = Queue.top();
    Queue.pop();

    BinaryBasicBlock *BBSrc = elmt.first;
    BinaryBasicBlock *BBDst = elmt.second;

    // Case 1: BBSrc and BBDst are the same. Ignore this edge
    if (BBSrc == BBDst || BBDst == *BasicBlocksLayout.begin())
      continue;

    int I = BBToClusterMap[BBSrc];
    int J = BBToClusterMap[BBDst];

    // Case 2: If they are already allocated at the same cluster, just increase
    // the weight of this cluster
    if (I == J) {
      ClusterEdges[I][I] += Weight[elmt];
      continue;
    }

    auto &ClusterA = Clusters[I];
    auto &ClusterB = Clusters[J];
    if (ClusterA.back() == BBSrc && ClusterB.front() == BBDst) {
      // Case 3: BBSrc is at the end of a cluster and BBDst is at the start,
      // allowing us to merge two clusters
      for (auto BB : ClusterB)
        BBToClusterMap[BB] = I;
      ClusterA.insert(ClusterA.end(), ClusterB.begin(), ClusterB.end());
      ClusterB.clear();
      // Iterate through all inter-cluster edges and transfer edges targeting
      // cluster B to cluster A.
      // It is bad to have to iterate though all edges when we could have a list
      // of predecessors for cluster B. However, it's not clear if it is worth
      // the added code complexity to create a data structure for clusters that
      // maintains a list of predecessors. Maybe change this if it becomes a
      // deal breaker.
      for (uint32_t K = 0, E = ClusterEdges.size(); K != E; ++K)
        ClusterEdges[K][I] += ClusterEdges[K][J];
    } else {
      // Case 4: Both BBSrc and BBDst are allocated in positions we cannot
      // merge them. Annotate the weight of this edge in the weight between
      // clusters to help us decide ordering between these clusters.
      ClusterEdges[I][J] += Weight[elmt];
    }
  }
  std::vector<uint32_t> Order;  // Cluster layout order

  // Here we have 3 conflicting goals as to how to layout clusters. If we want
  // to minimize jump offsets, we should put clusters with heavy inter-cluster
  // dependence as close as possible. If we want to maximize the probability
  // that all inter-cluster edges are predicted as not-taken, we should enforce
  // a topological order to make targets appear after sources, creating forward
  // branches. If we want to separate hot from cold blocks to maximize the
  // probability that unfrequently executed code doesn't pollute the cache, we
  // should put clusters in descending order of hotness.
  std::vector<double> AvgFreq;
  AvgFreq.resize(Clusters.size(), 0.0);
  for (uint32_t I = 0, E = Clusters.size(); I < E; ++I) {
    double Freq = 0.0;
    for (auto BB : Clusters[I]) {
      if (!BB->empty() && BB->size() != BB->getNumPseudos())
        Freq += ((double) BB->getExecutionCount()) /
                (BB->size() - BB->getNumPseudos());
    }
    AvgFreq[I] = Freq;
  }

  if (opts::PrintClusters) {
    for (uint32_t I = 0, E = Clusters.size(); I < E; ++I) {
      errs() << "Cluster number " << I << " (frequency: " << AvgFreq[I]
             << ") : ";
      auto Sep = "";
      for (auto BB : Clusters[I]) {
        errs() << Sep << BB->getName();
        Sep = ", ";
      }
      errs() << "\n";
    };
  }

  switch(Type) {
  case LT_OPTIMIZE: {
    for (uint32_t I = 0, E = Clusters.size(); I < E; ++I)
      if (!Clusters[I].empty())
        Order.push_back(I);
    break;
  }
  case LT_OPTIMIZE_BRANCH: {
    // Do a topological sort for clusters, prioritizing frequently-executed BBs
    // during the traversal.
    std::stack<uint32_t> Stack;
    std::vector<uint32_t> Status;
    std::vector<uint32_t> Parent;
    Status.resize(Clusters.size(), 0);
    Parent.resize(Clusters.size(), 0);
    constexpr uint32_t STACKED = 1;
    constexpr uint32_t VISITED = 2;
    Status[0] = STACKED;
    Stack.push(0);
    while (!Stack.empty()) {
      uint32_t I = Stack.top();
      if (!(Status[I] & VISITED)) {
        Status[I] |= VISITED;
        // Order successors by weight
        auto ClusterComp = [&ClusterEdges, I](uint32_t A, uint32_t B) {
          return ClusterEdges[I][A] > ClusterEdges[I][B];
        };
        std::priority_queue<uint32_t, std::vector<uint32_t>,
                            decltype(ClusterComp)> SuccQueue(ClusterComp);
        for (auto &Target: ClusterEdges[I]) {
          if (Target.second > 0 && !(Status[Target.first] & STACKED) &&
              !Clusters[Target.first].empty()) {
            Parent[Target.first] = I;
            Status[Target.first] = STACKED;
            SuccQueue.push(Target.first);
          }
        }
        while (!SuccQueue.empty()) {
          Stack.push(SuccQueue.top());
          SuccQueue.pop();
        }
        continue;
      }
      // Already visited this node
      Stack.pop();
      Order.push_back(I);
    }
    std::reverse(Order.begin(), Order.end());
    // Put unreachable clusters at the end
    for (uint32_t I = 0, E = Clusters.size(); I < E; ++I)
      if (!(Status[I] & VISITED) && !Clusters[I].empty())
        Order.push_back(I);

    // Sort nodes with equal precedence
    auto Beg = Order.begin();
    // Don't reorder the first cluster, which contains the function entry point
    ++Beg;
    std::stable_sort(Beg, Order.end(),
                     [&AvgFreq, &Parent](uint32_t A, uint32_t B) {
                       uint32_t P = Parent[A];
                       while (Parent[P] != 0) {
                         if (Parent[P] == B)
                           return false;
                         P = Parent[P];
                       }
                       P = Parent[B];
                       while (Parent[P] != 0) {
                         if (Parent[P] == A)
                           return true;
                         P = Parent[P];
                       }
                       return AvgFreq[A] > AvgFreq[B];
                     });
    break;
  }
  case LT_OPTIMIZE_CACHE: {
    // Order clusters based on average instruction execution frequency
    for (uint32_t I = 0, E = Clusters.size(); I < E; ++I)
      if (!Clusters[I].empty())
        Order.push_back(I);
    auto Beg = Order.begin();
    // Don't reorder the first cluster, which contains the function entry point
    ++Beg;
    std::stable_sort(Beg, Order.end(), [&AvgFreq](uint32_t A, uint32_t B) {
      return AvgFreq[A] > AvgFreq[B];
    });

    break;
  }
  default:
    llvm_unreachable("unexpected layout type");
  }

  if (opts::PrintClusters) {
    errs() << "New cluster order: ";
    auto Sep = "";
    for (auto O : Order) {
      errs() << Sep << O;
      Sep = ", ";
    }
    errs() << '\n';
  }

  BasicBlocksLayout.clear();
  for (auto I : Order) {
    auto &Cluster = Clusters[I];
    BasicBlocksLayout.insert(BasicBlocksLayout.end(), Cluster.begin(),
                             Cluster.end());
  }

  if (Split)
    splitFunction();
  fixBranches();
}

void BinaryFunction::solveOptimalLayout(bool Split) {
  std::vector<std::vector<uint64_t>> Weight;
  std::map<BinaryBasicBlock *, int> BBToIndex;
  std::vector<BinaryBasicBlock *> IndexToBB;

  DEBUG(dbgs() << "finding optimal block layout for " << getName() << "\n");

  unsigned N = BasicBlocksLayout.size();
  // Populating weight map and index map
  for (auto BB : BasicBlocksLayout) {
    BBToIndex[BB] = IndexToBB.size();
    IndexToBB.push_back(BB);
  }
  Weight.resize(N);
  for (auto BB : BasicBlocksLayout) {
    auto BI = BB->BranchInfo.begin();
    Weight[BBToIndex[BB]].resize(N);
    for (auto I : BB->successors()) {
      if (BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        Weight[BBToIndex[BB]][BBToIndex[I]] = BI->Count;
      ++BI;
    }
  }

  std::vector<std::vector<int64_t>> DP;
  DP.resize(1 << N);
  for (auto &Elmt : DP) {
    Elmt.resize(N, -1);
  }
  // Start with the entry basic block being allocated with cost zero
  DP[1][0] = 0;
  // Walk through TSP solutions using a bitmask to represent state (current set
  // of BBs in the layout)
  unsigned BestSet = 1;
  unsigned BestLast = 0;
  int64_t BestWeight = 0;
  for (unsigned Set = 1; Set < (1U << N); ++Set) {
    // Traverse each possibility of Last BB visited in this layout
    for (unsigned Last = 0; Last < N; ++Last) {
      // Case 1: There is no possible layout with this BB as Last
      if (DP[Set][Last] == -1)
        continue;

      // Case 2: There is a layout with this Set and this Last, and we try
      // to expand this set with New
      for (unsigned New = 1; New < N; ++New) {
        // Case 2a: BB "New" is already in this Set
        if ((Set & (1 << New)) != 0)
          continue;

        // Case 2b: BB "New" is not in this set and we add it to this Set and
        // record total weight of this layout with "New" as the last BB.
        unsigned NewSet = (Set | (1 << New));
        if (DP[NewSet][New] == -1)
          DP[NewSet][New] = DP[Set][Last] + (int64_t)Weight[Last][New];
        DP[NewSet][New] = std::max(DP[NewSet][New],
                                   DP[Set][Last] + (int64_t)Weight[Last][New]);

        if (DP[NewSet][New] > BestWeight) {
          BestWeight = DP[NewSet][New];
          BestSet = NewSet;
          BestLast = New;
        }
      }
    }
  }

  std::vector<BinaryBasicBlock *> PastLayout = BasicBlocksLayout;

  // Define final function layout based on layout that maximizes weight
  BasicBlocksLayout.clear();
  unsigned Last = BestLast;
  unsigned Set = BestSet;
  std::vector<bool> Visited;
  Visited.resize(N);
  Visited[Last] = true;
  BasicBlocksLayout.push_back(IndexToBB[Last]);
  Set = Set & ~(1U << Last);
  while (Set != 0) {
    int64_t Best = -1;
    for (unsigned I = 0; I < N; ++I) {
      if (DP[Set][I] == -1)
        continue;
      if (DP[Set][I] > Best) {
        Last = I;
        Best = DP[Set][I];
      }
    }
    Visited[Last] = true;
    BasicBlocksLayout.push_back(IndexToBB[Last]);
    Set = Set & ~(1U << Last);
  }
  std::reverse(BasicBlocksLayout.begin(), BasicBlocksLayout.end());

  // Finalize layout with BBs that weren't assigned to the layout
  for (auto BB : PastLayout) {
    if (Visited[BBToIndex[BB]] == false)
      BasicBlocksLayout.push_back(BB);
  }

  if (Split)
    splitFunction();
  fixBranches();
}

const BinaryBasicBlock *
BinaryFunction::getOriginalLayoutSuccessor(const BinaryBasicBlock *BB) const {
  auto I = std::upper_bound(BasicBlocks.begin(), BasicBlocks.end(), *BB);
  assert(I != BasicBlocks.begin() && "first basic block not at offset 0");

  if (I == BasicBlocks.end())
    return nullptr;
  return &*I;
}

void BinaryFunction::fixBranches() {
  auto &MIA = BC.MIA;

  for (unsigned I = 0, E = BasicBlocksLayout.size(); I != E; ++I) {
    BinaryBasicBlock *BB = BasicBlocksLayout[I];

    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    if (!MIA->analyzeBranch(BB->Instructions, TBB, FBB, CondBranch,
                            UncondBranch)) {
      continue;
    }

    // Check if the original fall-through for this block has been moved
    const MCSymbol *FT = nullptr;
    bool HotColdBorder = false;
    if (I + 1 != BasicBlocksLayout.size()) {
      FT = BasicBlocksLayout[I + 1]->getLabel();
      if (BB->IsCold != BasicBlocksLayout[I + 1]->IsCold)
        HotColdBorder = true;
    }
    const BinaryBasicBlock *OldFTBB = getOriginalLayoutSuccessor(BB);
    const MCSymbol *OldFT = nullptr;
    if (OldFTBB != nullptr)
      OldFT = OldFTBB->getLabel();

    // Case 1: There are no branches in this basic block and it just falls
    // through
    if (CondBranch == nullptr && UncondBranch == nullptr) {
      // Case 1a: Last instruction, excluding pseudos, is a return, so it does
      // *not* fall through to the next block.
      if (!BB->empty()) {
        auto LastInstIter = --BB->end();
        while (BC.MII->get(LastInstIter->getOpcode()).isPseudo() &&
               LastInstIter != BB->begin())
          --LastInstIter;
        if (MIA->isReturn(*LastInstIter))
          continue;
      }
      // Case 1b: Layout has changed and the fallthrough is not the same (or the
      // fallthrough got moved to a cold region). Need to add a new
      // unconditional branch to jump to the old fallthrough.
      if ((FT != OldFT || HotColdBorder) && OldFT != nullptr) {
        MCInst NewInst;
        if (!MIA->createUncondBranch(NewInst, OldFT, BC.Ctx.get()))
          llvm_unreachable("Target does not support creating new branches");
        BB->Instructions.emplace_back(std::move(NewInst));
      }
      // Case 1c: Layout hasn't changed, nothing to do.
      continue;
    }

    // Case 2: There is a single jump, unconditional, in this basic block
    if (CondBranch == nullptr) {
      // Case 2a: It jumps to the new fall-through, so we can delete it
      if (TBB == FT && !HotColdBorder) {
        BB->eraseInstruction(UncondBranch);
      }
      // Case 2b: If 2a doesn't happen, there is nothing we can do
      continue;
    }

    // Case 3: There is a single jump, conditional, in this basic block
    if (UncondBranch == nullptr) {
      // Case 3a: If the taken branch goes to the next block in the new layout,
      // invert this conditional branch logic so we can make this a fallthrough.
      if (TBB == FT && !HotColdBorder) {
        if (OldFT == nullptr) {
          errs() << "BOLT-ERROR: malfromed CFG for function " << getName()
                 << " in basic block " << BB->getName() << '\n';
        }
        assert(OldFT != nullptr && "malformed CFG");
        if (!MIA->reverseBranchCondition(*CondBranch, OldFT, BC.Ctx.get()))
          llvm_unreachable("Target does not support reversing branches");
        continue;
      }
      // Case 3b: Need to add a new unconditional branch because layout
      // has changed
      if ((FT != OldFT || HotColdBorder) && OldFT != nullptr) {
        MCInst NewInst;
        if (!MIA->createUncondBranch(NewInst, OldFT, BC.Ctx.get()))
          llvm_unreachable("Target does not support creating new branches");
        BB->Instructions.emplace_back(std::move(NewInst));
        continue;
      }
      // Case 3c: Old fall-through is the same as the new one, no need to change
      continue;
    }

    // Case 4: There are two jumps in this basic block, one conditional followed
    // by another unconditional.
    // Case 4a: If the unconditional jump target is the new fall through,
    // delete it.
    if (FBB == FT && !HotColdBorder) {
      BB->eraseInstruction(UncondBranch);
      continue;
    }
    // Case 4b: If the taken branch goes to the next block in the new layout,
    // invert this conditional branch logic so we can make this a fallthrough.
    // Now we don't need the unconditional jump anymore, so we also delete it.
    if (TBB == FT && !HotColdBorder) {
      if (!MIA->reverseBranchCondition(*CondBranch, FBB, BC.Ctx.get()))
        llvm_unreachable("Target does not support reversing branches");
      BB->eraseInstruction(UncondBranch);
      continue;
    }
    // Case 4c: Nothing interesting happening.
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
  BasicBlocks.front().CanOutline = false;
  for (auto &BB : BasicBlocks) {
    if (!BB.CanOutline)
      continue;
    if (BB.getExecutionCount() != 0) {
      BB.CanOutline = false;
      continue;
    }
    if (hasEHRanges()) {
      // We cannot move landing pads (or rather entry points for landing
      // pads).
      if (LandingPads.find(BB.getLabel()) != LandingPads.end()) {
        BB.CanOutline = false;
        continue;
      }
      // We cannot move a block that can throw since exception-handling
      // runtime cannot deal with split functions. However, if we can guarantee
      // that the block never throws, it is safe to move the block to
      // decrease the size of the function.
      for (auto &Instr : BB) {
        if (BC.MIA->isInvoke(Instr)) {
          BB.CanOutline = false;
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
    while (LandingPads.find((*FirstLP)->getLabel()) != LandingPads.end())
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
    BB->IsCold = true;
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
  for (auto &BB : BasicBlocks) {
    for (auto II = BB.begin(); II != BB.end(); ) {
      auto &Instr = *II;
      if (BC.MIA->isCFI(Instr)) {
        auto CFI = getCFIFor(Instr);
        if (CFI->getOperation() == MCCFIInstruction::OpGnuArgsSize) {
          CurrentGnuArgsSize = CFI->getOffset();
          // Delete DW_CFA_GNU_args_size instructions and only regenerate
          // during the final code emission. The information is embedded
          // inside call instructions.
          II = BB.Instructions.erase(II);
        } else {
          ++II;
        }
        continue;
      }

      if (BC.MIA->isInvoke(Instr)) {
        // Add the value of GNU_args_size as an extra operand if landing pad
        // is non-emptry.
        if (BC.MIA->getEHInfo(Instr).first) {
          Instr.addOperand(MCOperand::createImm(CurrentGnuArgsSize));
        }
      }
      ++II;
    }
  }
}

} // namespace bolt
} // namespace llvm
