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


namespace llvm {
namespace bolt {

namespace opts {

static cl::opt<bool>
AgressiveSplitting("split-all-cold",
                   cl::desc("outline as many cold basic blocks as possible"),
                   cl::Optional);

} // namespace opts

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
  OS << "Binary Function \"" << getName() << "\" " << Annotation << " {";
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
    if (!BB->Throwers.empty()) {
      OS << "  Throwers: ";
      auto Sep = "";
      for (auto Throw : BB->Throwers) {
        OS << Sep << Throw->getName();
        Sep = ", ";
      }
      OS << '\n';
    }

    Offset = RoundUpToAlignment(Offset, BB->getAlignment());

    // Note: offsets are imprecise since this is happening prior to relaxation.
    Offset = BC.printInstructions(OS, BB->begin(), BB->end(), Offset, this);

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

    if (!BB->LandingPads.empty()) {
      OS << "  Landing Pads: ";
      auto Sep = "";
      for (auto LP : BB->LandingPads) {
        OS << Sep << LP->getName();
        if (ExecutionCount != COUNT_NO_PROFILE) {
          OS << " (count: " << LP->ExecutionCount << ")";
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

  OS << "End of Function \"" << getName() << "\"\n\n";
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

  auto handleRIPOperand =
      [&](MCInst &Instruction, uint64_t Address, uint64_t Size) -> bool {
        uint64_t TargetAddress{0};
        MCSymbol *TargetSymbol{nullptr};
        if (!BC.MIA->evaluateRIPOperand(Instruction, Address, Size,
                                        TargetAddress)) {
          DEBUG(dbgs() << "BOLT: rip-relative operand can't be evaluated:\n";
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

    // Convert instruction to a shorter version that could be relaxed if needed.
    MIA->shortenInstruction(Instruction);

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
          // Add taken branch info.
          TakenBranches.push_back({Offset, TargetOffset});
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

    if (ULT.first && ULT.second) {
      Instruction.setLoc(
          findDebugLineInformationForInstructionAt(AbsoluteInstrAddr, ULT));
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

bool BinaryFunction::buildCFG() {

  auto &MIA = BC.MIA;

  auto BranchDataOrErr = BC.DR.getFuncBranchData(getNames());
  if (!BranchDataOrErr) {
    DEBUG(dbgs() << "no branch data found for \"" << getName() << "\"\n");
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
  for (auto BB : BasicBlocks) {
    BasicBlocksLayout.emplace_back(BB);
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
    if (BB->succ_size() == 0) {
      IsPrevFT = MIA->isTerminator(*LastInstIter) ? false : true;
    } else if (BB->succ_size() == 1) {
      IsPrevFT =  MIA->isConditionalBranch(*LastInstIter) ? true : false;
    } else {
      // Ends with 2 branches, with an indirect jump or it is a conditional
      // branch whose frequency has been inferred from LBR
      IsPrevFT = false;
    }

    PrevBB = BB;
  }

  if (!IsPrevFT) {
    // Possibly a call that does not return.
    DEBUG(dbgs() << "last block was marked as a fall-through\n");
  }

  // Add associated landing pad blocks to each basic block.
  for (auto BB : BasicBlocks) {
    if (LandingPads.find(BB->getLabel()) != LandingPads.end()) {
      MCSymbol *LP = BB->getLabel();
      for (unsigned I : LPToBBIndex.at(LP)) {
        BinaryBasicBlock *ThrowBB = getBasicBlockAtIndex(I);
        ThrowBB->addLandingPad(BB);
      }
    }
  }

  // Infer frequency for non-taken branches
  if (hasValidProfile())
    inferFallThroughCounts();

  // Update CFI information for each BB
  annotateCFIState();

  // Clean-up memory taken by instructions and labels.
  clearList(Instructions);
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

  if (!OrphanBranches.empty()) {
    errs() << "BOLT-WARNING: profile branches match only "
           << format("%.1f%%", ProfileMatchRatio * 100.0f) << " ("
           << (LocalProfileBranches.size() - OrphanBranches.size()) << '/'
           << LocalProfileBranches.size() << ") for function "
           << getName() << '\n';
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
      CurBB->ExecutionCount = ExecutionCount;
      continue;
    }
    CurBB->ExecutionCount = 0;
  }

  for (auto CurBB : BasicBlocks) {
    auto SuccCount = CurBB->BranchInfo.begin();
    for (auto Succ : CurBB->successors()) {
      // Do not update execution count of the entry block (when we have tail
      // calls). We already accounted for those when computing the func count.
      if (Succ == *BasicBlocks.begin()) {
        ++SuccCount;
        continue;
      }
      if (SuccCount->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        Succ->ExecutionCount += SuccCount->Count;
      ++SuccCount;
    }
  }

  // Update execution counts of landing pad blocks.
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
    for (auto &SuccCount : CurBB->BranchInfo) {
      if (SuccCount.Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        ReportedBranches += SuccCount.Count;
    }

    // Calculate frequency of throws from this node according to LBR data
    // for branching into associated landing pads. Since it is possible
    // for a landing pad to be associated with more than one basic blocks,
    // we may overestimate the frequency of throws for such blocks.
    uint64_t ReportedThrows = 0;
    for (BinaryBasicBlock *LP: CurBB->LandingPads) {
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
            << Twine::utohexstr(getAddress() + CurBB->getOffset()) << '\n';
    });

    // If there is a FT, the last successor will be it.
    auto &SuccCount = CurBB->BranchInfo.back();
    auto &Succ = CurBB->Successors.back();
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
    BinaryBasicBlock *CurBB = *CI;
    // Annotate this BB entry
    BBCFIState.emplace_back(State);

    // Advance state
    for (const auto &Instr : *CurBB) {
      auto *CFI = getCFIFor(Instr);
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
          errs() << "BOLT-WARNING: CFI rewriter detected nested CFI state while"
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

  BasicBlockOrderType NewLayout;
  std::unique_ptr<ReorderAlgorithm> Algo;

  // Cannot do optimal layout without profile.
  if (Type != LT_REVERSE && !hasValidProfile())
    return;

  if (Type == LT_REVERSE) {
    Algo.reset(new ReverseReorderAlgorithm());
  }
  else if (BasicBlocksLayout.size() <= FUNC_SIZE_THRESHOLD) {
    // Work on optimal solution if problem is small enough
    DEBUG(dbgs() << "finding optimal block layout for " << getName() << "\n");
    Algo.reset(new OptimalReorderAlgorithm());
  }
  else {
    DEBUG(dbgs() << "running block layout heuristics on " << getName() << "\n");

    std::unique_ptr<ClusterAlgorithm> CAlgo(new GreedyClusterAlgorithm());

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

    default:
      llvm_unreachable("unexpected layout type");
    }
  }

  Algo->reorderBasicBlocks(*this, NewLayout);
  BasicBlocksLayout.clear();
  BasicBlocksLayout.swap(NewLayout);

  if (Split)
    splitFunction();
  fixBranches();
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
    dbgs() << "BOLT-WARNING: Filename \"" << Filename << Annotation << Suffix
           << "\" exceeds the " << MAX_PATH << " size limit, truncating.\n";
    Filename.resize(MAX_PATH - (Suffix.size() + Annotation.size()));
  }
  Filename += Annotation;
  Filename += Suffix;
  return Filename;
}

}

void BinaryFunction::dumpGraph(raw_ostream& OS) const {
  OS << "strict digraph \"" << getName() << "\" {\n";
  for (auto *BB : BasicBlocks) {
    for (auto *Succ : BB->successors()) {
      OS << "\"" << BB->getName() << "\" -> "
         << "\"" << Succ->getName() << "\"\n";
    }
  }
  OS << "}\n";
}

void BinaryFunction::viewGraph() const {
  SmallString<MAX_PATH> Filename;
  if (auto EC = sys::fs::createTemporaryFile("bolt-cfg", "dot", Filename)) {
    dbgs() << "BOLT-WARNING: " << EC.message() << ", unable to create "
           << " bolt-cfg-XXXXX.dot temporary file.\n";
    return;
  }
  dumpGraphToFile(Filename.str());
  if (DisplayGraph(Filename)) {
    dbgs() << "BOLT-WARNING: Can't display " << Filename
           << " with graphviz.\n";
  }
  if (auto EC = sys::fs::remove(Filename)) {
    dbgs() << "BOLT-WARNING: " << EC.message() << ", failed to remove "
           << Filename.str() << "\n";
  }
}

void BinaryFunction::dumpGraphForPass(std::string Annotation) const {
  auto Filename = constructFilename(getName(), Annotation, ".dot");
  dbgs() << "BOLT-DEBUG: Dumping CFG to " << Filename << "\n";
  dumpGraphToFile(Filename);
}

void BinaryFunction::dumpGraphToFile(std::string Filename) const {
  std::error_code EC;
  raw_fd_ostream of(Filename, EC, sys::fs::F_None);
  if (EC) {
    dbgs() << "BOLT-WARNING: " << EC.message() << ", unable to open "
           << Filename << " for output.\n";
    return;
  }
  dumpGraph(of);
}

const BinaryBasicBlock *
BinaryFunction::getOriginalLayoutSuccessor(const BinaryBasicBlock *BB) const {
  auto I = std::upper_bound(begin(), end(), *BB);
  assert(I != begin() && "first basic block not at offset 0");

  if (I == end())
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
          errs() << "BOLT-ERROR: malformed CFG for function " << getName()
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
  BasicBlocks.front()->CanOutline = false;
  for (auto BB : BasicBlocks) {
    if (!BB->CanOutline)
      continue;
    if (BB->getExecutionCount() != 0) {
      BB->CanOutline = false;
      continue;
    }
    if (hasEHRanges()) {
      // We cannot move landing pads (or rather entry points for landing
      // pads).
      if (LandingPads.find(BB->getLabel()) != LandingPads.end()) {
        BB->CanOutline = false;
        continue;
      }
      // We cannot move a block that can throw since exception-handling
      // runtime cannot deal with split functions. However, if we can guarantee
      // that the block never throws, it is safe to move the block to
      // decrease the size of the function.
      for (auto &Instr : *BB) {
        if (BC.MIA->isInvoke(Instr)) {
          BB->CanOutline = false;
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
          II = BB->Instructions.erase(II);
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
      BBMerge->ExecutionCount = NewExecCount;
    }

    // Update BF's edge count for successors of this basic block.
    auto BBMergeSI = BBMerge->succ_begin();
    auto BII = BB->BranchInfo.begin();
    auto BIMergeI = BBMerge->BranchInfo.begin();
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

  if (PseudosDiffer) {
    errs() << "BOLT-WARNING: functions " << getName() << " and ";
    errs() << BF.getName() << " are identical, but have different";
    errs() << " pseudo instruction sequences.\n";
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

BinaryFunction::~BinaryFunction() {
  for (auto BB : BasicBlocks) {
    delete BB;
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
      auto BI = Latch->BranchInfo.begin();
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
      auto BI = Exiting->BranchInfo.begin();
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
  OS << "Loop Info for Function \"" << getName() << "\"";
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

} // namespace bolt
} // namespace llvm
