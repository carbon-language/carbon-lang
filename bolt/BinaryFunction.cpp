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


#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <string>

#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "flo"

namespace llvm {

namespace flo {

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

void BinaryFunction::print(raw_ostream &OS, bool PrintInstructions) const {
  StringRef SectionName;
  Section.getName(SectionName);
  OS << "Binary Function \"" << getName() << "\" {"
     << "\n  State       : "   << CurrentState
     << "\n  Address     : 0x" << Twine::utohexstr(Address)
     << "\n  Size        : 0x" << Twine::utohexstr(Size)
     << "\n  MaxSize     : 0x" << Twine::utohexstr(MaxSize)
     << "\n  Offset      : 0x" << Twine::utohexstr(FileOffset)
     << "\n  Section     : "   << SectionName
     << "\n  Orc Section : "   << getCodeSectionName()
     << "\n  IsSimple    : "   << IsSimple
     << "\n  BB count    : "   << BasicBlocks.size()
     << "\n  Image       : 0x" << Twine::utohexstr(ImageAddress)
     << "\n}\n";

  if (!PrintInstructions || !BC.InstPrinter)
    return;

  // Offset of the instruction in function.
  uint64_t Offset{0};

  if (BasicBlocks.empty() && !Instructions.empty()) {
    // Print before CFG was built.
    for (const auto &II : Instructions) {
      auto Offset = II.first;

      // Print label if exists at this offset.
      auto LI = Labels.find(Offset);
      if (LI != Labels.end())
        OS << LI->second->getName() << ":\n";

      auto &Instruction = II.second;
      OS << format("    %08" PRIx64 ": ", Offset);
      BC.InstPrinter->printInst(&Instruction, OS, "", *BC.STI);
      OS << "\n";
    }
  }

  for (const auto &BB : BasicBlocks) {
    OS << BB.getName() << " ("
       << BB.Instructions.size() << " instructions)\n";

    if (!BB.Predecessors.empty()) {
      OS << "  Predecessors: ";
      auto Sep = "";
      for (auto Pred : BB.Predecessors) {
        OS << Sep << Pred->getName();
        Sep = ", ";
      }
      OS << '\n';
    }

    Offset = RoundUpToAlignment(Offset, BB.getAlignment());

    for (auto &Instr : BB) {
      OS << format("    %08" PRIx64 ": ", Offset);
      BC.InstPrinter->printInst(&Instr, OS, "", *BC.STI);
      OS << "\n";

      // In case we need MCInst printer:
      // Instr.dump_pretty(OS, InstructionPrinter.get());

      // Calculate the size of the instruction.
      // Note: this is imprecise since happening prior to relaxation.
      SmallString<256> Code;
      SmallVector<MCFixup, 4> Fixups;
      raw_svector_ostream VecOS(Code);
      BC.MCE->encodeInstruction(Instr, VecOS, Fixups, *BC.STI);
      Offset += Code.size();
    }

    if (!BB.Successors.empty()) {
      OS << "  Successors: ";
      auto Sep = "";
      for (auto Succ : BB.Successors) {
        OS << Sep << Succ->getName();
        Sep = ", ";
      }
      OS << '\n';
    }

    OS << '\n';
  }

  OS << "End of Function \"" << getName() << "\"\n";
}

bool BinaryFunction::disassemble(ArrayRef<uint8_t> FunctionData) {
  assert(FunctionData.size() == getSize() &&
         "function size does not match raw data size");

  auto &Ctx = BC.Ctx;
  auto &MIA = BC.MIA;

  // Insert a label at the beginning of the function. This will be our first
  // basic block.
  Labels[0] = Ctx->createTempSymbol("BB0", false);

  bool IsSimple = true;
  for (uint64_t Offset = 0; IsSimple && (Offset < getSize()); ) {
    MCInst Instruction;
    uint64_t Size;
    if (!BC.DisAsm->getInstruction(Instruction,
                                   Size,
                                   FunctionData.slice(Offset),
                                   getAddress() + Offset,
                                   nulls(),
                                   nulls())) {
      // Ignore this function. Skip to the next one.
      IsSimple = false;
      break;
    }

    if (MIA->isIndirectBranch(Instruction)) {
      IsSimple = false;
      break;
    }

    if (MIA->isBranch(Instruction) || MIA->isCall(Instruction)) {
      uint64_t InstructionTarget = 0;
      uint64_t AbsoluteInstrAddr = getAddress() + Offset;
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
        MCSymbol *TargetSymbol{nullptr};
        uint64_t TargetOffset{0};

        if (IsCall && containsAddress(InstructionTarget)) {
          if (InstructionTarget == getAddress()) {
            // Recursive call.
            TargetSymbol = Ctx->getOrCreateSymbol(getName());
          } else {
            // Possibly an old-style PIC code
            DEBUG(dbgs() << "FLO: internal call detected at 0x"
                         << Twine::utohexstr(AbsoluteInstrAddr)
                         << " in function " << getName() << "\n");
            IsSimple = false;
            break;
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
            // This is a call regardless of the opcode (e.g. tail call).
            IsCall = true;
            // Check if we already have a symbol at this address.
            std::string Name;
            auto NI = BC.GlobalAddresses.find(InstructionTarget);
            if (NI != BC.GlobalAddresses.end()) {
              // Any registered name will do.
              Name = NI->second;
            } else {
              // Create a new symbol at the destination.
              Name = (Twine("FUNCat0x") +
                      Twine::utohexstr(InstructionTarget)).str();
              BC.GlobalAddresses.emplace(std::make_pair(InstructionTarget,
                                                        Name));
            }
            TargetSymbol =  Ctx->getOrCreateSymbol(Name);
            BC.GlobalSymbols[Name] = InstructionTarget;
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

      } else {
        // Indirect call
        IsSimple = false;
        break;
      }
    } else {
      if (MIA->hasRIPOperand(Instruction)) {
        DEBUG(dbgs() << "FLO: rip-relative instruction found "
                        "(not supported yet)\n");
        IsSimple = false;
        break;
      }
    }

    addInstruction(Offset, std::move(Instruction));

    Offset += Size;
  }

  setSimple(IsSimple);

  // TODO: clear memory if not simple function?

  // Update state.
  updateState(State::Disassembled);

  // Print the function in the new state.
  DEBUG(print(dbgs(), /* PrintInstructions = */ true));

  return true;
}

bool BinaryFunction::buildCFG() {

  auto &MIA = BC.MIA;

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
  for (auto &InstrInfo : Instructions) {
    auto LI = Labels.find(InstrInfo.first);
    if (LI != Labels.end()) {
      // Always create new BB at branch destination.
      PrevBB = InsertBB;
      InsertBB = addBasicBlock(LI->first, LI->second);
    }
    if (!InsertBB) {
      // It must be a fallthrough. Create a new block unless we see an
      // unconditional branch.
      assert(PrevBB && "no previous basic block for a fall through");
      if (MIA->isUnconditionalBranch(InstrInfo.second)) {
        // Temporarily restore inserter basic block.
        InsertBB = PrevBB;
      } else {
        InsertBB = addBasicBlock(InstrInfo.first,
                                 BC.Ctx->createTempSymbol("FT", true));
      }
    }

    InsertBB->addInstruction(InstrInfo.second);

    // How well do we detect tail calls here?
    if (MIA->isTerminator(InstrInfo.second)) {
      PrevBB = InsertBB;
      InsertBB = nullptr;
    }
  }

  // Intermediate dump.
  DEBUG(print(dbgs(), /* PrintInstructions = */ true));

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

    // TODO: add weights here.
    //
    FromBB->addSuccessor(ToBB);
  }

  // Add fall-through branches.
  PrevBB = nullptr;
  bool IsPrevFT = false; // Is previous block a fall-through.
  for (auto &BB : BasicBlocks) {
    if (IsPrevFT) {
      PrevBB->addSuccessor(&BB);
    }

    MCInst &LastInst = BB.back();
    if (BB.succ_size() == 0) {
      IsPrevFT = MIA->isTerminator(LastInst) ? false : true;
    } else if (BB.succ_size() == 1) {
      IsPrevFT =  MIA->isConditionalBranch(LastInst) ? true : false;
    } else {
      // Either ends with 2 branches, or with an indirect jump.
      IsPrevFT = false;
    }

    PrevBB = &BB;
  }

  if (!IsPrevFT) {
    // Possibly a call that does not return.
    DEBUG(dbgs() << "last block was marked as a fall-through\n");
  }

  // Clean-up memory taken by instructions and labels.
  clearInstructions();
  clearLabels();
  clearLocalBranches();

  // Update the state.
  CurrentState = State::CFG;

  // Print the function in the new state.
  DEBUG(print(dbgs(), /* PrintInstructions = */ true));

  return true;
}

} // namespace flo

} // namespace llvm
