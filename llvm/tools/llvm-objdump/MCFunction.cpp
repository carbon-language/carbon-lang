//===-- MCFunction.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the algorithm to break down a region of machine code
// into basic blocks and try to reconstruct a CFG from it.
//
//===----------------------------------------------------------------------===//

#include "MCFunction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <set>
using namespace llvm;

MCFunction
MCFunction::createFunctionFromMC(StringRef Name, const MCDisassembler *DisAsm,
                                 const MemoryObject &Region, uint64_t Start,
                                 uint64_t End, const MCInstrInfo *InstrInfo,
                                 raw_ostream &DebugOut) {
  std::set<uint64_t> Splits;
  Splits.insert(Start);
  std::vector<MCDecodedInst> Instructions;
  uint64_t Size;

  // Disassemble code and gather basic block split points.
  for (uint64_t Index = Start; Index < End; Index += Size) {
    MCInst Inst;

    if (DisAsm->getInstruction(Inst, Size, Region, Index, DebugOut)) {
      const MCInstrDesc &Desc = InstrInfo->get(Inst.getOpcode());
      if (Desc.isBranch()) {
        if (Desc.OpInfo[0].OperandType == MCOI::OPERAND_PCREL) {
          int64_t Imm = Inst.getOperand(0).getImm();
          // FIXME: Distinguish relocations from nop jumps.
          if (Imm != 0) {
            if (Index+Imm+Size >= End) {
              Instructions.push_back(MCDecodedInst(Index, Size, Inst));
              continue; // Skip branches that leave the function.
            }
            Splits.insert(Index+Imm+Size);
          }
        }
        Splits.insert(Index+Size);
      } else if (Desc.isReturn()) {
        Splits.insert(Index+Size);
      }

      Instructions.push_back(MCDecodedInst(Index, Size, Inst));
    } else {
      errs() << "warning: invalid instruction encoding\n";
      if (Size == 0)
        Size = 1; // skip illegible bytes
    }

  }

  MCFunction f(Name);

  // Create basic blocks.
  unsigned ii = 0, ie = Instructions.size();
  for (std::set<uint64_t>::iterator spi = Splits.begin(),
       spe = Splits.end(); spi != spe; ++spi) {
    MCBasicBlock BB;
    uint64_t BlockEnd = llvm::next(spi) == spe ? End : *llvm::next(spi);
    // Add instructions to the BB.
    for (; ii != ie; ++ii) {
      if (Instructions[ii].Address < *spi ||
          Instructions[ii].Address >= BlockEnd)
        break;
      BB.addInst(Instructions[ii]);
    }
    f.addBlock(*spi, BB);
  }

  // Calculate successors of each block.
  for (MCFunction::iterator i = f.begin(), e = f.end(); i != e; ++i) {
    MCBasicBlock &BB = i->second;
    if (BB.getInsts().empty()) continue;
    const MCDecodedInst &Inst = BB.getInsts().back();
    const MCInstrDesc &Desc = InstrInfo->get(Inst.Inst.getOpcode());

    if (Desc.isBranch()) {
      // PCRel branch, we know the destination.
      if (Desc.OpInfo[0].OperandType == MCOI::OPERAND_PCREL) {
        int64_t Imm = Inst.Inst.getOperand(0).getImm();
        if (Imm != 0)
          BB.addSucc(&f.getBlockAtAddress(Inst.Address+Inst.Size+Imm));
        // Conditional branches can also fall through to the next block.
        if (Desc.isConditionalBranch() && llvm::next(i) != e)
          BB.addSucc(&llvm::next(i)->second);
      } else {
        // Indirect branch. Bail and add all blocks of the function as a
        // successor.
        for (MCFunction::iterator i = f.begin(), e = f.end(); i != e; ++i)
          BB.addSucc(&i->second);
      }
    } else {
      // No branch. Fall through to the next block.
      if (!Desc.isReturn() && llvm::next(i) != e)
        BB.addSucc(&llvm::next(i)->second);
    }
  }

  return f;
}
