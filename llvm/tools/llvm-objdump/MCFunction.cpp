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
#include "llvm/MC/MCInstrAnalysis.h"
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
                                 uint64_t End, const MCInstrAnalysis *Ana,
                                 raw_ostream &DebugOut,
                                 SmallVectorImpl<uint64_t> &Calls) {
  std::vector<MCDecodedInst> Instructions;
  std::set<uint64_t> Splits;
  Splits.insert(Start);
  uint64_t Size;

  MCFunction f(Name);

  {
  DenseSet<uint64_t> VisitedInsts;
  SmallVector<uint64_t, 16> WorkList;
  WorkList.push_back(Start);
  // Disassemble code and gather basic block split points.
  while (!WorkList.empty()) {
    uint64_t Index = WorkList.pop_back_val();
    if (VisitedInsts.find(Index) != VisitedInsts.end())
      continue;

    for (;Index < End; Index += Size) {
      MCInst Inst;

      if (DisAsm->getInstruction(Inst, Size, Region, Index, DebugOut, nulls())){
        if (Ana->isBranch(Inst)) {
          uint64_t targ = Ana->evaluateBranch(Inst, Index, Size);
          if (targ != -1ULL && targ == Index+Size) {
            Instructions.push_back(MCDecodedInst(Index, Size, Inst));
            VisitedInsts.insert(Index);
            continue;
          }
          if (targ != -1ULL) {
            Splits.insert(targ);
            WorkList.push_back(targ);
            WorkList.push_back(Index+Size);
          }
          Splits.insert(Index+Size);
          Instructions.push_back(MCDecodedInst(Index, Size, Inst));
          VisitedInsts.insert(Index);
          break;
        } else if (Ana->isReturn(Inst)) {
          Splits.insert(Index+Size);
          Instructions.push_back(MCDecodedInst(Index, Size, Inst));
          VisitedInsts.insert(Index);
          break;
        } else if (Ana->isCall(Inst)) {
          uint64_t targ = Ana->evaluateBranch(Inst, Index, Size);
          if (targ != -1ULL && targ != Index+Size) {
            Calls.push_back(targ);
          }
        }

        Instructions.push_back(MCDecodedInst(Index, Size, Inst));
        VisitedInsts.insert(Index);
      } else {
        VisitedInsts.insert(Index);
        errs().write_hex(Index) << ": warning: invalid instruction encoding\n";
        if (Size == 0)
          Size = 1; // skip illegible bytes
      }
    }
  }
  }

  std::sort(Instructions.begin(), Instructions.end());

   // Create basic blocks.
  unsigned ii = 0, ie = Instructions.size();
  for (std::set<uint64_t>::iterator spi = Splits.begin(),
       spe = llvm::prior(Splits.end()); spi != spe; ++spi) {
    MCBasicBlock BB;
    uint64_t BlockEnd = *llvm::next(spi);
    // Add instructions to the BB.
    for (; ii != ie; ++ii) {
      if (Instructions[ii].Address < *spi ||
          Instructions[ii].Address >= BlockEnd)
        break;
      BB.addInst(Instructions[ii]);
    }
    f.addBlock(*spi, BB);
  }

  std::sort(f.Blocks.begin(), f.Blocks.end());

  // Calculate successors of each block.
  for (MCFunction::iterator i = f.begin(), e = f.end(); i != e; ++i) {
    MCBasicBlock &BB = i->second;
    if (BB.getInsts().empty()) continue;
    const MCDecodedInst &Inst = BB.getInsts().back();

    if (Ana->isBranch(Inst.Inst)) {
      uint64_t targ = Ana->evaluateBranch(Inst.Inst, Inst.Address, Inst.Size);
      if (targ == -1ULL) {
        // Indirect branch. Bail and add all blocks of the function as a
        // successor.
        for (MCFunction::iterator i = f.begin(), e = f.end(); i != e; ++i)
          BB.addSucc(i->first);
      } else if (targ != Inst.Address+Inst.Size)
        BB.addSucc(targ);
      // Conditional branches can also fall through to the next block.
      if (Ana->isConditionalBranch(Inst.Inst) && llvm::next(i) != e)
        BB.addSucc(llvm::next(i)->first);
    } else {
      // No branch. Fall through to the next block.
      if (!Ana->isReturn(Inst.Inst) && llvm::next(i) != e)
        BB.addSucc(llvm::next(i)->first);
    }
  }

  return f;
}
