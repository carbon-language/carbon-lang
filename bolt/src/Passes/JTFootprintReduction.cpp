//===--- JTFootprintReduction.cpp -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "JTFootprintReduction.h"
#include "llvm/Support/Options.h"

#define DEBUG_TYPE "JT"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<unsigned> Verbosity;
extern bool shouldProcess(const bolt::BinaryFunction &Function);

extern cl::opt<JumpTableSupportLevel> JumpTables;

static cl::opt<bool>
JTFootprintOnlyPIC("jt-footprint-optimize-for-icache",
  cl::desc("with jt-footprint-reduction, only process PIC jumptables and turn"
           " off other transformations that increase code size"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

void JTFootprintReduction::checkOpportunities(BinaryContext &BC,
                                              BinaryFunction &Function,
                                              DataflowInfoManager &Info) {
  std::map<JumpTable *, uint64_t> AllJTs;

  for (auto &BB : Function) {
    for (auto &Inst : BB) {
      auto *JumpTable = Function.getJumpTable(Inst);
      if (!JumpTable)
        continue;

      AllJTs[JumpTable] += BB.getKnownExecutionCount();
      ++IndJmps;

      if (BlacklistedJTs.count(JumpTable)) {
        ++IndJmpsDenied;
        continue;
      }

      uint64_t Scale;
      // Try a standard indirect jump matcher
      auto IndJmpMatcher = BC.MIB->matchIndJmp(
          BC.MIB->matchAnyOperand(), BC.MIB->matchImm(Scale),
          BC.MIB->matchReg(), BC.MIB->matchAnyOperand());
      if (!opts::JTFootprintOnlyPIC &&
          IndJmpMatcher->match(*BC.MRI, *BC.MIB,
                               MutableArrayRef<MCInst>(&*BB.begin(), &Inst + 1),
                               -1) &&
          Scale == 8) {
        if (Info.getLivenessAnalysis().scavengeRegAfter(&Inst))
          continue;
        BlacklistedJTs.insert(JumpTable);
        ++IndJmpsDenied;
        ++NumJTsNoReg;
        continue;
      }

      // Try a PIC matcher. The pattern we are looking for is a PIC JT ind jmp:
      //    addq    %rdx, %rsi
      //    addq    %rdx, %rdi
      //    leaq    DATAat0x402450(%rip), %r11
      //    movslq  (%r11,%rdx,4), %rcx
      //    addq    %r11, %rcx
      //    jmpq    *%rcx # JUMPTABLE @0x402450
      MCPhysReg BaseReg1;
      MCPhysReg BaseReg2;
      uint64_t Offset;
      auto PICIndJmpMatcher = BC.MIB->matchIndJmp(BC.MIB->matchAdd(
          BC.MIB->matchReg(BaseReg1),
          BC.MIB->matchLoad(BC.MIB->matchReg(BaseReg2), BC.MIB->matchImm(Scale),
                            BC.MIB->matchReg(), BC.MIB->matchImm(Offset))));
      auto PICBaseAddrMatcher = BC.MIB->matchIndJmp(
          BC.MIB->matchAdd(BC.MIB->matchLoadAddr(BC.MIB->matchSymbol()),
                           BC.MIB->matchAnyOperand()));
      if (!PICIndJmpMatcher->match(
              *BC.MRI, *BC.MIB,
              MutableArrayRef<MCInst>(&*BB.begin(), &Inst + 1), -1) ||
          Scale != 4 || BaseReg1 != BaseReg2 || Offset != 0 ||
          !PICBaseAddrMatcher->match(
              *BC.MRI, *BC.MIB,
              MutableArrayRef<MCInst>(&*BB.begin(), &Inst + 1), -1)) {
        BlacklistedJTs.insert(JumpTable);
        ++IndJmpsDenied;
        ++NumJTsBadMatch;
        continue;
      }
    }
  }

  // Statistics only
  for (const auto &JTFreq : AllJTs) {
    auto *JT = JTFreq.first;
    uint64_t CurScore = JTFreq.second;
    TotalJTScore += CurScore;
    if (!BlacklistedJTs.count(JT)) {
      OptimizedScore += CurScore;
      if (JT->EntrySize == 8)
        BytesSaved += JT->getSize() >> 1;
    }
  }
  TotalJTs += AllJTs.size();
  TotalJTsDenied += BlacklistedJTs.size();
}

bool JTFootprintReduction::tryOptimizeNonPIC(
    BinaryContext &BC, BinaryBasicBlock &BB,
    BinaryBasicBlock::iterator Inst, uint64_t JTAddr,
    JumpTable *JumpTable, DataflowInfoManager &Info) {
  if (opts::JTFootprintOnlyPIC)
    return false;

  MCOperand Base;
  uint64_t Scale;
  MCPhysReg Index;
  MCOperand Offset;
  auto IndJmpMatcher = BC.MIB->matchIndJmp(
      BC.MIB->matchAnyOperand(Base), BC.MIB->matchImm(Scale),
      BC.MIB->matchReg(Index), BC.MIB->matchAnyOperand(Offset));
  if (!IndJmpMatcher->match(*BC.MRI, *BC.MIB,
                            MutableArrayRef<MCInst>(&*BB.begin(), &*Inst + 1),
                            -1)) {
    return false;
  }

  assert(Scale == 8 && "Wrong scale");

  Scale = 4;
  IndJmpMatcher->annotate(*BC.MIB, "DeleteMe");

  auto &LA = Info.getLivenessAnalysis();
  MCPhysReg Reg = LA.scavengeRegAfter(&*Inst);
  assert(Reg != 0 && "Register scavenger failed!");
  auto RegOp = MCOperand::createReg(Reg);
  SmallVector<MCInst, 4> NewFrag;

  BC.MIB->createIJmp32Frag(NewFrag, Base, MCOperand::createImm(Scale),
                           MCOperand::createReg(Index), Offset, RegOp);
  BC.MIB->setJumpTable(NewFrag.back(), JTAddr, Index);

  JumpTable->OutputEntrySize = 4;

  BB.replaceInstruction(Inst, NewFrag.begin(), NewFrag.end());
  return true;
}

bool JTFootprintReduction::tryOptimizePIC(
    BinaryContext &BC, BinaryBasicBlock &BB,
    BinaryBasicBlock::iterator Inst, uint64_t JTAddr,
    JumpTable *JumpTable, DataflowInfoManager &Info) {
  MCPhysReg BaseReg;
  uint64_t Scale;
  MCPhysReg Index;
  MCOperand Offset;
  MCOperand JumpTableRef;
  auto PICIndJmpMatcher = BC.MIB->matchIndJmp(BC.MIB->matchAdd(
      BC.MIB->matchLoadAddr(BC.MIB->matchAnyOperand(JumpTableRef)),
      BC.MIB->matchLoad(BC.MIB->matchReg(BaseReg), BC.MIB->matchImm(Scale),
                        BC.MIB->matchReg(Index), BC.MIB->matchAnyOperand())));
  if (!PICIndJmpMatcher->match(*BC.MRI, *BC.MIB,
                              MutableArrayRef<MCInst>(&*BB.begin(), &*Inst + 1),
                               -1)) {
    return false;
  }

  assert(Scale == 4 && "Wrong scale");

  PICIndJmpMatcher->annotate(*BC.MIB, "DeleteMe");

  auto RegOp = MCOperand::createReg(BaseReg);
  SmallVector<MCInst, 4> NewFrag;

  BC.MIB->createIJmp32Frag(NewFrag, MCOperand::createReg(0),
                           MCOperand::createImm(Scale),
                           MCOperand::createReg(Index), JumpTableRef, RegOp);
  BC.MIB->setJumpTable(NewFrag.back(), JTAddr, Index);

  JumpTable->OutputEntrySize = 4;
  // DePICify
  JumpTable->Type = JumpTable::JTT_NORMAL;

  BB.replaceInstruction(Inst, NewFrag.begin(), NewFrag.end());
  return true;
}

void JTFootprintReduction::optimizeFunction(BinaryContext &BC,
                                            BinaryFunction &Function,
                                            DataflowInfoManager &Info) {
  for (auto &BB : Function) {
    if (!BB.getNumNonPseudos())
      continue;

    auto IndJmpRI = BB.getLastNonPseudo();
    auto IndJmp = std::prev(IndJmpRI.base());
    const auto JTAddr = BC.MIB->getJumpTable(*IndJmp);

    if (!JTAddr)
      continue;

    auto *JumpTable = Function.getJumpTable(*IndJmp);
    if (BlacklistedJTs.count(JumpTable))
      continue;

    if (tryOptimizeNonPIC(BC, BB, IndJmp, JTAddr, JumpTable, Info)
        || tryOptimizePIC(BC, BB, IndJmp, JTAddr, JumpTable, Info)) {
      Modified.insert(&Function);
      continue;
    }

    llvm_unreachable("Should either optimize PIC or NonPIC successfuly");
  }

  if (!Modified.count(&Function))
    return;

  for (auto &BB : Function) {
    for (auto I = BB.begin(); I != BB.end(); ) {
      if (BC.MIB->hasAnnotation(*I, "DeleteMe"))
        I = BB.eraseInstruction(I);
      else
        ++I;
    }
  }
}

void JTFootprintReduction::runOnFunctions(BinaryContext &BC) {
  if (opts::JumpTables == JTS_BASIC && BC.HasRelocations)
    return;

  std::unique_ptr<RegAnalysis> RA;
  std::unique_ptr<BinaryFunctionCallGraph> CG;
  if (!opts::JTFootprintOnlyPIC) {
    CG.reset(new BinaryFunctionCallGraph(buildCallGraph(BC)));
    RA.reset(new RegAnalysis(BC, &BC.getBinaryFunctions(), &*CG));
  }
  for (auto &BFIt : BC.getBinaryFunctions()) {
    auto &Function = BFIt.second;

    if (!Function.isSimple() || !opts::shouldProcess(Function))
      continue;

    if (Function.getKnownExecutionCount() == 0)
      continue;

    DataflowInfoManager Info(BC, Function, RA.get(), nullptr);
    BlacklistedJTs.clear();
    checkOpportunities(BC, Function, Info);
    optimizeFunction(BC, Function, Info);
  }

  if (TotalJTs == TotalJTsDenied) {
    outs() << "BOLT-INFO: JT Footprint reduction: no changes were made.\n";
    return;
  }

  outs() << "BOLT-INFO: JT Footprint reduction stats (simple funcs only):\n";
  if (OptimizedScore) {
    outs() << format("\t   %.2lf%%", (OptimizedScore * 100.0 / TotalJTScore))
           << " of dynamic JT entries were reduced.\n";
  }
  outs() << "\t   " << TotalJTs - TotalJTsDenied << " of " << TotalJTs
         << " jump tables affected.\n";
  outs() << "\t   " << IndJmps - IndJmpsDenied << " of " << IndJmps
         << " indirect jumps to JTs affected.\n";
  outs() << "\t   " << NumJTsBadMatch
         << " JTs discarded due to unsupported jump pattern.\n";
  outs() << "\t   " << NumJTsNoReg
         << " JTs discarded due to register unavailability.\n";
  outs() << "\t   " << BytesSaved
         << " bytes saved.\n";
}

} // namespace bolt
} // namespace llvm
