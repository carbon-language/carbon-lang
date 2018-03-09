//===- MCPlusBuilder.cpp - main interface for MCPlus-level instructions ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Create/analyze/modify instructions at MC+ level.
//
//===----------------------------------------------------------------------===//

#include "MCPlusBuilder.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <queue>

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;

bool MCPlusBuilder::evaluateBranch(const MCInst &Inst, uint64_t Addr,
                                   uint64_t Size, uint64_t &Target) const {
  return Analysis->evaluateBranch(Inst, Addr, Size, Target);
}

namespace {

const MCLandingPad *findLandingPad(const MCInst &Inst) {
  for (unsigned I = Inst.getNumOperands(); I > 0; --I) {
    const auto &Op = Inst.getOperand(I - 1);
    if (Op.isLandingPad()) {
      return Op.getLandingPad();
    }
  }
  return nullptr;
}

}

bool MCPlusBuilder::hasEHInfo(const MCInst &Inst) const {
  return findLandingPad(Inst) != nullptr;
}

MCLandingPad MCPlusBuilder::getEHInfo(const MCInst &Inst) const {
  const MCSymbol *LPSym = nullptr;
  uint64_t Action = 0;
  if (isCall(Inst)) {
    if (auto LP = findLandingPad(Inst)) {
      std::tie(LPSym, Action) = *LP;
    }
  }

  return std::make_pair(LPSym, Action);
}

// Add handler and action info for call instruction.
void MCPlusBuilder::addEHInfo(MCInst &Inst,
                                const MCLandingPad &LP,
                                MCContext *Ctx) const {
  if (isCall(Inst)) {
    assert(!hasEHInfo(Inst));
    Inst.addOperand(
      MCOperand::createLandingPad(new (*Ctx) MCLandingPad(LP)));
  }
}

int64_t MCPlusBuilder::getGnuArgsSize(const MCInst &Inst) const {
  for (unsigned I = Inst.getNumOperands(); I > 0; --I) {
    const auto &Op = Inst.getOperand(I - 1);
    if (Op.isGnuArgsSize()) {
      return Op.getGnuArgsSize();
    }
  }
  return -1LL;
}

void MCPlusBuilder::addGnuArgsSize(MCInst &Inst, int64_t GnuArgsSize) const {
  assert(GnuArgsSize >= 0 && "cannot set GNU_args_size to negative value");
  assert(getGnuArgsSize(Inst) == -1LL && "GNU_args_size already set");
  assert(isInvoke(Inst) && "GNU_args_size can only be set for invoke");

  Inst.addOperand(MCOperand::createGnuArgsSize(GnuArgsSize));
}

uint64_t MCPlusBuilder::getJumpTable(const MCInst &Inst) const {
  for (unsigned I = Inst.getNumOperands(); I > 0; --I) {
    const auto &Op = Inst.getOperand(I - 1);
    if (Op.isJumpTable()) {
      return Op.getJumpTable();
    }
  }
  return 0;
}

bool MCPlusBuilder::setJumpTable(MCContext *Ctx, MCInst &Inst, uint64_t Value,
                                   uint16_t IndexReg) const {
  if (!isIndirectBranch(Inst))
    return false;
  assert(getJumpTable(Inst) == 0 && "jump table already set");
  Inst.addOperand(MCOperand::createJumpTable(Value));
  addAnnotation<>(Ctx, Inst, "JTIndexReg", IndexReg);
  return true;
}

Optional<uint64_t>
MCPlusBuilder::getConditionalTailCall(const MCInst &Inst) const {
  for (unsigned I = Inst.getNumOperands(); I > 0; --I) {
    const auto &Op = Inst.getOperand(I - 1);
    if (Op.isConditionalTailCall()) {
      return Op.getConditionalTailCall();
    }
  }
  return NoneType();
}

bool
MCPlusBuilder::setConditionalTailCall(MCInst &Inst, uint64_t Dest) const {
  if (!isConditionalBranch(Inst))
    return false;

  for (unsigned I = Inst.getNumOperands(); I > 0; --I) {
    auto &Op = Inst.getOperand(I - 1);
    if (Op.isConditionalTailCall()) {
      Op.setConditionalTailCall(Dest);
      return true;
    }
  }

  Inst.addOperand(MCOperand::createConditionalTailCall(Dest));
  return true;
}

bool MCPlusBuilder::unsetConditionalTailCall(MCInst &Inst) const {
  for (auto OpI = Inst.begin(), OpE = Inst.end(); OpI != OpE; ++OpI) {
    if (OpI->isConditionalTailCall()) {
      Inst.erase(OpI);
      return true;
    }
  }

  return false;
}

namespace {

unsigned findAnnotationIndex(const MCInst &Inst, StringRef Name) {
  for (unsigned I = Inst.getNumOperands(); I > 0; --I) {
    const auto& Op = Inst.getOperand(I - 1);
    if (Op.isAnnotation() && Op.getAnnotation()->getName() == Name) {
      return I - 1;
    }
  }
  return Inst.getNumOperands();
}

}

bool MCPlusBuilder::hasAnnotation(const MCInst &Inst, StringRef Name) const {
  return findAnnotationIndex(Inst, Name) < Inst.getNumOperands();
}

bool MCPlusBuilder::removeAnnotation(MCInst &Inst, StringRef Name) const {
  const auto Idx = findAnnotationIndex(Inst, Name);
  if (Idx < Inst.getNumOperands()) {
    auto *Annotation = Inst.getOperand(Idx).getAnnotation();
    auto Itr = AnnotationPool.find(Annotation);
    if (Itr != AnnotationPool.end()) {
      AnnotationPool.erase(Itr);
      Annotation->~MCAnnotation();
    }
    Inst.erase(Inst.begin() + Idx);
    return true;
  }
  return false;
}

void MCPlusBuilder::removeAllAnnotations(MCInst &Inst) const {
  for (auto Idx = Inst.getNumOperands(); Idx > 0; --Idx) {
    auto &Op = Inst.getOperand(Idx - 1);
    if (Op.isAnnotation()) {
      auto *Annotation = Op.getAnnotation();
      auto Itr = AnnotationPool.find(Annotation);
      if (Itr != AnnotationPool.end()) {
        AnnotationPool.erase(Itr);
        Annotation->~MCAnnotation();
      }
      Inst.erase(Inst.begin() + Idx - 1);
    }
  }
}

bool MCPlusBuilder::renameAnnotation(MCInst &Inst,
                                       StringRef Before,
                                       StringRef After) const {
  const auto Idx = findAnnotationIndex(Inst, Before);
  if (Idx >= Inst.getNumOperands()) {
    return false;
  }
  auto *Annotation = Inst.getOperand(Idx).getAnnotation();
  auto PooledName = AnnotationNames.intern(After);
  AnnotationNameRefs.insert(PooledName);
  Annotation->setName(*PooledName);
  return true;
}

const MCAnnotation *
MCPlusBuilder::getAnnotation(const MCInst &Inst, StringRef Name) const {
  const auto Idx = findAnnotationIndex(Inst, Name);
  assert(Idx < Inst.getNumOperands());
  return Inst.getOperand(Idx).getAnnotation();
}

void MCPlusBuilder::getClobberedRegs(const MCInst &Inst,
                                       BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const auto &InstInfo = Info->get(Inst.getOpcode());

  const auto *ImplicitDefs = InstInfo.getImplicitDefs();
  for (unsigned I = 0, E = InstInfo.getNumImplicitDefs(); I != E; ++I) {
    Regs |= getAliases(ImplicitDefs[I], /*OnlySmaller=*/false);
  }

  for (unsigned I = 0, E = InstInfo.getNumDefs(); I != E; ++I) {
    const auto &Operand = Inst.getOperand(I);
    assert(Operand.isReg());
    Regs |= getAliases(Operand.getReg(), /*OnlySmaller=*/false);
  }
}

void MCPlusBuilder::getTouchedRegs(const MCInst &Inst,
                                     BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const auto &InstInfo = Info->get(Inst.getOpcode());

  const auto *ImplicitDefs = InstInfo.getImplicitDefs();
  for (unsigned I = 0, E = InstInfo.getNumImplicitDefs(); I != E; ++I) {
    Regs |= getAliases(ImplicitDefs[I], /*OnlySmaller=*/false);
  }
  const auto *ImplicitUses = InstInfo.getImplicitUses();
  for (unsigned I = 0, E = InstInfo.getNumImplicitUses(); I != E; ++I) {
    Regs |= getAliases(ImplicitUses[I], /*OnlySmaller=*/false);
  }

  for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
    if (!Inst.getOperand(I).isReg())
      continue;
    Regs |= getAliases(Inst.getOperand(I).getReg(), /*OnlySmaller=*/false);
  }
}

void MCPlusBuilder::getWrittenRegs(const MCInst &Inst,
                                     BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const auto &InstInfo = Info->get(Inst.getOpcode());

  const auto *ImplicitDefs = InstInfo.getImplicitDefs();
  for (unsigned I = 0, E = InstInfo.getNumImplicitDefs(); I != E; ++I) {
    Regs |= getAliases(ImplicitDefs[I], /*OnlySmaller=*/true);
  }

  for (unsigned I = 0, E = InstInfo.getNumDefs(); I != E; ++I) {
    const auto &Operand = Inst.getOperand(I);
    assert(Operand.isReg());
    Regs |= getAliases(Operand.getReg(), /*OnlySmaller=*/true);
  }
}

void MCPlusBuilder::getUsedRegs(const MCInst &Inst, BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const auto &InstInfo = Info->get(Inst.getOpcode());

  const auto *ImplicitUses = InstInfo.getImplicitUses();
  for (unsigned I = 0, E = InstInfo.getNumImplicitUses(); I != E; ++I) {
    Regs |= getAliases(ImplicitUses[I], /*OnlySmaller=*/true);
  }

  for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
    if (!Inst.getOperand(I).isReg())
      continue;
    Regs |= getAliases(Inst.getOperand(I).getReg(), /*OnlySmaller=*/true);
  }
}

const BitVector &
MCPlusBuilder::getAliases(MCPhysReg Reg,
                            bool OnlySmaller) const {
  // AliasMap caches a mapping of registers to the set of registers that
  // alias (are sub or superregs of itself, including itself).
  static std::vector<BitVector> AliasMap;
  static std::vector<MCPhysReg> SuperReg;

  if (AliasMap.size() > 0) {
    if (OnlySmaller)
      return AliasMap[Reg];
    return AliasMap[SuperReg[Reg]];
  }
  // Build alias map
  for (MCPhysReg I = 0, E = RegInfo->getNumRegs(); I != E; ++I) {
    BitVector BV(RegInfo->getNumRegs(), false);
    BV.set(I);
    AliasMap.emplace_back(std::move(BV));
    SuperReg.emplace_back(I);
  }
  std::queue<MCPhysReg> Worklist;
  // Propagate alias info upwards
  for (MCPhysReg I = 0, E = RegInfo->getNumRegs(); I != E; ++I) {
    Worklist.push(I);
  }
  while (!Worklist.empty()) {
    MCPhysReg I = Worklist.front();
    Worklist.pop();
    for (MCSubRegIterator SI(I, RegInfo); SI.isValid(); ++SI) {
      AliasMap[I] |= AliasMap[*SI];
    }
    for (MCSuperRegIterator SI(I, RegInfo); SI.isValid(); ++SI) {
      Worklist.push(*SI);
    }
  }
  // Propagate parent reg downwards
  for (MCPhysReg I = 0, E = RegInfo->getNumRegs(); I != E; ++I) {
    Worklist.push(I);
  }
  while (!Worklist.empty()) {
    MCPhysReg I = Worklist.front();
    Worklist.pop();
    for (MCSubRegIterator SI(I, RegInfo); SI.isValid(); ++SI) {
      SuperReg[*SI] = SuperReg[I];
      Worklist.push(*SI);
    }
  }

  DEBUG({
    dbgs() << "Dumping reg alias table:\n";
    for (MCPhysReg I = 0, E = RegInfo->getNumRegs(); I != E; ++I) {
      dbgs() << "Reg " << I << ": ";
      const BitVector &BV = AliasMap[SuperReg[I]];
      int Idx = BV.find_first();
      while (Idx != -1) {
        dbgs() << Idx << " ";
        Idx = BV.find_next(Idx);
      }
      dbgs() << "\n";
    }
  });

  if (OnlySmaller)
    return AliasMap[Reg];
  return AliasMap[SuperReg[Reg]];
}

uint8_t
MCPlusBuilder::getRegSize(MCPhysReg Reg) const {
  // SizeMap caches a mapping of registers to their sizes
  static std::vector<uint8_t> SizeMap;

  if (SizeMap.size() > 0) {
    return SizeMap[Reg];
  }
  SizeMap = std::vector<uint8_t>(RegInfo->getNumRegs());
  // Build size map
  for (auto I = RegInfo->regclass_begin(), E = RegInfo->regclass_end(); I != E;
       ++I) {
    for (MCPhysReg Reg : *I) {
      SizeMap[Reg] = I->getSize();
    }
  }

  return SizeMap[Reg];
}
