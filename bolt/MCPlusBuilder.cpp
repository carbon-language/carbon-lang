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

#include "MCPlus.h"
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
using namespace MCPlus;

Optional<MCLandingPad> MCPlusBuilder::getEHInfo(const MCInst &Inst) const {
  if (!isCall(Inst))
    return NoneType();
  auto LPSym = getAnnotationOpValue(Inst, MCAnnotation::kEHLandingPad);
  if (!LPSym)
    return NoneType();
  auto Action = getAnnotationOpValue(Inst, MCAnnotation::kEHAction);
  if (!Action)
    return NoneType();

  return std::make_pair(reinterpret_cast<const MCSymbol *>(*LPSym),
                        static_cast<uint64_t>(*Action));
}

void MCPlusBuilder::addEHInfo(MCInst &Inst, const MCLandingPad &LP) {
  if (isCall(Inst)) {
    assert(!getEHInfo(Inst));
    setAnnotationOpValue(Inst, MCAnnotation::kEHLandingPad,
                         reinterpret_cast<int64_t>(LP.first));
    setAnnotationOpValue(Inst, MCAnnotation::kEHAction,
                         static_cast<int64_t>(LP.second));
  }
}

int64_t MCPlusBuilder::getGnuArgsSize(const MCInst &Inst) const {
  auto Value = getAnnotationOpValue(Inst, MCAnnotation::kGnuArgsSize);
  if (!Value)
    return -1LL;
  return *Value;
}

void MCPlusBuilder::addGnuArgsSize(MCInst &Inst, int64_t GnuArgsSize) {
  assert(GnuArgsSize >= 0 && "cannot set GNU_args_size to negative value");
  assert(getGnuArgsSize(Inst) == -1LL && "GNU_args_size already set");
  assert(isInvoke(Inst) && "GNU_args_size can only be set for invoke");

  setAnnotationOpValue(Inst, MCAnnotation::kGnuArgsSize, GnuArgsSize);
}

uint64_t MCPlusBuilder::getJumpTable(const MCInst &Inst) const {
  auto Value = getAnnotationOpValue(Inst, MCAnnotation::kJumpTable);
  if (!Value)
    return 0;
  return *Value;
}

bool MCPlusBuilder::setJumpTable(MCInst &Inst, uint64_t Value,
                                 uint16_t IndexReg) {
  if (!isIndirectBranch(Inst))
    return false;
  assert(getJumpTable(Inst) == 0 && "jump table already set");
  setAnnotationOpValue(Inst, MCAnnotation::kJumpTable, Value);
  addAnnotation<>(Inst, "JTIndexReg", IndexReg);
  return true;
}

Optional<uint64_t>
MCPlusBuilder::getConditionalTailCall(const MCInst &Inst) const {
  auto Value = getAnnotationOpValue(Inst, MCAnnotation::kConditionalTailCall);
  if (!Value)
    return NoneType();
  return static_cast<uint64_t>(*Value);
}

bool
MCPlusBuilder::setConditionalTailCall(MCInst &Inst, uint64_t Dest) {
  if (!isConditionalBranch(Inst))
    return false;

  setAnnotationOpValue(Inst, MCAnnotation::kConditionalTailCall, Dest);
  return true;
}

bool MCPlusBuilder::unsetConditionalTailCall(MCInst &Inst) {
  if (!getConditionalTailCall(Inst))
    return false;
  removeAnnotation(Inst, MCAnnotation::kConditionalTailCall);
  return true;
}

bool MCPlusBuilder::hasAnnotation(const MCInst &Inst, unsigned Index) const {
  const auto *AnnotationInst = getAnnotationInst(Inst);
  if (!AnnotationInst)
    return false;

  return (bool)getAnnotationOpValue(Inst, Index);
}

bool MCPlusBuilder::removeAnnotation(MCInst &Inst, unsigned Index) {
  auto *AnnotationInst = getAnnotationInst(Inst);
  if (!AnnotationInst)
    return false;

  for (int I = AnnotationInst->getNumOperands() - 1; I >= 0; --I) {
    auto ImmValue = AnnotationInst->getOperand(I).getImm();
    if (extractAnnotationIndex(ImmValue) == Index) {
      AnnotationInst->erase(AnnotationInst->begin() + I);
      auto *Annotation =
        reinterpret_cast<MCAnnotation *>(extractAnnotationValue(ImmValue));
      auto Itr = AnnotationPool.find(Annotation);
      if (Itr != AnnotationPool.end()) {
        AnnotationPool.erase(Itr);
        Annotation->~MCAnnotation();
      }
      return true;
    }
  }

  return false;
}

void MCPlusBuilder::removeAllAnnotations(MCInst &Inst) {
  auto *AnnotationInst = getAnnotationInst(Inst);
  if (!AnnotationInst)
    return;

  for (int I = AnnotationInst->getNumOperands() - 1; I >= 0; --I) {
    auto ImmValue = AnnotationInst->getOperand(I).getImm();
    AnnotationInst->erase(std::prev(AnnotationInst->end()));
    auto *Annotation =
      reinterpret_cast<MCAnnotation *>(extractAnnotationValue(ImmValue));
    auto Itr = AnnotationPool.find(Annotation);
    if (Itr != AnnotationPool.end()) {
      AnnotationPool.erase(Itr);
      Annotation->~MCAnnotation();
    }
  }

  // Clear all attached MC+ info since it's no longer used.
  Inst.erase(std::prev(Inst.end()));
}

void
MCPlusBuilder::printAnnotations(const MCInst &Inst, raw_ostream &OS) const {
  const auto *AnnotationInst = getAnnotationInst(Inst);
  if (!AnnotationInst)
    return;

  for (unsigned I = 0; I < AnnotationInst->getNumOperands(); ++I) {
    const auto Imm = AnnotationInst->getOperand(I).getImm();
    const auto Index = extractAnnotationIndex(Imm);
    const auto Value = extractAnnotationValue(Imm);
    const auto *Annotation =
        reinterpret_cast<const MCAnnotation *>(Value);
    if (Index >= MCAnnotation::kGeneric) {
      OS << " # " << AnnotationNames[Index - MCAnnotation::kGeneric]
         << ": ";
      Annotation->print(OS);
    }
  }
}

bool MCPlusBuilder::evaluateBranch(const MCInst &Inst, uint64_t Addr,
                                   uint64_t Size, uint64_t &Target) const {
  return Analysis->evaluateBranch(Inst, Addr, Size, Target);
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
