//===-- lib/MC/MCObjectSymbolizer.cpp -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCObjectSymbolizer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRelocationInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;
using namespace object;

//===- MCMachObjectSymbolizer ---------------------------------------------===//

namespace {
class MCMachObjectSymbolizer : public MCObjectSymbolizer {
  const MachOObjectFile *MOOF;
  // __TEXT;__stubs support.
  uint64_t StubsStart;
  uint64_t StubsCount;
  uint64_t StubSize;
  uint64_t StubsIndSymIndex;

public:
  MCMachObjectSymbolizer(MCContext &Ctx,
                         std::unique_ptr<MCRelocationInfo> RelInfo,
                         const MachOObjectFile *MOOF);

  StringRef findExternalFunctionAt(uint64_t Addr) override;

  void tryAddingPcLoadReferenceComment(raw_ostream &cStream, int64_t Value,
                                       uint64_t Address) override;
};
} // End unnamed namespace

MCMachObjectSymbolizer::MCMachObjectSymbolizer(
    MCContext &Ctx, std::unique_ptr<MCRelocationInfo> RelInfo,
    const MachOObjectFile *MOOF)
  : MCObjectSymbolizer(Ctx, std::move(RelInfo), MOOF), MOOF(MOOF),
    StubsStart(0), StubsCount(0), StubSize(0), StubsIndSymIndex(0) {

  for (const SectionRef &Section : MOOF->sections()) {
    StringRef Name;
    Section.getName(Name);
    if (Name == "__stubs") {
      SectionRef StubsSec = Section;
      if (MOOF->is64Bit()) {
        MachO::section_64 S = MOOF->getSection64(StubsSec.getRawDataRefImpl());
        StubsIndSymIndex = S.reserved1;
        StubSize = S.reserved2;
      } else {
        MachO::section S = MOOF->getSection(StubsSec.getRawDataRefImpl());
        StubsIndSymIndex = S.reserved1;
        StubSize = S.reserved2;
      }
      assert(StubSize && "Mach-O stub entry size can't be zero!");
      StubsSec.getAddress(StubsStart);
      StubsSec.getSize(StubsCount);
      StubsCount /= StubSize;
    }
  }
}

StringRef MCMachObjectSymbolizer::findExternalFunctionAt(uint64_t Addr) {
  // FIXME: also, this can all be done at the very beginning, by iterating over
  // all stubs and creating the calls to outside functions. Is it worth it
  // though?
  if (!StubSize)
    return StringRef();
  uint64_t StubIdx = (Addr - StubsStart) / StubSize;
  if (StubIdx >= StubsCount)
    return StringRef();

  uint32_t SymtabIdx =
    MOOF->getIndirectSymbolTableEntry(MOOF->getDysymtabLoadCommand(), StubIdx);

  StringRef SymName;
  symbol_iterator SI = MOOF->symbol_begin();
  for (uint32_t i = 0; i != SymtabIdx; ++i)
    ++SI;
  SI->getName(SymName);
  assert(SI != MOOF->symbol_end() && "Stub wasn't found in the symbol table!");
  assert(SymName.front() == '_' && "Mach-O symbol doesn't start with '_'!");
  return SymName.substr(1);
}

void MCMachObjectSymbolizer::
tryAddingPcLoadReferenceComment(raw_ostream &cStream, int64_t Value,
                                uint64_t Address) {
  if (const RelocationRef *R = findRelocationAt(Address)) {
    const MCExpr *RelExpr = RelInfo->createExprForRelocation(*R);
    if (!RelExpr || RelExpr->EvaluateAsAbsolute(Value) == false)
      return;
  }
  uint64_t Addr = Value;
  if (const SectionRef *S = findSectionContaining(Addr)) {
    StringRef Name; S->getName(Name);
    uint64_t SAddr; S->getAddress(SAddr);
    if (Name == "__cstring") {
      StringRef Contents;
      S->getContents(Contents);
      Contents = Contents.substr(Addr - SAddr);
      cStream << " ## literal pool for: "
              << Contents.substr(0, Contents.find_first_of(0));
    }
  }
}

//===- MCObjectSymbolizer -------------------------------------------------===//

MCObjectSymbolizer::MCObjectSymbolizer(
  MCContext &Ctx, std::unique_ptr<MCRelocationInfo> RelInfo,
  const ObjectFile *Obj)
  : MCSymbolizer(Ctx, std::move(RelInfo)), Obj(Obj), SortedSections(),
    AddrToReloc() {}

bool MCObjectSymbolizer::
tryAddingSymbolicOperand(MCInst &MI, raw_ostream &cStream,
                         int64_t Value, uint64_t Address, bool IsBranch,
                         uint64_t Offset, uint64_t InstSize) {
  if (IsBranch) {
    StringRef ExtFnName = findExternalFunctionAt((uint64_t)Value);
    if (!ExtFnName.empty()) {
      MCSymbol *Sym = Ctx.GetOrCreateSymbol(ExtFnName);
      const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, Ctx);
      MI.addOperand(MCOperand::CreateExpr(Expr));
      return true;
    }
  }

  if (const RelocationRef *R = findRelocationAt(Address + Offset)) {
    if (const MCExpr *RelExpr = RelInfo->createExprForRelocation(*R)) {
      MI.addOperand(MCOperand::CreateExpr(RelExpr));
      return true;
    }
    // Only try to create a symbol+offset expression if there is no relocation.
    return false;
  }

  // Interpret Value as a branch target.
  if (IsBranch == false)
    return false;
  uint64_t UValue = Value;
  // FIXME: map instead of looping each time?
  for (const SymbolRef &Symbol : Obj->symbols()) {
    uint64_t SymAddr;
    Symbol.getAddress(SymAddr);
    uint64_t SymSize;
    Symbol.getSize(SymSize);
    StringRef SymName;
    Symbol.getName(SymName);
    SymbolRef::Type SymType;
    Symbol.getType(SymType);
    if (SymAddr == UnknownAddressOrSize || SymSize == UnknownAddressOrSize ||
        SymName.empty() || SymType != SymbolRef::ST_Function)
      continue;

    if ( SymAddr == UValue ||
        (SymAddr <= UValue && SymAddr + SymSize > UValue)) {
      MCSymbol *Sym = Ctx.GetOrCreateSymbol(SymName);
      const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, Ctx);
      if (SymAddr != UValue) {
        const MCExpr *Off = MCConstantExpr::Create(UValue - SymAddr, Ctx);
        Expr = MCBinaryExpr::CreateAdd(Expr, Off, Ctx);
      }
      MI.addOperand(MCOperand::CreateExpr(Expr));
      return true;
    }
  }
  return false;
}

void MCObjectSymbolizer::
tryAddingPcLoadReferenceComment(raw_ostream &cStream,
                                int64_t Value, uint64_t Address) {
}

StringRef MCObjectSymbolizer::findExternalFunctionAt(uint64_t Addr) {
  return StringRef();
}

MCObjectSymbolizer *MCObjectSymbolizer::createObjectSymbolizer(
    MCContext &Ctx, std::unique_ptr<MCRelocationInfo> RelInfo,
    const ObjectFile *Obj) {
  if (const MachOObjectFile *MOOF = dyn_cast<MachOObjectFile>(Obj))
    return new MCMachObjectSymbolizer(Ctx, std::move(RelInfo), MOOF);
  return new MCObjectSymbolizer(Ctx, std::move(RelInfo), Obj);
}

// SortedSections implementation.

static bool SectionStartsBefore(const SectionRef &S, uint64_t Addr) {
  uint64_t SAddr; S.getAddress(SAddr);
  return SAddr < Addr;
}

const SectionRef *MCObjectSymbolizer::findSectionContaining(uint64_t Addr) {
  if (SortedSections.empty())
    buildSectionList();

  SortedSectionList::iterator
    EndIt = SortedSections.end(),
    It = std::lower_bound(SortedSections.begin(), EndIt,
                          Addr, SectionStartsBefore);
  if (It == EndIt)
    return nullptr;
  uint64_t SAddr; It->getAddress(SAddr);
  uint64_t SSize; It->getSize(SSize);
  if (Addr >= SAddr + SSize)
    return nullptr;
  return &*It;
}

const RelocationRef *MCObjectSymbolizer::findRelocationAt(uint64_t Addr) {
  if (AddrToReloc.empty())
    buildRelocationByAddrMap();

  AddrToRelocMap::const_iterator RI = AddrToReloc.find(Addr);
  if (RI == AddrToReloc.end())
    return nullptr;
  return &RI->second;
}

void MCObjectSymbolizer::buildSectionList() {
  for (const SectionRef &Section : Obj->sections()) {
    bool RequiredForExec;
    Section.isRequiredForExecution(RequiredForExec);
    if (RequiredForExec == false)
      continue;
    uint64_t SAddr;
    Section.getAddress(SAddr);
    uint64_t SSize;
    Section.getSize(SSize);
    SortedSectionList::iterator It =
        std::lower_bound(SortedSections.begin(), SortedSections.end(), SAddr,
                         SectionStartsBefore);
    if (It != SortedSections.end()) {
      uint64_t FoundSAddr; It->getAddress(FoundSAddr);
      if (FoundSAddr < SAddr + SSize)
        llvm_unreachable("Inserting overlapping sections");
    }
    SortedSections.insert(It, Section);
  }
}

void MCObjectSymbolizer::buildRelocationByAddrMap() {
  for (const SectionRef &Section : Obj->sections()) {
    for (const RelocationRef &Reloc : Section.relocations()) {
      uint64_t Address;
      Reloc.getAddress(Address);
      // At a specific address, only keep the first relocation.
      if (AddrToReloc.find(Address) == AddrToReloc.end())
        AddrToReloc[Address] = Reloc;
    }
  }
}
