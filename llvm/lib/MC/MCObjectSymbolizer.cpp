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
#include "llvm/Object/MachO.h"
#include "llvm/Object/ELFObjectFile.h"
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
  MCMachObjectSymbolizer(MCContext &Ctx, OwningPtr<MCRelocationInfo> &RelInfo,
                         const MachOObjectFile *MOOF);

  StringRef findExternalFunctionAt(uint64_t Addr) LLVM_OVERRIDE;

  void tryAddingPcLoadReferenceComment(raw_ostream &cStream,
                                       int64_t Value,
                                       uint64_t Address) LLVM_OVERRIDE;
};
} // End unnamed namespace


MCMachObjectSymbolizer::
MCMachObjectSymbolizer(MCContext &Ctx, OwningPtr<MCRelocationInfo> &RelInfo,
                       const MachOObjectFile *MOOF)
    : MCObjectSymbolizer(Ctx, RelInfo, MOOF), MOOF(MOOF),
      StubsStart(0), StubsCount(0), StubSize(0), StubsIndSymIndex(0) {

  error_code ec;
  for (section_iterator SI = MOOF->begin_sections(), SE = MOOF->end_sections();
       SI != SE; SI.increment(ec)) {
    if (ec) break;
    StringRef Name; SI->getName(Name);
    if (Name == "__stubs") {
      SectionRef StubsSec = *SI;
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
  symbol_iterator SI = MOOF->begin_symbols();
  error_code ec;
  for (uint32_t i = 0; i != SymtabIdx; ++i) {
    SI.increment(ec);
  }
  SI->getName(SymName);
  assert(SI != MOOF->end_symbols() && "Stub wasn't found in the symbol table!");
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

MCObjectSymbolizer::MCObjectSymbolizer(MCContext &Ctx,
                                       OwningPtr<MCRelocationInfo> &RelInfo,
                                       const ObjectFile *Obj)
    : MCSymbolizer(Ctx, RelInfo), Obj(Obj), SortedSections(), AddrToReloc() {
}

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
  error_code ec;
  for (symbol_iterator SI = Obj->begin_symbols(), SE = Obj->end_symbols();
       SI != SE; SI.increment(ec)) {
    if (ec) break;
    uint64_t SymAddr; SI->getAddress(SymAddr);
    uint64_t SymSize; SI->getSize(SymSize);
    StringRef SymName; SI->getName(SymName);
    SymbolRef::Type SymType; SI->getType(SymType);
    if (SymAddr == UnknownAddressOrSize || SymSize == UnknownAddressOrSize
        || SymName.empty() || SymType != SymbolRef::ST_Function)
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

MCObjectSymbolizer *
MCObjectSymbolizer::createObjectSymbolizer(MCContext &Ctx,
                                           OwningPtr<MCRelocationInfo> &RelInfo,
                                           const ObjectFile *Obj) {
  if (const MachOObjectFile *MOOF = dyn_cast<MachOObjectFile>(Obj))
    return new MCMachObjectSymbolizer(Ctx, RelInfo, MOOF);
  return new MCObjectSymbolizer(Ctx, RelInfo, Obj);
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
    return 0;
  uint64_t SAddr; It->getAddress(SAddr);
  uint64_t SSize; It->getSize(SSize);
  if (Addr >= SAddr + SSize)
    return 0;
  return &*It;
}

const RelocationRef *MCObjectSymbolizer::findRelocationAt(uint64_t Addr) {
  if (AddrToReloc.empty())
    buildRelocationByAddrMap();

  AddrToRelocMap::const_iterator RI = AddrToReloc.find(Addr);
  if (RI == AddrToReloc.end())
    return 0;
  return &RI->second;
}

void MCObjectSymbolizer::buildSectionList() {
  error_code ec;
  for (section_iterator SI = Obj->begin_sections(), SE = Obj->end_sections();
                        SI != SE; SI.increment(ec)) {
    if (ec) break;

    bool RequiredForExec; SI->isRequiredForExecution(RequiredForExec);
    if (RequiredForExec == false)
      continue;
    uint64_t SAddr; SI->getAddress(SAddr);
    uint64_t SSize; SI->getSize(SSize);
    SortedSectionList::iterator It = std::lower_bound(SortedSections.begin(),
                                                      SortedSections.end(),
                                                      SAddr,
                                                      SectionStartsBefore);
    if (It != SortedSections.end()) {
      uint64_t FoundSAddr; It->getAddress(FoundSAddr);
      if (FoundSAddr < SAddr + SSize)
        llvm_unreachable("Inserting overlapping sections");
    }
    SortedSections.insert(It, *SI);
  }
}

void MCObjectSymbolizer::buildRelocationByAddrMap() {
  error_code ec;
  for (section_iterator SI = Obj->begin_sections(), SE = Obj->end_sections();
                        SI != SE; SI.increment(ec)) {
    if (ec) break;

    section_iterator RelSecI = SI->getRelocatedSection();
    if (RelSecI == Obj->end_sections())
      continue;

    uint64_t StartAddr; RelSecI->getAddress(StartAddr);
    uint64_t Size; RelSecI->getSize(Size);
    bool RequiredForExec; RelSecI->isRequiredForExecution(RequiredForExec);
    if (RequiredForExec == false || Size == 0)
      continue;
    for (relocation_iterator RI = SI->begin_relocations(),
                             RE = SI->end_relocations();
                             RI != RE;
                             RI.increment(ec)) {
      if (ec) break;
      // FIXME: libObject is inconsistent regarding error handling. The
      // overwhelming majority of methods always return object_error::success,
      // and assert for simple errors.. Here, ELFObjectFile::getRelocationOffset
      // asserts when the file type isn't ET_REL.
      // This workaround handles x86-64 elf, the only one that has a relocinfo.
      uint64_t Offset;
      if (Obj->isELF()) {
        const ELF64LEObjectFile *ELFObj = dyn_cast<ELF64LEObjectFile>(Obj);
        if (ELFObj == 0)
          break;
        if (ELFObj->getELFFile()->getHeader()->e_type == ELF::ET_REL) {
          RI->getOffset(Offset);
          Offset += StartAddr;
        } else {
          RI->getAddress(Offset);
        }
      } else {
        RI->getOffset(Offset);
        Offset += StartAddr;
      }
      // At a specific address, only keep the first relocation.
      if (AddrToReloc.find(Offset) == AddrToReloc.end())
        AddrToReloc[Offset] = *RI;
    }
  }
}
