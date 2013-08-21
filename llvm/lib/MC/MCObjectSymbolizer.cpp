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
public:
  MCMachObjectSymbolizer(MCContext &Ctx, OwningPtr<MCRelocationInfo> &RelInfo,
                         const MachOObjectFile *MOOF) {}

  void tryAddingPcLoadReferenceComment(raw_ostream &cStream,
                                       int64_t Value,
                                       uint64_t Address) LLVM_OVERRIDE;
};
} // End unnamed namespace


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
