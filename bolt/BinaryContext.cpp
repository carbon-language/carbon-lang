//===--- BinaryContext.cpp  - Interface for machine-level context ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryContext.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {
namespace bolt {

MCSymbol *BinaryContext::getOrCreateGlobalSymbol(uint64_t Address,
                                                 Twine Prefix) {
  MCSymbol *Symbol{nullptr};
  std::string Name;
  auto NI = GlobalAddresses.find(Address);
  if (NI != GlobalAddresses.end()) {
    // Even though there could be multiple names registered at the address,
    // we only use the first one.
    Name = NI->second;
  } else {
    Name = (Prefix + "0x" + Twine::utohexstr(Address)).str();
    assert(GlobalSymbols.find(Name) == GlobalSymbols.end() &&
           "created name is not unique");
    GlobalAddresses.emplace(std::make_pair(Address, Name));
  }

  Symbol = Ctx->lookupSymbol(Name);
  if (Symbol)
    return Symbol;

  Symbol = Ctx->getOrCreateSymbol(Name);
  GlobalSymbols[Name] = Address;

  return Symbol;
}


void BinaryContext::buildOffsetToDWARFCompileUnitMap() {
  for (const auto &CU : DwCtx->compile_units()) {
    OffsetToDwarfCU[CU->getOffset()] = CU.get();
  }
}

} // namespace bolt
} // namespace llvm
