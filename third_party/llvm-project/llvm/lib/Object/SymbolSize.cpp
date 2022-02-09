//===- SymbolSize.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SymbolSize.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Object/XCOFFObjectFile.h"

using namespace llvm;
using namespace object;

// Orders increasingly by (SectionID, Address).
int llvm::object::compareAddress(const SymEntry *A, const SymEntry *B) {
  if (A->SectionID != B->SectionID)
    return A->SectionID < B->SectionID ? -1 : 1;
  if (A->Address != B->Address)
    return A->Address < B->Address ? -1 : 1;
  return 0;
}

static unsigned getSectionID(const ObjectFile &O, SectionRef Sec) {
  if (auto *M = dyn_cast<MachOObjectFile>(&O))
    return M->getSectionID(Sec);
  if (isa<WasmObjectFile>(&O))
    return Sec.getIndex();
  if (isa<XCOFFObjectFile>(&O))
    return Sec.getIndex();
  return cast<COFFObjectFile>(O).getSectionID(Sec);
}

static unsigned getSymbolSectionID(const ObjectFile &O, SymbolRef Sym) {
  if (auto *M = dyn_cast<MachOObjectFile>(&O))
    return M->getSymbolSectionID(Sym);
  if (const auto *M = dyn_cast<WasmObjectFile>(&O))
    return M->getSymbolSectionId(Sym);
  if (const auto *M = dyn_cast<XCOFFObjectFile>(&O))
    return M->getSymbolSectionID(Sym);
  return cast<COFFObjectFile>(O).getSymbolSectionID(Sym);
}

std::vector<std::pair<SymbolRef, uint64_t>>
llvm::object::computeSymbolSizes(const ObjectFile &O) {
  std::vector<std::pair<SymbolRef, uint64_t>> Ret;

  if (const auto *E = dyn_cast<ELFObjectFileBase>(&O)) {
    auto Syms = E->symbols();
    if (Syms.empty())
      Syms = E->getDynamicSymbolIterators();
    for (ELFSymbolRef Sym : Syms)
      Ret.push_back({Sym, Sym.getSize()});
    return Ret;
  }

  // Collect sorted symbol addresses. Include dummy addresses for the end
  // of each section.
  std::vector<SymEntry> Addresses;
  unsigned SymNum = 0;
  for (symbol_iterator I = O.symbol_begin(), E = O.symbol_end(); I != E; ++I) {
    SymbolRef Sym = *I;
    Expected<uint64_t> ValueOrErr = Sym.getValue();
    if (!ValueOrErr)
      // TODO: Actually report errors helpfully.
      report_fatal_error(ValueOrErr.takeError());
    Addresses.push_back({I, *ValueOrErr, SymNum, getSymbolSectionID(O, Sym)});
    ++SymNum;
  }
  for (SectionRef Sec : O.sections()) {
    uint64_t Address = Sec.getAddress();
    uint64_t Size = Sec.getSize();
    Addresses.push_back(
        {O.symbol_end(), Address + Size, 0, getSectionID(O, Sec)});
  }

  if (Addresses.empty())
    return Ret;

  array_pod_sort(Addresses.begin(), Addresses.end(), compareAddress);

  // Compute the size as the gap to the next symbol
  for (unsigned I = 0, N = Addresses.size() - 1; I < N; ++I) {
    auto &P = Addresses[I];
    if (P.I == O.symbol_end())
      continue;

    // If multiple symbol have the same address, give both the same size.
    unsigned NextI = I + 1;
    while (NextI < N && Addresses[NextI].Address == P.Address)
      ++NextI;

    uint64_t Size = Addresses[NextI].Address - P.Address;
    P.Address = Size;
  }

  // Assign the sorted symbols in the original order.
  Ret.resize(SymNum);
  for (SymEntry &P : Addresses) {
    if (P.I == O.symbol_end())
      continue;
    Ret[P.Number] = {*P.I, P.Address};
  }
  return Ret;
}
