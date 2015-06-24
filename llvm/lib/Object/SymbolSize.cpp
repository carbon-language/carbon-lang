//===- SymbolSize.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SymbolSize.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ELFObjectFile.h"

using namespace llvm;
using namespace object;

namespace {
struct SymEntry {
  symbol_iterator I;
  uint64_t Address;
  unsigned Number;
  SectionRef Section;
};
}

static int compareAddress(const SymEntry *A, const SymEntry *B) {
  if (A->Section == B->Section)
    return A->Address - B->Address;
  if (A->Section < B->Section)
    return -1;
  if (A->Section == B->Section)
    return 0;
  return 1;
}

static int compareNumber(const SymEntry *A, const SymEntry *B) {
  return A->Number - B->Number;
}

ErrorOr<std::vector<std::pair<SymbolRef, uint64_t>>>
llvm::object::computeSymbolSizes(const ObjectFile &O) {
  std::vector<std::pair<SymbolRef, uint64_t>> Ret;

  if (const auto *E = dyn_cast<ELFObjectFileBase>(&O)) {
    for (SymbolRef Sym : E->symbols())
      Ret.push_back({Sym, E->getSymbolSize(Sym)});
    return Ret;
  }

  // Collect sorted symbol addresses. Include dummy addresses for the end
  // of each section.
  std::vector<SymEntry> Addresses;
  unsigned SymNum = 0;
  for (symbol_iterator I = O.symbol_begin(), E = O.symbol_end(); I != E; ++I) {
    SymbolRef Sym = *I;
    uint64_t Address;
    if (std::error_code EC = Sym.getAddress(Address))
      return EC;
    section_iterator SecI = O.section_end();
    if (std::error_code EC = Sym.getSection(SecI))
      return EC;
    Addresses.push_back({I, Address, SymNum, *SecI});
    ++SymNum;
  }
  for (const SectionRef Sec : O.sections()) {
    uint64_t Address = Sec.getAddress();
    uint64_t Size = Sec.getSize();
    Addresses.push_back({O.symbol_end(), Address + Size, 0, Sec});
  }
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

  // Put back in the original order and copy the result
  array_pod_sort(Addresses.begin(), Addresses.end(), compareNumber);
  for (SymEntry &P : Addresses) {
    if (P.I == O.symbol_end())
      continue;
    Ret.push_back({*P.I, P.Address});
  }
  return Ret;
}
