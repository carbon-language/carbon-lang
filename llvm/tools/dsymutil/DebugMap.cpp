//===- tools/dsymutil/DebugMap.cpp - Generic debug map representation -----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "DebugMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace llvm {
namespace dsymutil {

using namespace llvm::object;

DebugMapObject::DebugMapObject(StringRef ObjectFilename)
    : Filename(ObjectFilename) {}

bool DebugMapObject::addSymbol(StringRef Name, uint64_t ObjectAddress,
                               uint64_t LinkedAddress) {
  auto InsertResult = Symbols.insert(
      std::make_pair(Name, SymbolMapping(ObjectAddress, LinkedAddress)));
  return InsertResult.second;
}

void DebugMapObject::print(raw_ostream &OS) const {
  OS << getObjectFilename() << ":\n";
  // Sort the symbols in alphabetical order, like llvm-nm (and to get
  // deterministic output for testing).
  typedef std::pair<StringRef, SymbolMapping> Entry;
  std::vector<Entry> Entries;
  Entries.reserve(Symbols.getNumItems());
  for (const auto &Sym : make_range(Symbols.begin(), Symbols.end()))
    Entries.push_back(std::make_pair(Sym.getKey(), Sym.getValue()));
  std::sort(
      Entries.begin(), Entries.end(),
      [](const Entry &LHS, const Entry &RHS) { return LHS.first < RHS.first; });
  for (const auto &Sym : Entries) {
    OS << format("\t%016" PRIx64 " => %016" PRIx64 "\t%s\n",
                 Sym.second.ObjectAddress, Sym.second.BinaryAddress,
                 Sym.first.data());
  }
  OS << '\n';
}

#ifndef NDEBUG
void DebugMapObject::dump() const { print(errs()); }
#endif

DebugMapObject &DebugMap::addDebugMapObject(StringRef ObjectFilePath) {
  Objects.emplace_back(new DebugMapObject(ObjectFilePath));
  return *Objects.back();
}

const DebugMapObject::SymbolMapping *
DebugMapObject::lookupSymbol(StringRef SymbolName) const {
  StringMap<SymbolMapping>::const_iterator Sym = Symbols.find(SymbolName);
  if (Sym == Symbols.end())
    return nullptr;
  return &Sym->getValue();
}

void DebugMap::print(raw_ostream &OS) const {
  OS << "DEBUG MAP: " << BinaryTriple.getTriple()
     << "\n\tobject addr =>  executable addr\tsymbol name\n";
  for (const auto &Obj : objects())
    Obj->print(OS);
  OS << "END DEBUG MAP\n";
}

#ifndef NDEBUG
void DebugMap::dump() const { print(errs()); }
#endif
}
}
