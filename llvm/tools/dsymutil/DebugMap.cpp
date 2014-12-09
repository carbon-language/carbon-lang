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
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace llvm {

using namespace llvm::object;

DebugMapObject::DebugMapObject(StringRef ObjectFilename)
  : Filename(ObjectFilename) {}

bool DebugMapObject::addSymbol(StringRef Name, uint64_t ObjectAddress,
                               uint64_t LinkedAddress) {
  auto InsertResult = Symbols.insert(std::make_pair(Name,
                                                    SymbolMapping{ObjectAddress,
                                                                  LinkedAddress}));
  return InsertResult.second;
}

void DebugMapObject::print(raw_ostream& OS) const {
  OS << getObjectFilename() << ":\n";
  // Sort the symbols in alphabetical order, like llvm-nm (and to get
  // deterministic output for testing).
  typedef StringMapEntry<SymbolMapping> MapEntryTy;
  std::vector<const MapEntryTy *> Entries;
  Entries.reserve(Symbols.getNumItems());
  for (auto SymIt = Symbols.begin(), End = Symbols.end(); SymIt != End; ++SymIt)
    Entries.push_back(&*SymIt);
  std::sort(Entries.begin(), Entries.end(),
            [] (const MapEntryTy *LHS, const MapEntryTy *RHS) {
              return LHS->getKey() < RHS->getKey();
            });
  for (const auto *Entry: Entries) {
    const auto &Sym = Entry->getValue();
    OS << format("\t%016" PRIx64 " => %016" PRIx64 "\t%s\n",
                     Sym.ObjectAddress, Sym.BinaryAddress, Entry->getKeyData());
  }
  OS << '\n';
}

#ifndef NDEBUG
void DebugMapObject::dump() const {
  print(errs());
}
#endif

DebugMapObject& DebugMap::addDebugMapObject(StringRef ObjectFilePath) {
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

void DebugMap::print(raw_ostream& OS) const {
  OS << "DEBUG MAP:   object addr =>  executable addr\tsymbol name\n";
  for (const auto &Obj: objects())
    Obj->print(OS);
  OS << "END DEBUG MAP\n";
}

#ifndef NDEBUG
void DebugMap::dump() const {
  print(errs());
}
#endif
}
