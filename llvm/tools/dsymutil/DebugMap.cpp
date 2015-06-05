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
                               uint64_t LinkedAddress, uint32_t Size) {
  auto InsertResult = Symbols.insert(
      std::make_pair(Name, SymbolMapping(ObjectAddress, LinkedAddress, Size)));

  if (InsertResult.second)
    AddressToMapping[ObjectAddress] = &*InsertResult.first;
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
    OS << format("\t%016" PRIx64 " => %016" PRIx64 "+0x%x\t%s\n",
                 uint64_t(Sym.second.ObjectAddress),
                 uint64_t(Sym.second.BinaryAddress), uint32_t(Sym.second.Size),
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

const DebugMapObject::DebugMapEntry *
DebugMapObject::lookupSymbol(StringRef SymbolName) const {
  StringMap<SymbolMapping>::const_iterator Sym = Symbols.find(SymbolName);
  if (Sym == Symbols.end())
    return nullptr;
  return &*Sym;
}

const DebugMapObject::DebugMapEntry *
DebugMapObject::lookupObjectAddress(uint64_t Address) const {
  auto Mapping = AddressToMapping.find(Address);
  if (Mapping == AddressToMapping.end())
    return nullptr;
  return Mapping->getSecond();
}

void DebugMap::print(raw_ostream &OS) const {
  yaml::Output yout(OS, /* Ctxt = */ nullptr, /* WrapColumn = */ 0);
  yout << const_cast<DebugMap &>(*this);
}

#ifndef NDEBUG
void DebugMap::dump() const { print(errs()); }
#endif

ErrorOr<std::unique_ptr<DebugMap>>
DebugMap::parseYAMLDebugMap(StringRef InputFile, StringRef PrependPath,
                            bool Verbose) {
  auto ErrOrFile = MemoryBuffer::getFileOrSTDIN(InputFile);
  if (auto Err = ErrOrFile.getError())
    return Err;

  std::unique_ptr<DebugMap> Res;
  yaml::Input yin((*ErrOrFile)->getBuffer(), &PrependPath);
  yin >> Res;

  if (auto EC = yin.error())
    return EC;

  return std::move(Res);
}
}

namespace yaml {

// Normalize/Denormalize between YAML and a DebugMapObject.
struct MappingTraits<dsymutil::DebugMapObject>::YamlDMO {
  YamlDMO(IO &io) {}
  YamlDMO(IO &io, dsymutil::DebugMapObject &Obj);
  dsymutil::DebugMapObject denormalize(IO &IO);

  std::string Filename;
  std::vector<dsymutil::DebugMapObject::YAMLSymbolMapping> Entries;
};

void MappingTraits<std::pair<std::string, DebugMapObject::SymbolMapping>>::
    mapping(IO &io, std::pair<std::string, DebugMapObject::SymbolMapping> &s) {
  io.mapRequired("sym", s.first);
  io.mapRequired("objAddr", s.second.ObjectAddress);
  io.mapRequired("binAddr", s.second.BinaryAddress);
  io.mapOptional("size", s.second.Size);
}

void MappingTraits<dsymutil::DebugMapObject>::mapping(
    IO &io, dsymutil::DebugMapObject &DMO) {
  MappingNormalization<YamlDMO, dsymutil::DebugMapObject> Norm(io, DMO);
  io.mapRequired("filename", Norm->Filename);
  io.mapRequired("symbols", Norm->Entries);
}

void ScalarTraits<Triple>::output(const Triple &val, void *,
                                  llvm::raw_ostream &out) {
  out << val.str();
}

StringRef ScalarTraits<Triple>::input(StringRef scalar, void *, Triple &value) {
  value = Triple(scalar);
  return StringRef();
}

size_t
SequenceTraits<std::vector<std::unique_ptr<dsymutil::DebugMapObject>>>::size(
    IO &io, std::vector<std::unique_ptr<dsymutil::DebugMapObject>> &seq) {
  return seq.size();
}

dsymutil::DebugMapObject &
SequenceTraits<std::vector<std::unique_ptr<dsymutil::DebugMapObject>>>::element(
    IO &, std::vector<std::unique_ptr<dsymutil::DebugMapObject>> &seq,
    size_t index) {
  if (index >= seq.size()) {
    seq.resize(index + 1);
    seq[index].reset(new dsymutil::DebugMapObject);
  }
  return *seq[index];
}

void MappingTraits<dsymutil::DebugMap>::mapping(IO &io,
                                                dsymutil::DebugMap &DM) {
  io.mapRequired("triple", DM.BinaryTriple);
  io.mapOptional("objects", DM.Objects);
}

void MappingTraits<std::unique_ptr<dsymutil::DebugMap>>::mapping(
    IO &io, std::unique_ptr<dsymutil::DebugMap> &DM) {
  if (!DM)
    DM.reset(new DebugMap());
  io.mapRequired("triple", DM->BinaryTriple);
  io.mapOptional("objects", DM->Objects);
}

MappingTraits<dsymutil::DebugMapObject>::YamlDMO::YamlDMO(
    IO &io, dsymutil::DebugMapObject &Obj) {
  Filename = Obj.Filename;
  Entries.reserve(Obj.Symbols.size());
  for (auto &Entry : Obj.Symbols)
    Entries.push_back(std::make_pair(Entry.getKey(), Entry.getValue()));
}

dsymutil::DebugMapObject
MappingTraits<dsymutil::DebugMapObject>::YamlDMO::denormalize(IO &IO) {
  void *Ctxt = IO.getContext();
  StringRef PrependPath = *reinterpret_cast<StringRef *>(Ctxt);
  SmallString<80> Path(PrependPath);
  sys::path::append(Path, Filename);
  dsymutil::DebugMapObject Res(Path);
  for (auto &Entry : Entries) {
    auto &Mapping = Entry.second;
    Res.addSymbol(Entry.first, Mapping.ObjectAddress, Mapping.BinaryAddress,
                  Mapping.Size);
  }
  return Res;
}
}
}
