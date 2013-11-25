//===- lib/ReaderWriter/PECOFF/IdataPass.cpp ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IdataPass.h"

#include "lld/Core/File.h"
#include "lld/Core/Pass.h"
#include "lld/ReaderWriter/Simple.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <map>

namespace lld {
namespace pecoff {

static std::vector<uint8_t> stringRefToVector(StringRef name) {
  std::vector<uint8_t> ret(name.size() + 1);
  memcpy(&ret[0], name.data(), name.size());
  ret[name.size()] = 0;
  return ret;
}

static void addDir32NBReloc(COFFBaseDefinedAtom *atom, const Atom *target,
                            size_t offsetInAtom = 0) {
  atom->addReference(std::unique_ptr<COFFReference>(new COFFReference(
      target, offsetInAtom, llvm::COFF::IMAGE_REL_I386_DIR32NB)));
}

namespace idata {
class DLLNameAtom;
class HintNameAtom;
class ImportTableEntryAtom;

IdataAtom::IdataAtom(Context &context, std::vector<uint8_t> data)
    : COFFLinkerInternalAtom(context.dummyFile,
                             context.dummyFile.getNextOrdinal(), data) {
  context.file.addAtom(*this);
}

DLLNameAtom::DLLNameAtom(Context &context, StringRef name)
    : IdataAtom(context, stringRefToVector(name)) {
  context.dllNameAtoms.push_back(this);
}

HintNameAtom::HintNameAtom(Context &context, uint16_t hint,
                           StringRef importName)
    : IdataAtom(context, assembleRawContent(hint, importName)),
      _importName(importName) {
  context.hintNameAtoms.push_back(this);
}

std::vector<uint8_t> HintNameAtom::assembleRawContent(uint16_t hint,
                                                      StringRef importName) {
  size_t size =
      llvm::RoundUpToAlignment(sizeof(hint) + importName.size() + 1, 2);
  std::vector<uint8_t> ret(size);
  ret[importName.size()] = 0;
  ret[importName.size() - 1] = 0;
  *reinterpret_cast<llvm::support::ulittle16_t *>(&ret[0]) = hint;
  std::memcpy(&ret[2], importName.data(), importName.size());
  return ret;
}

std::vector<uint8_t>
ImportTableEntryAtom::assembleRawContent(uint32_t contents) {
  std::vector<uint8_t> ret(4);
  *reinterpret_cast<llvm::support::ulittle32_t *>(&ret[0]) = contents;
  return ret;
}

// Creates atoms for an import lookup table. The import lookup table is an
// array of pointers to hint/name atoms. The array needs to be terminated with
// the NULL entry.
void ImportDirectoryAtom::addRelocations(
    Context &context, StringRef loadName,
    const std::vector<COFFSharedLibraryAtom *> &sharedAtoms) {
  size_t lookupEnd = context.importLookupTables.size();
  size_t addressEnd = context.importAddressTables.size();

  // Create parallel arrays. The contents of the two are initially the
  // same. The PE/COFF loader overwrites the import address tables with the
  // pointers to the referenced items after loading the executable into
  // memory.
  addImportTableAtoms(context, sharedAtoms, false, context.importLookupTables);
  addImportTableAtoms(context, sharedAtoms, true, context.importAddressTables);

  addDir32NBReloc(this, context.importLookupTables[lookupEnd],
                  offsetof(ImportDirectoryTableEntry, ImportLookupTableRVA));
  addDir32NBReloc(this, context.importAddressTables[addressEnd],
                  offsetof(ImportDirectoryTableEntry, ImportAddressTableRVA));
  addDir32NBReloc(this, new (_alloc) DLLNameAtom(context, loadName),
                  offsetof(ImportDirectoryTableEntry, NameRVA));
}

void ImportDirectoryAtom::addImportTableAtoms(
    Context &context, const std::vector<COFFSharedLibraryAtom *> &sharedAtoms,
    bool shouldAddReference, std::vector<ImportTableEntryAtom *> &ret) const {
  for (COFFSharedLibraryAtom *atom : sharedAtoms) {
    ImportTableEntryAtom *entry = nullptr;
    if (atom->importName().empty()) {
      // Import by ordinal
      uint32_t hint = (1U << 31) | atom->hint();
      entry = new (_alloc) ImportTableEntryAtom(context, hint);
    } else {
      // Import by name
      entry = new (_alloc) ImportTableEntryAtom(context, 0);
      HintNameAtom *hintName = createHintNameAtom(context, atom);
      addDir32NBReloc(entry, hintName);
    }
    ret.push_back(entry);
    if (shouldAddReference)
      atom->setImportTableEntry(entry);
  }
  // Add the NULL entry.
  ret.push_back(new (_alloc) ImportTableEntryAtom(context, 0));
}

HintNameAtom *ImportDirectoryAtom::createHintNameAtom(
    Context &context, const COFFSharedLibraryAtom *atom) const {
  return new (_alloc) HintNameAtom(context, atom->hint(), atom->importName());
}

} // namespace idata

void IdataPass::perform(std::unique_ptr<MutableFile> &file) {
  if (file->sharedLibrary().size() == 0)
    return;

  idata::Context context(*file, _dummyFile);
  std::map<StringRef, std::vector<COFFSharedLibraryAtom *> > sharedAtoms =
      groupByLoadName(*file);
  for (auto i : sharedAtoms) {
    StringRef loadName = i.first;
    std::vector<COFFSharedLibraryAtom *> &atoms = i.second;
    createImportDirectory(context, loadName, atoms);
  }

  // All atoms, including those of tyep NullImportDirectoryAtom, are added to
  // context.file in the IdataAtom's constructor.
  new (_alloc) idata::NullImportDirectoryAtom(context);

  connectAtoms(context);
  createDataDirectoryAtoms(context);
  replaceSharedLibraryAtoms(context);
}

std::map<StringRef, std::vector<COFFSharedLibraryAtom *> >
IdataPass::groupByLoadName(MutableFile &file) {
  std::map<StringRef, COFFSharedLibraryAtom *> uniqueAtoms;
  for (const SharedLibraryAtom *atom : file.sharedLibrary())
    uniqueAtoms[atom->name()] = (COFFSharedLibraryAtom *)atom;

  std::map<StringRef, std::vector<COFFSharedLibraryAtom *> > ret;
  for (auto i : uniqueAtoms) {
    COFFSharedLibraryAtom *atom = i.second;
    ret[atom->loadName()].push_back(atom);
  }
  return ret;
}

void IdataPass::createImportDirectory(
    idata::Context &context, StringRef loadName,
    std::vector<COFFSharedLibraryAtom *> &dllAtoms) {
  new (_alloc) idata::ImportDirectoryAtom(context, loadName, dllAtoms);
}

template <typename T, typename U>
void IdataPass::appendAtoms(std::vector<T *> &vec1,
                            const std::vector<U *> &vec2) {
  vec1.insert(vec1.end(), vec2.begin(), vec2.end());
}

void IdataPass::connectAtoms(idata::Context &context) {
  std::vector<COFFBaseDefinedAtom *> atoms;
  appendAtoms(atoms, context.importDirectories);
  appendAtoms(atoms, context.importLookupTables);
  appendAtoms(atoms, context.importAddressTables);
  appendAtoms(atoms, context.dllNameAtoms);
  appendAtoms(atoms, context.hintNameAtoms);
  coff::connectAtomsWithLayoutEdge(atoms);
}

/// The addresses of the import dirctory and the import address table needs to
/// be set to the COFF Optional Data Directory header. A COFFDataDirectoryAtom
/// represents the data directory header. We create a COFFDataDirectoryAtom
/// and set relocations to them, so that the address will be set by the
/// writer.
void IdataPass::createDataDirectoryAtoms(idata::Context &context) {
  // CLR_RUNTIME_HEADER is the last index of the data directory.
  int nentries = llvm::COFF::CLR_RUNTIME_HEADER + 1;
  int entSize = sizeof(llvm::object::data_directory);
  std::vector<uint8_t> contents(nentries * entSize, 0);

  auto importTableOffset =
      llvm::COFF::DataDirectoryIndex::IMPORT_TABLE * entSize;
  auto iatOffset = llvm::COFF::DataDirectoryIndex::IAT * entSize;

  auto *importTableEntry = reinterpret_cast<llvm::object::data_directory *>(
      &contents[0] + importTableOffset);
  auto *iatEntry = reinterpret_cast<llvm::object::data_directory *>(
      &contents[0] + iatOffset);

  importTableEntry->Size =
      context.importDirectories.size() * context.importDirectories[0]->size();
  iatEntry->Size = context.importAddressTables.size() *
                   context.importAddressTables[0]->size();

  auto *dir = new (_alloc) coff::COFFDataDirectoryAtom(
      context.dummyFile, context.dummyFile.getNextOrdinal(),
      std::move(contents));
  addDir32NBReloc(dir, context.importDirectories[0], importTableOffset);
  addDir32NBReloc(dir, context.importAddressTables[0], iatOffset);

  context.file.addAtom(*dir);
}

/// Transforms a reference to a COFFSharedLibraryAtom to a real reference.
void IdataPass::replaceSharedLibraryAtoms(idata::Context &context) {
  for (const DefinedAtom *atom : context.file.defined()) {
    for (const Reference *ref : *atom) {
      const Atom *target = ref->target();
      auto *sharedAtom = dyn_cast<SharedLibraryAtom>(target);
      if (!sharedAtom)
        continue;
      auto *coffSharedAtom = (COFFSharedLibraryAtom *)sharedAtom;
      const DefinedAtom *entry = coffSharedAtom->getImportTableEntry();
      const_cast<Reference *>(ref)->setTarget(entry);
    }
  }
}

} // namespace pecoff
} // namespace lld
