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

static void addDir32NBReloc(COFFBaseDefinedAtom *atom, const Atom *target,
                            size_t offsetInAtom = 0) {
  atom->addReference(std::unique_ptr<COFFReference>(new COFFReference(
      target, offsetInAtom, llvm::COFF::IMAGE_REL_I386_DIR32NB)));
}

namespace idata {

IdataAtom::IdataAtom(Context &context, std::vector<uint8_t> data)
    : COFFLinkerInternalAtom(context.dummyFile,
                             context.dummyFile.getNextOrdinal(), data) {
  context.file.addAtom(*this);
}

HintNameAtom::HintNameAtom(Context &context, uint16_t hint,
                           StringRef importName)
    : IdataAtom(context, assembleRawContent(hint, importName)),
      _importName(importName) {}

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
  // Create parallel arrays. The contents of the two are initially the
  // same. The PE/COFF loader overwrites the import address tables with the
  // pointers to the referenced items after loading the executable into
  // memory.
  std::vector<ImportTableEntryAtom *> importLookupTables =
      createImportTableAtoms(context, sharedAtoms, false, ".idata.t");
  std::vector<ImportTableEntryAtom *> importAddressTables =
      createImportTableAtoms(context, sharedAtoms, true, ".idata.a");

  addDir32NBReloc(this, importLookupTables[0],
                  offsetof(ImportDirectoryTableEntry, ImportLookupTableRVA));
  addDir32NBReloc(this, importAddressTables[0],
                  offsetof(ImportDirectoryTableEntry, ImportAddressTableRVA));
  auto *atom = new (_alloc) coff::COFFStringAtom(
      context.dummyFile, context.dummyFile.getNextOrdinal(), ".idata",
      loadName);
  context.file.addAtom(*atom);
  addDir32NBReloc(this, atom, offsetof(ImportDirectoryTableEntry, NameRVA));
}

std::vector<ImportTableEntryAtom *> ImportDirectoryAtom::createImportTableAtoms(
    Context &context, const std::vector<COFFSharedLibraryAtom *> &sharedAtoms,
    bool shouldAddReference, StringRef sectionName) const {
  std::vector<ImportTableEntryAtom *> ret;
  for (COFFSharedLibraryAtom *atom : sharedAtoms) {
    ImportTableEntryAtom *entry = nullptr;
    if (atom->importName().empty()) {
      // Import by ordinal
      uint32_t hint = (1U << 31) | atom->hint();
      entry = new (_alloc) ImportTableEntryAtom(context, hint, sectionName);
    } else {
      // Import by name
      entry = new (_alloc) ImportTableEntryAtom(context, 0, sectionName);
      HintNameAtom *hintName =
          new (_alloc) HintNameAtom(context, atom->hint(), atom->importName());
      addDir32NBReloc(entry, hintName);
    }
    ret.push_back(entry);
    if (shouldAddReference)
      atom->setImportTableEntry(entry);
  }
  // Add the NULL entry.
  ret.push_back(new (_alloc) ImportTableEntryAtom(context, 0, sectionName));
  return ret;
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
    new (_alloc) idata::ImportDirectoryAtom(context, loadName, atoms);
  }

  // All atoms, including those of tyep NullImportDirectoryAtom, are added to
  // context.file in the IdataAtom's constructor.
  new (_alloc) idata::NullImportDirectoryAtom(context);

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
