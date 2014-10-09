//===- lib/ReaderWriter/PECOFF/IdataPass.cpp ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IdataPass.h"
#include "Pass.h"

#include "lld/Core/File.h"
#include "lld/Core/Pass.h"
#include "lld/Core/Simple.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <map>
#include <vector>

using llvm::object::delay_import_directory_table_entry;

namespace lld {
namespace pecoff {
namespace idata {

IdataAtom::IdataAtom(IdataContext &context, std::vector<uint8_t> data)
    : COFFLinkerInternalAtom(context.dummyFile,
                             context.dummyFile.getNextOrdinal(), data) {
  context.file.addAtom(*this);
}

HintNameAtom::HintNameAtom(IdataContext &context, uint16_t hint,
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
ImportTableEntryAtom::assembleRawContent(uint64_t rva, bool is64) {
  // The element size of the import table is 32 bit in PE and 64 bit
  // in PE+. In PE+, bits 62-31 are filled with zero.
  if (is64) {
    std::vector<uint8_t> ret(8);
    *reinterpret_cast<llvm::support::ulittle64_t *>(&ret[0]) = rva;
    return ret;
  }
  std::vector<uint8_t> ret(4);
  *reinterpret_cast<llvm::support::ulittle32_t *>(&ret[0]) = rva;
  return ret;
}

static std::vector<ImportTableEntryAtom *>
createImportTableAtoms(IdataContext &context,
                       const std::vector<COFFSharedLibraryAtom *> &sharedAtoms,
                       bool shouldAddReference, StringRef sectionName,
                       llvm::BumpPtrAllocator &alloc) {
  std::vector<ImportTableEntryAtom *> ret;
  for (COFFSharedLibraryAtom *atom : sharedAtoms) {
    ImportTableEntryAtom *entry = nullptr;
    if (atom->importName().empty()) {
      // Import by ordinal
      uint64_t hint = atom->hint();
      hint |= context.ctx.is64Bit() ? (uint64_t(1) << 63) : (uint64_t(1) << 31);
      entry = new (alloc) ImportTableEntryAtom(context, hint, sectionName);
    } else {
      // Import by name
      entry = new (alloc) ImportTableEntryAtom(context, 0, sectionName);
      HintNameAtom *hintName =
          new (alloc) HintNameAtom(context, atom->hint(), atom->importName());
      addDir32NBReloc(entry, hintName, context.ctx.getMachineType(), 0);
    }
    ret.push_back(entry);
    if (shouldAddReference)
      atom->setImportTableEntry(entry);
  }
  // Add the NULL entry.
  ret.push_back(new (alloc) ImportTableEntryAtom(context, 0, sectionName));
  return ret;
}

// Creates atoms for an import lookup table. The import lookup table is an
// array of pointers to hint/name atoms. The array needs to be terminated with
// the NULL entry.
void ImportDirectoryAtom::addRelocations(
    IdataContext &context, StringRef loadName,
    const std::vector<COFFSharedLibraryAtom *> &sharedAtoms) {
  // Create parallel arrays. The contents of the two are initially the
  // same. The PE/COFF loader overwrites the import address tables with the
  // pointers to the referenced items after loading the executable into
  // memory.
  std::vector<ImportTableEntryAtom *> importLookupTables =
      createImportTableAtoms(context, sharedAtoms, false, ".idata.t", _alloc);
  std::vector<ImportTableEntryAtom *> importAddressTables =
      createImportTableAtoms(context, sharedAtoms, true, ".idata.a", _alloc);

  addDir32NBReloc(this, importLookupTables[0], context.ctx.getMachineType(),
                  offsetof(ImportDirectoryTableEntry, ImportLookupTableRVA));
  addDir32NBReloc(this, importAddressTables[0], context.ctx.getMachineType(),
                  offsetof(ImportDirectoryTableEntry, ImportAddressTableRVA));
  auto *atom = new (_alloc)
      COFFStringAtom(context.dummyFile, context.dummyFile.getNextOrdinal(),
                     ".idata", loadName);
  context.file.addAtom(*atom);
  addDir32NBReloc(this, atom, context.ctx.getMachineType(),
                  offsetof(ImportDirectoryTableEntry, NameRVA));
}

// Create the contents for the delay-import table.
std::vector<uint8_t> DelayImportDirectoryAtom::createContent() {
  std::vector<uint8_t> r(sizeof(delay_import_directory_table_entry), 0);
  auto entry = reinterpret_cast<delay_import_directory_table_entry *>(&r[0]);
  // link.exe seems to set 1 to Attributes field, so do we.
  entry->Attributes = 1;
  return r;
}

// Create the data referred by the delay-import table.
void DelayImportDirectoryAtom::addRelocations(
    IdataContext &context, StringRef loadName,
    const std::vector<COFFSharedLibraryAtom *> &sharedAtoms) {
  // "NameTable" field
  std::vector<ImportTableEntryAtom *> nameTable =
      createImportTableAtoms(context, sharedAtoms, true, ".didat", _alloc);
  addDir32NBReloc(
      this, nameTable[0], context.ctx.getMachineType(),
      offsetof(delay_import_directory_table_entry, DelayImportNameTable));

  // "Name" field
  auto *atom = new (_alloc)
      COFFStringAtom(context.dummyFile, context.dummyFile.getNextOrdinal(),
                     ".didat", loadName);
  context.file.addAtom(*atom);
  addDir32NBReloc(this, atom, context.ctx.getMachineType(),
                  offsetof(delay_import_directory_table_entry, Name));
  // TODO: emit other fields
}

} // namespace idata

void IdataPass::perform(std::unique_ptr<MutableFile> &file) {
  if (file->sharedLibrary().empty())
    return;

  idata::IdataContext context(*file, _dummyFile, _ctx);
  std::map<StringRef, std::vector<COFFSharedLibraryAtom *>> sharedAtoms =
      groupByLoadName(*file);
  bool hasImports = false;
  bool hasDelayImports = false;

  // Create the import table and terminate it with the null entry.
  for (auto i : sharedAtoms) {
    StringRef loadName = i.first;
    if (_ctx.isDelayLoadDLL(loadName))
      continue;
    hasImports = true;
    std::vector<COFFSharedLibraryAtom *> &atoms = i.second;
    new (_alloc) idata::ImportDirectoryAtom(context, loadName, atoms);
  }
  if (hasImports)
    new (_alloc) idata::NullImportDirectoryAtom(context);

  // Create the delay import table and terminate it with the null entry.
  for (auto i : sharedAtoms) {
    StringRef loadName = i.first;
    if (!_ctx.isDelayLoadDLL(loadName))
      continue;
    hasDelayImports = true;
    std::vector<COFFSharedLibraryAtom *> &atoms = i.second;
    new (_alloc) idata::DelayImportDirectoryAtom(context, loadName, atoms);
  }
  if (hasDelayImports)
    new (_alloc) idata::DelayNullImportDirectoryAtom(context);

  replaceSharedLibraryAtoms(*file);
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
void IdataPass::replaceSharedLibraryAtoms(MutableFile &file) {
  for (const DefinedAtom *atom : file.defined()) {
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
