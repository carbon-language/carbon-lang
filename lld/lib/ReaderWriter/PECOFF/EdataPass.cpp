//===- lib/ReaderWriter/PECOFF/EdataPass.cpp ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Pass.h"
#include "EdataPass.h"

#include "lld/Core/File.h"
#include "lld/Core/Pass.h"
#include "lld/ReaderWriter/Simple.h"
#include "llvm/Support/Path.h"

#include <ctime>

using lld::pecoff::edata::EdataAtom;
using llvm::object::export_address_table_entry;
using llvm::object::export_directory_table_entry;

namespace lld {
namespace pecoff {

static bool
getExportedAtoms(const PECOFFLinkingContext &ctx, MutableFile *file,
                 std::vector<const DefinedAtom *> &ret) {
  std::map<StringRef, const DefinedAtom *> definedAtoms;
  for (const DefinedAtom *atom : file->defined())
    definedAtoms[atom->name()] = atom;

  for (StringRef dllExport : ctx.getDllExports()) {
    auto it = definedAtoms.find(ctx.decorateSymbol(dllExport));
    if (it == definedAtoms.end()) {
      llvm::errs() << "Symbol <" << dllExport
                   << "> is exported but not defined.\n";
      return false;
    }
    const DefinedAtom *atom = it->second;
    ret.push_back(atom);
  }
  return true;
}

static bool compare(const DefinedAtom *a, const DefinedAtom *b) {
  return a->name().compare(b->name()) < 0;
}

edata::EdataAtom *
EdataPass::createAddressTable(const std::vector<const DefinedAtom *> &atoms) {
  EdataAtom *addressTable = new (_alloc) EdataAtom(
    _file, sizeof(export_address_table_entry) * atoms.size());

  size_t offset = 0;
  for (const DefinedAtom *atom : atoms) {
    addDir32NBReloc(addressTable, atom, offset);
    offset += sizeof(export_address_table_entry);
  }
  return addressTable;
}

edata::EdataAtom *
EdataPass::createNamePointerTable(const std::vector<const DefinedAtom *> &atoms,
                                  MutableFile *file) {
  EdataAtom *table = new (_alloc) EdataAtom(_file, sizeof(uint32_t) * atoms.size());

  size_t offset = 0;
  for (const DefinedAtom *atom : atoms) {
    COFFStringAtom *stringAtom = new (_alloc) COFFStringAtom(
      _file, _file.getNextOrdinal(), ".edata", atom->name());
    file->addAtom(*stringAtom);
    addDir32NBReloc(table, stringAtom, offset);
    offset += sizeof(uint32_t);
  }
  return table;
}

edata::EdataAtom *EdataPass::createExportDirectoryTable(size_t numEntries) {
  EdataAtom *ret = new (_alloc) EdataAtom(_file, sizeof(export_directory_table_entry));
  auto *data = ret->getContents<export_directory_table_entry>();
  data->TimeDateStamp = time(nullptr);
  data->OrdinalBase = 1;
  data->AddressTableEntries = numEntries;
  data->NumberOfNamePointers = numEntries;
  return ret;
}

edata::EdataAtom *
EdataPass::createOrdinalTable(const std::vector<const DefinedAtom *> &atoms,
                              const std::vector<const DefinedAtom *> &sortedAtoms) {
  EdataAtom *ret = new (_alloc) EdataAtom(_file, sizeof(uint16_t) * atoms.size());
  uint16_t *data = ret->getContents<uint16_t>();

  std::map<const DefinedAtom *, size_t> ordinals;
  size_t ordinal = 0;
  for (const DefinedAtom *atom : atoms)
    ordinals[atom] = ordinal++;

  size_t index = 0;
  for (const DefinedAtom *atom : sortedAtoms)
    data[index++] = ordinals[atom];
  return ret;
}

void EdataPass::perform(std::unique_ptr<MutableFile> &file) {
  std::vector<const DefinedAtom *> atoms;
  if (!getExportedAtoms(_ctx, file.get(), atoms))
    return;
  if (atoms.empty())
    return;

  EdataAtom *table = createExportDirectoryTable(atoms.size());
  file->addAtom(*table);

  COFFStringAtom *dllName = new (_alloc) COFFStringAtom(
    _file, _file.getNextOrdinal(),
    ".edata", llvm::sys::path::filename(_ctx.outputPath()));
  file->addAtom(*dllName);
  addDir32NBReloc(table, dllName, offsetof(export_directory_table_entry, NameRVA));

  EdataAtom *addressTable = createAddressTable(atoms);
  file->addAtom(*addressTable);
  addDir32NBReloc(table, addressTable,
                  offsetof(export_directory_table_entry, ExportAddressTableRVA));

  std::vector<const DefinedAtom *> sortedAtoms(atoms);
  std::sort(sortedAtoms.begin(), sortedAtoms.end(), compare);
  EdataAtom *namePointerTable = createNamePointerTable(sortedAtoms, file.get());
  file->addAtom(*namePointerTable);
  addDir32NBReloc(table, namePointerTable,
                  offsetof(export_directory_table_entry, NamePointerRVA));

  EdataAtom *ordinalTable = createOrdinalTable(atoms, sortedAtoms);
  file->addAtom(*ordinalTable);
  addDir32NBReloc(table, ordinalTable,
                  offsetof(export_directory_table_entry, OrdinalTableRVA));
}

} // namespace pecoff
} // namespace lld
