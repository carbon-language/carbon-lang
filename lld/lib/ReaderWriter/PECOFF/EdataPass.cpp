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

static bool compare(const DefinedAtom *a, const DefinedAtom *b) {
  return a->name().compare(b->name()) < 0;
}

static bool getExportedAtoms(const PECOFFLinkingContext &ctx, MutableFile *file,
                             std::vector<const DefinedAtom *> &ret) {
  std::map<StringRef, const DefinedAtom *> definedAtoms;
  for (const DefinedAtom *atom : file->defined())
    definedAtoms[atom->name()] = atom;

  for (const PECOFFLinkingContext::ExportDesc &desc : ctx.getDllExports()) {
    auto it = definedAtoms.find(ctx.decorateSymbol(desc.name));
    if (it == definedAtoms.end()) {
      llvm::errs() << "Symbol <" << desc.name
                   << "> is exported but not defined.\n";
      return false;
    }
    const DefinedAtom *atom = it->second;
    ret.push_back(atom);
  }
  std::sort(ret.begin(), ret.end(), compare);
  return true;
}

edata::EdataAtom *
EdataPass::createAddressTable(const std::vector<const DefinedAtom *> &atoms) {
  EdataAtom *addressTable = new (_alloc)
      EdataAtom(_file, sizeof(export_address_table_entry) * atoms.size());

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
  EdataAtom *table =
      new (_alloc) EdataAtom(_file, sizeof(uint32_t) * atoms.size());

  size_t offset = 0;
  for (const DefinedAtom *atom : atoms) {
    COFFStringAtom *stringAtom = new (_alloc)
        COFFStringAtom(_file, _stringOrdinal++, ".edata", atom->name());
    file->addAtom(*stringAtom);
    addDir32NBReloc(table, stringAtom, offset);
    offset += sizeof(uint32_t);
  }
  return table;
}

edata::EdataAtom *EdataPass::createExportDirectoryTable(size_t numEntries) {
  EdataAtom *ret =
      new (_alloc) EdataAtom(_file, sizeof(export_directory_table_entry));
  auto *data = ret->getContents<export_directory_table_entry>();
  data->TimeDateStamp = time(nullptr);
  data->OrdinalBase = 1;
  data->AddressTableEntries = numEntries;
  data->NumberOfNamePointers = numEntries;
  return ret;
}

edata::EdataAtom *EdataPass::createOrdinalTable(
    const std::vector<const DefinedAtom *> &atoms) {
  EdataAtom *ret =
      new (_alloc) EdataAtom(_file, sizeof(uint16_t) * atoms.size());
  uint16_t *data = ret->getContents<uint16_t>();
  for (size_t i = 0, e = atoms.size(); i < e; ++i)
    data[i] = i;
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

  COFFStringAtom *dllName =
      new (_alloc) COFFStringAtom(_file, _stringOrdinal++, ".edata",
                                  llvm::sys::path::filename(_ctx.outputPath()));
  file->addAtom(*dllName);
  addDir32NBReloc(table, dllName,
                  offsetof(export_directory_table_entry, NameRVA));

  EdataAtom *addressTable = createAddressTable(atoms);
  file->addAtom(*addressTable);
  addDir32NBReloc(table, addressTable, offsetof(export_directory_table_entry,
                                                ExportAddressTableRVA));

  EdataAtom *namePointerTable = createNamePointerTable(atoms, file.get());
  file->addAtom(*namePointerTable);
  addDir32NBReloc(table, namePointerTable,
                  offsetof(export_directory_table_entry, NamePointerRVA));

  EdataAtom *ordinalTable = createOrdinalTable(atoms);
  file->addAtom(*ordinalTable);
  addDir32NBReloc(table, ordinalTable,
                  offsetof(export_directory_table_entry, OrdinalTableRVA));
}

} // namespace pecoff
} // namespace lld
