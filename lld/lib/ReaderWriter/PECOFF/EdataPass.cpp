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
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

#include <climits>
#include <ctime>
#include <utility>

using lld::pecoff::edata::EdataAtom;
using lld::pecoff::edata::TableEntry;
using llvm::object::export_address_table_entry;
using llvm::object::export_directory_table_entry;

namespace lld {
namespace pecoff {

static bool compare(const TableEntry &a, const TableEntry &b) {
  return a.exportName.compare(b.exportName) < 0;
}

static void assignOrdinals(PECOFFLinkingContext &ctx) {
  std::set<PECOFFLinkingContext::ExportDesc> exports;
  int maxOrdinal = -1;
  for (const PECOFFLinkingContext::ExportDesc &desc : ctx.getDllExports())
    maxOrdinal = std::max(maxOrdinal, desc.ordinal);
  int nextOrdinal = (maxOrdinal == -1) ? 1 : (maxOrdinal + 1);
  for (PECOFFLinkingContext::ExportDesc desc : ctx.getDllExports()) {
    if (desc.ordinal == -1)
      desc.ordinal = nextOrdinal++;
    exports.insert(desc);
  }
  ctx.getDllExports().swap(exports);
}

static bool getExportedAtoms(const PECOFFLinkingContext &ctx, MutableFile *file,
                             std::vector<TableEntry> &ret) {
  std::map<StringRef, const DefinedAtom *> definedAtoms;
  for (const DefinedAtom *atom : file->defined())
    definedAtoms[atom->name()] = atom;

  for (const PECOFFLinkingContext::ExportDesc &desc : ctx.getDllExports()) {
    auto it = definedAtoms.find(desc.name);
    if (it == definedAtoms.end()) {
      llvm::errs() << "Symbol <" << desc.name
                   << "> is exported but not defined.\n";
      return false;
    }
    const DefinedAtom *atom = it->second;
    ret.push_back(TableEntry(desc.name, desc.ordinal, atom, desc.noname));
  }
  std::sort(ret.begin(), ret.end(), compare);
  return true;
}

static std::pair<int, int> getOrdinalBase(std::vector<TableEntry> &entries) {
  int ordinalBase = INT_MAX;
  int maxOrdinal = -1;
  for (TableEntry &e : entries) {
    ordinalBase = std::min(ordinalBase, e.ordinal);
    maxOrdinal = std::max(maxOrdinal, e.ordinal);
  }
  return std::pair<int, int>(ordinalBase, maxOrdinal);
}

edata::EdataAtom *
EdataPass::createAddressTable(const std::vector<TableEntry> &entries,
                              int ordinalBase, int maxOrdinal) {
  EdataAtom *addressTable =
      new (_alloc) EdataAtom(_file, sizeof(export_address_table_entry) *
                                        (maxOrdinal - ordinalBase + 1));

  for (const TableEntry &e : entries) {
    int index = e.ordinal - ordinalBase;
    size_t offset = index * sizeof(export_address_table_entry);
    addDir32NBReloc(addressTable, e.atom, offset);
  }
  return addressTable;
}

edata::EdataAtom *
EdataPass::createNamePointerTable(const PECOFFLinkingContext &ctx,
                                  const std::vector<TableEntry> &entries,
                                  MutableFile *file) {
  EdataAtom *table =
      new (_alloc) EdataAtom(_file, sizeof(uint32_t) * entries.size());

  size_t offset = 0;
  for (const TableEntry &e : entries) {
    auto *stringAtom = new (_alloc) COFFStringAtom(
        _file, _stringOrdinal++, ".edata", ctx.undecorateSymbol(e.exportName));
    file->addAtom(*stringAtom);
    addDir32NBReloc(table, stringAtom, offset);
    offset += sizeof(uint32_t);
  }
  return table;
}

edata::EdataAtom *EdataPass::createExportDirectoryTable(
    const std::vector<edata::TableEntry> &namedEntries, int ordinalBase,
    int maxOrdinal) {
  EdataAtom *ret =
      new (_alloc) EdataAtom(_file, sizeof(export_directory_table_entry));
  auto *data = ret->getContents<export_directory_table_entry>();
  data->TimeDateStamp = time(nullptr);
  data->OrdinalBase = ordinalBase;
  data->AddressTableEntries = maxOrdinal - ordinalBase + 1;
  data->NumberOfNamePointers = namedEntries.size();
  return ret;
}

edata::EdataAtom *
EdataPass::createOrdinalTable(const std::vector<TableEntry> &entries,
                              int ordinalBase) {
  EdataAtom *ret =
      new (_alloc) EdataAtom(_file, sizeof(uint16_t) * entries.size());
  uint16_t *data = ret->getContents<uint16_t>();
  int i = 0;
  for (const TableEntry &e : entries)
    data[i++] = e.ordinal - ordinalBase;
  return ret;
}

void EdataPass::perform(std::unique_ptr<MutableFile> &file) {
  assignOrdinals(_ctx);

  std::vector<TableEntry> entries;
  if (!getExportedAtoms(_ctx, file.get(), entries))
    return;
  if (entries.empty())
    return;

  int ordinalBase, maxOrdinal;
  std::tie(ordinalBase, maxOrdinal) = getOrdinalBase(entries);

  std::vector<TableEntry> namedEntries;
  for (TableEntry &e : entries)
    if (!e.noname)
      namedEntries.push_back(e);

  EdataAtom *table =
      createExportDirectoryTable(namedEntries, ordinalBase, maxOrdinal);
  file->addAtom(*table);

  COFFStringAtom *dllName =
      new (_alloc) COFFStringAtom(_file, _stringOrdinal++, ".edata",
                                  llvm::sys::path::filename(_ctx.outputPath()));
  file->addAtom(*dllName);
  addDir32NBReloc(table, dllName,
                  offsetof(export_directory_table_entry, NameRVA));

  EdataAtom *addressTable =
      createAddressTable(entries, ordinalBase, maxOrdinal);
  file->addAtom(*addressTable);
  addDir32NBReloc(table, addressTable, offsetof(export_directory_table_entry,
                                                ExportAddressTableRVA));

  EdataAtom *namePointerTable =
      createNamePointerTable(_ctx, namedEntries, file.get());
  file->addAtom(*namePointerTable);
  addDir32NBReloc(table, namePointerTable,
                  offsetof(export_directory_table_entry, NamePointerRVA));

  EdataAtom *ordinalTable = createOrdinalTable(namedEntries, ordinalBase);
  file->addAtom(*ordinalTable);
  addDir32NBReloc(table, ordinalTable,
                  offsetof(export_directory_table_entry, OrdinalTableRVA));
}

} // namespace pecoff
} // namespace lld
