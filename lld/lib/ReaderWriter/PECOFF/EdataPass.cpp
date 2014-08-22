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
#include "lld/Core/Simple.h"
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

static StringRef removeStdcallSuffix(StringRef sym) {
  if (!sym.startswith("_"))
    return sym;
  StringRef trimmed = sym.rtrim("0123456789");
  if (sym.size() != trimmed.size() && trimmed.endswith("@"))
    return trimmed.drop_back();
  return sym;
}

static StringRef removeLeadingUnderscore(StringRef sym) {
  if (sym.startswith("_"))
    return sym.substr(1);
  return sym;
}

static bool getExportedAtoms(PECOFFLinkingContext &ctx, MutableFile *file,
                             std::vector<TableEntry> &ret) {
  std::map<StringRef, const DefinedAtom *> definedAtoms;
  for (const DefinedAtom *atom : file->defined())
    definedAtoms[removeStdcallSuffix(atom->name())] = atom;

  std::set<PECOFFLinkingContext::ExportDesc> exports;
  for (PECOFFLinkingContext::ExportDesc desc : ctx.getDllExports()) {
    auto it = definedAtoms.find(desc.name);
    if (it == definedAtoms.end()) {
      llvm::errs() << "Symbol <" << desc.name
                   << "> is exported but not defined.\n";
      return false;
    }
    const DefinedAtom *atom = it->second;

    // One can export a symbol with a different name than the symbol
    // name used in DLL. If such name is specified, use it in the
    // .edata section.
    StringRef exportName =
        desc.externalName.empty() ? desc.name : desc.externalName;
    ret.push_back(TableEntry(exportName, desc.ordinal, atom, desc.noname));

    if (desc.externalName.empty())
      desc.externalName = removeLeadingUnderscore(atom->name());
    exports.insert(desc);
  }
  ctx.getDllExports().swap(exports);
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
    addDir32NBReloc(addressTable, e.atom, _is64, offset);
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
    addDir32NBReloc(table, stringAtom, _is64, offset);
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
  addDir32NBReloc(table, dllName, _is64,
                  offsetof(export_directory_table_entry, NameRVA));

  EdataAtom *addressTable =
      createAddressTable(entries, ordinalBase, maxOrdinal);
  file->addAtom(*addressTable);
  addDir32NBReloc(
      table, addressTable, _is64,
      offsetof(export_directory_table_entry, ExportAddressTableRVA));

  EdataAtom *namePointerTable =
      createNamePointerTable(_ctx, namedEntries, file.get());
  file->addAtom(*namePointerTable);
  addDir32NBReloc(table, namePointerTable, _is64,
                  offsetof(export_directory_table_entry, NamePointerRVA));

  EdataAtom *ordinalTable = createOrdinalTable(namedEntries, ordinalBase);
  file->addAtom(*ordinalTable);
  addDir32NBReloc(table, ordinalTable, _is64,
                  offsetof(export_directory_table_entry, OrdinalTableRVA));
}

} // namespace pecoff
} // namespace lld
