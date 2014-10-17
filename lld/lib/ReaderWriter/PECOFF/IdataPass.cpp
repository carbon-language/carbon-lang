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

// Find "___delayLoadHelper2@8" (or "__delayLoadHelper2" on x64).
// This is not efficient but should be OK for now.
static const Atom *
findDelayLoadHelper(MutableFile &file, const PECOFFLinkingContext &ctx) {
  StringRef sym = ctx.getDelayLoadHelperName();
  for (const DefinedAtom *atom : file.defined())
    if (atom->name() == sym)
      return atom;
  std::string msg = (sym + " was not found").str();
  llvm_unreachable(msg.c_str());
}

// Create the data referred by the delay-import table.
void DelayImportDirectoryAtom::addRelocations(
    IdataContext &context, StringRef loadName,
    const std::vector<COFFSharedLibraryAtom *> &sharedAtoms) {
  // "ModuleHandle" field. This points to an array of pointer-size data
  // in ".data" section. Initially the array is initialized with zero.
  // The delay-load import helper will set DLL base address at runtime.
  auto *hmodule = new (_alloc) DelayImportAddressAtom(context);
  addDir32NBReloc(this, hmodule, context.ctx.getMachineType(),
                  offsetof(delay_import_directory_table_entry, ModuleHandle));

  // "NameTable" field. The data structure of this field is the same
  // as (non-delay) import table's Import Lookup Table. Contains
  // imported function names. This is a parallel array of AddressTable
  // field.
  std::vector<ImportTableEntryAtom *> nameTable =
      createImportTableAtoms(context, sharedAtoms, false, ".didat", _alloc);
  addDir32NBReloc(
      this, nameTable[0], context.ctx.getMachineType(),
      offsetof(delay_import_directory_table_entry, DelayImportNameTable));

  // "Name" field. This points to the NUL-terminated DLL name string.
  auto *name = new (_alloc)
      COFFStringAtom(context.dummyFile, context.dummyFile.getNextOrdinal(),
                     ".didat", loadName);
  context.file.addAtom(*name);
  addDir32NBReloc(this, name, context.ctx.getMachineType(),
                  offsetof(delay_import_directory_table_entry, Name));

  // "AddressTable" field. This points to an array of pointers, which
  // in turn pointing to delay-load functions.
  std::vector<DelayImportAddressAtom *> addrTable;
  for (int i = 0, e = sharedAtoms.size() + 1; i < e; ++i)
    addrTable.push_back(new (_alloc) DelayImportAddressAtom(context));
  for (int i = 0, e = sharedAtoms.size(); i < e; ++i)
    sharedAtoms[i]->setImportTableEntry(addrTable[i]);
  addDir32NBReloc(
      this, addrTable[0], context.ctx.getMachineType(),
      offsetof(delay_import_directory_table_entry, DelayImportAddressTable));

  const Atom *delayLoadHelper = findDelayLoadHelper(context.file, context.ctx);
  for (int i = 0, e = sharedAtoms.size(); i < e; ++i) {
    const DefinedAtom *loader = new (_alloc) DelayLoaderAtom(
        context, addrTable[i], this, delayLoadHelper);
    addDir64Reloc(addrTable[i], loader, context.ctx.getMachineType(), 0);
  }
}

DelayLoaderAtom::DelayLoaderAtom(IdataContext &context, const Atom *impAtom,
                                 const Atom *descAtom, const Atom *delayLoadHelperAtom)
    : IdataAtom(context, createContent(context.ctx.getMachineType())) {
  MachineTypes machine = context.ctx.getMachineType();
  switch (machine) {
  case llvm::COFF::IMAGE_FILE_MACHINE_I386:
    addDir32Reloc(this, impAtom, machine, 3);
    addDir32Reloc(this, descAtom, machine, 8);
    addRel32Reloc(this, delayLoadHelperAtom, machine, 13);
    break;
  case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
    addRel32Reloc(this, impAtom, machine, 36);
    addRel32Reloc(this, descAtom, machine, 43);
    addRel32Reloc(this, delayLoadHelperAtom, machine, 48);
    break;
  default:
    llvm::report_fatal_error("unsupported machine type");
  }
}

// DelayLoaderAtom contains a wrapper function for __delayLoadHelper2.
//
// __delayLoadHelper2 takes two pointers: a pointer to the delay-load
// table descripter and a pointer to _imp_ symbol for the function
// to be resolved.
//
// __delayLoadHelper2 looks at the table descriptor to know the DLL
// name, calls dlopen()-like function to load it, resolves all
// imported symbols, and then writes the resolved addresses to the
// import address table. It returns a pointer to the resolved
// function.
//
// __delayLoadHelper2 is defined in delayimp.lib.
std::vector<uint8_t>
DelayLoaderAtom::createContent(MachineTypes machine) const {
  static const uint8_t x86[] = {
    0x51,              // push  ecx
    0x52,              // push  edx
    0x68, 0, 0, 0, 0,  // push  offset ___imp__<FUNCNAME>
    0x68, 0, 0, 0, 0,  // push  offset ___DELAY_IMPORT_DESCRIPTOR_<DLLNAME>_dll
    0xE8, 0, 0, 0, 0,  // call  ___delayLoadHelper2@8
    0x5A,              // pop   edx
    0x59,              // pop   ecx
    0xFF, 0xE0,        // jmp   eax
  };
  static const uint8_t x64[] = {
    0x51,                               // push    rcx
    0x52,                               // push    rdx
    0x41, 0x50,                         // push    r8
    0x41, 0x51,                         // push    r9
    0x48, 0x83, 0xEC, 0x48,             // sub     rsp, 48h
    0x66, 0x0F, 0x7F, 0x04, 0x24,       // movdqa  xmmword ptr [rsp], xmm0
    0x66, 0x0F, 0x7F, 0x4C, 0x24, 0x10, // movdqa  xmmword ptr [rsp+10h], xmm1
    0x66, 0x0F, 0x7F, 0x54, 0x24, 0x20, // movdqa  xmmword ptr [rsp+20h], xmm2
    0x66, 0x0F, 0x7F, 0x5C, 0x24, 0x30, // movdqa  xmmword ptr [rsp+30h], xmm3
    0x48, 0x8D, 0x15, 0, 0, 0, 0,       // lea     rdx, [__imp_<FUNCNAME>]
    0x48, 0x8D, 0x0D, 0, 0, 0, 0,       // lea     rcx, [___DELAY_IMPORT_...]
    0xE8, 0, 0, 0, 0,                   // call    __delayLoadHelper2
    0x66, 0x0F, 0x6F, 0x04, 0x24,       // movdqa  xmm0, xmmword ptr [rsp]
    0x66, 0x0F, 0x6F, 0x4C, 0x24, 0x10, // movdqa  xmm1, xmmword ptr [rsp+10h]
    0x66, 0x0F, 0x6F, 0x54, 0x24, 0x20, // movdqa  xmm2, xmmword ptr [rsp+20h]
    0x66, 0x0F, 0x6F, 0x5C, 0x24, 0x30, // movdqa  xmm3, xmmword ptr [rsp+30h]
    0x48, 0x83, 0xC4, 0x48,             // add     rsp, 48h
    0x41, 0x59,                         // pop     r9
    0x41, 0x58,                         // pop     r8
    0x5A,                               // pop     rdx
    0x59,                               // pop     rcx
    0xFF, 0xE0,                         // jmp     rax
  };
  switch (machine) {
  case llvm::COFF::IMAGE_FILE_MACHINE_I386:
    return std::vector<uint8_t>(x86, x86 + sizeof(x86));
  case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
    return std::vector<uint8_t>(x64, x64 + sizeof(x64));
  default:
    llvm::report_fatal_error("unsupported machine type");
  }
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
