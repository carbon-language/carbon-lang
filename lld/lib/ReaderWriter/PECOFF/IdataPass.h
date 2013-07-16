//===- lib/ReaderWriter/PECOFF/IdataPass.h---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file \brief This linker pass creates atoms for the DLL import
/// information. The defined atoms constructed in this pass will go into .idata
/// section, unless .idata section is merged with other section such as .data.
///
/// For the details of the .idata section format, see Microsoft PE/COFF
/// Specification section 5.4, The .idata Section.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PE_COFF_IDATA_PASS_H_
#define LLD_READER_WRITER_PE_COFF_IDATA_PASS_H_

#include "Atoms.h"

#include "lld/Core/File.h"
#include "lld/Core/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <map>

using lld::coff::COFFBaseDefinedAtom;
using lld::coff::COFFDefinedAtom;
using lld::coff::COFFLinkerInternalAtom;
using lld::coff::COFFReference;
using lld::coff::COFFSharedLibraryAtom;
using lld::coff::COFFSharedLibraryAtom;
using llvm::COFF::ImportDirectoryTableEntry;
using std::map;
using std::vector;

namespace lld {
namespace pecoff {

namespace {
class DLLNameAtom;
class HintNameAtom;
class ImportTableEntryAtom;

void addDir32NBReloc(COFFBaseDefinedAtom *atom, const Atom *target,
                     size_t offsetInAtom = 0) {
  atom->addReference(std::unique_ptr<COFFReference>(new COFFReference(
      target, offsetInAtom, llvm::COFF::IMAGE_REL_I386_DIR32NB)));
}

// A state object of this pass.
struct Context {
  explicit Context(MutableFile &f) : file(f) {}

  MutableFile &file;

  // The object to accumulate idata atoms. Idata atoms need to be grouped by
  // type and be continuous in the output file. To force such layout, we
  // accumulate all atoms created in the pass in the following vectors, and add
  // layout edges when finishing the pass.
  vector<COFFBaseDefinedAtom *> importDirectories;
  vector<ImportTableEntryAtom *> importLookupTables;
  vector<ImportTableEntryAtom *> importAddressTables;
  vector<HintNameAtom *> hintNameAtoms;
  vector<DLLNameAtom *> dllNameAtoms;

  map<StringRef, COFFBaseDefinedAtom *> sharedToDefinedAtom;
};

/// The root class of all idata atoms.
class IdataAtom : public COFFLinkerInternalAtom {
public:
  virtual ContentType contentType() const { return typeData; }
  virtual ContentPermissions permissions() const { return permR__; }

protected:
  IdataAtom(MutableFile &file, vector<uint8_t> data)
      : COFFLinkerInternalAtom(file, data) {
    file.addAtom(*this);
  }
};

/// A DLLNameAtom contains a name of a DLL and is referenced by the Name RVA
/// field in the import directory table entry.
class DLLNameAtom : public IdataAtom {
public:
  DLLNameAtom(Context &ctx, StringRef name)
      : IdataAtom(ctx.file, stringRefToVector(name)) {
    ctx.dllNameAtoms.push_back(this);
  }

private:
  vector<uint8_t> stringRefToVector(StringRef name) {
    vector<uint8_t> ret(name.size() + 1);
    memcpy(&ret[0], name.data(), name.size());
    ret[name.size()] = 0;
    return std::move(ret);
  }
};

/// A HintNameAtom represents a symbol that will be imported from a DLL at
/// runtime. It consists with an optional hint, which is a small integer, and a
/// symbol name.
///
/// A hint is an index of the export pointer table in a DLL. If the import
/// library and DLL is in sync (i.e., ".lib" and ".dll" is for the same version
/// or the symbol ordinal is maintained by hand with ".exp" file), the PE/COFF
/// loader can find the symbol quickly.
class HintNameAtom : public IdataAtom {
public:
  HintNameAtom(Context &ctx, uint16_t hint, StringRef name)
      : IdataAtom(ctx.file, assembleRawContent(hint, name)), _name(name) {
    ctx.hintNameAtoms.push_back(this);
  }

  StringRef getContentString() { return _name; }

private:
  // The first two bytes of the content is a hint, followed by a null-terminated
  // symbol name. The total size needs to be multiple of 2.
  vector<uint8_t> assembleRawContent(uint16_t hint, StringRef name) {
    name = unmangle(name);
    size_t size = llvm::RoundUpToAlignment(sizeof(hint) + name.size() + 1, 2);
    vector<uint8_t> ret(size);
    ret[name.size()] = 0;
    ret[name.size() - 1] = 0;
    *reinterpret_cast<llvm::support::ulittle16_t *>(&ret[0]) = hint;
    std::memcpy(&ret[2], name.data(), name.size());
    return ret;
  }

  /// Undo name mangling. In Windows, the symbol name for function is encoded
  /// as "_name@X", where X is the number of bytes of the arguments.
  StringRef unmangle(StringRef mangledName) {
    assert(mangledName.startswith("_"));
    return mangledName.substr(1).split('@').first;
  }

  StringRef _name;
};

class ImportTableEntryAtom : public IdataAtom {
public:
  explicit ImportTableEntryAtom(Context &ctx)
      : IdataAtom(ctx.file, vector<uint8_t>(4, 0)) {}
};

/// An ImportDirectoryAtom includes information to load a DLL, including a DLL
/// name, symbols that will be resolved from the DLL, and the import address
/// table that are overwritten by the loader with the pointers to the referenced
/// items. The executable has one ImportDirectoryAtom per one imported DLL.
class ImportDirectoryAtom : public IdataAtom {
public:
  ImportDirectoryAtom(Context &ctx, StringRef loadName,
                      const vector<COFFSharedLibraryAtom *> &sharedAtoms)
      : IdataAtom(ctx.file, vector<uint8_t>(20, 0)) {
    addRelocations(ctx, loadName, sharedAtoms);
    ctx.importDirectories.push_back(this);
  }

private:
  void addRelocations(Context &ctx, StringRef loadName,
                      const vector<COFFSharedLibraryAtom *> &sharedAtoms) {
    size_t lookupEnd = ctx.importLookupTables.size();
    size_t addressEnd = ctx.importAddressTables.size();

    // Create parallel arrays. The contents of the two are initially the
    // same. The PE/COFF loader overwrites the import address tables with the
    // pointers to the referenced items after loading the executable into
    // memory.
    addImportTableAtoms(ctx, sharedAtoms, false, ctx.importLookupTables);
    addImportTableAtoms(ctx, sharedAtoms, true, ctx.importAddressTables);

    addDir32NBReloc(this, ctx.importLookupTables[lookupEnd],
                    offsetof(ImportDirectoryTableEntry, ImportLookupTableRVA));
    addDir32NBReloc(this, ctx.importAddressTables[addressEnd],
                    offsetof(ImportDirectoryTableEntry, ImportAddressTableRVA));
    addDir32NBReloc(this, new (_alloc) DLLNameAtom(ctx, loadName),
                    offsetof(ImportDirectoryTableEntry, NameRVA));
  }

  // Creates atoms for an import lookup table. The import lookup table is an
  // array of pointers to hint/name atoms. The array needs to be terminated with
  // the NULL entry.
  void addImportTableAtoms(Context &ctx,
                           const vector<COFFSharedLibraryAtom *> &sharedAtoms,
                           bool shouldAddReference,
                           vector<ImportTableEntryAtom *> &ret) const {
    for (COFFSharedLibraryAtom *shared : sharedAtoms) {
      HintNameAtom *hintName = createHintNameAtom(ctx, shared);
      ImportTableEntryAtom *entry = new (_alloc) ImportTableEntryAtom(ctx);
      addDir32NBReloc(entry, hintName);
      ret.push_back(entry);
      if (shouldAddReference)
        shared->setImportTableEntry(entry);
    }
    // Add the NULL entry.
    ret.push_back(new (_alloc) ImportTableEntryAtom(ctx));
  }

  HintNameAtom *createHintNameAtom(
      Context &ctx, const COFFSharedLibraryAtom *atom) const {
    return new (_alloc) HintNameAtom(ctx, atom->hint(), atom->unmangledName());
  }

  mutable llvm::BumpPtrAllocator _alloc;
};

/// The last NULL entry in the import directory.
class NullImportDirectoryAtom : public IdataAtom {
public:
  explicit NullImportDirectoryAtom(Context &ctx)
      : IdataAtom(ctx.file, vector<uint8_t>(20, 0)) {
    ctx.importDirectories.push_back(this);
  }
};

} // anonymous namespace

class IdataPass : public lld::Pass {
public:
  virtual void perform(MutableFile &file) {
    if (file.sharedLibrary().size() == 0)
      return;

    Context ctx(file);
    map<StringRef, vector<COFFSharedLibraryAtom *>> sharedAtoms =
        groupByLoadName(file);
    for (auto i : sharedAtoms) {
      StringRef loadName = i.first;
      vector<COFFSharedLibraryAtom *> &atoms = i.second;
      createImportDirectory(ctx, loadName, atoms);
    }
    new (_alloc) NullImportDirectoryAtom(ctx);
    connectAtoms(ctx);
    createDataDirectoryAtoms(ctx);
    replaceSharedLibraryAtoms(ctx);
  }

private:
  map<StringRef, vector<COFFSharedLibraryAtom *>>
  groupByLoadName(MutableFile &file) {
    map<StringRef, COFFSharedLibraryAtom *> uniqueAtoms;
    for (const SharedLibraryAtom *atom : file.sharedLibrary())
      uniqueAtoms[atom->name()] = (COFFSharedLibraryAtom *)atom;

    map<StringRef, vector<COFFSharedLibraryAtom *>> ret;
    for (auto i : uniqueAtoms) {
      COFFSharedLibraryAtom *atom = i.second;
      ret[atom->loadName()].push_back(atom);
    }
    return std::move(ret);
  }

  void
  createImportDirectory(Context &ctx, StringRef loadName,
                        vector<COFFSharedLibraryAtom *> &dllAtoms) {
    new (_alloc) ImportDirectoryAtom(ctx, loadName, dllAtoms);
  }

  void connectAtoms(Context &ctx) {
    coff::connectAtomsWithLayoutEdge(ctx.importDirectories);
    coff::connectAtomsWithLayoutEdge(ctx.importLookupTables);
    coff::connectAtomsWithLayoutEdge(ctx.importAddressTables);
    coff::connectAtomsWithLayoutEdge(ctx.hintNameAtoms);
    coff::connectAtomsWithLayoutEdge(ctx.dllNameAtoms);
  }

  /// The addresses of the import dirctory and the import address table needs to
  /// be set to the COFF Optional Data Directory header. A COFFDataDirectoryAtom
  /// represents an entry in the data directory header. We create atoms of class
  /// COFFDataDirectoryAtom and set relocations to them, so that the address
  /// will be set by the writer.
  void createDataDirectoryAtoms(Context &ctx) {
    auto *dir = new (_alloc) coff::COFFDataDirectoryAtom(
        ctx.file, llvm::COFF::DataDirectoryIndex::IMPORT_TABLE);
    addDir32NBReloc(dir, ctx.importDirectories[0]);
    ctx.file.addAtom(*dir);

    auto *iat = new (_alloc) coff::COFFDataDirectoryAtom(
        ctx.file, llvm::COFF::DataDirectoryIndex::IAT);
    addDir32NBReloc(iat, ctx.importAddressTables[0]);
    ctx.file.addAtom(*iat);
  }

  /// Transforms a reference to a COFFSharedLibraryAtom to a real reference.
  void replaceSharedLibraryAtoms(Context &ctx) {
    for (const DefinedAtom *atom : ctx.file.defined()) {
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

  llvm::BumpPtrAllocator _alloc;
};

} // namespace pecoff
} // namespace lld

#endif
