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
#include "lld/ReaderWriter/Simple.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
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
  explicit Context(MutableFile &f, File &g) : file(f), dummyFile(g) {}

  MutableFile &file;
  File &dummyFile;

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
  IdataAtom(Context &context, vector<uint8_t> data)
      : COFFLinkerInternalAtom(context.dummyFile, data) {
    context.file.addAtom(*this);
  }
};

/// A DLLNameAtom contains a name of a DLL and is referenced by the Name RVA
/// field in the import directory table entry.
class DLLNameAtom : public IdataAtom {
public:
  DLLNameAtom(Context &context, StringRef name)
      : IdataAtom(context, stringRefToVector(name)) {
    context.dllNameAtoms.push_back(this);
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
  HintNameAtom(Context &context, uint16_t hint, StringRef importName)
      : IdataAtom(context, assembleRawContent(hint, importName)),
        _importName(importName) {
    context.hintNameAtoms.push_back(this);
  }

  StringRef getContentString() { return _importName; }

private:
  // The first two bytes of the content is a hint, followed by a null-terminated
  // symbol name. The total size needs to be multiple of 2.
  vector<uint8_t> assembleRawContent(uint16_t hint, StringRef importName) {
    size_t size =
        llvm::RoundUpToAlignment(sizeof(hint) + importName.size() + 1, 2);
    vector<uint8_t> ret(size);
    ret[importName.size()] = 0;
    ret[importName.size() - 1] = 0;
    *reinterpret_cast<llvm::support::ulittle16_t *>(&ret[0]) = hint;
    std::memcpy(&ret[2], importName.data(), importName.size());
    return ret;
  }

  StringRef _importName;
};

class ImportTableEntryAtom : public IdataAtom {
public:
  explicit ImportTableEntryAtom(Context &context, uint32_t contents)
      : IdataAtom(context, assembleRawContent(contents)) {}

private:
  vector<uint8_t> assembleRawContent(uint32_t contents) {
    vector<uint8_t> ret(4);
    *reinterpret_cast<llvm::support::ulittle32_t *>(&ret[0]) = contents;
    return ret;
  }
};

/// An ImportDirectoryAtom includes information to load a DLL, including a DLL
/// name, symbols that will be resolved from the DLL, and the import address
/// table that are overwritten by the loader with the pointers to the referenced
/// items. The executable has one ImportDirectoryAtom per one imported DLL.
class ImportDirectoryAtom : public IdataAtom {
public:
  ImportDirectoryAtom(Context &context, StringRef loadName,
                      const vector<COFFSharedLibraryAtom *> &sharedAtoms)
      : IdataAtom(context, vector<uint8_t>(20, 0)) {
    addRelocations(context, loadName, sharedAtoms);
    context.importDirectories.push_back(this);
  }

private:
  void addRelocations(Context &context, StringRef loadName,
                      const vector<COFFSharedLibraryAtom *> &sharedAtoms) {
    size_t lookupEnd = context.importLookupTables.size();
    size_t addressEnd = context.importAddressTables.size();

    // Create parallel arrays. The contents of the two are initially the
    // same. The PE/COFF loader overwrites the import address tables with the
    // pointers to the referenced items after loading the executable into
    // memory.
    addImportTableAtoms(context, sharedAtoms, false,
                        context.importLookupTables);
    addImportTableAtoms(context, sharedAtoms, true,
                        context.importAddressTables);

    addDir32NBReloc(this, context.importLookupTables[lookupEnd],
                    offsetof(ImportDirectoryTableEntry, ImportLookupTableRVA));
    addDir32NBReloc(this, context.importAddressTables[addressEnd],
                    offsetof(ImportDirectoryTableEntry, ImportAddressTableRVA));
    addDir32NBReloc(this, new (_alloc) DLLNameAtom(context, loadName),
                    offsetof(ImportDirectoryTableEntry, NameRVA));
  }

  // Creates atoms for an import lookup table. The import lookup table is an
  // array of pointers to hint/name atoms. The array needs to be terminated with
  // the NULL entry.
  void addImportTableAtoms(Context &context,
                           const vector<COFFSharedLibraryAtom *> &sharedAtoms,
                           bool shouldAddReference,
                           vector<ImportTableEntryAtom *> &ret) const {
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

  HintNameAtom *createHintNameAtom(Context &context,
                                   const COFFSharedLibraryAtom *atom) const {
    return new (_alloc) HintNameAtom(context, atom->hint(), atom->importName());
  }

  mutable llvm::BumpPtrAllocator _alloc;
};

/// The last NULL entry in the import directory.
class NullImportDirectoryAtom : public IdataAtom {
public:
  explicit NullImportDirectoryAtom(Context &context)
      : IdataAtom(context, vector<uint8_t>(20, 0)) {
    context.importDirectories.push_back(this);
  }
};

} // anonymous namespace

// An instance of this class represents "input file" for atoms created in this
// pass. Only one instance of this class is created as a field of IdataPass.
class IdataPassFile : public SimpleFile {
public:
  IdataPassFile(const LinkingContext &ctx)
      : SimpleFile(ctx, "<idata-pass-file>") {
    setOrdinal(ctx.getNextOrdinalAndIncrement());
  }
};

class IdataPass : public lld::Pass {
public:
  IdataPass(const LinkingContext &ctx) : _dummyFile(ctx) {}

  virtual void perform(MutableFile &file) {
    if (file.sharedLibrary().size() == 0)
      return;

    Context context(file, _dummyFile);
    map<StringRef, vector<COFFSharedLibraryAtom *> > sharedAtoms =
        groupByLoadName(file);
    for (auto i : sharedAtoms) {
      StringRef loadName = i.first;
      vector<COFFSharedLibraryAtom *> &atoms = i.second;
      createImportDirectory(context, loadName, atoms);
    }

    auto nidatom = new (_alloc) NullImportDirectoryAtom(context);
    context.file.addAtom(*nidatom);

    connectAtoms(context);
    createDataDirectoryAtoms(context);
    replaceSharedLibraryAtoms(context);
    for (auto id : context.importDirectories)
      context.file.addAtom(*id);
    for (auto ilt : context.importLookupTables)
      context.file.addAtom(*ilt);
    for (auto iat : context.importAddressTables)
      context.file.addAtom(*iat);
    for (auto hna : context.hintNameAtoms)
      context.file.addAtom(*hna);
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

  void createImportDirectory(Context &context, StringRef loadName,
                             vector<COFFSharedLibraryAtom *> &dllAtoms) {
    new (_alloc) ImportDirectoryAtom(context, loadName, dllAtoms);
  }

  // Append vec2's elements at the end of vec1.
  template <typename T, typename U>
  void appendAtoms(vector<T *> &vec1, const vector<U *> &vec2) {
    vec1.insert(vec1.end(), vec2.begin(), vec2.end());
  }

  void connectAtoms(Context &context) {
    vector<COFFBaseDefinedAtom *> atoms;
    appendAtoms(atoms, context.importDirectories);
    appendAtoms(atoms, context.importLookupTables);
    appendAtoms(atoms, context.importAddressTables);
    appendAtoms(atoms, context.dllNameAtoms);
    appendAtoms(atoms, context.hintNameAtoms);

    coff::connectAtomsWithLayoutEdge(atoms);
  }

  /// The addresses of the import dirctory and the import address table needs to
  /// be set to the COFF Optional Data Directory header. A COFFDataDirectoryAtom
  /// represents an entry in the data directory header. We create atoms of class
  /// COFFDataDirectoryAtom and set relocations to them, so that the address
  /// will be set by the writer.
  void createDataDirectoryAtoms(Context &context) {
    auto *dir = new (_alloc) coff::COFFDataDirectoryAtom(
        context.dummyFile, llvm::COFF::DataDirectoryIndex::IMPORT_TABLE,
        context.importDirectories.size() *
            context.importDirectories[0]->size());
    addDir32NBReloc(dir, context.importDirectories[0]);
    context.file.addAtom(*dir);

    auto *iat = new (_alloc) coff::COFFDataDirectoryAtom(
        context.dummyFile, llvm::COFF::DataDirectoryIndex::IAT,
        context.importAddressTables.size() *
            context.importAddressTables[0]->size());
    addDir32NBReloc(iat, context.importAddressTables[0]);
    context.file.addAtom(*iat);
  }

  /// Transforms a reference to a COFFSharedLibraryAtom to a real reference.
  void replaceSharedLibraryAtoms(Context &context) {
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

  // A dummy file with which all the atoms created in the pass will be
  // associated. Atoms need to be associated to an input file even if it's not
  // read from a file, so we use this object.
  IdataPassFile _dummyFile;

  llvm::BumpPtrAllocator _alloc;
};

} // namespace pecoff
} // namespace lld

#endif
