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

#ifndef LLD_READER_WRITER_PE_COFF_IDATA_PASS_H
#define LLD_READER_WRITER_PE_COFF_IDATA_PASS_H

#include "Atoms.h"

#include "lld/Core/File.h"
#include "lld/Core/Pass.h"
#include "lld/Core/Simple.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "llvm/Support/COFF.h"

#include <algorithm>
#include <map>

using llvm::COFF::ImportDirectoryTableEntry;

namespace lld {
namespace pecoff {
namespace idata {

class DLLNameAtom;
class HintNameAtom;
class ImportTableEntryAtom;

// A state object of this pass.
struct IdataContext {
  IdataContext(MutableFile &f, VirtualFile &g, const PECOFFLinkingContext &c)
      : file(f), dummyFile(g), ctx(c) {}
  MutableFile &file;
  VirtualFile &dummyFile;
  const PECOFFLinkingContext &ctx;
};

/// The root class of all idata atoms.
class IdataAtom : public COFFLinkerInternalAtom {
public:
  SectionChoice sectionChoice() const override { return sectionCustomRequired; }
  StringRef customSectionName() const override { return ".idata"; }
  ContentType contentType() const override { return typeData; }
  ContentPermissions permissions() const override { return permR__; }

protected:
  IdataAtom(IdataContext &context, std::vector<uint8_t> data);
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
  HintNameAtom(IdataContext &context, uint16_t hint, StringRef importName);

  StringRef getContentString() { return _importName; }

private:
  std::vector<uint8_t> assembleRawContent(uint16_t hint, StringRef importName);
  StringRef _importName;
};

class ImportTableEntryAtom : public IdataAtom {
public:
  ImportTableEntryAtom(IdataContext &ctx, uint64_t contents,
                       StringRef sectionName)
      : IdataAtom(ctx, assembleRawContent(contents, ctx.ctx.is64Bit())),
        _sectionName(sectionName) {}

  StringRef customSectionName() const override {
    return _sectionName;
  };

private:
  std::vector<uint8_t> assembleRawContent(uint64_t contents, bool is64);
  StringRef _sectionName;
};

/// An ImportDirectoryAtom includes information to load a DLL, including a DLL
/// name, symbols that will be resolved from the DLL, and the import address
/// table that are overwritten by the loader with the pointers to the referenced
/// items. The executable has one ImportDirectoryAtom per one imported DLL.
class ImportDirectoryAtom : public IdataAtom {
public:
  ImportDirectoryAtom(IdataContext &context, StringRef loadName,
                      const std::vector<COFFSharedLibraryAtom *> &sharedAtoms)
      : IdataAtom(context, std::vector<uint8_t>(20, 0)) {
    addRelocations(context, loadName, sharedAtoms);
  }

  StringRef customSectionName() const override { return ".idata.d"; }

private:
  void addRelocations(IdataContext &context, StringRef loadName,
                      const std::vector<COFFSharedLibraryAtom *> &sharedAtoms);

  mutable llvm::BumpPtrAllocator _alloc;
};

/// The last NULL entry in the import directory.
class NullImportDirectoryAtom : public IdataAtom {
public:
  explicit NullImportDirectoryAtom(IdataContext &context)
      : IdataAtom(context, std::vector<uint8_t>(20, 0)) {}

  StringRef customSectionName() const override { return ".idata.d"; }
};

/// The class for the the delay-load import table.
class DelayImportDirectoryAtom : public IdataAtom {
public:
  DelayImportDirectoryAtom(
      IdataContext &context, StringRef loadName,
      const std::vector<COFFSharedLibraryAtom *> &sharedAtoms)
      : IdataAtom(context, createContent()) {
    addRelocations(context, loadName, sharedAtoms);
  }

  StringRef customSectionName() const override { return ".didat.d"; }

private:
  std::vector<uint8_t> createContent();
  void addRelocations(IdataContext &context, StringRef loadName,
                      const std::vector<COFFSharedLibraryAtom *> &sharedAtoms);

  mutable llvm::BumpPtrAllocator _alloc;
};

/// Terminator of the delay-load import table. The content of this atom is all
/// zero.
class DelayNullImportDirectoryAtom : public IdataAtom {
public:
  explicit DelayNullImportDirectoryAtom(IdataContext &context)
      : IdataAtom(context, createContent()) {}
  StringRef customSectionName() const override { return ".didat.d"; }

private:
  std::vector<uint8_t> createContent() const {
    return std::vector<uint8_t>(
        sizeof(llvm::object::delay_import_directory_table_entry), 0);
  }
};

class DelayImportAddressAtom : public IdataAtom {
public:
  explicit DelayImportAddressAtom(IdataContext &context)
      : IdataAtom(context, createContent(context.ctx)) {}
  StringRef customSectionName() const override { return ".data"; }
  ContentPermissions permissions() const override { return permRW_; }
  Alignment alignment() const override { return Alignment(3); }

private:
  std::vector<uint8_t> createContent(const PECOFFLinkingContext &ctx) const {
    return std::vector<uint8_t>(ctx.is64Bit() ? 8 : 4, 0);
  }
};

// DelayLoaderAtom contains a wrapper function for __delayLoadHelper2.
class DelayLoaderAtom : public IdataAtom {
public:
  DelayLoaderAtom(IdataContext &context, const Atom *impAtom,
                  const Atom *descAtom, const Atom *delayLoadHelperAtom);
  StringRef customSectionName() const override { return ".text"; }
  ContentPermissions permissions() const override { return permR_X; }
  Alignment alignment() const override { return Alignment(0); }

private:
  std::vector<uint8_t> createContent() const;
};

} // namespace idata

class IdataPass : public lld::Pass {
public:
  IdataPass(const PECOFFLinkingContext &ctx) : _dummyFile(ctx), _ctx(ctx) {}

  void perform(std::unique_ptr<MutableFile> &file) override;

private:
  std::map<StringRef, std::vector<COFFSharedLibraryAtom *>>
  groupByLoadName(MutableFile &file);

  void replaceSharedLibraryAtoms(MutableFile &file);

  // A dummy file with which all the atoms created in the pass will be
  // associated. Atoms need to be associated to an input file even if it's not
  // read from a file, so we use this object.
  VirtualFile _dummyFile;

  const PECOFFLinkingContext &_ctx;
  llvm::BumpPtrAllocator _alloc;
};

} // namespace pecoff
} // namespace lld

#endif
