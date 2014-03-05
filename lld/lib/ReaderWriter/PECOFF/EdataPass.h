//===- lib/ReaderWriter/PECOFF/EdataPass.h --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file \brief This linker pass creates atoms for the DLL export
/// information. The defined atoms constructed in this pass will go into .edata
/// section.
///
/// For the details of the .edata section format, see Microsoft PE/COFF
/// Specification section 5.3, The .edata Section.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PE_COFF_EDATA_PASS_H
#define LLD_READER_WRITER_PE_COFF_EDATA_PASS_H

#include "Atoms.h"

#include "lld/Core/File.h"
#include "lld/Core/Pass.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "lld/ReaderWriter/Simple.h"
#include "llvm/Support/COFF.h"

#include <map>

using llvm::COFF::ImportDirectoryTableEntry;

namespace lld {
namespace pecoff {
namespace edata {

struct TableEntry {
  TableEntry(StringRef exp, int ord, const DefinedAtom *a, bool n)
      : exportName(exp), ordinal(ord), atom(a), noname(n) {}
  StringRef exportName;
  int ordinal;
  const DefinedAtom *atom;
  bool noname;
};

/// The root class of all edata atoms.
class EdataAtom : public COFFLinkerInternalAtom {
public:
  EdataAtom(VirtualFile &file, size_t size)
      : COFFLinkerInternalAtom(file, file.getNextOrdinal(),
                               std::vector<uint8_t>(size)) {}

  SectionChoice sectionChoice() const override { return sectionCustomRequired; }
  StringRef customSectionName() const override { return ".edata"; }
  ContentType contentType() const override { return typeData; }
  ContentPermissions permissions() const override { return permR__; }

  template <typename T> T *getContents() const {
    return (T *)rawContent().data();
  }
};

} // namespace edata

class EdataPass : public lld::Pass {
public:
  EdataPass(PECOFFLinkingContext &ctx)
      : _ctx(ctx), _file(ctx), _stringOrdinal(1024) {}

  void perform(std::unique_ptr<MutableFile> &file) override;

private:
  edata::EdataAtom *
  createExportDirectoryTable(const std::vector<edata::TableEntry> &namedEntries,
                             int ordinalBase, int maxOrdinal);

  edata::EdataAtom *
  createAddressTable(const std::vector<edata::TableEntry> &entries,
                     int ordinalBase, int maxOrdinal);

  edata::EdataAtom *
  createNamePointerTable(const PECOFFLinkingContext &ctx,
                         const std::vector<edata::TableEntry> &entries,
                         MutableFile *file);

  edata::EdataAtom *
  createOrdinalTable(const std::vector<edata::TableEntry> &entries,
                     int ordinalBase);

  PECOFFLinkingContext &_ctx;
  VirtualFile _file;
  int _stringOrdinal;
  mutable llvm::BumpPtrAllocator _alloc;
};

} // namespace pecoff
} // namespace lld

#endif
