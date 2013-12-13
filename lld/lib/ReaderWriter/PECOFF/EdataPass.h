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

/// The root class of all edata atoms.
class EdataAtom : public COFFLinkerInternalAtom {
public:
  EdataAtom(VirtualFile &file, size_t size)
    : COFFLinkerInternalAtom(file, file.getNextOrdinal(),
                             std::vector<uint8_t>(size)) {}

  virtual SectionChoice sectionChoice() const { return sectionCustomRequired; }
  virtual StringRef customSectionName() const { return ".edata"; }
  virtual ContentType contentType() const { return typeData; }
  virtual ContentPermissions permissions() const { return permR__; }

  template<typename T> T *getContents() const {
    return (T *)rawContent().data();
  }
};

} // namespace edata

class EdataPass : public lld::Pass {
public:
  EdataPass(const PECOFFLinkingContext &ctx)
      : _ctx(ctx), _file(ctx), _stringOrdinal(1024) {}

  virtual void perform(std::unique_ptr<MutableFile> &file);

private:
  edata::EdataAtom *createExportDirectoryTable(size_t numEntries);
  edata::EdataAtom *createAddressTable(
    const std::vector<const DefinedAtom *> &atoms);
  edata::EdataAtom *
  createNamePointerTable(const std::vector<const DefinedAtom *> &atoms,
                         MutableFile *file);
  edata::EdataAtom *createOrdinalTable(
    const std::vector<const DefinedAtom *> &atoms,
    const std::vector<const DefinedAtom *> &sortedAtoms);

  const PECOFFLinkingContext &_ctx;
  VirtualFile _file;
  int _stringOrdinal;
  mutable llvm::BumpPtrAllocator _alloc;
};

} // namespace pecoff
} // namespace lld

#endif
