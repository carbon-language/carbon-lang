//===- lib/ReaderWriter/PECOFF/LoadConfigPass.h ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file \brief This linker pass creates an atom for Load Configuration
/// structure.
///
/// For the details of the Load Configuration structure, see Microsoft PE/COFF
/// Specification section 5.8. The Load Configuration Structure (Image Only).
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PE_COFF_LOAD_CONFIG_PASS_H
#define LLD_READER_WRITER_PE_COFF_LOAD_CONFIG_PASS_H

#include "Atoms.h"

#include "lld/Core/File.h"
#include "lld/Core/Pass.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "lld/ReaderWriter/Simple.h"

#include <map>

namespace lld {
namespace pecoff {
namespace loadcfg {

class LoadConfigAtom : public COFFLinkerInternalAtom {
public:
  LoadConfigAtom(VirtualFile &file, const DefinedAtom *sxdata, int count);

  virtual SectionChoice sectionChoice() const { return sectionCustomRequired; }
  virtual StringRef customSectionName() const { return ".loadcfg"; }
  virtual ContentType contentType() const { return typeData; }
  virtual ContentPermissions permissions() const { return permR__; }

  template <typename T> T *getContents() const {
    return (T *)rawContent().data();
  }
};

} // namespace loadcfg

class LoadConfigPass : public lld::Pass {
public:
  LoadConfigPass(PECOFFLinkingContext &ctx) : _ctx(ctx), _file(ctx) {}

  virtual void perform(std::unique_ptr<MutableFile> &file);

private:
  PECOFFLinkingContext &_ctx;
  VirtualFile _file;
  mutable llvm::BumpPtrAllocator _alloc;
};

} // namespace pecoff
} // namespace lld

#endif
