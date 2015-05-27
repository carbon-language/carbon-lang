//===- lib/ReaderWriter/ELF/Mips/MipsELFWriters.h -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_WRITERS_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_WRITERS_H

#include "DynamicLibraryWriter.h"
#include "ExecutableWriter.h"
#include "MipsLinkingContext.h"

namespace lld {
namespace elf {

template <class ELFT> class MipsTargetLayout;

template <typename ELFT> class MipsELFWriter {
public:
  MipsELFWriter(MipsLinkingContext &ctx, MipsTargetLayout<ELFT> &targetLayout);

  void setELFHeader(ELFHeader<ELFT> &elfHeader);

  void finalizeMipsRuntimeAtomValues();

  std::unique_ptr<RuntimeFile<ELFT>> createRuntimeFile();

private:
  MipsLinkingContext &_ctx;
  MipsTargetLayout<ELFT> &_targetLayout;

  void setAtomValue(StringRef name, uint64_t value);
};

template <class ELFT>
class MipsDynamicLibraryWriter : public DynamicLibraryWriter<ELFT> {
public:
  MipsDynamicLibraryWriter(MipsLinkingContext &ctx,
                           MipsTargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override;
  void createDefaultSections() override;

  std::error_code setELFHeader() override;

  unique_bump_ptr<SymbolTable<ELFT>> createSymbolTable() override;
  unique_bump_ptr<DynamicTable<ELFT>> createDynamicTable() override;
  unique_bump_ptr<DynamicSymbolTable<ELFT>> createDynamicSymbolTable() override;

private:
  MipsELFWriter<ELFT> _writeHelper;
  MipsTargetLayout<ELFT> &_targetLayout;
  unique_bump_ptr<Section<ELFT>> _reginfo;
};

template <class ELFT>
class MipsExecutableWriter : public ExecutableWriter<ELFT> {
public:
  MipsExecutableWriter(MipsLinkingContext &ctx, MipsTargetLayout<ELFT> &layout);

protected:
  void buildDynamicSymbolTable(const File &file) override;

  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override;
  void createDefaultSections() override;
  std::error_code setELFHeader() override;

  unique_bump_ptr<SymbolTable<ELFT>> createSymbolTable() override;
  unique_bump_ptr<DynamicTable<ELFT>> createDynamicTable() override;
  unique_bump_ptr<DynamicSymbolTable<ELFT>> createDynamicSymbolTable() override;

private:
  MipsELFWriter<ELFT> _writeHelper;
  MipsTargetLayout<ELFT> &_targetLayout;
  unique_bump_ptr<Section<ELFT>> _reginfo;
};

} // elf
} // lld

#endif
