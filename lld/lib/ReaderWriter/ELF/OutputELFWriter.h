//===- lib/ReaderWriter/ELF/OutputELFWriter.h ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_OUTPUT_WRITER_H
#define LLD_READER_WRITER_ELF_OUTPUT_WRITER_H

#include "ELFFile.h"
#include "TargetLayout.h"
#include "lld/Core/Writer.h"
#include "llvm/ADT/StringSet.h"

namespace lld {
class ELFLinkingContext;

namespace elf {
using namespace llvm;
using namespace llvm::object;

//  OutputELFWriter Class
//
/// \brief This acts as the base class for all the ELF writers that are output
/// for emitting an ELF output file. This class also acts as a common class for
/// creating static and dynamic executables. All the function in this class
/// can be overridden and an appropriate writer be created
template<class ELFT>
class OutputELFWriter : public ELFWriter {
public:
  typedef Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef Elf_Dyn_Impl<ELFT> Elf_Dyn;

  OutputELFWriter(ELFLinkingContext &ctx, TargetLayout<ELFT> &layout);

protected:
  // build the sections that need to be created
  virtual void createDefaultSections();

  // Build all the output sections
  void buildChunks(const File &file) override;

  // Build the output file
  virtual std::error_code buildOutput(const File &file);

  // Setup the ELF header.
  virtual std::error_code setELFHeader();

  // Write the file to the path specified
  std::error_code writeFile(const File &File, StringRef path) override;

  // Write to the output file.
  virtual std::error_code writeOutput(const File &file, StringRef path);

  // Get the size of the output file that the linker would emit.
  virtual uint64_t outputFileSize() const;

  // Build the atom to address map, this has to be called
  // before applying relocations
  virtual void buildAtomToAddressMap(const File &file);

  // Build the symbol table for static linking
  virtual void buildStaticSymbolTable(const File &file);

  // Build the dynamic symbol table for dynamic linking
  virtual void buildDynamicSymbolTable(const File &file);

  // Build the section header table
  virtual void buildSectionHeaderTable();

  // Assign sections that have no segments such as the symbol table,
  // section header table, string table etc
  virtual void assignSectionsWithNoSegments();

  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  // Finalize the default atom values
  virtual void finalizeDefaultAtomValues();

  // This is called by the write section to apply relocations
  uint64_t addressOfAtom(const Atom *atom) override {
    auto addr = _atomToAddressMap.find(atom);
    return addr == _atomToAddressMap.end() ? 0 : addr->second;
  }

  // This is a hook for creating default dynamic entries
  virtual void createDefaultDynamicEntries() {}

  /// \brief Create symbol table.
  virtual unique_bump_ptr<SymbolTable<ELFT>> createSymbolTable();

  /// \brief create dynamic table.
  virtual unique_bump_ptr<DynamicTable<ELFT>> createDynamicTable();

  /// \brief create dynamic symbol table.
  virtual unique_bump_ptr<DynamicSymbolTable<ELFT>>
      createDynamicSymbolTable();

  /// \brief Create entry in the dynamic symbols table for this atom.
  virtual bool isDynSymEntryRequired(const SharedLibraryAtom *sla) const {
    return _layout.isReferencedByDefinedAtom(sla);
  }

  /// \brief Create DT_NEEDED dynamic tage for the shared library.
  virtual bool isNeededTagRequired(const SharedLibraryAtom *sla) const {
    return false;
  }

  /// \brief Process undefined symbols that left after resolution step.
  virtual void processUndefinedSymbol(StringRef symName,
                                      RuntimeFile<ELFT> &file) const {}

  /// \brief Assign addresses to atoms marking section's start and end.
  void updateScopeAtomValues(StringRef sym, StringRef sec);

  llvm::BumpPtrAllocator _alloc;

  ELFLinkingContext &_ctx;
  TargetHandler &_targetHandler;

  typedef llvm::DenseMap<const Atom *, uint64_t> AtomToAddress;
  AtomToAddress _atomToAddressMap;
  TargetLayout<ELFT> &_layout;
  unique_bump_ptr<ELFHeader<ELFT>> _elfHeader;
  unique_bump_ptr<ProgramHeader<ELFT>> _programHeader;
  unique_bump_ptr<SymbolTable<ELFT>> _symtab;
  unique_bump_ptr<StringTable<ELFT>> _strtab;
  unique_bump_ptr<StringTable<ELFT>> _shstrtab;
  unique_bump_ptr<SectionHeader<ELFT>> _shdrtab;
  unique_bump_ptr<EHFrameHeader<ELFT>> _ehFrameHeader;
  /// \name Dynamic sections.
  /// @{
  unique_bump_ptr<DynamicTable<ELFT>> _dynamicTable;
  unique_bump_ptr<DynamicSymbolTable<ELFT>> _dynamicSymbolTable;
  unique_bump_ptr<StringTable<ELFT>> _dynamicStringTable;
  unique_bump_ptr<HashSection<ELFT>> _hashTable;
  llvm::StringSet<> _soNeeded;
  /// @}

private:
  static StringRef maybeGetSOName(Node *node);
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_OUTPUT_WRITER_H
