//===- lib/ReaderWriter/ELF/DynamicLibraryWriter.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_DYNAMIC_LIBRARY_WRITER_H
#define LLD_READER_WRITER_ELF_DYNAMIC_LIBRARY_WRITER_H

#include "lld/ReaderWriter/Writer.h"

#include "DefaultLayout.h"
#include "TargetLayout.h"
#include "ExecutableAtoms.h"

#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "llvm/ADT/StringSet.h"

namespace lld {
namespace elf {
using namespace llvm;
using namespace llvm::object;

template<class ELFT>
class DynamicLibraryWriter;

//===----------------------------------------------------------------------===//
//  DynamicLibraryWriter Class
//===----------------------------------------------------------------------===//
template<class ELFT>
class DynamicLibraryWriter : public ELFWriter {
public:
  typedef Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef Elf_Dyn_Impl<ELFT> Elf_Dyn;

  DynamicLibraryWriter(const ELFTargetInfo &ti);

private:
  // build the sections that need to be created
  void buildChunks(const File &file);
  virtual error_code writeFile(const File &File, StringRef path);
  void buildAtomToAddressMap();
  void buildStaticSymbolTable(const File &file);
  void buildDynamicSymbolTable(const File &file);
  void buildSectionHeaderTable();
  void assignSectionsWithNoSegments();
  void addDefaultAtoms();
  void addFiles(InputFiles&);
  void finalizeDefaultAtomValues();

  uint64_t addressOfAtom(const Atom *atom) {
    return _atomToAddressMap[atom];
  }

  void createDefaultSections();

  void createDefaultDynamicEntries() {}

  llvm::BumpPtrAllocator _alloc;

  const ELFTargetInfo &_targetInfo;
  TargetHandler<ELFT> &_targetHandler;

  typedef llvm::DenseMap<const Atom *, uint64_t> AtomToAddress;
  AtomToAddress _atomToAddressMap;
  TargetLayout<ELFT> *_layout;
  LLD_UNIQUE_BUMP_PTR(Header<ELFT>) _Header;
  LLD_UNIQUE_BUMP_PTR(ProgramHeader<ELFT>) _programHeader;
  LLD_UNIQUE_BUMP_PTR(SymbolTable<ELFT>) _symtab;
  LLD_UNIQUE_BUMP_PTR(StringTable<ELFT>) _strtab;
  LLD_UNIQUE_BUMP_PTR(StringTable<ELFT>) _shstrtab;
  LLD_UNIQUE_BUMP_PTR(SectionHeader<ELFT>) _shdrtab;
  /// \name Dynamic sections.
  /// @{
  LLD_UNIQUE_BUMP_PTR(DynamicTable<ELFT>) _dynamicTable;
  LLD_UNIQUE_BUMP_PTR(DynamicSymbolTable<ELFT>) _dynamicSymbolTable;
  LLD_UNIQUE_BUMP_PTR(StringTable<ELFT>) _dynamicStringTable;
  LLD_UNIQUE_BUMP_PTR(InterpSection<ELFT>) _interpSection;
  LLD_UNIQUE_BUMP_PTR(HashSection<ELFT>) _hashTable;
  llvm::StringSet<> _soNeeded;
  /// @}
  CRuntimeFile<ELFT> _runtimeFile;
};

//===----------------------------------------------------------------------===//
//  DynamicLibraryWriter
//===----------------------------------------------------------------------===//
template <class ELFT>
DynamicLibraryWriter<ELFT>::DynamicLibraryWriter(const ELFTargetInfo &ti)
    : _targetInfo(ti), _targetHandler(ti.getTargetHandler<ELFT>()),
      _runtimeFile(ti) {
  _layout = &_targetHandler.targetLayout();
}

template <class ELFT>
void DynamicLibraryWriter<ELFT>::buildChunks(const File &file) {
  for (const DefinedAtom *definedAtom : file.defined()) {
    _layout->addAtom(definedAtom);
  }
  for (const AbsoluteAtom *absoluteAtom : file.absolute())
    _layout->addAtom(absoluteAtom);
}

template <class ELFT>
void DynamicLibraryWriter<ELFT>::buildStaticSymbolTable(const File &file) {
  for (auto sec : _layout->sections())
    if (auto section = dyn_cast<AtomSection<ELFT>>(sec))
      for (const auto &atom : section->atoms())
        _symtab->addSymbol(atom->_atom, section->ordinal(), atom->_virtualAddr);
  for (auto &atom : _layout->absoluteAtoms())
    _symtab->addSymbol(atom->_atom, ELF::SHN_ABS, atom->_virtualAddr);
  for (const UndefinedAtom *a : file.undefined())
    _symtab->addSymbol(a, ELF::SHN_UNDEF);
}

template <class ELFT>
void DynamicLibraryWriter<ELFT>::buildDynamicSymbolTable(const File &file) {
  for (const auto sla : file.sharedLibrary()) {
    _dynamicSymbolTable->addSymbol(sla, ELF::SHN_UNDEF);
    _soNeeded.insert(sla->loadName());
  }
  // Add all the defined symbols to the dynamic symbol table
  // we need hooks into the Atom to find out which atoms need 
  // to be exported 
  for (auto sec : _layout->sections())
    if (auto section = dyn_cast<AtomSection<ELFT>>(sec))
      for (const auto &atom : section->atoms()) {
        const DefinedAtom *da = dyn_cast<const DefinedAtom>(atom->_atom);
        if (da && (da->scope() != DefinedAtom::scopeTranslationUnit)) 
          _dynamicSymbolTable->addSymbol(atom->_atom, section->ordinal(),
                                         atom->_virtualAddr, atom);
      }
  for (const auto &loadName : _soNeeded) {
    Elf_Dyn dyn;
    dyn.d_tag = DT_NEEDED;
    dyn.d_un.d_val = _dynamicStringTable->addString(loadName.getKey());
    _dynamicTable->addEntry(dyn);
  }
}

template <class ELFT> void DynamicLibraryWriter<ELFT>::buildAtomToAddressMap() {
  for (auto sec : _layout->sections())
    if (auto section = dyn_cast<AtomSection<ELFT>>(sec))
      for (const auto &atom : section->atoms())
        _atomToAddressMap[atom->_atom] = atom->_virtualAddr;
  // build the atomToAddressMap that contains absolute symbols too
  for (auto &atom : _layout->absoluteAtoms())
    _atomToAddressMap[atom->_atom] = atom->_virtualAddr;
}

template<class ELFT>
void DynamicLibraryWriter<ELFT>::buildSectionHeaderTable() {
  for (auto mergedSec : _layout->mergedSections()) {
    if (mergedSec->kind() != Chunk<ELFT>::K_ELFSection &&
        mergedSec->kind() != Chunk<ELFT>::K_AtomSection)
      continue;
    if (mergedSec->hasSegment())
      _shdrtab->appendSection(mergedSec);
  }
}

template<class ELFT>
void DynamicLibraryWriter<ELFT>::assignSectionsWithNoSegments() {
  for (auto mergedSec : _layout->mergedSections()) {
    if (mergedSec->kind() != Chunk<ELFT>::K_ELFSection &&
        mergedSec->kind() != Chunk<ELFT>::K_AtomSection)
      continue;
    if (!mergedSec->hasSegment())
      _shdrtab->appendSection(mergedSec);
  }
  _layout->assignOffsetsForMiscSections();
  for (auto sec : _layout->sections())
    if (auto section = dyn_cast<Section<ELFT>>(sec))
      if (!DefaultLayout<ELFT>::hasOutputSegment(section))
        _shdrtab->updateSection(section);
}

/// \brief Add absolute symbols by default. These are linker added
/// absolute symbols
template<class ELFT>
void DynamicLibraryWriter<ELFT>::addDefaultAtoms() {
  _runtimeFile.addAbsoluteAtom("__bss_start");
}

/// \brief Hook in lld to add CRuntime file 
template <class ELFT>
void DynamicLibraryWriter<ELFT>::addFiles(InputFiles &inputFiles) {
  addDefaultAtoms();
  inputFiles.prependFile(_runtimeFile);
  // Give a chance for the target to add atoms
  _targetHandler.addFiles(inputFiles);
}

/// Finalize the value of all the absolute symbols that we 
/// created
template<class ELFT>
void DynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues() {
  auto bssStartAtomIter = _layout->findAbsoluteAtom("__bss_start");

  assert(!(bssStartAtomIter == _layout->absoluteAtoms().end()) &&
         "Unable to find the absolute atoms that have been added by lld");

  auto phe = _programHeader->findProgramHeader(
      llvm::ELF::PT_LOAD, llvm::ELF::PF_W, llvm::ELF::PF_X);

  (*bssStartAtomIter)->_virtualAddr = (*phe)->p_vaddr + (*phe)->p_filesz;

  // Give a chance for the target to finalize its atom values
  _targetHandler.finalizeSymbolValues();
}

template <class ELFT>
error_code DynamicLibraryWriter<ELFT>::writeFile(const File &file, StringRef path) {
  buildChunks(file);

  // Call the preFlight callbacks to modify the sections and the atoms 
  // contained in them, in anyway the targets may want
  _layout->doPreFlight();

  // Create the default sections like the symbol table, string table, and the
  // section string table
  createDefaultSections();

  _dynamicTable->createDefaultEntries();
  buildDynamicSymbolTable(file);

  // Set the Layout
  _layout->assignSectionsToSegments();
  _layout->assignFileOffsets();
  _layout->assignVirtualAddress();

  // Finalize the default value of symbols that the linker adds
  finalizeDefaultAtomValues();

  // Build the Atom To Address map for applying relocations
  buildAtomToAddressMap();

  // Create symbol table and section string table
  buildStaticSymbolTable(file);

  // Finalize the layout by calling the finalize() functions
  _layout->finalize();

  // build Section Header table
  buildSectionHeaderTable();

  // assign Offsets and virtual addresses
  // for sections with no segments
  assignSectionsWithNoSegments();

  _dynamicTable->updateDynamicTable(_hashTable.get(),
                                    _dynamicSymbolTable.get());

  uint64_t totalSize = _shdrtab->fileOffset() + _shdrtab->fileSize();

  OwningPtr<FileOutputBuffer> buffer;
  error_code ec = FileOutputBuffer::create(path,
                                           totalSize, buffer,
                                           FileOutputBuffer::F_executable);
  if (ec)
    return ec;

  _Header->e_ident(ELF::EI_CLASS, _targetInfo.is64Bits() ? ELF::ELFCLASS64 :
                       ELF::ELFCLASS32);
  _Header->e_ident(ELF::EI_DATA, _targetInfo.isLittleEndian() ?
                       ELF::ELFDATA2LSB : ELF::ELFDATA2MSB);
  _Header->e_type(_targetInfo.getOutputType());
  _Header->e_machine(_targetInfo.getOutputMachine());

  if (!_targetHandler.doesOverrideHeader()) {
    _Header->e_ident(ELF::EI_VERSION, 1);
    _Header->e_ident(ELF::EI_OSABI, 0);
    _Header->e_version(1);
  } else {
    // override the contents of the ELF Header
    _targetHandler.setHeaderInfo(_Header.get());
  }
  _Header->e_phoff(_programHeader->fileOffset());
  _Header->e_shoff(_shdrtab->fileOffset());
  _Header->e_phentsize(_programHeader->entsize());
  _Header->e_phnum(_programHeader->numHeaders());
  _Header->e_shentsize(_shdrtab->entsize());
  _Header->e_shnum(_shdrtab->numHeaders());
  _Header->e_shstrndx(_shstrtab->ordinal());
  uint64_t virtualAddr = 0;
  _layout->findAtomAddrByName(_targetInfo.getEntry(), virtualAddr);

  // HACK: We have to write out the header and program header here even though
  // they are a member of a segment because only sections are written in the
  // following loop.
  _Header->write(this, *buffer);
  _programHeader->write(this, *buffer);

  for (auto section : _layout->sections())
    section->write(this, *buffer);

  return buffer->commit();
}

template<class ELFT>
void DynamicLibraryWriter<ELFT>::createDefaultSections() {
  _Header.reset(new (_alloc) Header<ELFT>(_targetInfo));
  _programHeader.reset(new (_alloc) ProgramHeader<ELFT>(_targetInfo));
  _layout->setHeader(_Header.get());
  _layout->setProgramHeader(_programHeader.get());

  _symtab.reset(new (_alloc) SymbolTable<ELFT>(
      _targetInfo, ".symtab", DefaultLayout<ELFT>::ORDER_SYMBOL_TABLE));
  _strtab.reset(new (_alloc) StringTable<ELFT>(
      _targetInfo, ".strtab", DefaultLayout<ELFT>::ORDER_STRING_TABLE));
  _shstrtab.reset(new (_alloc) StringTable<ELFT>(
      _targetInfo, ".shstrtab", DefaultLayout<ELFT>::ORDER_SECTION_STRINGS));
  _shdrtab.reset(new (_alloc) SectionHeader<ELFT>(
      _targetInfo, DefaultLayout<ELFT>::ORDER_SECTION_HEADERS));
  _layout->addSection(_symtab.get());
  _layout->addSection(_strtab.get());
  _layout->addSection(_shstrtab.get());
  _shdrtab->setStringSection(_shstrtab.get());
  _symtab->setStringSection(_strtab.get());
  _layout->addSection(_shdrtab.get());

  _dynamicTable.reset(new (_alloc) DynamicTable<ELFT>(
      _targetInfo, ".dynamic", DefaultLayout<ELFT>::ORDER_DYNAMIC));
  _dynamicStringTable.reset(new (_alloc) StringTable<ELFT>(
      _targetInfo, ".dynstr", DefaultLayout<ELFT>::ORDER_DYNAMIC_STRINGS,
      true));
  _dynamicSymbolTable.reset(new (_alloc) DynamicSymbolTable<ELFT>(
      _targetInfo, ".dynsym", DefaultLayout<ELFT>::ORDER_DYNAMIC_SYMBOLS));
  _interpSection.reset(new (_alloc) InterpSection<ELFT>(
      _targetInfo, ".interp", DefaultLayout<ELFT>::ORDER_INTERP,
      _targetInfo.getInterpreter()));
  _hashTable.reset(new (_alloc) HashSection<ELFT>(
      _targetInfo, ".hash", DefaultLayout<ELFT>::ORDER_HASH));
  _layout->addSection(_dynamicTable.get());
  _layout->addSection(_dynamicStringTable.get());
  _layout->addSection(_dynamicSymbolTable.get());
  _layout->addSection(_interpSection.get());
  _layout->addSection(_hashTable.get());
  _dynamicSymbolTable->setStringSection(_dynamicStringTable.get());
  if (_layout->hasDynamicRelocationTable())
    _layout->getDynamicRelocationTable()->setSymbolTable(
        _dynamicSymbolTable.get());
  if (_layout->hasPLTRelocationTable())
    _layout->getPLTRelocationTable()->setSymbolTable(
        _dynamicSymbolTable.get());

  // give a chance for the target to add sections
  _targetHandler.createDefaultSections();
}
} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_DYNAMIC_LIBRARY_WRITER_H
