//===- lib/ReaderWriter/ELF/WriterELF.cpp ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Writer.h"

#include "DefaultLayout.h"
#include "TargetLayout.h"
#include "ExecutableAtoms.h"

#include "lld/ReaderWriter/ELFTargetInfo.h"

using namespace llvm;
using namespace llvm::object;
namespace lld {
namespace elf {
template<class ELFT>
class ExecutableWriter;

//===----------------------------------------------------------------------===//
//  ExecutableWriter Class
//===----------------------------------------------------------------------===//
template<class ELFT>
class ExecutableWriter : public ELFWriter {
public:
  typedef Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef Elf_Sym_Impl<ELFT> Elf_Sym;

  ExecutableWriter(const ELFTargetInfo &ti);

private:
  // build the sections that need to be created
  void buildChunks(const File &file);
  virtual error_code writeFile(const File &File, StringRef path);
  void buildAtomToAddressMap();
  void buildSymbolTable ();
  void buildSectionHeaderTable();
  void assignSectionsWithNoSegments();
  void addAbsoluteUndefinedSymbols(const File &File);
  void addDefaultAtoms();
  void addFiles(InputFiles&);
  void finalizeDefaultAtomValues();

  uint64_t addressOfAtom(const Atom *atom) {
    return _atomToAddressMap[atom];
  }

  void createDefaultSections();

  const ELFTargetInfo &_targetInfo;
  TargetHandler<ELFT> &_targetHandler;

  typedef llvm::DenseMap<const Atom *, uint64_t> AtomToAddress;
  AtomToAddress _atomToAddressMap;
  llvm::BumpPtrAllocator _chunkAllocate;
  TargetLayout<ELFT> *_layout;
  Header<ELFT> *_Header;
  ProgramHeader<ELFT> *_programHeader;
  SymbolTable<ELFT> * _symtab;
  StringTable<ELFT> *_strtab;
  StringTable<ELFT> *_shstrtab;
  SectionHeader<ELFT> *_shdrtab;
  CRuntimeFile<ELFT> _runtimeFile;
};

//===----------------------------------------------------------------------===//
//  ExecutableWriter
//===----------------------------------------------------------------------===//
template <class ELFT>
ExecutableWriter<ELFT>::ExecutableWriter(const ELFTargetInfo &ti)
    : _targetInfo(ti), _targetHandler(ti.getTargetHandler<ELFT>()),
      _runtimeFile(ti) {
  _layout = &_targetHandler.targetLayout();
}

template <class ELFT>
void ExecutableWriter<ELFT>::buildChunks(const File &file) {
  for (const DefinedAtom *definedAtom : file.defined() ) {
    _layout->addAtom(definedAtom);
  }
  /// Add all the absolute atoms to the layout
  for (const AbsoluteAtom *absoluteAtom : file.absolute()) {
    _layout->addAtom(absoluteAtom);
  }
}

template<class ELFT>
void ExecutableWriter<ELFT>::buildSymbolTable () {
  for (auto sec : _layout->sections())
    if (auto section = dyn_cast<Section<ELFT>>(sec))
      for (const auto &atom : section->atoms())
        _symtab->addSymbol(atom->_atom, section->ordinal(), atom->_virtualAddr);
}

template<class ELFT>
void
ExecutableWriter<ELFT>::addAbsoluteUndefinedSymbols(const File &file) {
  // add all the absolute symbols that the layout contains to the output symbol
  // table
  for (auto &atom : _layout->absoluteAtoms())
    _symtab->addSymbol(atom->_atom, ELF::SHN_ABS, atom->_virtualAddr);
  for (const UndefinedAtom *a : file.undefined())
    _symtab->addSymbol(a, ELF::SHN_UNDEF);
}

template<class ELFT>
void ExecutableWriter<ELFT>::buildAtomToAddressMap () {
  for (auto sec : _layout->sections())
    if (auto section = dyn_cast<Section<ELFT>>(sec))
      for (const auto &atom : section->atoms())
        _atomToAddressMap[atom->_atom] = atom->_virtualAddr;
  // build the atomToAddressMap that contains absolute symbols too
  for (auto &atom : _layout->absoluteAtoms())
    _atomToAddressMap[atom->_atom] = atom->_virtualAddr;
}

template<class ELFT>
void ExecutableWriter<ELFT>::buildSectionHeaderTable() {
  for (auto mergedSec : _layout->mergedSections()) {
    if (mergedSec->kind() != Chunk<ELFT>::K_ELFSection)
      continue;
    if (mergedSec->hasSegment())
      _shdrtab->appendSection(mergedSec);
  }
}

template<class ELFT>
void ExecutableWriter<ELFT>::assignSectionsWithNoSegments() {
  for (auto mergedSec : _layout->mergedSections()) {
    if (mergedSec->kind() != Chunk<ELFT>::K_ELFSection)
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
void ExecutableWriter<ELFT>::addDefaultAtoms() {
  _runtimeFile.addUndefinedAtom(_targetInfo.getEntry());
  _runtimeFile.addAbsoluteAtom("__bss_start");
  _runtimeFile.addAbsoluteAtom("__bss_end");
  _runtimeFile.addAbsoluteAtom("_end");
  _runtimeFile.addAbsoluteAtom("end");
  _runtimeFile.addAbsoluteAtom("__init_array_start");
  _runtimeFile.addAbsoluteAtom("__init_array_end");
  _runtimeFile.addAbsoluteAtom("__rela_iplt_start");
  _runtimeFile.addAbsoluteAtom("__rela_iplt_end");
}

/// \brief Hook in lld to add CRuntime file 
template <class ELFT>
void ExecutableWriter<ELFT>::addFiles(InputFiles &inputFiles) {
  addDefaultAtoms();
  inputFiles.prependFile(_runtimeFile);
  // Give a chance for the target to add atoms
  _targetHandler.addFiles(inputFiles);
}

/// Finalize the value of all the absolute symbols that we 
/// created
template<class ELFT>
void ExecutableWriter<ELFT>::finalizeDefaultAtomValues() {
  auto bssStartAtomIter = _layout->findAbsoluteAtom("__bss_start");
  auto bssEndAtomIter = _layout->findAbsoluteAtom("__bss_end");
  auto underScoreEndAtomIter = _layout->findAbsoluteAtom("_end");
  auto endAtomIter = _layout->findAbsoluteAtom("end");
  auto initArrayStartIter = _layout->findAbsoluteAtom("__init_array_start");
  auto initArrayEndIter = _layout->findAbsoluteAtom("__init_array_end");
  auto realIpltStartIter = _layout->findAbsoluteAtom("__rela_iplt_start");
  auto realIpltEndIter = _layout->findAbsoluteAtom("__rela_iplt_end");

  auto startEnd = [&](typename DefaultLayout<ELFT>::AbsoluteAtomIterT start,
                      typename DefaultLayout<ELFT>::AbsoluteAtomIterT end,
                      StringRef sec) -> void {
    auto section = _layout->findOutputSection(sec);
    if (section) {
      (*start)->_virtualAddr = section->virtualAddr();
      (*end)->_virtualAddr = section->virtualAddr() + section->memSize();
    } else {
      (*start)->_virtualAddr = 0;
      (*end)->_virtualAddr = 0;
    }
  };

  startEnd(initArrayStartIter, initArrayEndIter, ".init_array");
  startEnd(realIpltStartIter, realIpltEndIter, ".rela.plt");

  assert(!(bssStartAtomIter == _layout->absoluteAtoms().end() ||
           bssEndAtomIter == _layout->absoluteAtoms().end() ||
           underScoreEndAtomIter == _layout->absoluteAtoms().end() ||
           endAtomIter == _layout->absoluteAtoms().end()) &&
         "Unable to find the absolute atoms that have been added by lld");

  auto phe = _programHeader->findProgramHeader(
      llvm::ELF::PT_LOAD, llvm::ELF::PF_W, llvm::ELF::PF_X);

  assert(!(phe == _programHeader->end()) &&
         "Can't find a data segment in the program header!");

  (*bssStartAtomIter)->_virtualAddr = (*phe)->p_vaddr + (*phe)->p_filesz;
  (*bssEndAtomIter)->_virtualAddr = (*phe)->p_vaddr + (*phe)->p_memsz;
  (*underScoreEndAtomIter)->_virtualAddr = (*phe)->p_vaddr + (*phe)->p_memsz;
  (*endAtomIter)->_virtualAddr = (*phe)->p_vaddr + (*phe)->p_memsz;
}

template<class ELFT>
error_code
ExecutableWriter<ELFT>::writeFile(const File &file, StringRef path) {
  buildChunks(file);
  // Create the default sections like the symbol table, string table, and the
  // section string table
  createDefaultSections();

  // Set the Layout
  _layout->assignSectionsToSegments();
  _layout->assignFileOffsets();
  _layout->assignVirtualAddress();

  // Finalize the default value of symbols that the linker adds
  finalizeDefaultAtomValues();

  // Build the Atom To Address map for applying relocations
  buildAtomToAddressMap();

  // Create symbol table and section string table
  buildSymbolTable();

  // add other symbols
  addAbsoluteUndefinedSymbols(file);

  // Finalize the layout by calling the finalize() functions
  _layout->finalize();

  // build Section Header table
  buildSectionHeaderTable();

  // assign Offsets and virtual addresses
  // for sections with no segments
  assignSectionsWithNoSegments();

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
    _targetHandler.setHeaderInfo(_Header);
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
  _Header->e_entry(virtualAddr);

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
void ExecutableWriter<ELFT>::createDefaultSections() {
  _Header = new Header<ELFT>(_targetInfo);
  _programHeader = new ProgramHeader<ELFT>(_targetInfo);
  _layout->setHeader(_Header);
  _layout->setProgramHeader(_programHeader);

  _symtab = new SymbolTable<
      ELFT>(_targetInfo, ".symtab", DefaultLayout<ELFT>::ORDER_SYMBOL_TABLE);
  _strtab = new StringTable<
      ELFT>(_targetInfo, ".strtab", DefaultLayout<ELFT>::ORDER_STRING_TABLE);
  _shstrtab = new StringTable<ELFT>(
      _targetInfo, ".shstrtab", DefaultLayout<ELFT>::ORDER_SECTION_STRINGS);
  _shdrtab = new SectionHeader<
      ELFT>(_targetInfo, DefaultLayout<ELFT>::ORDER_SECTION_HEADERS);
  _layout->addSection(_symtab);
  _layout->addSection(_strtab);
  _layout->addSection(_shstrtab);
  _shdrtab->setStringSection(_shstrtab);
  _symtab->setStringSection(_strtab);
  _layout->addSection(_shdrtab);

  // give a chance for the target to add sections
  _targetHandler.createDefaultSections();
}
} // namespace elf

std::unique_ptr<Writer> createWriterELF(const ELFTargetInfo &TI) {
  using llvm::object::ELFType;
  // Set the default layout to be the static executable layout
  // We would set the layout to a dynamic executable layout
  // if we came across any shared libraries in the process

  if (!TI.is64Bits() && TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ExecutableWriter<ELFType<support::little, 4, false>>(TI));
  else if (TI.is64Bits() && TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ExecutableWriter<ELFType<support::little, 8, true>>(TI));
  else if (!TI.is64Bits() && !TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ExecutableWriter<ELFType<support::big, 4, false>>(TI));
  else if (TI.is64Bits() && !TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ExecutableWriter<ELFType<support::big, 8, true>>(TI));

  llvm_unreachable("Invalid Options!");
}
} // namespace lld
