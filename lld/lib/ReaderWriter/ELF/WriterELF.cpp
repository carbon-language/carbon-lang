//===- lib/ReaderWriter/ELF/WriterELF.cpp ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "DefaultELFLayout.h"
#include "ExecutableAtoms.h"

using namespace llvm;
using namespace llvm::object;
namespace lld {
namespace elf {
template<class ELFT>
class ELFExecutableWriter;

//===----------------------------------------------------------------------===//
//  ELFExecutableWriter Class
//===----------------------------------------------------------------------===//
template<class ELFT>
class ELFExecutableWriter : public ELFWriter {
public:
  typedef Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef Elf_Sym_Impl<ELFT> Elf_Sym;

  ELFExecutableWriter(const ELFTargetInfo &ti);

private:
  // build the sections that need to be created
  void buildChunks(const lld::File &file);
  virtual error_code writeFile(const lld::File &File, StringRef path);
  void buildAtomToAddressMap();
  void buildSymbolTable ();
  void buildSectionHeaderTable();
  void assignSectionsWithNoSegments();
  void addAbsoluteUndefinedSymbols(const lld::File &File);
  void addDefaultAtoms();
  void addFiles(InputFiles&);
  void finalizeDefaultAtomValues();

  uint64_t addressOfAtom(const Atom *atom) {
    return _atomToAddressMap[atom];
  }

  KindHandler *kindHandler() { return _referenceKindHandler.get(); }

  void createDefaultSections();

  const ELFTargetInfo &_targetInfo;

  typedef llvm::DenseMap<const Atom*, uint64_t> AtomToAddress;
  std::unique_ptr<KindHandler> _referenceKindHandler;
  AtomToAddress _atomToAddressMap;
  llvm::BumpPtrAllocator _chunkAllocate;
  DefaultELFLayout<ELFT> *_layout;
  ELFHeader<ELFT> *_elfHeader;
  ELFProgramHeader<ELFT> *_programHeader;
  ELFSymbolTable<ELFT> * _symtab;
  ELFStringTable<ELFT> *_strtab;
  ELFStringTable<ELFT> *_shstrtab;
  ELFSectionHeader<ELFT> *_shdrtab;
  CRuntimeFileELF<ELFT> _runtimeFile;
};

//===----------------------------------------------------------------------===//
//  ELFExecutableWriter
//===----------------------------------------------------------------------===//
template<class ELFT>
ELFExecutableWriter<ELFT>::ELFExecutableWriter(const ELFTargetInfo &ti)
  : _targetInfo(ti)
  , _referenceKindHandler(KindHandler::makeHandler(
                              ti.getTriple().getArch(), ti.isLittleEndian()))
  , _runtimeFile(ti) {
  _layout = new DefaultELFLayout<ELFT>(ti);
}

template<class ELFT>
void ELFExecutableWriter<ELFT>::buildChunks(const lld::File &file){
  for (const DefinedAtom *definedAtom : file.defined() ) {
    _layout->addAtom(definedAtom);
  }
  /// Add all the absolute atoms to the layout
  for (const AbsoluteAtom *absoluteAtom : file.absolute()) {
    _layout->addAtom(absoluteAtom);
  }
}

template<class ELFT>
void ELFExecutableWriter<ELFT>::buildSymbolTable () {
  for (auto sec : _layout->sections())
    if (auto section = dyn_cast<Section<ELFT>>(sec))
      for (const auto &atom : section->atoms())
        _symtab->addSymbol(atom._atom, section->ordinal(), atom._virtualAddr);
}

template<class ELFT>
void
ELFExecutableWriter<ELFT>::addAbsoluteUndefinedSymbols(const lld::File &file) {
  // add all the absolute symbols that the layout contains to the output symbol
  // table
  for (auto &atom : _layout->absoluteAtoms())
    _symtab->addSymbol(atom.absoluteAtom(), ELF::SHN_ABS, atom.value());
  for (const UndefinedAtom *a : file.undefined())
    _symtab->addSymbol(a, ELF::SHN_UNDEF);
}

template<class ELFT>
void ELFExecutableWriter<ELFT>::buildAtomToAddressMap () {
  for (auto sec : _layout->sections())
    if (auto section = dyn_cast<Section<ELFT>>(sec))
      for (const auto &atom : section->atoms())
        _atomToAddressMap[atom._atom] = atom._virtualAddr;
  // build the atomToAddressMap that contains absolute symbols too
  for (auto &atom : _layout->absoluteAtoms())
    _atomToAddressMap[atom.absoluteAtom()] = atom.value();
}

template<class ELFT>
void ELFExecutableWriter<ELFT>::buildSectionHeaderTable() {
  for (auto mergedSec : _layout->mergedSections()) {
    if (mergedSec->kind() != Chunk<ELFT>::K_ELFSection)
      continue;
    if (mergedSec->hasSegment())
      _shdrtab->appendSection(mergedSec);
  }
}

template<class ELFT>
void ELFExecutableWriter<ELFT>::assignSectionsWithNoSegments() {
  for (auto mergedSec : _layout->mergedSections()) {
    if (mergedSec->kind() != Chunk<ELFT>::K_ELFSection)
      continue;
    if (!mergedSec->hasSegment())
      _shdrtab->appendSection(mergedSec);
  }
  _layout->assignOffsetsForMiscSections();
  for (auto sec : _layout->sections())
    if (auto section = dyn_cast<Section<ELFT>>(sec))
      if (!DefaultELFLayout<ELFT>::hasOutputSegment(section))
        _shdrtab->updateSection(section);
}

/// \brief Add absolute symbols by default. These are linker added
/// absolute symbols
template<class ELFT>
void ELFExecutableWriter<ELFT>::addDefaultAtoms() {
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
template<class ELFT>
void ELFExecutableWriter<ELFT>::addFiles(InputFiles &inputFiles) {
  addDefaultAtoms();
  inputFiles.prependFile(_runtimeFile);
}

/// Finalize the value of all the absolute symbols that we 
/// created
template<class ELFT>
void ELFExecutableWriter<ELFT>::finalizeDefaultAtomValues() {
  auto bssStartAtomIter = _layout->findAbsoluteAtom("__bss_start");
  auto bssEndAtomIter = _layout->findAbsoluteAtom("__bss_end");
  auto underScoreEndAtomIter = _layout->findAbsoluteAtom("_end");
  auto endAtomIter = _layout->findAbsoluteAtom("end");
  auto initArrayStartIter = _layout->findAbsoluteAtom("__init_array_start");
  auto initArrayEndIter = _layout->findAbsoluteAtom("__init_array_end");
  auto realIpltStartIter = _layout->findAbsoluteAtom("__rela_iplt_start");
  auto realIpltEndIter = _layout->findAbsoluteAtom("__rela_iplt_end");

  auto startEnd = [&](typename DefaultELFLayout<ELFT>::AbsoluteAtomIterT start,
                      typename DefaultELFLayout<ELFT>::AbsoluteAtomIterT end,
                      StringRef sec) -> void {
    auto section = _layout->findOutputSection(sec);
    if (section) {
      start->setValue(section->virtualAddr());
      end->setValue(section->virtualAddr() + section->memSize());
    } else {
      start->setValue(0);
      end->setValue(0);
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

  bssStartAtomIter->setValue((*phe)->p_vaddr + (*phe)->p_filesz);
  bssEndAtomIter->setValue((*phe)->p_vaddr + (*phe)->p_memsz);
  underScoreEndAtomIter->setValue((*phe)->p_vaddr + (*phe)->p_memsz);
  endAtomIter->setValue((*phe)->p_vaddr + (*phe)->p_memsz);
}

template<class ELFT>
error_code
ELFExecutableWriter<ELFT>::writeFile(const lld::File &file, StringRef path) {
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

  _elfHeader->e_ident(ELF::EI_CLASS, _targetInfo.is64Bits() ? ELF::ELFCLASS64
                                                            : ELF::ELFCLASS32);
  _elfHeader->e_ident(ELF::EI_DATA, _targetInfo.isLittleEndian()
                                    ? ELF::ELFDATA2LSB : ELF::ELFDATA2MSB);
  _elfHeader->e_ident(ELF::EI_VERSION, 1);
  _elfHeader->e_ident(ELF::EI_OSABI, 0);
  _elfHeader->e_type(_targetInfo.getOutputType());
  _elfHeader->e_machine(_targetInfo.getOutputMachine());
  _elfHeader->e_version(1);
  _elfHeader->e_entry(0ULL);
  _elfHeader->e_phoff(_programHeader->fileOffset());
  _elfHeader->e_shoff(_shdrtab->fileOffset());
  _elfHeader->e_phentsize(_programHeader->entsize());
  _elfHeader->e_phnum(_programHeader->numHeaders());
  _elfHeader->e_shentsize(_shdrtab->entsize());
  _elfHeader->e_shnum(_shdrtab->numHeaders());
  _elfHeader->e_shstrndx(_shstrtab->ordinal());
  uint64_t virtualAddr = 0;
  _layout->findAtomAddrByName(_targetInfo.getEntry(), virtualAddr);
  _elfHeader->e_entry(virtualAddr);

  // HACK: We have to write out the header and program header here even though
  // they are a member of a segment because only sections are written in the
  // following loop.
  _elfHeader->write(this, *buffer);
  _programHeader->write(this, *buffer);

  for (auto section : _layout->sections())
    section->write(this, *buffer);

  return buffer->commit();
}

template<class ELFT>
void ELFExecutableWriter<ELFT>::createDefaultSections() {
  _elfHeader = new ELFHeader<ELFT>(_targetInfo);
  _programHeader = new ELFProgramHeader<ELFT>(_targetInfo);
  _layout->setELFHeader(_elfHeader);
  _layout->setProgramHeader(_programHeader);

  _symtab = new ELFSymbolTable<
      ELFT>(_targetInfo, ".symtab", DefaultELFLayout<ELFT>::ORDER_SYMBOL_TABLE);
  _strtab = new ELFStringTable<
      ELFT>(_targetInfo, ".strtab", DefaultELFLayout<ELFT>::ORDER_STRING_TABLE);
  _shstrtab = new ELFStringTable<ELFT>(
      _targetInfo, ".shstrtab", DefaultELFLayout<ELFT>::ORDER_SECTION_STRINGS);
  _shdrtab = new ELFSectionHeader<
      ELFT>(_targetInfo, DefaultELFLayout<ELFT>::ORDER_SECTION_HEADERS);
  _layout->addSection(_symtab);
  _layout->addSection(_strtab);
  _layout->addSection(_shstrtab);
  _shdrtab->setStringSection(_shstrtab);
  _symtab->setStringSection(_strtab);
  _layout->addSection(_shdrtab);
}
} // namespace elf

std::unique_ptr<Writer> createWriterELF(const ELFTargetInfo &TI) {
  using llvm::object::ELFType;
  // Set the default layout to be the static executable layout
  // We would set the layout to a dynamic executable layout
  // if we came across any shared libraries in the process

  if (!TI.is64Bits() && TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ELFExecutableWriter<ELFType<support::little, 4, false>>(TI));
  else if (TI.is64Bits() && TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ELFExecutableWriter<ELFType<support::little, 8, true>>(TI));
  else if (!TI.is64Bits() && !TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ELFExecutableWriter<ELFType<support::big, 4, false>>(TI));
  else if (TI.is64Bits() && !TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ELFExecutableWriter<ELFType<support::big, 8, true>>(TI));

  llvm_unreachable("Invalid Options!");
}
} // namespace lld
