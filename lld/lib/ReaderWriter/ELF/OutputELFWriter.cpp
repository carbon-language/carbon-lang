//===- lib/ReaderWriter/ELF/OutputELFWriter.cpp --------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OutputELFWriter.h"
#include "lld/Core/SharedLibraryFile.h"
#include "lld/Core/Simple.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/Support/Path.h"

namespace lld {
namespace elf {

namespace {

template <class ELFT> class SymbolFile : public RuntimeFile<ELFT> {
public:
  SymbolFile(ELFLinkingContext &ctx)
      : RuntimeFile<ELFT>(ctx, "Dynamic absolute symbols") {}

  void addUndefinedAtom(StringRef) override {
    llvm_unreachable("Cannot add undefined atoms to resolve undefined symbols");
  }

  bool hasAtoms() const { return this->absolute().size(); }
};

template <class ELFT>
class DynamicSymbolFile : public SimpleArchiveLibraryFile {
  typedef std::function<void(StringRef, RuntimeFile<ELFT> &)> Resolver;

public:
  DynamicSymbolFile(ELFLinkingContext &ctx, Resolver resolver)
      : SimpleArchiveLibraryFile("Dynamically added runtime symbols"),
        _ctx(ctx), _resolver(resolver) {}

  File *find(StringRef sym, bool dataSymbolOnly) override {
    if (!_file)
      _file.reset(new (_alloc) SymbolFile<ELFT>(_ctx));

    assert(!_file->hasAtoms() && "The file shouldn't have atoms yet");
    _resolver(sym, *_file);
    // If atoms were added - release the file to the caller.
    return _file->hasAtoms() ? _file.release() : nullptr;
  }

private:
  ELFLinkingContext &_ctx;
  Resolver _resolver;

  // The allocator should go before bump pointers because of
  // reversed destruction order.
  llvm::BumpPtrAllocator _alloc;
  unique_bump_ptr<SymbolFile<ELFT>> _file;
};

} // end anon namespace

template <class ELFT>
OutputELFWriter<ELFT>::OutputELFWriter(ELFLinkingContext &ctx,
                                       TargetLayout<ELFT> &layout)
    : _ctx(ctx), _targetHandler(ctx.getTargetHandler()), _layout(layout) {}

template <class ELFT>
void OutputELFWriter<ELFT>::buildChunks(const File &file) {
  ScopedTask task(getDefaultDomain(), "buildChunks");
  for (const DefinedAtom *definedAtom : file.defined()) {
    DefinedAtom::ContentType contentType = definedAtom->contentType();
    // Dont add COMDAT group atoms and GNU linkonce atoms, as they are used for
    // symbol resolution.
    // TODO: handle partial linking.
    if (contentType == DefinedAtom::typeGroupComdat ||
        contentType == DefinedAtom::typeGnuLinkOnce)
      continue;
    _layout.addAtom(definedAtom);
  }
  for (const AbsoluteAtom *absoluteAtom : file.absolute())
    _layout.addAtom(absoluteAtom);
}

template <class ELFT>
void OutputELFWriter<ELFT>::buildStaticSymbolTable(const File &file) {
  ScopedTask task(getDefaultDomain(), "buildStaticSymbolTable");
  for (auto sec : _layout.sections())
    if (auto section = dyn_cast<AtomSection<ELFT>>(sec))
      for (const auto &atom : section->atoms())
        _symtab->addSymbol(atom->_atom, section->ordinal(), atom->_virtualAddr);
  for (auto &atom : _layout.absoluteAtoms())
    _symtab->addSymbol(atom->_atom, ELF::SHN_ABS, atom->_virtualAddr);
  for (const UndefinedAtom *a : file.undefined())
    _symtab->addSymbol(a, ELF::SHN_UNDEF);
}

// Returns the DSO name for a given input file if it's a shared library
// file and not marked as --as-needed.
template <class ELFT>
StringRef OutputELFWriter<ELFT>::maybeGetSOName(Node *node) {
  if (auto *fnode = dyn_cast<FileNode>(node))
    if (!fnode->asNeeded())
      if (auto *file = dyn_cast<SharedLibraryFile>(fnode->getFile()))
        return file->getDSOName();
  return "";
}

template <class ELFT>
void OutputELFWriter<ELFT>::buildDynamicSymbolTable(const File &file) {
  ScopedTask task(getDefaultDomain(), "buildDynamicSymbolTable");
  for (const auto &sla : file.sharedLibrary()) {
    if (isDynSymEntryRequired(sla)) {
      _dynamicSymbolTable->addSymbol(sla, ELF::SHN_UNDEF);
      _soNeeded.insert(sla->loadName());
      continue;
    }
    if (isNeededTagRequired(sla))
      _soNeeded.insert(sla->loadName());
  }
  for (const std::unique_ptr<Node> &node : _ctx.getNodes()) {
    StringRef soname = maybeGetSOName(node.get());
    if (!soname.empty())
      _soNeeded.insert(soname);
  }
  // Never mark the dynamic linker as DT_NEEDED
  _soNeeded.erase(sys::path::filename(_ctx.getInterpreter()));
  for (const auto &loadName : _soNeeded) {
    Elf_Dyn dyn;
    dyn.d_tag = DT_NEEDED;
    dyn.d_un.d_val = _dynamicStringTable->addString(loadName.getKey());
    _dynamicTable->addEntry(dyn);
  }
  const auto &rpathList = _ctx.getRpathList();
  if (!rpathList.empty()) {
    auto rpath =
        new (_alloc) std::string(join(rpathList.begin(), rpathList.end(), ":"));
    Elf_Dyn dyn;
    dyn.d_tag = _ctx.getEnableNewDtags() ? DT_RUNPATH : DT_RPATH;
    dyn.d_un.d_val = _dynamicStringTable->addString(*rpath);
    _dynamicTable->addEntry(dyn);
  }
  StringRef soname = _ctx.sharedObjectName();
  if (!soname.empty() && _ctx.getOutputELFType() == llvm::ELF::ET_DYN) {
    Elf_Dyn dyn;
    dyn.d_tag = DT_SONAME;
    dyn.d_un.d_val = _dynamicStringTable->addString(soname);
    _dynamicTable->addEntry(dyn);
  }
  // The dynamic symbol table need to be sorted earlier because the hash
  // table needs to be built using the dynamic symbol table. It would be
  // late to sort the symbols due to that in finalize. In the dynamic symbol
  // table finalize, we call the symbol table finalize and we don't want to
  // sort again
  _dynamicSymbolTable->sortSymbols();

  // Add the dynamic symbols into the hash table
  _dynamicSymbolTable->addSymbolsToHashTable();
}

template <class ELFT>
void OutputELFWriter<ELFT>::buildAtomToAddressMap(const File &file) {
  ScopedTask task(getDefaultDomain(), "buildAtomToAddressMap");
  int64_t totalAbsAtoms = _layout.absoluteAtoms().size();
  int64_t totalUndefinedAtoms = file.undefined().size();
  int64_t totalDefinedAtoms = 0;
  for (auto sec : _layout.sections())
    if (auto section = dyn_cast<AtomSection<ELFT>>(sec)) {
      totalDefinedAtoms += section->atoms().size();
      for (const auto &atom : section->atoms())
        _atomToAddressMap[atom->_atom] = atom->_virtualAddr;
    }
  // build the atomToAddressMap that contains absolute symbols too
  for (auto &atom : _layout.absoluteAtoms())
    _atomToAddressMap[atom->_atom] = atom->_virtualAddr;

  // Set the total number of atoms in the symbol table, so that appropriate
  // resizing of the string table can be done.
  // There's no such thing as symbol table if we're stripping all the symbols
  if (!_ctx.stripSymbols())
    _symtab->setNumEntries(totalDefinedAtoms + totalAbsAtoms +
                           totalUndefinedAtoms);
}

template <class ELFT> void OutputELFWriter<ELFT>::buildSectionHeaderTable() {
  ScopedTask task(getDefaultDomain(), "buildSectionHeaderTable");
  for (auto outputSection : _layout.outputSections()) {
    if (outputSection->kind() != Chunk<ELFT>::Kind::ELFSection &&
        outputSection->kind() != Chunk<ELFT>::Kind::AtomSection)
      continue;
    if (outputSection->hasSegment())
      _shdrtab->appendSection(outputSection);
  }
}

template <class ELFT>
void OutputELFWriter<ELFT>::assignSectionsWithNoSegments() {
  ScopedTask task(getDefaultDomain(), "assignSectionsWithNoSegments");
  for (auto outputSection : _layout.outputSections()) {
    if (outputSection->kind() != Chunk<ELFT>::Kind::ELFSection &&
        outputSection->kind() != Chunk<ELFT>::Kind::AtomSection)
      continue;
    if (!outputSection->hasSegment())
      _shdrtab->appendSection(outputSection);
  }
  _layout.assignFileOffsetsForMiscSections();
  for (auto sec : _layout.sections())
    if (auto section = dyn_cast<Section<ELFT>>(sec))
      if (!TargetLayout<ELFT>::hasOutputSegment(section))
        _shdrtab->updateSection(section);
}

template <class ELFT>
void OutputELFWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  // Add the virtual archive to resolve undefined symbols.
  // The file will be added later in the linking context.
  auto callback = [this](StringRef sym, RuntimeFile<ELFT> &file) {
    processUndefinedSymbol(sym, file);
  };
  _ctx.setUndefinesResolver(
      llvm::make_unique<DynamicSymbolFile<ELFT>>(_ctx, std::move(callback)));
  // Add script defined symbols
  auto file =
      llvm::make_unique<RuntimeFile<ELFT>>(_ctx, "Linker script runtime");
  for (auto &sym : this->_ctx.linkerScriptSema().getScriptDefinedSymbols())
    file->addAbsoluteAtom(sym.getKey());
  result.push_back(std::move(file));
}

template <class ELFT> void OutputELFWriter<ELFT>::finalizeDefaultAtomValues() {
  const llvm::StringSet<> &symbols =
      _ctx.linkerScriptSema().getScriptDefinedSymbols();
  for (auto &sym : symbols) {
    uint64_t res =
        _ctx.linkerScriptSema().getLinkerScriptExprValue(sym.getKey());
    AtomLayout *a = _layout.findAbsoluteAtom(sym.getKey());
    assert(a);
    a->_virtualAddr = res;
  }
}

template <class ELFT> void OutputELFWriter<ELFT>::createDefaultSections() {
  _elfHeader.reset(new (_alloc) ELFHeader<ELFT>(_ctx));
  _programHeader.reset(new (_alloc) ProgramHeader<ELFT>(_ctx));
  _layout.setHeader(_elfHeader.get());
  _layout.setProgramHeader(_programHeader.get());

  // Don't create .symtab and .strtab sections if we're going to
  // strip all the symbols.
  if (!_ctx.stripSymbols()) {
    _symtab = std::move(this->createSymbolTable());
    _strtab.reset(new (_alloc) StringTable<ELFT>(
        _ctx, ".strtab", TargetLayout<ELFT>::ORDER_STRING_TABLE));
    _layout.addSection(_symtab.get());
    _layout.addSection(_strtab.get());
    _symtab->setStringSection(_strtab.get());
  }

  _shstrtab.reset(new (_alloc) StringTable<ELFT>(
      _ctx, ".shstrtab", TargetLayout<ELFT>::ORDER_SECTION_STRINGS));
  _shdrtab.reset(new (_alloc) SectionHeader<ELFT>(
      _ctx, TargetLayout<ELFT>::ORDER_SECTION_HEADERS));
  _layout.addSection(_shstrtab.get());
  _shdrtab->setStringSection(_shstrtab.get());
  _layout.addSection(_shdrtab.get());

  for (auto sec : _layout.sections()) {
    // TODO: use findOutputSection
    auto section = dyn_cast<Section<ELFT>>(sec);
    if (!section || section->outputSectionName() != ".eh_frame")
      continue;
    _ehFrameHeader.reset(new (_alloc) EHFrameHeader<ELFT>(
        _ctx, ".eh_frame_hdr", _layout, TargetLayout<ELFT>::ORDER_EH_FRAMEHDR));
    _layout.addSection(_ehFrameHeader.get());
    break;
  }

  if (_ctx.isDynamic()) {
    _dynamicTable = std::move(createDynamicTable());
    _dynamicStringTable.reset(new (_alloc) StringTable<ELFT>(
        _ctx, ".dynstr", TargetLayout<ELFT>::ORDER_DYNAMIC_STRINGS, true));
    _dynamicSymbolTable = std::move(createDynamicSymbolTable());
    _hashTable.reset(new (_alloc) HashSection<ELFT>(
        _ctx, ".hash", TargetLayout<ELFT>::ORDER_HASH));
    // Set the hash table in the dynamic symbol table so that the entries in the
    // hash table can be created
    _dynamicSymbolTable->setHashTable(_hashTable.get());
    _hashTable->setSymbolTable(_dynamicSymbolTable.get());
    _layout.addSection(_dynamicTable.get());
    _layout.addSection(_dynamicStringTable.get());
    _layout.addSection(_dynamicSymbolTable.get());
    _layout.addSection(_hashTable.get());
    _dynamicSymbolTable->setStringSection(_dynamicStringTable.get());
    _dynamicTable->setSymbolTable(_dynamicSymbolTable.get());
    _dynamicTable->setHashTable(_hashTable.get());
    if (_layout.hasDynamicRelocationTable())
      _layout.getDynamicRelocationTable()->setSymbolTable(
          _dynamicSymbolTable.get());
    if (_layout.hasPLTRelocationTable())
      _layout.getPLTRelocationTable()->setSymbolTable(
          _dynamicSymbolTable.get());
  }
}

template <class ELFT>
unique_bump_ptr<SymbolTable<ELFT>> OutputELFWriter<ELFT>::createSymbolTable() {
  return unique_bump_ptr<SymbolTable<ELFT>>(new (_alloc) SymbolTable<ELFT>(
      this->_ctx, ".symtab", TargetLayout<ELFT>::ORDER_SYMBOL_TABLE));
}

/// \brief create dynamic table
template <class ELFT>
unique_bump_ptr<DynamicTable<ELFT>>
OutputELFWriter<ELFT>::createDynamicTable() {
  return unique_bump_ptr<DynamicTable<ELFT>>(new (_alloc) DynamicTable<ELFT>(
      this->_ctx, _layout, ".dynamic", TargetLayout<ELFT>::ORDER_DYNAMIC));
}

/// \brief create dynamic symbol table
template <class ELFT>
unique_bump_ptr<DynamicSymbolTable<ELFT>>
OutputELFWriter<ELFT>::createDynamicSymbolTable() {
  return unique_bump_ptr<DynamicSymbolTable<ELFT>>(
      new (_alloc)
          DynamicSymbolTable<ELFT>(this->_ctx, _layout, ".dynsym",
                                   TargetLayout<ELFT>::ORDER_DYNAMIC_SYMBOLS));
}

template <class ELFT>
std::error_code OutputELFWriter<ELFT>::buildOutput(const File &file) {
  ScopedTask buildTask(getDefaultDomain(), "ELF Writer buildOutput");
  buildChunks(file);

  // Create the default sections like the symbol table, string table, and the
  // section string table
  createDefaultSections();

  // Set the Layout
  _layout.assignSectionsToSegments();

  // Create the dynamic table entries
  if (_ctx.isDynamic()) {
    _dynamicTable->createDefaultEntries();
    buildDynamicSymbolTable(file);
  }

  // Call the preFlight callbacks to modify the sections and the atoms
  // contained in them, in anyway the targets may want
  _layout.doPreFlight();

  _layout.assignVirtualAddress();

  // Finalize the default value of symbols that the linker adds
  finalizeDefaultAtomValues();

  // Build the Atom To Address map for applying relocations
  buildAtomToAddressMap(file);

  // Create symbol table and section string table
  // Do it only if -s is not specified.
  if (!_ctx.stripSymbols())
    buildStaticSymbolTable(file);

  // Finalize the layout by calling the finalize() functions
  _layout.finalize();

  // build Section Header table
  buildSectionHeaderTable();

  // assign Offsets and virtual addresses
  // for sections with no segments
  assignSectionsWithNoSegments();

  if (_ctx.isDynamic())
    _dynamicTable->updateDynamicTable();

  return std::error_code();
}

template <class ELFT> std::error_code OutputELFWriter<ELFT>::setELFHeader() {
  _elfHeader->e_type(_ctx.getOutputELFType());
  _elfHeader->e_machine(_ctx.getOutputMachine());
  _elfHeader->e_ident(ELF::EI_VERSION, 1);
  _elfHeader->e_ident(ELF::EI_OSABI, 0);
  _elfHeader->e_version(1);
  _elfHeader->e_phoff(_programHeader->fileOffset());
  _elfHeader->e_shoff(_shdrtab->fileOffset());
  _elfHeader->e_phentsize(_programHeader->entsize());
  _elfHeader->e_phnum(_programHeader->numHeaders());
  _elfHeader->e_shentsize(_shdrtab->entsize());
  _elfHeader->e_shnum(_shdrtab->numHeaders());
  _elfHeader->e_shstrndx(_shstrtab->ordinal());
  if (const auto *al = _layout.findAtomLayoutByName(_ctx.entrySymbolName()))
    _elfHeader->e_entry(al->_virtualAddr);
  else
    _elfHeader->e_entry(0);

  return std::error_code();
}

template <class ELFT> uint64_t OutputELFWriter<ELFT>::outputFileSize() const {
  return _shdrtab->fileOffset() + _shdrtab->fileSize();
}

template <class ELFT>
std::error_code OutputELFWriter<ELFT>::writeOutput(const File &file,
                                                   StringRef path) {
  std::unique_ptr<FileOutputBuffer> buffer;
  ScopedTask createOutputTask(getDefaultDomain(), "ELF Writer Create Output");
  if (std::error_code ec = FileOutputBuffer::create(
          path, outputFileSize(), buffer, FileOutputBuffer::F_executable))
    return ec;
  createOutputTask.end();

  ScopedTask writeTask(getDefaultDomain(), "ELF Writer write to memory");

  // HACK: We have to write out the header and program header here even though
  // they are a member of a segment because only sections are written in the
  // following loop.

  // Finalize ELF Header / Program Headers.
  _elfHeader->finalize();
  _programHeader->finalize();

  _elfHeader->write(this, _layout, *buffer);
  _programHeader->write(this, _layout, *buffer);

  auto sections = _layout.sections();
  parallel_for_each(
      sections.begin(), sections.end(),
      [&](Chunk<ELFT> *section) { section->write(this, _layout, *buffer); });
  writeTask.end();

  ScopedTask commitTask(getDefaultDomain(), "ELF Writer commit to disk");
  return buffer->commit();
}

template <class ELFT>
std::error_code OutputELFWriter<ELFT>::writeFile(const File &file,
                                                 StringRef path) {
  if (std::error_code ec = buildOutput(file))
    return ec;
  if (std::error_code ec = setELFHeader())
    return ec;
  return writeOutput(file, path);
}

template <class ELFT>
void OutputELFWriter<ELFT>::updateScopeAtomValues(StringRef sym,
                                                  StringRef sec) {
  std::string start = ("__" + sym + "_start").str();
  std::string end = ("__" + sym + "_end").str();
  AtomLayout *s = _layout.findAbsoluteAtom(start);
  AtomLayout *e = _layout.findAbsoluteAtom(end);
  OutputSection<ELFT> *section = _layout.findOutputSection(sec);
  if (!s || !e)
    return;

  if (section) {
    s->_virtualAddr = section->virtualAddr();
    e->_virtualAddr = section->virtualAddr() + section->memSize();
  } else {
    s->_virtualAddr = 0;
    e->_virtualAddr = 0;
  }
}

template class OutputELFWriter<ELF32LE>;
template class OutputELFWriter<ELF32BE>;
template class OutputELFWriter<ELF64LE>;
template class OutputELFWriter<ELF64BE>;

} // namespace elf
} // namespace lld
