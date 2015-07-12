//===- lib/ReaderWriter/ELF/SectionChunks.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SectionChunks.h"
#include "TargetLayout.h"
#include "lld/Core/Parallel.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Dwarf.h"

namespace lld {
namespace elf {

template <class ELFT>
Section<ELFT>::Section(const ELFLinkingContext &ctx, StringRef sectionName,
                       StringRef chunkName, typename Chunk<ELFT>::Kind k)
    : Chunk<ELFT>(chunkName, k, ctx), _inputSectionName(sectionName),
      _outputSectionName(sectionName) {}

template <class ELFT> int Section<ELFT>::getContentType() const {
  if (_flags & llvm::ELF::SHF_EXECINSTR)
    return Chunk<ELFT>::ContentType::Code;
  else if (_flags & llvm::ELF::SHF_WRITE)
    return Chunk<ELFT>::ContentType::Data;
  else if (_flags & llvm::ELF::SHF_ALLOC)
    return Chunk<ELFT>::ContentType::Code;
  else
    return Chunk<ELFT>::ContentType::Unknown;
}

template <class ELFT>
AtomSection<ELFT>::AtomSection(const ELFLinkingContext &ctx,
                               StringRef sectionName, int32_t contentType,
                               int32_t permissions, int32_t order)
    : Section<ELFT>(ctx, sectionName, "AtomSection",
                    Chunk<ELFT>::Kind::AtomSection),
      _contentType(contentType), _contentPermissions(permissions) {
  this->setOrder(order);

  switch (contentType) {
  case DefinedAtom::typeCode:
  case DefinedAtom::typeDataFast:
  case DefinedAtom::typeData:
  case DefinedAtom::typeConstant:
  case DefinedAtom::typeGOT:
  case DefinedAtom::typeStub:
  case DefinedAtom::typeResolver:
  case DefinedAtom::typeThreadData:
    this->_type = SHT_PROGBITS;
    break;

  case DefinedAtom::typeThreadZeroFill:
  case DefinedAtom::typeZeroFillFast:
  case DefinedAtom::typeZeroFill:
    this->_type = SHT_NOBITS;
    break;

  case DefinedAtom::typeRONote:
  case DefinedAtom::typeRWNote:
    this->_type = SHT_NOTE;
    break;

  case DefinedAtom::typeNoAlloc:
    this->_type = SHT_PROGBITS;
    this->_isLoadedInMemory = false;
    break;
  }

  switch (permissions) {
  case DefinedAtom::permR__:
    this->_flags = SHF_ALLOC;
    break;
  case DefinedAtom::permR_X:
    this->_flags = SHF_ALLOC | SHF_EXECINSTR;
    break;
  case DefinedAtom::permRW_:
  case DefinedAtom::permRW_L:
    this->_flags = SHF_ALLOC | SHF_WRITE;
    if (_contentType == DefinedAtom::typeThreadData ||
        _contentType == DefinedAtom::typeThreadZeroFill)
      this->_flags |= SHF_TLS;
    break;
  case DefinedAtom::permRWX:
    this->_flags = SHF_ALLOC | SHF_WRITE | SHF_EXECINSTR;
    break;
  case DefinedAtom::perm___:
    this->_flags = 0;
    break;
  }
}

template <class ELFT>
void AtomSection<ELFT>::assignVirtualAddress(uint64_t addr) {
  parallel_for_each(_atoms.begin(), _atoms.end(), [&](AtomLayout *ai) {
    ai->_virtualAddr = addr + ai->_fileOffset;
  });
}

template <class ELFT>
void AtomSection<ELFT>::assignFileOffsets(uint64_t offset) {
  parallel_for_each(_atoms.begin(), _atoms.end(), [&](AtomLayout *ai) {
    ai->_fileOffset = offset + ai->_fileOffset;
  });
}

template <class ELFT>
const AtomLayout *
AtomSection<ELFT>::findAtomLayoutByName(StringRef name) const {
  for (auto ai : _atoms)
    if (ai->_atom->name() == name)
      return ai;
  return nullptr;
}

template <class ELFT>
std::string AtomSection<ELFT>::formatError(const std::string &errorStr,
                                           const AtomLayout &atom,
                                           const Reference &ref) const {
  StringRef kindValStr;
  if (!this->_ctx.registry().referenceKindToString(
          ref.kindNamespace(), ref.kindArch(), ref.kindValue(), kindValStr)) {
    kindValStr = "unknown";
  }

  return
      (Twine(errorStr) + " in file " + atom._atom->file().path() +
       ": reference from " + atom._atom->name() + "+" +
       Twine(ref.offsetInAtom()) + " to " + ref.target()->name() + "+" +
       Twine(ref.addend()) + " of type " + Twine(ref.kindValue()) + " (" +
       kindValStr + ")\n")
          .str();
}

/// Align the offset to the required modulus defined by the atom alignment
template <class ELFT>
uint64_t AtomSection<ELFT>::alignOffset(uint64_t offset,
                                        DefinedAtom::Alignment &atomAlign) {
  uint64_t requiredModulus = atomAlign.modulus;
  uint64_t alignment = atomAlign.value;
  uint64_t currentModulus = (offset % alignment);
  uint64_t retOffset = offset;
  if (currentModulus != requiredModulus) {
    if (requiredModulus > currentModulus)
      retOffset += requiredModulus - currentModulus;
    else
      retOffset += alignment + requiredModulus - currentModulus;
  }
  return retOffset;
}

// \brief Append an atom to a Section. The atom gets pushed into a vector
// contains the atom, the atom file offset, the atom virtual address
// the atom file offset is aligned appropriately as set by the Reader
template <class ELFT>
const AtomLayout *AtomSection<ELFT>::appendAtom(const Atom *atom) {
  const DefinedAtom *definedAtom = cast<DefinedAtom>(atom);

  DefinedAtom::Alignment atomAlign = definedAtom->alignment();
  uint64_t alignment = atomAlign.value;
  // Align the atom to the required modulus/ align the file offset and the
  // memory offset separately this is required so that BSS symbols are handled
  // properly as the BSS symbols only occupy memory size and not file size
  uint64_t fOffset = alignOffset(this->fileSize(), atomAlign);
  uint64_t mOffset = alignOffset(this->memSize(), atomAlign);
  switch (definedAtom->contentType()) {
  case DefinedAtom::typeCode:
  case DefinedAtom::typeConstant:
  case DefinedAtom::typeData:
  case DefinedAtom::typeDataFast:
  case DefinedAtom::typeZeroFillFast:
  case DefinedAtom::typeGOT:
  case DefinedAtom::typeStub:
  case DefinedAtom::typeResolver:
  case DefinedAtom::typeThreadData:
  case DefinedAtom::typeRONote:
  case DefinedAtom::typeRWNote:
    _atoms.push_back(new (_alloc) AtomLayout(atom, fOffset, 0));
    this->_fsize = fOffset + definedAtom->size();
    this->_msize = mOffset + definedAtom->size();
    DEBUG_WITH_TYPE("Section", llvm::dbgs()
                                   << "[" << this->name() << " " << this << "] "
                                   << "Adding atom: " << atom->name() << "@"
                                   << fOffset << "\n");
    break;
  case DefinedAtom::typeNoAlloc:
    _atoms.push_back(new (_alloc) AtomLayout(atom, fOffset, 0));
    this->_fsize = fOffset + definedAtom->size();
    DEBUG_WITH_TYPE("Section", llvm::dbgs()
                                   << "[" << this->name() << " " << this << "] "
                                   << "Adding atom: " << atom->name() << "@"
                                   << fOffset << "\n");
    break;
  case DefinedAtom::typeThreadZeroFill:
  case DefinedAtom::typeZeroFill:
    _atoms.push_back(new (_alloc) AtomLayout(atom, mOffset, 0));
    this->_msize = mOffset + definedAtom->size();
    break;
  default:
    llvm::dbgs() << definedAtom->contentType() << "\n";
    llvm_unreachable("Uexpected content type.");
  }
  // Set the section alignment to the largest alignment
  // std::max doesn't support uint64_t
  if (this->_alignment < alignment)
    this->_alignment = alignment;

  if (_atoms.size())
    return _atoms.back();
  return nullptr;
}

/// \brief convert the segment type to a String for diagnostics
///        and printing purposes
template <class ELFT> StringRef Section<ELFT>::segmentKindToStr() const {
  switch (_segmentType) {
  case llvm::ELF::PT_DYNAMIC:
    return "DYNAMIC";
  case llvm::ELF::PT_INTERP:
    return "INTERP";
  case llvm::ELF::PT_LOAD:
    return "LOAD";
  case llvm::ELF::PT_GNU_EH_FRAME:
    return "EH_FRAME";
  case llvm::ELF::PT_GNU_RELRO:
    return "GNU_RELRO";
  case llvm::ELF::PT_NOTE:
    return "NOTE";
  case llvm::ELF::PT_NULL:
    return "NULL";
  case llvm::ELF::PT_TLS:
    return "TLS";
  default:
    return "UNKNOWN";
  }
}

/// \brief Write the section and the atom contents to the buffer
template <class ELFT>
void AtomSection<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                              llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  bool success = true;

  // parallel_for_each() doesn't have deterministic order.  To guarantee
  // deterministic error output, collect errors in this vector and sort it
  // by atom file offset before printing all errors.
  std::vector<std::pair<size_t, std::string>> errors;
  parallel_for_each(_atoms.begin(), _atoms.end(), [&](AtomLayout *ai) {
    DEBUG_WITH_TYPE("Section", llvm::dbgs()
                                   << "Writing atom: " << ai->_atom->name()
                                   << " | " << ai->_fileOffset << "\n");
    const DefinedAtom *definedAtom = cast<DefinedAtom>(ai->_atom);
    if (!definedAtom->occupiesDiskSpace())
      return;
    // Copy raw content of atom to file buffer.
    ArrayRef<uint8_t> content = definedAtom->rawContent();
    uint64_t contentSize = content.size();
    if (contentSize == 0)
      return;
    uint8_t *atomContent = chunkBuffer + ai->_fileOffset;
    std::memcpy(atomContent, content.data(), contentSize);
    const TargetRelocationHandler &relHandler =
        this->_ctx.getTargetHandler().getRelocationHandler();
    for (const auto ref : *definedAtom) {
      if (std::error_code ec =
              relHandler.applyRelocation(*writer, buffer, *ai, *ref)) {
        std::lock_guard<std::mutex> lock(_outputMutex);
        errors.push_back(std::make_pair(ai->_fileOffset,
                                        formatError(ec.message(), *ai, *ref)));
        success = false;
      }
    }
  });
  if (!success) {
    std::sort(errors.begin(), errors.end());
    for (auto &&error : errors)
      llvm::errs() << error.second;
    llvm::report_fatal_error("relocating output");
  }
}

template <class ELFT>
void OutputSection<ELFT>::appendSection(Section<ELFT> *section) {
  if (section->alignment() > _alignment)
    _alignment = section->alignment();
  assert(!_link && "Section already has a link!");
  _link = section->getLink();
  _shInfo = section->getInfo();
  _entSize = section->getEntSize();
  _type = section->getType();
  if (_flags < section->getFlags())
    _flags = section->getFlags();
  section->setOutputSection(this, (_sections.size() == 0));
  _kind = section->kind();
  _sections.push_back(section);
}

template <class ELFT>
StringTable<ELFT>::StringTable(const ELFLinkingContext &ctx, const char *str,
                               int32_t order, bool dynamic)
    : Section<ELFT>(ctx, str, "StringTable") {
  // the string table has a NULL entry for which
  // add an empty string
  _strings.push_back("");
  this->_fsize = 1;
  this->_alignment = 1;
  this->setOrder(order);
  this->_type = SHT_STRTAB;
  if (dynamic) {
    this->_flags = SHF_ALLOC;
    this->_msize = this->_fsize;
  }
}

template <class ELFT> uint64_t StringTable<ELFT>::addString(StringRef symname) {
  if (symname.empty())
    return 0;
  StringMapTIter stringIter = _stringMap.find(symname);
  if (stringIter == _stringMap.end()) {
    _strings.push_back(symname);
    uint64_t offset = this->_fsize;
    this->_fsize += symname.size() + 1;
    if (this->_flags & SHF_ALLOC)
      this->_msize = this->_fsize;
    _stringMap[symname] = offset;
    return offset;
  }
  return stringIter->second;
}

template <class ELFT>
void StringTable<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &,
                              llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  for (auto si : _strings) {
    memcpy(dest, si.data(), si.size());
    dest += si.size();
    memcpy(dest, "", 1);
    dest += 1;
  }
}

/// ELF Symbol Table
template <class ELFT>
SymbolTable<ELFT>::SymbolTable(const ELFLinkingContext &ctx, const char *str,
                               int32_t order)
    : Section<ELFT>(ctx, str, "SymbolTable") {
  this->setOrder(order);
  Elf_Sym symbol;
  std::memset(&symbol, 0, sizeof(Elf_Sym));
  _symbolTable.push_back(SymbolEntry(nullptr, symbol, nullptr));
  this->_entSize = sizeof(Elf_Sym);
  this->_fsize = sizeof(Elf_Sym);
  this->_alignment = sizeof(Elf_Addr);
  this->_type = SHT_SYMTAB;
}

template <class ELFT>
void SymbolTable<ELFT>::addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                                       int64_t addr) {
  unsigned char binding = 0, type = 0;
  sym.st_size = da->size();
  DefinedAtom::ContentType ct;
  switch (ct = da->contentType()) {
  case DefinedAtom::typeCode:
  case DefinedAtom::typeStub:
    sym.st_value = addr;
    type = llvm::ELF::STT_FUNC;
    break;
  case DefinedAtom::typeResolver:
    sym.st_value = addr;
    type = llvm::ELF::STT_GNU_IFUNC;
    break;
  case DefinedAtom::typeDataFast:
  case DefinedAtom::typeData:
  case DefinedAtom::typeConstant:
    sym.st_value = addr;
    type = llvm::ELF::STT_OBJECT;
    break;
  case DefinedAtom::typeGOT:
    sym.st_value = addr;
    type = llvm::ELF::STT_NOTYPE;
    break;
  case DefinedAtom::typeZeroFill:
  case DefinedAtom::typeZeroFillFast:
    type = llvm::ELF::STT_OBJECT;
    sym.st_value = addr;
    break;
  case DefinedAtom::typeThreadData:
  case DefinedAtom::typeThreadZeroFill:
    type = llvm::ELF::STT_TLS;
    sym.st_value = addr;
    break;
  default:
    type = llvm::ELF::STT_NOTYPE;
  }
  if (da->customSectionName() == da->name())
    type = llvm::ELF::STT_SECTION;

  if (da->scope() == DefinedAtom::scopeTranslationUnit)
    binding = llvm::ELF::STB_LOCAL;
  else
    binding = llvm::ELF::STB_GLOBAL;

  sym.setBindingAndType(binding, type);
}

template <class ELFT>
void SymbolTable<ELFT>::addAbsoluteAtom(Elf_Sym &sym, const AbsoluteAtom *aa,
                                        int64_t addr) {
  unsigned char binding = 0, type = 0;
  type = llvm::ELF::STT_OBJECT;
  sym.st_shndx = llvm::ELF::SHN_ABS;
  switch (aa->scope()) {
  case AbsoluteAtom::scopeLinkageUnit:
    sym.setVisibility(llvm::ELF::STV_HIDDEN);
    binding = llvm::ELF::STB_LOCAL;
    break;
  case AbsoluteAtom::scopeTranslationUnit:
    binding = llvm::ELF::STB_LOCAL;
    break;
  case AbsoluteAtom::scopeGlobal:
    binding = llvm::ELF::STB_GLOBAL;
    break;
  }
  sym.st_value = addr;
  sym.setBindingAndType(binding, type);
}

template <class ELFT>
void SymbolTable<ELFT>::addSharedLibAtom(Elf_Sym &sym,
                                         const SharedLibraryAtom *aa) {
  unsigned char binding = 0, type = 0;
  if (aa->type() == SharedLibraryAtom::Type::Data) {
    type = llvm::ELF::STT_OBJECT;
    sym.st_size = aa->size();
  } else
    type = llvm::ELF::STT_FUNC;
  sym.st_shndx = llvm::ELF::SHN_UNDEF;
  binding = llvm::ELF::STB_GLOBAL;
  sym.setBindingAndType(binding, type);
}

template <class ELFT>
void SymbolTable<ELFT>::addUndefinedAtom(Elf_Sym &sym,
                                         const UndefinedAtom *ua) {
  unsigned char binding = 0, type = 0;
  sym.st_value = 0;
  type = llvm::ELF::STT_NOTYPE;
  if (ua->canBeNull())
    binding = llvm::ELF::STB_WEAK;
  else
    binding = llvm::ELF::STB_GLOBAL;
  sym.setBindingAndType(binding, type);
}

/// Add a symbol to the symbol Table, definedAtoms which get added to the symbol
/// section don't have their virtual addresses set at the time of adding the
/// symbol to the symbol table(Example: dynamic symbols), the addresses needs
/// to be updated in the table before writing the dynamic symbol table
/// information
template <class ELFT>
void SymbolTable<ELFT>::addSymbol(const Atom *atom, int32_t sectionIndex,
                                  uint64_t addr, const AtomLayout *atomLayout) {
  Elf_Sym symbol;

  if (atom->name().empty())
    return;

  symbol.st_name = _stringSection->addString(atom->name());
  symbol.st_size = 0;
  symbol.st_shndx = sectionIndex;
  symbol.st_value = 0;
  symbol.st_other = 0;
  symbol.setVisibility(llvm::ELF::STV_DEFAULT);

  // Add all the atoms
  if (const DefinedAtom *da = dyn_cast<const DefinedAtom>(atom))
    addDefinedAtom(symbol, da, addr);
  else if (const AbsoluteAtom *aa = dyn_cast<const AbsoluteAtom>(atom))
    addAbsoluteAtom(symbol, aa, addr);
  else if (isa<const SharedLibraryAtom>(atom))
    addSharedLibAtom(symbol, dyn_cast<SharedLibraryAtom>(atom));
  else
    addUndefinedAtom(symbol, dyn_cast<UndefinedAtom>(atom));

  // If --discard-all is on, don't add to the symbol table
  // symbols with local binding.
  if (this->_ctx.discardLocals() && symbol.getBinding() == llvm::ELF::STB_LOCAL)
    return;

  // Temporary locals are all the symbols which name starts with .L.
  // This is defined by the ELF standard.
  if (this->_ctx.discardTempLocals() && atom->name().startswith(".L"))
    return;

  _symbolTable.push_back(SymbolEntry(atom, symbol, atomLayout));
  this->_fsize += sizeof(Elf_Sym);
  if (this->_flags & SHF_ALLOC)
    this->_msize = this->_fsize;
}

template <class ELFT> void SymbolTable<ELFT>::finalize(bool sort) {
  // sh_info should be one greater than last symbol with STB_LOCAL binding
  // we sort the symbol table to keep all local symbols at the beginning
  if (sort)
    sortSymbols();

  uint16_t shInfo = 0;
  for (const auto &i : _symbolTable) {
    if (i._symbol.getBinding() != llvm::ELF::STB_LOCAL)
      break;
    shInfo++;
  }
  this->_info = shInfo;
  this->_link = _stringSection->ordinal();
  if (this->_outputSection) {
    this->_outputSection->setInfo(this->_info);
    this->_outputSection->setLink(this->_link);
  }
}

template <class ELFT>
void SymbolTable<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &,
                              llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  for (const auto &sti : _symbolTable) {
    memcpy(dest, &sti._symbol, sizeof(Elf_Sym));
    dest += sizeof(Elf_Sym);
  }
}

template <class ELFT>
DynamicSymbolTable<ELFT>::DynamicSymbolTable(const ELFLinkingContext &ctx,
                                             TargetLayout<ELFT> &layout,
                                             const char *str, int32_t order)
    : SymbolTable<ELFT>(ctx, str, order), _layout(layout) {
  this->_type = SHT_DYNSYM;
  this->_flags = SHF_ALLOC;
  this->_msize = this->_fsize;
}

template <class ELFT> void DynamicSymbolTable<ELFT>::addSymbolsToHashTable() {
  int index = 0;
  for (auto &ste : this->_symbolTable) {
    if (!ste._atom)
      _hashTable->addSymbol("", index);
    else
      _hashTable->addSymbol(ste._atom->name(), index);
    ++index;
  }
}

template <class ELFT> void DynamicSymbolTable<ELFT>::finalize() {
  // Defined symbols which have been added into the dynamic symbol table
  // don't have their addresses known until addresses have been assigned
  // so let's update the symbol values after they have got assigned
  for (auto &ste : this->_symbolTable) {
    const AtomLayout *atomLayout = ste._atomLayout;
    if (!atomLayout)
      continue;
    ste._symbol.st_value = atomLayout->_virtualAddr;
  }

  // Don't sort the symbols
  SymbolTable<ELFT>::finalize(false);
}

template <class ELFT>
RelocationTable<ELFT>::RelocationTable(const ELFLinkingContext &ctx,
                                       StringRef str, int32_t order)
    : Section<ELFT>(ctx, str, "RelocationTable") {
  this->setOrder(order);
  this->_flags = SHF_ALLOC;
  // Set the alignment properly depending on the target architecture
  this->_alignment = ELFT::Is64Bits ? 8 : 4;
  if (ctx.isRelaOutputFormat()) {
    this->_entSize = sizeof(Elf_Rela);
    this->_type = SHT_RELA;
  } else {
    this->_entSize = sizeof(Elf_Rel);
    this->_type = SHT_REL;
  }
}

template <class ELFT>
uint32_t RelocationTable<ELFT>::addRelocation(const DefinedAtom &da,
                                              const Reference &r) {
  _relocs.emplace_back(&da, &r);
  this->_fsize = _relocs.size() * this->_entSize;
  this->_msize = this->_fsize;
  return _relocs.size() - 1;
}

template <class ELFT>
bool RelocationTable<ELFT>::getRelocationIndex(const Reference &r,
                                               uint32_t &res) {
  auto rel = std::find_if(
      _relocs.begin(), _relocs.end(),
      [&](const std::pair<const DefinedAtom *, const Reference *> &p) {
        if (p.second == &r)
          return true;
        return false;
      });
  if (rel == _relocs.end())
    return false;
  res = std::distance(_relocs.begin(), rel);
  return true;
}

template <class ELFT>
bool RelocationTable<ELFT>::canModifyReadonlySection() const {
  for (const auto &rel : _relocs) {
    const DefinedAtom *atom = rel.first;
    if ((atom->permissions() & DefinedAtom::permRW_) != DefinedAtom::permRW_)
      return true;
  }
  return false;
}

template <class ELFT> void RelocationTable<ELFT>::finalize() {
  this->_link = _symbolTable ? _symbolTable->ordinal() : 0;
  if (this->_outputSection)
    this->_outputSection->setLink(this->_link);
}

template <class ELFT>
void RelocationTable<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                                  llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  for (const auto &rel : _relocs) {
    if (this->_ctx.isRelaOutputFormat()) {
      auto &r = *reinterpret_cast<Elf_Rela *>(dest);
      writeRela(writer, r, *rel.first, *rel.second);
      DEBUG_WITH_TYPE("ELFRelocationTable",
                      llvm::dbgs()
                          << rel.second->kindValue() << " relocation at "
                          << rel.first->name() << "@" << r.r_offset << " to "
                          << rel.second->target()->name() << "@" << r.r_addend
                          << "\n";);
    } else {
      auto &r = *reinterpret_cast<Elf_Rel *>(dest);
      writeRel(writer, r, *rel.first, *rel.second);
      DEBUG_WITH_TYPE("ELFRelocationTable",
                      llvm::dbgs() << rel.second->kindValue()
                                   << " relocation at " << rel.first->name()
                                   << "@" << r.r_offset << " to "
                                   << rel.second->target()->name() << "\n";);
    }
    dest += this->_entSize;
  }
}

template <class ELFT>
void RelocationTable<ELFT>::writeRela(ELFWriter *writer, Elf_Rela &r,
                                      const DefinedAtom &atom,
                                      const Reference &ref) {
  r.setSymbolAndType(getSymbolIndex(ref.target()), ref.kindValue(), false);
  r.r_offset = writer->addressOfAtom(&atom) + ref.offsetInAtom();
  // The addend is used only by relative relocations
  if (this->_ctx.isRelativeReloc(ref))
    r.r_addend = writer->addressOfAtom(ref.target()) + ref.addend();
  else
    r.r_addend = 0;
}

template <class ELFT>
void RelocationTable<ELFT>::writeRel(ELFWriter *writer, Elf_Rel &r,
                                     const DefinedAtom &atom,
                                     const Reference &ref) {
  r.setSymbolAndType(getSymbolIndex(ref.target()), ref.kindValue(), false);
  r.r_offset = writer->addressOfAtom(&atom) + ref.offsetInAtom();
}

template <class ELFT>
uint32_t RelocationTable<ELFT>::getSymbolIndex(const Atom *a) {
  return _symbolTable ? _symbolTable->getSymbolTableIndex(a)
                      : (uint32_t)STN_UNDEF;
}

template <class ELFT>
DynamicTable<ELFT>::DynamicTable(const ELFLinkingContext &ctx,
                                 TargetLayout<ELFT> &layout, StringRef str,
                                 int32_t order)
    : Section<ELFT>(ctx, str, "DynamicSection"), _layout(layout) {
  this->setOrder(order);
  this->_entSize = sizeof(Elf_Dyn);
  this->_alignment = ELFT::Is64Bits ? 8 : 4;
  // Reserve space for the DT_NULL entry.
  this->_fsize = sizeof(Elf_Dyn);
  this->_msize = sizeof(Elf_Dyn);
  this->_type = SHT_DYNAMIC;
  this->_flags = SHF_ALLOC;
}

template <class ELFT>
std::size_t DynamicTable<ELFT>::addEntry(int64_t tag, uint64_t val) {
  Elf_Dyn dyn;
  dyn.d_tag = tag;
  dyn.d_un.d_val = val;
  _entries.push_back(dyn);
  this->_fsize = (_entries.size() * sizeof(Elf_Dyn)) + sizeof(Elf_Dyn);
  this->_msize = this->_fsize;
  return _entries.size() - 1;
}

template <class ELFT>
void DynamicTable<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                               llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  // Add the null entry.
  Elf_Dyn d;
  d.d_tag = 0;
  d.d_un.d_val = 0;
  _entries.push_back(d);
  std::memcpy(dest, _entries.data(), this->_fsize);
}

template <class ELFT> void DynamicTable<ELFT>::createDefaultEntries() {
  bool isRela = this->_ctx.isRelaOutputFormat();
  _dt_hash = addEntry(DT_HASH, 0);
  _dt_strtab = addEntry(DT_STRTAB, 0);
  _dt_symtab = addEntry(DT_SYMTAB, 0);
  _dt_strsz = addEntry(DT_STRSZ, 0);
  _dt_syment = addEntry(DT_SYMENT, 0);
  if (_layout.hasDynamicRelocationTable()) {
    _dt_rela = addEntry(isRela ? DT_RELA : DT_REL, 0);
    _dt_relasz = addEntry(isRela ? DT_RELASZ : DT_RELSZ, 0);
    _dt_relaent = addEntry(isRela ? DT_RELAENT : DT_RELENT, 0);
    if (_layout.getDynamicRelocationTable()->canModifyReadonlySection())
      _dt_textrel = addEntry(DT_TEXTREL, 0);
  }
  if (_layout.hasPLTRelocationTable()) {
    _dt_pltrelsz = addEntry(DT_PLTRELSZ, 0);
    _dt_pltgot = addEntry(getGotPltTag(), 0);
    _dt_pltrel = addEntry(DT_PLTREL, isRela ? DT_RELA : DT_REL);
    _dt_jmprel = addEntry(DT_JMPREL, 0);
  }
}

template <class ELFT> void DynamicTable<ELFT>::doPreFlight() {
  auto initArray = _layout.findOutputSection(".init_array");
  auto finiArray = _layout.findOutputSection(".fini_array");
  if (initArray) {
    _dt_init_array = addEntry(DT_INIT_ARRAY, 0);
    _dt_init_arraysz = addEntry(DT_INIT_ARRAYSZ, 0);
  }
  if (finiArray) {
    _dt_fini_array = addEntry(DT_FINI_ARRAY, 0);
    _dt_fini_arraysz = addEntry(DT_FINI_ARRAYSZ, 0);
  }
  if (getInitAtomLayout())
    _dt_init = addEntry(DT_INIT, 0);
  if (getFiniAtomLayout())
    _dt_fini = addEntry(DT_FINI, 0);
}

template <class ELFT> void DynamicTable<ELFT>::finalize() {
  StringTable<ELFT> *dynamicStringTable = _dynamicSymbolTable->getStringTable();
  this->_link = dynamicStringTable->ordinal();
  if (this->_outputSection) {
    this->_outputSection->setType(this->_type);
    this->_outputSection->setInfo(this->_info);
    this->_outputSection->setLink(this->_link);
  }
}

template <class ELFT> void DynamicTable<ELFT>::updateDynamicTable() {
  StringTable<ELFT> *dynamicStringTable = _dynamicSymbolTable->getStringTable();
  _entries[_dt_hash].d_un.d_val = _hashTable->virtualAddr();
  _entries[_dt_strtab].d_un.d_val = dynamicStringTable->virtualAddr();
  _entries[_dt_symtab].d_un.d_val = _dynamicSymbolTable->virtualAddr();
  _entries[_dt_strsz].d_un.d_val = dynamicStringTable->memSize();
  _entries[_dt_syment].d_un.d_val = _dynamicSymbolTable->getEntSize();
  auto initArray = _layout.findOutputSection(".init_array");
  if (initArray) {
    _entries[_dt_init_array].d_un.d_val = initArray->virtualAddr();
    _entries[_dt_init_arraysz].d_un.d_val = initArray->memSize();
  }
  auto finiArray = _layout.findOutputSection(".fini_array");
  if (finiArray) {
    _entries[_dt_fini_array].d_un.d_val = finiArray->virtualAddr();
    _entries[_dt_fini_arraysz].d_un.d_val = finiArray->memSize();
  }
  if (const auto *al = getInitAtomLayout())
    _entries[_dt_init].d_un.d_val = getAtomVirtualAddress(al);
  if (const auto *al = getFiniAtomLayout())
    _entries[_dt_fini].d_un.d_val = getAtomVirtualAddress(al);
  if (_layout.hasDynamicRelocationTable()) {
    auto relaTbl = _layout.getDynamicRelocationTable();
    _entries[_dt_rela].d_un.d_val = relaTbl->virtualAddr();
    _entries[_dt_relasz].d_un.d_val = relaTbl->memSize();
    _entries[_dt_relaent].d_un.d_val = relaTbl->getEntSize();
  }
  if (_layout.hasPLTRelocationTable()) {
    auto relaTbl = _layout.getPLTRelocationTable();
    _entries[_dt_jmprel].d_un.d_val = relaTbl->virtualAddr();
    _entries[_dt_pltrelsz].d_un.d_val = relaTbl->memSize();
    auto gotplt = _layout.findOutputSection(".got.plt");
    _entries[_dt_pltgot].d_un.d_val = gotplt->virtualAddr();
  }
}

template <class ELFT>
const AtomLayout *DynamicTable<ELFT>::getInitAtomLayout() {
  auto al = _layout.findAtomLayoutByName(this->_ctx.initFunction());
  if (al && isa<DefinedAtom>(al->_atom))
    return al;
  return nullptr;
}

template <class ELFT>
const AtomLayout *DynamicTable<ELFT>::getFiniAtomLayout() {
  auto al = _layout.findAtomLayoutByName(this->_ctx.finiFunction());
  if (al && isa<DefinedAtom>(al->_atom))
    return al;
  return nullptr;
}

template <class ELFT>
InterpSection<ELFT>::InterpSection(const ELFLinkingContext &ctx, StringRef str,
                                   int32_t order, StringRef interp)
    : Section<ELFT>(ctx, str, "Dynamic:Interp"), _interp(interp) {
  this->setOrder(order);
  this->_alignment = 1;
  // + 1 for null term.
  this->_fsize = interp.size() + 1;
  this->_msize = this->_fsize;
  this->_type = SHT_PROGBITS;
  this->_flags = SHF_ALLOC;
}

template <class ELFT>
void InterpSection<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                                llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  std::memcpy(dest, _interp.data(), _interp.size());
}

template <class ELFT>
HashSection<ELFT>::HashSection(const ELFLinkingContext &ctx, StringRef name,
                               int32_t order)
    : Section<ELFT>(ctx, name, "Dynamic:Hash") {
  this->setOrder(order);
  this->_entSize = 4;
  this->_type = SHT_HASH;
  this->_flags = SHF_ALLOC;
  this->_alignment = ELFT::Is64Bits ? 8 : 4;
  this->_fsize = 0;
  this->_msize = 0;
}

template <class ELFT>
void HashSection<ELFT>::addSymbol(StringRef name, uint32_t index) {
  SymbolTableEntry ste;
  ste._name = name;
  ste._index = index;
  _entries.push_back(ste);
}

/// \brief Set the dynamic symbol table
template <class ELFT>
void HashSection<ELFT>::setSymbolTable(
    const DynamicSymbolTable<ELFT> *symbolTable) {
  _symbolTable = symbolTable;
}

template <class ELFT> void HashSection<ELFT>::doPreFlight() {
  // The number of buckets to use for a certain number of symbols.
  // If there are less than 3 symbols, 1 bucket will be used. If
  // there are less than 17 symbols, 3 buckets will be used, and so
  // forth. The bucket numbers are defined by GNU ld. We use the
  // same rules here so we generate hash sections with the same
  // size as those generated by GNU ld.
  uint32_t hashBuckets[] = {1,     3,     17,    37,     67,    97,   131,
                            197,   263,   521,   1031,   2053,  4099, 8209,
                            16411, 32771, 65537, 131101, 262147};
  int hashBucketsCount = sizeof(hashBuckets) / sizeof(uint32_t);

  unsigned int bucketsCount = 0;
  unsigned int dynSymCount = _entries.size();

  // Get the number of buckes that we want to use
  for (int i = 0; i < hashBucketsCount; ++i) {
    if (dynSymCount < hashBuckets[i])
      break;
    bucketsCount = hashBuckets[i];
  }
  _buckets.resize(bucketsCount);
  _chains.resize(_entries.size());

  // Create the hash table for the dynamic linker
  for (auto ai : _entries) {
    unsigned int dynsymIndex = ai._index;
    unsigned int bucketpos = llvm::object::elf_hash(ai._name) % bucketsCount;
    _chains[dynsymIndex] = _buckets[bucketpos];
    _buckets[bucketpos] = dynsymIndex;
  }

  this->_fsize = (2 + _chains.size() + _buckets.size()) * sizeof(uint32_t);
  this->_msize = this->_fsize;
}

template <class ELFT> void HashSection<ELFT>::finalize() {
  this->_link = _symbolTable ? _symbolTable->ordinal() : 0;
  if (this->_outputSection)
    this->_outputSection->setLink(this->_link);
}

template <class ELFT>
void HashSection<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                              llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  Elf_Word bucketChainCounts[2];
  bucketChainCounts[0] = _buckets.size();
  bucketChainCounts[1] = _chains.size();
  std::memcpy(dest, bucketChainCounts, sizeof(bucketChainCounts));
  dest += sizeof(bucketChainCounts);
  // write bucket values
  std::memcpy(dest, _buckets.data(), _buckets.size() * sizeof(Elf_Word));
  dest += _buckets.size() * sizeof(Elf_Word);
  // write chain values
  std::memcpy(dest, _chains.data(), _chains.size() * sizeof(Elf_Word));
}

template <class ELFT>
EHFrameHeader<ELFT>::EHFrameHeader(const ELFLinkingContext &ctx, StringRef name,
                                   TargetLayout<ELFT> &layout, int32_t order)
    : Section<ELFT>(ctx, name, "EHFrameHeader"), _layout(layout) {
  this->setOrder(order);
  this->_entSize = 0;
  this->_type = SHT_PROGBITS;
  this->_flags = SHF_ALLOC;
  this->_alignment = ELFT::Is64Bits ? 8 : 4;
  // Minimum size for empty .eh_frame_hdr.
  this->_fsize = 1 + 1 + 1 + 1 + 4;
  this->_msize = this->_fsize;
}

template <class ELFT> void EHFrameHeader<ELFT>::doPreFlight() {
  // TODO: Generate a proper binary search table.
}

template <class ELFT> void EHFrameHeader<ELFT>::finalize() {
  OutputSection<ELFT> *s = _layout.findOutputSection(".eh_frame");
  OutputSection<ELFT> *h = _layout.findOutputSection(".eh_frame_hdr");
  if (s && h)
    _ehFrameOffset = s->virtualAddr() - (h->virtualAddr() + 4);
}

template <class ELFT>
void EHFrameHeader<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                                llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  int pos = 0;
  dest[pos++] = 1; // version
  dest[pos++] = llvm::dwarf::DW_EH_PE_pcrel |
                llvm::dwarf::DW_EH_PE_sdata4; // eh_frame_ptr_enc
  dest[pos++] = llvm::dwarf::DW_EH_PE_omit;   // fde_count_enc
  dest[pos++] = llvm::dwarf::DW_EH_PE_omit;   // table_enc
  *reinterpret_cast<typename llvm::object::ELFFile<ELFT>::Elf_Sword *>(
      dest + pos) = _ehFrameOffset;
}

#define INSTANTIATE(klass)        \
  template class klass<ELF32LE>;  \
  template class klass<ELF32BE>;  \
  template class klass<ELF64LE>;  \
  template class klass<ELF64BE>

INSTANTIATE(AtomSection);
INSTANTIATE(DynamicSymbolTable);
INSTANTIATE(DynamicTable);
INSTANTIATE(EHFrameHeader);
INSTANTIATE(HashSection);
INSTANTIATE(InterpSection);
INSTANTIATE(OutputSection);
INSTANTIATE(RelocationTable);
INSTANTIATE(Section);
INSTANTIATE(StringTable);
INSTANTIATE(SymbolTable);

} // end namespace elf
} // end namespace lld
