//===- lib/ReaderWriter/ELF/Mips/MipsRelocationPass.cpp -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsELFFile.h"
#include "MipsLinkingContext.h"
#include "MipsRelocationPass.h"
#include "MipsTargetHandler.h"

#include "llvm/ADT/DenseSet.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm::ELF;

// Lazy resolver
static const uint8_t mipsGot0AtomContent[] = {
  0x00, 0x00, 0x00, 0x00
};

// Module pointer
static const uint8_t mipsGotModulePointerAtomContent[] = {
  0x00, 0x00, 0x00, 0x80
};

// TLS GD Entry
static const uint8_t mipsGotTlsGdAtomContent[] = {
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00
};

// PLT0 entry
static const uint8_t mipsPlt0AtomContent[] = {
  0x00, 0x00, 0x1c, 0x3c, // lui   $28, %hi(&GOTPLT[0])
  0x00, 0x00, 0x99, 0x8f, // lw    $25, %lo(&GOTPLT[0])($28)
  0x00, 0x00, 0x9c, 0x27, // addiu $28, $28, %lo(&GOTPLT[0])
  0x23, 0xc0, 0x1c, 0x03, // subu  $24, $24, $28
  0x21, 0x78, 0xe0, 0x03, // move  $15, $31
  0x82, 0xc0, 0x18, 0x00, // srl   $24, $24, 2
  0x09, 0xf8, 0x20, 0x03, // jalr  $25
  0xfe, 0xff, 0x18, 0x27  // subu  $24, $24, 2
};

// Regular PLT entry
static const uint8_t mipsPltAAtomContent[] = {
  0x00, 0x00, 0x0f, 0x3c, // lui   $15, %hi(.got.plt entry)
  0x00, 0x00, 0xf9, 0x8d, // l[wd] $25, %lo(.got.plt entry)($15)
  0x08, 0x00, 0x20, 0x03, // jr    $25
  0x00, 0x00, 0xf8, 0x25  // addiu $24, $15, %lo(.got.plt entry)
};

// LA25 stub entry
static const uint8_t mipsLA25AtomContent[] = {
  0x00, 0x00, 0x19, 0x3c, // lui   $25, %hi(func)
  0x00, 0x00, 0x00, 0x08, // j     func
  0x00, 0x00, 0x39, 0x27, // addiu $25, $25, %lo(func)
  0x00, 0x00, 0x00, 0x00  // nop
};

namespace {

/// \brief Abstract base class represent MIPS GOT entries.
class MipsGOTAtom : public GOTAtom {
public:
  MipsGOTAtom(const File &f) : GOTAtom(f, ".got") {}

  Alignment alignment() const override { return Alignment(2); }
};

/// \brief MIPS GOT entry initialized by zero.
class GOT0Atom : public MipsGOTAtom {
public:
  GOT0Atom(const File &f) : MipsGOTAtom(f) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsGot0AtomContent);
  }
};

/// \brief MIPS GOT entry initialized by zero.
class GOTModulePointerAtom : public MipsGOTAtom {
public:
  GOTModulePointerAtom(const File &f) : MipsGOTAtom(f) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsGotModulePointerAtomContent);
  }
};

/// \brief MIPS GOT TLS GD entry.
class GOTTLSGdAtom : public MipsGOTAtom {
public:
  GOTTLSGdAtom(const File &f) : MipsGOTAtom(f) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsGotTlsGdAtomContent);
  }
};

class PLT0Atom : public PLTAtom {
public:
  PLT0Atom(const File &f) : PLTAtom(f, ".plt") {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsPlt0AtomContent);
  }
};

class PLTAAtom : public PLTAtom {
public:
  PLTAAtom(const File &f) : PLTAtom(f, ".plt") {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsPltAAtomContent);
  }
};

/// \brief MIPS GOT PLT entry
class GOTPLTAtom : public GOTAtom {
public:
  GOTPLTAtom(const File &f) : GOTAtom(f, ".got.plt") {}

  Alignment alignment() const override { return Alignment(2); }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsGot0AtomContent);
  }
};

/// \brief LA25 stub atom
class LA25Atom : public PLTAtom {
public:
  LA25Atom(const File &f) : PLTAtom(f, ".text") {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsLA25AtomContent);
  }
};

class RelocationPassFile : public SimpleFile {
public:
  RelocationPassFile(const ELFLinkingContext &ctx)
      : SimpleFile("RelocationPassFile") {
    setOrdinal(ctx.getNextOrdinalAndIncrement());
  }

  llvm::BumpPtrAllocator _alloc;
};

template <typename ELFT> class RelocationPass : public Pass {
public:
  RelocationPass(MipsLinkingContext &ctx);

  void perform(std::unique_ptr<MutableFile> &mf) override;

private:
  /// \brief Reference to the linking context.
  const MipsLinkingContext &_ctx;

  /// \brief Owner of all the Atoms created by this pass.
  RelocationPassFile _file;

  /// \brief Map Atoms and addend to local GOT entries.
  typedef std::pair<const Atom *, int64_t> LocalGotMapKeyT;
  llvm::DenseMap<LocalGotMapKeyT, GOTAtom *> _gotLocalMap;

  /// \brief Map Atoms to global GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotGlobalMap;

  /// \brief Map Atoms to TLS GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotTLSMap;

  /// \brief Map Atoms to TLS GD GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotTLSGdMap;

  /// \brief GOT entry for the R_MIPS_TLS_LDM relocation.
  GOTTLSGdAtom *_gotLDMEntry;

  /// \brief the list of local GOT atoms.
  std::vector<GOTAtom *> _localGotVector;

  /// \brief the list of global GOT atoms.
  std::vector<GOTAtom *> _globalGotVector;

  /// \brief the list of TLS GOT atoms.
  std::vector<GOTAtom *> _tlsGotVector;

  /// \brief Map Atoms to their PLT entries.
  llvm::DenseMap<const Atom *, PLTAtom *> _pltMap;

  /// \brief Map Atoms to their Object entries.
  llvm::DenseMap<const Atom *, ObjectAtom *> _objectMap;

  /// \brief Map Atoms to their LA25 entries.
  llvm::DenseMap<const Atom *, LA25Atom *> _la25Map;

  /// \brief Atoms referenced by static relocations.
  llvm::DenseSet<const Atom *> _hasStaticRelocations;

  /// \brief Atoms require pointers equality.
  llvm::DenseSet<const Atom *> _requiresPtrEquality;

  /// \brief References which are candidates for converting
  /// to the R_MIPS_REL32 relocation.
  std::vector<Reference *> _rel32Candidates;

  /// \brief the list of PLT atoms.
  std::vector<PLTAtom *> _pltVector;

  /// \brief the list of GOTPLT atoms.
  std::vector<GOTAtom *> _gotpltVector;

  /// \brief the list of Object entries.
  std::vector<ObjectAtom *> _objectVector;

  /// \brief the list of LA25 entries.
  std::vector<LA25Atom *> _la25Vector;

  /// \brief Handle a specific reference.
  void handleReference(const MipsELFDefinedAtom<ELFT> &atom, Reference &ref);

  /// \brief Collect information about the reference to use it
  /// later in the handleReference() routine.
  void collectReferenceInfo(const MipsELFDefinedAtom<ELFT> &atom,
                            Reference &ref);

  void handlePlain(Reference &ref);
  void handle26(Reference &ref);
  void handleGOT(Reference &ref);
  void handleGPRel(const MipsELFDefinedAtom<ELFT> &atom, Reference &ref);

  const GOTAtom *getLocalGOTEntry(const Reference &ref);
  const GOTAtom *getGlobalGOTEntry(const Atom *a);
  const GOTAtom *getTLSGOTEntry(const Atom *a);
  const GOTAtom *getTLSGdGOTEntry(const Atom *a);
  const GOTAtom *getTLSLdmGOTEntry(const Atom *a);
  PLTAtom *getPLTEntry(const Atom *a);
  const LA25Atom *getLA25Entry(const Atom *a);
  const ObjectAtom *getObjectEntry(const SharedLibraryAtom *a);

  bool isLocal(const Atom *a) const;
  bool isLocalCall(const Atom *a) const;
  bool isDynamic(const Atom *atom) const;
  bool requireLA25Stub(const Atom *a) const;
  bool requirePLTEntry(Reference &ref);
  bool requireCopy(Reference &ref);
  void configurePLTReference(Reference &ref);
  void createPLTHeader();
  bool mightBeDynamic(const MipsELFDefinedAtom<ELFT> &atom,
                      const Reference &ref) const;

  static void addSingleReference(SimpleELFDefinedAtom *src, const Atom *tgt,
                                 uint16_t relocType);
};

template <typename ELFT>
RelocationPass<ELFT>::RelocationPass(MipsLinkingContext &ctx)
    : _ctx(ctx), _file(ctx), _gotLDMEntry(nullptr) {
  _localGotVector.push_back(new (_file._alloc) GOT0Atom(_file));
  _localGotVector.push_back(new (_file._alloc) GOTModulePointerAtom(_file));
}

template <typename ELFT>
void RelocationPass<ELFT>::perform(std::unique_ptr<MutableFile> &mf) {
  for (const auto &atom : mf->defined())
    for (const auto &ref : *atom)
      collectReferenceInfo(*cast<MipsELFDefinedAtom<ELFT>>(atom),
                           const_cast<Reference &>(*ref));

  // Process all references.
  for (const auto &atom : mf->defined())
    for (const auto &ref : *atom)
      handleReference(*cast<MipsELFDefinedAtom<ELFT>>(atom),
                      const_cast<Reference &>(*ref));

  // Create R_MIPS_REL32 relocations.
  for (auto *ref : _rel32Candidates) {
    if (!isDynamic(ref->target()))
      continue;
    if (_pltMap.count(ref->target()))
      continue;
    ref->setKindValue(R_MIPS_REL32);
    if (!isLocalCall(ref->target()))
      getGlobalGOTEntry(ref->target());
  }

  uint64_t ordinal = 0;

  for (auto &got : _localGotVector) {
    DEBUG_WITH_TYPE("MipsGOT", llvm::dbgs() << "[ GOT ] Adding L "
                                            << got->name() << "\n");
    got->setOrdinal(ordinal++);
    mf->addAtom(*got);
  }

  for (auto &got : _globalGotVector) {
    DEBUG_WITH_TYPE("MipsGOT", llvm::dbgs() << "[ GOT ] Adding G "
                                            << got->name() << "\n");
    got->setOrdinal(ordinal++);
    mf->addAtom(*got);
  }

  for (auto &got : _tlsGotVector) {
    got->setOrdinal(ordinal++);
    mf->addAtom(*got);
  }

  for (auto &plt : _pltVector) {
    DEBUG_WITH_TYPE("MipsGOT", llvm::dbgs() << "[ PLT ] Adding " << plt->name()
                                            << "\n");
    plt->setOrdinal(ordinal++);
    mf->addAtom(*plt);
  }

  for (auto &gotplt : _gotpltVector) {
    DEBUG_WITH_TYPE("MipsGOT", llvm::dbgs() << "[ GOTPLT ] Adding "
                                            << gotplt->name() << "\n");
    gotplt->setOrdinal(ordinal++);
    mf->addAtom(*gotplt);
  }

  for (auto obj : _objectVector) {
    obj->setOrdinal(ordinal++);
    mf->addAtom(*obj);
  }

  for (auto la25 : _la25Vector) {
    la25->setOrdinal(ordinal++);
    mf->addAtom(*la25);
  }
}

template <typename ELFT>
void RelocationPass<ELFT>::handleReference(const MipsELFDefinedAtom<ELFT> &atom,
                                           Reference &ref) {
  if (!ref.target())
    return;
  if (ref.kindNamespace() != lld::Reference::KindNamespace::ELF)
    return;
  assert(ref.kindArch() == Reference::KindArch::Mips);
  switch (ref.kindValue()) {
  case R_MIPS_32:
  case R_MIPS_PC32:
  case R_MIPS_HI16:
  case R_MIPS_LO16:
    // FIXME (simon): Handle dynamic/static linking differently.
    handlePlain(ref);
    break;
  case R_MIPS_26:
    handle26(ref);
    break;
  case R_MIPS_GOT16:
  case R_MIPS_CALL16:
    handleGOT(ref);
    break;
  case R_MIPS_GPREL32:
    handleGPRel(atom, ref);
    break;
  case R_MIPS_TLS_DTPREL_HI16:
  case R_MIPS_TLS_DTPREL_LO16:
    ref.setAddend(ref.addend() - atom.file().getDTPOffset());
    break;
  case R_MIPS_TLS_TPREL_HI16:
  case R_MIPS_TLS_TPREL_LO16:
    ref.setAddend(ref.addend() - atom.file().getTPOffset());
    break;
  case R_MIPS_TLS_GD:
    ref.setTarget(getTLSGdGOTEntry(ref.target()));
    break;
  case R_MIPS_TLS_LDM:
    ref.setTarget(getTLSLdmGOTEntry(ref.target()));
    break;
  case R_MIPS_TLS_GOTTPREL:
    ref.setTarget(getTLSGOTEntry(ref.target()));
    break;
  }
}

template <typename ELFT>
static bool isConstrainSym(const MipsELFDefinedAtom<ELFT> &atom, Reference &ref) {
  if ((atom.section()->sh_flags & SHF_ALLOC) == 0)
    return false;
  switch (ref.kindValue()) {
  case R_MIPS_NONE:
  case R_MIPS_JALR:
  case R_MIPS_GPREL32:
    return false;
  default:
    return true;
  }
}

template <typename ELFT>
void
RelocationPass<ELFT>::collectReferenceInfo(const MipsELFDefinedAtom<ELFT> &atom,
                                           Reference &ref) {
  if (!ref.target())
    return;
  if (ref.kindNamespace() != lld::Reference::KindNamespace::ELF)
    return;
  if (!isConstrainSym(atom, ref))
    return;

  if (mightBeDynamic(atom, ref))
    _rel32Candidates.push_back(&ref);
  else
    _hasStaticRelocations.insert(ref.target());

  if (ref.kindValue() != R_MIPS_CALL16 && ref.kindValue() != R_MIPS_26)
    _requiresPtrEquality.insert(ref.target());
}

template <typename ELFT>
bool RelocationPass<ELFT>::isLocal(const Atom *a) const {
  if (auto *da = dyn_cast<DefinedAtom>(a))
    return da->scope() == Atom::scopeTranslationUnit;
  return false;
}

template <typename ELFT>
static bool isMipsReadonly(const MipsELFDefinedAtom<ELFT> &atom) {
  auto secFlags = atom.section()->sh_flags;
  auto secType = atom.section()->sh_type;

  if ((secFlags & SHF_ALLOC) == 0)
    return false;
  if (secType == SHT_NOBITS)
    return false;
  if ((secFlags & SHF_WRITE) != 0)
    return false;
  return true;
}

template <typename ELFT>
bool RelocationPass<ELFT>::mightBeDynamic(const MipsELFDefinedAtom<ELFT> &atom,
                                          const Reference &ref) const {
  auto refKind = ref.kindValue();

  if (refKind == R_MIPS_CALL16 || refKind == R_MIPS_GOT16)
    return true;

  if (refKind != R_MIPS_32)
    return false;
  if ((atom.section()->sh_flags & SHF_ALLOC) == 0)
    return false;

  if (_ctx.getOutputELFType() == llvm::ELF::ET_DYN)
    return true;
  if (!isMipsReadonly(atom))
    return true;
  if (atom.file().isPIC())
    return true;

  return false;
}

template <typename ELFT>
bool RelocationPass<ELFT>::requirePLTEntry(Reference &ref) {
  if (!_hasStaticRelocations.count(ref.target()))
    return false;
  const auto *sa = dyn_cast<ELFDynamicAtom<ELFT>>(ref.target());
  if (sa && sa->type() != SharedLibraryAtom::Type::Code)
    return false;
  const auto *da = dyn_cast<ELFDefinedAtom<ELFT>>(ref.target());
  if (da && da->contentType() != DefinedAtom::typeCode)
    return false;
  if (isLocalCall(ref.target()))
    return false;
  return true;
}

template <typename ELFT>
bool RelocationPass<ELFT>::requireCopy(Reference &ref) {
  if (!_hasStaticRelocations.count(ref.target()))
    return false;
  const auto *sa = dyn_cast<ELFDynamicAtom<ELFT>>(ref.target());
  return sa && sa->type() == SharedLibraryAtom::Type::Data;
}

template <typename ELFT>
void RelocationPass<ELFT>::configurePLTReference(Reference &ref) {
  const Atom *atom = ref.target();

  auto *plt = getPLTEntry(atom);
  ref.setTarget(plt);

  if (_hasStaticRelocations.count(atom) && _requiresPtrEquality.count(atom))
    addSingleReference(plt, atom, LLD_R_MIPS_STO_PLT);
}

template <typename ELFT>
bool RelocationPass<ELFT>::isDynamic(const Atom *atom) const {
  const auto *da = dyn_cast<const DefinedAtom>(atom);
  if (da && da->dynamicExport() == DefinedAtom::dynamicExportAlways)
    return true;

  const auto *sa = dyn_cast<SharedLibraryAtom>(atom);
  if (sa)
    return true;

  if (_ctx.getOutputELFType() == llvm::ELF::ET_DYN) {
    if (da && da->scope() != DefinedAtom::scopeTranslationUnit)
      return true;

    const auto *ua = dyn_cast<UndefinedAtom>(atom);
    if (ua)
      return true;
  }

  return false;
}

template <typename ELFT>
void RelocationPass<ELFT>::handlePlain(Reference &ref) {
  if (!isDynamic(ref.target()))
      return;

  if (requirePLTEntry(ref))
    configurePLTReference(ref);
  else if (requireCopy(ref))
    ref.setTarget(getObjectEntry(cast<SharedLibraryAtom>(ref.target())));
}

template <typename ELFT> void RelocationPass<ELFT>::handle26(Reference &ref) {
  if (ref.kindValue() == R_MIPS_26 && !isLocal(ref.target())) {
    ref.setKindValue(LLD_R_MIPS_GLOBAL_26);

    if (requireLA25Stub(ref.target()))
      const_cast<Reference &>(ref).setTarget(getLA25Entry(ref.target()));
  }

  const auto *sla = dyn_cast<SharedLibraryAtom>(ref.target());
  if (sla && sla->type() == SharedLibraryAtom::Type::Code)
    configurePLTReference(ref);
}

template <typename ELFT> void RelocationPass<ELFT>::handleGOT(Reference &ref) {
  if (isLocalCall(ref.target()))
    ref.setTarget(getLocalGOTEntry(ref));
  else
    ref.setTarget(getGlobalGOTEntry(ref.target()));
}

template <typename ELFT>
void RelocationPass<ELFT>::handleGPRel(const MipsELFDefinedAtom<ELFT> &atom,
                                       Reference &ref) {
  assert(ref.kindValue() == R_MIPS_GPREL32);
  ref.setAddend(ref.addend() + atom.file().getGP0());
}

template <typename ELFT>
bool RelocationPass<ELFT>::isLocalCall(const Atom *a) const {
  Atom::Scope scope;
  if (auto *da = dyn_cast<DefinedAtom>(a))
    scope = da->scope();
  else if (auto *aa = dyn_cast<AbsoluteAtom>(a))
    scope = aa->scope();
  else
    return false;

  // Local and hidden symbols must be local.
  if (scope == Atom::scopeTranslationUnit || scope == Atom::scopeLinkageUnit)
    return true;

  // Calls to external symbols defined in an executable file resolved locally.
  if (_ctx.getOutputELFType() == llvm::ELF::ET_EXEC)
    return true;

  return false;
}

template <typename ELFT>
bool RelocationPass<ELFT>::requireLA25Stub(const Atom *a) const {
  if (auto *da = dyn_cast<DefinedAtom>(a))
    return static_cast<const MipsELFDefinedAtom<ELFT> *>(da)->file().isPIC();
  return false;
}

template <typename ELFT>
const GOTAtom *RelocationPass<ELFT>::getLocalGOTEntry(const Reference &ref) {
  const Atom *a = ref.target();
  LocalGotMapKeyT key(a, ref.addend());

  auto got = _gotLocalMap.find(key);
  if (got != _gotLocalMap.end())
    return got->second;

  auto ga = new (_file._alloc) GOT0Atom(_file);
  _gotLocalMap[key] = ga;

  _localGotVector.push_back(ga);

  if (isLocal(a))
    ga->addReferenceELF_Mips(LLD_R_MIPS_32_HI16, 0, a, ref.addend());
  else
    ga->addReferenceELF_Mips(R_MIPS_32, 0, a, 0);

  DEBUG_WITH_TYPE("MipsGOT", {
    ga->_name = "__got_";
    ga->_name += a->name();
    llvm::dbgs() << "[ GOT ] Create L " << a->name() << "\n";
  });

  return ga;
}

template <typename ELFT>
const GOTAtom *RelocationPass<ELFT>::getGlobalGOTEntry(const Atom *a) {
  auto got = _gotGlobalMap.find(a);
  if (got != _gotGlobalMap.end())
    return got->second;

  auto ga = new (_file._alloc) GOT0Atom(_file);
  _gotGlobalMap[a] = ga;

  _globalGotVector.push_back(ga);
  ga->addReferenceELF_Mips(LLD_R_MIPS_GLOBAL_GOT, 0, a, 0);

  if (const DefinedAtom *da = dyn_cast<DefinedAtom>(a))
    ga->addReferenceELF_Mips(R_MIPS_32, 0, da, 0);

  DEBUG_WITH_TYPE("MipsGOT", {
    ga->_name = "__got_";
    ga->_name += a->name();
    llvm::dbgs() << "[ GOT ] Create G " << a->name() << "\n";
  });

  return ga;
}

template <typename ELFT>
const GOTAtom *RelocationPass<ELFT>::getTLSGOTEntry(const Atom *a) {
  auto got = _gotTLSMap.find(a);
  if (got != _gotTLSMap.end())
    return got->second;

  auto ga = new (_file._alloc) GOT0Atom(_file);
  _gotTLSMap[a] = ga;

  _tlsGotVector.push_back(ga);
  ga->addReferenceELF_Mips(R_MIPS_TLS_TPREL32, 0, a, 0);

  return ga;
}

template <typename ELFT>
const GOTAtom *RelocationPass<ELFT>::getTLSGdGOTEntry(const Atom *a) {
  auto got = _gotTLSGdMap.find(a);
  if (got != _gotTLSGdMap.end())
    return got->second;

  auto ga = new (_file._alloc) GOTTLSGdAtom(_file);
  _gotTLSGdMap[a] = ga;

  _tlsGotVector.push_back(ga);
  ga->addReferenceELF_Mips(R_MIPS_TLS_DTPMOD32, 0, a, 0);
  ga->addReferenceELF_Mips(R_MIPS_TLS_DTPREL32, 4, a, 0);

  return ga;
}

template <typename ELFT>
const GOTAtom *RelocationPass<ELFT>::getTLSLdmGOTEntry(const Atom *a) {
  if (_gotLDMEntry)
    return _gotLDMEntry;

  _gotLDMEntry = new (_file._alloc) GOTTLSGdAtom(_file);
  _tlsGotVector.push_back(_gotLDMEntry);
  _gotLDMEntry->addReferenceELF_Mips(R_MIPS_TLS_DTPMOD32, 0, _gotLDMEntry, 0);

  return _gotLDMEntry;
}

template <typename ELFT> void RelocationPass<ELFT>::createPLTHeader() {
  assert(_pltVector.empty() && _gotpltVector.empty());

  auto pa = new (_file._alloc) PLT0Atom(_file);
  _pltVector.push_back(pa);

  auto ga0 = new (_file._alloc) GOTPLTAtom(_file);
  _gotpltVector.push_back(ga0);
  auto ga1 = new (_file._alloc) GOTPLTAtom(_file);
  _gotpltVector.push_back(ga1);

  // Setup reference to fixup the PLT0 entry.
  pa->addReferenceELF_Mips(LLD_R_MIPS_HI16, 0, ga0, 0);
  pa->addReferenceELF_Mips(LLD_R_MIPS_LO16, 4, ga0, 0);
  pa->addReferenceELF_Mips(LLD_R_MIPS_LO16, 8, ga0, 0);

  DEBUG_WITH_TYPE("MipsGOT", {
    pa->_name = "__plt0";
    llvm::dbgs() << "[ PLT ] Create PLT0\n";
    ga0->_name = "__gotplt0";
    llvm::dbgs() << "[ GOTPLT ] Create GOTPLT0\n";
    ga1->_name = "__gotplt1";
    llvm::dbgs() << "[ GOTPLT ] Create GOTPLT1\n";
  });
}

template <typename ELFT>
void RelocationPass<ELFT>::addSingleReference(SimpleELFDefinedAtom *src,
                                              const Atom *tgt,
                                              uint16_t relocType) {
  for (const auto &r : *src)
    if (r->kindNamespace() == lld::Reference::KindNamespace::ELF &&
        r->kindValue() == relocType && r->target() == tgt)
      break;
  src->addReferenceELF_Mips(relocType, 0, tgt, 0);
}

template <typename ELFT>
PLTAtom *RelocationPass<ELFT>::getPLTEntry(const Atom *a) {
  auto plt = _pltMap.find(a);
  if (plt != _pltMap.end())
    return plt->second;

  if (_pltVector.empty())
    createPLTHeader();

  auto pa = new (_file._alloc) PLTAAtom(_file);
  _pltMap[a] = pa;
  _pltVector.push_back(pa);

  auto ga = new (_file._alloc) GOTPLTAtom(_file);
  _gotpltVector.push_back(ga);

  // Setup reference to fixup the PLT entry.
  pa->addReferenceELF_Mips(LLD_R_MIPS_HI16, 0, ga, 0);
  pa->addReferenceELF_Mips(LLD_R_MIPS_LO16, 4, ga, 0);
  pa->addReferenceELF_Mips(LLD_R_MIPS_LO16, 12, ga, 0);

  // Setup reference to assign initial value to the .got.plt entry.
  ga->addReferenceELF_Mips(R_MIPS_32, 0, _pltVector.front(), 0);
  // Create dynamic relocation to adjust the .got.plt entry at runtime.
  ga->addReferenceELF_Mips(R_MIPS_JUMP_SLOT, 0, a, 0);

  DEBUG_WITH_TYPE("MipsGOT", {
    pa->_name = "__plt_";
    pa->_name += a->name();
    llvm::dbgs() << "[ PLT ] Create " << a->name() << "\n";
    ga->_name = "__got_plt_";
    ga->_name += a->name();
    llvm::dbgs() << "[ GOTPLT ] Create " << a->name() << "\n";
  });

  return pa;
}

template <typename ELFT>
const LA25Atom *RelocationPass<ELFT>::getLA25Entry(const Atom *a) {
  auto la25 = _la25Map.find(a);
  if (la25 != _la25Map.end())
    return la25->second;

  auto sa = new (_file._alloc) LA25Atom(_file);
  _la25Map[a] = sa;
  _la25Vector.push_back(sa);

  // Setup reference to fixup the LA25 stub entry.
  sa->addReferenceELF_Mips(R_MIPS_HI16, 0, a, 0);
  sa->addReferenceELF_Mips(R_MIPS_26, 4, a, 0);
  sa->addReferenceELF_Mips(R_MIPS_LO16, 8, a, 0);

  DEBUG_WITH_TYPE("MipsGOT", {
    sa->_name = ".pic.";
    sa->_name += a->name();
  });

  return sa;
}

template <typename ELFT>
const ObjectAtom *
RelocationPass<ELFT>::getObjectEntry(const SharedLibraryAtom *a) {
  auto obj = _objectMap.find(a);
  if (obj != _objectMap.end())
    return obj->second;

  auto oa = new (_file._alloc) ObjectAtom(_file, a);
  oa->addReferenceELF_Mips(R_MIPS_COPY, 0, oa, 0);
  oa->_name = a->name();
  oa->_size = a->size();

  _objectMap[a] = oa;
  _objectVector.push_back(oa);

  return oa;
}

} // end anon namespace

std::unique_ptr<Pass>
lld::elf::createMipsRelocationPass(MipsLinkingContext &ctx) {
  switch (ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Pass>(new RelocationPass<Mips32ElELFType>(ctx));
  case llvm::ELF::ET_REL:
    return std::unique_ptr<Pass>();
  default:
    llvm_unreachable("Unhandled output file type");
  }
}
