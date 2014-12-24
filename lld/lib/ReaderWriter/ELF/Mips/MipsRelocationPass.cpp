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

// Regular PLT0 entry
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

// microMIPS PLT0 entry
static const uint8_t micromipsPlt0AtomContent[] = {
  0x80, 0x79, 0x00, 0x00, // addiupc $3,  (&GOTPLT[0]) - .
  0x23, 0xff, 0x00, 0x00, // lw      $25, 0($3)
  0x35, 0x05,             // subu    $2,  $2, $3
  0x25, 0x25,             // srl     $2,  $2, 2
  0x02, 0x33, 0xfe, 0xff, // subu    $24, $2, 2
  0xff, 0x0d,             // move    $15, $31
  0xf9, 0x45,             // jalrs   $25
  0x83, 0x0f,             // move    $28, $3
  0x00, 0x0c              // nop
};

// Regular PLT entry
static const uint8_t mipsPltAAtomContent[] = {
  0x00, 0x00, 0x0f, 0x3c, // lui   $15, %hi(.got.plt entry)
  0x00, 0x00, 0xf9, 0x8d, // l[wd] $25, %lo(.got.plt entry)($15)
  0x08, 0x00, 0x20, 0x03, // jr    $25
  0x00, 0x00, 0xf8, 0x25  // addiu $24, $15, %lo(.got.plt entry)
};

// microMIPS PLT entry
static const uint8_t micromipsPltAtomContent[] = {
  0x00, 0x79, 0x00, 0x00, // addiupc $2, (.got.plt entry) - .
  0x22, 0xff, 0x00, 0x00, // lw $25, 0($2)
  0x99, 0x45,             // jr $25
  0x02, 0x0f              // move $24, $2
};

// LA25 stub entry
static const uint8_t mipsLA25AtomContent[] = {
  0x00, 0x00, 0x19, 0x3c, // lui   $25, %hi(func)
  0x00, 0x00, 0x00, 0x08, // j     func
  0x00, 0x00, 0x39, 0x27, // addiu $25, $25, %lo(func)
  0x00, 0x00, 0x00, 0x00  // nop
};

// microMIPS LA25 stub entry
static const uint8_t micromipsLA25AtomContent[] = {
  0xb9, 0x41, 0x00, 0x00, // lui   $25, %hi(func)
  0x00, 0xd4, 0x00, 0x00, // j     func
  0x39, 0x33, 0x00, 0x00, // addiu $25, $25, %lo(func)
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

class GOTPLTAtom : public GOTAtom {
public:
  GOTPLTAtom(const File &f) : GOTAtom(f, ".got.plt") {}
  GOTPLTAtom(const Atom *a, const File &f) : GOTAtom(f, ".got.plt") {
    // Create dynamic relocation to adjust the .got.plt entry at runtime.
    addReferenceELF_Mips(R_MIPS_JUMP_SLOT, 0, a, 0);
  }

  /// Setup reference to assign initial value to the .got.plt entry.
  void setPLT0(const PLTAtom *plt0) {
    addReferenceELF_Mips(R_MIPS_32, 0, plt0, 0);
  }

  Alignment alignment() const override { return Alignment(2); }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsGot0AtomContent);
  }
};

class PLT0Atom : public PLTAtom {
public:
  PLT0Atom(const Atom *got, const File &f) : PLTAtom(f, ".plt") {
    // Setup reference to fixup the PLT0 entry.
    addReferenceELF_Mips(LLD_R_MIPS_HI16, 0, got, 0);
    addReferenceELF_Mips(LLD_R_MIPS_LO16, 4, got, 0);
    addReferenceELF_Mips(LLD_R_MIPS_LO16, 8, got, 0);
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsPlt0AtomContent);
  }
};

class PLT0MicroAtom : public PLTAtom {
public:
  PLT0MicroAtom(const Atom *got, const File &f) : PLTAtom(f, ".plt") {
    // Setup reference to fixup the PLT0 entry.
    addReferenceELF_Mips(R_MICROMIPS_PC23_S2, 0, got, 0);
  }

  CodeModel codeModel() const override { return codeMipsMicro; }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(micromipsPlt0AtomContent);
  }
};

class PLTAAtom : public PLTAtom {
public:
  PLTAAtom(const GOTPLTAtom *got, const File &f) : PLTAtom(f, ".plt") {
    // Setup reference to fixup the PLT entry.
    addReferenceELF_Mips(LLD_R_MIPS_HI16, 0, got, 0);
    addReferenceELF_Mips(LLD_R_MIPS_LO16, 4, got, 0);
    addReferenceELF_Mips(LLD_R_MIPS_LO16, 12, got, 0);
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsPltAAtomContent);
  }
};

class PLTMicroAtom : public PLTAtom {
public:
  PLTMicroAtom(const GOTPLTAtom *got, const File &f) : PLTAtom(f, ".plt") {
    // Setup reference to fixup the microMIPS PLT entry.
    addReferenceELF_Mips(R_MICROMIPS_PC23_S2, 0, got, 0);
  }

  Alignment alignment() const override { return Alignment(1); }
  CodeModel codeModel() const override { return codeMipsMicro; }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(micromipsPltAtomContent);
  }
};

class LA25Atom : public PLTAtom {
public:
  LA25Atom(const File &f) : PLTAtom(f, ".text") {}
};

class LA25RegAtom : public LA25Atom {
public:
  LA25RegAtom(const Atom *a, const File &f) : LA25Atom(f) {
    // Setup reference to fixup the LA25 stub entry.
    addReferenceELF_Mips(R_MIPS_HI16, 0, a, 0);
    addReferenceELF_Mips(R_MIPS_26, 4, a, 0);
    addReferenceELF_Mips(R_MIPS_LO16, 8, a, 0);
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsLA25AtomContent);
  }
};

class LA25MicroAtom : public LA25Atom {
public:
  LA25MicroAtom(const Atom *a, const File &f) : LA25Atom(f) {
    // Setup reference to fixup the microMIPS LA25 stub entry.
    addReferenceELF_Mips(R_MICROMIPS_HI16, 0, a, 0);
    addReferenceELF_Mips(R_MICROMIPS_26_S1, 4, a, 0);
    addReferenceELF_Mips(R_MICROMIPS_LO16, 8, a, 0);
  }

  CodeModel codeModel() const override { return codeMipsMicro; }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(micromipsLA25AtomContent);
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

  /// \brief GOT entry for the R_xxxMIPS_TLS_LDM relocations.
  GOTTLSGdAtom *_gotLDMEntry;

  /// \brief the list of local GOT atoms.
  std::vector<GOTAtom *> _localGotVector;

  /// \brief the list of global GOT atoms.
  std::vector<GOTAtom *> _globalGotVector;

  /// \brief the list of TLS GOT atoms.
  std::vector<GOTAtom *> _tlsGotVector;

  /// \brief Map Atoms to their GOTPLT entries.
  llvm::DenseMap<const Atom *, GOTPLTAtom *> _gotpltMap;

  /// \brief Map Atoms to their PLT entries.
  llvm::DenseMap<const Atom *, PLTAAtom *> _pltRegMap;
  llvm::DenseMap<const Atom *, PLTMicroAtom *> _pltMicroMap;

  /// \brief Map Atoms to their Object entries.
  llvm::DenseMap<const Atom *, ObjectAtom *> _objectMap;

  /// \brief Map Atoms to their LA25 entries.
  llvm::DenseMap<const Atom *, LA25RegAtom *> _la25RegMap;
  llvm::DenseMap<const Atom *, LA25MicroAtom *> _la25MicroMap;

  /// \brief Atoms referenced by static relocations.
  llvm::DenseSet<const Atom *> _hasStaticRelocations;

  /// \brief Atoms require pointers equality.
  llvm::DenseSet<const Atom *> _requiresPtrEquality;

  /// \brief References which are candidates for converting
  /// to the R_MIPS_REL32 relocation.
  std::vector<Reference *> _rel32Candidates;

  /// \brief the list of PLT atoms.
  std::vector<PLTAtom *> _pltRegVector;
  std::vector<PLTAtom *> _pltMicroVector;

  /// \brief the list of GOTPLT atoms.
  std::vector<GOTPLTAtom *> _gotpltVector;

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

  void handlePlain(const MipsELFDefinedAtom<ELFT> &atom, Reference &ref);
  void handle26(const MipsELFDefinedAtom<ELFT> &atom, Reference &ref);
  void handleGOT(Reference &ref);
  void handleGPRel(const MipsELFDefinedAtom<ELFT> &atom, Reference &ref);

  const GOTAtom *getLocalGOTEntry(const Reference &ref);
  const GOTAtom *getGlobalGOTEntry(const Atom *a);
  const GOTAtom *getTLSGOTEntry(const Atom *a);
  const GOTAtom *getTLSGdGOTEntry(const Atom *a);
  const GOTAtom *getTLSLdmGOTEntry(const Atom *a);
  const GOTPLTAtom *getGOTPLTEntry(const Atom *a);
  const PLTAtom *getPLTEntry(const Atom *a);
  const PLTAtom *getPLTRegEntry(const Atom *a);
  const PLTAtom *getPLTMicroEntry(const Atom *a);
  const LA25Atom *getLA25Entry(const Atom *target, bool isMicroMips);
  const LA25Atom *getLA25RegEntry(const Atom *a);
  const LA25Atom *getLA25MicroEntry(const Atom *a);
  const ObjectAtom *getObjectEntry(const SharedLibraryAtom *a);

  PLTAtom *createPLTHeader(bool isMicroMips);

  bool isLocal(const Atom *a) const;
  bool isLocalCall(const Atom *a) const;
  bool isDynamic(const Atom *atom) const;
  bool requireLA25Stub(const Atom *a) const;
  bool requirePLTEntry(Reference &ref);
  bool requireCopy(Reference &ref);
  bool mightBeDynamic(const MipsELFDefinedAtom<ELFT> &atom,
                      const Reference &ref) const;
  bool hasPLTEntry(const Atom *atom) const;
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
    if (!isDynamic(ref->target()) || hasPLTEntry(ref->target()))
      continue;
    ref->setKindValue(R_MIPS_REL32);
    if (!isLocalCall(ref->target()))
      getGlobalGOTEntry(ref->target());
  }

  uint64_t ordinal = 0;

  for (auto &got : _localGotVector) {
    got->setOrdinal(ordinal++);
    mf->addAtom(*got);
  }

  for (auto &got : _globalGotVector) {
    got->setOrdinal(ordinal++);
    mf->addAtom(*got);
  }

  for (auto &got : _tlsGotVector) {
    got->setOrdinal(ordinal++);
    mf->addAtom(*got);
  }

  // Create and emit PLT0 entry.
  PLTAtom *plt0Atom = nullptr;
  if (!_pltRegVector.empty())
    plt0Atom = createPLTHeader(false);
  else if (!_pltMicroVector.empty())
    plt0Atom = createPLTHeader(true);

  if (plt0Atom) {
    plt0Atom->setOrdinal(ordinal++);
    mf->addAtom(*plt0Atom);
  }

  // Emit regular PLT entries firts.
  for (auto &plt : _pltRegVector) {
    plt->setOrdinal(ordinal++);
    mf->addAtom(*plt);
  }

  // microMIPS PLT entries come after regular ones.
  for (auto &plt : _pltMicroVector) {
    plt->setOrdinal(ordinal++);
    mf->addAtom(*plt);
  }

  // Assign PLT0 to GOTPLT entries.
  assert(_gotpltMap.empty() || plt0Atom);
  for (auto &a: _gotpltMap)
    a.second->setPLT0(plt0Atom);

  for (auto &gotplt : _gotpltVector) {
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
  case R_MICROMIPS_HI16:
  case R_MICROMIPS_LO16:
    // FIXME (simon): Handle dynamic/static linking differently.
    handlePlain(atom, ref);
    break;
  case R_MIPS_26:
  case R_MICROMIPS_26_S1:
    handle26(atom, ref);
    break;
  case R_MIPS_GOT16:
  case R_MIPS_CALL16:
  case R_MICROMIPS_GOT16:
  case R_MICROMIPS_CALL16:
    handleGOT(ref);
    break;
  case R_MIPS_GPREL32:
    handleGPRel(atom, ref);
    break;
  case R_MIPS_TLS_DTPREL_HI16:
  case R_MIPS_TLS_DTPREL_LO16:
  case R_MICROMIPS_TLS_DTPREL_HI16:
  case R_MICROMIPS_TLS_DTPREL_LO16:
    ref.setAddend(ref.addend() - atom.file().getDTPOffset());
    break;
  case R_MIPS_TLS_TPREL_HI16:
  case R_MIPS_TLS_TPREL_LO16:
  case R_MICROMIPS_TLS_TPREL_HI16:
  case R_MICROMIPS_TLS_TPREL_LO16:
    ref.setAddend(ref.addend() - atom.file().getTPOffset());
    break;
  case R_MIPS_TLS_GD:
  case R_MICROMIPS_TLS_GD:
    ref.setTarget(getTLSGdGOTEntry(ref.target()));
    break;
  case R_MIPS_TLS_LDM:
  case R_MICROMIPS_TLS_LDM:
    ref.setTarget(getTLSLdmGOTEntry(ref.target()));
    break;
  case R_MIPS_TLS_GOTTPREL:
  case R_MICROMIPS_TLS_GOTTPREL:
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
  case R_MICROMIPS_JALR:
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

  if (ref.kindValue() != R_MIPS_CALL16 &&
      ref.kindValue() != R_MICROMIPS_CALL16 && ref.kindValue() != R_MIPS_26 &&
      ref.kindValue() != R_MICROMIPS_26_S1)
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

  if (refKind == R_MIPS_CALL16 || refKind == R_MIPS_GOT16 ||
      refKind == R_MICROMIPS_CALL16 || refKind == R_MICROMIPS_GOT16)
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
bool RelocationPass<ELFT>::hasPLTEntry(const Atom *atom) const {
  return _pltRegMap.count(atom) || _pltMicroMap.count(atom);
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
static bool isMicroMips(const MipsELFDefinedAtom<ELFT> &atom) {
  return atom.codeModel() == DefinedAtom::codeMipsMicro ||
         atom.codeModel() == DefinedAtom::codeMipsMicroPIC;
}

template <typename ELFT>
const LA25Atom *RelocationPass<ELFT>::getLA25Entry(const Atom *target,
                                                   bool isMicroMips) {
  return isMicroMips ? getLA25MicroEntry(target) : getLA25RegEntry(target);
}

template <typename ELFT>
const PLTAtom *RelocationPass<ELFT>::getPLTEntry(const Atom *a) {
  bool hasMicroCode = _ctx.getMergedELFFlags() & EF_MIPS_MICROMIPS;

  // If file contains microMIPS code try to reuse compressed PLT entry...
  if (hasMicroCode) {
    auto microPLT = _pltMicroMap.find(a);
    if (microPLT != _pltMicroMap.end())
      return microPLT->second;
  }

  // ... then try to reuse a regular PLT entry ...
  auto regPLT = _pltRegMap.find(a);
  if (regPLT != _pltRegMap.end())
    return regPLT->second;

  // ... and finally prefer to create new compressed PLT entry.
  return hasMicroCode ? getPLTMicroEntry(a) : getPLTRegEntry(a);
}

template <typename ELFT>
void RelocationPass<ELFT>::handlePlain(const MipsELFDefinedAtom<ELFT> &atom,
                                       Reference &ref) {
  if (!isDynamic(ref.target()))
      return;

  if (requirePLTEntry(ref))
    ref.setTarget(getPLTEntry(ref.target()));
  else if (requireCopy(ref))
    ref.setTarget(getObjectEntry(cast<SharedLibraryAtom>(ref.target())));
}

template <typename ELFT>
void RelocationPass<ELFT>::handle26(const MipsELFDefinedAtom<ELFT> &atom,
                                    Reference &ref) {
  bool isMicro = ref.kindValue() == R_MICROMIPS_26_S1;

  const auto *sla = dyn_cast<SharedLibraryAtom>(ref.target());
  if (sla && sla->type() == SharedLibraryAtom::Type::Code)
    ref.setTarget(isMicro ? getPLTMicroEntry(sla) : getPLTRegEntry(sla));

  if (requireLA25Stub(ref.target()))
    ref.setTarget(getLA25Entry(ref.target(), isMicro));

  if (!isLocal(ref.target()))
    switch (ref.kindValue()) {
    case R_MIPS_26:
      ref.setKindValue(LLD_R_MIPS_GLOBAL_26);
      break;
    case R_MICROMIPS_26_S1:
      ref.setKindValue(LLD_R_MICROMIPS_GLOBAL_26_S1);
      break;
    default:
      llvm_unreachable("Unexpected relocation kind");
    }
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
  if (isLocal(a))
    return false;
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

template <typename ELFT>
PLTAtom *RelocationPass<ELFT>::createPLTHeader(bool isMicroMips) {
  auto ga1 = new (_file._alloc) GOTPLTAtom(_file);
  _gotpltVector.insert(_gotpltVector.begin(), ga1);
  auto ga0 = new (_file._alloc) GOTPLTAtom(_file);
  _gotpltVector.insert(_gotpltVector.begin(), ga0);

  if (isMicroMips)
    return new (_file._alloc) PLT0MicroAtom(ga0, _file);
  else
    return new (_file._alloc) PLT0Atom(ga0, _file);
}

template <typename ELFT>
const GOTPLTAtom *RelocationPass<ELFT>::getGOTPLTEntry(const Atom *a) {
  auto it = _gotpltMap.find(a);
  if (it != _gotpltMap.end())
    return it->second;

  auto ga = new (_file._alloc) GOTPLTAtom(a, _file);
  _gotpltMap[a] = ga;
  _gotpltVector.push_back(ga);
  return ga;
}

template <typename ELFT>
const PLTAtom *RelocationPass<ELFT>::getPLTRegEntry(const Atom *a) {
  auto plt = _pltRegMap.find(a);
  if (plt != _pltRegMap.end())
    return plt->second;

  auto pa = new (_file._alloc) PLTAAtom(getGOTPLTEntry(a), _file);
  _pltRegMap[a] = pa;
  _pltRegVector.push_back(pa);

  // Check that 'a' dynamic symbol table record should point to the PLT.
  if (_hasStaticRelocations.count(a) && _requiresPtrEquality.count(a))
    pa->addReferenceELF_Mips(LLD_R_MIPS_STO_PLT, 0, a, 0);

  return pa;
}

template <typename ELFT>
const PLTAtom *RelocationPass<ELFT>::getPLTMicroEntry(const Atom *a) {
  auto plt = _pltMicroMap.find(a);
  if (plt != _pltMicroMap.end())
    return plt->second;

  auto pa = new (_file._alloc) PLTMicroAtom(getGOTPLTEntry(a), _file);
  _pltMicroMap[a] = pa;
  _pltMicroVector.push_back(pa);

  // Check that 'a' dynamic symbol table record should point to the PLT.
  if (_hasStaticRelocations.count(a) && _requiresPtrEquality.count(a))
    pa->addReferenceELF_Mips(LLD_R_MIPS_STO_PLT, 0, a, 0);

  return pa;
}

template <typename ELFT>
const LA25Atom *RelocationPass<ELFT>::getLA25RegEntry(const Atom *a) {
  auto la25 = _la25RegMap.find(a);
  if (la25 != _la25RegMap.end())
    return la25->second;

  auto sa = new (_file._alloc) LA25RegAtom(a, _file);
  _la25RegMap[a] = sa;
  _la25Vector.push_back(sa);

  return sa;
}

template <typename ELFT>
const LA25Atom *RelocationPass<ELFT>::getLA25MicroEntry(const Atom *a) {
  auto la25 = _la25MicroMap.find(a);
  if (la25 != _la25MicroMap.end())
    return la25->second;

  auto sa = new (_file._alloc) LA25MicroAtom(a, _file);
  _la25MicroMap[a] = sa;
  _la25Vector.push_back(sa);

  return sa;
}

template <typename ELFT>
const ObjectAtom *
RelocationPass<ELFT>::getObjectEntry(const SharedLibraryAtom *a) {
  auto obj = _objectMap.find(a);
  if (obj != _objectMap.end())
    return obj->second;

  auto oa = new (_file._alloc) ObjectAtom(_file);
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
