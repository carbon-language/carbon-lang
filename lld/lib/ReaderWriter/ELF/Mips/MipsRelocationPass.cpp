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
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00
};

// Module pointer
static const uint8_t mipsGotModulePointerAtomContent[] = {
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x80
};

// TLS GD Entry
static const uint8_t mipsGotTlsGdAtomContent[] = {
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00,
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

// R6 PLT entry
static const uint8_t mipsR6PltAAtomContent[] = {
  0x00, 0x00, 0x0f, 0x3c, // lui   $15, %hi(.got.plt entry)
  0x00, 0x00, 0xf9, 0x8d, // l[wd] $25, %lo(.got.plt entry)($15)
  0x09, 0x00, 0x20, 0x03, // jr    $25
  0x00, 0x00, 0xf8, 0x25  // addiu $24, $15, %lo(.got.plt entry)
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

  Alignment alignment() const override { return 4; }
};

/// \brief MIPS GOT entry initialized by zero.
template <typename ELFT> class GOT0Atom : public MipsGOTAtom {
public:
  GOT0Atom(const File &f) : MipsGOTAtom(f) {}

  ArrayRef<uint8_t> rawContent() const override;
};

template <> ArrayRef<uint8_t> GOT0Atom<ELF32LE>::rawContent() const {
  return llvm::makeArrayRef(mipsGot0AtomContent).slice(4);
}
template <> ArrayRef<uint8_t> GOT0Atom<ELF64LE>::rawContent() const {
  return llvm::makeArrayRef(mipsGot0AtomContent);
}

/// \brief MIPS GOT entry initialized by zero.
template <typename ELFT> class GOTModulePointerAtom : public MipsGOTAtom {
public:
  GOTModulePointerAtom(const File &f) : MipsGOTAtom(f) {}

  ArrayRef<uint8_t> rawContent() const override;
};

template <>
ArrayRef<uint8_t> GOTModulePointerAtom<ELF32LE>::rawContent() const {
  return llvm::makeArrayRef(mipsGotModulePointerAtomContent).slice(4);
}
template <>
ArrayRef<uint8_t> GOTModulePointerAtom<ELF64LE>::rawContent() const {
  return llvm::makeArrayRef(mipsGotModulePointerAtomContent);
}

/// \brief MIPS GOT TLS GD entry.
template <typename ELFT> class GOTTLSGdAtom : public MipsGOTAtom {
public:
  GOTTLSGdAtom(const File &f) : MipsGOTAtom(f) {}

  ArrayRef<uint8_t> rawContent() const override;
};

template <> ArrayRef<uint8_t> GOTTLSGdAtom<ELF32LE>::rawContent() const {
    return llvm::makeArrayRef(mipsGotTlsGdAtomContent).slice(8);
}

template <> ArrayRef<uint8_t> GOTTLSGdAtom<ELF64LE>::rawContent() const {
    return llvm::makeArrayRef(mipsGotTlsGdAtomContent);
}

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

  Alignment alignment() const override { return 4; }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsGot0AtomContent).slice(4);
  }
};

class PLT0Atom : public PLTAtom {
public:
  PLT0Atom(const Atom *got, const File &f) : PLTAtom(f, ".plt") {
    // Setup reference to fixup the PLT0 entry.
    addReferenceELF_Mips(R_MIPS_HI16, 0, got, 0);
    addReferenceELF_Mips(R_MIPS_LO16, 4, got, 0);
    addReferenceELF_Mips(R_MIPS_LO16, 8, got, 0);
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
    addReferenceELF_Mips(R_MIPS_HI16, 0, got, 0);
    addReferenceELF_Mips(R_MIPS_LO16, 4, got, 0);
    addReferenceELF_Mips(R_MIPS_LO16, 12, got, 0);
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsPltAAtomContent);
  }
};

class PLTR6Atom : public PLTAAtom {
public:
  PLTR6Atom(const GOTPLTAtom *got, const File &f) : PLTAAtom(got, f) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(mipsR6PltAAtomContent);
  }
};

class PLTMicroAtom : public PLTAtom {
public:
  PLTMicroAtom(const GOTPLTAtom *got, const File &f) : PLTAtom(f, ".plt") {
    // Setup reference to fixup the microMIPS PLT entry.
    addReferenceELF_Mips(R_MICROMIPS_PC23_S2, 0, got, 0);
  }

  Alignment alignment() const override { return 2; }
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

class MipsGlobalOffsetTableAtom : public GlobalOffsetTableAtom {
public:
  MipsGlobalOffsetTableAtom(const File &f) : GlobalOffsetTableAtom(f) {}

  StringRef customSectionName() const override { return ".got"; }
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

  std::error_code perform(SimpleFile &mf) override;

private:
  /// \brief Reference to the linking context.
  const MipsLinkingContext &_ctx;

  /// \brief Owner of all the Atoms created by this pass.
  RelocationPassFile _file;

  /// \brief Map Atoms and addend to local GOT entries.
  typedef std::pair<const Atom *, int64_t> LocalGotMapKeyT;
  llvm::DenseMap<LocalGotMapKeyT, GOTAtom *> _gotLocalMap;
  llvm::DenseMap<LocalGotMapKeyT, GOTAtom *> _gotLocalPageMap;

  /// \brief Map Atoms to global GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotGlobalMap;

  /// \brief Map Atoms to TLS GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotTLSMap;

  /// \brief Map Atoms to TLS GD GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotTLSGdMap;

  /// \brief GOT entry for the R_xxxMIPS_TLS_LDM relocations.
  GOTTLSGdAtom<ELFT> *_gotLDMEntry = nullptr;

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
  std::error_code collectReferenceInfo(const MipsELFDefinedAtom<ELFT> &atom,
                                       Reference &ref);

  /// \brief Check that the relocation is valid for the current linking mode.
  std::error_code validateRelocation(const DefinedAtom &atom,
                                     const Reference &ref) const;

  void handlePlain(const MipsELFDefinedAtom<ELFT> &atom, Reference &ref);
  void handle26(const MipsELFDefinedAtom<ELFT> &atom, Reference &ref);
  void handleGOT(Reference &ref);

  const GOTAtom *getLocalGOTEntry(const Reference &ref);
  const GOTAtom *getLocalGOTPageEntry(const Reference &ref);
  const GOTAtom *getGlobalGOTEntry(const Atom *a);
  const GOTAtom *getTLSGOTEntry(const Atom *a, Reference::Addend addend);
  const GOTAtom *getTLSGdGOTEntry(const Atom *a, Reference::Addend addend);
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
  bool requirePLTEntry(const Atom *a) const;
  bool requireCopy(const Atom *a) const;
  bool mightBeDynamic(const MipsELFDefinedAtom<ELFT> &atom,
                      Reference::KindValue refKind) const;
  bool hasPLTEntry(const Atom *atom) const;

  /// \brief Linked files contain microMIPS code.
  bool isMicroMips();
  /// \brief Linked files contain MIPS R6 code.
  bool isMipsR6();
};

template <typename ELFT>
RelocationPass<ELFT>::RelocationPass(MipsLinkingContext &ctx)
    : _ctx(ctx), _file(ctx) {
  _localGotVector.push_back(new (_file._alloc) GOT0Atom<ELFT>(_file));
  _localGotVector.push_back(new (_file._alloc)
                                GOTModulePointerAtom<ELFT>(_file));
}

template <typename ELFT>
std::error_code RelocationPass<ELFT>::perform(SimpleFile &mf) {
  for (const auto &atom : mf.defined())
    for (const auto &ref : *atom) {
      const auto &da = *cast<MipsELFDefinedAtom<ELFT>>(atom);
      if (auto ec = collectReferenceInfo(da, const_cast<Reference &>(*ref)))
        return ec;
    }

  // Process all references.
  for (const auto &atom : mf.defined())
    for (const auto &ref : *atom)
      handleReference(*cast<MipsELFDefinedAtom<ELFT>>(atom),
                      const_cast<Reference &>(*ref));

  // Create R_MIPS_REL32 relocations.
  for (auto *ref : _rel32Candidates) {
    if (!isDynamic(ref->target()) || hasPLTEntry(ref->target()))
      continue;
    ref->setKindValue(R_MIPS_REL32);
    if (ELFT::Is64Bits)
      static_cast<MipsELFReference<ELFT> *>(ref)->setTag(R_MIPS_64);
    if (!isLocalCall(ref->target()))
      getGlobalGOTEntry(ref->target());
  }

  uint64_t ordinal = 0;

  if (!_localGotVector.empty() || !_globalGotVector.empty() ||
      !_tlsGotVector.empty()) {
    SimpleDefinedAtom *ga = new (_file._alloc) MipsGlobalOffsetTableAtom(_file);
    ga->setOrdinal(ordinal++);
    mf.addAtom(*ga);
  }

  for (auto &got : _localGotVector) {
    got->setOrdinal(ordinal++);
    mf.addAtom(*got);
  }

  for (auto &got : _globalGotVector) {
    got->setOrdinal(ordinal++);
    mf.addAtom(*got);
  }

  for (auto &got : _tlsGotVector) {
    got->setOrdinal(ordinal++);
    mf.addAtom(*got);
  }

  // Create and emit PLT0 entry.
  PLTAtom *plt0Atom = nullptr;
  if (!_pltRegVector.empty())
    plt0Atom = createPLTHeader(false);
  else if (!_pltMicroVector.empty())
    plt0Atom = createPLTHeader(true);

  if (plt0Atom) {
    plt0Atom->setOrdinal(ordinal++);
    mf.addAtom(*plt0Atom);
  }

  // Emit regular PLT entries firts.
  for (auto &plt : _pltRegVector) {
    plt->setOrdinal(ordinal++);
    mf.addAtom(*plt);
  }

  // microMIPS PLT entries come after regular ones.
  for (auto &plt : _pltMicroVector) {
    plt->setOrdinal(ordinal++);
    mf.addAtom(*plt);
  }

  // Assign PLT0 to GOTPLT entries.
  assert(_gotpltMap.empty() || plt0Atom);
  for (auto &a: _gotpltMap)
    a.second->setPLT0(plt0Atom);

  for (auto &gotplt : _gotpltVector) {
    gotplt->setOrdinal(ordinal++);
    mf.addAtom(*gotplt);
  }

  for (auto obj : _objectVector) {
    obj->setOrdinal(ordinal++);
    mf.addAtom(*obj);
  }

  for (auto la25 : _la25Vector) {
    la25->setOrdinal(ordinal++);
    mf.addAtom(*la25);
  }

  return std::error_code();
}

template <typename ELFT>
void RelocationPass<ELFT>::handleReference(const MipsELFDefinedAtom<ELFT> &atom,
                                           Reference &ref) {
  if (!ref.target())
    return;
  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return;
  assert(ref.kindArch() == Reference::KindArch::Mips);
  switch (ref.kindValue()) {
  case R_MIPS_32:
  case R_MIPS_PC32:
  case R_MIPS_HI16:
  case R_MIPS_LO16:
  case R_MIPS_PCHI16:
  case R_MIPS_PCLO16:
  case R_MICROMIPS_HI16:
  case R_MICROMIPS_LO16:
  case R_MICROMIPS_HI0_LO16:
    // FIXME (simon): Handle dynamic/static linking differently.
    handlePlain(atom, ref);
    break;
  case R_MIPS_26:
  case R_MICROMIPS_26_S1:
    handle26(atom, ref);
    break;
  case R_MIPS_EH:
  case R_MIPS_GOT16:
  case R_MIPS_CALL16:
  case R_MIPS_GOT_HI16:
  case R_MIPS_GOT_LO16:
  case R_MIPS_CALL_HI16:
  case R_MIPS_CALL_LO16:
  case R_MICROMIPS_GOT16:
  case R_MICROMIPS_CALL16:
  case R_MICROMIPS_GOT_HI16:
  case R_MICROMIPS_GOT_LO16:
  case R_MICROMIPS_CALL_HI16:
  case R_MICROMIPS_CALL_LO16:
  case R_MIPS_GOT_DISP:
  case R_MIPS_GOT_PAGE:
  case R_MICROMIPS_GOT_DISP:
  case R_MICROMIPS_GOT_PAGE:
    handleGOT(ref);
    break;
  case R_MIPS_GOT_OFST:
  case R_MICROMIPS_GOT_OFST:
    // Nothing to do. We create GOT page entry in the R_MIPS_GOT_PAGE handler.
    break;
  case R_MIPS_GPREL16:
  case R_MICROMIPS_GPREL16:
  case R_MICROMIPS_GPREL7_S2:
  case R_MIPS_LITERAL:
  case R_MICROMIPS_LITERAL:
    if (isLocal(ref.target()))
      ref.setAddend(ref.addend() + atom.file().getGP0());
    break;
  case R_MIPS_GPREL32:
    ref.setAddend(ref.addend() + atom.file().getGP0());
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
    ref.setTarget(getTLSGdGOTEntry(ref.target(), ref.addend()));
    break;
  case R_MIPS_TLS_LDM:
  case R_MICROMIPS_TLS_LDM:
    ref.setTarget(getTLSLdmGOTEntry(ref.target()));
    break;
  case R_MIPS_TLS_GOTTPREL:
  case R_MICROMIPS_TLS_GOTTPREL:
    ref.setTarget(getTLSGOTEntry(ref.target(), ref.addend()));
    break;
  }
}

template <typename ELFT>
static bool isConstrainSym(const MipsELFDefinedAtom<ELFT> &atom,
                           Reference::KindValue refKind) {
  if ((atom.section()->sh_flags & SHF_ALLOC) == 0)
    return false;
  switch (refKind) {
  case R_MIPS_NONE:
  case R_MIPS_JALR:
  case R_MICROMIPS_JALR:
  case R_MIPS_GPREL16:
  case R_MIPS_GPREL32:
  case R_MICROMIPS_GPREL16:
  case R_MICROMIPS_GPREL7_S2:
  case R_MIPS_LITERAL:
  case R_MICROMIPS_LITERAL:
    return false;
  default:
    return true;
  }
}

template <typename ELFT>
std::error_code
RelocationPass<ELFT>::collectReferenceInfo(const MipsELFDefinedAtom<ELFT> &atom,
                                           Reference &ref) {
  if (!ref.target())
    return std::error_code();
  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return std::error_code();

  auto refKind = ref.kindValue();
  if (refKind == R_MIPS_EH && this->_ctx.mipsPcRelEhRel())
    ref.setKindValue(R_MIPS_PC32);

  if (auto ec = validateRelocation(atom, ref))
    return ec;

  if (!isConstrainSym(atom, refKind))
    return std::error_code();

  if (mightBeDynamic(atom, refKind))
    _rel32Candidates.push_back(&ref);
  else
    _hasStaticRelocations.insert(ref.target());

  if (refKind != R_MIPS_CALL16 && refKind != R_MICROMIPS_CALL16 &&
      refKind != R_MIPS_26 && refKind != R_MICROMIPS_26_S1 &&
      refKind != R_MIPS_GOT_HI16 && refKind != R_MIPS_GOT_LO16 &&
      refKind != R_MIPS_CALL_HI16 && refKind != R_MIPS_CALL_LO16 &&
      refKind != R_MICROMIPS_GOT_HI16 && refKind != R_MICROMIPS_GOT_LO16 &&
      refKind != R_MICROMIPS_CALL_HI16 && refKind != R_MICROMIPS_CALL_LO16 &&
      refKind != R_MIPS_EH)
    _requiresPtrEquality.insert(ref.target());
  return std::error_code();
}

static std::error_code
make_reject_for_shared_lib_reloc_error(const ELFLinkingContext &ctx,
                                       const DefinedAtom &atom,
                                       const Reference &ref) {
  StringRef kindValStr = "unknown";
  ctx.registry().referenceKindToString(ref.kindNamespace(), ref.kindArch(),
                                       ref.kindValue(), kindValStr);

  return make_dynamic_error_code(Twine(kindValStr) + " (" +
                                 Twine(ref.kindValue()) +
                                 ") relocation cannot be used "
                                 "when making a shared object, recompile " +
                                 atom.file().path() + " with -fPIC");
}

static std::error_code
make_external_gprel32_reloc_error(const ELFLinkingContext &ctx,
                                  const DefinedAtom &atom,
                                  const Reference &ref) {
  return make_dynamic_error_code(
      "R_MIPS_GPREL32 (12) relocation cannot be used "
      "against external symbol " +
      ref.target()->name() + " in file " + atom.file().path());
}

template <typename ELFT>
std::error_code
RelocationPass<ELFT>::validateRelocation(const DefinedAtom &atom,
                                         const Reference &ref) const {
  if (!ref.target())
    return std::error_code();

  if (ref.kindValue() == R_MIPS_GPREL32 && !isLocal(ref.target()))
    return make_external_gprel32_reloc_error(this->_ctx, atom, ref);

  if (this->_ctx.getOutputELFType() != ET_DYN)
    return std::error_code();

  switch (ref.kindValue()) {
  case R_MIPS16_HI16:
  case R_MIPS_HI16:
  case R_MIPS_HIGHER:
  case R_MIPS_HIGHEST:
  case R_MICROMIPS_HI16:
  case R_MICROMIPS_HIGHER:
  case R_MICROMIPS_HIGHEST:
    // For shared object we accepts "high" relocations
    // against the "_gp_disp" symbol only.
    if (ref.target()->name() != "_gp_disp")
      return make_reject_for_shared_lib_reloc_error(this->_ctx, atom, ref);
    break;
  case R_MIPS16_26:
  case R_MIPS_26:
  case R_MICROMIPS_26_S1:
    // These relocations are position dependent
    // and not acceptable in a shared object.
    return make_reject_for_shared_lib_reloc_error(this->_ctx, atom, ref);
  default:
    break;
  }
  return std::error_code();
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
                                          Reference::KindValue refKind) const {
  if (refKind == R_MIPS_CALL16 || refKind == R_MIPS_GOT16 ||
      refKind == R_MICROMIPS_CALL16 || refKind == R_MICROMIPS_GOT16 ||
      refKind == R_MIPS_GOT_HI16 || refKind == R_MIPS_GOT_LO16 ||
      refKind == R_MIPS_CALL_HI16 || refKind == R_MIPS_CALL_LO16 ||
      refKind == R_MICROMIPS_GOT_HI16 || refKind == R_MICROMIPS_GOT_LO16 ||
      refKind == R_MICROMIPS_CALL_HI16 || refKind == R_MICROMIPS_CALL_LO16)
    return true;

  if (refKind != R_MIPS_32 && refKind != R_MIPS_64)
    return false;
  if ((atom.section()->sh_flags & SHF_ALLOC) == 0)
    return false;

  if (_ctx.getOutputELFType() == ET_DYN)
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

template <typename ELFT> bool RelocationPass<ELFT>::isMicroMips() {
  TargetHandler &handler = this->_ctx.getTargetHandler();
  return static_cast<MipsTargetHandler<ELFT> &>(handler)
      .getAbiInfoHandler()
      .isMicroMips();
}

template <typename ELFT> bool RelocationPass<ELFT>::isMipsR6() {
  TargetHandler &handler = this->_ctx.getTargetHandler();
  return static_cast<MipsTargetHandler<ELFT> &>(handler)
      .getAbiInfoHandler()
      .isMipsR6();
}

template <typename ELFT>
bool RelocationPass<ELFT>::requirePLTEntry(const Atom *a) const {
  if (!_hasStaticRelocations.count(a))
    return false;
  const auto *sa = dyn_cast<ELFDynamicAtom<ELFT>>(a);
  if (sa && sa->type() != SharedLibraryAtom::Type::Code)
    return false;
  const auto *da = dyn_cast<ELFDefinedAtom<ELFT>>(a);
  if (da && da->contentType() != DefinedAtom::typeCode)
    return false;
  if (isLocalCall(a))
    return false;
  return true;
}

template <typename ELFT>
bool RelocationPass<ELFT>::requireCopy(const Atom *a) const {
  if (!_hasStaticRelocations.count(a))
    return false;
  const auto *sa = dyn_cast<ELFDynamicAtom<ELFT>>(a);
  return sa && sa->type() == SharedLibraryAtom::Type::Data;
}

template <typename ELFT>
bool RelocationPass<ELFT>::isDynamic(const Atom *atom) const {
  const auto *da = dyn_cast<const DefinedAtom>(atom);
  if (da && da->dynamicExport() == DefinedAtom::dynamicExportAlways)
    return true;
  if (isa<SharedLibraryAtom>(atom))
    return true;
  if (_ctx.getOutputELFType() != ET_DYN)
    return false;
  if (da && da->scope() != DefinedAtom::scopeTranslationUnit)
    return true;
  return isa<UndefinedAtom>(atom);
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
  // If file contains microMIPS code try to reuse compressed PLT entry...
  if (isMicroMips()) {
    auto microPLT = _pltMicroMap.find(a);
    if (microPLT != _pltMicroMap.end())
      return microPLT->second;
  }

  // ... then try to reuse a regular PLT entry ...
  auto regPLT = _pltRegMap.find(a);
  if (regPLT != _pltRegMap.end())
    return regPLT->second;

  // ... and finally prefer to create new compressed PLT entry.
  return isMicroMips() ? getPLTMicroEntry(a) : getPLTRegEntry(a);
}

template <typename ELFT>
void RelocationPass<ELFT>::handlePlain(const MipsELFDefinedAtom<ELFT> &atom,
                                       Reference &ref) {
  if (!isDynamic(ref.target()))
      return;

  if (requirePLTEntry(ref.target()))
    ref.setTarget(getPLTEntry(ref.target()));
  else if (requireCopy(ref.target()))
    ref.setTarget(getObjectEntry(cast<SharedLibraryAtom>(ref.target())));
}

template <typename ELFT>
void RelocationPass<ELFT>::handle26(const MipsELFDefinedAtom<ELFT> &atom,
                                    Reference &ref) {
  bool isMicro = ref.kindValue() == R_MICROMIPS_26_S1;
  assert((isMicro || ref.kindValue() == R_MIPS_26) && "Unexpected relocation");

  const auto *sla = dyn_cast<SharedLibraryAtom>(ref.target());
  if (sla && sla->type() == SharedLibraryAtom::Type::Code)
    ref.setTarget(isMicro ? getPLTMicroEntry(sla) : getPLTRegEntry(sla));

  if (requireLA25Stub(ref.target()))
    ref.setTarget(getLA25Entry(ref.target(), isMicro));

  if (!isLocal(ref.target())) {
    if (isMicro)
      ref.setKindValue(LLD_R_MICROMIPS_GLOBAL_26_S1);
    else
      ref.setKindValue(LLD_R_MIPS_GLOBAL_26);
  }
}

template <typename ELFT> void RelocationPass<ELFT>::handleGOT(Reference &ref) {
  if (!isLocalCall(ref.target())) {
    ref.setTarget(getGlobalGOTEntry(ref.target()));
    return;
  }

  if (ref.kindValue() == R_MIPS_GOT_PAGE ||
      ref.kindValue() == R_MICROMIPS_GOT_PAGE)
    ref.setTarget(getLocalGOTPageEntry(ref));
  else if (ref.kindValue() == R_MIPS_GOT_DISP ||
           ref.kindValue() == R_MIPS_GOT_HI16 ||
           ref.kindValue() == R_MIPS_GOT_LO16 ||
           ref.kindValue() == R_MIPS_CALL_HI16 ||
           ref.kindValue() == R_MIPS_CALL_LO16 ||
           ref.kindValue() == R_MICROMIPS_GOT_DISP ||
           ref.kindValue() == R_MICROMIPS_GOT_HI16 ||
           ref.kindValue() == R_MICROMIPS_GOT_LO16 ||
           ref.kindValue() == R_MICROMIPS_CALL_HI16 ||
           ref.kindValue() == R_MICROMIPS_CALL_LO16 ||
           ref.kindValue() == R_MIPS_EH)
    ref.setTarget(getLocalGOTEntry(ref));
  else if (isLocal(ref.target()))
    ref.setTarget(getLocalGOTPageEntry(ref));
  else
    ref.setTarget(getLocalGOTEntry(ref));
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
  if (_ctx.getOutputELFType() == ET_EXEC)
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

  auto ga = new (_file._alloc) GOT0Atom<ELFT>(_file);
  _gotLocalMap[key] = ga;

  _localGotVector.push_back(ga);

  Reference::KindValue relKind = ELFT::Is64Bits ? R_MIPS_64 : R_MIPS_32;
  ga->addReferenceELF_Mips(relKind, 0, a, 0);

  return ga;
}

template <typename ELFT>
const GOTAtom *
RelocationPass<ELFT>::getLocalGOTPageEntry(const Reference &ref) {
  const Atom *a = ref.target();
  LocalGotMapKeyT key(a, ref.addend());

  auto got = _gotLocalPageMap.find(key);
  if (got != _gotLocalPageMap.end())
    return got->second;

  auto ga = new (_file._alloc) GOT0Atom<ELFT>(_file);
  _gotLocalPageMap[key] = ga;

  _localGotVector.push_back(ga);

  Reference::KindValue relKind =
      ELFT::Is64Bits ? LLD_R_MIPS_64_HI16 : LLD_R_MIPS_32_HI16;
  ga->addReferenceELF_Mips(relKind, 0, a, ref.addend());

  return ga;
}

template <typename ELFT>
const GOTAtom *RelocationPass<ELFT>::getGlobalGOTEntry(const Atom *a) {
  auto got = _gotGlobalMap.find(a);
  if (got != _gotGlobalMap.end())
    return got->second;

  auto ga = new (_file._alloc) GOT0Atom<ELFT>(_file);
  _gotGlobalMap[a] = ga;

  _globalGotVector.push_back(ga);
  ga->addReferenceELF_Mips(LLD_R_MIPS_GLOBAL_GOT, 0, a, 0);

  if (const DefinedAtom *da = dyn_cast<DefinedAtom>(a))
    ga->addReferenceELF_Mips(R_MIPS_32, 0, da, 0);

  return ga;
}

template <typename ELFT>
const GOTAtom *RelocationPass<ELFT>::getTLSGOTEntry(const Atom *a,
                                                    Reference::Addend addend) {
  auto got = _gotTLSMap.find(a);
  if (got != _gotTLSMap.end())
    return got->second;

  auto ga = new (_file._alloc) GOT0Atom<ELFT>(_file);
  _gotTLSMap[a] = ga;

  _tlsGotVector.push_back(ga);
  Reference::KindValue relKind =
      ELFT::Is64Bits ? R_MIPS_TLS_TPREL64 : R_MIPS_TLS_TPREL32;
  ga->addReferenceELF_Mips(relKind, 0, a, addend);

  return ga;
}

template <typename ELFT>
const GOTAtom *
RelocationPass<ELFT>::getTLSGdGOTEntry(const Atom *a,
                                       Reference::Addend addend) {
  auto got = _gotTLSGdMap.find(a);
  if (got != _gotTLSGdMap.end())
    return got->second;

  auto ga = new (_file._alloc) GOTTLSGdAtom<ELFT>(_file);
  _gotTLSGdMap[a] = ga;

  _tlsGotVector.push_back(ga);
  if (ELFT::Is64Bits) {
    ga->addReferenceELF_Mips(R_MIPS_TLS_DTPMOD64, 0, a, addend);
    ga->addReferenceELF_Mips(R_MIPS_TLS_DTPREL64, 8, a, addend);
  } else {
    ga->addReferenceELF_Mips(R_MIPS_TLS_DTPMOD32, 0, a, addend);
    ga->addReferenceELF_Mips(R_MIPS_TLS_DTPREL32, 4, a, addend);
  }

  return ga;
}

template <typename ELFT>
const GOTAtom *RelocationPass<ELFT>::getTLSLdmGOTEntry(const Atom *a) {
  if (_gotLDMEntry)
    return _gotLDMEntry;

  _gotLDMEntry = new (_file._alloc) GOTTLSGdAtom<ELFT>(_file);
  _tlsGotVector.push_back(_gotLDMEntry);
  if (ELFT::Is64Bits)
    _gotLDMEntry->addReferenceELF_Mips(R_MIPS_TLS_DTPMOD64, 0, _gotLDMEntry, 0);
  else
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

  PLTAAtom *pa = isMipsR6()
                     ? new (_file._alloc) PLTR6Atom(getGOTPLTEntry(a), _file)
                     : new (_file._alloc) PLTAAtom(getGOTPLTEntry(a), _file);
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

static std::unique_ptr<Pass> createPass(MipsLinkingContext &ctx) {
  switch (ctx.getTriple().getArch()) {
  case llvm::Triple::mipsel:
    return llvm::make_unique<RelocationPass<ELF32LE>>(ctx);
  case llvm::Triple::mips64el:
    return llvm::make_unique<RelocationPass<ELF64LE>>(ctx);
  default:
    llvm_unreachable("Unhandled arch");
  }
}

std::unique_ptr<Pass>
lld::elf::createMipsRelocationPass(MipsLinkingContext &ctx) {
  switch (ctx.getOutputELFType()) {
  case ET_EXEC:
  case ET_DYN:
    return createPass(ctx);
  case ET_REL:
    return nullptr;
  default:
    llvm_unreachable("Unhandled output file type");
  }
}
