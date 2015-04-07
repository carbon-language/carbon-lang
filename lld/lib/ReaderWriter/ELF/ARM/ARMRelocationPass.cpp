//===--------- lib/ReaderWriter/ELF/ARM/ARMRelocationPass.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the relocation processing pass for ARM. This includes
///   GOT and PLT entries, TLS, COPY, and ifunc.
///
/// This also includes additional behavior that gnu-ld and gold implement but
/// which is not specified anywhere.
///
//===----------------------------------------------------------------------===//

#include "ARMRelocationPass.h"
#include "ARMLinkingContext.h"
#include "Atoms.h"
#include "lld/Core/Simple.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm::ELF;

namespace {
// ARM B/BL instructions of static relocation veneer.
// TODO: consider different instruction set for archs below ARMv5
// (one as for Thumb may be used though it's less optimal).
static const uint8_t Veneer_ARM_B_BL_StaticAtomContent[8] = {
    0x04, 0xf0, 0x1f, 0xe5,  // ldr pc, [pc, #-4]
    0x00, 0x00, 0x00, 0x00   // <target_symbol_address>
};

// Thumb B/BL instructions of static relocation veneer.
// TODO: consider different instruction set for archs above ARMv5
// (one as for ARM may be used since it's more optimal).
static const uint8_t Veneer_THM_B_BL_StaticAtomContent[8] = {
    0x78, 0x47,              // bx pc
    0x00, 0x00,              // nop
    0xfe, 0xff, 0xff, 0xea   // b <target_symbol_address>
};

// .got values
static const uint8_t ARMGotAtomContent[4] = {0};

// .plt values (other entries)
static const uint8_t ARMPltAtomContent[12] = {
    0x00, 0xc0, 0x8f, 0xe2,  // add ip, pc, #offset[G0]
    0x00, 0xc0, 0x8c, 0xe2,  // add ip, ip, #offset[G1]
    0x00, 0xf0, 0xbc, 0xe5   // ldr pc, [ip, #offset[G2]]!
};

// Veneer for switching from Thumb to ARM code for PLT entries.
static const uint8_t ARMPltVeneerAtomContent[4] = {
    0x78, 0x47,              // bx pc
    0x00, 0x00               // nop
};

#ifdef NDEBUG
// Determine proper names for mapping symbols.
static std::string getMappingAtomName(DefinedAtom::CodeModel model,
                                      const std::string &part) {
  switch (model) {
  case DefinedAtom::codeARM_a:
    return part.empty() ? "$a" : "$a." + part;
  case DefinedAtom::codeARM_d:
    return part.empty() ? "$d" : "$d." + part;
  case DefinedAtom::codeARM_t:
    return part.empty() ? "$t" : "$t." + part;
  default:
    llvm_unreachable("Wrong code model of mapping atom");
  }
}
#endif

/// \brief Atoms that hold veneer code.
class VeneerAtom : public SimpleELFDefinedAtom {
  StringRef _section;

public:
  VeneerAtom(const File &f, StringRef secName)
      : SimpleELFDefinedAtom(f), _section(secName) {}

  Scope scope() const override { return DefinedAtom::scopeTranslationUnit; }

  SectionChoice sectionChoice() const override {
    return DefinedAtom::sectionBasedOnContent;
  }

  StringRef customSectionName() const override { return _section; }

  ContentType contentType() const override {
    return DefinedAtom::typeCode;
  }

  uint64_t size() const override { return rawContent().size(); }

  ContentPermissions permissions() const override { return permR_X; }

  Alignment alignment() const override { return 4; }

  StringRef name() const override { return _name; }
  std::string _name;
};

/// \brief Atoms that hold veneer for statically relocated
/// ARM B/BL instructions.
class Veneer_ARM_B_BL_StaticAtom : public VeneerAtom {
public:
  Veneer_ARM_B_BL_StaticAtom(const File &f, StringRef secName)
      : VeneerAtom(f, secName) {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(Veneer_ARM_B_BL_StaticAtomContent);
  }
};

/// \brief Atoms that hold veneer for statically relocated
/// Thumb B/BL instructions.
class Veneer_THM_B_BL_StaticAtom : public VeneerAtom {
public:
  Veneer_THM_B_BL_StaticAtom(const File &f, StringRef secName)
      : VeneerAtom(f, secName) {}

  DefinedAtom::CodeModel codeModel() const override {
    return DefinedAtom::codeARMThumb;
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(Veneer_THM_B_BL_StaticAtomContent);
  }
};

/// \brief Atoms that are used by ARM dynamic linking
class ARMGOTAtom : public GOTAtom {
public:
  ARMGOTAtom(const File &f) : GOTAtom(f, ".got") {}

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(ARMGotAtomContent);
  }

  Alignment alignment() const override { return 4; }

protected:
  // Constructor for PLTGOT atom.
  ARMGOTAtom(const File &f, StringRef secName) : GOTAtom(f, secName) {}
};

class ARMGOTPLTAtom : public ARMGOTAtom {
public:
  ARMGOTPLTAtom(const File &f) : ARMGOTAtom(f, ".got.plt") {}
};

/// \brief PLT entry atom.
/// Serves as a mapping symbol in the release mode.
class ARMPLTAtom : public PLTAtom {
public:
  ARMPLTAtom(const File &f, const std::string &name)
      : PLTAtom(f, ".plt") {
#ifndef NDEBUG
    _name = name;
#else
    // Don't move the code to any base classes since
    // virtual codeModel method would return wrong value.
    _name = getMappingAtomName(codeModel(), name);
#endif
  }

  DefinedAtom::CodeModel codeModel() const override {
#ifndef NDEBUG
    return DefinedAtom::codeNA;
#else
    return DefinedAtom::codeARM_a;
#endif
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(ARMPltAtomContent);
  }

  Alignment alignment() const override { return 4; }

  StringRef name() const override { return _name; }

private:
  std::string _name;
};

/// \brief Veneer atom for PLT entry.
/// Serves as a mapping symbol in the release mode.
class ARMPLTVeneerAtom : public PLTAtom {
public:
  ARMPLTVeneerAtom(const File &f, const std::string &name)
      : PLTAtom(f, ".plt") {
#ifndef NDEBUG
    _name = name;
#else
    // Don't move the code to any base classes since
    // virtual codeModel method would return wrong value.
    _name = getMappingAtomName(codeModel(), name);
#endif
  }

  DefinedAtom::CodeModel codeModel() const override {
#ifndef NDEBUG
    return DefinedAtom::codeARMThumb;
#else
    return DefinedAtom::codeARM_t;
#endif
  }

  ArrayRef<uint8_t> rawContent() const override {
    return llvm::makeArrayRef(ARMPltVeneerAtomContent);
  }

  Alignment alignment() const override { return 4; }

  StringRef name() const override { return _name; }

private:
  std::string _name;
};

class ELFPassFile : public SimpleFile {
public:
  ELFPassFile(const ELFLinkingContext &eti) : SimpleFile("ELFPassFile") {
    setOrdinal(eti.getNextOrdinalAndIncrement());
  }

  llvm::BumpPtrAllocator _alloc;
};

/// \brief CRTP base for handling relocations.
template <class Derived> class ARMRelocationPass : public Pass {
  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    DEBUG_WITH_TYPE(
        "ARM", llvm::dbgs() << "\t" << LLVM_FUNCTION_NAME << "()"
            << ":   Name of Defined Atom: " << atom.name().str();
        llvm::dbgs() << "   kindValue: " << ref.kindValue() << "\n");
    if (ref.kindNamespace() != Reference::KindNamespace::ELF)
      return;
    assert(ref.kindArch() == Reference::KindArch::ARM);
    switch (ref.kindValue()) {
    case R_ARM_ABS32:
    case R_ARM_REL32:
    case R_ARM_TARGET1:
    case R_ARM_MOVW_ABS_NC:
    case R_ARM_MOVT_ABS:
    case R_ARM_THM_MOVW_ABS_NC:
    case R_ARM_THM_MOVT_ABS:
    case R_ARM_THM_CALL:
    case R_ARM_CALL:
    case R_ARM_JUMP24:
    case R_ARM_THM_JUMP24:
    case R_ARM_THM_JUMP11:
      static_cast<Derived *>(this)->handleIFUNC(atom, ref);
      static_cast<Derived *>(this)->handleVeneer(atom, ref);
      break;
    case R_ARM_TLS_IE32:
      static_cast<Derived *>(this)->handleTLSIE32(ref);
      break;
    case R_ARM_GOT_BREL:
      static_cast<Derived *>(this)->handleGOT(ref);
      break;
    default:
      break;
    }
  }

protected:
  std::error_code handleVeneer(const DefinedAtom &atom, const Reference &ref) {
    const VeneerAtom *(Derived::*getVeneer)(const DefinedAtom *, StringRef) =
        nullptr;
    const auto kindValue = ref.kindValue();
    switch (kindValue) {
    case R_ARM_JUMP24:
      getVeneer = &Derived::getVeneer_ARM_B_BL;
      break;
    case R_ARM_THM_JUMP24:
      getVeneer = &Derived::getVeneer_THM_B_BL;
      break;
    default:
      return std::error_code();
    }

    // Target symbol and relocated place should have different
    // instruction sets in order a veneer to be generated in between.
    const auto *target = dyn_cast<DefinedAtom>(ref.target());
    if (!target || isThumbCode(target) == isThumbCode(&atom))
      return std::error_code();

    // TODO: For unconditional jump instructions (R_ARM_CALL and R_ARM_THM_CALL)
    // fixup isn't possible without veneer generation for archs below ARMv5.

    // Veneers may only be generated for STT_FUNC target symbols
    // or for symbols located in sections different to the place of relocation.
    StringRef secName = atom.customSectionName();
    if (DefinedAtom::typeCode != target->contentType() &&
        !target->customSectionName().equals(secName)) {
      StringRef kindValStr;
      if (!this->_ctx.registry().referenceKindToString(
              ref.kindNamespace(), ref.kindArch(), kindValue, kindValStr)) {
        kindValStr = "unknown";
      }

      std::string errStr =
          (Twine("Reference of type ") + Twine(kindValue) + " (" + kindValStr +
           ") from " + atom.name() + "+" + Twine(ref.offsetInAtom()) + " to " +
           ref.target()->name() + "+" + Twine(ref.addend()) +
           " cannot be effected without a veneer").str();

      llvm_unreachable(errStr.c_str());
    }

    assert(getVeneer && "The veneer handler is missing");
    const Atom *veneer =
        (static_cast<Derived *>(this)->*getVeneer)(target, secName);

    assert(veneer && "The veneer is not set");
    const_cast<Reference &>(ref).setTarget(veneer);
    return std::error_code();
  }

  std::error_code handleTLSIE32(const Reference &ref) {
    if (const auto *target = dyn_cast<DefinedAtom>(ref.target())) {
      const_cast<Reference &>(ref).setTarget(
          static_cast<Derived *>(this)->getTLSTPOFF32(target));
      return std::error_code();
    }
    llvm_unreachable("R_ARM_TLS_IE32 reloc targets wrong atom type");
  }

  /// \brief Create a GOT entry for TLS with reloc type and addend specified.
  template <Reference::KindValue R_ARM_TLS, Reference::Addend A = 0>
  const GOTAtom *getGOTTLSEntry(const DefinedAtom *da) {
    if (auto got = _gotAtoms.lookup(da))
      return got;
    auto g = new (_file._alloc) ARMGOTAtom(_file);
    g->addReferenceELF_ARM(R_ARM_TLS, 0, da, A);
#ifndef NDEBUG
    g->_name = "__got_tls_";
    g->_name += da->name();
#endif
    _gotAtoms[da] = g;
    return g;
  }

  /// \brief get a veneer for a PLT entry.
  const PLTAtom *getPLTVeneer(const DefinedAtom *da, PLTAtom *pa,
                              StringRef source) {
    std::string name = "__plt_from_thumb";
    name += source;
    name += da->name();
    // Create veneer for PLT entry.
    auto va = new (_file._alloc) ARMPLTVeneerAtom(_file, name);
    // Fake reference to show connection between veneer and PLT entry.
    va->addReferenceELF_ARM(R_ARM_NONE, 0, pa, 0);

    _pltAtoms[da] = PLTWithVeneer(pa, va);
    return va;
  }

  typedef const GOTAtom *(Derived::*GOTFactory)(const DefinedAtom *);

  /// \brief get a PLT entry referencing PLTGOT entry.
  ///
  /// If the entry does not exist, both GOT and PLT entry are created.
  const PLTAtom *getPLTEntry(const DefinedAtom *da, bool fromThumb,
                             GOTFactory gotFactory, StringRef source) {
    auto pltVeneer = _pltAtoms.lookup(da);
    if (!pltVeneer.empty()) {
      // Return clean PLT entry provided it is ARM code.
      if (!fromThumb)
        return pltVeneer._plt;

      // Check if veneer is present for Thumb to ARM transition.
      if (pltVeneer._veneer)
        return pltVeneer._veneer;

      // Create veneer for existing PLT entry.
      return getPLTVeneer(da, pltVeneer._plt, source);
    }

    // Create specific GOT entry.
    const auto *ga = (static_cast<Derived *>(this)->*gotFactory)(da);
    assert(_gotAtoms.lookup(da) == ga &&
           "GOT entry should be added to the map");
    assert(ga->customSectionName() == ".got.plt" &&
           "GOT entry should be in a special section");

    std::string name = "__plt";
    name += source;
    name += da->name();
    // Create PLT entry for the GOT entry.
    auto pa = new (_file._alloc) ARMPLTAtom(_file, name);
    pa->addReferenceELF_ARM(R_ARM_ALU_PC_G0_NC, 0, ga, -8);
    pa->addReferenceELF_ARM(R_ARM_ALU_PC_G1_NC, 4, ga, -4);
    pa->addReferenceELF_ARM(R_ARM_LDR_PC_G2, 8, ga, 0);

    // Since all PLT entries are in ARM code, Thumb to ARM
    // switching should be added if the relocated place contais Thumb code.
    if (fromThumb)
      return getPLTVeneer(da, pa, source);

    // Otherwise just add PLT entry and return it to the caller.
    _pltAtoms[da] = PLTWithVeneer(pa);
    return pa;
  }

  /// \brief Create the GOT entry for a given IFUNC Atom.
  const GOTAtom *createIFUNCGOTEntry(const DefinedAtom *da) {
    assert(!_gotAtoms.lookup(da) && "IFUNC GOT entry already exists");
    auto g = new (_file._alloc) ARMGOTPLTAtom(_file);
    g->addReferenceELF_ARM(R_ARM_ABS32, 0, da, 0);
    g->addReferenceELF_ARM(R_ARM_IRELATIVE, 0, da, 0);
#ifndef NDEBUG
    g->_name = "__got_ifunc_";
    g->_name += da->name();
#endif
    _gotAtoms[da] = g;
    return g;
  }

  /// \brief get the PLT entry for a given IFUNC Atom.
  const PLTAtom *getIFUNCPLTEntry(const DefinedAtom *da, bool fromThumb) {
    return getPLTEntry(da, fromThumb, &Derived::createIFUNCGOTEntry, "_ifunc_");
  }

  /// \brief Redirect the call to the PLT stub for the target IFUNC.
  ///
  /// This create a PLT and GOT entry for the IFUNC if one does not exist. The
  /// GOT entry and a IRELATIVE relocation to the original target resolver.
  std::error_code handleIFUNC(const DefinedAtom &atom, const Reference &ref) {
    auto target = dyn_cast<const DefinedAtom>(ref.target());
    if (target && target->contentType() == DefinedAtom::typeResolver) {
      const_cast<Reference &>(ref).setTarget(
          getIFUNCPLTEntry(target, isThumbCode(atom.codeModel())));
    }
    return std::error_code();
  }

  /// \brief Create a GOT entry containing 0.
  const GOTAtom *getNullGOT() {
    if (!_null) {
      _null = new (_file._alloc) ARMGOTPLTAtom(_file);
#ifndef NDEBUG
      _null->_name = "__got_null";
#endif
    }
    return _null;
  }

  const GOTAtom *getGOT(const DefinedAtom *da) {
    if (auto got = _gotAtoms.lookup(da))
      return got;
    auto g = new (_file._alloc) ARMGOTAtom(_file);
    g->addReferenceELF_ARM(R_ARM_ABS32, 0, da, 0);
#ifndef NDEBUG
    g->_name = "__got_";
    g->_name += da->name();
#endif
    _gotAtoms[da] = g;
    return g;
  }

public:
  ARMRelocationPass(const ELFLinkingContext &ctx) : _file(ctx), _ctx(ctx) {}

  /// \brief Do the pass.
  ///
  /// The goal here is to first process each reference individually. Each call
  /// to handleReference may modify the reference itself and/or create new
  /// atoms which must be stored in one of the maps below.
  ///
  /// After all references are handled, the atoms created during that are all
  /// added to mf.
  void perform(std::unique_ptr<SimpleFile> &mf) override {
    ScopedTask task(getDefaultDomain(), "ARM GOT/PLT Pass");
    DEBUG_WITH_TYPE(
        "ARM", llvm::dbgs() << "Undefined Atoms" << "\n";
        for (const auto &atom
             : mf->undefined()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        }

        llvm::dbgs() << "Shared Library Atoms" << "\n";
        for (const auto &atom
             : mf->sharedLibrary()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        }

        llvm::dbgs() << "Absolute Atoms" << "\n";
        for (const auto &atom
             : mf->absolute()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        }

        llvm::dbgs() << "Defined Atoms" << "\n";
        for (const auto &atom
             : mf->defined()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        });

    // Process all references.
    for (const auto &atom : mf->defined()) {
      for (const auto &ref : *atom) {
        handleReference(*atom, *ref);
      }
    }

    // Add all created atoms to the link.
    uint64_t ordinal = 0;
    for (auto &pltKV : _pltAtoms) {
      auto &plt = pltKV.second;
      if (auto *v = plt._veneer) {
        v->setOrdinal(ordinal++);
        mf->addAtom(*v);
      }
      auto *p = plt._plt;
      p->setOrdinal(ordinal++);
      mf->addAtom(*p);
    }
    if (_null) {
      _null->setOrdinal(ordinal++);
      mf->addAtom(*_null);
    }
    for (auto &gotKV : _gotAtoms) {
      auto &got = gotKV.second;
      got->setOrdinal(ordinal++);
      mf->addAtom(*got);
    }
    for (auto &veneerKV : _veneerAtoms) {
      auto &veneer = veneerKV.second;
      veneer->setOrdinal(ordinal++);
      mf->addAtom(*veneer);
    }
  }

protected:
  /// \brief Owner of all the Atoms created by this pass.
  ELFPassFile _file;
  const ELFLinkingContext &_ctx;

  /// \brief Map Atoms to their GOT entries.
  llvm::MapVector<const Atom *, GOTAtom *> _gotAtoms;

  /// \brief Map Atoms to their PLT entries depending on the code model.
  struct PLTWithVeneer {
    PLTWithVeneer(PLTAtom *p = nullptr, PLTAtom *v = nullptr)
        : _plt(p), _veneer(v) {}

    bool empty() const {
      assert((_plt || !_veneer) && "Veneer appears without PLT entry");
      return !_plt && !_veneer;
    }

    PLTAtom *_plt;
    PLTAtom *_veneer;
  };
  llvm::MapVector<const Atom *, PLTWithVeneer> _pltAtoms;

  /// \brief Map Atoms to their veneers.
  llvm::MapVector<const Atom *, VeneerAtom *> _veneerAtoms;

  /// \brief GOT entry that is always 0. Used for undefined weaks.
  GOTAtom *_null = nullptr;
};

/// This implements the static relocation model. Meaning GOT and PLT entries are
/// not created for references that can be directly resolved. These are
/// converted to a direct relocation. For entries that do require a GOT or PLT
/// entry, that entry is statically bound.
///
/// TLS always assumes module 1 and attempts to remove indirection.
class ARMStaticRelocationPass final
    : public ARMRelocationPass<ARMStaticRelocationPass> {
public:
  ARMStaticRelocationPass(const elf::ARMLinkingContext &ctx)
      : ARMRelocationPass(ctx) {}

  /// \brief Get the veneer for ARM B/BL instructions.
  const VeneerAtom *getVeneer_ARM_B_BL(const DefinedAtom *da,
                                       StringRef secName) {
    if (auto veneer = _veneerAtoms.lookup(da))
      return veneer;
    auto v = new (_file._alloc) Veneer_ARM_B_BL_StaticAtom(_file, secName);
    v->addReferenceELF_ARM(R_ARM_ABS32, 4, da, 0);

    v->_name = "__";
    v->_name += da->name();
    v->_name += "_from_arm";

    _veneerAtoms[da] = v;
    return v;
  }

  /// \brief Get the veneer for Thumb B/BL instructions.
  const VeneerAtom *getVeneer_THM_B_BL(const DefinedAtom *da,
                                       StringRef secName) {
    if (auto veneer = _veneerAtoms.lookup(da))
      return veneer;
    auto v = new (_file._alloc) Veneer_THM_B_BL_StaticAtom(_file, secName);
    v->addReferenceELF_ARM(R_ARM_JUMP24, 4, da, 0);

    v->_name = "__";
    v->_name += da->name();
    v->_name += "_from_thumb";

    _veneerAtoms[da] = v;
    return v;
  }

  /// \brief Create a GOT entry for R_ARM_TLS_TPOFF32 reloc.
  const GOTAtom *getTLSTPOFF32(const DefinedAtom *da) {
    return getGOTTLSEntry<R_ARM_TLS_LE32>(da);
  }

  std::error_code handleGOT(const Reference &ref) {
    if (isa<UndefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getNullGOT());
    else if (const auto *da = dyn_cast<DefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getGOT(da));
    return std::error_code();
  }
};

} // end of anon namespace

std::unique_ptr<Pass>
lld::elf::createARMRelocationPass(const ARMLinkingContext &ctx) {
  switch (ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    if (ctx.isDynamic())
      llvm_unreachable("Unhandled output file type");
    return llvm::make_unique<ARMStaticRelocationPass>(ctx);
  default:
    llvm_unreachable("Unhandled output file type");
  }
}
