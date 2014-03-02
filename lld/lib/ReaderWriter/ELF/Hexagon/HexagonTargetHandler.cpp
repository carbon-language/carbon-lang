//===- lib/ReaderWriter/ELF/Hexagon/HexagonTargetHandler.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonExecutableWriter.h"
#include "HexagonDynamicLibraryWriter.h"
#include "HexagonTargetHandler.h"
#include "HexagonLinkingContext.h"

using namespace lld;
using namespace elf;
using namespace llvm::ELF;

using llvm::makeArrayRef;

HexagonTargetHandler::HexagonTargetHandler(HexagonLinkingContext &context)
    : DefaultTargetHandler(context), _hexagonLinkingContext(context),
      _hexagonRuntimeFile(new HexagonRuntimeFile<HexagonELFType>(context)),
      _hexagonTargetLayout(new HexagonTargetLayout<HexagonELFType>(context)),
      _hexagonRelocationHandler(new HexagonTargetRelocationHandler(
          context, *_hexagonTargetLayout.get())) {}

std::unique_ptr<Writer> HexagonTargetHandler::getWriter() {
  switch (_hexagonLinkingContext.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return std::unique_ptr<Writer>(
        new elf::HexagonExecutableWriter<HexagonELFType>(
            _hexagonLinkingContext, *_hexagonTargetLayout.get()));
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Writer>(
        new elf::HexagonDynamicLibraryWriter<HexagonELFType>(
            _hexagonLinkingContext, *_hexagonTargetLayout.get()));
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

using namespace llvm::ELF;

// .got atom
const uint8_t hexagonGotAtomContent[4] = { 0 };
// .got.plt atom (entry 0)
const uint8_t hexagonGotPlt0AtomContent[16] = { 0 };
// .got.plt atom (all other entries)
const uint8_t hexagonGotPltAtomContent[4] = { 0 };
// .plt (entry 0)
const uint8_t hexagonPlt0AtomContent[28] = {
  0x00, 0x40, 0x00, 0x00, // { immext (#0)
  0x1c, 0xc0, 0x49, 0x6a, //   r28 = add (pc, ##GOT0@PCREL) } # address of GOT0
  0x0e, 0x42, 0x9c, 0xe2, // { r14 -= add (r28, #16)  # offset of GOTn from GOTa
  0x4f, 0x40, 0x9c, 0x91, //   r15 = memw (r28 + #8)  # object ID at GOT2
  0x3c, 0xc0, 0x9c, 0x91, //   r28 = memw (r28 + #4) }# dynamic link at GOT1
  0x0e, 0x42, 0x0e, 0x8c, // { r14 = asr (r14, #2)    # index of PLTn
  0x00, 0xc0, 0x9c, 0x52, //   jumpr r28 }            # call dynamic linker
};

// .plt (other entries)
const uint8_t hexagonPltAtomContent[16] = {
  0x00, 0x40, 0x00, 0x00, // { immext (#0)
  0x0e, 0xc0, 0x49, 0x6a, //   r14 = add (pc, ##GOTn@PCREL) } # address of GOTn
  0x1c, 0xc0, 0x8e, 0x91, // r28 = memw (r14)                 # contents of GOTn
  0x00, 0xc0, 0x9c, 0x52, // jumpr r28                        # call it
};

class HexagonGOTAtom : public GOTAtom {
public:
  HexagonGOTAtom(const File &f) : GOTAtom(f, ".got") {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return makeArrayRef(hexagonGotAtomContent);
  }

  virtual Alignment alignment() const { return Alignment(2); }
};

class HexagonGOTPLTAtom : public GOTAtom {
public:
  HexagonGOTPLTAtom(const File &f) : GOTAtom(f, ".got.plt") {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return makeArrayRef(hexagonGotPltAtomContent);
  }

  virtual Alignment alignment() const { return Alignment(2); }
};

class HexagonGOTPLT0Atom : public GOTAtom {
public:
  HexagonGOTPLT0Atom(const File &f) : GOTAtom(f, ".got.plt") {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return makeArrayRef(hexagonGotPlt0AtomContent);
  }

  virtual Alignment alignment() const { return Alignment(3); }
};

class HexagonPLT0Atom : public PLT0Atom {
public:
  HexagonPLT0Atom(const File &f) : PLT0Atom(f) {
#ifndef NDEBUG
    _name = ".PLT0";
#endif
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    return makeArrayRef(hexagonPlt0AtomContent);
  }
};

class HexagonPLTAtom : public PLTAtom {

public:
  HexagonPLTAtom(const File &f, StringRef secName) : PLTAtom(f, secName) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return makeArrayRef(hexagonPltAtomContent);
  }
};

class ELFPassFile : public SimpleFile {
public:
  ELFPassFile(const ELFLinkingContext &eti) : SimpleFile("ELFPassFile") {
    setOrdinal(eti.getNextOrdinalAndIncrement());
  }

  llvm::BumpPtrAllocator _alloc;
};

/// \brief Create GOT and PLT entries for relocations. Handles standard GOT/PLT
template <class Derived> class GOTPLTPass : public Pass {
  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    if (ref.kindNamespace() != Reference::KindNamespace::ELF)
      return;
    assert(ref.kindArch() == Reference::KindArch::Hexagon);
    switch (ref.kindValue()) {
    case R_HEX_PLT_B22_PCREL:
    case R_HEX_B22_PCREL:
      static_cast<Derived *>(this)->handlePLT32(ref);
      break;
    case R_HEX_GOT_LO16:
    case R_HEX_GOT_HI16:
    case R_HEX_GOT_32_6_X:
    case R_HEX_GOT_16_X:
    case R_HEX_GOT_11_X:
      static_cast<Derived *>(this)->handleGOTREL(ref);
      break;
    }
  }

protected:
  /// \brief Create a GOT entry containing 0.
  const GOTAtom *getNullGOT() {
    if (!_null) {
      _null = new (_file._alloc) HexagonGOTPLTAtom(_file);
#ifndef NDEBUG
      _null->_name = "__got_null";
#endif
    }
    return _null;
  }

public:
  GOTPLTPass(const ELFLinkingContext &ctx)
      : _file(ctx), _null(nullptr), _PLT0(nullptr), _got0(nullptr) {}

  /// \brief Do the pass.
  ///
  /// The goal here is to first process each reference individually. Each call
  /// to handleReference may modify the reference itself and/or create new
  /// atoms which must be stored in one of the maps below.
  ///
  /// After all references are handled, the atoms created during that are all
  /// added to mf.
  virtual void perform(std::unique_ptr<MutableFile> &mf) {
    // Process all references.
    for (const auto &atom : mf->defined())
      for (const auto &ref : *atom)
        handleReference(*atom, *ref);

    // Add all created atoms to the link.
    uint64_t ordinal = 0;
    if (_PLT0) {
      _PLT0->setOrdinal(ordinal++);
      mf->addAtom(*_PLT0);
    }
    for (auto &plt : _pltVector) {
      plt->setOrdinal(ordinal++);
      mf->addAtom(*plt);
    }
    if (_null) {
      _null->setOrdinal(ordinal++);
      mf->addAtom(*_null);
    }
    if (_got0) {
      _got0->setOrdinal(ordinal++);
      mf->addAtom(*_got0);
    }
    for (auto &got : _gotVector) {
      got->setOrdinal(ordinal++);
      mf->addAtom(*got);
    }
  }

protected:
  /// \brief Owner of all the Atoms created by this pass.
  ELFPassFile _file;

  /// \brief Map Atoms to their GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotMap;

  /// \brief Map Atoms to their PLT entries.
  llvm::DenseMap<const Atom *, PLTAtom *> _pltMap;

  /// \brief the list of GOT/PLT atoms
  std::vector<GOTAtom *> _gotVector;
  std::vector<PLTAtom *> _pltVector;

  /// \brief GOT entry that is always 0. Used for undefined weaks.
  GOTAtom *_null;

  /// \brief The got and plt entries for .PLT0. This is used to call into the
  /// dynamic linker for symbol resolution.
  /// @{
  PLT0Atom *_PLT0;
  GOTAtom *_got0;
  /// @}
};

class DynamicGOTPLTPass final : public GOTPLTPass<DynamicGOTPLTPass> {
public:
  DynamicGOTPLTPass(const elf::HexagonLinkingContext &ctx) : GOTPLTPass(ctx) {
    _got0 = new (_file._alloc) HexagonGOTPLT0Atom(_file);
#ifndef NDEBUG
    _got0->_name = "__got0";
#endif
  }

  const PLT0Atom *getPLT0() {
    if (_PLT0)
      return _PLT0;
    _PLT0 = new (_file._alloc) HexagonPLT0Atom(_file);
    _PLT0->addReferenceELF_Hexagon(R_HEX_B32_PCREL_X, 0, _got0, 0);
    _PLT0->addReferenceELF_Hexagon(R_HEX_6_PCREL_X, 4, _got0, 4);
    DEBUG_WITH_TYPE("PLT", llvm::dbgs() << "[ PLT0/GOT0 ] "
                                        << "Adding plt0/got0 \n");
    return _PLT0;
  }

  const PLTAtom *getPLTEntry(const Atom *a) {
    auto plt = _pltMap.find(a);
    if (plt != _pltMap.end())
      return plt->second;
    auto ga = new (_file._alloc) HexagonGOTPLTAtom(_file);
    ga->addReferenceELF_Hexagon(R_HEX_JMP_SLOT, 0, a, 0);
    auto pa = new (_file._alloc) HexagonPLTAtom(_file, ".plt");
    pa->addReferenceELF_Hexagon(R_HEX_B32_PCREL_X, 0, ga, 0);
    pa->addReferenceELF_Hexagon(R_HEX_6_PCREL_X, 4, ga, 4);

    // Point the got entry to the PLT0 atom initially
    ga->addReferenceELF_Hexagon(R_HEX_32, 0, getPLT0(), 0);
#ifndef NDEBUG
    ga->_name = "__got_";
    ga->_name += a->name();
    pa->_name = "__plt_";
    pa->_name += a->name();
    DEBUG_WITH_TYPE("PLT", llvm::dbgs() << "[" << a->name() << "] "
                                        << "Adding plt/got: " << pa->_name
                                        << "/" << ga->_name << "\n");
#endif
    _gotMap[a] = ga;
    _pltMap[a] = pa;
    _gotVector.push_back(ga);
    _pltVector.push_back(pa);
    return pa;
  }

  const GOTAtom *getGOTEntry(const Atom *a) {
    auto got = _gotMap.find(a);
    if (got != _gotMap.end())
      return got->second;
    auto ga = new (_file._alloc) HexagonGOTAtom(_file);
    ga->addReferenceELF_Hexagon(R_HEX_GLOB_DAT, 0, a, 0);

#ifndef NDEBUG
    ga->_name = "__got_";
    ga->_name += a->name();
    DEBUG_WITH_TYPE("GOT", llvm::dbgs() << "[" << a->name() << "] "
                                        << "Adding got: " << ga->_name << "\n");
#endif
    _gotMap[a] = ga;
    _gotVector.push_back(ga);
    return ga;
  }

  error_code handleGOTREL(const Reference &ref) {
    // Turn this so that the target is set to the GOT entry
    const_cast<Reference &>(ref).setTarget(getGOTEntry(ref.target()));
    return error_code::success();
  }

  error_code handlePLT32(const Reference &ref) {
    // Turn this into a PC32 to the PLT entry.
    assert(ref.kindNamespace() == Reference::KindNamespace::ELF);
    assert(ref.kindArch() == Reference::KindArch::Hexagon);
    const_cast<Reference &>(ref).setKindValue(R_HEX_B22_PCREL);
    const_cast<Reference &>(ref).setTarget(getPLTEntry(ref.target()));
    return error_code::success();
  }
};

void elf::HexagonLinkingContext::addPasses(PassManager &pm) {
  if (isDynamic())
    pm.add(std::unique_ptr<Pass>(new DynamicGOTPLTPass(*this)));
  ELFLinkingContext::addPasses(pm);
}

void HexagonTargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::Hexagon, kindStrings);
}

const Registry::KindStrings HexagonTargetHandler::kindStrings[] = {
  LLD_KIND_STRING_ENTRY(R_HEX_NONE),
  LLD_KIND_STRING_ENTRY(R_HEX_B22_PCREL),
  LLD_KIND_STRING_ENTRY(R_HEX_B15_PCREL),
  LLD_KIND_STRING_ENTRY(R_HEX_B7_PCREL),
  LLD_KIND_STRING_ENTRY(R_HEX_LO16),
  LLD_KIND_STRING_ENTRY(R_HEX_HI16),
  LLD_KIND_STRING_ENTRY(R_HEX_32),
  LLD_KIND_STRING_ENTRY(R_HEX_16),
  LLD_KIND_STRING_ENTRY(R_HEX_8),
  LLD_KIND_STRING_ENTRY(R_HEX_GPREL16_0),
  LLD_KIND_STRING_ENTRY(R_HEX_GPREL16_1),
  LLD_KIND_STRING_ENTRY(R_HEX_GPREL16_2),
  LLD_KIND_STRING_ENTRY(R_HEX_GPREL16_3),
  LLD_KIND_STRING_ENTRY(R_HEX_HL16),
  LLD_KIND_STRING_ENTRY(R_HEX_B13_PCREL),
  LLD_KIND_STRING_ENTRY(R_HEX_B9_PCREL),
  LLD_KIND_STRING_ENTRY(R_HEX_B32_PCREL_X),
  LLD_KIND_STRING_ENTRY(R_HEX_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_B22_PCREL_X),
  LLD_KIND_STRING_ENTRY(R_HEX_B15_PCREL_X),
  LLD_KIND_STRING_ENTRY(R_HEX_B13_PCREL_X),
  LLD_KIND_STRING_ENTRY(R_HEX_B9_PCREL_X),
  LLD_KIND_STRING_ENTRY(R_HEX_B7_PCREL_X),
  LLD_KIND_STRING_ENTRY(R_HEX_16_X),
  LLD_KIND_STRING_ENTRY(R_HEX_12_X),
  LLD_KIND_STRING_ENTRY(R_HEX_11_X),
  LLD_KIND_STRING_ENTRY(R_HEX_10_X),
  LLD_KIND_STRING_ENTRY(R_HEX_9_X),
  LLD_KIND_STRING_ENTRY(R_HEX_8_X),
  LLD_KIND_STRING_ENTRY(R_HEX_7_X),
  LLD_KIND_STRING_ENTRY(R_HEX_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_32_PCREL),
  LLD_KIND_STRING_ENTRY(R_HEX_COPY),
  LLD_KIND_STRING_ENTRY(R_HEX_GLOB_DAT),
  LLD_KIND_STRING_ENTRY(R_HEX_JMP_SLOT),
  LLD_KIND_STRING_ENTRY(R_HEX_RELATIVE),
  LLD_KIND_STRING_ENTRY(R_HEX_PLT_B22_PCREL),
  LLD_KIND_STRING_ENTRY(R_HEX_GOTREL_LO16),
  LLD_KIND_STRING_ENTRY(R_HEX_GOTREL_HI16),
  LLD_KIND_STRING_ENTRY(R_HEX_GOTREL_32),
  LLD_KIND_STRING_ENTRY(R_HEX_GOT_LO16),
  LLD_KIND_STRING_ENTRY(R_HEX_GOT_HI16),
  LLD_KIND_STRING_ENTRY(R_HEX_GOT_32),
  LLD_KIND_STRING_ENTRY(R_HEX_GOT_16),
  LLD_KIND_STRING_ENTRY(R_HEX_DTPMOD_32),
  LLD_KIND_STRING_ENTRY(R_HEX_DTPREL_LO16),
  LLD_KIND_STRING_ENTRY(R_HEX_DTPREL_HI16),
  LLD_KIND_STRING_ENTRY(R_HEX_DTPREL_32),
  LLD_KIND_STRING_ENTRY(R_HEX_DTPREL_16),
  LLD_KIND_STRING_ENTRY(R_HEX_GD_PLT_B22_PCREL),
  LLD_KIND_STRING_ENTRY(R_HEX_GD_GOT_LO16),
  LLD_KIND_STRING_ENTRY(R_HEX_GD_GOT_HI16),
  LLD_KIND_STRING_ENTRY(R_HEX_GD_GOT_32),
  LLD_KIND_STRING_ENTRY(R_HEX_GD_GOT_16),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_LO16),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_HI16),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_32),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_GOT_LO16),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_GOT_HI16),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_GOT_32),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_GOT_16),
  LLD_KIND_STRING_ENTRY(R_HEX_TPREL_LO16),
  LLD_KIND_STRING_ENTRY(R_HEX_TPREL_HI16),
  LLD_KIND_STRING_ENTRY(R_HEX_TPREL_32),
  LLD_KIND_STRING_ENTRY(R_HEX_TPREL_16),
  LLD_KIND_STRING_ENTRY(R_HEX_6_PCREL_X),
  LLD_KIND_STRING_ENTRY(R_HEX_GOTREL_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_GOTREL_16_X),
  LLD_KIND_STRING_ENTRY(R_HEX_GOTREL_11_X),
  LLD_KIND_STRING_ENTRY(R_HEX_GOT_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_GOT_16_X),
  LLD_KIND_STRING_ENTRY(R_HEX_GOT_11_X),
  LLD_KIND_STRING_ENTRY(R_HEX_DTPREL_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_DTPREL_16_X),
  LLD_KIND_STRING_ENTRY(R_HEX_DTPREL_11_X),
  LLD_KIND_STRING_ENTRY(R_HEX_GD_GOT_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_GD_GOT_16_X),
  LLD_KIND_STRING_ENTRY(R_HEX_GD_GOT_11_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_16_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_GOT_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_GOT_16_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_GOT_11_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_16_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_GOT_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_GOT_16_X),
  LLD_KIND_STRING_ENTRY(R_HEX_IE_GOT_11_X),
  LLD_KIND_STRING_ENTRY(R_HEX_TPREL_32_6_X),
  LLD_KIND_STRING_ENTRY(R_HEX_TPREL_16_X),
  LLD_KIND_STRING_ENTRY(R_HEX_TPREL_11_X),
  LLD_KIND_STRING_END
};
