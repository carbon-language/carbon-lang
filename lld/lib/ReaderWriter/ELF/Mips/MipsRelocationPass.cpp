//===- lib/ReaderWriter/ELF/Mips/MipsRelocationPass.cpp -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsLinkingContext.h"
#include "MipsRelocationPass.h"

#include "Atoms.h"

namespace {

using namespace lld;
using namespace lld::elf;
using namespace llvm::ELF;

// Lazy resolver
const uint8_t mipsGot0AtomContent[] = { 0x00, 0x00, 0x00, 0x00 };

// Module pointer
const uint8_t mipsGotModulePointerAtomContent[] = { 0x00, 0x00, 0x00, 0x80 };

// PLT0 entry
const uint8_t mipsPlt0AtomContent[] = {
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
const uint8_t mipsPltAAtomContent[] = {
  0x00, 0x00, 0x0f, 0x3c, // lui   $15, %hi(.got.plt entry)
  0x00, 0x00, 0xf9, 0x8d, // l[wd] $25, %lo(.got.plt entry)($15)
  0x08, 0x00, 0x20, 0x03, // jr    $25
  0x00, 0x00, 0xf8, 0x25  // addiu $24, $15, %lo(.got.plt entry)
};

/// \brief Abstract base class represent MIPS GOT entries.
class MipsGOTAtom : public GOTAtom {
public:
  MipsGOTAtom(const File &f) : GOTAtom(f, ".got") {}

  virtual Alignment alignment() const { return Alignment(2); }
};

/// \brief MIPS GOT entry initialized by zero.
class GOT0Atom : public MipsGOTAtom {
public:
  GOT0Atom(const File &f) : MipsGOTAtom(f) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return llvm::makeArrayRef(mipsGot0AtomContent);
  }
};

/// \brief MIPS GOT entry initialized by zero.
class GOTModulePointerAtom : public MipsGOTAtom {
public:
  GOTModulePointerAtom(const File &f) : MipsGOTAtom(f) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return llvm::makeArrayRef(mipsGotModulePointerAtomContent);
  }
};

class PLT0Atom : public PLTAtom {
public:
  PLT0Atom(const File &f) : PLTAtom(f, ".plt") {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return llvm::makeArrayRef(mipsPlt0AtomContent);
  }
};

class PLTAAtom : public PLTAtom {
public:
  PLTAAtom(const File &f) : PLTAtom(f, ".plt") {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return llvm::makeArrayRef(mipsPltAAtomContent);
  }
};

/// \brief MIPS GOT PLT entry
class GOTPLTAtom : public GOTAtom {
public:
  GOTPLTAtom(const File &f) : GOTAtom(f, ".got.plt") {}

  virtual Alignment alignment() const { return Alignment(2); }

  virtual ArrayRef<uint8_t> rawContent() const {
    return llvm::makeArrayRef(mipsGot0AtomContent);
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

class RelocationPass : public Pass {
public:
  RelocationPass(MipsLinkingContext &context)
      : _context(context), _file(context) {
    _localGotVector.push_back(new (_file._alloc) GOT0Atom(_file));
    _localGotVector.push_back(new (_file._alloc) GOTModulePointerAtom(_file));
  }

  virtual void perform(std::unique_ptr<MutableFile> &mf) {
    // Process all references.
    for (const auto &atom : mf->defined())
      for (const auto &ref : *atom)
        handleReference(*atom, *ref);

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

    for (auto &plt : _pltVector) {
      DEBUG_WITH_TYPE("MipsGOT", llvm::dbgs() << "[ PLT ] Adding "
                                              << plt->name() << "\n");
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
  }

private:
  /// \brief Reference to the linking context.
  const MipsLinkingContext &_context;

  /// \brief Owner of all the Atoms created by this pass.
  RelocationPassFile _file;

  /// \brief Map Atoms to their GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotMap;

  /// \brief the list of local GOT atoms.
  std::vector<GOTAtom *> _localGotVector;

  /// \brief the list of global GOT atoms.
  std::vector<GOTAtom *> _globalGotVector;

  /// \brief Map Atoms to their PLT entries.
  llvm::DenseMap<const Atom *, PLTAtom *> _pltMap;

  /// \brief Map Atoms to their Object entries.
  llvm::DenseMap<const Atom *, ObjectAtom *> _objectMap;

  /// \brief the list of PLT atoms.
  std::vector<PLTAtom *> _pltVector;

  /// \brief the list of GOTPLT atoms.
  std::vector<GOTAtom *> _gotpltVector;

  /// \brief the list of Object entries.
  std::vector<ObjectAtom *> _objectVector;

  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    if (ref.kindNamespace() != lld::Reference::KindNamespace::ELF)
      return;
    assert(ref.kindArch() == Reference::KindArch::Mips);
    switch (ref.kindValue()) {
    case R_MIPS_32:
    case R_MIPS_HI16:
    case R_MIPS_LO16:
      // FIXME (simon): Handle dynamic/static linking differently.
      handlePlain(ref);
      break;
    case R_MIPS_26:
      handlePLT(ref);
      break;
    case R_MIPS_GOT16:
    case R_MIPS_CALL16:
      handleGOT(ref);
      break;
    }
  }

  bool isLocal(const Atom *a) {
    if (auto *da = dyn_cast<DefinedAtom>(a))
      return da->scope() == Atom::scopeTranslationUnit;
    return false;
  }

  void handlePlain(const Reference &ref) {
    if (!ref.target())
      return;
    auto sla = dyn_cast<SharedLibraryAtom>(ref.target());
    if (sla && sla->type() == SharedLibraryAtom::Type::Data)
      const_cast<Reference &>(ref).setTarget(getObjectEntry(sla));
  }

  void handlePLT(const Reference &ref) {
    if (ref.kindValue() == R_MIPS_26 && !isLocal(ref.target()))
      const_cast<Reference &>(ref).setKindValue(LLD_R_MIPS_GLOBAL_26);

    if (isa<SharedLibraryAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getPLTEntry(ref.target()));
  }

  void handleGOT(const Reference &ref) {
    if (ref.kindValue() == R_MIPS_GOT16 && !isLocal(ref.target()))
      const_cast<Reference &>(ref).setKindValue(LLD_R_MIPS_GLOBAL_GOT16);

    const_cast<Reference &>(ref).setTarget(getGOTEntry(ref.target()));
  }

  bool requireLocalGOT(const Atom *a) {
    Atom::Scope scope;
    if (auto *da = dyn_cast<DefinedAtom>(a))
      scope = da->scope();
    else if (auto *aa = dyn_cast<AbsoluteAtom>(a))
      scope = aa->scope();
    else
      return false;

    // Local and hidden symbols must be local.
    if (scope == Atom::scopeTranslationUnit ||
        scope == Atom::scopeLinkageUnit)
      return true;

    // External symbol defined in an executable file requires a local GOT entry.
    if (_context.getOutputELFType() == llvm::ELF::ET_EXEC)
      return true;

    return false;
  }

  const GOTAtom *getGOTEntry(const Atom *a) {
    auto got = _gotMap.find(a);
    if (got != _gotMap.end())
      return got->second;

    auto ga = new (_file._alloc) GOT0Atom(_file);
    _gotMap[a] = ga;

    bool localGOT = requireLocalGOT(a);

    if (localGOT)
      _localGotVector.push_back(ga);
    else {
      _globalGotVector.push_back(ga);
      ga->addReferenceELF_Mips(LLD_R_MIPS_GLOBAL_GOT, 0, a, 0);
    }

    if (const DefinedAtom *da = dyn_cast<DefinedAtom>(a))
      ga->addReferenceELF_Mips(R_MIPS_32, 0, da, 0);

    DEBUG_WITH_TYPE("MipsGOT", {
      ga->_name = "__got_";
      ga->_name += a->name();
      llvm::dbgs() << "[ GOT ] Create " << (localGOT ? "L " : "G ") << a->name()
                   << "\n";
    });

    return ga;
  }

  void createPLTHeader() {
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

  const PLTAtom *getPLTEntry(const Atom *a) {
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

  const ObjectAtom *getObjectEntry(const SharedLibraryAtom *a) {
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
};

} // end anon namespace

std::unique_ptr<Pass>
lld::elf::createMipsRelocationPass(MipsLinkingContext &ctx) {
  switch (ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Pass>(new RelocationPass(ctx));
  case llvm::ELF::ET_REL:
    return std::unique_ptr<Pass>();
  default:
    llvm_unreachable("Unhandled output file type");
  }
}
