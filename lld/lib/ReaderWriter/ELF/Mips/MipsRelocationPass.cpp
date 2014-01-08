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
  RelocationPass(MipsLinkingContext &context) : _file(context) {
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
  }

private:
  /// \brief Owner of all the Atoms created by this pass.
  RelocationPassFile _file;

  /// \brief Map Atoms to their GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotMap;

  /// \brief the list of local GOT atoms.
  std::vector<GOTAtom *> _localGotVector;

  /// \brief the list of global GOT atoms.
  std::vector<GOTAtom *> _globalGotVector;

  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    if (ref.kindNamespace() != lld::Reference::KindNamespace::ELF)
      return;
    assert(ref.kindArch() == Reference::KindArch::Mips);
    switch (ref.kindValue()) {
    case R_MIPS_GOT16:
    case R_MIPS_CALL16:
      handleGOT(ref);
      break;
    }
  }

  bool isLocal(const Atom *a) {
    return isa<DefinedAtom>(a) &&
           dyn_cast<DefinedAtom>(a)->scope() == Atom::scopeTranslationUnit;
  }

  void handleGOT(const Reference &ref) {
    if (ref.kindValue() == R_MIPS_GOT16 && !isLocal(ref.target()))
      const_cast<Reference &>(ref).setKindValue(LLD_R_MIPS_GLOBAL_GOT16);

    const_cast<Reference &>(ref).setTarget(getGOTEntry(ref.target()));
  }

  bool requireLocalGOT(const Atom *a) {
    Atom::Scope scope;
    if (isa<DefinedAtom>(a))
      scope = dyn_cast<DefinedAtom>(a)->scope();
    else if (isa<AbsoluteAtom>(a))
      scope = dyn_cast<AbsoluteAtom>(a)->scope();
    else
      return false;

    // Local and hidden symbols must be local.
    if (scope == Atom::scopeTranslationUnit ||
        scope == Atom::scopeLinkageUnit)
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
