//===- lib/ReaderWriter/ELF/Mips/MipsLinkingContext.cpp -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "MipsLinkingContext.h"
#include "MipsTargetHandler.h"

#include "llvm/ADT/StringSwitch.h"

using namespace lld;
using namespace lld::elf;

namespace {

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
class MipsGOT0Atom : public MipsGOTAtom {
public:
  MipsGOT0Atom(const File &f) : MipsGOTAtom(f) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return llvm::makeArrayRef(mipsGot0AtomContent);
  }
};

/// \brief MIPS GOT entry initialized by zero.
class MipsGOTModulePointerAtom : public MipsGOTAtom {
public:
  MipsGOTModulePointerAtom(const File &f) : MipsGOTAtom(f) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return llvm::makeArrayRef(mipsGotModulePointerAtomContent);
  }
};

class MipsGOTPassFile : public SimpleFile {
public:
  MipsGOTPassFile(const ELFLinkingContext &ctx)
      : SimpleFile("MipsGOTPassFile") {
    setOrdinal(ctx.getNextOrdinalAndIncrement());
  }

  llvm::BumpPtrAllocator _alloc;
};

class MipsGOTPass : public Pass {
public:
  MipsGOTPass(MipsLinkingContext &context)
      : _file(context), _got0(new (_file._alloc) MipsGOT0Atom(_file)),
        _got1(new (_file._alloc) MipsGOTModulePointerAtom(_file)) {
    _localGotVector.push_back(_got0);
    _localGotVector.push_back(_got1);
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
  MipsGOTPassFile _file;

  /// \brief GOT header entries.
  GOTAtom *_got0;
  GOTAtom *_got1;

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

  void handleGOT(const Reference &ref) {
    const_cast<Reference &>(ref).setTarget(getEntry(ref.target()));
  }

  const GOTAtom *getEntry(const Atom *a) {
    auto got = _gotMap.find(a);
    if (got != _gotMap.end())
      return got->second;

    const DefinedAtom *da = dyn_cast<DefinedAtom>(a);
    bool isLocal = (da && da->scope() == Atom::scopeTranslationUnit);

    auto ga = new (_file._alloc) MipsGOT0Atom(_file);
    _gotMap[a] = ga;
    if (isLocal)
      _localGotVector.push_back(ga);
    else {
      if (da)
        ga->addReferenceELF_Mips(R_MIPS_32, 0, a, 0);
      else
        ga->addReferenceELF_Mips(R_MIPS_NONE, 0, a, 0);
      _globalGotVector.push_back(ga);
    }

    DEBUG_WITH_TYPE("MipsGOT", {
      ga->_name = "__got_";
      ga->_name += a->name();
      llvm::dbgs() << "[ GOT ] Create " << (isLocal ? "L " : "G ") << a->name()
                   << "\n";
    });

    return ga;
  }
};

} // end anon namespace

MipsLinkingContext::MipsLinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, std::unique_ptr<TargetHandlerBase>(
                                    new MipsTargetHandler(*this))) {}

MipsTargetLayout<Mips32ElELFType> &MipsLinkingContext::getTargetLayout() {
  auto &layout = getTargetHandler<Mips32ElELFType>().targetLayout();
  return static_cast<MipsTargetLayout<Mips32ElELFType> &>(layout);
}

const MipsTargetLayout<Mips32ElELFType> &
MipsLinkingContext::getTargetLayout() const {
  auto &layout = getTargetHandler<Mips32ElELFType>().targetLayout();
  return static_cast<MipsTargetLayout<Mips32ElELFType> &>(layout);
}

bool MipsLinkingContext::isLittleEndian() const {
  return Mips32ElELFType::TargetEndianness == llvm::support::little;
}

void MipsLinkingContext::addPasses(PassManager &pm) {
  switch (getOutputELFType()) {
  case llvm::ELF::ET_DYN:
    pm.add(std::unique_ptr<Pass>(new MipsGOTPass(*this)));
    break;
  default:
    llvm_unreachable("Unhandled output file type");
  }

  ELFLinkingContext::addPasses(pm);
}
