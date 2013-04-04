//===- lib/ReaderWriter/ELF/Hexagon/HexagonTargetHandler.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonDynamicAtoms.h"
#include "HexagonTargetHandler.h"
#include "HexagonTargetInfo.h"

using namespace lld;
using namespace elf;

using namespace llvm::ELF;

HexagonTargetHandler::HexagonTargetHandler(HexagonTargetInfo &targetInfo)
    : DefaultTargetHandler(targetInfo), _targetLayout(targetInfo),
      _relocationHandler(targetInfo, *this, _targetLayout),
      _hexagonRuntimeFile(targetInfo) {}

namespace {

using namespace llvm::ELF;

class ELFPassFile : public SimpleFile {
public:
  ELFPassFile(const ELFTargetInfo &eti) : SimpleFile(eti, "ELFPassFile") {}

  llvm::BumpPtrAllocator _alloc;
};

/// \brief Create GOT and PLT entries for relocations. Handles standard GOT/PLT
template <class Derived> class GOTPLTPass : public Pass {
  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    switch (ref.kind()) {
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
  GOTPLTPass(const ELFTargetInfo &ti)
      : _file(ti), _null(nullptr), _PLT0(nullptr), _got0(nullptr) {}

  /// \brief Do the pass.
  ///
  /// The goal here is to first process each reference individually. Each call
  /// to handleReference may modify the reference itself and/or create new
  /// atoms which must be stored in one of the maps below.
  ///
  /// After all references are handled, the atoms created during that are all
  /// added to mf.
  virtual void perform(MutableFile &mf) {
    // Process all references.
    for (const auto &atom : mf.defined())
      for (const auto &ref : *atom)
        handleReference(*atom, *ref);

    // Add all created atoms to the link.
    uint64_t ordinal = 0;
    if (_PLT0) {
      _PLT0->setOrdinal(ordinal++);
      mf.addAtom(*_PLT0);
    }
    for (auto &plt : _pltVector) {
      plt->setOrdinal(ordinal++);
      mf.addAtom(*plt);
    }
    if (_null) {
      _null->setOrdinal(ordinal++);
      mf.addAtom(*_null);
    }
    if (_got0) {
      _got0->setOrdinal(ordinal++);
      mf.addAtom(*_got0);
    }
    for (auto &got : _gotVector) {
      got->setOrdinal(ordinal++);
      mf.addAtom(*got);
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

class DynamicGOTPLTPass LLVM_FINAL : public GOTPLTPass<DynamicGOTPLTPass> {
public:
  DynamicGOTPLTPass(const elf::HexagonTargetInfo &ti) : GOTPLTPass(ti) {
    _got0 = new (_file._alloc) HexagonGOTPLT0Atom(_file);
#ifndef NDEBUG
    _got0->_name = "__got0";
#endif
  }

  const PLT0Atom *getPLT0() {
    if (_PLT0)
      return _PLT0;
    _PLT0 = new (_file._alloc) HexagonPLT0Atom(_file);
    _PLT0->addReference(R_HEX_B32_PCREL_X, 0, _got0, 0);
    _PLT0->addReference(R_HEX_6_PCREL_X, 4, _got0, 4);
    DEBUG_WITH_TYPE("PLT", llvm::dbgs() << "[ PLT0/GOT0 ] "
                                        << "Adding plt0/got0 \n");
    return _PLT0;
  }

  const PLTAtom *getPLTEntry(const Atom *a) {
    auto plt = _pltMap.find(a);
    if (plt != _pltMap.end())
      return plt->second;
    auto ga = new (_file._alloc) HexagonGOTPLTAtom(_file);
    ga->addReference(R_HEX_JMP_SLOT, 0, a, 0);
    auto pa = new (_file._alloc) HexagonPLTAtom(_file, ".plt");
    pa->addReference(R_HEX_B32_PCREL_X, 0, ga, 0);
    pa->addReference(R_HEX_6_PCREL_X, 4, ga, 4);

    // Point the got entry to the PLT0 atom initially
    ga->addReference(R_HEX_32, 0, getPLT0(), 0);
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
    ga->addReference(R_HEX_GLOB_DAT, 0, a, 0);

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

  ErrorOr<void> handleGOTREL(const Reference &ref) {
    // Turn this so that the target is set to the GOT entry
    const_cast<Reference &>(ref).setTarget(getGOTEntry(ref.target()));
    return error_code::success();
  }

  ErrorOr<void> handlePLT32(const Reference &ref) {
    // Turn this into a PC32 to the PLT entry.
    const_cast<Reference &>(ref).setKind(R_HEX_B22_PCREL);
    const_cast<Reference &>(ref).setTarget(getPLTEntry(ref.target()));
    return error_code::success();
  }
};
} // end anonymous namespace

void elf::HexagonTargetInfo::addPasses(PassManager &pm) const {
  if (isDynamic())
    pm.add(std::unique_ptr<Pass>(new DynamicGOTPLTPass(*this)));
  ELFTargetInfo::addPasses(pm);
}
