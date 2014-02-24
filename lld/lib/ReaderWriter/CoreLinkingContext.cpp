//===- lib/ReaderWriter/CoreLinkingContext.cpp ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/CoreLinkingContext.h"

#include "lld/Core/Pass.h"
#include "lld/Core/PassManager.h"
#include "lld/Passes/LayoutPass.h"
#include "lld/Passes/RoundTripYAMLPass.h"
#include "lld/ReaderWriter/Simple.h"

#include "llvm/ADT/ArrayRef.h"

using namespace lld;

namespace {

/// \brief Simple atom created by the stubs pass.
class TestingStubAtom : public DefinedAtom {
public:
  TestingStubAtom(const File &F, const Atom &) : _file(F) {
    static uint32_t lastOrdinal = 0;
    _ordinal = lastOrdinal++;
  }

  virtual const File &file() const { return _file; }

  virtual StringRef name() const { return StringRef(); }

  virtual uint64_t ordinal() const { return _ordinal; }

  virtual uint64_t size() const { return 0; }

  virtual Scope scope() const { return DefinedAtom::scopeLinkageUnit; }

  virtual Interposable interposable() const { return DefinedAtom::interposeNo; }

  virtual Merge merge() const { return DefinedAtom::mergeNo; }

  virtual ContentType contentType() const { return DefinedAtom::typeStub; }

  virtual Alignment alignment() const { return Alignment(0, 0); }

  virtual SectionChoice sectionChoice() const {
    return DefinedAtom::sectionBasedOnContent;
  }

  virtual StringRef customSectionName() const { return StringRef(); }

  virtual SectionPosition sectionPosition() const { return sectionPositionAny; }

  virtual DeadStripKind deadStrip() const {
    return DefinedAtom::deadStripNormal;
  }

  virtual ContentPermissions permissions() const {
    return DefinedAtom::permR_X;
  }

  virtual bool isAlias() const { return false; }

  virtual ArrayRef<uint8_t> rawContent() const { return ArrayRef<uint8_t>(); }

  virtual reference_iterator begin() const {
    return reference_iterator(*this, nullptr);
  }

  virtual reference_iterator end() const {
    return reference_iterator(*this, nullptr);
  }

  virtual const Reference *derefIterator(const void *iter) const {
    return nullptr;
  }

  virtual void incrementIterator(const void *&iter) const {}

private:
  const File &_file;
  uint32_t _ordinal;
};

/// \brief Simple atom created by the GOT pass.
class TestingGOTAtom : public DefinedAtom {
public:
  TestingGOTAtom(const File &F, const Atom &) : _file(F) {
    static uint32_t lastOrdinal = 0;
    _ordinal = lastOrdinal++;
  }

  virtual const File &file() const { return _file; }

  virtual StringRef name() const { return StringRef(); }

  virtual uint64_t ordinal() const { return _ordinal; }

  virtual uint64_t size() const { return 0; }

  virtual Scope scope() const { return DefinedAtom::scopeLinkageUnit; }

  virtual Interposable interposable() const { return DefinedAtom::interposeNo; }

  virtual Merge merge() const { return DefinedAtom::mergeNo; }

  virtual ContentType contentType() const { return DefinedAtom::typeGOT; }

  virtual Alignment alignment() const { return Alignment(3, 0); }

  virtual SectionChoice sectionChoice() const {
    return DefinedAtom::sectionBasedOnContent;
  }

  virtual StringRef customSectionName() const { return StringRef(); }

  virtual SectionPosition sectionPosition() const { return sectionPositionAny; }

  virtual DeadStripKind deadStrip() const {
    return DefinedAtom::deadStripNormal;
  }

  virtual ContentPermissions permissions() const {
    return DefinedAtom::permRW_;
  }

  virtual bool isAlias() const { return false; }

  virtual ArrayRef<uint8_t> rawContent() const { return ArrayRef<uint8_t>(); }

  virtual reference_iterator begin() const {
    return reference_iterator(*this, nullptr);
  }

  virtual reference_iterator end() const {
    return reference_iterator(*this, nullptr);
  }

  virtual const Reference *derefIterator(const void *iter) const {
    return nullptr;
  }

  virtual void incrementIterator(const void *&iter) const {}

private:
  const File &_file;
  uint32_t _ordinal;
};

class TestingPassFile : public SimpleFile {
public:
  TestingPassFile(const LinkingContext &ctx) : SimpleFile("Testing pass") {}

  virtual void addAtom(const Atom &atom) {
    if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(&atom))
      _definedAtoms._atoms.push_back(defAtom);
    else
      llvm_unreachable("atom has unknown definition kind");
  }

  virtual DefinedAtomRange definedAtoms() {
    return range<std::vector<const DefinedAtom *>::iterator>(
        _definedAtoms._atoms.begin(), _definedAtoms._atoms.end());
  }

  virtual const atom_collection<DefinedAtom> &defined() const {
    return _definedAtoms;
  }
  virtual const atom_collection<UndefinedAtom> &undefined() const {
    return _undefinedAtoms;
  }
  virtual const atom_collection<SharedLibraryAtom> &sharedLibrary() const {
    return _sharedLibraryAtoms;
  }
  virtual const atom_collection<AbsoluteAtom> &absolute() const {
    return _absoluteAtoms;
  }

private:
  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;
};


class TestingStubsPass : public StubsPass {
public:
  TestingStubsPass(const LinkingContext &ctx) : _file(TestingPassFile(ctx)) {}

  virtual bool noTextRelocs() { return true; }

  virtual bool isCallSite(const Reference &ref) {
    if (ref.kindNamespace() != Reference::KindNamespace::testing)
      return false;
    return (ref.kindValue() == CoreLinkingContext::TEST_RELOC_CALL32);
  }

  virtual const DefinedAtom *getStub(const Atom &target) {
    const DefinedAtom *result = new TestingStubAtom(_file, target);
    _file.addAtom(*result);
    return result;
  }

  virtual void addStubAtoms(MutableFile &mergedFile) {
    for (const DefinedAtom *stub : _file.defined()) {
      mergedFile.addAtom(*stub);
    }
  }

private:
  TestingPassFile _file;
};

class TestingGOTPass : public GOTPass {
public:
  TestingGOTPass(const LinkingContext &ctx) : _file(TestingPassFile(ctx)) {}

  virtual bool noTextRelocs() { return true; }

  virtual bool isGOTAccess(const Reference &ref, bool &canBypassGOT) {
    if (ref.kindNamespace() != Reference::KindNamespace::testing)
      return false;
    switch (ref.kindValue()) {
    case CoreLinkingContext::TEST_RELOC_GOT_LOAD32:
      canBypassGOT = true;
      return true;
    case CoreLinkingContext::TEST_RELOC_GOT_USE32:
      canBypassGOT = false;
      return true;
    }
    return false;
  }

  virtual void updateReferenceToGOT(const Reference *ref, bool targetIsNowGOT) {
    const_cast<Reference *>(ref)->setKindValue(
        targetIsNowGOT ? CoreLinkingContext::TEST_RELOC_PCREL32
                       : CoreLinkingContext::TEST_RELOC_LEA32_WAS_GOT);
  }

  virtual const DefinedAtom *makeGOTEntry(const Atom &target) {
    return new TestingGOTAtom(_file, target);
  }

private:
  TestingPassFile _file;
};

} // anonymous namespace

CoreLinkingContext::CoreLinkingContext() {}

bool CoreLinkingContext::validateImpl(raw_ostream &) {
  _writer = createWriterYAML(*this);
  return true;
}

void CoreLinkingContext::addPasses(PassManager &pm) {
  for (StringRef name : _passNames) {
    if (name.equals("layout"))
      pm.add(std::unique_ptr<Pass>(new LayoutPass(registry())));
    else if (name.equals("GOT"))
      pm.add(std::unique_ptr<Pass>(new TestingGOTPass(*this)));
    else if (name.equals("stubs"))
      pm.add(std::unique_ptr<Pass>(new TestingStubsPass(*this)));
    else
      llvm_unreachable("bad pass name");
  }
}

Writer &CoreLinkingContext::writer() const { return *_writer; }

