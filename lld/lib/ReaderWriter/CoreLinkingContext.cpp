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

  const File &file() const override { return _file; }

  StringRef name() const override { return StringRef(); }

  uint64_t ordinal() const override { return _ordinal; }

  uint64_t size() const override { return 0; }

  Scope scope() const override { return DefinedAtom::scopeLinkageUnit; }

  Interposable interposable() const override { return DefinedAtom::interposeNo; }

  Merge merge() const override { return DefinedAtom::mergeNo; }

  ContentType contentType() const override { return DefinedAtom::typeStub; }

  Alignment alignment() const override { return Alignment(0, 0); }

  SectionChoice sectionChoice() const override {
    return DefinedAtom::sectionBasedOnContent;
  }

  StringRef customSectionName() const override { return StringRef(); }

  SectionPosition sectionPosition() const override { return sectionPositionAny; }

  DeadStripKind deadStrip() const override {
    return DefinedAtom::deadStripNormal;
  }

  ContentPermissions permissions() const override {
    return DefinedAtom::permR_X;
  }

  ArrayRef<uint8_t> rawContent() const override { return ArrayRef<uint8_t>(); }

  reference_iterator begin() const override {
    return reference_iterator(*this, nullptr);
  }

  reference_iterator end() const override {
    return reference_iterator(*this, nullptr);
  }

  const Reference *derefIterator(const void *iter) const override {
    return nullptr;
  }

  void incrementIterator(const void *&iter) const override {}

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

  const File &file() const override { return _file; }

  StringRef name() const override { return StringRef(); }

  uint64_t ordinal() const override { return _ordinal; }

  uint64_t size() const override { return 0; }

  Scope scope() const override { return DefinedAtom::scopeLinkageUnit; }

  Interposable interposable() const override { return DefinedAtom::interposeNo; }

  Merge merge() const override { return DefinedAtom::mergeNo; }

  ContentType contentType() const override { return DefinedAtom::typeGOT; }

  Alignment alignment() const override { return Alignment(3, 0); }

  SectionChoice sectionChoice() const override {
    return DefinedAtom::sectionBasedOnContent;
  }

  StringRef customSectionName() const override { return StringRef(); }

  SectionPosition sectionPosition() const override { return sectionPositionAny; }

  DeadStripKind deadStrip() const override {
    return DefinedAtom::deadStripNormal;
  }

  ContentPermissions permissions() const override {
    return DefinedAtom::permRW_;
  }

  ArrayRef<uint8_t> rawContent() const override { return ArrayRef<uint8_t>(); }

  reference_iterator begin() const override {
    return reference_iterator(*this, nullptr);
  }

  reference_iterator end() const override {
    return reference_iterator(*this, nullptr);
  }

  const Reference *derefIterator(const void *iter) const override {
    return nullptr;
  }

  void incrementIterator(const void *&iter) const override {}

private:
  const File &_file;
  uint32_t _ordinal;
};

class TestingPassFile : public SimpleFile {
public:
  TestingPassFile(const LinkingContext &ctx) : SimpleFile("Testing pass") {}

  void addAtom(const Atom &atom) override {
    if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(&atom))
      _definedAtoms._atoms.push_back(defAtom);
    else
      llvm_unreachable("atom has unknown definition kind");
  }

  DefinedAtomRange definedAtoms() override {
    return range<std::vector<const DefinedAtom *>::iterator>(
        _definedAtoms._atoms.begin(), _definedAtoms._atoms.end());
  }

  const atom_collection<DefinedAtom> &defined() const override {
    return _definedAtoms;
  }
  const atom_collection<UndefinedAtom> &undefined() const override {
    return _undefinedAtoms;
  }
  const atom_collection<SharedLibraryAtom> &sharedLibrary() const override {
    return _sharedLibraryAtoms;
  }
  const atom_collection<AbsoluteAtom> &absolute() const override {
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

  bool noTextRelocs() override { return true; }

  bool isCallSite(const Reference &ref) override {
    if (ref.kindNamespace() != Reference::KindNamespace::testing)
      return false;
    return (ref.kindValue() == CoreLinkingContext::TEST_RELOC_CALL32);
  }

  const DefinedAtom *getStub(const Atom &target) override {
    const DefinedAtom *result = new TestingStubAtom(_file, target);
    _file.addAtom(*result);
    return result;
  }

  void addStubAtoms(MutableFile &mergedFile) override {
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

  bool noTextRelocs() override { return true; }

  bool isGOTAccess(const Reference &ref, bool &canBypassGOT) override {
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

  void updateReferenceToGOT(const Reference *ref, bool targetIsNowGOT) override {
    const_cast<Reference *>(ref)->setKindValue(
        targetIsNowGOT ? CoreLinkingContext::TEST_RELOC_PCREL32
                       : CoreLinkingContext::TEST_RELOC_LEA32_WAS_GOT);
  }

  const DefinedAtom *makeGOTEntry(const Atom &target) override {
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
