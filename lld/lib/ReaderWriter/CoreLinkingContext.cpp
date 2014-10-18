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
#include "lld/Core/Simple.h"
#include "lld/Passes/LayoutPass.h"
#include "lld/Passes/RoundTripYAMLPass.h"
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
    else
      llvm_unreachable("bad pass name");
  }
}

Writer &CoreLinkingContext::writer() const { return *_writer; }
