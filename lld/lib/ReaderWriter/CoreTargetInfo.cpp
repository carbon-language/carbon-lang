//===- lib/ReaderWriter/CoreTargetInfo.cpp --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/CoreTargetInfo.h"

#include "lld/Core/Pass.h"
#include "lld/Core/PassManager.h"
#include "lld/Passes/LayoutPass.h"

#include "llvm/ADT/ArrayRef.h"


using namespace lld;

namespace {

/// \brief Simple atom created by the stubs pass.
class TestingStubAtom : public DefinedAtom {
public:
  TestingStubAtom(const File &F, const Atom&) : _file(F) {
    static uint32_t lastOrdinal = 0;
    _ordinal = lastOrdinal++;
  }

  virtual const File &file() const {
    return _file;
  }

  virtual StringRef name() const {
    return StringRef();
  }

  virtual uint64_t ordinal() const {
    return _ordinal;
  }

  virtual uint64_t size() const {
    return 0;
  }

  virtual Scope scope() const {
    return DefinedAtom::scopeLinkageUnit;
  }

  virtual Interposable interposable() const {
    return DefinedAtom::interposeNo;
  }

  virtual Merge merge() const {
    return DefinedAtom::mergeNo;
  }

  virtual ContentType contentType() const  {
    return DefinedAtom::typeStub;
  }

  virtual Alignment alignment() const {
    return Alignment(0, 0);
  }

  virtual SectionChoice sectionChoice() const {
    return DefinedAtom::sectionBasedOnContent;
  }

  virtual StringRef customSectionName() const {
    return StringRef();
  }
  
  virtual SectionPosition sectionPosition() const {
    return sectionPositionAny;
  }
  
  virtual DeadStripKind deadStrip() const {
    return DefinedAtom::deadStripNormal;
  }

  virtual ContentPermissions permissions() const  {
    return DefinedAtom::permR_X;
  }

  virtual bool isThumb() const {
    return false;
  }

  virtual bool isAlias() const {
    return false;
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>();
  }

  virtual reference_iterator begin() const {
    return reference_iterator(*this, nullptr);
  }

  virtual reference_iterator end() const {
    return reference_iterator(*this, nullptr);
  }

  virtual const Reference *derefIterator(const void *iter) const {
    return nullptr;
  }

  virtual void incrementIterator(const void *&iter) const {
  }

private:
  const File &_file;
  uint32_t _ordinal;
};

/// \brief Simple atom created by the GOT pass.
class TestingGOTAtom : public DefinedAtom {
public:
  TestingGOTAtom(const File &F, const Atom&) : _file(F) {
    static uint32_t lastOrdinal = 0;
    _ordinal = lastOrdinal++;
  }

  virtual const File &file() const {
    return _file;
  }

  virtual StringRef name() const {
    return StringRef();
  }

  virtual uint64_t ordinal() const {
    return _ordinal;
  }

  virtual uint64_t size() const {
    return 0;
  }

  virtual Scope scope() const {
    return DefinedAtom::scopeLinkageUnit;
  }

  virtual Interposable interposable() const {
    return DefinedAtom::interposeNo;
  }

  virtual Merge merge() const {
    return DefinedAtom::mergeNo;
  }

  virtual ContentType contentType() const  {
    return DefinedAtom::typeGOT;
  }

  virtual Alignment alignment() const {
    return Alignment(3, 0);
  }

  virtual SectionChoice sectionChoice() const {
    return DefinedAtom::sectionBasedOnContent;
  }

  virtual StringRef customSectionName() const {
    return StringRef();
  }

  virtual SectionPosition sectionPosition() const {
    return sectionPositionAny;
  }

  virtual DeadStripKind deadStrip() const {
    return DefinedAtom::deadStripNormal;
  }

  virtual ContentPermissions permissions() const  {
    return DefinedAtom::permRW_;
  }

  virtual bool isThumb() const {
    return false;
  }

  virtual bool isAlias() const {
    return false;
  }

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>();
  }

  virtual reference_iterator begin() const {
    return reference_iterator(*this, nullptr);
  }

  virtual reference_iterator end() const {
    return reference_iterator(*this, nullptr);
  }

  virtual const Reference *derefIterator(const void *iter) const {
    return nullptr;
  }

  virtual void incrementIterator(const void *&iter) const {
  }

private:
  const File &_file;
  uint32_t _ordinal;
};

class TestingPassFile : public MutableFile {
public:
  TestingPassFile(const TargetInfo &ti) : MutableFile(ti, "Testing pass") {}

  virtual void addAtom(const Atom &atom) {
    if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(&atom))
      _definedAtoms._atoms.push_back(defAtom);
    else
      llvm_unreachable("atom has unknown definition kind");
  }

  virtual DefinedAtomRange definedAtoms() {
    return range<std::vector<const DefinedAtom*>::iterator>(
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
  atom_collection_vector<DefinedAtom>       _definedAtoms;
  atom_collection_vector<UndefinedAtom>     _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>      _absoluteAtoms;
};

struct TestingKindMapping {
  const char     *string;
  int32_t         value;
  bool            isBranch;
  bool            isGotLoad;
  bool            isGotUse;
};

//
// Table of fixup kinds in YAML documents used for testing
//
const TestingKindMapping sKinds[] = {
    {"in-group",      -3, false,  false, false},
    {"layout-after",  -2, false,  false, false},
    {"layout-before", -1, false,  false, false},
    {"call32",         2, true,  false, false},
    {"pcrel32",        3, false, false, false},
    {"gotLoad32",      7, false, true,  true},
    {"gotUse32",       9, false, false, true},
    {"lea32wasGot",    8, false, false, false},
    {nullptr,          0, false, false, false}
  };

class TestingStubsPass : public StubsPass {
public:
  TestingStubsPass(const TargetInfo &ti) : _file(TestingPassFile(ti))
  {}

  virtual bool noTextRelocs() {
    return true;
  }

  virtual bool isCallSite(int32_t kind) {
    for (const TestingKindMapping *p = sKinds; p->string != nullptr; ++p) {
      if (kind == p->value)
        return p->isBranch;
    }
    return false;
  }

  virtual const DefinedAtom *getStub(const Atom &target) {
    const DefinedAtom *result = new TestingStubAtom(_file, target);
    _file.addAtom(*result);
    return result;
  }

  virtual void addStubAtoms(MutableFile &mergedFile) {
    for (const DefinedAtom *stub : _file.defined() ) {
      mergedFile.addAtom(*stub);
    }
  }

private:
  TestingPassFile _file;
};

class TestingGOTPass : public GOTPass {
public:
  TestingGOTPass(const TargetInfo &ti) : _file(TestingPassFile(ti))
  {}

  virtual bool noTextRelocs() {
    return true;
  }

  virtual bool isGOTAccess(int32_t kind, bool &canBypassGOT) {
    for (const TestingKindMapping *p = sKinds; p->string != nullptr; ++p) {
      if (kind == p->value) {
        canBypassGOT = p->isGotLoad;
        return p->isGotUse || p->isGotLoad;
      }
    }
    return false;
  }

  virtual void updateReferenceToGOT(const Reference *ref, bool targetIsNowGOT) {
    if (targetIsNowGOT)
      const_cast<Reference*>(ref)->setKind(3); // pcrel32
    else
      const_cast<Reference*>(ref)->setKind(8); // lea32wasGot
  }

  virtual const DefinedAtom *makeGOTEntry(const Atom &target) {
    return new TestingGOTAtom(_file, target);
  }

private:
  TestingPassFile _file;
};

} // anonymous namespace


CoreTargetInfo::CoreTargetInfo() {
}


void CoreTargetInfo::addPasses(PassManager &pm) const {
  for (StringRef name : _passNames) {
    if ( name.equals("layout") )
      pm.add(std::unique_ptr<Pass>((new LayoutPass())));
    else if ( name.equals("GOT") )
      pm.add(std::unique_ptr<Pass>(new TestingGOTPass(*this)));
    else if ( name.equals("stubs") )
      pm.add(std::unique_ptr<Pass>(new TestingStubsPass(*this)));
    else
      llvm_unreachable("bad pass name");
  }
}

error_code CoreTargetInfo::parseFile(std::unique_ptr<MemoryBuffer> &mb,
                            std::vector<std::unique_ptr<File>> &result) const {
  if (!_reader)
    _reader = createReaderYAML(*this);
  return _reader->parseFile(mb,result);
}

Writer &CoreTargetInfo::writer() const {
  if (!_writer)
    _writer = createWriterYAML(*this);
  return *_writer;
}


ErrorOr<Reference::Kind> 
CoreTargetInfo::relocKindFromString(StringRef str) const {
  for (const TestingKindMapping *p = sKinds; p->string != nullptr; ++p) {
    if (str.equals(p->string))
      return p->value;
  }
  return make_error_code(yaml_reader_error::illegal_value);
}

ErrorOr<std::string> 
CoreTargetInfo::stringFromRelocKind(Reference::Kind kind) const {
  for (const TestingKindMapping *p = sKinds; p->string != nullptr; ++p) {
    if (kind == p->value)
      return std::string(p->string);
  }
  return make_error_code(yaml_reader_error::illegal_value);
}



