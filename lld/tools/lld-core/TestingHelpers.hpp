//===- tools/lld/TestingWriter.hpp - Linker Core Test Support -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_TOOLS_TESTING_HELPERS_H_
#define LLD_TOOLS_TESTING_HELPERS_H_

#include "lld/Core/Atom.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Pass.h"
#include "lld/Core/Resolver.h"
#include "lld/ReaderWriter/WriterYAML.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"

#include <vector>

using namespace lld;

//
// Simple atom created by the stubs pass.
//
class TestingStubAtom : public DefinedAtom {
public:
        TestingStubAtom(const File& f, const Atom& shlib) :
                        _file(f), _shlib(shlib) {
          static uint32_t lastOrdinal = 0;
          _ordinal = lastOrdinal++; 
        }

  virtual const File& file() const {
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
    return Alignment(0,0);
  }
  
  virtual SectionChoice sectionChoice() const {
    return DefinedAtom::sectionBasedOnContent;
  }
    
  virtual StringRef customSectionName() const {
    return StringRef();
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
  
  virtual const Reference* derefIterator(const void* iter) const {
    return nullptr;
  }
  
  virtual void incrementIterator(const void*& iter) const {
  
  }
  
private:
  const File&               _file;
  const Atom&               _shlib;
  uint32_t                  _ordinal;
};




//
// Simple atom created by the GOT pass.
//
class TestingGOTAtom : public DefinedAtom {
public:
        TestingGOTAtom(const File& f, const Atom& shlib) :
                        _file(f), _shlib(shlib) {
          static uint32_t lastOrdinal = 0;
          _ordinal = lastOrdinal++; 
        }

  virtual const File& file() const {
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
    return Alignment(3,0);
  }
  
  virtual SectionChoice sectionChoice() const {
    return DefinedAtom::sectionBasedOnContent;
  }
    
  virtual StringRef customSectionName() const {
    return StringRef();
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
  
  virtual const Reference* derefIterator(const void* iter) const {
    return nullptr;
  }
  
  virtual void incrementIterator(const void*& iter) const {
  
  }
  
private:
  const File&               _file;
  const Atom&               _shlib;
  uint32_t                  _ordinal;
};



class TestingPassFile : public File {
public:
  TestingPassFile() : File("Testing pass") {
  }
  
  virtual void addAtom(const Atom &atom) {
    if (const DefinedAtom* defAtom = dyn_cast<DefinedAtom>(&atom)) {
      _definedAtoms._atoms.push_back(defAtom);
    } 
    else {
      assert(0 && "atom has unknown definition kind");
    }
  }
  
  virtual const atom_collection<DefinedAtom>& defined() const {
    return _definedAtoms;
  }
  virtual const atom_collection<UndefinedAtom>& undefined() const {
    return _undefinedAtoms;
  }
  virtual const atom_collection<SharedLibraryAtom>& sharedLibrary() const {
    return _sharedLibraryAtoms;
  }
  virtual const atom_collection<AbsoluteAtom>& absolute() const {
    return _absoluteAtoms;
  }
    
private:
  atom_collection_vector<DefinedAtom>         _definedAtoms;
  atom_collection_vector<UndefinedAtom>       _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom>   _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>        _absoluteAtoms;
};



struct TestingKindMapping {
  const char*           string;
  Reference::Kind       value;
  bool                  isBranch;
  bool                  isGotLoad;
  bool                  isGotUse;
};

//
// Table of fixup kinds in YAML documents used for testing
//
const TestingKindMapping sKinds[] = {
    { "call32",         1,    true,  false, false},
    { "pcrel32",        2,    false, false, false },
    { "gotLoad32",      3,    false, true,  true },
    { "gotUse32",       4,    false, false, true },
    { "lea32wasGot",    5,    false, false, false },
    { nullptr,          0,    false, false, false }
  };



class TestingStubsPass : public StubsPass {
public:
  virtual bool noTextRelocs() {
    return true;
  }

  virtual bool isCallSite(Reference::Kind kind) {
    for (const TestingKindMapping* p = sKinds; p->string != nullptr; ++p) {
      if ( kind == p->value )
        return p->isBranch;
    }
    return false;
  }

  virtual const DefinedAtom* getStub(const Atom& target) {
    const DefinedAtom *result = new TestingStubAtom(_file, target);
    _file.addAtom(*result);
    return result;
  }


  virtual void addStubAtoms(File &mergedFile) {
    for (const DefinedAtom *stub : _file.defined() ) {
      mergedFile.addAtom(*stub);
    }
  }
  
private:
  TestingPassFile    _file;
};



class TestingGOTPass : public GOTPass {
public:
  virtual bool noTextRelocs() {
    return true;
  }

  virtual bool isGOTAccess(Reference::Kind kind, bool &canBypassGOT) {
    for (const TestingKindMapping* p = sKinds; p->string != nullptr; ++p) {
      if ( kind == p->value ) {
        canBypassGOT = p->isGotLoad;
        return (p->isGotUse || p->isGotLoad);
      }
    }
    return false;
  }

  virtual void updateReferenceToGOT(const Reference *ref, bool targetIsNowGOT) {
    if ( targetIsNowGOT )
      (const_cast<Reference*>(ref))->setKind(2); // pcrel32
    else
      (const_cast<Reference*>(ref))->setKind(5); // lea32wasGot
  }

  virtual const DefinedAtom* makeGOTEntry(const Atom &target) {
    return new TestingGOTAtom(_file, target);
  }
  
private:
  TestingPassFile    _file;
};


class TestingWriterOptionsYAML : public lld::WriterOptionsYAML {
public:
  TestingWriterOptionsYAML(bool stubs, bool got)
    : _doStubs(stubs), _doGOT(got) {
  }

  virtual StubsPass *stubPass() const {
    if ( _doStubs )
      return const_cast<TestingStubsPass*>(&_stubsPass);
    else
      return nullptr;
  }
  
  virtual GOTPass *gotPass() const {
     if ( _doGOT )
      return const_cast<TestingGOTPass*>(&_gotPass);
    else
      return nullptr;
  }
  
  virtual StringRef kindToString(Reference::Kind value) const {
    for (const TestingKindMapping* p = sKinds; p->string != nullptr; ++p) {
      if ( value == p->value)
        return p->string;
    }
    return StringRef("???");
  }
private:
  bool              _doStubs;
  bool              _doGOT;
  TestingStubsPass  _stubsPass;
  TestingGOTPass    _gotPass;
};


class TestingReaderOptionsYAML : public lld::ReaderOptionsYAML {
  virtual Reference::Kind kindFromString(StringRef kindName) const {
    for (const TestingKindMapping* p = sKinds; p->string != nullptr; ++p) {
      if ( kindName.equals(p->string) )
        return p->value;
    }
    int k;
    if (kindName.getAsInteger(0, k))
      k = 0;
    return k;
  }
};



#endif // LLD_TOOLS_TESTING_HELPERS_H_
