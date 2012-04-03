//===- Core/YamlReader.cpp - Reads YAML -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/YamlReader.h"
#include "YamlKeyValues.h"
#include "lld/Core/Atom.h"
#include "lld/Core/AbsoluteAtom.h"
#include "lld/Core/Error.h"
#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Platform.h"
#include "lld/Core/Reference.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/UndefinedAtom.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"

#include <cstring>
#include <vector>



namespace lld {
namespace yaml {

namespace {

class YAML {
public:
  struct Entry {
    Entry(const char *k, const char *v, std::vector<uint8_t>* vs, 
          int d, bool bd, bool bs)
      : key(strdup(k))
      , value(v ? strdup(v) : nullptr)
      , valueSequenceBytes(vs)
      , depth(d)
      , beginSequence(bs)
      , beginDocument(bd) {}

    const char *          key;
    const char *          value;
    std::vector<uint8_t>* valueSequenceBytes;
    int                   depth;
    bool                  beginSequence;
    bool                  beginDocument;
  };

  static void parse(llvm::MemoryBuffer *mb, std::vector<const Entry *>&);

private:
  enum State {
    start,
    inHeaderComment,
    inTripleDash,
    inTriplePeriod,
    inDocument,
    inKey,
    inSpaceBeforeValue,
    inValue,
    inValueSequence,
    inValueSequenceEnd
  };
};


void YAML::parse(llvm::MemoryBuffer *mb, std::vector<const Entry *> &entries) {
  State state = start;
  char key[64];
  char value[64];
  char *p = nullptr;
  unsigned int lineNumber = 1;
  int depth = 0;
  bool nextKeyIsStartOfDocument = false;
  bool nextKeyIsStartOfSequence = false;
  std::vector<uint8_t>* sequenceBytes = nullptr;
  unsigned contentByte = 0;
  for (const char *s = mb->getBufferStart(); s < mb->getBufferEnd(); ++s) {
    char c = *s;
    if (c == '\n')
      ++lineNumber;
    switch (state) {
    case start:
      if (c == '#')
        state = inHeaderComment;
      else if (c == '-') {
        p = &key[0];
        *p++ = c;
        state = inTripleDash;
      }
      break;
    case inHeaderComment:
      if (c == '\n') {
        state = start;
      }
      break;
    case inTripleDash:
      if (c == '-') {
        *p++ = c;
      } else if (c == '\n') {
        *p = '\0';
        if (strcmp(key, "---") != 0)
          return;
        depth = 0;
        state = inDocument;
        nextKeyIsStartOfDocument = true;
      } else {
        return;
      }
      break;
    case inTriplePeriod:
      if (c == '.') {
        *p++ = c;
      } else if (c == '\n') {
        *p = '\0';
        if (strcmp(key, "...") != 0)
          return;
        depth = 0;
        state = inHeaderComment;
      } else {
        return;
      }
      break;
    case inDocument:
      if (isalnum(c)) {
        state = inKey;
        p = &key[0];
        *p++ = c;
      } else if (c == '-') {
        if (depth == 0) {
          p = &key[0];
          *p++ = c;
          state = inTripleDash;
        } else {
          nextKeyIsStartOfSequence = true;
          ++depth;
        }
      } else if (c == ' ') {
        ++depth;
      } else if (c == '.') {
        p = &key[0];
        *p++ = c;
        state = inTriplePeriod;
      } else if (c == '\n') {
        // ignore empty lines
        depth = 0;
      } else if (c == '\t') {
        llvm::report_fatal_error("TAB character found in yaml file");
      } else {
        return;
      }
      break;
    case inKey:
      if (isalnum(c) || (c == '-')) {
        *p++ = c;
      } else if (c == ':') {
        *p = '\0';
        state = inSpaceBeforeValue;
      } else if (c == '\n') {
        *p = '\0';
        if (strcmp(key, "---") == 0)
          state = inDocument;
        else
          return;
      } else {
        return;
      }
      break;
    case inSpaceBeforeValue:
      if (isalnum(c) || (c == '-') || (c == '_')) {
        p = &value[0];
        *p++ = c;
        state = inValue;
      } else if (c == '\n') {
        entries.push_back(new Entry(key, "", nullptr, depth,
                                    nextKeyIsStartOfDocument,
                                    nextKeyIsStartOfSequence));
        nextKeyIsStartOfSequence = false;
        nextKeyIsStartOfDocument = false;
        state = inDocument;
        depth = 0;
      } else if (c == '[') {
        contentByte = 0;
        sequenceBytes = new std::vector<uint8_t>();
        state = inValueSequence;
      } else if (c == ' ') {
        // eat space
      } else if (c == '\t') {
        llvm::report_fatal_error("TAB character found in yaml file");
      } else {
        return;
      }
      break;
    case inValue:
      if (c == '\n') {
        *p = '\0';
        entries.push_back(new Entry(key, value, nullptr, depth,
                                    nextKeyIsStartOfDocument,
                                    nextKeyIsStartOfSequence));
        nextKeyIsStartOfSequence = false;
        nextKeyIsStartOfDocument = false;
        state = inDocument;
        depth = 0;
      }
      else {
        *p++ = c;
      } 
      break;
    case inValueSequence:
      if (c == ']') {
        sequenceBytes->push_back(contentByte);
        state = inValueSequenceEnd;
      }
      else if ( (c == ' ') || (c == '\n') ) {
        // eat white space
      }
      else if (c == ',') {
        sequenceBytes->push_back(contentByte);
      }
      else if ( isdigit(c) ) {
        contentByte = (contentByte << 4) | (c-'0');
      } 
      else if ( ('a' <= tolower(c)) && (tolower(c) <= 'f') ) {
        contentByte = (contentByte << 4) | (tolower(c)-'a'+10);
      }
      else {
        llvm::report_fatal_error("non-hex digit found in content [ ]");
      }
      break;
    case inValueSequenceEnd:
      if (c == '\n') {
        entries.push_back(new Entry(key, nullptr, sequenceBytes, depth,
                                    nextKeyIsStartOfDocument,
                                    nextKeyIsStartOfSequence));
        nextKeyIsStartOfSequence = false;
        nextKeyIsStartOfDocument = false;
        state = inDocument;
        depth = 0;
      }
      break;
    }
  }
}



class YAMLReference : public Reference {
public: 
                YAMLReference() : _target(nullptr), _targetName(nullptr),
                                   _offsetInAtom(0), _addend(0), _kind(0) { }

  virtual uint64_t offsetInAtom() const {
    return _offsetInAtom;
  }
  
  virtual Kind kind() const {
    return _kind;
  }
  
  virtual void setKind(Kind k) {
    _kind = k;
  }

  virtual const Atom* target() const {
    return _target;
  }
  
  virtual Addend addend() const {
    return _addend;
  }

  virtual void setTarget(const Atom* newAtom) {
    _target = newAtom;
  }

  const Atom*  _target;
  const char*  _targetName;
  uint64_t     _offsetInAtom;
  Addend       _addend;
  Kind         _kind;
};



class YAMLDefinedAtom;

class YAMLFile : public File {
public:
  YAMLFile()
    : File("path")
    , _lastRefIndex(0) {}

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

  virtual void addAtom(const Atom&) {
    assert(0 && "cannot add atoms to YAML files");
  }
   
  void bindTargetReferences();
  void addDefinedAtom(YAMLDefinedAtom* atom, const char* refName);
  void addUndefinedAtom(UndefinedAtom* atom);
  void addSharedLibraryAtom(SharedLibraryAtom* atom);
  void addAbsoluteAtom(AbsoluteAtom* atom);
  Atom* findAtom(const char* name);
  
  struct NameAtomPair {
                 NameAtomPair(const char* n, Atom* a) : name(n), atom(a) {}
    const char*  name;
    Atom*        atom;
  };

  atom_collection_vector<DefinedAtom>         _definedAtoms;
  atom_collection_vector<UndefinedAtom>       _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom>   _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>        _absoluteAtoms;
  std::vector<YAMLReference>                  _references;
  std::vector<NameAtomPair>                   _nameToAtomMapping;
  unsigned int                                _lastRefIndex;
};



class YAMLDefinedAtom : public DefinedAtom {
public:
  YAMLDefinedAtom( uint32_t ord
          , YAMLFile& file
          , DefinedAtom::Scope scope
          , DefinedAtom::ContentType type
          , DefinedAtom::SectionChoice sectionChoice
          , DefinedAtom::Interposable interpose
          , DefinedAtom::Merge merge
          , DefinedAtom::DeadStripKind deadStrip
          , DefinedAtom::ContentPermissions perms
          , bool isThumb
          , bool isAlias
          , DefinedAtom::Alignment alignment
          , const char* name
          , const char* sectionName
          , uint64_t size
          , std::vector<uint8_t>* content)
    : _file(file)
    , _name(name)
    , _sectionName(sectionName)
    , _size(size)
    , _ord(ord)
    , _content(content)
    , _alignment(alignment)
    , _scope(scope)
    , _type(type)
    , _sectionChoice(sectionChoice)
    , _interpose(interpose)
    , _merge(merge)
    , _deadStrip(deadStrip)
    , _permissions(perms)
    , _isThumb(isThumb)
    , _isAlias(isAlias)
    , _refStartIndex(file._lastRefIndex)
    , _refEndIndex(file._references.size()) {
    file._lastRefIndex = _refEndIndex;
  }

  virtual const class File& file() const {
    return _file;
  }

  virtual StringRef name() const {
    if (_name == nullptr)
      return StringRef();
    else
      return _name;
  }

 virtual uint64_t size() const {
    return (_content ? _content->size() : _size);
  }

  virtual DefinedAtom::Scope scope() const {
    return _scope;
  }
  
  virtual DefinedAtom::Interposable interposable() const {
    return _interpose;
  }
  
  virtual DefinedAtom::Merge merge() const {
    return _merge;
  }

  virtual DefinedAtom::ContentType contentType() const {
    return _type;
  }
    
  virtual DefinedAtom::Alignment alignment() const {
    return _alignment;
  }
  
  virtual DefinedAtom::SectionChoice sectionChoice() const {
    return _sectionChoice;
  }

  virtual StringRef customSectionName() const {
    return _sectionName;
  }
    
  virtual DefinedAtom::DeadStripKind deadStrip() const {
    return _deadStrip;
  }
    
  virtual DefinedAtom::ContentPermissions permissions() const {
    return _permissions;
  }
  
  virtual bool isThumb() const {
    return _isThumb;
  }
    
  virtual bool isAlias() const {
    return _isAlias;
  }
  
 ArrayRef<uint8_t> rawContent() const {
    if (_content != nullptr)
      return ArrayRef<uint8_t>(*_content);
    else
      return ArrayRef<uint8_t>();
  }
 
  virtual uint64_t ordinal() const {
    return _ord;
  }

  DefinedAtom::reference_iterator referencesBegin() const {
    uintptr_t index = _refStartIndex;
    const void* it = reinterpret_cast<const void*>(index);
    return reference_iterator(*this, it);
  }
    
  DefinedAtom::reference_iterator referencesEnd() const {
    uintptr_t index = _refEndIndex;
    const void* it = reinterpret_cast<const void*>(index);
    return reference_iterator(*this, it);
  }

  const Reference* derefIterator(const void* it) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(it);
    assert(index >= _refStartIndex);
    assert(index < _refEndIndex);
    assert(index < _file._references.size());
    return &_file._references[index];
  }
    
  void incrementIterator(const void*& it) const {
    uintptr_t index = reinterpret_cast<uintptr_t>(it);
    ++index;
    it = reinterpret_cast<const void*>(index);
  }

  
    
  void bindTargetReferences() const {
    for (unsigned int i=_refStartIndex; i < _refEndIndex; ++i) {
      const char* targetName = _file._references[i]._targetName;
      Atom* targetAtom = _file.findAtom(targetName);
      _file._references[i]._target = targetAtom;
    }
  }
  
private:
  YAMLFile&                   _file;
  const char *                _name;
  const char *                _sectionName;
  unsigned long               _size;
  uint32_t                    _ord;
  std::vector<uint8_t>*       _content;
  DefinedAtom::Alignment      _alignment;
  DefinedAtom::Scope          _scope;
  DefinedAtom::ContentType    _type;
  DefinedAtom::SectionChoice  _sectionChoice;
  DefinedAtom::Interposable   _interpose;
  DefinedAtom::Merge          _merge;
  DefinedAtom::DeadStripKind  _deadStrip;
  DefinedAtom::ContentPermissions _permissions;
  bool                        _isThumb;
  bool                        _isAlias;
  unsigned int                _refStartIndex;
  unsigned int                _refEndIndex;
};


class YAMLUndefinedAtom : public UndefinedAtom {
public:
        YAMLUndefinedAtom(YAMLFile& f, int32_t ord, const char* nm, 
                          UndefinedAtom::CanBeNull cbn)
            : _file(f), _name(nm), _ordinal(ord), _canBeNull(cbn) { }

  virtual const class File& file() const {
    return _file;
  }

  virtual StringRef name() const {
    return _name;
  }

  virtual CanBeNull canBeNull() const {
    return _canBeNull;
  }
  
 
private:
  YAMLFile&                   _file;
  const char *                _name;
  uint32_t                    _ordinal;
  UndefinedAtom::CanBeNull     _canBeNull;
};



class YAMLSharedLibraryAtom : public SharedLibraryAtom {
public:
        YAMLSharedLibraryAtom(YAMLFile& f, int32_t ord, const char* nm, 
                                const char* ldnm, bool cbn)
            : _file(f), _name(nm), _ordinal(ord), 
              _loadName(ldnm), _canBeNull(cbn) { }

  virtual const class File& file() const {
    return _file;
  }

  virtual StringRef name() const {
    return _name;
  }

  virtual StringRef loadName() const {
    return _loadName;
  }

  virtual bool canBeNullAtRuntime() const {
    return _canBeNull;
  }
  
 
private:
  YAMLFile&                   _file;
  const char *                _name;
  uint32_t                    _ordinal;
  const char *                _loadName;
  bool                        _canBeNull;
};



class YAMLAbsoluteAtom : public AbsoluteAtom {
public:
        YAMLAbsoluteAtom(YAMLFile& f, int32_t ord, const char* nm, uint64_t v)
            : _file(f), _name(nm), _ordinal(ord), _value(v) { }

  virtual const class File& file() const {
    return _file;
  }

  virtual StringRef name() const {
    return _name;
  }

  virtual uint64_t value() const {
    return _value;
  }

private:
  YAMLFile&        _file;
  const char *     _name;
  uint32_t         _ordinal;
  uint64_t         _value;
};



void YAMLFile::bindTargetReferences() {
    for (defined_iterator it = definedAtomsBegin(); it != definedAtomsEnd(); ++it) {
      const YAMLDefinedAtom* atom = reinterpret_cast<const YAMLDefinedAtom*>(*it);   
      atom->bindTargetReferences();
    }
}

Atom* YAMLFile::findAtom(const char* name) {
  for (std::vector<NameAtomPair>::const_iterator it = _nameToAtomMapping.begin();
                                    it != _nameToAtomMapping.end(); ++it) {
    if ( strcmp(name, it->name) == 0 )
      return it->atom;
  }
  llvm::report_fatal_error("reference to atom that does not exist");
}

void YAMLFile::addDefinedAtom(YAMLDefinedAtom* atom, const char* refName) {
  _definedAtoms._atoms.push_back(atom);
  assert(refName != nullptr);
  _nameToAtomMapping.push_back(NameAtomPair(refName, atom));
}

void YAMLFile::addUndefinedAtom(UndefinedAtom* atom) {
  _undefinedAtoms._atoms.push_back(atom);
  _nameToAtomMapping.push_back(NameAtomPair(atom->name().data(), atom));
}

void YAMLFile::addSharedLibraryAtom(SharedLibraryAtom* atom) {
  _sharedLibraryAtoms._atoms.push_back(atom);
  _nameToAtomMapping.push_back(NameAtomPair(atom->name().data(), atom));
}

void YAMLFile::addAbsoluteAtom(AbsoluteAtom* atom) {
  _absoluteAtoms._atoms.push_back(atom);
  _nameToAtomMapping.push_back(NameAtomPair(atom->name().data(), atom));
}


class YAMLAtomState {
public:
  YAMLAtomState(Platform& platform);
  
  void setName(const char *n);
  void setRefName(const char *n);
  void setAlign2(const char *n);

  void setFixupKind(const char *n);
  void setFixupTarget(const char *n);
  void addFixup(YAMLFile *f);

  void makeAtom(YAMLFile&);

  Platform&                   _platform;
  const char *                _name;
  const char *                _refName;
  const char *                _sectionName;
  const char*                 _loadName;
  unsigned long long          _size;
  uint64_t                    _value;
  uint32_t                    _ordinal;
  std::vector<uint8_t>*       _content;
  DefinedAtom::Alignment      _alignment;
  Atom::Definition            _definition;
  DefinedAtom::Scope          _scope;
  DefinedAtom::ContentType    _type;
  DefinedAtom::SectionChoice  _sectionChoice;
  DefinedAtom::Interposable   _interpose;
  DefinedAtom::Merge          _merge;
  DefinedAtom::DeadStripKind  _deadStrip;
  DefinedAtom::ContentPermissions _permissions;
  bool                        _isThumb;
  bool                        _isAlias;
  UndefinedAtom::CanBeNull    _canBeNull;
  YAMLReference               _ref;
};


YAMLAtomState::YAMLAtomState(Platform& platform)
  : _platform(platform)
  , _name(nullptr)
  , _refName(nullptr)
  , _sectionName(nullptr)
  , _loadName(nullptr)
  , _size(0)
  , _value(0)
  , _ordinal(0)
  , _content(nullptr)
  , _alignment(0, 0)
  , _definition(KeyValues::definitionDefault)
  , _scope(KeyValues::scopeDefault)
  , _type(KeyValues::contentTypeDefault)
  , _sectionChoice(KeyValues::sectionChoiceDefault)
  , _interpose(KeyValues::interposableDefault)
  , _merge(KeyValues::mergeDefault)
  , _deadStrip(KeyValues::deadStripKindDefault)
  , _permissions(KeyValues::permissionsDefault)
  , _isThumb(KeyValues::isThumbDefault)
  , _isAlias(KeyValues::isAliasDefault) 
  , _canBeNull(KeyValues::canBeNullDefault)
  {
  }


void YAMLAtomState::makeAtom(YAMLFile& f) {
  if ( _definition == Atom::definitionRegular ) {
    YAMLDefinedAtom *a = new YAMLDefinedAtom(_ordinal, f, _scope, _type,
                          _sectionChoice, _interpose, _merge, _deadStrip,
                          _permissions, _isThumb, _isAlias, 
                          _alignment, _name, _sectionName, _size, _content);
    f.addDefinedAtom(a, _refName ? _refName : _name);
    ++_ordinal;
  }
  else if ( _definition == Atom::definitionUndefined ) {
    UndefinedAtom *a = new YAMLUndefinedAtom(f, _ordinal, _name, _canBeNull);
    f.addUndefinedAtom(a);
    ++_ordinal;
  }
  else if ( _definition == Atom::definitionSharedLibrary ) {
    bool nullable = (_canBeNull == UndefinedAtom::canBeNullAtRuntime);
    SharedLibraryAtom *a = new YAMLSharedLibraryAtom(f, _ordinal, _name, 
                                                      _loadName, nullable);
    f.addSharedLibraryAtom(a);
    ++_ordinal;
  }
   else if ( _definition == Atom::definitionAbsolute ) {
    AbsoluteAtom *a = new YAMLAbsoluteAtom(f, _ordinal, _name, _value);
    f.addAbsoluteAtom(a);
    ++_ordinal;
  }
 
  // reset state for next atom
  _name             = nullptr;
  _refName          = nullptr;
  _sectionName      = nullptr;
  _loadName         = nullptr;
  _size             = 0;
  _value            = 0;
  _ordinal          = 0;
  _content          = nullptr;
  _alignment.powerOf2= 0;
  _alignment.modulus = 0;
  _definition       = KeyValues::definitionDefault;
  _scope            = KeyValues::scopeDefault;
  _type             = KeyValues::contentTypeDefault;
  _sectionChoice    = KeyValues::sectionChoiceDefault;
  _interpose        = KeyValues::interposableDefault;
  _merge            = KeyValues::mergeDefault;
  _deadStrip        = KeyValues::deadStripKindDefault;
  _permissions      = KeyValues::permissionsDefault;
  _isThumb          = KeyValues::isThumbDefault;
  _isAlias          = KeyValues::isAliasDefault;
  _canBeNull        = KeyValues::canBeNullDefault;
  _ref._target       = nullptr;
  _ref._targetName   = nullptr;
  _ref._addend       = 0;
  _ref._offsetInAtom = 0;
  _ref._kind         = 0;
}

void YAMLAtomState::setName(const char *n) {
  _name = n;
}

void YAMLAtomState::setRefName(const char *n) {
  _refName = n;
}

void YAMLAtomState::setAlign2(const char *s) {
  if (StringRef(s).getAsInteger(10, _alignment.powerOf2))
    _alignment.powerOf2 = 1;
}

void YAMLAtomState::setFixupKind(const char *s) {
  _ref._kind = _platform.kindFromString(StringRef(s));
}

void YAMLAtomState::setFixupTarget(const char *s) {
  _ref._targetName = s;
}


void YAMLAtomState::addFixup(YAMLFile *f) {
  f->_references.push_back(_ref);
  // clear for next ref
  _ref._target       = nullptr;
  _ref._targetName   = nullptr;
  _ref._addend       = 0;
  _ref._offsetInAtom = 0;
  _ref._kind         = 0;
}


} // anonymous namespace





/// parseObjectText - Parse the specified YAML formatted MemoryBuffer
/// into lld::File object(s) and append each to the specified vector<File*>.
error_code parseObjectText( llvm::MemoryBuffer *mb
                          , Platform& platform
                          , std::vector<const File *> &result) {
  std::vector<const YAML::Entry *> entries;
  YAML::parse(mb, entries);

  YAMLFile *file = nullptr;
  YAMLAtomState atomState(platform);
  bool inAtoms       = false;
  bool inFixups      = false;
  int depthForAtoms  = -1;
  int depthForFixups = -1;
  int lastDepth      = -1;
  bool haveAtom      = false;
  bool haveFixup     = false;

  for (std::vector<const YAML::Entry *>::iterator it = entries.begin();
       it != entries.end(); ++it) {
    const YAML::Entry *entry = *it;

    if (entry->beginDocument) {
      if (file != nullptr) {
        if (haveAtom) {
          atomState.makeAtom(*file);
          haveAtom = false;
        }
        file->bindTargetReferences();
        result.push_back(file);
      }
      file = new YAMLFile();
      inAtoms = false;
      depthForAtoms = -1;
    }
    if (lastDepth > entry->depth) {
      // end of fixup sequence
      if (haveFixup) {
        atomState.addFixup(file);
        haveFixup = false;
      }
    }

    if (inAtoms && (depthForAtoms == -1)) {
      depthForAtoms = entry->depth;
    }
    if (inFixups && (depthForFixups == -1)) {
      depthForFixups = entry->depth;
    }
    if (strcmp(entry->key, "atoms") == 0) {
      inAtoms = true;
    }
    if (inAtoms) {
      if (depthForAtoms == entry->depth) {
        if (entry->beginSequence) {
          if (haveAtom) {
            atomState.makeAtom(*file);
            haveAtom = false;
          }
        }
        if (strcmp(entry->key, KeyValues::nameKeyword) == 0) {
          atomState.setName(entry->value);
          haveAtom = true;
        } 
        else if (strcmp(entry->key, KeyValues::refNameKeyword) == 0) {
          atomState.setRefName(entry->value);
          haveAtom = true;
        } 
        else if (strcmp(entry->key, KeyValues::definitionKeyword) == 0) {
          atomState._definition = KeyValues::definition(entry->value);
          haveAtom = true;
        } 
        else if (strcmp(entry->key, KeyValues::scopeKeyword) == 0) {
          atomState._scope = KeyValues::scope(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::contentTypeKeyword) == 0) {
          atomState._type = KeyValues::contentType(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::deadStripKindKeyword) == 0) {
          atomState._deadStrip = KeyValues::deadStripKind(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::sectionChoiceKeyword) == 0) {
          atomState._sectionChoice = KeyValues::sectionChoice(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::mergeKeyword) == 0) {
          atomState._merge = KeyValues::merge(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::interposableKeyword) == 0) {
          atomState._interpose = KeyValues::interposable(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::isThumbKeyword) == 0) {
          atomState._isThumb = KeyValues::isThumb(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::isAliasKeyword) == 0) {
          atomState._isAlias = KeyValues::isAlias(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::canBeNullKeyword) == 0) {
          atomState._canBeNull = KeyValues::canBeNull(entry->value);
          if ( atomState._definition == Atom::definitionSharedLibrary ) {
            if ( atomState._canBeNull == UndefinedAtom::canBeNullAtBuildtime )
              return make_error_code(yaml_reader_error::illegal_value);
          }
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::sectionNameKeyword) == 0) {
          atomState._sectionName = entry->value;
          haveAtom = true;
        } 
        else if (strcmp(entry->key, KeyValues::sizeKeyword) == 0) {
          StringRef val = entry->value;
          if (val.getAsInteger(0, atomState._size))
            return make_error_code(yaml_reader_error::illegal_value);
          haveAtom = true;
        } 
        else if (strcmp(entry->key, KeyValues::contentKeyword) == 0) {
          atomState._content = entry->valueSequenceBytes;
          haveAtom = true;
        } 
        else if (strcmp(entry->key, "align2") == 0) {
          atomState.setAlign2(entry->value);
          haveAtom = true;
        } 
        else if (strcmp(entry->key, KeyValues::fixupsKeyword) == 0) {
          inFixups = true;
        }
        else if (strcmp(entry->key, KeyValues::loadNameKeyword) == 0) {
          atomState._loadName = entry->value;
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::valueKeyword) == 0) {
          llvm::APInt Val;
          StringRef(entry->value).getAsInteger(0, Val);
          atomState._value = Val.getZExtValue();
          haveAtom = true;
        }
        else {
          return make_error_code(yaml_reader_error::unknown_keyword);
        }
      } 
      else if (depthForFixups == entry->depth) {
        if (entry->beginSequence) {
          if (haveFixup) {
            atomState.addFixup(file);
            haveFixup = false;
          }
        }
        if (strcmp(entry->key, KeyValues::fixupsKindKeyword) == 0) {
          atomState.setFixupKind(entry->value);
          haveFixup = true;
        } 
        else if (strcmp(entry->key, KeyValues::fixupsOffsetKeyword) == 0) {
          if (StringRef(entry->value).getAsInteger(0,
               atomState._ref._offsetInAtom))
            return make_error_code(yaml_reader_error::illegal_value);
          haveFixup = true;
        } 
        else if (strcmp(entry->key, KeyValues::fixupsTargetKeyword) == 0) {
          atomState.setFixupTarget(entry->value);
          haveFixup = true;
        }
        else if (strcmp(entry->key, KeyValues::fixupsAddendKeyword) == 0) {
          StringRef Addend(entry->value);
          if (Addend.getAsInteger(0, atomState._ref._addend))
            return make_error_code(yaml_reader_error::illegal_value);
          haveFixup = true;
        }
      }
    }
    lastDepth = entry->depth;
  }
  if (haveAtom) {
    atomState.makeAtom(*file);
  }
  if (file != nullptr) {
    file->bindTargetReferences();
    result.push_back(file);
  }
  return make_error_code(yaml_reader_error::success);
}


//
// Fill in vector<File*> from path to input text file.
//
error_code parseObjectTextFileOrSTDIN( StringRef path
                                     , Platform&  platform
                                     , std::vector<const File*>& result) {
  OwningPtr<llvm::MemoryBuffer> mb;
  if (error_code ec = llvm::MemoryBuffer::getFileOrSTDIN(path, mb))
    return ec;

  return parseObjectText(mb.get(), platform, result);
}

} // namespace yaml
} // namespace lld
