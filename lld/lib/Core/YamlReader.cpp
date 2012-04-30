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
#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Platform.h"
#include "lld/Core/Reference.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/UndefinedAtom.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/YAMLParser.h"

#include <cstring>
#include <set>
#include <type_traits>
#include <vector>

using namespace lld;

static bool getAs(const llvm::yaml::ScalarNode *SN, bool &Result) {
  SmallString<4> Storage;
  StringRef Value = SN->getValue(Storage);
  if (Value == "true")
    Result = true;
  else if (Value == "false")
    Result = false;
  else
    return false;
  return true;
}

template<class T>
typename std::enable_if<std::numeric_limits<T>::is_integer, bool>::type
getAs(const llvm::yaml::ScalarNode *SN, T &Result) {
  SmallString<4> Storage;
  StringRef Value = SN->getValue(Storage);
  if (Value.getAsInteger(0, Result))
    return false;
  return true;
}

namespace lld {
namespace yaml {

class YAMLReference : public Reference {
public:
  YAMLReference()
    : _target(nullptr)
    , _offsetInAtom(0)
    , _addend(0)
    , _kind(0)
  {}

  virtual uint64_t offsetInAtom() const {
    return _offsetInAtom;
  }

  virtual Kind kind() const {
    return _kind;
  }

  virtual void setKind(Kind k) {
    _kind = k;
  }

  virtual const Atom *target() const {
    return _target;
  }

  virtual Addend addend() const {
    return _addend;
  }

  virtual void setAddend(Addend a) {
    _addend = a;
  }

  virtual void setTarget(const Atom *newAtom) {
    _target = newAtom;
  }

  const Atom *_target;
  StringRef   _targetName;
  uint64_t    _offsetInAtom;
  Addend      _addend;
  Kind        _kind;
};

class YAMLDefinedAtom;

class YAMLFile : public ArchiveLibraryFile {
public:
  YAMLFile()
    : ArchiveLibraryFile("<anonymous>")
    , _lastRefIndex(0)
    , _kind(File::kindObject)
    , _inArchive(false) {
  }

  virtual File::Kind kind() const {
    return _kind;
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

  virtual void addAtom(const Atom&) {
    assert(0 && "cannot add atoms to YAML files");
  }

  virtual const File *find(StringRef name, bool dataSymbolOnly) const;

  void bindTargetReferences();
  void addDefinedAtom(YAMLDefinedAtom *atom, StringRef refName);
  void addUndefinedAtom(UndefinedAtom *atom);
  void addSharedLibraryAtom(SharedLibraryAtom *atom);
  void addAbsoluteAtom(AbsoluteAtom *atom);
  Atom *findAtom(StringRef name);
  void addMember(StringRef);
  void setName(StringRef);

  struct NameAtomPair {
                 NameAtomPair(StringRef n, Atom *a) : name(n), atom(a) {}
    StringRef name;
    Atom     *atom;
  };

  atom_collection_vector<DefinedAtom>       _definedAtoms;
  atom_collection_vector<UndefinedAtom>     _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom>      _absoluteAtoms;
  std::vector<YAMLReference>                _references;
  std::vector<NameAtomPair>                 _nameToAtomMapping;
  std::vector<StringRef>                    _memberNames;
  std::vector<std::unique_ptr<YAMLFile>>    _memberFiles;
  unsigned int                              _lastRefIndex;
  File::Kind                                _kind;
  bool                                      _inArchive;
};

class YAMLDefinedAtom : public DefinedAtom {
public:
  YAMLDefinedAtom( uint32_t ord
          , YAMLFile &file
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
          , StringRef name
          , StringRef sectionName
          , uint64_t size
          , std::vector<uint8_t> content)
    : _file(file)
    , _name(name)
    , _sectionName(sectionName)
    , _size(size)
    , _ord(ord)
    , _content(std::move(content))
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

  virtual const class File &file() const {
    return _file;
  }

  virtual StringRef name() const {
    return _name;
  }

  virtual uint64_t size() const {
    return _content.empty() ? _size : _content.size();
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
    return ArrayRef<uint8_t>(_content);
  }

  virtual uint64_t ordinal() const {
    return _ord;
  }

  DefinedAtom::reference_iterator begin() const {
    uintptr_t index = _refStartIndex;
    const void* it = reinterpret_cast<const void*>(index);
    return reference_iterator(*this, it);
  }

  DefinedAtom::reference_iterator end() const {
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
      StringRef targetName = _file._references[i]._targetName;
      Atom *targetAtom = _file.findAtom(targetName);
      _file._references[i]._target = targetAtom;
    }
  }

private:
  YAMLFile                   &_file;
  StringRef                   _name;
  StringRef                   _sectionName;
  unsigned long               _size;
  uint32_t                    _ord;
  std::vector<uint8_t>        _content;
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
  YAMLUndefinedAtom( YAMLFile &f
                   , int32_t ord
                   , StringRef name
                   , UndefinedAtom::CanBeNull cbn)
    : _file(f)
    , _name(name)
    , _ordinal(ord)
    , _canBeNull(cbn) {
  }

  virtual const class File &file() const {
    return _file;
  }

  virtual StringRef name() const {
    return _name;
  }

  virtual CanBeNull canBeNull() const {
    return _canBeNull;
  }

private:
  YAMLFile                &_file;
  StringRef                _name;
  uint32_t                 _ordinal;
  UndefinedAtom::CanBeNull _canBeNull;
};

class YAMLSharedLibraryAtom : public SharedLibraryAtom {
public:
  YAMLSharedLibraryAtom( YAMLFile &f
                       , int32_t ord
                       , StringRef name
                       , StringRef loadName
                       , bool cbn)
    : _file(f)
    , _name(name)
    , _ordinal(ord)
    , _loadName(loadName)
    , _canBeNull(cbn) {
  }

  virtual const class File &file() const {
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
  YAMLFile &_file;
  StringRef _name;
  uint32_t  _ordinal;
  StringRef _loadName;
  bool      _canBeNull;
};

class YAMLAbsoluteAtom : public AbsoluteAtom {
public:
  YAMLAbsoluteAtom(YAMLFile &f, int32_t ord, StringRef name, uint64_t v)
    : _file(f)
    , _name(name)
    , _ordinal(ord)
    , _value(v) {
  }

  virtual const class File &file() const {
    return _file;
  }

  virtual StringRef name() const {
    return _name;
  }

  virtual uint64_t value() const {
    return _value;
  }

private:
  YAMLFile &_file;
  StringRef _name;
  uint32_t  _ordinal;
  uint64_t  _value;
};

void YAMLFile::bindTargetReferences() {
  for (const DefinedAtom *defAtom : _definedAtoms) {
    const YAMLDefinedAtom *atom =
      reinterpret_cast<const YAMLDefinedAtom*>(defAtom);
    atom->bindTargetReferences();
  }
}

Atom *YAMLFile::findAtom(StringRef name) {
  for (auto &ci : _nameToAtomMapping) {
    if (ci.name == name)
      return ci.atom;
  }
  llvm::report_fatal_error("reference to atom that does not exist");
}

void YAMLFile::addDefinedAtom(YAMLDefinedAtom *atom, StringRef refName) {
  _definedAtoms._atoms.push_back(atom);
  _nameToAtomMapping.push_back(NameAtomPair(refName, atom));
}

void YAMLFile::addUndefinedAtom(UndefinedAtom *atom) {
  _undefinedAtoms._atoms.push_back(atom);
  _nameToAtomMapping.push_back(NameAtomPair(atom->name(), atom));
}

void YAMLFile::addSharedLibraryAtom(SharedLibraryAtom *atom) {
  _sharedLibraryAtoms._atoms.push_back(atom);
  _nameToAtomMapping.push_back(NameAtomPair(atom->name(), atom));
}

void YAMLFile::addAbsoluteAtom(AbsoluteAtom *atom) {
  _absoluteAtoms._atoms.push_back(atom);
  _nameToAtomMapping.push_back(NameAtomPair(atom->name(), atom));
}

void YAMLFile::addMember(StringRef name) {
  _memberNames.push_back(name);
}

void YAMLFile::setName(StringRef name) {
  _path = StringRef(name);
}

const File *YAMLFile::find(StringRef name, bool dataSymbolOnly) const {
  for (auto &file : _memberFiles) {
    for (const DefinedAtom *atom : file->defined() ) {
      if (name == atom->name())
        return file.get();
    }
  }
  return nullptr;
}

class YAMLAtomState {
public:
  YAMLAtomState(Platform &platform);

  void setName(StringRef n);
  void setRefName(StringRef n);
  void setAlign2(StringRef n);

  void setFixupKind(StringRef n);
  void setFixupTarget(StringRef n);
  void addFixup(YAMLFile *f);

  void makeAtom(YAMLFile&);

  Platform                   &_platform;
  StringRef                   _name;
  StringRef                   _refName;
  StringRef                   _sectionName;
  StringRef                   _loadName;
  unsigned long long          _size;
  uint64_t                    _value;
  uint32_t                    _ordinal;
  std::vector<uint8_t>        _content;
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


YAMLAtomState::YAMLAtomState(Platform &platform)
  : _platform(platform)
  , _size(0)
  , _value(0)
  , _ordinal(0)
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
  , _canBeNull(KeyValues::canBeNullDefault) {
}

void YAMLAtomState::makeAtom(YAMLFile &f) {
  if (_definition == Atom::definitionRegular) {
    YAMLDefinedAtom *a =
      new YAMLDefinedAtom( _ordinal
                         , f
                         , _scope
                         , _type
                         , _sectionChoice
                         , _interpose
                         , _merge
                         , _deadStrip
                         , _permissions
                         , _isThumb
                         , _isAlias
                         , _alignment
                         , _name
                         , _sectionName
                         , _size
                         , _content
                         );
    f.addDefinedAtom(a, !_refName.empty() ? _refName : _name);
    ++_ordinal;
  } else if (_definition == Atom::definitionUndefined) {
    UndefinedAtom *a = new YAMLUndefinedAtom(f, _ordinal, _name, _canBeNull);
    f.addUndefinedAtom(a);
    ++_ordinal;
  } else if (_definition == Atom::definitionSharedLibrary) {
    bool nullable = (_canBeNull == UndefinedAtom::canBeNullAtRuntime);
    SharedLibraryAtom *a = new YAMLSharedLibraryAtom(f, _ordinal, _name,
                                                      _loadName, nullable);
    f.addSharedLibraryAtom(a);
    ++_ordinal;
  } else if (_definition == Atom::definitionAbsolute) {
    AbsoluteAtom *a = new YAMLAbsoluteAtom(f, _ordinal, _name, _value);
    f.addAbsoluteAtom(a);
    ++_ordinal;
  }

  // reset state for next atom
  _name               = StringRef();
  _refName            = StringRef();
  _sectionName        = StringRef();
  _loadName           = StringRef();
  _size               = 0;
  _value              = 0;
  _ordinal            = 0;
  _content.clear();
  _alignment.powerOf2 = 0;
  _alignment.modulus  = 0;
  _definition         = KeyValues::definitionDefault;
  _scope              = KeyValues::scopeDefault;
  _type               = KeyValues::contentTypeDefault;
  _sectionChoice      = KeyValues::sectionChoiceDefault;
  _interpose          = KeyValues::interposableDefault;
  _merge              = KeyValues::mergeDefault;
  _deadStrip          = KeyValues::deadStripKindDefault;
  _permissions        = KeyValues::permissionsDefault;
  _isThumb            = KeyValues::isThumbDefault;
  _isAlias            = KeyValues::isAliasDefault;
  _canBeNull          = KeyValues::canBeNullDefault;
  _ref._target        = nullptr;
  _ref._targetName    = StringRef();
  _ref._addend        = 0;
  _ref._offsetInAtom  = 0;
  _ref._kind          = 0;
}

void YAMLAtomState::setName(StringRef n) {
  _name = n;
}

void YAMLAtomState::setRefName(StringRef n) {
  _refName = n;
}

void YAMLAtomState::setAlign2(StringRef s) {
  if (StringRef(s).getAsInteger(10, _alignment.powerOf2))
    _alignment.powerOf2 = 1;
}

void YAMLAtomState::setFixupKind(StringRef s) {
  _ref._kind = _platform.kindFromString(StringRef(s));
}

void YAMLAtomState::setFixupTarget(StringRef s) {
  _ref._targetName = s;
}

void YAMLAtomState::addFixup(YAMLFile *f) {
  f->_references.push_back(_ref);
  // clear for next ref
  _ref._target       = nullptr;
  _ref._targetName   = StringRef();
  _ref._addend       = 0;
  _ref._offsetInAtom = 0;
  _ref._kind         = 0;
}

llvm::error_code parseFixup( llvm::yaml::MappingNode *mn
                           , llvm::yaml::Stream &s
                           , Platform &p
                           , YAMLFile &f
                           , YAMLAtomState &yas) {
  using namespace llvm::yaml;
  llvm::SmallString<32> storage;

  for (auto &keyval : *mn) {
    ScalarNode *key = llvm::dyn_cast<ScalarNode>(keyval.getKey());
    if (!key) {
      s.printError(key, "Expected a scalar value");
      return make_error_code(yaml_reader_error::illegal_value);
    }
    ScalarNode *value = llvm::dyn_cast<ScalarNode>(keyval.getValue());
    if (!value) {
      s.printError(value, "Expected a scalar value");
      return make_error_code(yaml_reader_error::illegal_value);
    }
    llvm::StringRef keyValue = key->getValue(storage);
    if (keyValue == KeyValues::fixupsOffsetKeyword) {      
      if (!getAs(value, yas._ref._offsetInAtom)) {
        s.printError(value, "Invalid value for offset");
        return make_error_code(yaml_reader_error::illegal_value);
      }
    } else if (keyValue == KeyValues::fixupsKindKeyword) {
      yas._ref._kind = p.kindFromString(value->getValue(storage));
    } else if (keyValue == KeyValues::fixupsTargetKeyword) {
      // FIXME: string lifetime.
      yas._ref._targetName = value->getValue(storage);
    } else if (keyValue == KeyValues::fixupsAddendKeyword) {
      if (!getAs(value, yas._ref._addend)) {
        s.printError(value, "Invalid value for addend");
        return make_error_code(yaml_reader_error::illegal_value);
      }
    } else {
      s.printError(key, "Unrecognized key");
      return make_error_code(yaml_reader_error::unknown_keyword);
    }
  }
  yas.addFixup(&f);
  return make_error_code(yaml_reader_error::success);
}

llvm::error_code parseAtom( llvm::yaml::MappingNode *mn
                          , llvm::yaml::Stream &s
                          , Platform &p
                          , YAMLFile &f) {
  using namespace llvm::yaml;
  YAMLAtomState yas(p);
  llvm::SmallString<32> storage;

  for (MappingNode::iterator i = mn->begin(), e = mn->end(); i != e; ++i) {
    ScalarNode *Key = llvm::dyn_cast<ScalarNode>(i->getKey());
    if (!Key)
      return make_error_code(yaml_reader_error::illegal_value);
    llvm::StringRef KeyValue = Key->getValue(storage);
    if (KeyValue == KeyValues::contentKeyword) {
      ScalarNode *scalarValue = llvm::dyn_cast<ScalarNode>(i->getValue());
      if (scalarValue) {
        yas._type = KeyValues::contentType(scalarValue->getValue(storage));
      } else {
        SequenceNode *Value = llvm::dyn_cast<SequenceNode>(i->getValue());
        if (!Value) {
          s.printError(i->getValue(), "Expected a sequence");
          return make_error_code(yaml_reader_error::illegal_value);
        }
        for (SequenceNode::iterator ci = Value->begin(), ce = Value->end();
                                    ci != ce; ++ci) {
          ScalarNode *Entry = llvm::dyn_cast<ScalarNode>(&*ci);
          if (!Entry) {
            s.printError(i->getValue(), "Expected a scalar value");
            return make_error_code(yaml_reader_error::illegal_value);
          }
          unsigned int ContentByte;
          if (Entry->getValue(storage).getAsInteger(16, ContentByte)) {
            s.printError(i->getValue(), "Invalid content byte");
            return make_error_code(yaml_reader_error::illegal_value);
          }
          if (ContentByte > 0xFF) {
            s.printError(i->getValue(), "Byte out of range (> 0xFF)");
            return make_error_code(yaml_reader_error::illegal_value);
          }
          yas._content.push_back(ContentByte & 0xFF);
        }
      }
    } else if (KeyValue == KeyValues::fixupsKeyword) {
      SequenceNode *Value = llvm::dyn_cast<SequenceNode>(i->getValue());
      if (!Value) {
        s.printError(i->getValue(), "Expected a sequence");
        return make_error_code(yaml_reader_error::illegal_value);
      }
      for (auto &i : *Value) {
        MappingNode *Fixup = llvm::dyn_cast<MappingNode>(&i);
        if (!Fixup) {
          s.printError(&i, "Expected a map");
          return make_error_code(yaml_reader_error::illegal_value);
        }
        if (error_code ec = parseFixup(Fixup, s, p, f, yas))
          return ec;
      }
    } else {
      // The rest of theses all require value to be a scalar.
      ScalarNode *Value = llvm::dyn_cast<ScalarNode>(i->getValue());
      if (!Value) {
        s.printError(i->getValue(), "Expected a scalar value");
        return make_error_code(yaml_reader_error::illegal_value);
      }
      if (KeyValue == KeyValues::nameKeyword) {
        // FIXME: String lifetime.
        yas.setName(Value->getValue(storage));
      } else if (KeyValue == KeyValues::refNameKeyword) {
        // FIXME: String lifetime.
        yas.setRefName(Value->getValue(storage));
      } else if (KeyValue == KeyValues::valueKeyword) {
        if (!getAs(Value, yas._value)) {
          s.printError(Value, "Invalid value for value");
          return make_error_code(yaml_reader_error::illegal_value);
        }
      } else if (KeyValue == KeyValues::loadNameKeyword)
        // FIXME: String lifetime.
        yas._loadName = Value->getValue(storage);
      else if (KeyValue == KeyValues::definitionKeyword)
        yas._definition = KeyValues::definition(Value->getValue(storage));
      else if (KeyValue == KeyValues::scopeKeyword)
        yas._scope = KeyValues::scope(Value->getValue(storage));
      else if (KeyValue == KeyValues::contentTypeKeyword)
        yas._type = KeyValues::contentType(Value->getValue(storage));
      else if (KeyValue == KeyValues::deadStripKindKeyword)
        yas._deadStrip = KeyValues::deadStripKind(Value->getValue(storage));
      else if (KeyValue == KeyValues::sectionChoiceKeyword)
        yas._sectionChoice = KeyValues::sectionChoice(Value->getValue(storage));
      else if (KeyValue == KeyValues::mergeKeyword)
        yas._merge = KeyValues::merge(Value->getValue(storage));
      else if (KeyValue == KeyValues::interposableKeyword)
        yas._interpose = KeyValues::interposable(Value->getValue(storage));
      else if (KeyValue == KeyValues::isThumbKeyword) {
        if (!getAs(Value, yas._isThumb)) {
          s.printError(Value, "Invalid value for isThumb");
          return make_error_code(yaml_reader_error::illegal_value);
        }
      } else if (KeyValue == KeyValues::isAliasKeyword) {
        if (!getAs(Value, yas._isAlias)) {
          s.printError(Value, "Invalid value for isAlias");
          return make_error_code(yaml_reader_error::illegal_value);
        }
      } else if (KeyValue == KeyValues::canBeNullKeyword) {
        yas._canBeNull = KeyValues::canBeNull(Value->getValue(storage));
        if (yas._definition == Atom::definitionSharedLibrary
            && yas._canBeNull == UndefinedAtom::canBeNullAtBuildtime) {
          s.printError(Value, "Invalid value for can be null");
          return make_error_code(yaml_reader_error::illegal_value);
        }
      } else if (KeyValue == KeyValues::sectionNameKeyword)
        // FIXME: String lifetime.
        yas._sectionName = Value->getValue(storage);
      else if (KeyValue == KeyValues::sizeKeyword) {
        if (!getAs(Value, yas._size)) {
          s.printError(Value, "Invalid value for size");
          return make_error_code(yaml_reader_error::illegal_value);
        }
      } else if (KeyValue == "align2")
        yas.setAlign2(Value->getValue(storage));
      else {
        s.printError(Key, "Unrecognized key");
        return make_error_code(yaml_reader_error::unknown_keyword);
      }
    }
  }
  yas.makeAtom(f);
  return make_error_code(yaml_reader_error::success);
}

llvm::error_code parseAtoms( llvm::yaml::SequenceNode *atoms
                           , llvm::yaml::Stream &s
                           , Platform &p
                           , YAMLFile &f) {
  using namespace llvm::yaml;

  for (auto &atom : *atoms) {
    if (MappingNode *a = llvm::dyn_cast<MappingNode>(&atom)) {
      if (llvm::error_code ec = parseAtom(a, s, p, f))
        return ec;
    } else {
      s.printError(&atom, "Expected map");
      return make_error_code(yaml_reader_error::illegal_value);
    }
  }
  return make_error_code(yaml_reader_error::success);
}

llvm::error_code parseArchive( llvm::yaml::SequenceNode *archive
                             , llvm::yaml::Stream &s
                             , Platform &p
                             , YAMLFile &f) {
  using namespace llvm::yaml;
  llvm::SmallString<32> storage;

  for (auto &member : *archive) {
    std::unique_ptr<YAMLFile> mf(new YAMLFile);
    MappingNode *mem = llvm::dyn_cast<MappingNode>(&member);
    if (!mem) {
      s.printError(&member, "Expected map");
      return make_error_code(yaml_reader_error::illegal_value);
    }
    for (auto &keyVal : *mem) {
      ScalarNode *key = llvm::dyn_cast<ScalarNode>(keyVal.getKey());
      if (!key) {
        s.printError(keyVal.getKey(), "Expected scalar value");
        return make_error_code(yaml_reader_error::illegal_value);
      }
      if (key->getValue(storage) == "name") {
        ScalarNode *value = llvm::dyn_cast<ScalarNode>(keyVal.getValue());
        if (!value) {
          s.printError(keyVal.getValue(), "Expected scalar value");
          return make_error_code(yaml_reader_error::illegal_value);
        }
        // FIXME: String lifetime.
        mf->setName(value->getValue(storage));
      } else if (key->getValue(storage) == "atoms") {
        SequenceNode *atoms = llvm::dyn_cast<SequenceNode>(keyVal.getValue());
        if (!atoms) {
          s.printError(keyVal.getValue(), "Expected sequence");
          return make_error_code(yaml_reader_error::illegal_value);
        }
        if (error_code ec = parseAtoms(atoms, s, p, *mf))
          return ec;
      } else {
        s.printError(key, "Unrecognized key");
        return make_error_code(yaml_reader_error::unknown_keyword);
      }
    }
    f._memberFiles.push_back(std::move(mf));
  }
  return make_error_code(yaml_reader_error::success);
}

/// parseObjectText - Parse the specified YAML formatted MemoryBuffer
/// into lld::File object(s) and append each to the specified vector<File*>.
error_code parseObjectText( llvm::MemoryBuffer *mb
                          , Platform& platform
                          , std::vector<std::unique_ptr<const File>> &result) {
  using namespace llvm::yaml;
  llvm::SourceMgr sm;
  Stream stream(mb->getBuffer(), sm);

  llvm::SmallString<32> storage;
  for (Document &d : stream) {
    std::unique_ptr<YAMLFile> CurFile(new YAMLFile);
    if (llvm::isa<NullNode>(d.getRoot()))
      continue; // Empty files are allowed.
    MappingNode *n = llvm::dyn_cast<MappingNode>(d.getRoot());
    if (!n) {
      stream.printError(d.getRoot(), "Expected map");
      return make_error_code(yaml_reader_error::illegal_value);
    }
    for (MappingNode::iterator mi = n->begin(), me = n->end(); mi != me; ++mi) {
      ScalarNode *key = llvm::dyn_cast<ScalarNode>(mi->getKey());
      if (!key) {
        stream.printError(mi->getValue(), "Expected scalar value");
        return make_error_code(yaml_reader_error::illegal_value);
      }
      if (key->getValue(storage) == "atoms") {
        SequenceNode *Atoms = llvm::dyn_cast<SequenceNode>(mi->getValue());
        if (!Atoms) {
          stream.printError(mi->getValue(), "Expected sequence");
          return make_error_code(yaml_reader_error::illegal_value);
        }
        if (error_code ec = parseAtoms(Atoms, stream, platform, *CurFile))
          return ec;
      } else if (key->getValue(storage) == "archive") {
        CurFile->_kind = YAMLFile::kindArchiveLibrary;
        SequenceNode *members = llvm::dyn_cast<SequenceNode>(mi->getValue());
        if (!members) {
          stream.printError(mi->getValue(), "Expected sequence");
          return make_error_code(yaml_reader_error::illegal_value);
        }
        if (error_code ec = parseArchive( members
                                        , stream
                                        , platform
                                        , *CurFile))
          return ec;
      } else {
        stream.printError(key, "Unrecognized key");
        return make_error_code(yaml_reader_error::unknown_keyword);
      }
    }
    if (stream.failed())
      return make_error_code(yaml_reader_error::illegal_value);
    CurFile->bindTargetReferences();
    result.emplace_back(CurFile.release());
  }

  return make_error_code(yaml_reader_error::success);
}

//
// Fill in vector<File*> from path to input text file.
//
error_code parseObjectTextFileOrSTDIN( StringRef path
                                     , Platform&  platform
                                     , std::vector<
                                         std::unique_ptr<const File>>& result) {
  OwningPtr<llvm::MemoryBuffer> mb;
  if (error_code ec = llvm::MemoryBuffer::getFileOrSTDIN(path, mb))
    return ec;

  return parseObjectText(mb.take(), platform, result);
}

} // namespace yaml
} // namespace lld
