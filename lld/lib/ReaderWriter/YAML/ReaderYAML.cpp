//===- lib/ReaderWriter/YAML/ReaderYAML.cpp - Reads YAML object files -----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/ReaderYAML.h"

#include "lld/Core/AbsoluteAtom.h"
#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/Atom.h"
#include "lld/Core/Error.h"
#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/UndefinedAtom.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/YAMLParser.h"

#include <cstring>
#include <vector>

#include "YamlKeyValues.h"


namespace lld {
namespace yaml {


///
/// Concrete instance of lld::Reference created parsing YAML object files
///
class YAMLReference : public Reference {
public:
  YAMLReference()
    : _target(nullptr)
    , _targetNameNode(nullptr)
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

  typedef llvm::yaml::ScalarNode ScalarNode;
  
  const Atom *_target;
  ScalarNode * _targetNameNode;
  uint64_t    _offsetInAtom;
  Addend      _addend;
  Kind        _kind;
};


///
/// Concrete instance of lld::File created parsing YAML object files.
///
class YAMLFile : public ArchiveLibraryFile {
public:
  YAMLFile()
    : ArchiveLibraryFile("<anonymous>")
    , _lastRefIndex(0)
    , _kind(File::kindObject) {
  }

  ~YAMLFile();
  
  // Depending on the YAML description, this file can be either an
  // lld::ArchiveLibraryFile or lld::File.
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

  // Standard way that archives are searched.
  virtual const File *find(StringRef name, bool dataSymbolOnly) const;

  error_code bindTargetReferences(llvm::yaml::Stream &stream);
  
  void addDefinedAtom(class YAMLDefinedAtom *atom, StringRef refName);
  void addUndefinedAtom(UndefinedAtom *atom);
  void addSharedLibraryAtom(SharedLibraryAtom *atom);
  void addAbsoluteAtom(AbsoluteAtom *atom);
  Atom *findAtom(StringRef name);
  void addMember(StringRef);
  void setName(StringRef);

  StringRef copyString(StringRef);
  
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
  std::vector<std::unique_ptr<YAMLFile>>    _memberFiles;
  std::vector<char*>                        _stringCopies;
  unsigned int                              _lastRefIndex;
  File::Kind                                _kind;
};



///
/// Concrete instance of lld::DefinedAtom created parsing YAML object files.
///
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
          , std::vector<uint8_t>& content)
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

  // Convert each target name to a pointer to an atom object
  error_code bindTargetReferences(llvm::yaml::Stream &stream) const {
    for (unsigned int i=_refStartIndex; i < _refEndIndex; ++i) {
      llvm::SmallString<32> storage;
      llvm::yaml::ScalarNode *node = _file._references[i]._targetNameNode;
      StringRef name = node->getValue(storage);
      Atom *targetAtom = _file.findAtom(name);
      if ( targetAtom ) {
        _file._references[i]._target = targetAtom;
      }
      else {
        stream.printError(node, "Fixup has target '" + name 
                            + "' which does not exist");
        return make_error_code(yaml_reader_error::illegal_value);
      }
    }
    return make_error_code(yaml_reader_error::success);
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



///
/// Concrete instance of lld::UndefinedAtom created parsing YAML object files.
///
class YAMLUndefinedAtom : public UndefinedAtom {
public:
  YAMLUndefinedAtom( YAMLFile &f
                   , int32_t
                   , StringRef name
                   , UndefinedAtom::CanBeNull cbn)
    : _file(f)
    , _name(name)
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
  UndefinedAtom::CanBeNull _canBeNull;
};



///
/// Concrete instance of lld::SharedLibraryAtom created parsing YAML files.
///
class YAMLSharedLibraryAtom : public SharedLibraryAtom {
public:
  YAMLSharedLibraryAtom( YAMLFile &f
                       , int32_t
                       , StringRef name
                       , StringRef loadName
                       , bool cbn)
    : _file(f)
    , _name(name)
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
  StringRef _loadName;
  bool      _canBeNull;
};



///
/// Concrete instance of lld::AbsoluteAtom created parsing YAML object files.
///
class YAMLAbsoluteAtom : public AbsoluteAtom {
public:
  YAMLAbsoluteAtom(YAMLFile &f, int32_t, StringRef name, uint64_t v, Atom::Scope scope)
    : _file(f)
    , _name(name)
    , _value(v) 
    , _scope(scope){
  }

  virtual const class File &file() const {
    return _file;
  }

  virtual Scope scope() const {
    return _scope;
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
  uint64_t  _value;
  Atom::Scope _scope;
};




//===----------------------------------------------------------------------===//
//  YAMLFile methods
//===----------------------------------------------------------------------===//

YAMLFile::~YAMLFile() {
  for (char *s : _stringCopies) {
    delete [] s;
  }
}


error_code YAMLFile::bindTargetReferences(llvm::yaml::Stream &stream) {
  error_code ec;
  for (const DefinedAtom *defAtom : _definedAtoms) {
    const YAMLDefinedAtom *atom =
      reinterpret_cast<const YAMLDefinedAtom*>(defAtom);
    ec = atom->bindTargetReferences(stream);
    if ( ec )
      return ec;
  }
  return ec;
}

Atom *YAMLFile::findAtom(StringRef name) {
  for (auto &ci : _nameToAtomMapping) {
    if (ci.name == name)
      return ci.atom;
  }
  return nullptr;
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

void YAMLFile::setName(StringRef name) {
  _path = StringRef(name);
}


// Allocate a new copy of this string and keep track of allocations
// in _stringCopies, so they can be freed when YAMLFile is destroyed.
StringRef YAMLFile::copyString(StringRef str) {
  char* s = new char[str.size()];
  memcpy(s, str.data(), str.size());
  _stringCopies.push_back(s);
  return StringRef(s, str.size());
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



///
/// The state machine that drives the YAMLParser stream and instantiates
/// Files and Atoms.  This class also buffers all the attribures for the 
/// current atom and current fixup.  Once all attributes are accumulated,  
/// a new atom or fixup instance is instantiated.
///
class YAMLState {
public:
  YAMLState(const ReaderOptionsYAML &opts, llvm::yaml::Stream *s, YAMLFile *f);

  void        parse(llvm::yaml::Node *node, StringRef keyword, 
                                        llvm::yaml::Node *keywordNode=nullptr);
  error_code  error() { return _error; }
  
private:
  typedef llvm::yaml::Node Node;
  typedef llvm::yaml::ScalarNode ScalarNode;
  typedef llvm::yaml::SequenceNode SequenceNode;
  typedef llvm::yaml::MappingNode MappingNode;
  typedef llvm::yaml::Stream Stream;

  void resetState();
  void setAlign2(StringRef n);

  void makeReference();
  void makeAtom(Node *node);
  void makeDefinedAtom(Node *node);
  void makeUndefinedAtom(Node *node);
  void makeSharedLibraryAtom(Node *node);
  void makeAbsoluteAtom(Node *node);
 
  void parseMemberName(ScalarNode *node);
  void parseAtomName(ScalarNode *node);
  void parseAtomRefName(ScalarNode *node);
  void parseAtomType(ScalarNode *node);
  void parseAtomScope(ScalarNode *node);
  void parseAtomDefinition(ScalarNode *node);
  void parseAtomDeadStrip(ScalarNode *node);
  void parseAtomSectionChoice(ScalarNode *node);
  void parseAtomInterposable(ScalarNode *node);
  void parseAtomMerge(ScalarNode *node);
  void parseAtomIsThumb(ScalarNode *node);
  void parseAtomIsAlias(ScalarNode *node);
  void parseAtomSectionName(ScalarNode *node);
  void parseAtomSize(ScalarNode *node);
  void parseAtomPermissions(ScalarNode *node);
  void parseAtomCanBeNull(ScalarNode *node);
  void parseFixUpOffset(ScalarNode *node);
  void parseFixUpKind(ScalarNode *node);
  void parseFixUpTarget(ScalarNode *node);
  void parseFixUpAddend(ScalarNode *node);
  void parseAtomContentByte(ScalarNode *node);
  void parseAtomLoadName(ScalarNode *node);
  void parseAtomValue(ScalarNode *node);

  StringRef extractString(ScalarNode *node);

  typedef void (YAMLState:: *ParseScalar)(ScalarNode *node);
  typedef void (YAMLState:: *ParseSeq)(SequenceNode *node);
  typedef void (YAMLState:: *ParseMap)(MappingNode *node);

  enum State { inError, inTop, inDoc, inArch, inMemb, 
              inAtoms, inAtom, inFixUps, inFixUp, inBytes };
  struct Transistion {
    State         state;
    const char*   keyword;
    State         newState;
    ParseScalar   customAction;
  };

  static const char* stateName(State);

  void moveToState(State s);
  void returnToState(State s, Node *node);
  
  static const Transistion _s_transistions[];

  const ReaderOptionsYAML    &_options;
  error_code                  _error;
  llvm::yaml::Stream         *_stream;
  YAMLFile                   *_file;
  YAMLFile                   *_archiveFile;
  State                       _state;
  StringRef                   _name;
  StringRef                   _refName;
  StringRef                   _sectionName;
  StringRef                   _loadName;
  StringRef                   _memberName;
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
  bool                        _hasDefinedAtomAttributes;
  bool                        _hasUndefinedAtomAttributes;
  bool                        _hasSharedLibraryAtomAttributes;
  bool                        _hasAbsoluteAtomAttributes;
};


//
// This transition table is the heart of the state machine.  
// The table is read left-to-right columns A,B,C,D as:  
//    If the state is A and key B is seen switch to state C then
//    if D is not nullptr call that method with the key's value,
//    if D is nullptr, recursively parse in the new state.
//
const YAMLState::Transistion YAMLState::_s_transistions[] = {
  { inTop,   "<root>",         inDoc,   nullptr                            },
  { inDoc,   "archive",        inArch,  nullptr                            },
  { inArch,  "<any-seq-item>", inMemb,  nullptr                            },
  { inMemb,  "atoms",          inAtoms, nullptr                            },
  { inMemb,  "name",           inMemb,  &YAMLState::parseMemberName        },
  { inDoc,   "atoms",          inAtoms, nullptr                            },
  { inAtoms, "<any-seq-item>", inAtom,  nullptr                            },
  { inAtom,  "name",           inAtom,  &YAMLState::parseAtomName          },
  { inAtom,  "ref-name",       inAtom,  &YAMLState::parseAtomRefName       },
  { inAtom,  "type",           inAtom,  &YAMLState::parseAtomType          },
  { inAtom,  "scope",          inAtom,  &YAMLState::parseAtomScope         },
  { inAtom,  "definition",     inAtom,  &YAMLState::parseAtomDefinition    },
  { inAtom,  "dead-strip",     inAtom,  &YAMLState::parseAtomDeadStrip     },
  { inAtom,  "section-choice", inAtom,  &YAMLState::parseAtomSectionChoice },
  { inAtom,  "interposable",   inAtom,  &YAMLState::parseAtomInterposable  },
  { inAtom,  "merge",          inAtom,  &YAMLState::parseAtomMerge         },
  { inAtom,  "is-thumb",       inAtom,  &YAMLState::parseAtomIsThumb       },
  { inAtom,  "is-alias",       inAtom,  &YAMLState::parseAtomIsAlias       },
  { inAtom,  "section-name",   inAtom,  &YAMLState::parseAtomSectionName   },
  { inAtom,  "size",           inAtom,  &YAMLState::parseAtomSize          },
  { inAtom,  "permissions",    inAtom,  &YAMLState::parseAtomPermissions   },
  { inAtom,  "can-be-null",    inAtom,  &YAMLState::parseAtomCanBeNull     },
  { inAtom,  "content",        inBytes, nullptr                            },
  { inAtom,  "fixups",         inFixUps,nullptr                            },
  { inBytes, "<any-seq-item>", inBytes, &YAMLState::parseAtomContentByte   },
  { inFixUps,"<any-seq-item>", inFixUp, nullptr                            },
  { inFixUp, "offset",         inFixUp, &YAMLState::parseFixUpOffset       },
  { inFixUp, "kind",           inFixUp, &YAMLState::parseFixUpKind         },
  { inFixUp, "target",         inFixUp, &YAMLState::parseFixUpTarget       },
  { inFixUp, "addend",         inFixUp, &YAMLState::parseFixUpAddend       },
  { inAtom,  "load-name",      inAtom,  &YAMLState::parseAtomLoadName      },
  { inAtom,  "value",          inAtom,  &YAMLState::parseAtomValue         },
  { inError,  nullptr,         inAtom,  nullptr                            },
};



YAMLState::YAMLState(const ReaderOptionsYAML &opts, Stream *stream, 
                                                                YAMLFile *file)
  : _options(opts)
  , _error(make_error_code(yaml_reader_error::success))
  , _stream(stream)
  , _file(file)
  , _archiveFile(nullptr)
  , _state(inTop) 
  , _alignment(0, 0) {
  this->resetState();
}

void YAMLState::makeAtom(Node *node) {
  switch (_definition ) {
    case Atom::definitionRegular:
      this->makeDefinedAtom(node);
      break;
    case Atom::definitionUndefined:
      this->makeUndefinedAtom(node);
      break;
    case Atom::definitionSharedLibrary:
      this->makeSharedLibraryAtom(node);
      break;
    case Atom::definitionAbsolute:
      this->makeAbsoluteAtom(node);
      break;
  }
  ++_ordinal;
  
  // reset state for next atom
  this->resetState();
}

void YAMLState::makeDefinedAtom(Node *node) {
  if ( _hasAbsoluteAtomAttributes ) {
    _stream->printError(node, "Defined atom '" + _name 
                          + "' has attributes only allowed on absolute atoms");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  if ( _hasSharedLibraryAtomAttributes ) {
    _stream->printError(node, "Defined atom '" + _name 
                    + "' has attributes only allowed on shared library atoms");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }

  YAMLDefinedAtom *a = new YAMLDefinedAtom(_ordinal, *_file, _scope, _type
                         , _sectionChoice, _interpose, _merge, _deadStrip
                         , _permissions, _isThumb, _isAlias, _alignment
                         , _name, _sectionName, _size, _content);
    _file->addDefinedAtom(a, !_refName.empty() ? _refName : _name);
}

void YAMLState::makeUndefinedAtom(Node *node) {
  if ( _hasDefinedAtomAttributes ) {
    _stream->printError(node, "Undefined atom '" + _name 
                          + "' has attributes only allowed on defined atoms");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  if ( _hasAbsoluteAtomAttributes ) {
    _stream->printError(node, "Defined atom '" + _name 
                          + "' has attributes only allowed on absolute atoms");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  UndefinedAtom *a = new YAMLUndefinedAtom(*_file, _ordinal, _name, _canBeNull);
  _file->addUndefinedAtom(a);
}

void YAMLState::makeSharedLibraryAtom(Node *node) {
  if ( _hasDefinedAtomAttributes ) {
    _stream->printError(node, "SharedLibrary atom '" + _name 
                          + "' has attributes only allowed on defined atoms");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  if ( _hasAbsoluteAtomAttributes ) {
    _stream->printError(node, "Defined atom '" + _name 
                          + "' has attributes only allowed on absolute atoms");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  bool nullable = (_canBeNull == UndefinedAtom::canBeNullAtRuntime);
  SharedLibraryAtom *a = new YAMLSharedLibraryAtom(*_file, _ordinal, _name,
                                                    _loadName, nullable);
  _file->addSharedLibraryAtom(a);
}

void YAMLState::makeAbsoluteAtom(Node *node) {
  if ( _hasDefinedAtomAttributes ) {
    _stream->printError(node, "Absolute atom '" + _name 
                          + "' has attributes only allowed on defined atoms");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  if ( _hasSharedLibraryAtomAttributes ) {
    _stream->printError(node, "Absolute atom '" + _name 
                    + "' has attributes only allowed on shared library atoms");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  AbsoluteAtom *a = new YAMLAbsoluteAtom(*_file, _ordinal, _name, _value, 
                                         _scope);
  _file->addAbsoluteAtom(a);
}



void YAMLState::resetState() {
  _name               = StringRef();
  _refName            = StringRef();
  _sectionName        = StringRef();
  _loadName           = StringRef();
  _memberName         = StringRef();
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
  _ref._targetNameNode= nullptr;
  _ref._addend        = 0;
  _ref._offsetInAtom  = 0;
  _ref._kind          = 0;
  
  _hasDefinedAtomAttributes = false;
  _hasUndefinedAtomAttributes = false;
  _hasSharedLibraryAtomAttributes = false;
  _hasAbsoluteAtomAttributes = false;
}


void YAMLState::makeReference() {
  _file->_references.push_back(_ref);
  // clear for next ref
  _ref._target        = nullptr;
  _ref._targetNameNode= nullptr;
  _ref._addend        = 0;
  _ref._offsetInAtom  = 0;
  _ref._kind          = 0;
}



void YAMLState::setAlign2(StringRef s) {
  if (StringRef(s).getAsInteger(10, _alignment.powerOf2))
    _alignment.powerOf2 = 1;
}


// For debug logging
const char* YAMLState::stateName(State s) {
  switch ( s ) {
    case inError:
      return "inError";
    case inTop:
      return "inTop";
    case inDoc:
      return "inDoc";
    case inArch:
      return "inArch";
    case inMemb:
      return "inMemb";
    case inAtoms:
      return "inAtoms";
    case inAtom:
      return "inAtom";
    case inFixUps:
      return "inFixUps";
    case inFixUp:
      return "inFixUp";
    case inBytes:
      return "inBytes";
  }
  return "unknown case";
}

// Called by parse() when recursing and switching to a new state.
void YAMLState::moveToState(State newState) {
  if ( newState == _state )
    return;
  DEBUG_WITH_TYPE("objtxt", llvm::dbgs() << "moveToState(" << stateName(newState) 
                     << "), _state=" << stateName(_state) << "\n");
  
  if ( newState == inArch ) {
    // Seen "archive:", repurpose existing YAMLFile to be archive file
    _file->_kind = File::kindArchiveLibrary;
    _archiveFile = _file;
    _file = nullptr;
  }
  
  if ( newState == inMemb ) {
    assert(_state == inArch);
    // Make new YAMLFile for this member
    std::unique_ptr<YAMLFile> memberFile(new YAMLFile);
    _file = memberFile.get();
    assert(_archiveFile != nullptr);
    _archiveFile->_memberFiles.emplace_back(memberFile.release());
  }

  _state = newState;
}

// Called by parse() when returning from recursion and restoring the old state.
void YAMLState::returnToState(State prevState, Node *node) {
  if ( prevState == _state )
    return;
  DEBUG_WITH_TYPE("objtxt", llvm::dbgs() 
                     << "returnToState(" << stateName(prevState) 
                     << "), _state=" << stateName(_state) << "\n");
  // If done with an atom, instantiate an object for it.
  if ( (_state == inAtom) && (prevState == inAtoms) )
    this->makeAtom(node);
  // If done wit a fixup, instantiate an object for it.
  if ( (_state == inFixUp) && (prevState == inFixUps) )
    this->makeReference();
  _state = prevState;
}

// If a string in the yaml document is quoted in a way that there is no
// contiguous range of bytes that a StringRef can point to, then we make
// a copy of the string and have the StringRef point to that.
StringRef YAMLState::extractString(ScalarNode *node) {
  llvm::SmallString<32> storage;
  StringRef str = node->getValue(storage);
  //if ( str.data() == storage.begin() ) {
    str = _file->copyString(str);
  //}
  return str;
}


void YAMLState::parseMemberName(ScalarNode *node) {
   _memberName = extractString(node);
}

void YAMLState::parseAtomName(ScalarNode *node) {
   _name = extractString(node);
}

void YAMLState::parseAtomRefName(ScalarNode *node) {
   _refName = extractString(node);
}

void YAMLState::parseAtomScope(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::scope(node->getValue(storage), _scope) ) {
    _stream->printError(node, "Invalid value for 'scope:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
}

void YAMLState::parseAtomDefinition(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::definition(node->getValue(storage), _definition) ) {
    _stream->printError(node, "Invalid value for 'definition:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
}

void YAMLState::parseAtomType(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::contentType(node->getValue(storage), _type) ) {
    _stream->printError(node, "Invalid value for 'type:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomDeadStrip(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::deadStripKind(node->getValue(storage), _deadStrip) ) {
    _stream->printError(node, "Invalid value for 'dead-strip:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomSectionChoice(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::sectionChoice(node->getValue(storage), _sectionChoice) ) {
    _stream->printError(node, "Invalid value for 'section-choice:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomInterposable(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::interposable(node->getValue(storage), _interpose) ) {
    _stream->printError(node, "Invalid value for 'interposable:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomMerge(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::merge(node->getValue(storage), _merge) ) {
    _stream->printError(node, "Invalid value for 'merge:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomIsThumb(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::isThumb(node->getValue(storage), _isThumb) ) {
    _stream->printError(node, "Invalid value for 'thumb:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomIsAlias(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::isAlias(node->getValue(storage), _isAlias) ) {
    _stream->printError(node, "Invalid value for 'is-alias:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomSectionName(ScalarNode *node) {
  _sectionName = extractString(node);
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomSize(ScalarNode *node) {
  llvm::SmallString<32> storage;
  StringRef offsetStr = node->getValue(storage);
  if ( offsetStr.getAsInteger(0, _size) ) {
    _stream->printError(node, "Invalid value for atom 'size:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomPermissions(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::permissions(node->getValue(storage), _permissions) ) {
    _stream->printError(node, "Invalid value for 'permissions:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomCanBeNull(ScalarNode *node) {
  llvm::SmallString<32> storage;
  if ( KeyValues::canBeNull(node->getValue(storage), _canBeNull) ) {
    _stream->printError(node, "Invalid value for 'can-be-null:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
}

void YAMLState::parseFixUpOffset(ScalarNode *node) {
  llvm::SmallString<32> storage;
  StringRef offsetStr = node->getValue(storage);
  if ( offsetStr.getAsInteger(0, _ref._offsetInAtom) ) {
    _stream->printError(node, "Invalid value for fixup 'offset:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseFixUpKind(ScalarNode *node) {
  llvm::SmallString<32> storage;
  _ref._kind = _options.kindFromString(node->getValue(storage));
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseFixUpTarget(ScalarNode *node) {
  _ref._targetNameNode = node;
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseFixUpAddend(ScalarNode *node) {
  llvm::SmallString<32> storage;
  StringRef offsetStr = node->getValue(storage);
  if ( offsetStr.getAsInteger(0, _ref._addend) ) {
    _stream->printError(node, "Invalid value for fixup 'addend:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomContentByte(ScalarNode *node) {
  llvm::SmallString<32> storage;
  StringRef str = node->getValue(storage);
  unsigned int contentByte;
  if ( str.getAsInteger(16, contentByte) ) {
    _stream->printError(node, "Invalid content hex byte '0x" + str + "'");
    _error = make_error_code(yaml_reader_error::illegal_value);
    return;
  }
  if (contentByte > 0xFF) {
    _stream->printError(node, "Content hex byte out of range (0x" 
                                                       + str + " > 0xFF)");
    _error = make_error_code(yaml_reader_error::illegal_value);
    return;
  }
  _content.push_back(contentByte & 0xFF);
  _hasDefinedAtomAttributes = true;
}

void YAMLState::parseAtomLoadName(ScalarNode *node) {
  _loadName = extractString(node);
  _hasSharedLibraryAtomAttributes = true;
}


void YAMLState::parseAtomValue(ScalarNode *node) {
  llvm::SmallString<32> storage;
  StringRef offsetStr = node->getValue(storage);
  if ( offsetStr.getAsInteger(0, _value) ) {
    _stream->printError(node, "Invalid value for fixup 'addend:'");
    _error = make_error_code(yaml_reader_error::illegal_value);
  }
  _hasAbsoluteAtomAttributes = true;
}

//
// This is the parsing engine that walks the nodes in the yaml document
// stream.  It is table driven.  See _s_transistions.
//
void YAMLState::parse(Node *node, StringRef keyword, Node *keywordNode) {
  using namespace llvm::yaml;
  DEBUG_WITH_TYPE("objtxt", llvm::dbgs() << "parse(" << keyword << "), _state=" 
                     << stateName(_state) << "\n");
  if ( _error )
    return;
  State savedState = _state;
  for(const Transistion* t=_s_transistions; t->state != inError; ++t) {
    if ( t->state != _state )
      continue;
    if ( ! keyword.equals(t->keyword) )
      continue;    
    ParseScalar action = t->customAction;
    this->moveToState(t->newState);
    if ( ScalarNode *sc = llvm::dyn_cast<ScalarNode>(node) ) {
      if ( action ) {
        (*this.*action)(sc);
      }
      else {
        _stream->printError(node, "unexpected scalar");
        _error = make_error_code(yaml_reader_error::illegal_value);
      }
    }
    else if ( SequenceNode *seq = llvm::dyn_cast<SequenceNode>(node) ) {
      if ( action ) {
        _stream->printError(node, "unexpected sequence");
        _error = make_error_code(yaml_reader_error::illegal_value);
      }
      else {
        for (Node &seqEntry : *seq ) {
          this->parse(&seqEntry, StringRef("<any-seq-item>"));
          if ( _error )
            break;
        }
      }
    }
    else if ( MappingNode *map = llvm::dyn_cast<MappingNode>(node) ) {
      if ( action ) {
        _stream->printError(node, "unexpected map");
        _error = make_error_code(yaml_reader_error::illegal_value);
      }
      else {
        llvm::SmallString<32> storage;
        for (auto &keyVal : *map) {
          ScalarNode *keyScalar = llvm::dyn_cast<ScalarNode>(keyVal.getKey());
          llvm::StringRef keyStr = keyScalar->getValue(storage);
          this->parse(keyVal.getValue(), keyStr, keyScalar);
          if ( _error )
            break;
        }
      }
    }
    else {
      _stream->printError(node, "unexpected node type");
      _error = make_error_code(yaml_reader_error::illegal_value);
    }
    this->returnToState(savedState, node);
    return;
  }
  switch (_state) {
    case inAtom:
      _stream->printError(keywordNode, "Unknown atom attribute '" 
                                        + keyword + ":'");
      break;
    case inFixUp:
      _stream->printError(keywordNode, "Unknown fixup attribute '" 
                                        + keyword + ":'");
      break;
    case inDoc:
      _stream->printError(keywordNode, "Unknown file attribute '" 
                                        + keyword + ":'");
      break;
    default:
      _stream->printError(keywordNode, "Unknown keyword '" 
                                        + keyword + ":'");
  }
  _error = make_error_code(yaml_reader_error::illegal_value);
}


/// parseFile - Parse the specified YAML formatted MemoryBuffer
/// into lld::File object(s) and append each to the specified vector<File*>.
static error_code parseFile(std::unique_ptr<MemoryBuffer> &mb,
                      const ReaderOptionsYAML &options,
                      std::vector<std::unique_ptr<File>> &result) {
  llvm::SourceMgr       srcMgr;
  llvm::yaml::Stream    stream(mb->getBuffer(), srcMgr);

  for (llvm::yaml::Document &d : stream) {
    std::unique_ptr<yaml::YAMLFile> curFile(new yaml::YAMLFile);
    if (llvm::isa<llvm::yaml::NullNode>(d.getRoot()))
      continue; // Empty files are allowed.
    yaml::YAMLState yamlState(options, &stream, curFile.get());
    yamlState.parse(d.getRoot(), StringRef("<root>"));

    if ( stream.failed() ) 
      return make_error_code(yaml_reader_error::illegal_value);
    if ( yamlState.error() ) 
      return yamlState.error();
    
    error_code ec = curFile->bindTargetReferences(stream);
    if ( ec )
      return ec;
    result.emplace_back(curFile.release());
  }

  return make_error_code(yaml_reader_error::success);
}



} // namespace yaml



class ReaderYAML: public Reader {
public:
  ReaderYAML(const ReaderOptionsYAML &options) : _options(options) {
  }

  error_code parseFile(std::unique_ptr<MemoryBuffer> mb,
                       std::vector<std::unique_ptr<File>> &result) {
    return lld::yaml::parseFile(mb, _options, result);
  }

private:
  const ReaderOptionsYAML &_options;
};



Reader* createReaderYAML(const ReaderOptionsYAML &options) {
  return new ReaderYAML(options);
}

ReaderOptionsYAML::ReaderOptionsYAML() {
}

ReaderOptionsYAML::~ReaderOptionsYAML() {
}




} // namespace lld
