//===- Core/YamlReader.cpp - Reads YAML -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "YamlKeyValues.h"

#include "lld/Core/YamlReader.h"
#include "lld/Core/Atom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"

#include <vector>



namespace lld {
namespace yaml {

enum yaml_reader_errors {
  success = 0,
  unknown_keyword,
  illegal_value
};

class reader_error_category : public llvm::_do_message {
public:
  virtual const char* name() const {
    return "lld.yaml.reader";
  }
  virtual std::string message(int ev) const;
};

const reader_error_category reader_error_category_singleton;

std::string reader_error_category::message(int ev) const {
  switch (ev) {
  case success: 
    return "Success";
  case unknown_keyword:
    return "Unknown keyword found in yaml file";
  case illegal_value: 
    return "Bad value found in yaml file";
  default:
    llvm_unreachable("An enumerator of yaml_reader_errors does not have a "
                     "message defined.");
  }
}

inline llvm::error_code make_error_code(yaml_reader_errors e) {
  return llvm::error_code(static_cast<int>(e), reader_error_category_singleton);
}


class YAML {
public:
  struct Entry {
    Entry(const char *k, const char *v, std::vector<uint8_t>* vs, 
          int d, bool bd, bool bs)
      : key(strdup(k))
      , value(v ? strdup(v) : NULL)
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
  char *p = NULL;
  unsigned int lineNumber = 1;
  int depth = 0;
  bool nextKeyIsStartOfDocument = false;
  bool nextKeyIsStartOfSequence = false;
  std::vector<uint8_t>* sequenceBytes = NULL;
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
        entries.push_back(new Entry(key, "", NULL, depth,
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
      if (isalnum(c) || (c == '-') || (c == '_')) {
        *p++ = c;
      } else if (c == '\n') {
        *p = '\0';
        entries.push_back(new Entry(key, value, NULL, depth,
                                    nextKeyIsStartOfDocument,
                                    nextKeyIsStartOfSequence));
        nextKeyIsStartOfSequence = false;
        nextKeyIsStartOfDocument = false;
        state = inDocument;
        depth = 0;
      }
      break;
    case inValueSequence:
      if (c == ']') {
        sequenceBytes->push_back(contentByte);
        state = inValueSequenceEnd;
      }
      else if (c == ' ') {
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
        entries.push_back(new Entry(key, NULL, sequenceBytes, depth,
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

class YAMLFile : public File {
public:
  YAMLFile()
    : File("path")
    , _lastRefIndex(0) {}

  virtual bool forEachAtom(File::AtomHandler &) const;
  virtual bool justInTimeforEachAtom(llvm::StringRef name,
                                     File::AtomHandler &) const;

  std::vector<DefinedAtom*> _definedAtoms;
  std::vector<UndefinedAtom*> _undefinedAtoms;
  std::vector<Reference> _references;
  unsigned int _lastRefIndex;
};

bool YAMLFile::forEachAtom(File::AtomHandler &handler) const {
  handler.doFile(*this);
  for (std::vector<DefinedAtom *>::const_iterator it = _definedAtoms.begin();
       it != _definedAtoms.end(); ++it) {
    handler.doDefinedAtom(**it);
  }
  return true;
}

bool YAMLFile::justInTimeforEachAtom(llvm::StringRef name,
                                     File::AtomHandler &handler) const {
  return false;
}


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
          , bool internalName
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
    , _internalName(internalName)
    , _isThumb(isThumb)
    , _isAlias(isAlias)
    , _refStartIndex(file._lastRefIndex)
    , _refEndIndex(file._references.size()) {
    file._lastRefIndex = _refEndIndex;
  }

  virtual const class File& file() const {
    return _file;
  }

  virtual llvm::StringRef name() const {
    return _name;
  }

  virtual bool internalName() const {
    return _internalName;
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

  virtual llvm::StringRef customSectionName() const {
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
  
 llvm::ArrayRef<uint8_t> rawContent() const {
    if ( _content != NULL ) 
      return llvm::ArrayRef<uint8_t>(*_content);
    else
      return llvm::ArrayRef<uint8_t>();
  }
 
  virtual uint64_t ordinal() const {
    return _ord;
  }

  
  virtual Reference::iterator referencesBegin() const {
    if (_file._references.size() < _refStartIndex)
      return (Reference::iterator)&_file._references[_refStartIndex];
    return 0;
  }
  
  virtual Reference::iterator referencesEnd() const {
    if (_file._references.size() < _refEndIndex)
      return (Reference::iterator)&_file._references[_refEndIndex];
    return 0;
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
  bool                        _internalName;
  bool                        _isThumb;
  bool                        _isAlias;
  unsigned int                _refStartIndex;
  unsigned int                _refEndIndex;
};


class YAMLAtomState {
public:
  YAMLAtomState();

  void setName(const char *n);
  void setAlign2(const char *n);

  void setFixupKind(const char *n);
  void setFixupOffset(const char *n);
  void setFixupTarget(const char *n);
  void addFixup(YAMLFile *f);

  void makeAtom(YAMLFile&);

  const char *                _name;
  const char *                _sectionName;
  unsigned long long          _size;
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
  bool                        _internalName;
  bool                        _isThumb;
  bool                        _isAlias;
  Reference                   _ref;
};


YAMLAtomState::YAMLAtomState()
  : _name(NULL)
  , _sectionName(NULL)
  , _size(0)
  , _ordinal(0)
  , _content(NULL) 
  , _alignment(0, 0)
  , _definition(KeyValues::definitionDefault)
  , _scope(KeyValues::scopeDefault)
  , _type(KeyValues::contentTypeDefault)
  , _sectionChoice(KeyValues::sectionChoiceDefault)
  , _interpose(KeyValues::interposableDefault)
  , _merge(KeyValues::mergeDefault)
  , _deadStrip(KeyValues::deadStripKindDefault)
  , _permissions(KeyValues::permissionsDefault)
  , _internalName(KeyValues::internalNameDefault)
  , _isThumb(KeyValues::isThumbDefault)
  , _isAlias(KeyValues::isAliasDefault) 
  {
  _ref.target       = NULL;
  _ref.addend       = 0;
  _ref.offsetInAtom = 0;
  _ref.kind         = 0;
  _ref.flags        = 0;
}


void YAMLAtomState::makeAtom(YAMLFile& f) {
  if ( _definition == Atom::definitionRegular ) {
    DefinedAtom *a = new YAMLDefinedAtom(_ordinal, f, _scope, _type,
                          _sectionChoice, _interpose, _merge, _deadStrip,
                          _permissions,  _internalName, _isThumb, _isAlias, 
                          _alignment, _name, _sectionName, _size, _content);

    f._definedAtoms.push_back(a);
    ++_ordinal;
  }
  
  // reset state for next atom
  _name             = NULL;
  _sectionName      = NULL;
  _size             = 0;
  _ordinal          = 0;
  _content          = NULL;
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
  _ref.target       = NULL;
  _ref.addend       = 0;
  _ref.offsetInAtom = 0;
  _ref.kind         = 0;
  _ref.flags        = 0;
}

void YAMLAtomState::setName(const char *n) {
  _name = n;
}


void YAMLAtomState::setAlign2(const char *s) {
  llvm::StringRef str(s);
  uint32_t res;
  str.getAsInteger(10, res);
  _alignment.powerOf2 = static_cast<uint16_t>(res);
}


void YAMLAtomState::setFixupKind(const char *s) {
  if (strcmp(s, "pcrel32") == 0)
    _ref.kind = 1;
  else if (strcmp(s, "call32") == 0)
    _ref.kind = 2;
  else
    llvm::report_fatal_error("bad fixup kind value");
}

void YAMLAtomState::setFixupOffset(const char *s) {
  if ((s[0] == '0') && (s[1] == 'x'))
    llvm::StringRef(s).getAsInteger(16, _ref.offsetInAtom);
  else
    llvm::StringRef(s).getAsInteger(10, _ref.offsetInAtom);
}

void YAMLAtomState::setFixupTarget(const char *s) {
}

void YAMLAtomState::addFixup(YAMLFile *f) {
  f->_references.push_back(_ref);
  // clear for next ref
  _ref.target       = NULL;
  _ref.addend       = 0;
  _ref.offsetInAtom = 0;
  _ref.kind         = 0;
  _ref.flags        = 0;
}

llvm::error_code parseObjectText( llvm::MemoryBuffer *mb
                                , std::vector<File *> &result) {
  std::vector<const YAML::Entry *> entries;
  YAML::parse(mb, entries);

  YAMLFile *file = NULL;
  YAMLAtomState atomState;
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
      if (file != NULL) {
        if (haveAtom) {
          atomState.makeAtom(*file);
          haveAtom = false;
        }
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
        else if (strcmp(entry->key, KeyValues::internalNameKeyword) == 0) {
          atomState._internalName = KeyValues::internalName(entry->value);
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
        else if (strcmp(entry->key, KeyValues::sectionNameKeyword) == 0) {
          atomState._sectionName = entry->value;
          haveAtom = true;
        } 
        else if (strcmp(entry->key, KeyValues::sizeKeyword) == 0) {
          llvm::StringRef val = entry->value;
          if ( val.getAsInteger(0, atomState._size) ) 
            return make_error_code(illegal_value);
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
        else if (strcmp(entry->key, "fixups") == 0) {
          inFixups = true;
        }
        else {
          return make_error_code(unknown_keyword);
        }
      } 
      else if (depthForFixups == entry->depth) {
        if (entry->beginSequence) {
          if (haveFixup) {
            atomState.addFixup(file);
            haveFixup = false;
          }
        }
        if (strcmp(entry->key, "kind") == 0) {
          atomState.setFixupKind(entry->value);
          haveFixup = true;
        } else if (strcmp(entry->key, "offset") == 0) {
          atomState.setFixupOffset(entry->value);
          haveFixup = true;
        } else if (strcmp(entry->key, "target") == 0) {
          atomState.setFixupTarget(entry->value);
          haveFixup = true;
        }
      }
    }
    lastDepth = entry->depth;
  }
  if (haveAtom) {
    atomState.makeAtom(*file);
  }

  result.push_back(file);
  return make_error_code(success);
}

//
// Fill in vector<File*> from path to input text file.
//
llvm::error_code parseObjectTextFileOrSTDIN(llvm::StringRef path
                                 , std::vector<File*>& result) {
  llvm::OwningPtr<llvm::MemoryBuffer> mb;
  llvm::error_code ec = llvm::MemoryBuffer::getFileOrSTDIN(path, mb);
  if ( ec ) 
      return ec;
      
  return parseObjectText(mb.get(), result);
}


} // namespace yaml
} // namespace lld
