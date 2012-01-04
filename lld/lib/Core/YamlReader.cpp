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
    Entry(const char *k, const char *v, int d, bool bd, bool bs)
      : key(strdup(k))
      , value(strdup(v))
      , depth(d)
      , beginSequence(bs)
      , beginDocument(bd) {}

    const char *key;
    const char *value;
    int         depth;
    bool        beginSequence;
    bool        beginDocument;
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
        entries.push_back(new Entry(key, "", depth,
                                    nextKeyIsStartOfDocument,
                                    nextKeyIsStartOfSequence));
        nextKeyIsStartOfSequence = false;
        nextKeyIsStartOfDocument = false;
        state = inDocument;
        depth = 0;
      } else if (c == '[') {
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
        entries.push_back(new Entry(key, value, depth,
                                    nextKeyIsStartOfDocument,
                                    nextKeyIsStartOfSequence));
        nextKeyIsStartOfSequence = false;
        nextKeyIsStartOfDocument = false;
        state = inDocument;
        depth = 0;
      }
      break;
    case inValueSequence:
      if (c == ']')
        state = inValueSequenceEnd;
      break;
    case inValueSequenceEnd:
      if (c == '\n') {
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

  std::vector<Atom *> _atoms;
  std::vector<Reference> _references;
  unsigned int _lastRefIndex;
};

bool YAMLFile::forEachAtom(File::AtomHandler &handler) const {
  handler.doFile(*this);
  for (std::vector<Atom *>::const_iterator it = _atoms.begin();
       it != _atoms.end(); ++it) {
    handler.doAtom(**it);
  }
  return true;
}

bool YAMLFile::justInTimeforEachAtom(llvm::StringRef name,
                                     File::AtomHandler &handler) const {
  return false;
}


class YAMLAtom : public Atom {
public:
  YAMLAtom( uint64_t ord
          , Definition d
          , Scope s
          , ContentType ct
          , SectionChoice sc
          , bool intn
          , bool md
          , bool ah
          , DeadStripKind dsk
          , bool tb
          , bool al
          , Alignment a
          , YAMLFile& f
          , const char *n
          , const char* sn
          , uint64_t sz)
    : Atom(ord, d, s, ct, sc, intn, md, ah, dsk, tb, al, a)
    , _file(f)
    , _name(n)
    , _sectionName(sn)
    , _size(sz)
    , _refStartIndex(f._lastRefIndex)
    , _refEndIndex(f._references.size()) {
    f._lastRefIndex = _refEndIndex;
  }

  virtual const class File& file() const {
    return _file;
  }

  virtual bool translationUnitSource(const char* *dir, const char* *name) const{
    return false;
  }

  virtual llvm::StringRef name() const {
    return _name;
  }
  
  virtual llvm::StringRef customSectionName() const {
    return _sectionName;
  }

  virtual uint64_t objectAddress() const {
    return 0;
  }

  virtual uint64_t size() const {
    return _size;
  }

  virtual void copyRawContent(uint8_t buffer[]) const { }
  virtual Reference::iterator referencesBegin() const;
  virtual Reference::iterator referencesEnd() const;
private:
  YAMLFile&      _file;
  const char *   _name;
  const char *   _sectionName;
  unsigned long  _size;
  unsigned int   _refStartIndex;
  unsigned int   _refEndIndex;
};

Reference::iterator YAMLAtom::referencesBegin() const {
  if (_file._references.size() < _refStartIndex)
    return (Reference::iterator)&_file._references[_refStartIndex];
  return 0;
}

Reference::iterator YAMLAtom::referencesEnd() const {
  if (_file._references.size() < _refEndIndex)
    return (Reference::iterator)&_file._references[_refEndIndex];
  return 0;
}

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

  uint64_t  _ordinal;
  long long _size;
  const char *_name;
  Atom::Alignment _align;
  Atom::ContentType _type;
  Atom::Scope _scope;
  Atom::Definition _def;
  Atom::SectionChoice _sectionChoice;
  bool _internalName;
  bool _mergeDuplicates;
  Atom::DeadStripKind _deadStrip;
  bool _thumb;
  bool _alias;
  bool _autoHide;
  const char *_sectionName;
  Reference _ref;
};

YAMLAtomState::YAMLAtomState()
  : _ordinal(0)
  , _size(0)
  , _name(NULL)
  , _align(0, 0)
  , _type(KeyValues::contentTypeDefault)
  , _scope(KeyValues::scopeDefault)
  , _def(KeyValues::definitionDefault)
  , _internalName(KeyValues::internalNameDefault)
  , _mergeDuplicates(KeyValues::mergeDuplicatesDefault)
  , _deadStrip(KeyValues::deadStripKindDefault)
  , _thumb(KeyValues::isThumbDefault)
  , _alias(KeyValues::isAliasDefault) 
  , _autoHide(KeyValues::autoHideDefault)
  , _sectionName(NULL) {
  _ref.target       = NULL;
  _ref.addend       = 0;
  _ref.offsetInAtom = 0;
  _ref.kind         = 0;
  _ref.flags        = 0;
}

void YAMLAtomState::makeAtom(YAMLFile& f) {
  Atom *a = new YAMLAtom(_ordinal, _def, _scope, _type, _sectionChoice,
                         _internalName, _mergeDuplicates, _autoHide,  
                         _deadStrip, _thumb, _alias, _align, f, 
                         _name, _sectionName, _size);

  f._atoms.push_back(a);
  ++_ordinal;
  
  // reset state for next atom
  _name             = NULL;
  _align.powerOf2   = 0;
  _align.modulus    = 0;
  _type             = KeyValues::contentTypeDefault;
  _scope            = KeyValues::scopeDefault;
  _def              = KeyValues::definitionDefault;
  _sectionChoice    = KeyValues::sectionChoiceDefault;
  _internalName     = KeyValues::internalNameDefault;
  _mergeDuplicates  = KeyValues::mergeDuplicatesDefault;
  _deadStrip        = KeyValues::deadStripKindDefault;
  _thumb            = KeyValues::isThumbDefault;
  _alias            = KeyValues::isAliasDefault;
  _autoHide         = KeyValues::autoHideDefault;
  _sectionName      = NULL;
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
  _align.powerOf2 = static_cast<uint16_t>(res);
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
          atomState._def = KeyValues::definition(entry->value);
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
        else if (strcmp(entry->key, KeyValues::mergeDuplicatesKeyword) == 0) {
          atomState._mergeDuplicates = KeyValues::mergeDuplicates(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::autoHideKeyword) == 0) {
          atomState._autoHide = KeyValues::autoHide(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::isThumbKeyword) == 0) {
          atomState._thumb = KeyValues::isThumb(entry->value);
          haveAtom = true;
        }
        else if (strcmp(entry->key, KeyValues::isAliasKeyword) == 0) {
          atomState._alias = KeyValues::isAlias(entry->value);
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
          // TO DO: switch to content mode
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
