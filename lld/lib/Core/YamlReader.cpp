//===- Core/YamlReader.cpp - Reads YAML -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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

namespace { const llvm::error_code success; }

namespace lld {
namespace yaml {
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
  YAMLAtom( Definition d
          , Combine c
          , Scope s
          , ContentType ct
          , SectionChoice sc
          , bool uvn
          , bool dds
          , bool tb
          , bool al
          , Alignment a
          , YAMLFile *f
          , const char *n)
    : Atom(d, c, s, ct, sc, uvn, dds, tb, al, a)
    , _file(f)
    , _name(n)
    , _size(0)
    , _refStartIndex(f->_lastRefIndex)
    , _refEndIndex(f->_references.size()) {
    f->_lastRefIndex = _refEndIndex;
  }

  virtual const class File *file() const {
    return _file;
  }

  virtual bool translationUnitSource(const char* *dir, const char* *name) const{
    return false;
  }

  virtual llvm::StringRef name() const {
    return _name;
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
  YAMLFile *_file;
  const char *_name;
  unsigned long _size;
  unsigned int _refStartIndex;
  unsigned int _refEndIndex;
};

Reference::iterator YAMLAtom::referencesBegin() const {
  if (_file->_references.size() < _refStartIndex)
    return (Reference::iterator)&_file->_references[_refStartIndex];
  return 0;
}

Reference::iterator YAMLAtom::referencesEnd() const {
  if (_file->_references.size() < _refEndIndex)
    return (Reference::iterator)&_file->_references[_refEndIndex];
  return 0;
}

class YAMLAtomState {
public:
  YAMLAtomState();

  void setName(const char *n);
  void setScope(const char *n);
  void setType(const char *n);
  void setAlign2(const char *n);
  void setDefinition(const char *n);

  void setFixupKind(const char *n);
  void setFixupOffset(const char *n);
  void setFixupTarget(const char *n);
  void addFixup(YAMLFile *f);

  void makeAtom(YAMLFile *);

private:
  const char *_name;
  Atom::Alignment _align;
  Atom::Combine _combine;
  Atom::ContentType _type;
  Atom::Scope _scope;
  Atom::Definition _def;
  Atom::SectionChoice _sectionChoice;
  bool _userVisibleName;
  bool _dontDeadStrip;
  bool _thumb;
  bool _alias;
  Reference _ref;
};

YAMLAtomState::YAMLAtomState()
  : _name(NULL)
  , _align(0, 0)
  , _combine(Atom::combineNever)
  , _type(Atom::typeData)
  , _scope(Atom::scopeGlobal)
  , _userVisibleName(true)
  , _dontDeadStrip(false)
  , _thumb(false)
  , _alias(false) {
  _ref.target       = NULL;
  _ref.addend       = 0;
  _ref.offsetInAtom = 0;
  _ref.kind         = 0;
  _ref.flags        = 0;
}

void YAMLAtomState::makeAtom(YAMLFile *f) {
  Atom *a = new YAMLAtom(_def, _combine, _scope, _type, _sectionChoice,
                         _userVisibleName, _dontDeadStrip, _thumb, _alias,
                         _align, f, _name);

  f->_atoms.push_back(a);

  // reset state for next atom
  _name             = NULL;
  _align.powerOf2   = 0;
  _align.modulus    = 0;
  _combine          = Atom::combineNever;
  _type             = Atom::typeData;
  _scope            = Atom::scopeGlobal;
  _def              = Atom::definitionRegular;
  _sectionChoice    = Atom::sectionBasedOnContent;
  _userVisibleName  = true;
  _dontDeadStrip    = false;
  _thumb            = false;
  _alias            = false;
  _ref.target       = NULL;
  _ref.addend       = 0;
  _ref.offsetInAtom = 0;
  _ref.kind         = 0;
  _ref.flags        = 0;
}

void YAMLAtomState::setName(const char *n) {
  _name = n;
}

void YAMLAtomState::setScope(const char *s) {
  if (strcmp(s, "global") == 0)
    _scope = Atom::scopeGlobal;
  else if (strcmp(s, "hidden") == 0)
    _scope = Atom::scopeLinkageUnit;
  else if (strcmp(s, "static") == 0)
    _scope = Atom::scopeTranslationUnit;
  else
    llvm::report_fatal_error("bad scope value");
}

void YAMLAtomState::setType(const char *s) {
  if (strcmp(s, "code") == 0)
    _type = Atom::typeCode;
  else if (strcmp(s, "c-string") == 0)
    _type = Atom::typeCString;
  else if (strcmp(s, "zero-fill") == 0)
    _type = Atom::typeZeroFill;
  else if (strcmp(s, "data") == 0)
    _type = Atom::typeData;
  else
    llvm::report_fatal_error("bad type value");
}

void YAMLAtomState::setAlign2(const char *s) {
  llvm::StringRef str(s);
  uint32_t res;
  str.getAsInteger(10, res);
  _align.powerOf2 = static_cast<uint16_t>(res);
}

void YAMLAtomState::setDefinition(const char *s) {
  if (strcmp(s, "regular") == 0)
    _def = Atom::definitionRegular;
  else if (strcmp(s, "tentative") == 0)
    _def = Atom::definitionTentative;
  else if (strcmp(s, "absolute") == 0)
    _def = Atom::definitionAbsolute;
  else
    llvm::report_fatal_error("bad definition value");
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
          atomState.makeAtom(file);
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
            atomState.makeAtom(file);
            haveAtom = false;
          }
        }
        if (strcmp(entry->key, "name") == 0) {
          atomState.setName(entry->value);
          haveAtom = true;
        } else if (strcmp(entry->key, "scope") == 0) {
          atomState.setScope(entry->value);
          haveAtom = true;
        } else if (strcmp(entry->key, "type") == 0) {
          atomState.setType(entry->value);
          haveAtom = true;
        } else if (strcmp(entry->key, "align2") == 0) {
          atomState.setAlign2(entry->value);
          haveAtom = true;
        } else if (strcmp(entry->key, "definition") == 0) {
          atomState.setDefinition(entry->value);
          haveAtom = true;
        } else if (strcmp(entry->key, "fixups") == 0) {
          inFixups = true;
        }
      } else if (depthForFixups == entry->depth) {
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
    atomState.makeAtom(file);
  }

  result.push_back(file);
  return success;
}
} // namespace yaml
} // namespace lld
