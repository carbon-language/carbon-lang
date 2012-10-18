//===- lib/ReaderWriter/YAML/WriterYAML.cpp - Writes YAML object files ----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/WriterYAML.h"

#include "lld/Core/Atom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

#include "YamlKeyValues.h"

#include <vector>

namespace lld {
namespace yaml {

///
/// In most cases, atoms names are unambiguous, so references can just
/// use the atom name as the target (e.g. target: foo).  But in a few
/// cases that does not work, so ref-names are added.  These are labels
/// used only in yaml.  The labels do not exist in the Atom model.
///
/// One need for ref-names are when atoms have no user supplied name
/// (e.g. c-string literal).  Another case is when two object files with
/// identically named static functions are merged (ld -r) into one object file.
/// In that case referencing the function by name is ambiguous, so a unique
/// ref-name is added.
///
class RefNameBuilder {
public:
  RefNameBuilder(const File& file)
                : _collisionCount(0), _unnamedCounter(0) {
    // visit all atoms
    for( const DefinedAtom *atom : file.defined() ) {
      // Build map of atoms names to detect duplicates
      if ( ! atom->name().empty() )
        buildDuplicateNameMap(*atom);

      // Find references to unnamed atoms and create ref-names for them.
      for (const Reference *ref : *atom) {
        // create refname for any unnamed reference target
        const Atom *target = ref->target();
        if ( (target != nullptr) && target->name().empty() ) {
          std::string Storage;
          llvm::raw_string_ostream Buffer(Storage);
          Buffer << llvm::format("L%03d", _unnamedCounter++);
          _refNames[target] = Buffer.str();
        }
      }
    }
    for( const UndefinedAtom *undefAtom : file.undefined() ) {
      buildDuplicateNameMap(*undefAtom);
    }
    for( const SharedLibraryAtom *shlibAtom : file.sharedLibrary() ) {
      buildDuplicateNameMap(*shlibAtom);
    }
    for( const AbsoluteAtom *absAtom : file.absolute() ) {
      buildDuplicateNameMap(*absAtom);
    }


  }

  void buildDuplicateNameMap(const Atom& atom) {
    assert(!atom.name().empty());
    NameToAtom::iterator pos = _nameMap.find(atom.name());
    if ( pos != _nameMap.end() ) {
      // Found name collision, give each a unique ref-name.
      std::string Storage;
      llvm::raw_string_ostream Buffer(Storage);
      Buffer << atom.name() << llvm::format(".%03d", ++_collisionCount);
      _refNames[&atom] = Buffer.str();
      const Atom* prevAtom = pos->second;
      AtomToRefName::iterator pos2 = _refNames.find(prevAtom);
      if ( pos2 == _refNames.end() ) {
        // only create ref-name for previous if none already created
        Buffer << prevAtom->name() << llvm::format(".%03d", ++_collisionCount);
        _refNames[prevAtom] = Buffer.str();
      }
    }
    else {
      // First time we've seen this name, just add it to map.
      _nameMap[atom.name()] = &atom;
    }
  }

  bool hasRefName(const Atom* atom) {
     return _refNames.count(atom);
  }

  StringRef refName(const Atom *atom) {
     return _refNames.find(atom)->second;
  }

private:
  typedef llvm::StringMap<const Atom*> NameToAtom;
  typedef llvm::DenseMap<const Atom*, std::string> AtomToRefName;

  unsigned int      _collisionCount;
  unsigned int      _unnamedCounter;
  NameToAtom        _nameMap;
  AtomToRefName     _refNames;
};


///
/// Helper class for writeObjectText() to write out atoms in yaml format.
///
class AtomWriter {
public:
  AtomWriter(const File& file, const WriterOptionsYAML &options, 
                                                            RefNameBuilder& rnb)
    : _file(file), _options(options), _rnb(rnb), _firstAtom(true) { }


  void write(raw_ostream &out) {
    // write header
    out << "---\n";

    // visit all atoms
    for( const DefinedAtom *atom : _file.defined() ) {
      writeDefinedAtom(*atom, out);
    }
    for( const UndefinedAtom *undefAtom : _file.undefined() ) {
      writeUndefinedAtom(*undefAtom, out);
    }
    for( const SharedLibraryAtom *shlibAtom : _file.sharedLibrary() ) {
      writeSharedLibraryAtom(*shlibAtom, out);
    }
    for( const AbsoluteAtom *absAtom : _file.absolute() ) {
      writeAbsoluteAtom(*absAtom, out);
    }

    out << "...\n";
  }


  void writeDefinedAtom(const DefinedAtom &atom, raw_ostream &out) {
    if ( _firstAtom ) {
      out << "atoms:\n";
      _firstAtom = false;
    }
    else {
      // add blank line between atoms for readability
      out << "\n";
    }

    bool hasDash = false;
    if ( !atom.name().empty() ) {
      out   << "    - "
            << "name:"
            << spacePadding(strlen("name"))
            << atom.name()
            << "\n";
      hasDash = true;
    }

    if ( _rnb.hasRefName(&atom) ) {
      out   << (hasDash ? "      " : "    - ")
            << "ref-name:"
            << spacePadding(strlen("ref-name"))
            << _rnb.refName(&atom)
            << "\n";
      hasDash = true;
    }

    if ( atom.definition() != KeyValues::definitionDefault ) {
      out   << (hasDash ? "      " : "    - ")
            << "definition:"
            << spacePadding(strlen("definition"))
            << KeyValues::definition(atom.definition())
            << "\n";
      hasDash = true;
    }

    if ( atom.scope() != KeyValues::scopeDefault ) {
      out   << (hasDash ? "      " : "    - ")
            << "scope:"
            << spacePadding(strlen("scope"))
            << KeyValues::scope(atom.scope())
            << "\n";
    }

     if ( atom.interposable() != KeyValues::interposableDefault ) {
      out   << "      "
            << "interposable:"
            << spacePadding(strlen("interposable"))
            << KeyValues::interposable(atom.interposable())
            << "\n";
    }

    if ( atom.merge() != KeyValues::mergeDefault ) {
      out   << "      "
            << "merge:"
            << spacePadding(strlen("merge"))
            << KeyValues::merge(atom.merge())
            << "\n";
    }

    if ( atom.contentType() != KeyValues::contentTypeDefault ) {
      out   << "      "
            << "type:"
            << spacePadding(strlen("type"))
            << KeyValues::contentType(atom.contentType())
            << "\n";
    }

    if ( atom.deadStrip() != KeyValues::deadStripKindDefault ) {
      out   << "      "
            << "dead-strip:"
            << spacePadding(strlen("dead-strip"))
            << KeyValues::deadStripKind(atom.deadStrip())
            << "\n";
    }

    if ( atom.sectionChoice() != KeyValues::sectionChoiceDefault ) {
      out   << "      "
            << "section-choice:"
            << spacePadding(strlen("section-choice"))
            << KeyValues::sectionChoice(atom.sectionChoice())
            << "\n";
      assert( ! atom.customSectionName().empty() );
      out   << "      "
            << "section-name:"
            << spacePadding(strlen("section-name"))
            << atom.customSectionName()
            << "\n";
    }

    if ( atom.isThumb() != KeyValues::isThumbDefault ) {
      out   << "      "
            << "is-thumb:"
            << spacePadding(strlen("is-thumb"))
            << KeyValues::isThumb(atom.isThumb())
            << "\n";
    }

    if ( atom.isAlias() != KeyValues::isAliasDefault ) {
      out   << "      "
            << "is-alias:"
            << spacePadding(strlen("is-alias"))
            << KeyValues::isAlias(atom.isAlias())
            << "\n";
    }

    if ( (atom.contentType() != DefinedAtom::typeZeroFill)
                                   && (atom.size() != 0) ) {
      out   << "      "
            << "content:"
            << spacePadding(strlen("content"))
            << "[ ";
      ArrayRef<uint8_t> arr = atom.rawContent();
      bool needComma = false;
      for (unsigned int i=0; i < arr.size(); ++i) {
        if ( needComma )
          out  << ", ";
        if ( ((i % 12) == 0) && (i != 0) ) {
          out << "\n                           ";
        }
        out  << hexdigit(arr[i] >> 4);
        out  << hexdigit(arr[i] & 0x0F);
        needComma = true;
      }
      out  << " ]\n";
    }

    bool wroteFirstFixup = false;
    for (const Reference *ref : atom) {
      if ( !wroteFirstFixup ) {
        out  << "      fixups:\n";
        wroteFirstFixup = true;
      }
      out   << "      - "
            << "offset:"
            << spacePadding(strlen("offset"))
            << ref->offsetInAtom()
            << "\n";
      out   << "        "
            << "kind:"
            << spacePadding(strlen("kind"))
            << _options.kindToString(ref->kind())
            << "\n";
      const Atom* target = ref->target();
      if (target != nullptr) {
        StringRef refName = target->name();
        if ( _rnb.hasRefName(target) )
          refName = _rnb.refName(target);
        assert(!refName.empty());
        out   << "        "
              << "target:"
              << spacePadding(strlen("target"))
              << refName
              << "\n";
      }
      if ( ref->addend() != 0 ) {
        out   << "        "
              << "addend:"
              << spacePadding(strlen("addend"))
              << ref->addend()
              << "\n";
      }
    }
  }


  void writeUndefinedAtom(const UndefinedAtom &atom, raw_ostream &out) {
    if ( _firstAtom ) {
      out  << "atoms:\n";
      _firstAtom = false;
    }
    else {
      // add blank line between atoms for readability
      out  << "\n";
    }

    out   << "    - "
          << "name:"
          << spacePadding(strlen("name"))
          << atom.name()
          << "\n";

    out   << "      "
          << "definition:"
          << spacePadding(strlen("definition"))
          << KeyValues::definition(atom.definition())
          << "\n";

    if ( atom.canBeNull() != KeyValues::canBeNullDefault ) {
      out   << "      "
            << "can-be-null:"
            << spacePadding(strlen("can-be-null"))
            << KeyValues::canBeNull(atom.canBeNull())
            << "\n";
    }
  }

  void writeSharedLibraryAtom(const SharedLibraryAtom &atom, raw_ostream &out) {
    if ( _firstAtom ) {
      out  << "atoms:\n";
      _firstAtom = false;
    }
    else {
      // add blank line between atoms for readability
      out  << "\n";
    }

    out   << "    - "
          << "name:"
          << spacePadding(strlen("name"))
          << atom.name()
          << "\n";

    out   << "      "
          << "definition:"
          << spacePadding(strlen("definition"))
          << KeyValues::definition(atom.definition())
          << "\n";

    if ( !atom.loadName().empty() ) {
      out   << "      "
            << "load-name:"
            << spacePadding(strlen("load-name"))
            << atom.loadName()
            << "\n";
    }

    if ( atom.canBeNullAtRuntime() ) {
      out   << "      "
            << "can-be-null:"
            << spacePadding(strlen("can-be-null"))
            << KeyValues::canBeNull(UndefinedAtom::canBeNullAtRuntime)
            << "\n";
    }
   }

  void writeAbsoluteAtom(const AbsoluteAtom &atom, raw_ostream &out) {
     if ( _firstAtom ) {
      out << "atoms:\n";
      _firstAtom = false;
    }
    else {
      // add blank line between atoms for readability
      out << "\n";
    }

    out   << "    - "
          << "name:"
          << spacePadding(strlen("name"))
          << atom.name()
          << "\n";

    out   << "      "
          << "definition:"
          << spacePadding(strlen("definition"))
          << KeyValues::definition(atom.definition())
          << "\n";

    if ( atom.scope() != KeyValues::scopeDefault ) {
    out   << "      "
            << "scope:"
            << spacePadding(strlen("scope"))
            << KeyValues::scope(atom.scope())
            << "\n";
    }

    out   << "      "
          << "value:"
          << spacePadding(strlen("value"))
          << "0x";
     out.write_hex(atom.value());
     out << "\n";
   }


private:
  // return a string of the correct number of spaces to align value
  const char* spacePadding(size_t keyLen) {
    const char* spaces = "                  ";
    assert(strlen(spaces) > keyLen);
    return &spaces[keyLen];
  }

  char hexdigit(uint8_t nibble) {
    if ( nibble < 0x0A )
      return '0' + nibble;
    else
      return 'A' + nibble - 0x0A;
  }

  const File                      &_file;
  const WriterOptionsYAML         &_options;
  RefNameBuilder                  &_rnb;
  bool                             _firstAtom;
};




class Writer : public lld::Writer {
public:
  Writer(const WriterOptionsYAML &options) : _options(options) {
  }
  
  virtual error_code writeFile(const lld::File &file, StringRef path) {
    // Create stream to path.
    std::string errorInfo;
    llvm::raw_fd_ostream out(path.data(), errorInfo);
    if (!errorInfo.empty())
      return llvm::make_error_code(llvm::errc::no_such_file_or_directory);

    // Figure what ref-name labels are needed.
    RefNameBuilder rnb(file);

    // Write out all atoms.
    AtomWriter writer(file, _options, rnb);
    writer.write(out);
    return error_code::success();
  }
  
  virtual StubsPass *stubPass() {
    return _options.stubPass();
  }
  
  virtual GOTPass *gotPass() {
    return _options.gotPass();
  }
  
  
private:
  const WriterOptionsYAML &_options;
};


} // namespace yaml


Writer* createWriterYAML(const WriterOptionsYAML &options) {
  return new lld::yaml::Writer(options);
}

WriterOptionsYAML::WriterOptionsYAML() {
}

WriterOptionsYAML::~WriterOptionsYAML() {
}


} // namespace lld
