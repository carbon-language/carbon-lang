//===- Core/YamlWriter.cpp - Writes YAML ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "YamlKeyValues.h"

#include "lld/Core/YamlWriter.h"
#include "lld/Core/Atom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"

#include <vector>

namespace lld {
namespace yaml {

namespace {
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
class RefNameBuilder : public File::AtomHandler, 
                       public DefinedAtom::ReferenceHandler {
public:
  RefNameBuilder() { }

  virtual void doReference(const Reference& ref) {
    // create refname for any unnamed reference target
    if ( ref.target()->name().empty() ) {
      char* buffer;
      asprintf(&buffer, "L%03d", _unnamedCounter++);
      _refNames[ref.target()] = buffer;
    }
  }

  virtual void doFile(const File &) { }
  
  virtual void doDefinedAtom(const DefinedAtom& atom) {
    // Build map of atoms names to detect duplicates
    if ( ! atom.name().empty() )
      buildDuplicateNameMap(atom);
    
    // Find references to unnamed atoms and create ref-names for them.
    _unnamedCounter = 0;
    atom.forEachReference(*this);
  }
  
  virtual void doUndefinedAtom(const UndefinedAtom& atom) {
    buildDuplicateNameMap(atom);
  }
  
  virtual void doSharedLibraryAtom(const SharedLibraryAtom& atom) {
    buildDuplicateNameMap(atom);
  }

  virtual void doAbsoluteAtom(const AbsoluteAtom& atom) {
    buildDuplicateNameMap(atom);
  }
                         
  void buildDuplicateNameMap(const Atom& atom) {
    assert(!atom.name().empty());
    NameToAtom::iterator pos = _nameMap.find(atom.name());
    if ( pos != _nameMap.end() ) {
      // Found name collision, give each a unique ref-name.
      char* buffer;
      asprintf(&buffer, "%s.%03d", atom.name().data(), ++_collisionCount);
      _refNames[&atom] = buffer;
      const Atom* prevAtom = pos->second;
      AtomToRefName::iterator pos2 = _refNames.find(prevAtom);
      if ( pos2 == _refNames.end() ) {
        // only create ref-name for previous if none already created
        asprintf(&buffer, "%s.%03d", prevAtom->name().data(), ++_collisionCount);
        _refNames[prevAtom] = buffer;
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
  
  const char* refName(const Atom* atom) {
     return _refNames.find(atom)->second;
  }
  
private:
  struct MyMappingInfo {
    static llvm::StringRef getEmptyKey() { return llvm::StringRef(); }
    static llvm::StringRef getTombstoneKey() { return llvm::StringRef(" ", 0); }
    static unsigned getHashValue(llvm::StringRef const val) {
                                               return llvm::HashString(val); }
    static bool isEqual(llvm::StringRef const lhs, 
                        llvm::StringRef const rhs) { return lhs.equals(rhs); }
  };
  typedef llvm::DenseMap<llvm::StringRef, const Atom*, MyMappingInfo> NameToAtom;
  typedef llvm::DenseMap<const Atom*, const char*> AtomToRefName;
  
  unsigned int      _collisionCount;
  unsigned int      _unnamedCounter;
  NameToAtom        _nameMap;
  AtomToRefName     _refNames;
};


///
/// Helper class for writeObjectText() to write out atoms in yaml format.
///
class AtomWriter : public File::AtomHandler,
                   public DefinedAtom::ReferenceHandler {
public:
  AtomWriter(RefNameBuilder& rnb, llvm::raw_ostream &out) 
    : _out(out), _rnb(rnb), _firstAtom(true) { }

  virtual void doFile(const class File &) { _firstAtom = true; }
  
  virtual void doDefinedAtom(const class DefinedAtom &atom) {
    if ( _firstAtom ) {
      _out << "atoms:\n";
      _firstAtom = false;
    }
    else {
      // add blank line between atoms for readability
      _out << "\n";
    }
    
    bool hasDash = false;
    if ( !atom.name().empty() ) {
      _out  << "    - "
            << KeyValues::nameKeyword
            << ":"
            << spacePadding(KeyValues::nameKeyword)
            << atom.name() 
            << "\n";
      hasDash = true;
    }
     
    if ( _rnb.hasRefName(&atom) ) {
      _out  << (hasDash ? "      " : "    - ")
            << KeyValues::refNameKeyword
            << ":"
            << spacePadding(KeyValues::refNameKeyword)
            << _rnb.refName(&atom) 
            << "\n";
      hasDash = true;
    }
    
    if ( atom.definition() != KeyValues::definitionDefault ) {
      _out  << (hasDash ? "      " : "    - ")
            << KeyValues::definitionKeyword 
            << ":"
            << spacePadding(KeyValues::definitionKeyword)
            << KeyValues::definition(atom.definition()) 
            << "\n";
      hasDash = true;
    }
    
    if ( atom.scope() != KeyValues::scopeDefault ) {
      _out  << (hasDash ? "      " : "    - ")
            << KeyValues::scopeKeyword 
            << ":"
            << spacePadding(KeyValues::scopeKeyword)
            << KeyValues::scope(atom.scope()) 
            << "\n";
      hasDash = true;
    }
    
     if ( atom.interposable() != KeyValues::interposableDefault ) {
      _out  << "      " 
            << KeyValues::interposableKeyword 
            << ":"
            << spacePadding(KeyValues::interposableKeyword)
            << KeyValues::interposable(atom.interposable()) 
            << "\n";
    }
    
    if ( atom.merge() != KeyValues::mergeDefault ) {
      _out  << "      " 
            << KeyValues::mergeKeyword 
            << ":"
            << spacePadding(KeyValues::mergeKeyword)
            << KeyValues::merge(atom.merge()) 
            << "\n";
    }
    
    if ( atom.contentType() != KeyValues::contentTypeDefault ) {
      _out  << "      " 
            << KeyValues::contentTypeKeyword 
            << ":"
            << spacePadding(KeyValues::contentTypeKeyword)
            << KeyValues::contentType(atom.contentType()) 
            << "\n";
    }

    if ( atom.deadStrip() != KeyValues::deadStripKindDefault ) {
      _out  << "      " 
            << KeyValues::deadStripKindKeyword 
            << ":"
            << spacePadding(KeyValues::deadStripKindKeyword)
            << KeyValues::deadStripKind(atom.deadStrip()) 
            << "\n";
    }

    if ( atom.sectionChoice() != KeyValues::sectionChoiceDefault ) {
      _out  << "      " 
            << KeyValues::sectionChoiceKeyword 
            << ":"
            << spacePadding(KeyValues::sectionChoiceKeyword)
            << KeyValues::sectionChoice(atom.sectionChoice()) 
            << "\n";
      assert( ! atom.customSectionName().empty() );
      _out  << "      " 
            << KeyValues::sectionNameKeyword 
            << ":"
            << spacePadding(KeyValues::sectionNameKeyword)
            << atom.customSectionName()
            << "\n";
    }

    if ( atom.isThumb() != KeyValues::isThumbDefault ) {
      _out  << "      " 
            << KeyValues::isThumbKeyword 
            << ":"
            << spacePadding(KeyValues::isThumbKeyword)
            << KeyValues::isThumb(atom.isThumb()) 
            << "\n";
    }

    if ( atom.isAlias() != KeyValues::isAliasDefault ) {
      _out  << "      " 
            << KeyValues::isAliasKeyword 
            << ":"
            << spacePadding(KeyValues::isAliasKeyword)
            << KeyValues::isAlias(atom.isAlias()) 
            << "\n";
    }

    if ( (atom.contentType() != DefinedAtom::typeZeroFill) 
                                   && (atom.size() != 0) ) {
      _out  << "      " 
            << KeyValues::contentKeyword 
            << ":"
            << spacePadding(KeyValues::contentKeyword)
            << "[ ";
      llvm::ArrayRef<uint8_t> arr = atom.rawContent();
      bool needComma = false;
      for (unsigned int i=0; i < arr.size(); ++i) {
        if ( needComma )
          _out << ", ";
        _out << hexdigit(arr[i] >> 4);
        _out << hexdigit(arr[i] & 0x0F);
        needComma = true;
      }
      _out << " ]\n";
    }

    _wroteFirstFixup = false;
    atom.forEachReference(*this);
  }
    
  virtual void doReference(const Reference& ref) {
    if ( !_wroteFirstFixup ) {
      _out << "      fixups:\n";
      _wroteFirstFixup = true;
    }
    _out  << "      - "
          << KeyValues::fixupsOffsetKeyword
          << ":"
          << spacePadding(KeyValues::fixupsOffsetKeyword)
          << ref.offsetInAtom()
          << "\n";
    _out  << "        "
          << KeyValues::fixupsKindKeyword
          << ":"
          << spacePadding(KeyValues::fixupsKindKeyword)
          << ref.kind()
          << "\n";
    const Atom* target = ref.target();
    if ( target != NULL ) {
      llvm::StringRef refName = target->name();
      if ( _rnb.hasRefName(target) )
        refName = _rnb.refName(target);
      assert(!refName.empty());
      _out  << "        "
            << KeyValues::fixupsTargetKeyword
            << ":"
            << spacePadding(KeyValues::fixupsTargetKeyword)
            << refName 
            << "\n";
    }
    if ( ref.addend() != 0 ) {
      _out  << "        "
            << KeyValues::fixupsAddendKeyword
            << ":"
            << spacePadding(KeyValues::fixupsAddendKeyword)
            << ref.addend()
            << "\n";
    }
  }


  virtual void doUndefinedAtom(const class UndefinedAtom &atom) {
    if ( _firstAtom ) {
      _out << "atoms:\n";
      _firstAtom = false;
    }
    else {
      // add blank line between atoms for readability
      _out << "\n";
    }
        
    _out  << "    - "
          << KeyValues::nameKeyword
          << ":"
          << spacePadding(KeyValues::nameKeyword)
          << atom.name() 
          << "\n";

    _out  << "      " 
          << KeyValues::definitionKeyword 
          << ":"
          << spacePadding(KeyValues::definitionKeyword)
          << KeyValues::definition(atom.definition()) 
          << "\n";

    if ( atom.canBeNull() != KeyValues::canBeNullDefault ) {
      _out  << "      " 
            << KeyValues::canBeNullKeyword 
            << ":"
            << spacePadding(KeyValues::canBeNullKeyword)
            << KeyValues::canBeNull(atom.canBeNull()) 
            << "\n";
    }
  }

   virtual void doSharedLibraryAtom(const SharedLibraryAtom& atom) {
    if ( _firstAtom ) {
      _out << "atoms:\n";
      _firstAtom = false;
    }
    else {
      // add blank line between atoms for readability
      _out << "\n";
    }
        
    _out  << "    - "
          << KeyValues::nameKeyword
          << ":"
          << spacePadding(KeyValues::nameKeyword)
          << atom.name() 
          << "\n";

    _out  << "      " 
          << KeyValues::definitionKeyword 
          << ":"
          << spacePadding(KeyValues::definitionKeyword)
          << KeyValues::definition(atom.definition()) 
          << "\n";

    if ( !atom.loadName().empty() ) {
      _out  << "      " 
            << KeyValues::loadNameKeyword 
            << ":"
            << spacePadding(KeyValues::loadNameKeyword)
            << atom.loadName()
            << "\n";
    }

    if ( atom.canBeNullAtRuntime() ) {
      _out  << "      " 
            << KeyValues::canBeNullKeyword 
            << ":"
            << spacePadding(KeyValues::canBeNullKeyword)
            << KeyValues::canBeNull(UndefinedAtom::canBeNullAtRuntime) 
            << "\n";
    }
   }
   
   virtual void doAbsoluteAtom(const AbsoluteAtom& atom) {
     if ( _firstAtom ) {
      _out << "atoms:\n";
      _firstAtom = false;
    }
    else {
      // add blank line between atoms for readability
      _out << "\n";
    }
        
    _out  << "    - "
          << KeyValues::nameKeyword
          << ":"
          << spacePadding(KeyValues::nameKeyword)
          << atom.name() 
          << "\n";

    _out  << "      " 
          << KeyValues::definitionKeyword 
          << ":"
          << spacePadding(KeyValues::definitionKeyword)
          << KeyValues::definition(atom.definition()) 
          << "\n";
    
    _out  << "      " 
          << KeyValues::valueKeyword 
          << ":"
          << spacePadding(KeyValues::valueKeyword)
          << "0x";
     _out.write_hex(atom.value());
     _out << "\n";
   }
                     

private:
  // return a string of the correct number of spaces to align value
  const char* spacePadding(const char* key) {
    const char* spaces = "                  ";
    assert(strlen(spaces) > strlen(key));
    return &spaces[strlen(key)];
  }

  char hexdigit(uint8_t nibble) {
    if ( nibble < 0x0A )
      return '0' + nibble;
    else
      return 'A' + nibble - 0x0A;
  }

  llvm::raw_ostream&  _out;
  RefNameBuilder      _rnb;
  bool                _firstAtom;
  bool                _wroteFirstFixup;
};

} // anonymous namespace



///
/// writeObjectText - writes the lld::File object as in YAML
/// format to the specified stream.
///
void writeObjectText(const File &file, llvm::raw_ostream &out) {
  // Figure what ref-name labels are needed
  RefNameBuilder rnb;
  file.forEachAtom(rnb);
  
  // Write out all atoms
  AtomWriter h(rnb, out);
  out << "---\n";
  file.forEachAtom(h);
  out << "...\n";
}

} // namespace yaml
} // namespace lld
