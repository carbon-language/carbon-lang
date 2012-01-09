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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"

#include <vector>

namespace lld {
namespace yaml {

class Handler : public File::AtomHandler {
public:
  Handler(llvm::raw_ostream &out) : _out(out), _firstAtom(true) { }

  virtual void doFile(const class File &) { _firstAtom = true; }
  
  virtual void doAtom(const class Atom &atom) {
      // add blank line between atoms for readability
      if ( !_firstAtom )
        _out << "\n";
      _firstAtom = false;
        
      _out  << "    - "
            << KeyValues::nameKeyword
            << ":"
            << spacePadding(KeyValues::nameKeyword)
            << atom.name() 
            << "\n";
    
    if ( atom.internalName() != KeyValues::internalNameDefault ) {
      _out  << "      " 
            << KeyValues::internalNameKeyword 
            << ":"
            << spacePadding(KeyValues::internalNameKeyword)
            << KeyValues::internalName(atom.internalName()) 
            << "\n";
    }
    
    if ( atom.definition() != KeyValues::definitionDefault ) {
      _out  << "      " 
            << KeyValues::definitionKeyword 
            << ":"
            << spacePadding(KeyValues::definitionKeyword)
            << KeyValues::definition(atom.definition()) 
            << "\n";
    }
    
    if ( atom.scope() != KeyValues::scopeDefault ) {
      _out  << "      " 
            << KeyValues::scopeKeyword 
            << ":"
            << spacePadding(KeyValues::scopeKeyword)
            << KeyValues::scope(atom.scope()) 
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

    if ( atom.mergeDuplicates() != KeyValues::mergeDuplicatesDefault ) {
      _out  << "      " 
            << KeyValues::mergeDuplicatesKeyword 
            << ":"
            << spacePadding(KeyValues::mergeDuplicatesKeyword)
            << KeyValues::mergeDuplicates(atom.mergeDuplicates()) 
            << "\n";
    }

    if ( atom.autoHide() != KeyValues::autoHideDefault ) {
      _out  << "      " 
            << KeyValues::autoHideKeyword 
            << ":"
            << spacePadding(KeyValues::autoHideKeyword)
            << KeyValues::autoHide(atom.autoHide()) 
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

     
    if ( atom.contentType() != Atom::typeZeroFill ) {
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

    if (atom.referencesBegin() != atom.referencesEnd()) {
      _out << "      fixups:\n";
      for (Reference::iterator it = atom.referencesBegin(),
           end = atom.referencesEnd(); it != end; ++it) {
        _out << "      - kind:      " << it->kind << "\n";
        _out << "        offset:    " << it->offsetInAtom << "\n";
      }
    }

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
  bool                _firstAtom;
};

void writeObjectText(File &file, llvm::raw_ostream &out) {
  Handler h(out);
  out << "---\n";
  out << "atoms:\n";
  file.forEachAtom(h);
  out << "...\n";
}

} // namespace yaml
} // namespace lld
