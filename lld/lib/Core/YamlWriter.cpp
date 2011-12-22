//===- Core/YamlWriter.cpp - Writes YAML ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/YamlWriter.h"
#include "lld/Core/Atom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"

#include <vector>

namespace lld {
namespace yaml {

class Handler : public File::AtomHandler {
public:
  Handler(llvm::raw_ostream &out) : _out(out) { }

  virtual void doFile(const class File &) { }
  virtual void doAtom(const class Atom &atom) {
    _out << "    - name:             " << atom.name() << "\n";
    
    if ( atom.internalName() )
      _out << "      internal-name:     true\n";
      
    if ( atom.definition() != Atom::definitionRegular )
      _out << "      definition:       " << definitionString(atom.definition()) <<"\n";
      
    if ( atom.scope() != Atom::scopeTranslationUnit )
      _out << "      scope:            " << scopeString(atom.scope()) << "\n";
      
    _out << "      type:             " << typeString(atom.contentType()) << "\n";
    
    if ( atom.mergeDuplicates() )
      _out << "      merge-duplicates: true\n";
      
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
  const char *scopeString(Atom::Scope scope) {
    switch (scope) {
    case Atom::scopeTranslationUnit:
      return "static";
    case Atom::scopeLinkageUnit:
      return "hidden";
    case Atom::scopeGlobal:
      return "global";
    }
    return "???";
  }

  const char *typeString(Atom::ContentType type) {
    switch (type) {
    case Atom::typeCode:
      return "code";
    case Atom::typeCString:
      return "c-string";
    case Atom::typeZeroFill:
      return "zero-fill";
    case Atom::typeData:
      return "data";
    default:
      return "???";
    }
  }

  const char *definitionString(Atom::Definition def) {
    switch (def) {
    case Atom::definitionRegular:
      return "regular";
    case Atom::definitionWeak:
      return "weak";
    case Atom::definitionTentative:
      return "tentative";
    case Atom::definitionAbsolute:
      return "absolute";
    default:
      return "???";
    }
  }

  llvm::raw_ostream &_out;
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
