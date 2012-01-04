//===- Core/YamlKeyValues.h - Reads YAML ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_YAML_KEY_VALUES_H_
#define LLD_CORE_YAML_KEY_VALUES_H_

#include "lld/Core/Atom.h"


namespace lld {
namespace yaml {

class KeyValues {
public:
  static const char* const          nameKeyword;
  static const char* const          sectionNameKeyword;
  static const char* const          contentKeyword;
  static const char* const          sizeKeyword;
  

  static const char* const          scopeKeyword;
  static const Atom::Scope          scopeDefault;
  static Atom::Scope                scope(const char*);
  static const char*                scope(Atom::Scope);
  
  static const char* const          definitionKeyword;
  static const Atom::Definition     definitionDefault;
  static Atom::Definition           definition(const char*);
  static const char*                definition(Atom::Definition);

  static const char* const          contentTypeKeyword;
  static const Atom::ContentType    contentTypeDefault;
  static Atom::ContentType          contentType(const char*);
  static const char*                contentType(Atom::ContentType);

  static const char* const          deadStripKindKeyword;
  static const Atom::DeadStripKind  deadStripKindDefault;
  static Atom::DeadStripKind        deadStripKind(const char*);
  static const char*                deadStripKind(Atom::DeadStripKind);

  static const char* const          sectionChoiceKeyword;
  static const Atom::SectionChoice  sectionChoiceDefault;
  static Atom::SectionChoice        sectionChoice(const char*);
  static const char*                sectionChoice(Atom::SectionChoice);

  static const char* const          internalNameKeyword;
  static const bool                 internalNameDefault;
  static bool                       internalName(const char*);
  static const char*                internalName(bool);

  static const char* const          mergeDuplicatesKeyword;
  static const bool                 mergeDuplicatesDefault;
  static bool                       mergeDuplicates(const char*);
  static const char*                mergeDuplicates(bool);

  static const char* const          autoHideKeyword;
  static const bool                 autoHideDefault;
  static bool                       autoHide(const char*);
  static const char*                autoHide(bool);

  static const char* const          isThumbKeyword;
  static const bool                 isThumbDefault;
  static bool                       isThumb(const char*);
  static const char*                isThumb(bool);

  static const char* const          isAliasKeyword;
  static const bool                 isAliasDefault;
  static bool                       isAlias(const char*);
  static const char*                isAlias(bool);

};

} // namespace yaml
} // namespace lld

#endif // LLD_CORE_YAML_KEY_VALUES_H_

