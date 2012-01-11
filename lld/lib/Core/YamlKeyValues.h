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
#include "lld/Core/DefinedAtom.h"


namespace lld {
namespace yaml {

class KeyValues {
public:
  static const char* const                nameKeyword;
  static const char* const                sectionNameKeyword;
  static const char* const                contentKeyword;
  static const char* const                sizeKeyword;
  
  static const char* const                definitionKeyword;
  static const Atom::Definition           definitionDefault;
  static Atom::Definition                 definition(const char*);
  static const char*                      definition(Atom::Definition);

  static const char* const                scopeKeyword;
  static const DefinedAtom::Scope         scopeDefault;
  static DefinedAtom::Scope               scope(const char*);
  static const char*                      scope(DefinedAtom::Scope);
  
  static const char* const                contentTypeKeyword;
  static const DefinedAtom::ContentType   contentTypeDefault;
  static DefinedAtom::ContentType         contentType(const char*);
  static const char*                      contentType(DefinedAtom::ContentType);

  static const char* const                deadStripKindKeyword;
  static const DefinedAtom::DeadStripKind deadStripKindDefault;
  static DefinedAtom::DeadStripKind       deadStripKind(const char*);
  static const char*                      deadStripKind(DefinedAtom::DeadStripKind);

  static const char* const                sectionChoiceKeyword;
  static const DefinedAtom::SectionChoice sectionChoiceDefault;
  static DefinedAtom::SectionChoice       sectionChoice(const char*);
  static const char*                      sectionChoice(DefinedAtom::SectionChoice);

  static const char* const                interposableKeyword;
  static const DefinedAtom::Interposable  interposableDefault;
  static DefinedAtom::Interposable        interposable(const char*);
  static const char*                      interposable(DefinedAtom::Interposable);

  static const char* const                mergeKeyword;
  static const DefinedAtom::Merge         mergeDefault;
  static DefinedAtom::Merge               merge(const char*);
  static const char*                      merge(DefinedAtom::Merge);

  static const char* const                      permissionsKeyword;
  static const DefinedAtom::ContentPermissions  permissionsDefault;
  static DefinedAtom::ContentPermissions        permissions(const char*);
  static const char*                            permissions(DefinedAtom::ContentPermissions);

  static const char* const                internalNameKeyword;
  static const bool                       internalNameDefault;
  static bool                             internalName(const char*);
  static const char*                      internalName(bool);

  static const char* const                isThumbKeyword;
  static const bool                       isThumbDefault;
  static bool                             isThumb(const char*);
  static const char*                      isThumb(bool);

  static const char* const                isAliasKeyword;
  static const bool                       isAliasDefault;
  static bool                             isAlias(const char*);
  static const char*                      isAlias(bool);

};

} // namespace yaml
} // namespace lld

#endif // LLD_CORE_YAML_KEY_VALUES_H_

