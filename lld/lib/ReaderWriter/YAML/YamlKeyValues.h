//===- lib/ReaderWriter/YAML/YamlKeyValues.h ------------------------------===//
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
#include "lld/Core/UndefinedAtom.h"
#include "lld/Core/File.h"

namespace lld {
namespace yaml {

class KeyValues {
public:

  static const char* const                definitionKeyword;
  static const Atom::Definition           definitionDefault;
  static bool                             definition(StringRef, Atom::Definition&);
  static const char*                      definition(Atom::Definition);

  static const char* const                scopeKeyword;
  static const DefinedAtom::Scope         scopeDefault;
  static bool                             scope(StringRef, DefinedAtom::Scope&);
  static const char*                      scope(Atom::Scope);

  static const char* const                contentTypeKeyword;
  static const DefinedAtom::ContentType   contentTypeDefault;
  static bool                             contentType(StringRef, DefinedAtom::ContentType&);
  static const char*                      contentType(DefinedAtom::ContentType);

  static const char* const                deadStripKindKeyword;
  static const DefinedAtom::DeadStripKind deadStripKindDefault;
  static bool                             deadStripKind(StringRef, DefinedAtom::DeadStripKind&);
  static const char*                      deadStripKind(DefinedAtom::DeadStripKind);

  static const char* const                sectionChoiceKeyword;
  static const DefinedAtom::SectionChoice sectionChoiceDefault;
  static bool                             sectionChoice(StringRef,  DefinedAtom::SectionChoice&);
  static const char*                      sectionChoice(DefinedAtom::SectionChoice);

  static const char* const                interposableKeyword;
  static const DefinedAtom::Interposable  interposableDefault;
  static bool                             interposable(StringRef, DefinedAtom::Interposable&);
  static const char*                      interposable(DefinedAtom::Interposable);

  static const char* const                mergeKeyword;
  static const DefinedAtom::Merge         mergeDefault;
  static bool                             merge(StringRef, DefinedAtom::Merge&);
  static const char*                      merge(DefinedAtom::Merge);

  static const char* const                      permissionsKeyword;
  static const DefinedAtom::ContentPermissions  permissionsDefault;
  static bool                                   permissions(StringRef, DefinedAtom::ContentPermissions&);
  static const char*                            permissions(DefinedAtom::ContentPermissions);

  static const char* const                isThumbKeyword;
  static const bool                       isThumbDefault;
  static bool                             isThumb(StringRef, bool&);
  static const char*                      isThumb(bool);

  static const char* const                isAliasKeyword;
  static const bool                       isAliasDefault;
  static bool                             isAlias(StringRef, bool&);
  static const char*                      isAlias(bool);

  static const char* const                canBeNullKeyword;
  static const UndefinedAtom::CanBeNull   canBeNullDefault;
  static bool                             canBeNull(StringRef, UndefinedAtom::CanBeNull&);
  static const char*                      canBeNull(UndefinedAtom::CanBeNull);

};

} // namespace yaml
} // namespace lld

#endif // LLD_CORE_YAML_KEY_VALUES_H_
