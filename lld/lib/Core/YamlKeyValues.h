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
#include "lld/Core/UndefinedAtom.h"
#include "lld/Core/File.h"

namespace lld {
namespace yaml {

class KeyValues {
public:
  static const char* const                nameKeyword;
  static const char* const                refNameKeyword;
  static const char* const                sectionNameKeyword;
  static const char* const                contentKeyword;
  static const char* const                sizeKeyword;
  static const char* const                loadNameKeyword;
  static const char* const                valueKeyword;
  static const char* const                fixupsKeyword;
  static const char* const                fileAtomsKeyword;
  static const char* const                fileMembersKeyword;

  static const char* const                fileKindKeyword;
  static const File::Kind                 fileKindDefault;
  static File::Kind                       fileKind(StringRef);
  static const char*                      fileKind(File::Kind);

  static const char* const                definitionKeyword;
  static const Atom::Definition           definitionDefault;
  static Atom::Definition                 definition(StringRef);
  static const char*                      definition(Atom::Definition);

  static const char* const                scopeKeyword;
  static const DefinedAtom::Scope         scopeDefault;
  static DefinedAtom::Scope               scope(StringRef);
  static const char*                      scope(DefinedAtom::Scope);

  static const char* const                contentTypeKeyword;
  static const DefinedAtom::ContentType   contentTypeDefault;
  static DefinedAtom::ContentType         contentType(StringRef);
  static const char*                      contentType(DefinedAtom::ContentType);

  static const char* const                deadStripKindKeyword;
  static const DefinedAtom::DeadStripKind deadStripKindDefault;
  static DefinedAtom::DeadStripKind       deadStripKind(StringRef);
  static const char*                      deadStripKind(DefinedAtom::DeadStripKind);

  static const char* const                sectionChoiceKeyword;
  static const DefinedAtom::SectionChoice sectionChoiceDefault;
  static DefinedAtom::SectionChoice       sectionChoice(StringRef);
  static const char*                      sectionChoice(DefinedAtom::SectionChoice);

  static const char* const                interposableKeyword;
  static const DefinedAtom::Interposable  interposableDefault;
  static DefinedAtom::Interposable        interposable(StringRef);
  static const char*                      interposable(DefinedAtom::Interposable);

  static const char* const                mergeKeyword;
  static const DefinedAtom::Merge         mergeDefault;
  static DefinedAtom::Merge               merge(StringRef);
  static const char*                      merge(DefinedAtom::Merge);

  static const char* const                      permissionsKeyword;
  static const DefinedAtom::ContentPermissions  permissionsDefault;
  static DefinedAtom::ContentPermissions        permissions(StringRef);
  static const char*                            permissions(DefinedAtom::ContentPermissions);

  static const char* const                isThumbKeyword;
  static const bool                       isThumbDefault;
  static const char*                      isThumb(bool);

  static const char* const                isAliasKeyword;
  static const bool                       isAliasDefault;
  static const char*                      isAlias(bool);

  static const char* const                canBeNullKeyword;
  static const UndefinedAtom::CanBeNull   canBeNullDefault;
  static UndefinedAtom::CanBeNull         canBeNull(StringRef);
  static const char*                      canBeNull(UndefinedAtom::CanBeNull);


  static const char* const                fixupsKindKeyword;
  static const char* const                fixupsOffsetKeyword;
  static const char* const                fixupsTargetKeyword;
  static const char* const                fixupsAddendKeyword;

};

} // namespace yaml
} // namespace lld

#endif // LLD_CORE_YAML_KEY_VALUES_H_
