//===- lib/ReaderWriter/YAML/YamlKeyValues.cpp ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "YamlKeyValues.h"

#include "llvm/Support/ErrorHandling.h"
#include "lld/Core/File.h"

#include <cstring>

namespace lld {
namespace yaml {


const DefinedAtom::Definition         KeyValues::definitionDefault = Atom::definitionRegular;
const DefinedAtom::Scope              KeyValues::scopeDefault = DefinedAtom::scopeTranslationUnit;
const DefinedAtom::ContentType        KeyValues::contentTypeDefault = DefinedAtom::typeData;
const DefinedAtom::DeadStripKind      KeyValues::deadStripKindDefault = DefinedAtom::deadStripNormal;
const DefinedAtom::SectionChoice      KeyValues::sectionChoiceDefault = DefinedAtom::sectionBasedOnContent;
const DefinedAtom::Interposable       KeyValues::interposableDefault = DefinedAtom::interposeNo;
const DefinedAtom::Merge              KeyValues::mergeDefault = DefinedAtom::mergeNo;
const DefinedAtom::ContentPermissions KeyValues::permissionsDefault = DefinedAtom::permR__;
const bool                            KeyValues::isThumbDefault = false;
const bool                            KeyValues::isAliasDefault = false;
const UndefinedAtom::CanBeNull        KeyValues::canBeNullDefault = UndefinedAtom::canBeNullNever;




struct DefinitionMapping {
  const char*       string;
  Atom::Definition  value;
};

static const DefinitionMapping defMappings[] = {
  { "regular",        Atom::definitionRegular },
  { "absolute",       Atom::definitionAbsolute },
  { "undefined",      Atom::definitionUndefined },
  { "shared-library", Atom::definitionSharedLibrary },
  { nullptr,          Atom::definitionRegular }
};

bool KeyValues::definition(StringRef s, Atom::Definition &out)
{
  for (const DefinitionMapping* p = defMappings; p->string != nullptr; ++p) {
    if (s == p->string) {
      out = p->value;
      return false;
    }
  }
  return true;
}

const char* KeyValues::definition(Atom::Definition s) {
  for (const DefinitionMapping* p = defMappings; p->string != nullptr; ++p) {
    if ( p->value == s )
      return p->string;
  }
  llvm::report_fatal_error("bad definition value");
}





struct ScopeMapping {
  const char* string;
  DefinedAtom::Scope value;
};

static const ScopeMapping scopeMappings[] = {
  { "global", Atom::scopeGlobal },
  { "hidden", Atom::scopeLinkageUnit },
  { "static", Atom::scopeTranslationUnit },
  { nullptr,  Atom::scopeGlobal }
};

bool KeyValues::scope(StringRef s, DefinedAtom::Scope &out)
{
  for (const ScopeMapping* p = scopeMappings; p->string != nullptr; ++p) {
    if (s == p->string) {
      out = p->value;
      return false;
    }
  }
  return true;
}

const char* KeyValues::scope(Atom::Scope s) {
  for (const ScopeMapping* p = scopeMappings; p->string != nullptr; ++p) {
    if ( p->value == s )
      return p->string;
  }
  llvm::report_fatal_error("bad scope value");
}







struct ContentTypeMapping {
  const char*       string;
  DefinedAtom::ContentType  value;
};

static const ContentTypeMapping typeMappings[] = {
  { "unknown",        DefinedAtom::typeUnknown },
  { "code",           DefinedAtom::typeCode },
  { "stub",           DefinedAtom::typeStub },
  { "stub-helper",    DefinedAtom::typeStubHelper },
  { "resolver",       DefinedAtom::typeResolver },
  { "constant",       DefinedAtom::typeConstant },
  { "c-string",       DefinedAtom::typeCString },
  { "utf16-string",   DefinedAtom::typeUTF16String },
  { "CFI",            DefinedAtom::typeCFI },
  { "LSDA",           DefinedAtom::typeLSDA },
  { "literal-4",      DefinedAtom::typeLiteral4 },
  { "literal-8",      DefinedAtom::typeLiteral8 },
  { "literal-16",     DefinedAtom::typeLiteral16 },
  { "data",           DefinedAtom::typeData },
  { "zero-fill",      DefinedAtom::typeZeroFill },
  { "cf-string",      DefinedAtom::typeCFString },
  { "got",            DefinedAtom::typeGOT },
  { "lazy-pointer",   DefinedAtom::typeLazyPointer },
  { "initializer-ptr",DefinedAtom::typeInitializerPtr },
  { "terminator-ptr", DefinedAtom::typeTerminatorPtr },
  { "c-string-ptr",   DefinedAtom::typeCStringPtr },
  { "objc1-class",    DefinedAtom::typeObjC1Class },
  { "objc1-class-ptr",DefinedAtom::typeObjCClassPtr },
  { "objc2-cat-ptr",  DefinedAtom::typeObjC2CategoryList },
  { "tlv-thunk",      DefinedAtom::typeThunkTLV },
  { "tlv-data",       DefinedAtom::typeTLVInitialData },
  { "tlv-zero-fill",  DefinedAtom::typeTLVInitialZeroFill },
  { "tlv-init-ptr",   DefinedAtom::typeTLVInitializerPtr },
  { nullptr,          DefinedAtom::typeUnknown }
};

bool KeyValues::contentType(StringRef s, DefinedAtom::ContentType &out)
{
  for (const ContentTypeMapping* p = typeMappings; p->string != nullptr; ++p) {
    if (s == p->string) {
      out = p->value;
      return false;
    }
  }
  return true;
}

const char* KeyValues::contentType(DefinedAtom::ContentType s) {
  for (const ContentTypeMapping* p = typeMappings; p->string != nullptr; ++p) {
    if ( p->value == s )
      return p->string;
  }
  llvm::report_fatal_error("bad content type value");
}







struct DeadStripMapping {
  const char*           string;
  DefinedAtom::DeadStripKind   value;
};

static const DeadStripMapping dsMappings[] = {
  { "normal",         DefinedAtom::deadStripNormal },
  { "never",          DefinedAtom::deadStripNever },
  { "always",         DefinedAtom::deadStripAlways },
  { nullptr,          DefinedAtom::deadStripNormal }
};

bool KeyValues::deadStripKind(StringRef s, DefinedAtom::DeadStripKind &out)
{
  for (const DeadStripMapping* p = dsMappings; p->string != nullptr; ++p) {
    if (s == p->string) {
      out = p->value;
      return false;
    }
  }
  return true;
}

const char* KeyValues::deadStripKind(DefinedAtom::DeadStripKind dsk) {
  for (const DeadStripMapping* p = dsMappings; p->string != nullptr; ++p) {
    if ( p->value == dsk )
      return p->string;
  }
  llvm::report_fatal_error("bad dead strip value");
}





struct InterposableMapping {
  const char*           string;
  DefinedAtom::Interposable   value;
};

static const InterposableMapping interMappings[] = {
  { "no",           DefinedAtom::interposeNo },
  { "yes",          DefinedAtom::interposeYes },
  { "yesAndWeak",   DefinedAtom::interposeYesAndRuntimeWeak },
  { nullptr,        DefinedAtom::interposeNo }
};

bool KeyValues::interposable(StringRef s, DefinedAtom::Interposable &out)
{
  for (const InterposableMapping* p = interMappings; p->string != nullptr; ++p){
    if (s == p->string) {
      out = p->value;
      return false;
    }
  }
  return true;
}

const char* KeyValues::interposable(DefinedAtom::Interposable in) {
  for (const InterposableMapping* p = interMappings; p->string != nullptr; ++p){
    if ( p->value == in )
      return p->string;
  }
  llvm::report_fatal_error("bad interposable value");
}






struct MergeMapping {
  const char*          string;
  DefinedAtom::Merge   value;
};

static const MergeMapping mergeMappings[] = {
  { "no",             DefinedAtom::mergeNo },
  { "asTentative",    DefinedAtom::mergeAsTentative },
  { "asWeak",         DefinedAtom::mergeAsWeak },
  { "asAddressedWeak",DefinedAtom::mergeAsWeakAndAddressUsed },
  { nullptr,          DefinedAtom::mergeNo }
};

bool KeyValues::merge(StringRef s, DefinedAtom::Merge& out)
{
  for (const MergeMapping* p = mergeMappings; p->string != nullptr; ++p) {
    if (s == p->string) {
      out = p->value;
      return false;
    }
  }
  return true;
}

const char* KeyValues::merge(DefinedAtom::Merge in) {
  for (const MergeMapping* p = mergeMappings; p->string != nullptr; ++p) {
    if ( p->value == in )
      return p->string;
  }
  llvm::report_fatal_error("bad merge value");
}






struct SectionChoiceMapping {
  const char*                 string;
  DefinedAtom::SectionChoice  value;
};

static const SectionChoiceMapping sectMappings[] = {
  { "content",         DefinedAtom::sectionBasedOnContent },
  { "custom",          DefinedAtom::sectionCustomPreferred },
  { "custom-required", DefinedAtom::sectionCustomRequired },
  { nullptr,           DefinedAtom::sectionBasedOnContent }
};

bool KeyValues::sectionChoice(StringRef s, DefinedAtom::SectionChoice &out)
{
  for (const SectionChoiceMapping* p = sectMappings; p->string != nullptr; ++p){
    if (s == p->string) {
      out = p->value;
      return false;
    }
  }
  return true;
}

const char* KeyValues::sectionChoice(DefinedAtom::SectionChoice s) {
  for (const SectionChoiceMapping* p = sectMappings; p->string != nullptr; ++p){
    if ( p->value == s )
      return p->string;
  }
  llvm::report_fatal_error("bad dead strip value");
}







struct PermissionsMapping {
  const char*                      string;
  DefinedAtom::ContentPermissions  value;
};

static const PermissionsMapping permMappings[] = {
  { "---",    DefinedAtom::perm___  },
  { "r--",    DefinedAtom::permR__  },
  { "r-x",    DefinedAtom::permR_X  },
  { "rw-",    DefinedAtom::permRW_  },
  { "rw-l",   DefinedAtom::permRW_L },
  { nullptr,  DefinedAtom::perm___  }
};

bool KeyValues::permissions(StringRef s, DefinedAtom::ContentPermissions &out)
{
  for (const PermissionsMapping* p = permMappings; p->string != nullptr; ++p) {
    if (s == p->string) {
      out = p->value;
      return false;
    }
  }
  return true;
}

const char* KeyValues::permissions(DefinedAtom::ContentPermissions s) {
  for (const PermissionsMapping* p = permMappings; p->string != nullptr; ++p) {
    if ( p->value == s )
      return p->string;
  }
  llvm::report_fatal_error("bad permissions value");
}


bool KeyValues::isThumb(StringRef s, bool &out)
{
  if ( s.equals("true") ) {
    out = true;
    return false;
  }
  
  if ( s.equals("false") ) {
    out = false;
    return false;
  }

  return true;
}

const char* KeyValues::isThumb(bool b) {
  return b ? "true" : "false";
}


bool KeyValues::isAlias(StringRef s, bool &out)
{
  if ( s.equals("true") ) {
    out = true;
    return false;
  }
  
  if ( s.equals("false") ) {
    out = false;
    return false;
  }

  return true;
}

const char* KeyValues::isAlias(bool b) {
  return b ? "true" : "false";
}




struct CanBeNullMapping {
  const char*               string;
  UndefinedAtom::CanBeNull  value;
};

static const CanBeNullMapping cbnMappings[] = {
  { "never",         UndefinedAtom::canBeNullNever },
  { "at-runtime",    UndefinedAtom::canBeNullAtRuntime },
  { "at-buildtime",  UndefinedAtom::canBeNullAtBuildtime },
  { nullptr,         UndefinedAtom::canBeNullNever }
};


bool KeyValues::canBeNull(StringRef s, UndefinedAtom::CanBeNull &out)
{
  for (const CanBeNullMapping* p = cbnMappings; p->string != nullptr; ++p) {
    if (s == p->string) {
      out = p->value;
      return false;
    }
  }
  return true;
}

const char* KeyValues::canBeNull(UndefinedAtom::CanBeNull c) {
  for (const CanBeNullMapping* p = cbnMappings; p->string != nullptr; ++p) {
    if ( p->value == c )
      return p->string;
  }
  llvm::report_fatal_error("bad can-be-null value");
}







} // namespace yaml
} // namespace lld
