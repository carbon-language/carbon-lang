//===- Core/YamlKeyValues.cpp - Reads YAML --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string.h>

#include "YamlKeyValues.h"

#include "llvm/Support/ErrorHandling.h"

namespace lld {
namespace yaml {


const char* const KeyValues::nameKeyword            = "name";
const char* const KeyValues::definitionKeyword      = "definition";
const char* const KeyValues::scopeKeyword           = "scope";
const char* const KeyValues::contentTypeKeyword     = "type";
const char* const KeyValues::deadStripKindKeyword   = "dead-strip";
const char* const KeyValues::sectionChoiceKeyword   = "section-choice";
const char* const KeyValues::internalNameKeyword    = "internal-name";
const char* const KeyValues::interposableKeyword    = "interposable";
const char* const KeyValues::mergeKeyword           = "merge";
const char* const KeyValues::isThumbKeyword         = "is-thumb";
const char* const KeyValues::isAliasKeyword         = "is-alias";
const char* const KeyValues::sectionNameKeyword     = "section-name";
const char* const KeyValues::contentKeyword         = "content";
const char* const KeyValues::sizeKeyword            = "size";
const char* const KeyValues::permissionsKeyword      = "permissions";


const DefinedAtom::Definition         KeyValues::definitionDefault = Atom::definitionRegular;
const DefinedAtom::Scope              KeyValues::scopeDefault = DefinedAtom::scopeTranslationUnit;
const DefinedAtom::ContentType        KeyValues::contentTypeDefault = DefinedAtom::typeData;
const DefinedAtom::DeadStripKind      KeyValues::deadStripKindDefault = DefinedAtom::deadStripNormal;
const DefinedAtom::SectionChoice      KeyValues::sectionChoiceDefault = DefinedAtom::sectionBasedOnContent;
const DefinedAtom::Interposable       KeyValues::interposableDefault = DefinedAtom::interposeNo;
const DefinedAtom::Merge              KeyValues::mergeDefault = DefinedAtom::mergeNo;
const DefinedAtom::ContentPermissions KeyValues::permissionsDefault = DefinedAtom::permR__;
const bool                            KeyValues::internalNameDefault = false;
const bool                            KeyValues::isThumbDefault = false;
const bool                            KeyValues::isAliasDefault = false;





struct DefinitionMapping {
	const char*       string;
	Atom::Definition  value;
};

static const DefinitionMapping defMappings[] = {
	{ "regular",        Atom::definitionRegular },
	{ "absolute",       Atom::definitionAbsolute },
	{ "undefined",      Atom::definitionUndefined },
	{ "shared-library", Atom::definitionSharedLibrary },
  { NULL,             Atom::definitionRegular }
};

Atom::Definition KeyValues::definition(const char* s)
{
	for (const DefinitionMapping* p = defMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad definition value");
}

const char* KeyValues::definition(Atom::Definition s) {
	for (const DefinitionMapping* p = defMappings; p->string != NULL; ++p) {
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
	{ "global", DefinedAtom::scopeGlobal },
	{ "hidden", DefinedAtom::scopeLinkageUnit },
	{ "static", DefinedAtom::scopeTranslationUnit },
  { NULL,     DefinedAtom::scopeGlobal }
};
  
DefinedAtom::Scope KeyValues::scope(const char* s)
{
	for (const ScopeMapping* p = scopeMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad scope value");
}

const char* KeyValues::scope(DefinedAtom::Scope s) {
	for (const ScopeMapping* p = scopeMappings; p->string != NULL; ++p) {
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
  { NULL,             DefinedAtom::typeUnknown }
};

DefinedAtom::ContentType KeyValues::contentType(const char* s)
{
	for (const ContentTypeMapping* p = typeMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad content type value");
}

const char* KeyValues::contentType(DefinedAtom::ContentType s) {
	for (const ContentTypeMapping* p = typeMappings; p->string != NULL; ++p) {
    if ( p->value == s )
      return p->string;
  }
  llvm::report_fatal_error("bad content type value");
}







struct DeadStripMapping {
	const char*           string;
	DefinedAtom::DeadStripKind   value;
};

static const DeadStripMapping deadStripMappings[] = {
	{ "normal",         DefinedAtom::deadStripNormal },
	{ "never",          DefinedAtom::deadStripNever },
	{ "always",         DefinedAtom::deadStripAlways },
  { NULL,             DefinedAtom::deadStripNormal }
};

DefinedAtom::DeadStripKind KeyValues::deadStripKind(const char* s)
{
	for (const DeadStripMapping* p = deadStripMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad dead strip value");
}

const char* KeyValues::deadStripKind(DefinedAtom::DeadStripKind dsk) {
	for (const DeadStripMapping* p = deadStripMappings; p->string != NULL; ++p) {
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
  { NULL,           DefinedAtom::interposeNo }
};

DefinedAtom::Interposable KeyValues::interposable(const char* s)
{
	for (const InterposableMapping* p = interMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad interposable value");
}

const char* KeyValues::interposable(DefinedAtom::Interposable in) {
	for (const InterposableMapping* p = interMappings; p->string != NULL; ++p) {
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
  { NULL,             DefinedAtom::mergeNo }
};

DefinedAtom::Merge KeyValues::merge(const char* s)
{
	for (const MergeMapping* p = mergeMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad merge value");
}

const char* KeyValues::merge(DefinedAtom::Merge in) {
	for (const MergeMapping* p = mergeMappings; p->string != NULL; ++p) {
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
  { NULL,              DefinedAtom::sectionBasedOnContent }
};

DefinedAtom::SectionChoice KeyValues::sectionChoice(const char* s)
{
	for (const SectionChoiceMapping* p = sectMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad dead strip value");
}

const char* KeyValues::sectionChoice(DefinedAtom::SectionChoice s) {
	for (const SectionChoiceMapping* p = sectMappings; p->string != NULL; ++p) {
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
	{ "content",         DefinedAtom::perm___ },
	{ "custom",          DefinedAtom::permR__ },
	{ "custom-required", DefinedAtom::permR_X },
	{ "custom-required", DefinedAtom::permRW_ },
	{ "custom-required", DefinedAtom::permRW_L },
  { NULL,              DefinedAtom::perm___ }
};

DefinedAtom::ContentPermissions KeyValues::permissions(const char* s)
{
	for (const PermissionsMapping* p = permMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad permissions value");
}

const char* KeyValues::permissions(DefinedAtom::ContentPermissions s) {
	for (const PermissionsMapping* p = permMappings; p->string != NULL; ++p) {
    if ( p->value == s )
      return p->string;
  }
  llvm::report_fatal_error("bad permissions value");
}







bool KeyValues::internalName(const char* s)
{
  if ( strcmp(s, "true") == 0 )
    return true;
  else if ( strcmp(s, "false") == 0 )
    return false;
  llvm::report_fatal_error("bad internal-name value");
}

const char* KeyValues::internalName(bool b) {
  return b ? "true" : "false";
}




bool KeyValues::isThumb(const char* s)
{
  if ( strcmp(s, "true") == 0 )
    return true;
  else if ( strcmp(s, "false") == 0 )
    return false;
  llvm::report_fatal_error("bad is-thumb value");
}

const char* KeyValues::isThumb(bool b) {
  return b ? "true" : "false";
}




bool KeyValues::isAlias(const char* s)
{
  if ( strcmp(s, "true") == 0 )
    return true;
  else if ( strcmp(s, "false") == 0 )
    return false;
  llvm::report_fatal_error("bad is-alias value");
}

const char* KeyValues::isAlias(bool b) {
  return b ? "true" : "false";
}






} // namespace yaml
} // namespace lld
