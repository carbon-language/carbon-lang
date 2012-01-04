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
const char* const KeyValues::scopeKeyword           = "scope";
const char* const KeyValues::definitionKeyword      = "definition";
const char* const KeyValues::contentTypeKeyword     = "type";
const char* const KeyValues::deadStripKindKeyword   = "dead-strip";
const char* const KeyValues::sectionChoiceKeyword   = "section-choice";
const char* const KeyValues::internalNameKeyword    = "internal-name";
const char* const KeyValues::mergeDuplicatesKeyword = "merge-duplicates";
const char* const KeyValues::autoHideKeyword        = "auto-hide";
const char* const KeyValues::isThumbKeyword         = "is-thumb";
const char* const KeyValues::isAliasKeyword         = "is-alias";
const char* const KeyValues::sectionNameKeyword     = "section-name";
const char* const KeyValues::contentKeyword         = "content";
const char* const KeyValues::sizeKeyword            = "size";


const Atom::Scope         KeyValues::scopeDefault = Atom::scopeTranslationUnit;
const Atom::Definition    KeyValues::definitionDefault = Atom::definitionRegular;
const Atom::ContentType   KeyValues::contentTypeDefault = Atom::typeData;
const Atom::DeadStripKind KeyValues::deadStripKindDefault = Atom::deadStripNormal;
const Atom::SectionChoice KeyValues::sectionChoiceDefault = Atom::sectionBasedOnContent;
const bool                KeyValues::internalNameDefault = false;
const bool                KeyValues::mergeDuplicatesDefault = false;
const bool                KeyValues::autoHideDefault = false;
const bool                KeyValues::isThumbDefault = false;
const bool                KeyValues::isAliasDefault = false;


struct ScopeMapping {
	const char* string;
	Atom::Scope value;
};

static const ScopeMapping scopeMappings[] = {
	{ "global", Atom::scopeGlobal },
	{ "hidden", Atom::scopeLinkageUnit },
	{ "static", Atom::scopeTranslationUnit },
  { NULL,     Atom::scopeGlobal }
};
  
Atom::Scope KeyValues::scope(const char* s)
{
	for (const ScopeMapping* p = scopeMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad scope value");
}

const char* KeyValues::scope(Atom::Scope s) {
	for (const ScopeMapping* p = scopeMappings; p->string != NULL; ++p) {
    if ( p->value == s )
      return p->string;
  }
  llvm::report_fatal_error("bad scope value");
}





struct DefinitionMapping {
	const char*       string;
	Atom::Definition  value;
};

static const DefinitionMapping defMappings[] = {
	{ "regular",        Atom::definitionRegular },
	{ "weak",           Atom::definitionWeak },
	{ "tentative",      Atom::definitionTentative },
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





struct ContentTypeMapping {
	const char*       string;
	Atom::ContentType  value;
};

static const ContentTypeMapping typeMappings[] = {
	{ "unknown",        Atom::typeUnknown },
	{ "code",           Atom::typeCode },
	{ "resolver",       Atom::typeResolver },
	{ "constant",       Atom::typeConstant },
	{ "c-string",       Atom::typeCString },
	{ "utf16-string",   Atom::typeUTF16String },
	{ "CFI",            Atom::typeCFI },
	{ "LSDA",           Atom::typeLSDA },
	{ "literal-4",      Atom::typeLiteral4 },
	{ "literal-8",      Atom::typeLiteral8 },
	{ "literal-16",     Atom::typeLiteral16 },
	{ "data",           Atom::typeData },
	{ "zero-fill",      Atom::typeZeroFill },
	{ "cf-string",      Atom::typeCFString },
	{ "initializer-ptr",Atom::typeInitializerPtr },
	{ "terminator-ptr", Atom::typeTerminatorPtr },
	{ "c-string-ptr",   Atom::typeCStringPtr },
	{ "objc1-class",    Atom::typeObjC1Class },
	{ "objc1-class-ptr",Atom::typeObjCClassPtr },
	{ "objc2-cat-ptr",  Atom::typeObjC2CategoryList },
	{ "tlv-thunk",      Atom::typeThunkTLV },
	{ "tlv-data",       Atom::typeTLVInitialData },
	{ "tlv-zero-fill",  Atom::typeTLVInitialZeroFill },
	{ "tlv-init-ptr",   Atom::typeTLVInitializerPtr },
  { NULL,             Atom::typeUnknown }
};

Atom::ContentType KeyValues::contentType(const char* s)
{
	for (const ContentTypeMapping* p = typeMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad content type value");
}

const char* KeyValues::contentType(Atom::ContentType s) {
	for (const ContentTypeMapping* p = typeMappings; p->string != NULL; ++p) {
    if ( p->value == s )
      return p->string;
  }
  llvm::report_fatal_error("bad content type value");
}







struct DeadStripMapping {
	const char*           string;
	Atom::DeadStripKind   value;
};

static const DeadStripMapping deadStripMappings[] = {
	{ "normal",         Atom::deadStripNormal },
	{ "never",          Atom::deadStripNever },
	{ "always",         Atom::deadStripAlways },
  { NULL,             Atom::deadStripNormal }
};

Atom::DeadStripKind KeyValues::deadStripKind(const char* s)
{
	for (const DeadStripMapping* p = deadStripMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad dead strip value");
}

const char* KeyValues::deadStripKind(Atom::DeadStripKind dsk) {
	for (const DeadStripMapping* p = deadStripMappings; p->string != NULL; ++p) {
    if ( p->value == dsk )
      return p->string;
  }
  llvm::report_fatal_error("bad dead strip value");
}






struct SectionChoiceMapping {
	const char*           string;
	Atom::SectionChoice   value;
};

static const SectionChoiceMapping sectMappings[] = {
	{ "content",         Atom::sectionBasedOnContent },
	{ "custom",          Atom::sectionCustomPreferred },
	{ "custom-required", Atom::sectionCustomRequired },
  { NULL,              Atom::sectionBasedOnContent }
};

Atom::SectionChoice KeyValues::sectionChoice(const char* s)
{
	for (const SectionChoiceMapping* p = sectMappings; p->string != NULL; ++p) {
    if ( strcmp(p->string, s) == 0 )
      return p->value;
  }
  llvm::report_fatal_error("bad dead strip value");
}

const char* KeyValues::sectionChoice(Atom::SectionChoice s) {
	for (const SectionChoiceMapping* p = sectMappings; p->string != NULL; ++p) {
    if ( p->value == s )
      return p->string;
  }
  llvm::report_fatal_error("bad dead strip value");
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






bool KeyValues::mergeDuplicates(const char* s)
{
  if ( strcmp(s, "true") == 0 )
    return true;
  else if ( strcmp(s, "false") == 0 )
    return false;
  llvm::report_fatal_error("bad merge-duplicates value");
}

const char* KeyValues::mergeDuplicates(bool b) {
  return b ? "true" : "false";
}






bool KeyValues::autoHide(const char* s)
{
  if ( strcmp(s, "true") == 0 )
    return true;
  else if ( strcmp(s, "false") == 0 )
    return false;
  llvm::report_fatal_error("bad auto-hide value");
}

const char* KeyValues::autoHide(bool b) {
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
