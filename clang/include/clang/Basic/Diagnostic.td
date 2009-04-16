//===--- Diagnostic.td - C Language Family Diagnostic Handling ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TableGen core definitions for the diagnostics
//  and diagnostic control.
//
//===----------------------------------------------------------------------===//

// Define the diagnostic mappings.
class DiagMapping;
def MAP_IGNORE  : DiagMapping;
def MAP_WARNING : DiagMapping;
def MAP_ERROR   : DiagMapping;
def MAP_FATAL   : DiagMapping;

// Define the diagnostic classes.
class DiagClass;
def CLASS_NOTE      : DiagClass;
def CLASS_WARNING   : DiagClass;
def CLASS_EXTENSION : DiagClass;
def CLASS_ERROR     : DiagClass;

// Diagnostic Groups.
class DiagGroup<string Name, list<DiagGroup> subgroups = []> {
  string GroupName = Name;
  list<DiagGroup> SubGroups = subgroups;
}
class InGroup<DiagGroup G> { DiagGroup Group = G; }
//class IsGroup<string Name> { DiagGroup Group = DiagGroup<Name>; }

def ImplicitFunctionDeclare : DiagGroup<"implicit-function-declaration">;
def Trigraphs : DiagGroup<"trigraphs">;

// Empty DiagGroups: these are recognized by clang but ignored.
def : DiagGroup<"aggregate-return">;
def : DiagGroup<"bad-function-cast">;
def : DiagGroup<"cast-align">;
def : DiagGroup<"cast-qual">;
def : DiagGroup<"char-align">;
def : DiagGroup<"char-subscripts">;
def : DiagGroup<"declaration-after-statement">;
def : DiagGroup<"error-implicit-function-declaration">;
def : DiagGroup<"format-security">;
def : DiagGroup<"format=2">;
def : DiagGroup<"format">;
def : DiagGroup<"four-char-constants">;
def : DiagGroup<"init-self">;
def : DiagGroup<"inline">;
def : DiagGroup<"int-to-pointer-cast">;
def : DiagGroup<"missing-braces">;
def : DiagGroup<"missing-declarations">;
def : DiagGroup<"missing-format-attribute">;
def : DiagGroup<"nested-externs">;
def : DiagGroup<"newline-eof">;
def : DiagGroup<"no-#warnings">;
def : DiagGroup<"no-comment">;
def : DiagGroup<"format-y2k">;
def : DiagGroup<"long-long">;
def : DiagGroup<"missing-field-initializers">;
def : DiagGroup<"nonportable-cfstrings">;
def : DiagGroup<"old-style-definition">;
def : DiagGroup<"packed">;
def : DiagGroup<"parentheses">;
def : DiagGroup<"pointer-arith">;
def : DiagGroup<"pointer-to-int-cast">;
def : DiagGroup<"redundant-decls">;
def : DiagGroup<"return-type">;
def : DiagGroup<"shadow">;
def : DiagGroup<"shorten-64-to-32">;
def : DiagGroup<"sign-compare">;
def : DiagGroup<"strict-overflow=">;
def : DiagGroup<"strict-overflow">;
def : DiagGroup<"strict-prototypes">;
def : DiagGroup<"strict-selector-match">;
def Switch         : DiagGroup<"switch">;
def Uninitialized  : DiagGroup<"uninitialized">;
def UnknownPragmas : DiagGroup<"unknown-pragmas">;
def : DiagGroup<"unused-function">;
def : DiagGroup<"unused-label">;
def : DiagGroup<"unused-parameter">;
def UnusedValue    : DiagGroup<"unused-value">;
def UnusedVariable : DiagGroup<"unused-variable">;
def : DiagGroup<"variadic-macros">;
def VolatileRegisterVar : DiagGroup<"volatile-register-var">;
def : DiagGroup<"write-strings">;

def : DiagGroup<"extra">;
def : DiagGroup<"">;  // -W
def : DiagGroup<"most">;

// Aggregation warning settings.
def : DiagGroup<"all", [
    ImplicitFunctionDeclare,
    Switch,
    Trigraphs,
    Uninitialized,
    UnknownPragmas,
    UnusedValue,
    UnusedVariable,
    VolatileRegisterVar
 ]>;


// All diagnostics emitted by the compiler are an indirect subclass of this.
class Diagnostic<string text, DiagClass DC, DiagMapping defaultmapping> {
  /// Component is specified by the file with a big let directive.
  string      Component = ?;
  string      Text = text;
  DiagClass   Class = DC;
  DiagMapping DefaultMapping = defaultmapping;
  DiagGroup   Group;
}

class Error<string str>     : Diagnostic<str, CLASS_ERROR, MAP_ERROR>;
class Warning<string str>   : Diagnostic<str, CLASS_WARNING, MAP_WARNING>;
class Extension<string str> : Diagnostic<str, CLASS_EXTENSION, MAP_IGNORE>;
class ExtWarn<string str>   : Diagnostic<str, CLASS_EXTENSION, MAP_WARNING>;
class Note<string str>      : Diagnostic<str, CLASS_NOTE, MAP_FATAL/*ignored*/>;


class DefaultIgnore { DiagMapping DefaultMapping = MAP_IGNORE; }
class DefaultWarn   { DiagMapping DefaultMapping = MAP_WARNING; }
class DefaultError  { DiagMapping DefaultMapping = MAP_ERROR; }
class DefaultFatal  { DiagMapping DefaultMapping = MAP_FATAL; }

// Definitions for Diagnostics.
include "DiagnosticASTKinds.td"
include "DiagnosticAnalysisKinds.td"
include "DiagnosticCommonKinds.td"
include "DiagnosticDriverKinds.td"
include "DiagnosticFrontendKinds.td"
include "DiagnosticLexKinds.td"
include "DiagnosticParseKinds.td"
include "DiagnosticSemaKinds.td"

