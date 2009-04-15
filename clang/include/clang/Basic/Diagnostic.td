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


/*
class Option<string name, list<OptionControlled> members> : OptionControlled {
  string Name = name;
  list<OptionControlled> Members = members;
}

// Definitions for options ("warning groups").
include "DiagnosticOptions.td"

*/

// All diagnostics emitted by the compiler are an indirect subclass of this.
class Diagnostic<string text, DiagClass DC, DiagMapping defaultmapping> {
  /// Component is specified by the file with a big let directive.
  string Component = ?;
  string Text = text;
  DiagClass Class = DC;
  DiagMapping DefaultMapping = defaultmapping;
}

class Error<string str>     : Diagnostic<str, CLASS_ERROR, MAP_ERROR>;
class Warning<string str>   : Diagnostic<str, CLASS_WARNING, MAP_WARNING>;
class Extension<string str> : Diagnostic<str, CLASS_EXTENSION, MAP_IGNORE>;
class ExtWarn<string str>   : Diagnostic<str, CLASS_EXTENSION, MAP_WARNING>;
class Note<string str>      : Diagnostic<str, CLASS_NOTE, MAP_FATAL/*ignored*/>;

// Definitions for Diagnostics.
include "DiagnosticASTKinds.td"
include "DiagnosticAnalysisKinds.td"
include "DiagnosticCommonKinds.td"
include "DiagnosticDriverKinds.td"
include "DiagnosticFrontendKinds.td"
include "DiagnosticLexKinds.td"
include "DiagnosticParseKinds.td"
include "DiagnosticSemaKinds.td"

