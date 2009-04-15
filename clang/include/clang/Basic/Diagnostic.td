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

/*
class Option<string name, list<OptionControlled> members> : OptionControlled {
  string Name = name;
  list<OptionControlled> Members = members;
}

// Definitions for options ("warning groups").
include "DiagnosticOptions.td"

*/

// All diagnostics emitted by the compiler are an indirect subclass of this.
class Diagnostic<string text, DiagMapping defaultmapping> {
  /// Component is specified by the file with a big let directive.
  string Component = ?;
  string Text = text;
  DiagMapping DefaultMapping = defaultmapping;
}

class Error<string text>     : Diagnostic<text, MAP_ERROR>;
class Warning<string text>   : Diagnostic<text, MAP_WARNING>;
class Extension<string text> : Diagnostic<text, MAP_IGNORE>;
class ExtWarn<string text>   : Diagnostic<text, MAP_WARNING>;
class Note<string text>      : Diagnostic<text, MAP_FATAL /*ignored*/>;

// Definitions for Diagnostics.
include "DiagnosticASTKinds.td"
include "DiagnosticAnalysisKinds.td"
include "DiagnosticCommonKinds.td"
include "DiagnosticDriverKinds.td"
include "DiagnosticFrontendKinds.td"
include "DiagnosticLexKinds.td"
include "DiagnosticParseKinds.td"
include "DiagnosticSemaKinds.td"

