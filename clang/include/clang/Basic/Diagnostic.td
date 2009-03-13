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

// All diagnostics emitted by the compiler are an indirect subclass of this.
class Diagnostic<string text> {
  string Component = ?;
  string Text = text;
}

class Error<string text> : Diagnostic<text>;
class Note<string text> : Diagnostic<text>;

// Anything that can be controlled by an option subclasses this.
class OptionControlled;

class Warning<string text> : Diagnostic<text>, OptionControlled {
  string DefaultMapping = "warning";
}
// Special cases of warnings.
class Extension<string text> : Warning<text> {
  let DefaultMapping = "ignore";
}
class ExtWarn<string text> : Warning<text>;



class Option<string name, list<OptionControlled> members> : OptionControlled {
  string Name = name;
  list<OptionControlled> Members = members;
}

include "DiagnosticASTKinds.td"
include "DiagnosticAnalysisKinds.td"
include "DiagnosticCommonKinds.td"
include "DiagnosticDriverKinds.td"
include "DiagnosticFrontendKinds.td"
include "DiagnosticLexKinds.td"
include "DiagnosticParseKinds.td"
include "DiagnosticSemaKinds.td"

