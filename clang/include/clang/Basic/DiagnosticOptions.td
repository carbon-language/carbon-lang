//===--- DiagnosticOptions.td - C Language Family Diagnostic Handling -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TableGen definitions for the diagnostics options.
//
//===----------------------------------------------------------------------===//

def UnusedMacrosDiags : Option<"unused-macros", [pp_macro_not_used]>;
