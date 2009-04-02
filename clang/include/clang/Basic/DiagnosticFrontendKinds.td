//==--- DiagnosticFrontendKinds.td - frontend diagnostics -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

let Component = "Frontend" in {

def err_fe_unknown_triple : Error<
  "unknown target triple '%0', please use -triple or -arch">;
def err_fe_error_reading : Error<"error reading '%0'">;
def err_fe_error_reading_stdin : Error<"error reading stdin">;

def note_fixit_applied : Note<"FIX-IT applied suggested code changes">;
def note_fixit_in_macro : Note<
    "FIX-IT unable to apply suggested code changes in a macro">;
def note_fixit_failed : Note<
    "FIX-IT unable to apply suggested code changes">;
def note_fixit_unfixed_error : Note<"FIX-IT detected an error it cannot fix">;
def warn_fixit_no_changes : Note<
    "FIX-IT detected errors it could not fix; no output will be generated">;
}
