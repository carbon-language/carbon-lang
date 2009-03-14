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

}
