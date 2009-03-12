//==--- DiagnosticDriverKinds.td - libdriver diagnostics ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

let Component = "Driver" in {

def err_drv_no_such_file : Error<"no such file or directory: '%0'">
def err_drv_unsupported_opt : Error<"unsupported option '%0'">
def err_drv_unknown_stdin_type : Error<
    "-E or -x required when input is from standard input">
def err_drv_unknown_language : Error<"language not recognized: '%0'">

}
