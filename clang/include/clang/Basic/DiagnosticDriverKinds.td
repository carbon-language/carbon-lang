//==--- DiagnosticDriverKinds.td - libdriver diagnostics ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

let Component = "Driver" in {

def err_drv_no_such_file : Error<"no such file or directory: '%0'">;
def err_drv_unsupported_opt : Error<"unsupported option '%0'">;
def err_drv_unknown_stdin_type : Error<
  "-E or -x required when input is from standard input">;
def err_drv_unknown_language : Error<"language not recognized: '%0'">;
def err_drv_invalid_opt_with_multiple_archs : Error<
  "option '%0' cannot be used with multiple -arch options">;
def err_drv_invalid_output_with_multiple_archs : Error<
  "cannot use '%0' output with multiple -arch options">;
def err_drv_no_input_files : Error<"no input files">;
def err_drv_use_of_Z_option : Error<
  "unsupported use of internal gcc -Z option '%0'">;

def warn_drv_input_file_unused : Warning<
  "%0: '%1' input file unused when '%2' is present">;

}
