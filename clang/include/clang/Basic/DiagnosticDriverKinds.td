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
def err_drv_output_argument_with_multiple_files : Error<
  "cannot specify -o when generating multiple output files">;
def err_drv_unable_to_make_temp : Error<
  "unable to make temporary file: %0">;
def err_drv_unable_to_remove_file : Error<
  "unable to remove file: %0">;
def err_drv_command_failure : Error<
  "unable to execute command: %0">;
def err_drv_invalid_darwin_version : Error<
  "invalid Darwin version number: %0">;
def err_drv_missing_argument : Error<
  "argument to '%0' is missing (expected %1 %plural{1:value|:values}1)">;
def err_drv_invalid_Xarch_argument : Error<
  "invalid Xarch argument: '%0'">;
def err_drv_argument_only_allowed_with : Error<
  "invalid argument '%0' only allowed with '%1'">;
def err_drv_argument_not_allowed_with : Error<
  "invalid argument '%0' not allowed with '%1'">;
def err_drv_invalid_version_number : Error<
  "invalid version number in '%0'">;
def err_drv_no_linker_llvm_support : Error<
  "'%0': unable to pass LLVM bit-code files to linker">;

def warn_drv_input_file_unused : Warning<
  "%0: '%1' input unused when '%2' is present">;
def warn_drv_unused_argument : Warning<
  "argument unused during compilation: '%0'">;
def warn_drv_pipe_ignored_with_save_temps : Warning<
  "-pipe ignored because -save-temps specified">;
def warn_drv_not_using_clang_cpp : Warning<
  "not using the clang prepreprocessor due to user override">;
def warn_drv_not_using_clang_cxx : Warning<
  "not using the clang compiler for C++ inputs">;
def warn_drv_not_using_clang_arch : Warning<
  "not using the clang compiler for the '%0' architecture">;
def warn_drv_clang_unsupported : Warning<
  "the clang compiler does not yet support '%0'">;

}
