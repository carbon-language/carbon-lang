//==--- DiagnosticSerializationKinds.td - serialization diagnostics -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

let Component = "Serialization" in {

def err_fe_unable_to_read_pch_file : Error<
    "unable to read PCH file %0: '%1'">;
def err_fe_not_a_pch_file : Error<
    "input is not a PCH file: '%0'">;
def err_fe_pch_malformed : Error<
    "malformed or corrupted PCH file: '%0'">, DefaultFatal;
def err_fe_pch_malformed_block : Error<
    "malformed block record in PCH file: '%0'">, DefaultFatal;
def err_fe_pch_error_at_end_block : Error<
    "error at end of module block in PCH file: '%0'">, DefaultFatal;
def err_fe_pch_file_modified : Error<
    "file '%0' has been modified since the precompiled header was built">,
    DefaultFatal;
def err_fe_pch_file_overridden : Error<
    "file '%0' from the precompiled header has been overridden">;

def warn_pch_target_triple : Error<
    "PCH file was compiled for the target '%0' but the current translation "
    "unit is being compiled for target '%1'">;
def err_pch_langopt_mismatch : Error<"%0 was %select{disabled|enabled}1 in "
    "PCH file but is currently %select{disabled|enabled}2">;
def err_pch_langopt_value_mismatch : Error<
  "%0 differs in PCH file vs. current file">;
  
def warn_pch_version_too_old : Error<
    "PCH file uses an older PCH format that is no longer supported">;
def warn_pch_version_too_new : Error<
    "PCH file uses a newer PCH format that cannot be read">;
def warn_pch_different_branch : Error<
    "PCH file built from a different branch (%0) than the compiler (%1)">;
def err_pch_with_compiler_errors : Error<
    "PCH file contains compiler errors">;
def warn_cmdline_conflicting_macro_def : Error<
    "definition of the macro '%0' conflicts with the definition used to "
    "build the precompiled header">;
def note_pch_macro_defined_as : Note<
    "definition of macro '%0' in the precompiled header">;
def warn_cmdline_missing_macro_defs : Warning<
    "macro definitions used to build the precompiled header are missing">;
def note_using_macro_def_from_pch : Note<
    "using this macro definition from precompiled header">;
def warn_macro_name_used_in_pch : Error<
    "definition of macro %0 conflicts with an identifier used in the "
    "precompiled header">;
def warn_pch_compiler_options_mismatch : Error<
    "compiler options used when building the precompiled header differ from "
    "the options used when using the precompiled header">;

def err_not_a_pch_file : Error<
    "'%0' does not appear to be a precompiled header file">, DefaultFatal;
}
