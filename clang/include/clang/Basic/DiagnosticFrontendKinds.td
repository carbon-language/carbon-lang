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

// PCH reader
def warn_pch_target_triple : Warning<
    "PCH file was compiled for the target '%0' but the current translation "
    "unit is being compiled for target '%1'">;
def note_ignoring_pch : Note<
    "ignoring precompiled header '%0'">;
def warn_pch_c99 : Warning<
    "C99 support was %select{disabled|enabled}0 in PCH file but is "
    "currently %select{disabled|enabled}1">;
def warn_pch_cplusplus : Warning<
    "C++ support was %select{disabled|enabled}0 in PCH file but is "
    "currently %select{disabled|enabled}1">;
def warn_pch_cplusplus0x : Warning<
    "C++0x support was %select{disabled|enabled}0 in PCH file but is "
    "currently %select{disabled|enabled}1">;
def warn_pch_objective_c : Warning<
    "Objective-C support was %select{disabled|enabled}0 in PCH file but is "
    "currently %select{disabled|enabled}1">;
def warn_pch_objective_c2 : Warning<
    "Objective-C 2.0 support was %select{disabled|enabled}0 in PCH file but "
    "is currently %select{disabled|enabled}1">;
def warn_pch_nonfragile_abi : Warning<
    "PCH file was compiled with the %select{32-bit|non-fragile}0 Objective-C "
    "ABI but the %select{32-bit|non-fragile}1 Objective-C ABI is selected">;
def warn_pch_extensions : Warning<
    "extensions were %select{enabled|disabled}0 in PCH file but are "
    "currently %select{enabled|disabled}1">;
def warn_pch_gnu_extensions : Warning<
    "GNU extensions were %select{disabled|enabled}0 in PCH file but are "
    "currently %select{disabled|enabled}1">;
def warn_pch_microsoft_extensions : Warning<
    "Microsoft extensions were %select{disabled|enabled}0 in PCH file but are "
    "currently %select{disabled|enabled}1">;
def warn_pch_heinous_extensions : Warning<
    "heinous extensions were %select{disabled|enabled}0 in PCH file but are "
    "currently %select{disabled|enabled}1">;
def warn_pch_lax_vector_conversions : Warning<
    "lax vector conversions were %select{disabled|enabled}0 in PCH file but "
    "are currently %select{disabled|enabled}1">;
def warn_pch_exceptions : Warning<
    "exceptions were %select{disabled|enabled}0 in PCH file but "
    "are currently %select{disabled|enabled}1">;
def warn_pch_objc_runtime : Warning<
    "PCH file was compiled with the %select{NeXT|GNU}0 runtime but the "
    "%select{NeXT|GNU}1 runtime is selected">;
def warn_pch_freestanding : Warning<
    "PCH file was compiled with a %select{hosted|freestanding}0  "
    "implementation but a %select{hosted|freestanding}1 implementation "
    "is selected">;
def warn_pch_builtins : Warning<
    "PCH file was compiled with builtins %select{enabled|disabled}0 but "
    "builtins are currently %select{enabled|disabled}1">;
def warn_pch_thread_safe_statics : Warning<
    "PCH file was compiled %select{without|with}0 thread-safe statics but"
    "thread-safe statics are currently %select{disabled|enabled}1">;
def warn_pch_blocks : Warning<
    "blocks were %select{disabled|enabled}0 in PCH file but "
    "are currently %select{disabled|enabled}1">;
def warn_pch_math_errno : Warning<
    "math functions %select{do not respect|respect}0 'errno' in PCH "
    "file but they are currently set to %select{not respect|respect}1 "
    "'errno'">;
def warn_pch_overflow_checking : Warning<
    "signed integer overflow checking was %select{disabled|enabled}0 in PCH "
    "file but is currently %select{disabled|enabled}1">;
def warn_pch_optimize : Warning<
    "the macro '__OPTIMIZE__' was %select{not defined|defined}0 in "
    "the PCH file but is currently %select{undefined|defined}1">;
def warn_pch_optimize_size : Warning<
    "the macro '__OPTIMIZE_SIZE__' was %select{not defined|defined}0 in "
    "the PCH file but is currently %select{undefined|defined}1">;
def warn_pch_static : Warning<
    "the PCH file was compiled %select{dynamic|static}0 but the "
    "current translation unit is being compiled as %select{dynamic|static}1">;
def warn_pch_pic_level : Warning<
    "PCH file was compiled with PIC level %0, but the current translation "
    "unit will be compiled with PIC level %1">;
def warn_pch_gnu_inline : Warning<
    "PCH file was compiled with %select{C99|GNU|}0 inline semantics but "
    "%select{C99|GNU}1 inline semantics are currently selected">;
def warn_pch_no_inline : Warning<
    "the macro '__NO_INLINE__' was %select{not defined|defined}0 in "
    "the PCH file but is currently %select{undefined|defined}1">;
def warn_pch_gc_mode : Warning<
    "the PCH file was built with %select{no||hybrid}0 garbage collection but "
    "the current translation unit will compiled with %select{no||hybrid}1 "
    "garbage collection">;
def warn_pch_version_too_old : Warning<
    "PCH file uses an older PCH format that is no longer supported">;
def warn_pch_version_too_new : Warning<
    "PCH file uses a newer PCH format that cannot be read">;
def warn_cmdline_conflicting_macro_def : Warning<
    "definition of the macro '%0' conflicts with the definition used to "
    "build the precompiled header">;
def note_pch_macro_defined_as : Note<
    "definition of macro '%0' in the precompiled header">;
def warn_cmdline_missing_macro_defs : Warning<
    "macro definitions used to build the precompiled header are missing">;
def note_using_macro_def_from_pch : Note<
    "using this macro definition from precompiled header">;

}
