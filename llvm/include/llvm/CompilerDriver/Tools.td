//===- Tools.td - Tools description for LLVMC2 -------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains descriptions of the various build tools run by llvmc2.
//
//===----------------------------------------------------------------------===//

def OptList : OptionList<[
 (switch_option "emit-llvm",
    (help "Emit LLVM .ll files instead of native object files")),
 (switch_option "E",
    (help "Stop after the preprocessing stage, do not run the compiler")),
 (switch_option "fsyntax-only",
    (help "Stop after checking the input for syntax errors")),
 (switch_option "opt",
    (help "Enable opt")),
 (switch_option "S",
    (help "Stop after compilation, do not assemble")),
 (switch_option "c",
    (help "Compile and assemble, but do not link")),
 (switch_option "pthread",
    (help "Enable threads")),
 (parameter_option "linker",
    (help "Choose linker (possible values: gcc, g++)")),
 (parameter_list_option "include",
    (help "Include the named file prior to preprocessing")),
 (prefix_list_option "I",
    (help "Add a directory to include path")),
 (prefix_list_option "Wa,",
    (help "Pass options to assembler")),
 (prefix_list_option "L",
    (help "Add a directory to link path")),
 (prefix_list_option "l",
    (help "Search a library when linking")),
 (prefix_list_option "Wl,",
    (help "Pass options to linker"))
]>;

class llvm_gcc_based <string cmd_prefix, string in_lang, string E_ext> : Tool<
[(in_language in_lang),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line (case
            (switch_on "E"),
              (case (not_empty "o"),
                    !strconcat(cmd_prefix, " -E $INFILE -o $OUTFILE"),
                    (default),
                    !strconcat(cmd_prefix, " -E $INFILE")),
            (switch_on "fsyntax-only"),
              !strconcat(cmd_prefix, " -fsyntax-only $INFILE"),
            (and (switch_on "S"), (switch_on "emit-llvm")),
              !strconcat(cmd_prefix, " -S $INFILE -o $OUTFILE -emit-llvm"),
            (default),
              !strconcat(cmd_prefix, " -c $INFILE -o $OUTFILE -emit-llvm"))),
 (actions
     (case
         (switch_on "E"), [(stop_compilation), (output_suffix E_ext)],
         (and (switch_on "emit-llvm"), (switch_on "S")),
              [(output_suffix "ll"), (stop_compilation)],
         (and (switch_on "emit-llvm"), (switch_on "c")), (stop_compilation),
         (switch_on "fsyntax-only"), (stop_compilation),
         (not_empty "include"), (forward "include"),
         (not_empty "I"), (forward "I"))),
 (sink)
]>;

def llvm_gcc_c : llvm_gcc_based<"llvm-gcc -x c", "c", "i">;
def llvm_gcc_cpp : llvm_gcc_based<"llvm-g++ -x c++", "c++", "i">;
def llvm_gcc_m : llvm_gcc_based<"llvm-gcc -x objective-c", "objective-c", "mi">;
def llvm_gcc_mxx : llvm_gcc_based<"llvm-gcc -x objective-c++",
                                  "objective-c++", "mi">;

def opt : Tool<
[(in_language "llvm-bitcode"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line "opt -f $INFILE -o $OUTFILE")
]>;

def llvm_as : Tool<
[(in_language "llvm-assembler"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line "llvm-as $INFILE -o $OUTFILE")
]>;

def llvm_gcc_assembler : Tool<
[(in_language "assembler"),
 (out_language "object-code"),
 (output_suffix "o"),
 (cmd_line "llvm-gcc -c -x assembler $INFILE -o $OUTFILE"),
 (actions (case
          (switch_on "c"), (stop_compilation),
          (not_empty "Wa,"), (unpack_values "Wa,")))
]>;

def llc : Tool<
[(in_language "llvm-bitcode"),
 (out_language "assembler"),
 (output_suffix "s"),
 (cmd_line "llc -relocation-model=pic -f $INFILE -o $OUTFILE"),
 (actions (case (switch_on "S"), (stop_compilation)))
]>;

// Base class for linkers
class llvm_gcc_based_linker <string cmd_prefix> : Tool<
[(in_language "object-code"),
 (out_language "executable"),
 (output_suffix "out"),
 (cmd_line !strconcat(cmd_prefix, " $INFILE -o $OUTFILE")),
 (join),
 (actions (case
          (switch_on "pthread"), (append_cmd "-lpthread"),
          (not_empty "L"), (forward "L"),
          (not_empty "l"), (forward "l"),
          (not_empty "Wl,"), (unpack_values "Wl,")))
]>;

// Default linker
def llvm_gcc_linker : llvm_gcc_based_linker<"llvm-gcc">;
// Alternative linker for C++
def llvm_gcc_cpp_linker : llvm_gcc_based_linker<"llvm-g++">;

// Language map

def LanguageMap : LanguageMap<
    [LangToSuffixes<"c++", ["cc", "cp", "cxx", "cpp", "CPP", "c++", "C"]>,
     LangToSuffixes<"c", ["c"]>,
     LangToSuffixes<"c-cpp-output", ["i"]>,
     LangToSuffixes<"objective-c-cpp-output", ["mi"]>,
     LangToSuffixes<"objective-c++", ["mm"]>,
     LangToSuffixes<"objective-c", ["m"]>,
     LangToSuffixes<"assembler", ["s"]>,
     LangToSuffixes<"assembler-with-cpp", ["S"]>,
     LangToSuffixes<"llvm-assembler", ["ll"]>,
     LangToSuffixes<"llvm-bitcode", ["bc"]>,
     LangToSuffixes<"object-code", ["o"]>,
     LangToSuffixes<"executable", ["out"]>
     ]>;
