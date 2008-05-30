//===- Tools.td - Tools description for the LLVMCC  --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains descriptions of the various build tools run by llvmcc.
//
//===----------------------------------------------------------------------===//

// TOTHINK: Open issue: should we use DAG lists in Tool specifications
// or change to something like

// def LLVMGccC : < Tool<
// [ InLanguage<"c">,
//   PrefixListOption<"Wl", [UnpackValues, PropertyName<Arg>, ...]>
//  ...] ?

// DAG lists look more aesthetically pleasing to me.

def llvm_gcc_c : Tool<
[(in_language "c"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line (case
            (switch_on "E"),
              "llvm-g++ -E -x c $INFILE -o $OUTFILE",
            (default),
              "llvm-g++ -c -x c $INFILE -o $OUTFILE -emit-llvm")),
 (switch_option "E", (stop_compilation),(output_suffix "i"),
   (help "Stop after the preprocessing stage, do not run the compiler")),
 (sink)
]>;

def llvm_gcc_cpp : Tool<
[(in_language "c++"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line (case
            (switch_on "E"),
              "llvm-g++ -E -x c++ $INFILE -o $OUTFILE",
            (default),
              "llvm-g++ -c -x c++ $INFILE -o $OUTFILE -emit-llvm")),
 (switch_option "E", (stop_compilation), (output_suffix "i")),
 (sink)
]>;

def opt : Tool<
[(in_language "llvm-bitcode"),
 (out_language "llvm-bitcode"),
 (switch_option "opt", (help "Enable opt")),
 (output_suffix "bc"),
 (cmd_line "opt $INFILE -o $OUTFILE")
]>;

def llvm_as : Tool<
[(in_language "llvm-assembler"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line "llvm-as $INFILE -o $OUTFILE")
]>;

def llc : Tool<
[(in_language "llvm-bitcode"),
 (out_language "assembler"),
 (output_suffix "s"),
 (switch_option "S", (stop_compilation),
                (help "Stop after compilation, do not assemble")),
 (cmd_line "llc -f $INFILE -o $OUTFILE")
]>;

def llvm_gcc_assembler : Tool<
[(in_language "assembler"),
 (out_language "object-code"),
 (output_suffix "o"),
 (cmd_line "llvm-gcc -c -x assembler $INFILE -o $OUTFILE"),
 (switch_option "c", (stop_compilation),
                (help "Compile and assemble, but do not link")),
 (prefix_list_option "Wa,", (unpack_values), (help "pass options to assembler"))
]>;

// Default linker
def llvm_gcc_linker : Tool<
[(in_language "object-code"),
 (out_language "executable"),
 (output_suffix "out"),
 (cmd_line "llvm-gcc $INFILE -o $OUTFILE"),
 (join),
 (prefix_list_option "L", (forward), (help "add a directory to link path")),
 (prefix_list_option "l", (forward), (help "search a library when linking")),
 (prefix_list_option "Wl,", (unpack_values), (help "pass options to linker"))
]>;

// Alternative linker for C++
def llvm_gcc_cpp_linker : Tool<
[(in_language "object-code"),
 (out_language "executable"),
 (output_suffix "out"),
 (cmd_line "llvm-g++ $INFILE -o $OUTFILE"),
 (join),
 (parameter_option "linker",
                   (help "Choose linker (possible values: gcc, g++)")),
 (prefix_list_option "L", (forward)),
 (prefix_list_option "l", (forward)),
 (prefix_list_option "Wl,", (unpack_values))
]>;

// Language map

def LanguageMap : LanguageMap<
    [LangToSuffixes<"c++", ["cc", "cp", "cxx", "cpp", "CPP", "c++", "C"]>,
     LangToSuffixes<"c", ["c"]>,
     LangToSuffixes<"assembler", ["s"]>,
     LangToSuffixes<"llvm-assembler", ["ll"]>,
     LangToSuffixes<"llvm-bitcode", ["bc"]>,
     LangToSuffixes<"object-code", ["o"]>,
     LangToSuffixes<"executable", ["out"]>
     ]>;
