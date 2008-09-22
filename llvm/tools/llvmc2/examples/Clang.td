// A (first stab at a) replacement for the Clang's ccc script.
// To compile, use this command:
//    make TOOLNAME=ccc2 GRAPH=examples/Clang.td

include "Common.td"


// TOFIX: Add an explicit option list for aliases and things like this.
def Options : OptionList<[
(switch_option "E",
    (help "Stop after the preprocessing stage, do not run the compiler"))
]>;

class clang_base<string language, dag cmdline> : Tool<
[(in_language language),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line cmdline),
 (switch_option "E", (stop_compilation), (output_suffix "i")),
 (sink)
]>;

def clang_c : clang_base<"c",
(case
(switch_on "E"),
    (case
    (not_empty "o"),
        "clang -E -x c $INFILE -o $OUTFILE",
    (default),
        "clang -E -x c $INFILE"),
(default),
    "clang -emit-llvm-bc -x c $INFILE -o $OUTFILE")>;

def clang_cpp : clang_base<"c++",
(case
(switch_on "E"),
    (case
    (not_empty "o"),
        "clang -E -x c++ $INFILE -o $OUTFILE",
    (default),
        "clang -E -x c++ $INFILE"),
(default),
    "clang -emit-llvm-bc -x c++ $INFILE -o $OUTFILE")>;

def clang_objective_c : clang_base<"objective-c",
(case
(switch_on "E"),
    (case
    (not_empty "o"),
        "clang -E -x objective-c $INFILE -o $OUTFILE",
    (default),
        "clang -E -x objective-c $INFILE"),
(default),
    "clang -emit-llvm-bc -x objective-c $INFILE -o $OUTFILE")>;

// Default linker
def llvm_ld : Tool<
[(in_language "llvm-bitcode"),
 (out_language "executable"),
 (output_suffix "out"),
 (cmd_line "llvm-ld -native -disable-internalize $INFILE -o $OUTFILE"),
 (prefix_list_option "L", (forward), (help "Specify a library search path")),
 (join)
]>;

// Language map

def LanguageMap : LanguageMap<
    [LangToSuffixes<"c++", ["cc", "cp", "cxx", "cpp", "CPP", "c++", "C"]>,
     LangToSuffixes<"c", ["c"]>,
     LangToSuffixes<"objective-c", ["m"]>,
     LangToSuffixes<"c-cpp-output", ["i"]>,
     LangToSuffixes<"objective-c-cpp-output", ["mi"]>
     ]>;

// Compilation graph

def CompilationGraph : CompilationGraph<[
    Edge<root, clang_c>,
    Edge<root, clang_cpp>,
    Edge<root, clang_objective_c>,
    Edge<clang_c, llvm_ld>,
    Edge<clang_cpp, llvm_ld>,
    Edge<clang_objective_c, llvm_ld>
    ]>;

