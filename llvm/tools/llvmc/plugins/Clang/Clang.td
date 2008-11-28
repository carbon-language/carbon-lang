// A (first stab at a) replacement for the Clang's ccc script.
// To compile, use this command:
//    cd $LLVMC2_DIR
//    make DRIVER_NAME=ccc2 BUILTIN_PLUGINS=Clang

include "llvm/CompilerDriver/Common.td"

def Options : OptionList<[
(switch_option "E",
    (help "Stop after the preprocessing stage, do not run the compiler"))
]>;

class clang_based<string language, string cmd> : Tool<
[(in_language language),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line (case
           (switch_on "E"),
           (case
                (not_empty "o"),
                !strconcat(cmd, " -E $INFILE -o $OUTFILE"),
                (default),
                !strconcat(cmd, " -E $INFILE")),
          (default),
          !strconcat(cmd, " -emit-llvm-bc $INFILE -o $OUTFILE"))),
 (switch_option "E", (stop_compilation), (output_suffix "i")),
 (sink)
]>;

def clang_c : clang_based<"c", "clang -x c">;
def clang_cpp : clang_based<"c++", "clang -x c++">;
def clang_objective_c : clang_based<"objective-c", "clang -x objective-c">;
def clang_objective_cpp : clang_based<"objective-c++",
    "clang -x objective-c++">;

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
    Edge<"root", "clang_c">,
    Edge<"root", "clang_cpp">,
    Edge<"root", "clang_objective_c">,
    Edge<"clang_c", "llvm_ld">,
    Edge<"clang_cpp", "llvm_ld">,
    Edge<"clang_objective_c", "llvm_ld">
    ]>;
