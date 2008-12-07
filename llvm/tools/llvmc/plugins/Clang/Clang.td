// A replacement for the Clang's ccc script.
// Depends on the Base plugin.
// To compile, use this command:
//    cd $LLVMC2_DIR
//    make DRIVER_NAME=ccc2 BUILTIN_PLUGINS=Clang

include "llvm/CompilerDriver/Common.td"

def Options : OptionList<[
(extern_switch "E",
    (help "Stop after the preprocessing stage, do not run the compiler")),
(extern_list "L", (help "Specify a library search path")),
(switch_option "clang", (help "Use Clang instead of llvm-gcc"))
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
 (actions (case (switch_on "E"),
                [(stop_compilation), (output_suffix "i")])),
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
 (actions (case (not_empty "L"), (forward "L"))),
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
    OptionalEdge<"root", "clang_c",
                         (case (switch_on "clang"), (inc_weight))>,
    OptionalEdge<"root", "clang_cpp",
                         (case (switch_on "clang"), (inc_weight))>,
    OptionalEdge<"root", "clang_objective_c",
                         (case (switch_on "clang"), (inc_weight))>,
    Edge<"clang_c", "llvm_ld">,
    Edge<"clang_cpp", "llvm_ld">,
    Edge<"clang_objective_c", "llvm_ld">
    ]>;
