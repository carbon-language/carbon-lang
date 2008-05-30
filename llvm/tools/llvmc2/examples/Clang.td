// A (first stab at a) replacement for the Clang's ccc script.
// To compile, use this command:
//    make TOOLNAME=ccc GRAPH=examples/Clang.td

include "Common.td"

def clang : Tool<
[(in_language ["c", "c++", "objective-c"]),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line (case (switch_on "E"), "clang -E $INFILE -o $OUTFILE",
                 (default), "clang -emit-llvm-bc $INFILE -o $OUTFILE")),
 (switch_option "E", (stop_compilation), (output_suffix "i"),
   (help "Stop after the preprocessing stage, do not run the compiler")),
 (sink)
]>;

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
    Edge<root, clang>,
    Edge<clang, llvm_ld>
    ]>;

