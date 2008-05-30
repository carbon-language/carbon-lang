// A simple wrapper for gcc.
// To compile, use this command:
//    make TOOLNAME=llvmc_simple GRAPH=examples/Simple.td

include "Common.td"

def gcc : Tool<
[(in_language "c"),
 (out_language "executable"),
 (output_suffix "out"),
 (cmd_line "gcc $INFILE -o $OUTFILE"),
 (sink)
]>;

def LanguageMap : LanguageMap<[LangToSuffixes<"c", ["c"]>]>;

def CompilationGraph : CompilationGraph<[Edge<root, gcc>]>;
