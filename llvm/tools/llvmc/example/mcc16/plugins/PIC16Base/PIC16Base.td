//===- PIC16Base.td - PIC16 toolchain driver ---------------*- tablegen -*-===//
//
// A basic driver for the PIC16 toolchain.
//
//===----------------------------------------------------------------------===//

include "llvm/CompilerDriver/Common.td"

// Options

def OptionList : OptionList<[
 (switch_option "g",
    (help "Enable Debugging")),
 (switch_option "S",
    (help "Stop after compilation, do not assemble")),
 (parameter_option "I",
    (help "Add a directory to include path")),
 (parameter_option "pre-RA-sched",
    (help "Example of an option that is passed to llc")),
 (prefix_list_option "Wa,",
    (help "Pass options to native assembler")),
 (prefix_list_option "Wl,",
    (help "Pass options to native linker")),
 (prefix_list_option "Wllc,",
    (help "Pass options to llc")),
 (prefix_list_option "Wo,",
    (help "Pass options to llvm-ld"))
]>;

// Tools

def clang_cc : Tool<[
 (in_language "c"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line "$CALL(GetBinDir)clang-cc -I $CALL(GetStdHeadersDir) -triple=pic16- -emit-llvm-bc $INFILE -o $OUTFILE"),
 (actions (case
          (not_empty "I"), (forward "I"))),
 (sink)
]>;

def llvm_ld : Tool<[
 (in_language "llvm-bitcode"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line "llvm-ld -f -link-as-library $INFILE -o $OUTFILE"),
 (actions (case
          (switch_on "g"), (append_cmd "-disable-opt"),
          (not_empty "Wo,"), (unpack_values "Wo,")))
]>;

def llvm_ld_lto : Tool<[
 (in_language "llvm-bitcode"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line "llvm-ld -link-as-library $INFILE -o $OUTFILE"),
 (actions (case
          (switch_on "g"), (append_cmd "-disable-opt"),
          (not_empty "Wo,"), (unpack_values "Wo,"))),
 (join)
]>;

def llc : Tool<[
 (in_language "llvm-bitcode"),
 (out_language "assembler"),
 (output_suffix "s"),
 (cmd_line "llc -march=pic16 -f $INFILE -o $OUTFILE"),
 (actions (case
          (switch_on "S"), (stop_compilation),
          (not_empty "Wllc,"), (unpack_values "Wllc,"),
          (not_empty "pre-RA-sched"), (forward "pre-RA-sched")))
]>;

def gpasm : Tool<[
 (in_language "assembler"),
 (out_language "object-code"),
 (output_suffix "o"),
 (cmd_line "gpasm -I $CALL(GetStdAsmHeadersDir) $INFILE -o $OUTFILE"),
 (actions (case
          (not_empty "Wa,"), (unpack_values "Wa,")))
]>;

def mplink : Tool<[
 (in_language "object-code"),
 (out_language "executable"),
 (output_suffix "out"),
 (cmd_line "mplink /k $CALL(GetStdLinkerScriptsDir) /l $CALL(GetStdLibsDir) $INFILE -o $OUTFILE"),
 (actions (case
          (not_empty "Wl,"), (unpack_values "Wl,"))),
 (join)
]>;

// Language map

def LanguageMap : LanguageMap<[
    LangToSuffixes<"c", ["c"]>,
    LangToSuffixes<"c-cpp-output", ["i"]>,
    LangToSuffixes<"assembler", ["s"]>,
    LangToSuffixes<"assembler-with-cpp", ["S"]>,
    LangToSuffixes<"llvm-assembler", ["ll"]>,
    LangToSuffixes<"llvm-bitcode", ["bc"]>,
    LangToSuffixes<"object-code", ["o"]>,
    LangToSuffixes<"executable", ["out"]>
]>;

// Compilation graph

def CompilationGraph : CompilationGraph<[
    Edge<"root", "clang_cc">,
    Edge<"clang_cc", "llvm_ld_lto">,
    Edge<"llvm_ld_lto", "llc">,
    OptionalEdge<"clang_cc", "llvm_ld", (case (switch_on "S"), (inc_weight))>,
    Edge<"llvm_ld", "llc">,
    Edge<"llc", "gpasm">,
    Edge<"gpasm", "mplink">
]>;
