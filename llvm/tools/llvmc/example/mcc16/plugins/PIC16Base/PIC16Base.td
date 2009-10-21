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
 (switch_option "E",
    (help "Stop after preprocessing, do not compile")),
 (switch_option "S",
    (help "Stop after compilation, do not assemble")),
 (switch_option "bc",
    (help "Stop after b-code generation, do not compile")),
 (switch_option "c",
    (help "Stop after assemble, do not link")),
 (prefix_list_option "I",
    (help "Add a directory to include path")),
 (prefix_list_option "L",
    (help "Add a directory to library path")),
 (prefix_list_option "K",
    (help "Add a directory to linker script search path")),
 (parameter_option "l",
    (help "Specify a library to link")),
 (parameter_option "k",
    (help "Specify a linker script")),
 (parameter_option "m",
    (help "Generate linker map file with the given name")),
 (prefix_list_option "D",
    (help "Define a macro")),
 (switch_option "O0",
    (help "Do not optimize")),
// (switch_option "O1",
//    (help "Optimization level 1")),
// (switch_option "O2",
//    (help "Optimization level 2. (Default)")),
// (parameter_option "pre-RA-sched",
//    (help "Example of an option that is passed to llc")),
 (prefix_list_option "Wa,",
    (help "Pass options to native assembler")),
 (prefix_list_option "Wl,",
    (help "Pass options to native linker"))
// (prefix_list_option "Wllc,",
//    (help "Pass options to llc")),
// (prefix_list_option "Wo,",
//    (help "Pass options to llvm-ld"))
]>;

// Tools
class clang_based<string language, string cmd, string ext_E> : Tool<
[(in_language language),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line (case
           (switch_on "E"),
           (case 
              (not_empty "o"), !strconcat(cmd, " -E $INFILE -o $OUTFILE"),
              (default), !strconcat(cmd, " -E $INFILE")),
           (default), !strconcat(cmd, " $INFILE -o $OUTFILE"))),
 (actions (case 
                (and (multiple_input_files), (or (switch_on "S"), (switch_on "c"))),
              (error "cannot specify -o with -c or -S with multiple files"),
                (switch_on "E"), [(stop_compilation), (output_suffix ext_E)],
                (switch_on "bc"),[(stop_compilation), (output_suffix "bc")],
                (switch_on "g"), (append_cmd "-g"),
                (not_empty "D"), (forward "D"),
                (not_empty "I"), (forward "I"))),
 (sink)
]>;

def clang_cc : clang_based<"c", "$CALL(GetBinDir)clang-cc                                                        -I $CALL(GetStdHeadersDir) -triple=pic16-                                       -emit-llvm-bc ", "i">;

//def clang_cc : Tool<[
// (in_language "c"),
// (out_language "llvm-bitcode"),
// (output_suffix "bc"),
// (cmd_line "$CALL(GetBinDir)clang-cc -I $CALL(GetStdHeadersDir) -triple=pic16- -emit-llvm-bc "),
// (cmd_line kkkkk
// (actions (case
//          (switch_on "g"), (append_cmd "g"),
//          (not_empty "I"), (forward "I"))),
// (sink)
//]>;


// pre-link-and-lto step.
def llvm_ld : Tool<[
 (in_language "llvm-bitcode"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line "$CALL(GetBinDir)llvm-ld -L $CALL(GetStdLibsDir) -disable-gvn -instcombine -disable-inlining                   $INFILE -b $OUTFILE -l std"),
 (actions (case
          (switch_on "O0"), (append_cmd "-disable-opt"))),
 (join)
]>;

// optimize single file
def llvm_ld_optimizer : Tool<[
 (in_language "llvm-bitcode"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
 (cmd_line "$CALL(GetBinDir)llvm-ld -disable-gvn -instcombine -disable-inlining                   $INFILE -b $OUTFILE"),
 (actions (case
          (switch_on "O0"), (append_cmd "-disable-opt")))
]>;

// optimizer step.
def pic16passes : Tool<[
 (in_language "llvm-bitcode"),
 (out_language "llvm-bitcode"),
 (output_suffix "obc"),
 (cmd_line "$CALL(GetBinDir)opt -pic16cg -pic16overlay $INFILE -f -o $OUTFILE"),
 (actions (case
          (switch_on "O0"), (append_cmd "-disable-opt")))
]>;

def llc : Tool<[
 (in_language "llvm-bitcode"),
 (out_language "assembler"),
 (output_suffix "s"),
 (cmd_line "$CALL(GetBinDir)llc -march=pic16 -disable-jump-tables -pre-RA-sched=list-burr -regalloc=pbqp -f $INFILE -o $OUTFILE"),
 (actions (case
          (switch_on "S"), (stop_compilation)))
//          (not_empty "Wllc,"), (unpack_values "Wllc,"),
//         (not_empty "pre-RA-sched"), (forward "pre-RA-sched")))
]>;

def gpasm : Tool<[
 (in_language "assembler"),
 (out_language "object-code"),
 (output_suffix "o"),
 (cmd_line "$CALL(GetBinDir)gpasm -r decimal -p p16F1937 -I $CALL(GetStdAsmHeadersDir) -C -c -q $INFILE -o $OUTFILE"),
 (actions (case
          (switch_on "c"), (stop_compilation),
          (switch_on "g"), (append_cmd "-g"),
          (not_empty "Wa,"), (unpack_values "Wa,")))
]>;

def mplink : Tool<[
 (in_language "object-code"),
 (out_language "executable"),
 (output_suffix "cof"),
 (cmd_line "$CALL(GetBinDir)mplink.exe -k $CALL(GetStdLinkerScriptsDir) -l $CALL(GetStdLibsDir) -p 16f1937  intrinsics.lib devices.lib $INFILE -o $OUTFILE"),
 (actions (case
          (not_empty "Wl,"), (unpack_values "Wl,"),
          (not_empty "L"), (forward_as "L", "-l"),
          (not_empty "K"), (forward_as "K", "-k"),
          (not_empty "m"), (forward "m"),
//          (not_empty "l"), [(unpack_values "l"),(append_cmd ".lib")])),
          (not_empty "k"), (unpack_values "k"),
          (not_empty "l"), (unpack_values "l"))),
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
    LangToSuffixes<"executable", ["cof"]>
]>;

// Compilation graph

def CompilationGraph : CompilationGraph<[
    Edge<"root", "clang_cc">,
    Edge<"root", "llvm_ld">,
    OptionalEdge<"root", "llvm_ld_optimizer", (case 
                                         (switch_on "S"), (inc_weight),
                                         (switch_on "c"), (inc_weight))>,
    Edge<"root", "gpasm">,
    Edge<"root", "mplink">,
    Edge<"clang_cc", "llvm_ld">,
    OptionalEdge<"clang_cc", "llvm_ld_optimizer", (case 
                                         (switch_on "S"), (inc_weight),
                                         (switch_on "c"), (inc_weight))>,
    Edge<"llvm_ld", "pic16passes">,
    Edge<"llvm_ld_optimizer", "pic16passes">,
    Edge<"pic16passes", "llc">,
    Edge<"llc", "gpasm">,
    Edge<"gpasm", "mplink">
]>;
