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
 (prefix_option "p",
    (help "Specify part name")),
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
 (switch_option "X",
    (help "Do not invoke mp2hex to create an output hex file.")),
 (switch_option "O0",
    (help "Do not optimize")),
 (switch_option "O1",
    (help "Optimization Level 1.")),
 (switch_option "O2",
    (help "Optimization Level 2.")),
 (switch_option "O3",
    (help "Optimization Level 3.")),
 (switch_option "Od",
    (help "Perform Debug-safe Optimizations only.")),
 (switch_option "w",
    (help "Disable all warnings.")),
// (switch_option "O1",
//    (help "Optimization level 1")),
// (switch_option "O2",
//    (help "Optimization level 2. (Default)")),
// (parameter_option "pre-RA-sched",
//    (help "Example of an option that is passed to llc")),
 (parameter_option "regalloc",
    (help "Register allocator to use.(possible values: simple, linearscan, pbqp, local. default = linearscan)")),
 (prefix_list_option "Wa,", (comma_separated),
    (help "Pass options to assembler (Run 'gpasm -help' for assembler options)")),
 (prefix_list_option "Wl,", (comma_separated),
    (help "Pass options to linker (Run 'mplink -help' for linker options)"))
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
                (switch_on "w"), (append_cmd "-w"),
                (switch_on "O1"), (append_cmd ""),
                (switch_on "O2"), (append_cmd ""),
                (switch_on "O3"), (append_cmd ""),
                (switch_on "Od"), (append_cmd ""),
                (not_empty "D"), (forward "D"),
                (not_empty "I"), (forward "I"),
                (switch_on "O0"), (append_cmd "-O0"),
                (default), (append_cmd "-O1")))
// (sink)
]>;

def clang_cc : clang_based<"c", "$CALL(GetBinDir)clang -cc1                                                        -I $CALL(GetStdHeadersDir)                                                      -D $CALL(GetLowerCasePartDefine)                                                -D $CALL(GetUpperCasePartDefine) -triple=pic16-                                 -emit-llvm-bc ", "i">;

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
 (cmd_line "$CALL(GetBinDir)llvm-ld -L $CALL(GetStdLibsDir) -disable-gvn -disable-licm-promotion -disable-mem2reg $INFILE -b $OUTFILE -l std"),
 (actions (case
          (switch_on "O0"), (append_cmd "-disable-opt"),
          (switch_on "O1"), (append_cmd "-disable-opt"),
// Whenever O3 is not specified on the command line, default i.e. disable-inlining will always be added.
          (switch_on "O2"), (append_cmd ""), 
          (switch_on "O3"), (append_cmd ""),
          (default), (append_cmd "-disable-inlining"))),
 (join)
]>;

// optimize single file
def llvm_ld_optimizer : Tool<[
 (in_language "llvm-bitcode"),
 (out_language "llvm-bitcode"),
 (output_suffix "bc"),
// FIXME: we are still not disabling licm-promotion.
// -disable-licm-promotion and building stdn library causes c16-71 to fail.
 (cmd_line "$CALL(GetBinDir)llvm-ld -disable-gvn -disable-mem2reg                              $INFILE -b $OUTFILE"),
 (actions (case
          (switch_on "O0"), (append_cmd "-disable-opt"),
          (switch_on "O1"), (append_cmd "-disable-opt"),
// Whenever O3 is not specified on the command line, default i.e. disable-inlining will always be added.
          (switch_on "O2"), (append_cmd ""), 
          (switch_on "O3"), (append_cmd ""),
          (default), (append_cmd "-disable-inlining")))
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
 (cmd_line "$CALL(GetBinDir)llc -march=pic16 -disable-jump-tables -pre-RA-sched=list-burr -f $INFILE -o $OUTFILE"),
 (actions (case
          (switch_on "S"), (stop_compilation),
//          (not_empty "Wllc,"), (unpack_values "Wllc,"),
//         (not_empty "pre-RA-sched"), (forward "pre-RA-sched")))
         (not_empty "regalloc"), (forward "regalloc"),
         (empty "regalloc"), (append_cmd "-regalloc=linearscan")))
]>;

def gpasm : Tool<[
 (in_language "assembler"),
 (out_language "object-code"),
 (output_suffix "o"),
 (cmd_line "$CALL(GetBinDir)gpasm -z -r decimal -I $CALL(GetStdAsmHeadersDir) -C -c -w 2 $INFILE -o $OUTFILE"),
 (actions (case
          (switch_on "c"), (stop_compilation),
          (switch_on "g"), (append_cmd "-g"),
          (not_empty "p"), (forward "p"),
          (empty "p"), (append_cmd "-p 16f1xxx"),
          (not_empty "Wa,"), (forward_value "Wa,")))
]>;

def mplink : Tool<[
 (in_language "object-code"),
 (out_language "executable"),
 (output_suffix "cof"),
 (cmd_line "$CALL(GetBinDir)mplink -e -k $CALL(GetStdLinkerScriptsDir) -l $CALL(GetStdLibsDir) intrinsics.lib stdn.lib $INFILE -o $OUTFILE"),
 (actions (case
          (not_empty "Wl,"), (forward_value "Wl,"),
          (switch_on "X"), (append_cmd "-x"),
          (not_empty "L"), (forward_as "L", "-l"),
          (not_empty "K"), (forward_as "K", "-k"),
          (not_empty "m"), (forward "m"),
          (not_empty "p"), [(forward "p"), (append_cmd "-c")],
          (empty "p"), (append_cmd "-p 16f1xxx -c"),
//          (not_empty "l"), [(unpack_values "l"),(append_cmd ".lib")])),
          (not_empty "k"), (forward "k"),
          (not_empty "l"), (forward "l"))),
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
