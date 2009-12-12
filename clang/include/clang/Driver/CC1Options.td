//===--- CC1Options.td - Options for clang -cc1 ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by clang -cc1.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "OptParser.td"

//===----------------------------------------------------------------------===//
// Target Options
//===----------------------------------------------------------------------===//

def target_abi : Separate<"-target-abi">,
  HelpText<"Target a particular ABI type">;
def mcpu : Separate<"-mcpu">,
  HelpText<"Target a specific cpu type ('-mcpu help' for details)">;
def target_feature : Separate<"-target-feature">,
  HelpText<"Target specific attributes">;
def triple : Separate<"-triple">,
  HelpText<"Specify target triple (e.g. i686-apple-darwin9)">;
def triple_EQ : Joined<"-triple=">, Alias<triple>;

//===----------------------------------------------------------------------===//
// Analyzer Options
//===----------------------------------------------------------------------===//

def analysis_CFGDump : Flag<"-cfg-dump">,
  HelpText<"Display Control-Flow Graphs">;
def analysis_CFGView : Flag<"-cfg-view">,
  HelpText<"View Control-Flow Graphs using GraphViz">;
def analysis_DisplayLiveVariables : Flag<"-dump-live-variables">,
  HelpText<"Print results of live variable analysis">;
def analysis_SecuritySyntacticChecks : Flag<"-warn-security-syntactic">,
  HelpText<"Perform quick security checks that require no data flow">;
def analysis_WarnDeadStores : Flag<"-warn-dead-stores">,
  HelpText<"Warn about stores to dead variables">;
def analysis_WarnUninitVals : Flag<"-warn-uninit-values">,
  HelpText<"Warn about uses of uninitialized variables">;
def analysis_WarnObjCMethSigs : Flag<"-warn-objc-methodsigs">,
  HelpText<"Warn about Objective-C method signatures with type incompatibilities">;
def analysis_WarnObjCDealloc : Flag<"-warn-objc-missing-dealloc">,
  HelpText<"Warn about Objective-C classes that lack a correct implementation of -dealloc">;
def analysis_WarnObjCUnusedIvars : Flag<"-warn-objc-unused-ivars">,
  HelpText<"Warn about private ivars that are never used">;
def analysis_CheckerCFRef : Flag<"-checker-cfref">,
  HelpText<"Run the [Core] Foundation reference count checker">;
def analysis_WarnSizeofPointer : Flag<"-warn-sizeof-pointer">,
  HelpText<"Warn about unintended use of sizeof() on pointer expressions">;
def analysis_InlineCall : Flag<"-inline-call">,
  HelpText<"Experimental transfer function inlining callees when its definition is available.">;

def analyzer_store : Separate<"-analyzer-store">,
  HelpText<"Source Code Analysis - Abstract Memory Store Models">;
def analyzer_store_EQ : Joined<"-analyzer-store=">, Alias<analyzer_store>;

def analyzer_constraints : Separate<"-analyzer-constraints">,
  HelpText<"Source Code Analysis - Symbolic Constraint Engines">;
def analyzer_constraints_EQ : Joined<"-analyzer-constraints=">,
  Alias<analyzer_constraints>;

def analyzer_output : Separate<"-analyzer-output">,
  HelpText<"Source Code Analysis - Output Options">;
def analyzer_output_EQ : Joined<"-analyzer-output=">,
  Alias<analyzer_output>;

def analyzer_opt_analyze_headers : Flag<"-analyzer-opt-analyze-headers">,
  HelpText<"Force the static analyzer to analyze functions defined in header files">;
def analyzer_opt_analyze_nested_blocks : Flag<"-analyzer-opt-analyze-nested-blocks">,
  HelpText<"Analyze the definitions of blocks in addition to functions">;  
def analyzer_display_progress : Flag<"-analyzer-display-progress">,
  HelpText<"Emit verbose output about the analyzer's progress">;
def analyzer_experimental_checks : Flag<"-analyzer-experimental-checks">,
  HelpText<"Use experimental path-sensitive checks">;
def analyzer_experimental_internal_checks :
  Flag<"-analyzer-experimental-internal-checks">,
  HelpText<"Use new default path-sensitive checks currently in testing">;
def analyze_function : Separate<"-analyze-function">,
  HelpText<"Run analysis on specific function">;
def analyze_function_EQ : Joined<"-analyze-function=">, Alias<analyze_function>;
def analyzer_eagerly_assume : Flag<"-analyzer-eagerly-assume">,
  HelpText<"Eagerly assume the truth/falseness of some symbolic constraints">;
def analyzer_no_purge_dead : Flag<"-analyzer-no-purge-dead">,
  HelpText<"Don't remove dead symbols, bindings, and constraints before processing a statement">;
def trim_egraph : Flag<"-trim-egraph">,
  HelpText<"Only show error-related paths in the analysis graph">;
def analyzer_viz_egraph_graphviz : Flag<"-analyzer-viz-egraph-graphviz">,
  HelpText<"Display exploded graph using GraphViz">;
def analyzer_viz_egraph_ubigraph : Flag<"-analyzer-viz-egraph-ubigraph">,
  HelpText<"Display exploded graph using Ubigraph">;

//===----------------------------------------------------------------------===//
// CodeGen Options
//===----------------------------------------------------------------------===//

def disable_llvm_optzns : Flag<"-disable-llvm-optzns">,
  HelpText<"Don't run LLVM optimization passes">;
def disable_red_zone : Flag<"-disable-red-zone">,
  HelpText<"Do not emit code that uses the red zone.">;
def g : Flag<"-g">, HelpText<"Generate source level debug information">;
def fcatch_undefined_behavior : Flag<"-fcatch-undefined-behavior">,
    HelpText<"Generate runtime checks for undefined behavior.">;
def fno_common : Flag<"-fno-common">,
  HelpText<"Compile common globals like normal definitions">;
def no_implicit_float : Flag<"-no-implicit-float">,
  HelpText<"Don't generate implicit floating point instructions (x86-only)">;
def fno_merge_all_constants : Flag<"-fno-merge-all-constants">,
  HelpText<"Disallow merging of constants.">;
def masm_verbose : Flag<"-masm-verbose">,
  HelpText<"Generate verbose assembly output">;
def mcode_model : Separate<"-mcode-model">,
  HelpText<"The code model to use">;
def mdebug_pass : Separate<"-mdebug-pass">,
  HelpText<"Enable additional debug output">;
def mdisable_fp_elim : Flag<"-mdisable-fp-elim">,
  HelpText<"Disable frame pointer elimination optimization">;
def mfloat_abi : Separate<"-mfloat-abi">,
  HelpText<"The float ABI to use">;
def mlimit_float_precision : Separate<"-mlimit-float-precision">,
  HelpText<"Limit float precision to the given value">;
def mno_zero_initialized_in_bss : Flag<"-mno-zero-initialized-in-bss">,
  HelpText<"Do not put zero initialized data in the BSS">;
def msoft_float : Flag<"-msoft-float">,
  HelpText<"Use software floating point">;
def mrelocation_model : Separate<"-mrelocation-model">,
  HelpText<"The relocation model to use">;
def munwind_tables : Flag<"-munwind-tables">,
  HelpText<"Generate unwinding tables for all functions">;
def O : Joined<"-O">, HelpText<"Optimization level">;
def Os : Flag<"-Os">, HelpText<"Optimize for size">;

//===----------------------------------------------------------------------===//
// Dependency Output Options
//===----------------------------------------------------------------------===//

def dependency_file : Separate<"-dependency-file">,
  HelpText<"Filename (or -) to write dependency output to">;
def sys_header_deps : Flag<"-sys-header-deps">,
  HelpText<"Include system headers in dependency output">;
def MT : Separate<"-MT">, HelpText<"Specify target for dependency">;
def MP : Flag<"-MP">,
  HelpText<"Create phony target for each dependency (other than main file)">;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def dump_build_information : Separate<"-dump-build-information">,
  MetaVarName<"<filename>">,
  HelpText<"output a dump of some build information to a file">;
def fno_show_column : Flag<"-fno-show-column">,
  HelpText<"Do not include column number on diagnostics">;
def fno_show_source_location : Flag<"-fno-show-source-location">,
  HelpText<"Do not include source location information with diagnostics">;
def fno_caret_diagnostics : Flag<"-fno-caret-diagnostics">,
  HelpText<"Do not include source line and caret with diagnostics">;
def fno_diagnostics_fixit_info : Flag<"-fno-diagnostics-fixit-info">,
  HelpText<"Do not include fixit information in diagnostics">;
def w : Flag<"-w">, HelpText<"Suppress all warnings">;
def pedantic : Flag<"-pedantic">;
def pedantic_errors : Flag<"-pedantic-errors">;

// This gets all -W options, including -Werror, -W[no-]system-headers, etc.  The
// driver has stripped off -Wa,foo etc.  The driver has also translated -W to
// -Wextra, so we don't need to worry about it.
def W : Joined<"-W">;

def fdiagnostics_print_source_range_info : Flag<"-fdiagnostics-print-source-range-info">,
  HelpText<"Print source range spans in numeric form">;
def fdiagnostics_show_option : Flag<"-fdiagnostics-show-option">,
  HelpText<"Print diagnostic name with mappable diagnostics">;
def fmessage_length : Separate<"-fmessage-length">, MetaVarName<"<N>">,
  HelpText<"Format message diagnostics so that they fit within N columns or fewer, when possible.">;
def fcolor_diagnostics : Flag<"-fcolor-diagnostics">,
  HelpText<"Use colors in diagnostics">;
def Wno_rewrite_macros : Flag<"-Wno-rewrite-macros">,
  HelpText<"Silence ObjC rewriting warnings">;
def verify : Flag<"-verify">,
  HelpText<"Verify emitted diagnostics and warnings">;

//===----------------------------------------------------------------------===//
// Frontend Options
//===----------------------------------------------------------------------===//

// This isn't normally used, it is just here so we can parse a
// CompilerInvocation out of a driver-derived argument vector.
def cc1 : Flag<"-cc1">;

def code_completion_at : Separate<"-code-completion-at">,
  MetaVarName<"<file>:<line>:<column>">,
  HelpText<"Dump code-completion information at a location">;
def remap_file : Separate<"-remap-file">,
  MetaVarName<"<from>;<to>">,
  HelpText<"Replace the contents of the <from> file with the contents of the <to> file">;
def code_completion_at_EQ : Joined<"-code-completion-at=">,
  Alias<code_completion_at>;
def no_code_completion_debug_printer : Flag<"-no-code-completion-debug-printer">,
  HelpText<"Don't use the \"debug\" code-completion print">;
def code_completion_macros : Flag<"-code-completion-macros">,
  HelpText<"Include macros in code-completion results">;
def disable_free : Flag<"-disable-free">,
  HelpText<"Disable freeing of memory on exit">;
def empty_input_only : Flag<"-empty-input-only">,
  HelpText<"Force running on an empty input file">;
def help : Flag<"-help">,
  HelpText<"Print this help text">;
def _help : Flag<"--help">, Alias<help>;
def x : Separate<"-x">, HelpText<"Input language type">;
def cxx_inheritance_view : Separate<"-cxx-inheritance-view">,
  MetaVarName<"<class name>">,
  HelpText<"View C++ inheritance for a specified class">;
def fixit_at : Separate<"-fixit-at">, MetaVarName<"<source location>">,
  HelpText<"Perform Fix-It modifications at the given source location">;
def o : Separate<"-o">, MetaVarName<"<path>">, HelpText<"Specify output file">;
def load : Separate<"-load">, MetaVarName<"<dsopath>">,
  HelpText<"Load the named plugin (dynamic shared object)">;
def plugin : Separate<"-plugin">,
  HelpText<"Use the named plugin action (use \"help\" to list available options)">;
def version : Flag<"-version">,
  HelpText<"Print the compiler version">;
def _version : Flag<"--version">, Alias<version>;

def Action_Group : OptionGroup<"<action group>">;
let Group = Action_Group in {

def Eonly : Flag<"-Eonly">,
  HelpText<"Just run preprocessor, no output (for timings)">;
def E : Flag<"-E">,
  HelpText<"Run preprocessor, emit preprocessed file">;
def dump_raw_tokens : Flag<"-dump-raw-tokens">,
  HelpText<"Lex file in raw mode and dump raw tokens">;
def analyze : Flag<"-analyze">,
  HelpText<"Run static analysis engine">;
def dump_tokens : Flag<"-dump-tokens">,
  HelpText<"Run preprocessor, dump internal rep of tokens">;
def parse_noop : Flag<"-parse-noop">,
  HelpText<"Run parser with noop callbacks (for timings)">;
def fsyntax_only : Flag<"-fsyntax-only">,
  HelpText<"Run parser and perform semantic analysis">;
def fixit : Flag<"-fixit">,
  HelpText<"Apply fix-it advice to the input source">;
def parse_print_callbacks : Flag<"-parse-print-callbacks">,
  HelpText<"Run parser and print each callback invoked">;
def emit_html : Flag<"-emit-html">,
  HelpText<"Output input source as HTML">;
def ast_print : Flag<"-ast-print">,
  HelpText<"Build ASTs and then pretty-print them">;
def ast_print_xml : Flag<"-ast-print-xml">,
  HelpText<"Build ASTs and then print them in XML format">;
def ast_dump : Flag<"-ast-dump">,
  HelpText<"Build ASTs and then debug dump them">;
def ast_view : Flag<"-ast-view">,
  HelpText<"Build ASTs and view them with GraphViz">;
def print_decl_contexts : Flag<"-print-decl-contexts">,
  HelpText<"Print DeclContexts and their Decls">;
def dump_record_layouts : Flag<"-dump-record-layouts">,
  HelpText<"Dump record layout information">;
def emit_pth : Flag<"-emit-pth">,
  HelpText<"Generate pre-tokenized header file">;
def emit_pch : Flag<"-emit-pch">,
  HelpText<"Generate pre-compiled header file">;
def S : Flag<"-S">,
  HelpText<"Emit native assembly code">;
def emit_llvm : Flag<"-emit-llvm">,
  HelpText<"Build ASTs then convert to LLVM, emit .ll file">;
def emit_llvm_bc : Flag<"-emit-llvm-bc">,
  HelpText<"Build ASTs then convert to LLVM, emit .bc file">;
def emit_llvm_only : Flag<"-emit-llvm-only">,
  HelpText<"Build ASTs and convert to LLVM, discarding output">;
def rewrite_test : Flag<"-rewrite-test">,
  HelpText<"Rewriter playground">;
def rewrite_objc : Flag<"-rewrite-objc">,
  HelpText<"Rewrite ObjC into C (code rewriter example)">;
def rewrite_macros : Flag<"-rewrite-macros">,
  HelpText<"Expand macros without full preprocessing">;
def rewrite_blocks : Flag<"-rewrite-blocks">,
  HelpText<"Rewrite Blocks to C">;

}

def relocatable_pch : Flag<"-relocatable-pch">,
  HelpText<"Whether to build a relocatable precompiled header">;
def print_stats : Flag<"-print-stats">,
  HelpText<"Print performance metrics and statistics">;
def ftime_report : Flag<"-ftime-report">,
  HelpText<"Print the amount of time each phase of compilation takes">;

//===----------------------------------------------------------------------===//
// Language Options
//===----------------------------------------------------------------------===//

def fno_builtin : Flag<"-fno-builtin">,
  HelpText<"Disable implicit builtin knowledge of functions">;
def faltivec : Flag<"-faltivec">,
  HelpText<"Enable AltiVec vector initializer syntax">;
def faccess_control : Flag<"-faccess-control">,
  HelpText<"Enable C++ access control">;
def fdollars_in_identifiers : Flag<"-fdollars-in-identifiers">,
  HelpText<"Allow '$' in identifiers">;
def femit_all_decls : Flag<"-femit-all-decls">,
  HelpText<"Emit all declarations, even if unused">;
def fblocks : Flag<"-fblocks">,
  HelpText<"enable the 'blocks' language feature">;
def fheinous_gnu_extensions : Flag<"-fheinous-gnu-extensions">;  
def fexceptions : Flag<"-fexceptions">,
  HelpText<"Enable support for exception handling">;
def ffreestanding : Flag<"-ffreestanding">,
  HelpText<"Assert that the compilation takes place in a freestanding environment">;
def fgnu_runtime : Flag<"-fgnu-runtime">,
  HelpText<"Generate output compatible with the standard GNU Objective-C runtime">;
def std_EQ : Joined<"-std=">,
  HelpText<"Language standard to compile for">;
def fms_extensions : Flag<"-fms-extensions">,
  HelpText<"Accept some non-standard constructs used in Microsoft header files ">;
def main_file_name : Separate<"-main-file-name">,
  HelpText<"Main file name to use for debug info">;
def fno_elide_constructors : Flag<"-fno-elide-constructors">,
  HelpText<"Disable C++ copy constructor elision">;
def fno_lax_vector_conversions : Flag<"-fno-lax-vector-conversions">,
  HelpText<"Disallow implicit conversions between vectors with a different number of elements or different element types">;
def fno_math_errno : Flag<"-fno-math-errno">,
  HelpText<"Don't require math functions to respect errno">;
def fno_signed_char : Flag<"-fno-signed-char">,
  HelpText<"Char is unsigned">;
def fno_operator_names : Flag<"-fno-operator-names">,
  HelpText<"Do not treat C++ operator name keywords as synonyms for operators">;
def fconstant_string_class : Separate<"-fconstant-string-class">,
  MetaVarName<"<class name>">,
  HelpText<"Specify the class to use for constant Objective-C string objects.">;
def fobjc_gc : Flag<"-fobjc-gc">,
  HelpText<"Enable Objective-C garbage collection">;
def fobjc_gc_only : Flag<"-fobjc-gc-only">,
  HelpText<"Use GC exclusively for Objective-C related memory management">;
def print_ivar_layout : Flag<"-print-ivar-layout">,
  HelpText<"Enable Objective-C Ivar layout bitmap print trace">;
def fobjc_nonfragile_abi : Flag<"-fobjc-nonfragile-abi">,
  HelpText<"enable objective-c's nonfragile abi">;
def ftrapv : Flag<"-ftrapv">,
  HelpText<"Trap on integer overflow">;
def pic_level : Separate<"-pic-level">,
  HelpText<"Value for __PIC__">;
def pthread : Flag<"-pthread">,
  HelpText<"Support POSIX threads in generated code">;
def fpascal_strings : Flag<"-fpascal-strings">,
  HelpText<"Recognize and construct Pascal-style string literals">;
def fno_rtti : Flag<"-fno-rtti">,
  HelpText<"Disable generation of rtti information">;
def fshort_wchar : Flag<"-fshort-wchar">,
  HelpText<"Force wchar_t to be a short unsigned int">;
def static_define : Flag<"-static-define">,
  HelpText<"Should __STATIC__ be defined">;
def stack_protector : Separate<"-stack-protector">,
  HelpText<"Enable stack protectors">;
def fvisibility : Separate<"-fvisibility">,
  HelpText<"Default symbol visibility">;
def ftemplate_depth : Separate<"-ftemplate-depth">,
  HelpText<"Maximum depth of recursive template instantiation">;
def trigraphs : Flag<"-trigraphs">,
  HelpText<"Process trigraph sequences">;
def fwritable_strings : Flag<"-fwritable-strings">,
  HelpText<"Store string literals as writable data">;

//===----------------------------------------------------------------------===//
// Header Search Options
//===----------------------------------------------------------------------===//

def nostdinc : Flag<"-nostdinc">,
  HelpText<"Disable standard #include directories">;
def nobuiltininc : Flag<"-nobuiltininc">,
  HelpText<"Disable builtin #include directories">;
def F : JoinedOrSeparate<"-F">, MetaVarName<"<directory>">,
  HelpText<"Add directory to framework include search path">;
def I : JoinedOrSeparate<"-I">, MetaVarName<"<directory>">,
  HelpText<"Add directory to include search path">;
def idirafter : JoinedOrSeparate<"-idirafter">, MetaVarName<"<directory>">,
  HelpText<"Add directory to AFTER include search path">;
def iquote : JoinedOrSeparate<"-iquote">, MetaVarName<"<directory>">,
  HelpText<"Add directory to QUOTE include search path">;
def isystem : JoinedOrSeparate<"-isystem">, MetaVarName<"<directory>">,
  HelpText<"Add directory to SYSTEM include search path">;
def iprefix : JoinedOrSeparate<"-iprefix">, MetaVarName<"<prefix>">,
  HelpText<"Set the -iwithprefix/-iwithprefixbefore prefix">;
def iwithprefix : JoinedOrSeparate<"-iwithprefix">, MetaVarName<"<dir>">,
  HelpText<"Set directory to SYSTEM include search path with prefix">;
def iwithprefixbefore : JoinedOrSeparate<"-iwithprefixbefore">,
  MetaVarName<"<dir>">,
  HelpText<"Set directory to include search path with prefix">;
def isysroot : JoinedOrSeparate<"-isysroot">, MetaVarName<"<dir>">,
  HelpText<"Set the system root directory (usually /)">;
def v : Flag<"-v">, HelpText<"Enable verbose output">;

//===----------------------------------------------------------------------===//
// Preprocessor Options
//===----------------------------------------------------------------------===//

def D : JoinedOrSeparate<"-D">, MetaVarName<"<macro>">,
  HelpText<"Predefine the specified macro">;
def include_ : JoinedOrSeparate<"-include">, MetaVarName<"<file>">, EnumName<"include">,
  HelpText<"Include file before parsing">;
def imacros : JoinedOrSeparate<"-imacros">, MetaVarName<"<file>">,
  HelpText<"Include macros from file before parsing">;
def include_pch : Separate<"-include-pch">, MetaVarName<"<file>">,
  HelpText<"Include precompiled header file">;
def include_pth : Separate<"-include-pth">, MetaVarName<"<file>">,
  HelpText<"Include file before parsing">;
def token_cache : Separate<"-token-cache">, MetaVarName<"<path>">,
  HelpText<"Use specified token cache file">;
def U : JoinedOrSeparate<"-U">, MetaVarName<"<macro>">,
  HelpText<"Undefine the specified macro">;
def undef : Flag<"-undef">, MetaVarName<"<macro>">,
  HelpText<"undef all system defines">;

//===----------------------------------------------------------------------===//
// Preprocessed Output Options
//===----------------------------------------------------------------------===//

def P : Flag<"-P">,
  HelpText<"Disable linemarker output in -E mode">;
def C : Flag<"-C">,
  HelpText<"Enable comment output in -E mode">;
def CC : Flag<"-CC">,
  HelpText<"Enable comment output in -E mode, even from macro expansions">;
def dM : Flag<"-dM">,
  HelpText<"Print macro definitions in -E mode instead of normal output">;
def dD : Flag<"-dD">,
  HelpText<"Print macro definitions in -E mode in addition to normal output">;
