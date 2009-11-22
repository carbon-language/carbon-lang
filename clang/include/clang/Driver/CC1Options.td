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

// Target Options

def target_abi : Separate<"-target-abi">,
  HelpText<"Target a particular ABI type">;
def mcpu : Separate<"-mcpu">,
  HelpText<"Target a specific cpu type ('-mcpu help' for details)">;
def target_feature : Separate<"-target-feature">,
  HelpText<"Target specific attributes">;
def triple : Separate<"-triple">,
  HelpText<"Specify target triple (e.g. i686-apple-darwin9)">;

// Analyzer Options

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
  HelpText<"Experimental transfer function inling callees when its definition is available.">;

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

// CodeGen Options

def disable_llvm_optzns : Flag<"-disable-llvm-optzns">,
  HelpText<"Don't run LLVM optimization passes">;
def disable_red_zone : Flag<"-disable-red-zone">,
  HelpText<"Do not emit code that uses the red zone.">;
def g : Flag<"-g">, HelpText<"Generate source level debug information">;
def fno_common : Flag<"-fno-common">,
  HelpText<"Compile common globals like normal definitions">;
def no_implicit_float : Flag<"-no-implicit-float">,
  HelpText<"Don't generate implicit floating point instructions (x86-only)">;
def fno_merge_all_constants : Flag<"-fno-merge-all-constants">,
  HelpText<"Disallow merging of constants.">;
def O : Joined<"-O">, HelpText<"Optimization level">;
def Os : Flag<"-Os">, HelpText<"Optimize for size">;

// Dependency Output Options
def dependency_file : Separate<"-dependency-file">,
  HelpText<"Filename (or -) to write dependency output to">;
def sys_header_deps : Flag<"-sys-header-deps">,
  HelpText<"Include system headers in dependency output">;
def MT : Separate<"-MT">, HelpText<"Specify target for dependency">;
def MP : Flag<"-MP">,
  HelpText<"Create phony target for each dependency (other than main file)">;
