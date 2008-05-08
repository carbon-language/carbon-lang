//===- Tools.td - Common definitions for LLVMCC  -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains common definitions used in llvmcc tool description files.
//
//===----------------------------------------------------------------------===//

class Tool<list<dag> l> {
      list<dag> properties = l;
}

// Special Tool instance - the root node of the compilation graph.

def root : Tool<[]>;

// Possible Tool properties

def in_language;
def out_language;
def output_suffix;
def cmd_line;
def join;
def sink;

// Possible option types

def switch_option;
def parameter_option;
def parameter_list_option;
def prefix_option;
def prefix_list_option;

// Possible option properties

def append_cmd;
def forward;
def stop_compilation;
def unpack_values;
def help;
def required;

// Possible edge properties

// 'Atomic' properties.
def switch_on;
def parameter_equals;
def element_in_list;
def if_input_languages_contain;

// Edge property combinators.
def weight;
def and;
def or;

// Map from suffixes to language names

class LangToSuffixes<string str, list<string> lst> {
      string lang = str;
      list<string> suffixes = lst;
}

class LanguageMap<list<LangToSuffixes> lst> {
      list<LangToSuffixes> map = lst;
}

// Compilation graph

class EdgeBase<Tool t1, Tool t2, list<dag> lst> {
      Tool a = t1;
      Tool b = t2;
      list<dag> props = lst;
}

class Edge<Tool t1, Tool t2> : EdgeBase<t1, t2, []>;

// Edge and SimpleEdge are synonyms.
class SimpleEdge<Tool t1, Tool t2> : EdgeBase<t1, t2, []>;

// Optionally enabled edge.
class OptionalEdge<Tool t1, Tool t2, list<dag> lst> : EdgeBase<t1, t2, lst>;

class CompilationGraph<list<EdgeBase> lst> {
      list<EdgeBase> edges = lst;
}
