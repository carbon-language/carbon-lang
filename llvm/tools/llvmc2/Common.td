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

// Special Tool instance - root of all toolchains

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

// Map from suffixes to language names

class LangToSuffixes<string str, list<string> lst> {
      string lang = str;
      list<string> suffixes = lst;
}

class LanguageMap<list<LangToSuffixes> lst> {
      list<LangToSuffixes> map = lst;
}

// Compilation graph

class Edge<Tool t1, Tool t2> {
      Tool a = t1;
      Tool b = t2;
}

class CompilationGraph<list<Edge> lst> {
      list<Edge> edges = lst;
}
