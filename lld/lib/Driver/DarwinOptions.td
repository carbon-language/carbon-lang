include "llvm/Option/OptParser.td"

// output kinds
def relocatable : Flag<["-"], "r">, HelpText<"create relocatable output">;
def static : Flag<["-"], "static">;
def dynamic : Flag<["-"], "dynamic">;
def dylib : Flag<["-"], "dylib">;
def bundle : Flag<["-"], "bundle">;
def execute : Flag<["-"], "execute">;
def preload : Flag<["-"], "preload">;

// linking options
def dead_strip : Flag<["-"], "dead_strip">;
def entry : Separate<["-"], "e">, HelpText<"entry symbol name">;
def mllvm : Separate<["-"], "mllvm">, HelpText<"Options to pass to LLVM">;
def arch : Separate<["-"], "arch">, HelpText<"architecture">;
def macosx_version_min : Separate<["-"], "macosx_version_min">, 
                          HelpText<"min Mac OS X version">;
def ios_version_min : Separate<["-"], "ios_version_min">, 
                          HelpText<"min iOS version">;
def ios_simulator_version_min : Separate<["-"], "ios_simulator_version_min">, 
                          HelpText<"min iOS simulator version">;

// search options
def L : Joined<["-"], "L">;

// inputs and outputs
def output : Separate<["-"], "o">, HelpText<"output file name">;


