include "llvm/Option/OptParser.td"

def output : Separate<["-"], "o">;
def entry  : Separate<["-"], "e">;

def dead_strip              : Flag<["--"], "dead-strip">;
def undefines_are_errors    : Flag<["--"], "undefines-are-errors">;
def keep_globals            : Flag<["--"], "keep-globals">;
def commons_search_archives : Flag<["--"], "commons-search-archives">;

def add_pass       : Separate<["--"], "add-pass">;

def target : Separate<["-"], "target">, HelpText<"Target triple to link for">;
def mllvm : Separate<["-"], "mllvm">, HelpText<"Options to pass to LLVM">;

