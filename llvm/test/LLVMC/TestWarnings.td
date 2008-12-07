// Check that the compiler warns about unused options.
// This should fail because the output is printed on stderr.
// RUN: tblgen -I $srcroot/include --gen-llvmc %s | grep extern
// XFAIL: *

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[(extern_switch "Wall"),
    (extern_parameter "std"), (extern_list "L")]>;
