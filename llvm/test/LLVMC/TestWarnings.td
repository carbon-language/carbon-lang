// Check that the compiler warns about unused options!.
// RUN: tblgen -I $srcroot/include --gen-llvmc %s | grep extern
// XFAIL: *

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[(extern_switch "Wall"),
    (extern_parameter "std"), (extern_list "L")]>;
