// Check that warnings about unused options are really emitted.
// This should fail because the output is printed on stderr.
// RUN: tblgen -I %p/../../include --gen-llvmc %s |& grep "option '-Wall' has no effect!"
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[(switch_option "Wall", (extern))]>;
