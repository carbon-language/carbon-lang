// Purpose:
//    Check that bad keyword args in \DexLabel are reported.
//    Use --binary switch to trick dexter into skipping the build step.
//
// RUN: not %dexter_base test --binary %s --debugger 'lldb' -- %s | FileCheck %s
// CHECK: parser error:{{.*}}err_label_kwarg.cpp(8): unexpected named args: bad_arg

// DexLabel('test', bad_arg=0)
