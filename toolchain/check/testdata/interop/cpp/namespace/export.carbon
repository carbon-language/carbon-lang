// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/namespace/export.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/namespace/export.carbon

// --- alias_identity.carbon

import Cpp;

namespace M;
alias N = M;

class M.C {}

inline Cpp '''
// OK, valid redeclaration: M and N are the same namespace.
namespace A = Carbon::M;
namespace A = Carbon::N;

// Valid: M::C and N::C are the same class.
Carbon::M::C *pmc;
Carbon::N::C *pnc = pmc;
''';
