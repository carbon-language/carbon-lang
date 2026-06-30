// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_redundant_generic_deduced.carbon

library "[[@TEST_NAME]]";

fn F[generic T: type]();

// --- fail_redundant_generic_class_param.carbon

library "[[@TEST_NAME]]";

class C(generic T: type) {}

// --- fail_redundant_runtime_explicit.carbon

library "[[@TEST_NAME]]";

fn G(runtime x: i32);

