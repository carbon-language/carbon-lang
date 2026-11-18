// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/form.carbon

package Core library "prelude/parts/form";

private fn MakeForm() -> type = "form.make_type";
alias Form = MakeForm();
