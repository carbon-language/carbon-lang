// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/cpp_file.h"

namespace Carbon::SemIR {

auto CppFile::VisitLocalTopLevelDecls(
    llvm::function_ref<void(const clang::Decl*)> visitor) const -> void {
  ast_unit_->visitLocalTopLevelDecls(
      &visitor, [](void* erased_visitor_ptr, const clang::Decl* decl) {
        auto* visitor_ptr = static_cast<decltype(visitor)*>(erased_visitor_ptr);
        (*visitor_ptr)(decl);
        return true;
      });
}

}  // namespace Carbon::SemIR
