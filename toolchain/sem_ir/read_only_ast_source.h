// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_READ_ONLY_AST_SOURCE_H_
#define CARBON_TOOLCHAIN_SEM_IR_READ_ONLY_AST_SOURCE_H_

#include "clang/Sema/ExternalSemaSource.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

class ReadOnlyASTSource : public clang::ExternalSemaSource {
 public:
  explicit ReadOnlyASTSource(const File& sem_ir) : sem_ir_(sem_ir) {}

  auto layoutRecordType(
      const clang::RecordDecl* record_decl, uint64_t& size, uint64_t& alignment,
      llvm::DenseMap<const clang::FieldDecl*, uint64_t>& field_offsets,
      llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>&
          base_offsets,
      llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>&
          vbase_offsets) -> bool override;

 private:
  const File& sem_ir_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_READ_ONLY_AST_SOURCE_H_
