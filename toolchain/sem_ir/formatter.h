// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_
#define CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_

#include <memory>

#include "llvm/Support/raw_ostream.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

// Formatter for printing textual Semantics IR.
class Formatter {
 public:
  explicit Formatter(const Lex::TokenizedBuffer& tokenized_buffer,
                     const Parse::Tree& parse_tree, const File& sem_ir);
  ~Formatter();

  auto Print(llvm::raw_ostream& out) -> void;
  auto PrintPartialTrailingCodeBlock(llvm::ArrayRef<SemIR::InstId> block,
                                     int indent, llvm::raw_ostream& out)
      -> void;
  auto PrintInst(SemIR::InstId inst_id, int indent, llvm::raw_ostream& out)
      -> void;

 private:
  class FormatterImpl;

  std::unique_ptr<FormatterImpl> formatter_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_
