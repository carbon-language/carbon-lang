// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_
#define CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_

#include "llvm/Support/raw_ostream.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst_namer.h"

namespace Carbon::SemIR {

// Formatter for printing textual Semantics IR.
class Formatter {
 public:
  explicit Formatter(const Lex::TokenizedBuffer& tokenized_buffer,
                     const Parse::Tree& parse_tree, const File& sem_ir);
  ~Formatter();

  // Prints the full IR.
  auto Print(llvm::raw_ostream& out) -> void;

 private:
  const File& sem_ir_;
  // Caches naming between Print calls.
  InstNamer inst_namer_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FORMATTER_H_
