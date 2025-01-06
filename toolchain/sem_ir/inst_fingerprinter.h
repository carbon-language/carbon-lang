// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_FINGERPRINTER_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_FINGERPRINTER_H_

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Computes fingerprints for instructions. These fingerprints are intended to be
// stable across compilations and across minor changes to the compiler.
class InstFingerprinter {
 public:
  explicit InstFingerprinter(const File& sem_ir);

  // Gets or computes a fingerprint for the given instruction.
  auto GetOrCompute(InstId inst_id) -> uint64_t;

 private:
  const File* sem_ir_;

  // The fingerprint for each instruction, indexed by the InstId's index. Zero
  // is used for fingerprints that haven't been computed yet; the fingerprint of
  // an instruction is never zero.
  std::vector<uint64_t> fingerprints_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_FINGERPRINTER_H_
