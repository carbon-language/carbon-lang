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
  // Gets or computes a fingerprint for the given instruction.
  auto GetOrCompute(const File* file, InstId inst_id) -> uint64_t;

 private:
  // The fingerprint for each instruction that has had its fingerprint computed,
  // indexed by the InstId's index.
  Map<std::pair<const File*, InstId>, uint64_t> fingerprints_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_FINGERPRINTER_H_
