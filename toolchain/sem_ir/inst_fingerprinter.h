// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_FINGERPRINTER_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_FINGERPRINTER_H_

#include "toolchain/base/fixed_size_value_store.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Computes fingerprints for instructions. These fingerprints are intended to be
// stable across compilations and across minor changes to the compiler.
class InstFingerprinter {
 public:
  explicit InstFingerprinter(int total_ir_count)
      : fingerprints_(FilesFingerprintStores::MakeWithExplicitSizeFrom(
            total_ir_count, [] {
              return FingerprintStore::MakeForOverwriteWithExplicitSize(0);
            })) {}

  // Gets or computes a fingerprint for the given instruction.
  auto GetOrCompute(const File* file, InstId inst_id) -> uint64_t;

  // Gets or computes a fingerprint for the given instruction block.
  auto GetOrCompute(const File* file, InstBlockId inst_block_id) -> uint64_t;

  // Gets or computes a fingerprint for the given impl.
  auto GetOrCompute(const File* file, ImplId impl_id) -> uint64_t;

  // Gets or computes a fingerprint for the given C++ overload set.
  auto GetOrCompute(const File* file, CppOverloadSetId overload_set_id)
      -> uint64_t;

 private:
  // The fingerprint for each instruction that has had its fingerprint computed,
  // indexed by the InstId's index.
  //
  // TODO: Experiment with also caching fingerprints for instruction blocks once
  // we can get realistic performance measurements for this. This would simplify
  // the `GetOrCompute` overload for `InstBlockId`s, and may save some work if
  // the same canonical inst block is used by multiple instructions, for example
  // as a specific argument list.
  using FingerprintStore = FixedSizeValueStore<InstId, uint64_t>;
  using FilesFingerprintStores =
      FixedSizeValueStore<CheckIRId, FingerprintStore>;
  FilesFingerprintStores fingerprints_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_FINGERPRINTER_H_
