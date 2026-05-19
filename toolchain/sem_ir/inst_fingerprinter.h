// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_FINGERPRINTER_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_FINGERPRINTER_H_

#include <memory>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "toolchain/base/fixed_size_value_store.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

class HashFingerprintStore;
class StringFingerprintStore;

// Computes fingerprints for instructions. These fingerprints are intended to be
// stable across compilations and across minor changes to the compiler.
template <typename StoreT, typename ResultT>
class InstFingerprinterTemplate {
 public:
  using StoreType = StoreT;
  using ResultType = ResultT;

  explicit InstFingerprinterTemplate(int total_ir_count);
  ~InstFingerprinterTemplate();

  // Gets or computes a fingerprint for the given instruction.
  auto GetOrCompute(const File* file, InstId inst_id) -> ResultType;

  // Gets or computes a fingerprint for the given instruction block.
  auto GetOrCompute(const File* file, InstBlockId inst_block_id) -> ResultType;

  // Gets or computes a fingerprint for the given impl.
  auto GetOrCompute(const File* file, ImplId impl_id) -> ResultType;

  // Gets or computes a fingerprint for the given C++ overload set.
  auto GetOrCompute(const File* file, CppOverloadSetId overload_set_id)
      -> ResultType;

 private:
  std::unique_ptr<StoreT> store_;
};

using HashInstFingerprinter =
    InstFingerprinterTemplate<HashFingerprintStore, uint64_t>;
using StringInstFingerprinter =
    InstFingerprinterTemplate<StringFingerprintStore, llvm::StringRef>;
using InstFingerprinter = HashInstFingerprinter;

extern template class InstFingerprinterTemplate<HashFingerprintStore, uint64_t>;
extern template class InstFingerprinterTemplate<StringFingerprintStore,
                                                llvm::StringRef>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_FINGERPRINTER_H_
