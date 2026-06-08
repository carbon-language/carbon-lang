// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/shared_value_stores.h"

#include "toolchain/base/canonical_value_store_impl.h"
#include "toolchain/base/value_store_impl.h"

namespace Carbon {

template class CanonicalValueStore<FloatId, llvm::APFloat>;
template class CanonicalValueStore<IdentifierId, llvm::StringRef>;
template class CanonicalValueStore<StringLiteralValueId, llvm::StringRef>;
template class ValueStore<FloatId, llvm::APFloat>;
template class ValueStore<IdentifierId, llvm::StringRef>;
template class ValueStore<RealId, Real>;
template class ValueStore<StringLiteralValueId, llvm::StringRef>;

auto SharedValueStores::OutputYaml(
    std::optional<llvm::StringRef> filename) const -> Yaml::OutputMapping {
  return Yaml::OutputMapping([&, filename](Yaml::OutputMapping::Map map) {
    if (filename) {
      map.Add("filename", *filename);
    }
    map.Add("shared_values",
            Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
              map.Add("ints", ints_.OutputYaml());
              map.Add("reals", reals_.OutputYaml());
              map.Add("floats", floats_.OutputYaml());
              map.Add("identifiers", identifiers_.OutputYaml());
              map.Add("strings", string_literals_.OutputYaml());
            }));
  });
}

}  // namespace Carbon
