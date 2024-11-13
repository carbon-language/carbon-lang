// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_SHARED_VALUE_STORES_H_
#define CARBON_TOOLCHAIN_BASE_SHARED_VALUE_STORES_H_

#include "toolchain/base/int.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

// Stores that will be used across compiler phases for a given compilation unit.
// This is provided mainly so that they don't need to be passed separately.
class SharedValueStores : public Yaml::Printable<SharedValueStores> {
 public:
  // Provide types that can be used by APIs to forward access to these stores.
  using IntStore = IntStore;
  using RealStore = ValueStore<RealId>;
  using FloatStore = CanonicalValueStore<FloatId>;
  using IdentifierStore = CanonicalValueStore<IdentifierId>;
  using StringLiteralStore = CanonicalValueStore<StringLiteralValueId>;

  explicit SharedValueStores() = default;

  // Not copyable or movable.
  SharedValueStores(const SharedValueStores&) = delete;
  auto operator=(const SharedValueStores&) -> SharedValueStores& = delete;

  auto identifiers() -> IdentifierStore& { return identifiers_; }
  auto identifiers() const -> const IdentifierStore& { return identifiers_; }
  auto ints() -> IntStore& { return ints_; }
  auto ints() const -> const IntStore& { return ints_; }
  auto reals() -> RealStore& { return reals_; }
  auto reals() const -> const RealStore& { return reals_; }
  auto floats() -> FloatStore& { return floats_; }
  auto floats() const -> const FloatStore& { return floats_; }
  auto string_literal_values() -> StringLiteralStore& {
    return string_literals_;
  }
  auto string_literal_values() const -> const StringLiteralStore& {
    return string_literals_;
  }

  auto OutputYaml(std::optional<llvm::StringRef> filename = std::nullopt) const
      -> Yaml::OutputMapping {
    return Yaml::OutputMapping([&, filename](Yaml::OutputMapping::Map map) {
      if (filename) {
        map.Add("filename", *filename);
      }
      map.Add("shared_values",
              Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                map.Add("ints", ints_.OutputYaml());
                map.Add("reals", reals_.OutputYaml());
                map.Add("identifiers", identifiers_.OutputYaml());
                map.Add("strings", string_literals_.OutputYaml());
              }));
    });
  }

  // Collects memory usage for the various shared stores.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "ints_"), ints_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "reals_"), reals_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "floats_"), floats_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "identifiers_"),
                      identifiers_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "string_literals_"),
                      string_literals_);
  }

 private:
  IntStore ints_;
  RealStore reals_;
  FloatStore floats_;

  IdentifierStore identifiers_;
  StringLiteralStore string_literals_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_SHARED_VALUE_STORES_H_
