// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/bundle.h"

#include "toolchain/base/block_value_store_impl.h"
#include "toolchain/base/value_store_impl.h"

namespace Carbon::SemIR {

auto BundleStore::OutputYaml() const -> Yaml::OutputMapping {
  return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
    for (auto bundle_id : store_.ids()) {
      AddYamlMapEntry(map, bundle_id);
    }
  });
}

auto BundleStore::OutputBundleYaml(RawBundleId bundle_id) const
    -> Yaml::OutputMapping {
  return Yaml::OutputMapping([this, bundle_id](Yaml::OutputMapping::Map map) {
    AddYamlMapEntry(map, bundle_id);
  });
}

auto BundleStore::CollectMemUsage(MemUsage& mem_usage,
                                  llvm::StringRef label) const -> void {
  store_.CollectMemUsage(mem_usage, label);
  bundle_kind_cache_.CollectMemUsage(mem_usage, label);
}

auto BundleStore::BundleString(RawBundleId bundle_id) const -> std::string {
  RawStringOstream out;
  llvm::ListSeparator sep;
  out << "{";
  for (auto [i, raw_id] : llvm::enumerate(store_.Get(bundle_id))) {
    PrintBundleField(out, sep, i, raw_id);
  }
  out << "}";
  return out.TakeStr();
}

auto BundleStore::AddYamlMapEntry(Yaml::OutputMapping::Map& map,
                                  RawBundleId bundle_id) const -> void {
  auto kind_set = bundle_kind_cache_.GetWithDefault(bundle_id, IdKindSet{});
  if (kind_set.empty()) {
    map.Add(PrintToString(bundle_id),
            Yaml::OutputScalar(BundleString(bundle_id)));
  } else if (kind_set.size() == 1) {
    IdAndKind typed_bundle_id(*kind_set.begin(), bundle_id.index);
    auto bundle_string = typed_bundle_id.Dispatch<std::string>(
        [this](auto id) { return BundleString(id); });
    map.Add(PrintToString(bundle_id), Yaml::OutputScalar(bundle_string));
  } else {
    map.Add(PrintToString(bundle_id), YamlBundleMap(bundle_id, kind_set));
  }
}

auto BundleStore::YamlBundleMap(RawBundleId bundle_id,
                                const IdKindSet& kinds) const
    -> Yaml::OutputMapping {
  return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
    for (auto [i, id_kind] : llvm::enumerate(kinds)) {
      IdAndKind typed_bundle_id(id_kind, bundle_id.index);
      // TODO: make this a YAML sequence instead of a map.
      map.Add(llvm::itostr(i),
              typed_bundle_id.Dispatch<std::string>(
                  [this](auto id) { return BundleString(id); }));
    }
  });
}

}  // namespace Carbon::SemIR
