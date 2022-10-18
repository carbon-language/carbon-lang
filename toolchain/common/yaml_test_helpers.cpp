// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/common/yaml_test_helpers.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/YAMLParser.h"

namespace Carbon::Testing::Yaml {

static auto Parse(llvm::yaml::Node* node) -> Value {
  if (!node) {
    return Value{ErrorValue()};
  }

  switch (node->getType()) {
    case llvm::yaml::Node::NK_Null:
      return Value{NullValue()};

    case llvm::yaml::Node::NK_Scalar: {
      llvm::SmallString<128> storage;
      return Value{
          llvm::cast<llvm::yaml::ScalarNode>(*node).getValue(storage).str()};
    }

    case llvm::yaml::Node::NK_BlockScalar:
      return Value{
          llvm::cast<llvm::yaml::BlockScalarNode>(*node).getValue().str()};

    case llvm::yaml::Node::NK_Mapping: {
      MappingValue v;
      for (llvm::yaml::KeyValueNode& kv :
           llvm::cast<llvm::yaml::MappingNode>(*node)) {
        Value key = Parse(kv.getKey());
        Value value = Parse(kv.getValue());
        v.emplace_back(std::move(key), std::move(value));
      }
      return Value{std::move(v)};
    }

    case llvm::yaml::Node::NK_Sequence: {
      SequenceValue v;
      for (llvm::yaml::Node& n : llvm::cast<llvm::yaml::SequenceNode>(*node)) {
        v.push_back(Parse(&n));
      }
      return Value{std::move(v)};
    }

    case llvm::yaml::Node::NK_Alias:
      return Value{AliasValue()};

    case llvm::yaml::Node::NK_KeyValue:
      llvm_unreachable("should only exist as child of mapping");
  }

  llvm_unreachable("unknown yaml node kind");
}

auto Value::FromText(llvm::StringRef text) -> SequenceValue {
  llvm::SourceMgr sm;
  llvm::yaml::Stream yaml_stream(text, sm);

  SequenceValue result;
  for (llvm::yaml::Document& document : yaml_stream) {
    result.push_back(Parse(document.getRoot()));
  }
  return result;
}

auto operator<<(std::ostream& os, const Value& v) -> std::ostream& {
  // Variant visitor that prints the value in the form of code to recreate the
  // value.
  struct Printer {
    auto operator()(NullValue /*v*/) -> void { out << "Yaml::NullValue()"; }
    auto operator()(AliasValue /*v*/) -> void { out << "Yaml::AliasValue()"; }
    auto operator()(ErrorValue /*v*/) -> void { out << "Yaml::ErrorValue()"; }
    auto operator()(const ScalarValue& v) -> void { out << std::quoted(v); }
    auto operator()(const MappingValue& v) -> void {
      out << "Yaml::MappingValue{";
      bool first = true;
      for (const auto& [key, value] : v) {
        if (first) {
          first = false;
        } else {
          out << ", ";
        }
        out << "{" << key << ", " << value << "}";
      }
      out << "}";
    }
    auto operator()(const SequenceValue& v) -> void {
      out << "Yaml::SequenceValue{";
      bool first = true;
      for (const auto& value : v) {
        if (first) {
          first = false;
        } else {
          out << ", ";
        }
        out << value;
      }
      out << "}";
    }

    std::ostream& out;
  };
  std::visit(Printer{os}, v);
  return os;
}

}  // namespace Carbon::Testing::Yaml
