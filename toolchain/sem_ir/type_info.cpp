// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/type_info.h"

#include "common/raw_string_ostream.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto ValueRepr::Print(llvm::raw_ostream& out) const -> void {
  out << "{kind: ";
  switch (kind) {
    case Unknown:
      out << "unknown";
      break;
    case None:
      out << "none";
      break;
    case Copy:
      out << "copy";
      break;
    case Pointer:
      out << "pointer";
      break;
    case Custom:
      out << "custom";
      break;
  }
  out << ", type: " << type_id << "}";
}

auto CompleteTypeInfo::Print(llvm::raw_ostream& out) const -> void {
  out << "{value_rep: " << value_repr << "}";
}

auto ValueRepr::ForType(const File& file, TypeId type_id) -> ValueRepr {
  return file.types().GetValueRepr(type_id);
}

auto ValueRepr::IsCopyOfObjectRepr(const File& file, TypeId orig_type_id) const
    -> bool {
  // If aggregate_kind is ValueAggregate, then the representations are known to
  // be different in some way even, if they're represented by the same type.
  return (kind == SemIR::ValueRepr::Copy || kind == SemIR::ValueRepr::None) &&
         aggregate_kind != SemIR::ValueRepr::ValueAggregate &&
         type_id == file.types().GetObjectRepr(orig_type_id);
}

auto InitRepr::ForType(const File& file, TypeId type_id) -> InitRepr {
  auto value_rep = ValueRepr::ForType(file, type_id);
  switch (value_rep.kind) {
    case ValueRepr::None:
      return {.kind = InitRepr::None};

    case ValueRepr::Copy:
      // TODO: Use in-place initialization for types that have non-trivial
      // destructive move.
      return {.kind = InitRepr::ByCopy};

    case ValueRepr::Pointer:
    case ValueRepr::Custom:
      return {.kind = InitRepr::InPlace};

    case ValueRepr::Unknown:
      return {.kind = InitRepr::Incomplete};
  }
}

auto NumericTypeLiteralInfo::ForType(const File& file, ClassType class_type)
    -> NumericTypeLiteralInfo {
  // Quickly rule out any class that's not a specific.
  if (!class_type.specific_id.has_value()) {
    return NumericTypeLiteralInfo::Invalid;
  }

  // The class must be declared in the `Core` package.
  const auto& class_info = file.classes().Get(class_type.class_id);
  if (!class_info.scope_id.has_value() ||
      !file.name_scopes().IsCorePackage(
          file.name_scopes().Get(class_info.scope_id).parent_scope_id())) {
    return NumericTypeLiteralInfo::Invalid;
  }

  // The class's name must be the name corresponding to a type literal.
  auto name_ident = file.names().GetAsStringIfIdentifier(class_info.name_id);
  if (!name_ident) {
    return NumericTypeLiteralInfo::Invalid;
  }
  Kind kind = llvm::StringSwitch<Kind>(*name_ident)
                  .Case("Int", Int)
                  .Case("UInt", UInt)
                  .Case("Float", Float)
                  .Default(None);
  if (kind == None) {
    return NumericTypeLiteralInfo::Invalid;
  }

  // There must be exactly one argument.
  const auto& specific = file.specifics().Get(class_type.specific_id);
  auto args = file.inst_blocks().Get(specific.args_id);
  if (args.size() != 1) {
    return NumericTypeLiteralInfo::Invalid;
  }

  // And the argument must be an integer value.
  auto width_arg = file.insts().TryGetAs<IntValue>(args[0]);
  if (!width_arg) {
    return NumericTypeLiteralInfo::Invalid;
  }
  return {.kind = kind, .bit_width_id = width_arg->int_id};
}

auto NumericTypeLiteralInfo::PrintLiteral(const File& file,
                                          llvm::raw_ostream& out) const
    -> void {
  CARBON_CHECK(is_valid());
  out << static_cast<char>(kind);
  file.ints().Get(bit_width_id).print(out, /*isSigned=*/false);
}

auto NumericTypeLiteralInfo::GetLiteralAsString(const File& file) const
    -> std::string {
  RawStringOstream out;
  PrintLiteral(file, out);
  return out.TakeStr();
}

}  // namespace Carbon::SemIR
