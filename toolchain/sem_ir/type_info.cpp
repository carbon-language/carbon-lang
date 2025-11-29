// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/type_info.h"

#include <string>

#include "common/raw_string_ostream.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto ValueRepr::Print(llvm::raw_ostream& out) const -> void {
  out << "{kind: ";
  switch (kind) {
    case Unknown:
      out << "unknown";
      break;
    case Dependent:
      out << "dependent";
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
  return (kind == ValueRepr::Copy || kind == ValueRepr::None) &&
         aggregate_kind != ValueRepr::ValueAggregate &&
         type_id == file.types().GetObjectRepr(orig_type_id);
}

auto InitRepr::ForType(const File& file, TypeId type_id) -> InitRepr {
  auto value_rep = ValueRepr::ForType(file, type_id);
  switch (value_rep.kind) {
    case ValueRepr::None:
      return {.kind = InitRepr::None};

    case ValueRepr::Dependent:
      return {.kind = InitRepr::Dependent};

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
  if (!file.name_scopes().IsInCorePackageRoot(class_info.scope_id)) {
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

// Returns whether this kind of recognized type should have a generic argument
// list.
static auto ExpectsArgs(RecognizedTypeInfo::Kind kind) -> bool {
  return kind == RecognizedTypeInfo::Optional;
}

auto RecognizedTypeInfo::ForType(const File& file, ClassType class_type)
    -> RecognizedTypeInfo {
  auto args_id = SemIR::InstBlockId::None;

  if (class_type.specific_id.has_value()) {
    auto numeric = NumericTypeLiteralInfo::ForType(file, class_type);
    if (numeric.is_valid()) {
      return {.kind = Numeric, .numeric = numeric};
    }
    args_id = file.specifics().Get(class_type.specific_id).args_id;
  }

  // The class must be declared in the `Core` package. We check for up to one
  // level of enclosing namespace.
  const auto& class_info = file.classes().Get(class_type.class_id);
  auto parent_scope_name_id = SemIR::NameId::None;
  if (!file.name_scopes().IsInCorePackageRoot(class_info.scope_id)) {
    if (!file.name_scopes().IsInCorePackageRoot(class_info.parent_scope_id)) {
      return {.kind = None};
    }
    parent_scope_name_id =
        file.name_scopes().Get(class_info.parent_scope_id).name_id();
    if (!parent_scope_name_id.has_value()) {
      return {.kind = None};
    }
  }

  // The class's name must be the name corresponding to a type literal.
  auto name_ident = file.names().GetAsStringIfIdentifier(class_info.name_id);
  if (!name_ident) {
    return {.kind = None};
  }

  if (!parent_scope_name_id.has_value()) {
    Kind kind = llvm::StringSwitch<Kind>(*name_ident)
                    .Case("Char", Char)
                    .Case("Optional", Optional)
                    .Case("String", Str)
                    .Default(None);
    if (ExpectsArgs(kind) == args_id.has_value()) {
      return {.kind = kind, .args_id = args_id};
    }
  }

  auto parent_name_ident =
      file.names().GetAsStringIfIdentifier(parent_scope_name_id);
  if (parent_name_ident == "CppCompat") {
    Kind kind = llvm::StringSwitch<Kind>(*name_ident)
                    .Case("Long32", CppLong32)
                    .Case("ULong32", CppULong32)
                    .Case("LongLong64", CppLongLong64)
                    .Case("ULongLong64", CppULongLong64)
                    .Case("NullptrT", CppNullptrT)
                    .Case("VoidBase", CppVoidBase)
                    .Default(None);
    if (ExpectsArgs(kind) == args_id.has_value()) {
      return {.kind = kind, .args_id = args_id};
    }
  }

  return {.kind = None};
}

// Prints a `Core.CppCompat` integer type name. Typically, when the C++ integer
// type width matches the Carbon type width, prints
// `Cpp.builtin_type` and returns true. Otherwise returns false.
static auto PrintCppCompatLiteral(
    const File& file, clang::CanQualType clang::ASTContext::* qual_type_member,
    unsigned int carbon_bit_width, llvm::StringRef cpp_builtin_name,
    llvm::raw_ostream& out) -> bool {
  if (file.clang_ast_unit()) {
    const clang::ASTContext& ast_context =
        file.clang_ast_unit()->getASTContext();
    if (ast_context.getIntWidth(ast_context.*qual_type_member) ==
        carbon_bit_width) {
      out << "Cpp." << cpp_builtin_name;
      return true;
    }
  }
  return false;
}

auto RecognizedTypeInfo::PrintLiteral(const File& file,
                                      llvm::raw_ostream& out) const -> bool {
  switch (kind) {
    case None:
      CARBON_FATAL("Printing invalid type literal");
    case Numeric:
      numeric.PrintLiteral(file, out);
      return true;
    case Char:
      out << "char";
      return true;
    case CppLong32:
      return PrintCppCompatLiteral(file, &clang::ASTContext::LongTy, 32, "long",
                                   out);
    case CppULong32:
      return PrintCppCompatLiteral(file, &clang::ASTContext::UnsignedLongTy, 32,
                                   "unsigned_long", out);
    case CppLongLong64:
      return PrintCppCompatLiteral(file, &clang::ASTContext::LongLongTy, 64,
                                   "long_long", out);
    case CppULongLong64:
      return PrintCppCompatLiteral(file, &clang::ASTContext::UnsignedLongLongTy,
                                   64, "unsigned_long_long", out);
    case CppNullptrT:
      if (file.clang_ast_unit()) {
        out << "Cpp.nullptr_t";
        return true;
      }
      break;
    case CppVoidBase:
      if (file.clang_ast_unit()) {
        out << "Cpp.void";
        return true;
      }
      break;
    case Optional:
      break;
    case Str:
      out << "str";
      return true;
  }

  return false;
}

}  // namespace Carbon::SemIR
