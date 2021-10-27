// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/value.h"

#include <algorithm>

#include "common/check.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
#include "executable_semantics/interpreter/action.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

auto StructValue::FindField(const std::string& name) const
    -> std::optional<Nonnull<const Value*>> {
  for (const NamedValue& element : elements_) {
    if (element.name == name) {
      return element.value;
    }
  }
  return std::nullopt;
}

namespace {

auto GetMember(Nonnull<Arena*> arena, Nonnull<const Value*> v,
               const std::string& f, SourceLocation source_loc)
    -> Nonnull<const Value*> {
  switch (v->kind()) {
    case Value::Kind::StructValue: {
      std::optional<Nonnull<const Value*>> field =
          cast<StructValue>(*v).FindField(f);
      if (field == std::nullopt) {
        FATAL_RUNTIME_ERROR(source_loc) << "member " << f << " not in " << *v;
      }
      return *field;
    }
    case Value::Kind::NominalClassValue: {
      std::optional<Nonnull<const Value*>> field =
          cast<StructValue>(cast<NominalClassValue>(*v).inits()).FindField(f);
      if (field == std::nullopt) {
        FATAL_RUNTIME_ERROR(source_loc) << "member " << f << " not in " << *v;
      }
      return *field;
    }
    case Value::Kind::ChoiceType: {
      const auto& choice = cast<ChoiceType>(*v);
      if (!choice.FindAlternative(f)) {
        FATAL_RUNTIME_ERROR(source_loc)
            << "alternative " << f << " not in " << *v;
      }
      return arena->New<AlternativeConstructorValue>(f, choice.name());
    }
    default:
      FATAL() << "field access not allowed for value " << *v;
  }
}

}  // namespace

auto Value::GetField(Nonnull<Arena*> arena, const FieldPath& path,
                     SourceLocation source_loc) const -> Nonnull<const Value*> {
  Nonnull<const Value*> value(this);
  for (const std::string& field : path.components_) {
    value = GetMember(arena, value, field, source_loc);
  }
  return value;
}

namespace {

auto SetFieldImpl(Nonnull<Arena*> arena, Nonnull<const Value*> value,
                  std::vector<std::string>::const_iterator path_begin,
                  std::vector<std::string>::const_iterator path_end,
                  Nonnull<const Value*> field_value, SourceLocation source_loc)
    -> Nonnull<const Value*> {
  if (path_begin == path_end) {
    return field_value;
  }
  switch (value->kind()) {
    case Value::Kind::StructValue: {
      std::vector<NamedValue> elements = cast<StructValue>(*value).elements();
      auto it = std::find_if(elements.begin(), elements.end(),
                             [path_begin](const NamedValue& element) {
                               return element.name == *path_begin;
                             });
      if (it == elements.end()) {
        FATAL_RUNTIME_ERROR(source_loc)
            << "field " << *path_begin << " not in " << *value;
      }
      it->value = SetFieldImpl(arena, it->value, path_begin + 1, path_end,
                               field_value, source_loc);
      return arena->New<StructValue>(elements);
    }
    case Value::Kind::NominalClassValue: {
      return SetFieldImpl(arena, &cast<NominalClassValue>(*value).inits(),
                          path_begin, path_end, field_value, source_loc);
    }
    case Value::Kind::TupleValue: {
      std::vector<Nonnull<const Value*>> elements =
          cast<TupleValue>(*value).elements();
      // TODO(geoffromer): update FieldPath to hold integers as well as strings.
      int index = std::stoi(*path_begin);
      if (index < 0 || static_cast<size_t>(index) >= elements.size()) {
        FATAL_RUNTIME_ERROR(source_loc)
            << "index " << *path_begin << " out of range in " << *value;
      }
      elements[index] = SetFieldImpl(arena, elements[index], path_begin + 1,
                                     path_end, field_value, source_loc);
      return arena->New<TupleValue>(elements);
    }
    default:
      FATAL() << "field access not allowed for value " << *value;
  }
}

}  // namespace

auto Value::SetField(Nonnull<Arena*> arena, const FieldPath& path,
                     Nonnull<const Value*> field_value,
                     SourceLocation source_loc) const -> Nonnull<const Value*> {
  return SetFieldImpl(arena, Nonnull<const Value*>(this),
                      path.components_.begin(), path.components_.end(),
                      field_value, source_loc);
}

void Value::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case Value::Kind::AlternativeConstructorValue: {
      const auto& alt = cast<AlternativeConstructorValue>(*this);
      out << alt.choice_name() << "." << alt.alt_name();
      break;
    }
    case Value::Kind::BindingPlaceholderValue: {
      const auto& placeholder = cast<BindingPlaceholderValue>(*this);
      if (placeholder.name().has_value()) {
        out << *placeholder.name();
      } else {
        out << "_";
      }
      out << ": " << placeholder.type();
      break;
    }
    case Value::Kind::AlternativeValue: {
      const auto& alt = cast<AlternativeValue>(*this);
      out << "alt " << alt.choice_name() << "." << alt.alt_name() << " "
          << alt.argument();
      break;
    }
    case Value::Kind::StructValue: {
      const auto& struct_val = cast<StructValue>(*this);
      out << "{";
      llvm::ListSeparator sep;
      for (const NamedValue& element : struct_val.elements()) {
        out << sep << "." << element.name << " = " << *element.value;
      }
      out << "}";
      break;
    }
    case Value::Kind::NominalClassValue: {
      const auto& s = cast<NominalClassValue>(*this);
      out << cast<NominalClassType>(s.type()).name() << s.inits();
      break;
    }
    case Value::Kind::TupleValue: {
      out << "(";
      llvm::ListSeparator sep;
      for (Nonnull<const Value*> element : cast<TupleValue>(*this).elements()) {
        out << sep << *element;
      }
      out << ")";
      break;
    }
    case Value::Kind::IntValue:
      out << cast<IntValue>(*this).value();
      break;
    case Value::Kind::BoolValue:
      out << (cast<BoolValue>(*this).value() ? "true" : "false");
      break;
    case Value::Kind::FunctionValue:
      out << "fun<" << cast<FunctionValue>(*this).declaration().name() << ">";
      break;
    case Value::Kind::PointerValue:
      out << "ptr<" << cast<PointerValue>(*this).value() << ">";
      break;
    case Value::Kind::BoolType:
      out << "Bool";
      break;
    case Value::Kind::IntType:
      out << "i32";
      break;
    case Value::Kind::TypeType:
      out << "Type";
      break;
    case Value::Kind::AutoType:
      out << "auto";
      break;
    case Value::Kind::ContinuationType:
      out << "Continuation";
      break;
    case Value::Kind::PointerType:
      out << cast<PointerType>(*this).type() << "*";
      break;
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*this);
      out << "fn ";
      if (fn_type.deduced().size() > 0) {
        out << "[";
        unsigned int i = 0;
        for (const auto& deduced : fn_type.deduced()) {
          if (i != 0) {
            out << ", ";
          }
          out << deduced.name << ":! " << *deduced.type;
          ++i;
        }
        out << "]";
      }
      out << fn_type.parameters() << " -> " << fn_type.return_type();
      break;
    }
    case Value::Kind::StructType: {
      out << "{";
      llvm::ListSeparator sep;
      for (const auto& [name, type] : cast<StructType>(*this).fields()) {
        out << sep << "." << name << ": " << *type;
      }
      out << "}";
      break;
    }
    case Value::Kind::NominalClassType:
      out << "class " << cast<NominalClassType>(*this).name();
      break;
    case Value::Kind::ChoiceType:
      out << "choice " << cast<ChoiceType>(*this).name();
      break;
    case Value::Kind::VariableType:
      out << cast<VariableType>(*this).name();
      break;
    case Value::Kind::ContinuationValue: {
      out << "{";
      llvm::ListSeparator sep(" :: ");
      for (Nonnull<const Action*> action :
           cast<ContinuationValue>(*this).stack()) {
        out << sep << *action;
      }
      out << "}";
      break;
    }
    case Value::Kind::StringType:
      out << "String";
      break;
    case Value::Kind::StringValue:
      out << "\"";
      out.write_escaped(cast<StringValue>(*this).value());
      out << "\"";
      break;
  }
}

auto TypeEqual(Nonnull<const Value*> t1, Nonnull<const Value*> t2) -> bool {
  if (t1->kind() != t2->kind()) {
    return false;
  }
  switch (t1->kind()) {
    case Value::Kind::PointerType:
      return TypeEqual(&cast<PointerType>(*t1).type(),
                       &cast<PointerType>(*t2).type());
    case Value::Kind::FunctionType: {
      const auto& fn1 = cast<FunctionType>(*t1);
      const auto& fn2 = cast<FunctionType>(*t2);
      return TypeEqual(&fn1.parameters(), &fn2.parameters()) &&
             TypeEqual(&fn1.return_type(), &fn2.return_type());
    }
    case Value::Kind::StructType: {
      const auto& struct1 = cast<StructType>(*t1);
      const auto& struct2 = cast<StructType>(*t2);
      if (struct1.fields().size() != struct2.fields().size()) {
        return false;
      }
      for (size_t i = 0; i < struct1.fields().size(); ++i) {
        if (struct1.fields()[i].name != struct2.fields()[i].name ||
            !TypeEqual(struct1.fields()[i].value, struct2.fields()[i].value)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::NominalClassType:
      return cast<NominalClassType>(*t1).name() ==
             cast<NominalClassType>(*t2).name();
    case Value::Kind::ChoiceType:
      return cast<ChoiceType>(*t1).name() == cast<ChoiceType>(*t2).name();
    case Value::Kind::TupleValue: {
      const auto& tup1 = cast<TupleValue>(*t1);
      const auto& tup2 = cast<TupleValue>(*t2);
      if (tup1.elements().size() != tup2.elements().size()) {
        return false;
      }
      for (size_t i = 0; i < tup1.elements().size(); ++i) {
        if (!TypeEqual(tup1.elements()[i], tup2.elements()[i])) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::ContinuationType:
    case Value::Kind::TypeType:
    case Value::Kind::StringType:
      return true;
    case Value::Kind::VariableType:
      return cast<VariableType>(*t1).name() == cast<VariableType>(*t2).name();
    default:
      FATAL() << "TypeEqual used to compare non-type values\n"
              << *t1 << "\n"
              << *t2;
  }
}

// Returns true if the two values are equal and returns false otherwise.
//
// This function implements the `==` operator of Carbon.
auto ValueEqual(Nonnull<const Value*> v1, Nonnull<const Value*> v2,
                SourceLocation source_loc) -> bool {
  if (v1->kind() != v2->kind()) {
    return false;
  }
  switch (v1->kind()) {
    case Value::Kind::IntValue:
      return cast<IntValue>(*v1).value() == cast<IntValue>(*v2).value();
    case Value::Kind::BoolValue:
      return cast<BoolValue>(*v1).value() == cast<BoolValue>(*v2).value();
    case Value::Kind::PointerValue:
      return cast<PointerValue>(*v1).value() == cast<PointerValue>(*v2).value();
    case Value::Kind::FunctionValue: {
      std::optional<Nonnull<const Statement*>> body1 =
          cast<FunctionValue>(*v1).declaration().body();
      std::optional<Nonnull<const Statement*>> body2 =
          cast<FunctionValue>(*v2).declaration().body();
      return body1.has_value() == body2.has_value() &&
             (!body1.has_value() || *body1 == *body2);
    }
    case Value::Kind::TupleValue: {
      const std::vector<Nonnull<const Value*>>& elements1 =
          cast<TupleValue>(*v1).elements();
      const std::vector<Nonnull<const Value*>>& elements2 =
          cast<TupleValue>(*v2).elements();
      if (elements1.size() != elements2.size()) {
        return false;
      }
      for (size_t i = 0; i < elements1.size(); ++i) {
        if (!ValueEqual(elements1[i], elements2[i], source_loc)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::StructValue: {
      const auto& struct_v1 = cast<StructValue>(*v1);
      const auto& struct_v2 = cast<StructValue>(*v2);
      CHECK(struct_v1.elements().size() == struct_v2.elements().size());
      for (size_t i = 0; i < struct_v1.elements().size(); ++i) {
        CHECK(struct_v1.elements()[i].name == struct_v2.elements()[i].name);
        if (!ValueEqual(struct_v1.elements()[i].value,
                        struct_v2.elements()[i].value, source_loc)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::StringValue:
      return cast<StringValue>(*v1).value() == cast<StringValue>(*v2).value();
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::AutoType:
    case Value::Kind::StructType:
    case Value::Kind::NominalClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
      return TypeEqual(v1, v2);
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
      FATAL() << "ValueEqual does not support this kind of value: " << *v1;
  }
}

auto ChoiceType::FindAlternative(std::string_view name) const
    -> std::optional<Nonnull<const Value*>> {
  for (const NamedValue& alternative : alternatives_) {
    if (alternative.name == name) {
      return alternative.value;
    }
  }
  return std::nullopt;
}

}  // namespace Carbon
