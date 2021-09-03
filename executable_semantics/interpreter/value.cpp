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

auto FindInVarValues(const std::string& field, const VarValues& inits)
    -> std::optional<Nonnull<const Value*>> {
  for (auto& i : inits) {
    if (i.first == field) {
      return i.second;
    }
  }
  return std::nullopt;
}

auto FieldsEqual(const VarValues& ts1, const VarValues& ts2) -> bool {
  if (ts1.size() == ts2.size()) {
    for (auto& iter1 : ts1) {
      auto t2 = FindInVarValues(iter1.first, ts2);
      if (!t2) {
        return false;
      }
      if (!TypeEqual(iter1.second, *t2)) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

auto StructValue::FindField(const std::string& name) const
    -> std::optional<Nonnull<const Value*>> {
  for (const TupleElement& element : elements_) {
    if (element.name == name) {
      return element.value;
    }
  }
  return std::nullopt;
}

auto TupleValue::FindField(const std::string& name) const
    -> std::optional<Nonnull<const Value*>> {
  for (const TupleElement& element : elements) {
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
          cast<TupleValue>(*cast<NominalClassValue>(*v).Inits()).FindField(f);
      if (field == std::nullopt) {
        FATAL_RUNTIME_ERROR(source_loc) << "member " << f << " not in " << *v;
      }
      return *field;
    }
    case Value::Kind::TupleValue: {
      std::optional<Nonnull<const Value*>> field =
          cast<TupleValue>(*v).FindField(f);
      if (!field) {
        FATAL_RUNTIME_ERROR(source_loc) << "field " << f << " not in " << *v;
      }
      return *field;
    }
    case Value::Kind::ChoiceType: {
      const auto& choice = cast<ChoiceType>(*v);
      if (!FindInVarValues(f, choice.Alternatives())) {
        FATAL_RUNTIME_ERROR(source_loc)
            << "alternative " << f << " not in " << *v;
      }
      return arena->New<AlternativeConstructorValue>(f, choice.Name());
    }
    default:
      FATAL() << "field access not allowed for value " << *v;
  }
}

}  // namespace

auto Value::GetField(Nonnull<Arena*> arena, const FieldPath& path,
                     SourceLocation source_loc) const -> Nonnull<const Value*> {
  Nonnull<const Value*> value(this);
  for (const std::string& field : path.components) {
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
      std::vector<TupleElement> elements = cast<StructValue>(*value).elements();
      auto it = std::find_if(elements.begin(), elements.end(),
                             [path_begin](const TupleElement& element) {
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
      return SetFieldImpl(arena, cast<NominalClassValue>(*value).Inits(),
                          path_begin, path_end, field_value, source_loc);
    }
    case Value::Kind::TupleValue: {
      std::vector<TupleElement> elements = cast<TupleValue>(*value).Elements();
      auto it = std::find_if(elements.begin(), elements.end(),
                             [path_begin](const TupleElement& element) {
                               return element.name == *path_begin;
                             });
      if (it == elements.end()) {
        FATAL_RUNTIME_ERROR(source_loc)
            << "field " << *path_begin << " not in " << *value;
      }
      it->value = SetFieldImpl(arena, it->value, path_begin + 1, path_end,
                               field_value, source_loc);
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
                      path.components.begin(), path.components.end(),
                      field_value, source_loc);
}

void Value::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case Value::Kind::AlternativeConstructorValue: {
      const auto& alt = cast<AlternativeConstructorValue>(*this);
      out << alt.ChoiceName() << "." << alt.AltName();
      break;
    }
    case Value::Kind::BindingPlaceholderValue: {
      const auto& placeholder = cast<BindingPlaceholderValue>(*this);
      if (placeholder.Name().has_value()) {
        out << *placeholder.Name();
      } else {
        out << "_";
      }
      out << ": " << *placeholder.Type();
      break;
    }
    case Value::Kind::AlternativeValue: {
      const auto& alt = cast<AlternativeValue>(*this);
      out << "alt " << alt.ChoiceName() << "." << alt.AltName() << " "
          << *alt.Argument();
      break;
    }
    case Value::Kind::StructValue: {
      const auto& struct_val = cast<StructValue>(*this);
      out << "{";
      llvm::ListSeparator sep;
      for (const TupleElement& element : struct_val.elements()) {
        out << sep << "." << element.name << " = " << *element.value;
      }
      out << "}";
      break;
    }
    case Value::Kind::NominalClassValue: {
      const auto& s = cast<NominalClassValue>(*this);
      out << cast<NominalClassType>(*s.Type()).Name() << *s.Inits();
      break;
    }
    case Value::Kind::TupleValue: {
      out << "(";
      llvm::ListSeparator sep;
      for (const TupleElement& element : cast<TupleValue>(*this).Elements()) {
        out << sep << element.name << " = " << *element.value;
      }
      out << ")";
      break;
    }
    case Value::Kind::IntValue:
      out << cast<IntValue>(*this).Val();
      break;
    case Value::Kind::BoolValue:
      out << (cast<BoolValue>(*this).Val() ? "true" : "false");
      break;
    case Value::Kind::FunctionValue:
      out << "fun<" << cast<FunctionValue>(*this).Name() << ">";
      break;
    case Value::Kind::PointerValue:
      out << "ptr<" << cast<PointerValue>(*this).Val() << ">";
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
      out << *cast<PointerType>(*this).Type() << "*";
      break;
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*this);
      out << "fn ";
      if (fn_type.Deduced().size() > 0) {
        out << "[";
        unsigned int i = 0;
        for (const auto& deduced : fn_type.Deduced()) {
          if (i != 0) {
            out << ", ";
          }
          out << deduced.name << ":! " << *deduced.type;
          ++i;
        }
        out << "]";
      }
      out << *fn_type.Param() << " -> " << *fn_type.Ret();
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
      out << "class " << cast<NominalClassType>(*this).Name();
      break;
    case Value::Kind::ChoiceType:
      out << "choice " << cast<ChoiceType>(*this).Name();
      break;
    case Value::Kind::VariableType:
      out << cast<VariableType>(*this).Name();
      break;
    case Value::Kind::ContinuationValue: {
      out << "{";
      llvm::ListSeparator sep(" :: ");
      for (Nonnull<const Action*> action :
           *cast<ContinuationValue>(*this).Stack()) {
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
      out.write_escaped(cast<StringValue>(*this).Val());
      out << "\"";
      break;
  }
}

auto CopyVal(Nonnull<Arena*> arena, Nonnull<const Value*> val,
             SourceLocation source_loc) -> Nonnull<const Value*> {
  switch (val->kind()) {
    case Value::Kind::TupleValue: {
      std::vector<TupleElement> elements;
      for (const TupleElement& element : cast<TupleValue>(*val).Elements()) {
        elements.push_back(
            {.name = element.name,
             .value = CopyVal(arena, element.value, source_loc)});
      }
      return arena->New<TupleValue>(std::move(elements));
    }
    case Value::Kind::AlternativeValue: {
      const auto& alt = cast<AlternativeValue>(*val);
      Nonnull<const Value*> arg = CopyVal(arena, alt.Argument(), source_loc);
      return arena->New<AlternativeValue>(alt.AltName(), alt.ChoiceName(), arg);
    }
    case Value::Kind::StructValue: {
      std::vector<TupleElement> elements;
      for (const TupleElement& element : cast<StructValue>(*val).elements()) {
        elements.push_back(
            {.name = element.name,
             .value = CopyVal(arena, element.value, source_loc)});
      }
      return arena->New<StructValue>(std::move(elements));
    }
    case Value::Kind::NominalClassValue: {
      const auto& s = cast<NominalClassValue>(*val);
      Nonnull<const Value*> inits = CopyVal(arena, s.Inits(), source_loc);
      return arena->New<NominalClassValue>(s.Type(), inits);
    }
    case Value::Kind::IntValue:
      return arena->New<IntValue>(cast<IntValue>(*val).Val());
    case Value::Kind::BoolValue:
      return arena->New<BoolValue>(cast<BoolValue>(*val).Val());
    case Value::Kind::FunctionValue: {
      const auto& fn_value = cast<FunctionValue>(*val);
      return arena->New<FunctionValue>(fn_value.Name(), fn_value.Param(),
                                       fn_value.Body());
    }
    case Value::Kind::PointerValue:
      return arena->New<PointerValue>(cast<PointerValue>(*val).Val());
    case Value::Kind::ContinuationValue:
      return arena->New<ContinuationValue>(
          cast<ContinuationValue>(*val).Stack());
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*val);
      return arena->New<FunctionType>(
          fn_type.Deduced(), CopyVal(arena, fn_type.Param(), source_loc),
          CopyVal(arena, fn_type.Ret(), source_loc));
    }
    case Value::Kind::PointerType:
      return arena->New<PointerType>(
          CopyVal(arena, cast<PointerType>(*val).Type(), source_loc));
    case Value::Kind::IntType:
      return arena->New<IntType>();
    case Value::Kind::BoolType:
      return arena->New<BoolType>();
    case Value::Kind::TypeType:
      return arena->New<TypeType>();
    case Value::Kind::AutoType:
      return arena->New<AutoType>();
    case Value::Kind::ContinuationType:
      return arena->New<ContinuationType>();
    case Value::Kind::StringType:
      return arena->New<StringType>();
    case Value::Kind::StringValue:
      return arena->New<StringValue>(cast<StringValue>(*val).Val());
    case Value::Kind::StructType: {
      VarValues fields;
      for (const auto& [name, type] : cast<StructType>(*val).fields()) {
        fields.push_back({name, CopyVal(arena, type, source_loc)});
      }
      return arena->New<StructType>(fields);
    }
    case Value::Kind::VariableType:
    case Value::Kind::NominalClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
      // TODO: These should be copied so that they don't get destructed.
      return val;
  }
}

auto TypeEqual(Nonnull<const Value*> t1, Nonnull<const Value*> t2) -> bool {
  if (t1->kind() != t2->kind()) {
    return false;
  }
  switch (t1->kind()) {
    case Value::Kind::PointerType:
      return TypeEqual(cast<PointerType>(*t1).Type(),
                       cast<PointerType>(*t2).Type());
    case Value::Kind::FunctionType: {
      const auto& fn1 = cast<FunctionType>(*t1);
      const auto& fn2 = cast<FunctionType>(*t2);
      return TypeEqual(fn1.Param(), fn2.Param()) &&
             TypeEqual(fn1.Ret(), fn2.Ret());
    }
    case Value::Kind::StructType: {
      const auto& struct1 = cast<StructType>(*t1);
      const auto& struct2 = cast<StructType>(*t2);
      if (struct1.fields().size() != struct2.fields().size()) {
        return false;
      }
      for (size_t i = 0; i < struct1.fields().size(); ++i) {
        if (struct1.fields()[i].first != struct2.fields()[i].first ||
            !TypeEqual(struct1.fields()[i].second,
                       struct2.fields()[i].second)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::NominalClassType:
      return cast<NominalClassType>(*t1).Name() ==
             cast<NominalClassType>(*t2).Name();
    case Value::Kind::ChoiceType:
      return cast<ChoiceType>(*t1).Name() == cast<ChoiceType>(*t2).Name();
    case Value::Kind::TupleValue: {
      const auto& tup1 = cast<TupleValue>(*t1);
      const auto& tup2 = cast<TupleValue>(*t2);
      if (tup1.Elements().size() != tup2.Elements().size()) {
        return false;
      }
      for (size_t i = 0; i < tup1.Elements().size(); ++i) {
        if (tup1.Elements()[i].name != tup2.Elements()[i].name ||
            !TypeEqual(tup1.Elements()[i].value, tup2.Elements()[i].value)) {
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
      return cast<VariableType>(*t1).Name() == cast<VariableType>(*t2).Name();
    default:
      FATAL() << "TypeEqual used to compare non-type values\n"
              << *t1 << "\n"
              << *t2;
  }
}

// Returns true if all the fields of the two tuples contain equal values
// and returns false otherwise.
static auto FieldsValueEqual(const std::vector<TupleElement>& ts1,
                             const std::vector<TupleElement>& ts2,
                             SourceLocation source_loc) -> bool {
  if (ts1.size() != ts2.size()) {
    return false;
  }
  for (const TupleElement& element : ts1) {
    auto iter = std::find_if(
        ts2.begin(), ts2.end(),
        [&](const TupleElement& e2) { return e2.name == element.name; });
    if (iter == ts2.end()) {
      return false;
    }
    if (!ValueEqual(element.value, iter->value, source_loc)) {
      return false;
    }
  }
  return true;
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
      return cast<IntValue>(*v1).Val() == cast<IntValue>(*v2).Val();
    case Value::Kind::BoolValue:
      return cast<BoolValue>(*v1).Val() == cast<BoolValue>(*v2).Val();
    case Value::Kind::PointerValue:
      return cast<PointerValue>(*v1).Val() == cast<PointerValue>(*v2).Val();
    case Value::Kind::FunctionValue: {
      std::optional<Nonnull<const Statement*>> body1 =
          cast<FunctionValue>(*v1).Body();
      std::optional<Nonnull<const Statement*>> body2 =
          cast<FunctionValue>(*v2).Body();
      return body1.has_value() == body2.has_value() &&
             (!body1.has_value() || *body1 == *body2);
    }
    case Value::Kind::TupleValue:
      return FieldsValueEqual(cast<TupleValue>(*v1).Elements(),
                              cast<TupleValue>(*v2).Elements(), source_loc);
    case Value::Kind::StructValue:
      return FieldsValueEqual(cast<StructValue>(*v1).elements(),
                              cast<StructValue>(*v2).elements(), source_loc);
    case Value::Kind::StringValue:
      return cast<StringValue>(*v1).Val() == cast<StringValue>(*v2).Val();
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

}  // namespace Carbon
