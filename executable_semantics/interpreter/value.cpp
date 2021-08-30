// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/value.h"

#include <algorithm>

#include "common/check.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

auto FindInVarValues(const std::string& field, const VarValues& inits)
    -> const Value* {
  for (auto& i : inits) {
    if (i.first == field) {
      return i.second;
    }
  }
  return nullptr;
}

auto FieldsEqual(const VarValues& ts1, const VarValues& ts2) -> bool {
  if (ts1.size() == ts2.size()) {
    for (auto& iter1 : ts1) {
      auto t2 = FindInVarValues(iter1.first, ts2);
      if (t2 == nullptr) {
        return false;
      }
      if (!TypeEqual(iter1.second, t2)) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

auto TupleValue::FindField(const std::string& name) const -> const Value* {
  for (const TupleElement& element : elements) {
    if (element.name == name) {
      return element.value;
    }
  }
  return nullptr;
}

namespace {

auto GetMember(const Value* v, const std::string& f, SourceLocation loc)
    -> const Value* {
  switch (v->Tag()) {
    case Value::Kind::StructValue: {
      const Value* field =
          cast<TupleValue>(*cast<StructValue>(*v).Inits()).FindField(f);
      if (field == nullptr) {
        FATAL_RUNTIME_ERROR(loc) << "member " << f << " not in " << *v;
      }
      return field;
    }
    case Value::Kind::TupleValue: {
      const Value* field = cast<TupleValue>(*v).FindField(f);
      if (field == nullptr) {
        FATAL_RUNTIME_ERROR(loc) << "field " << f << " not in " << *v;
      }
      return field;
    }
    case Value::Kind::ChoiceType: {
      const auto& choice = cast<ChoiceType>(*v);
      if (FindInVarValues(f, choice.Alternatives()) == nullptr) {
        FATAL_RUNTIME_ERROR(loc) << "alternative " << f << " not in " << *v;
      }
      return global_arena->RawNew<AlternativeConstructorValue>(f,
                                                               choice.Name());
    }
    default:
      FATAL() << "field access not allowed for value " << *v;
  }
}

}  // namespace

auto Value::GetField(const FieldPath& path, SourceLocation loc) const
    -> const Value* {
  const Value* value = this;
  for (const std::string& field : path.components) {
    value = GetMember(value, field, loc);
  }
  return value;
}

namespace {

auto SetFieldImpl(const Value* value,
                  std::vector<std::string>::const_iterator path_begin,
                  std::vector<std::string>::const_iterator path_end,
                  const Value* field_value, SourceLocation loc)
    -> const Value* {
  if (path_begin == path_end) {
    return field_value;
  }
  switch (value->Tag()) {
    case Value::Kind::StructValue: {
      return SetFieldImpl(cast<StructValue>(*value).Inits(), path_begin,
                          path_end, field_value, loc);
    }
    case Value::Kind::TupleValue: {
      std::vector<TupleElement> elements = cast<TupleValue>(*value).Elements();
      auto it = std::find_if(elements.begin(), elements.end(),
                             [path_begin](const TupleElement& element) {
                               return element.name == *path_begin;
                             });
      if (it == elements.end()) {
        FATAL_RUNTIME_ERROR(loc)
            << "field " << *path_begin << " not in " << *value;
      }
      it->value =
          SetFieldImpl(it->value, path_begin + 1, path_end, field_value, loc);
      return global_arena->RawNew<TupleValue>(elements);
    }
    default:
      FATAL() << "field access not allowed for value " << *value;
  }
}

}  // namespace

auto Value::SetField(const FieldPath& path, const Value* field_value,
                     SourceLocation loc) const -> const Value* {
  return SetFieldImpl(this, path.components.begin(), path.components.end(),
                      field_value, loc);
}

void Value::Print(llvm::raw_ostream& out) const {
  switch (Tag()) {
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
      const auto& s = cast<StructValue>(*this);
      out << cast<ClassType>(*s.Type()).Name() << *s.Inits();
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
    case Value::Kind::ClassType:
      out << "struct " << cast<ClassType>(*this).Name();
      break;
    case Value::Kind::ChoiceType:
      out << "choice " << cast<ChoiceType>(*this).Name();
      break;
    case Value::Kind::VariableType:
      out << cast<VariableType>(*this).Name();
      break;
    case Value::Kind::ContinuationValue:
      out << "continuation";
      // TODO: Find a way to print useful information about the continuation
      // without creating a dependency cycle.
      break;
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

auto CopyVal(const Value* val, SourceLocation loc) -> const Value* {
  switch (val->Tag()) {
    case Value::Kind::TupleValue: {
      std::vector<TupleElement> elements;
      for (const TupleElement& element : cast<TupleValue>(*val).Elements()) {
        elements.push_back(
            {.name = element.name, .value = CopyVal(element.value, loc)});
      }
      return global_arena->RawNew<TupleValue>(std::move(elements));
    }
    case Value::Kind::AlternativeValue: {
      const auto& alt = cast<AlternativeValue>(*val);
      const Value* arg = CopyVal(alt.Argument(), loc);
      return global_arena->RawNew<AlternativeValue>(alt.AltName(),
                                                    alt.ChoiceName(), arg);
    }
    case Value::Kind::StructValue: {
      const auto& s = cast<StructValue>(*val);
      const Value* inits = CopyVal(s.Inits(), loc);
      return global_arena->RawNew<StructValue>(s.Type(), inits);
    }
    case Value::Kind::IntValue:
      return global_arena->RawNew<IntValue>(cast<IntValue>(*val).Val());
    case Value::Kind::BoolValue:
      return global_arena->RawNew<BoolValue>(cast<BoolValue>(*val).Val());
    case Value::Kind::FunctionValue: {
      const auto& fn_value = cast<FunctionValue>(*val);
      return global_arena->RawNew<FunctionValue>(
          fn_value.Name(), fn_value.Param(), fn_value.Body());
    }
    case Value::Kind::PointerValue:
      return global_arena->RawNew<PointerValue>(cast<PointerValue>(*val).Val());
    case Value::Kind::ContinuationValue:
      // Copying a continuation is "shallow".
      return val;
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*val);
      return global_arena->RawNew<FunctionType>(fn_type.Deduced(),
                                                CopyVal(fn_type.Param(), loc),
                                                CopyVal(fn_type.Ret(), loc));
    }
    case Value::Kind::PointerType:
      return global_arena->RawNew<PointerType>(
          CopyVal(cast<PointerType>(*val).Type(), loc));
    case Value::Kind::IntType:
      return global_arena->RawNew<IntType>();
    case Value::Kind::BoolType:
      return global_arena->RawNew<BoolType>();
    case Value::Kind::TypeType:
      return global_arena->RawNew<TypeType>();
    case Value::Kind::AutoType:
      return global_arena->RawNew<AutoType>();
    case Value::Kind::ContinuationType:
      return global_arena->RawNew<ContinuationType>();
    case Value::Kind::StringType:
      return global_arena->RawNew<StringType>();
    case Value::Kind::StringValue:
      return global_arena->RawNew<StringValue>(cast<StringValue>(*val).Val());
    case Value::Kind::VariableType:
    case Value::Kind::ClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
      // TODO: These should be copied so that they don't get destructed.
      return val;
  }
}

auto TypeEqual(const Value* t1, const Value* t2) -> bool {
  if (t1->Tag() != t2->Tag()) {
    return false;
  }
  switch (t1->Tag()) {
    case Value::Kind::PointerType:
      return TypeEqual(cast<PointerType>(*t1).Type(),
                       cast<PointerType>(*t2).Type());
    case Value::Kind::FunctionType: {
      const auto& fn1 = cast<FunctionType>(*t1);
      const auto& fn2 = cast<FunctionType>(*t2);
      return TypeEqual(fn1.Param(), fn2.Param()) &&
             TypeEqual(fn1.Ret(), fn2.Ret());
    }
    case Value::Kind::ClassType:
      return cast<ClassType>(*t1).Name() == cast<ClassType>(*t2).Name();
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
                             SourceLocation loc) -> bool {
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
    if (!ValueEqual(element.value, iter->value, loc)) {
      return false;
    }
  }
  return true;
}

// Returns true if the two values are equal and returns false otherwise.
//
// This function implements the `==` operator of Carbon.
auto ValueEqual(const Value* v1, const Value* v2, SourceLocation loc) -> bool {
  if (v1->Tag() != v2->Tag()) {
    return false;
  }
  switch (v1->Tag()) {
    case Value::Kind::IntValue:
      return cast<IntValue>(*v1).Val() == cast<IntValue>(*v2).Val();
    case Value::Kind::BoolValue:
      return cast<BoolValue>(*v1).Val() == cast<BoolValue>(*v2).Val();
    case Value::Kind::PointerValue:
      return cast<PointerValue>(*v1).Val() == cast<PointerValue>(*v2).Val();
    case Value::Kind::FunctionValue: {
      std::optional<Ptr<const Statement>> body1 =
          cast<FunctionValue>(*v1).Body();
      std::optional<Ptr<const Statement>> body2 =
          cast<FunctionValue>(*v2).Body();
      return body1.has_value() == body2.has_value() &&
             (!body1.has_value() || *body1 == *body2);
    }
    case Value::Kind::TupleValue:
      return FieldsValueEqual(cast<TupleValue>(*v1).Elements(),
                              cast<TupleValue>(*v2).Elements(), loc);
    case Value::Kind::StringValue:
      return cast<StringValue>(*v1).Val() == cast<StringValue>(*v2).Val();
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::AutoType:
    case Value::Kind::ClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
      return TypeEqual(v1, v2);
    case Value::Kind::StructValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
      FATAL() << "ValueEqual does not support this kind of value: " << *v1;
  }
}

}  // namespace Carbon
