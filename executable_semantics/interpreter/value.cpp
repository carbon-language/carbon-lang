// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/value.h"

#include <algorithm>

#include "common/check.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon {

auto Value::GetIntValue() const -> int {
  return std::get<IntValue>(value).value;
}

auto Value::GetBoolValue() const -> bool {
  return std::get<BoolValue>(value).value;
}

auto Value::GetFunctionValue() const -> const FunctionValue& {
  return std::get<FunctionValue>(value);
}

auto Value::GetStructValue() const -> const StructValue& {
  return std::get<StructValue>(value);
}

auto Value::GetAlternativeConstructorValue() const
    -> const AlternativeConstructorValue& {
  return std::get<AlternativeConstructorValue>(value);
}

auto Value::GetAlternativeValue() const -> const AlternativeValue& {
  return std::get<AlternativeValue>(value);
}

auto Value::GetTupleValue() const -> const TupleValue& {
  return std::get<TupleValue>(value);
}

auto Value::GetPointerValue() const -> Address {
  return std::get<PointerValue>(value).value;
}

auto Value::GetBindingPlaceholderValue() const
    -> const BindingPlaceholderValue& {
  return std::get<BindingPlaceholderValue>(value);
}

auto Value::GetFunctionType() const -> const FunctionType& {
  return std::get<FunctionType>(value);
}

auto Value::GetPointerType() const -> const PointerType& {
  return std::get<PointerType>(value);
}

auto Value::GetStructType() const -> const StructType& {
  return std::get<StructType>(value);
}

auto Value::GetChoiceType() const -> const ChoiceType& {
  return std::get<ChoiceType>(value);
}

auto Value::GetVariableType() const -> const VariableType& {
  return std::get<VariableType>(value);
}

auto Value::GetContinuationValue() const -> const ContinuationValue& {
  return std::get<ContinuationValue>(value);
}

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
      if (not TypeEqual(iter1.second, t2)) {
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

auto Value::MakeIntValue(int i) -> const Value* {
  auto* v = new Value();
  v->value = IntValue({.value = i});
  return v;
}

auto Value::MakeBoolValue(bool b) -> const Value* {
  auto* v = new Value();
  v->value = BoolValue({.value = b});
  return v;
}

auto Value::MakeFunctionValue(std::string name, const Value* param,
                              const Statement* body) -> const Value* {
  auto* v = new Value();
  v->value =
      FunctionValue({.name = std::move(name), .param = param, .body = body});
  return v;
}

auto Value::MakePointerValue(Address addr) -> const Value* {
  auto* v = new Value();
  v->value = PointerValue({.value = addr});
  return v;
}

auto Value::MakeStructValue(const Value* type, const Value* inits)
    -> const Value* {
  auto* v = new Value();
  v->value = StructValue({.type = type, .inits = inits});
  return v;
}

auto Value::MakeTupleValue(std::vector<TupleElement> elements) -> const Value* {
  auto* v = new Value();
  v->value = TupleValue({.elements = std::move(elements)});
  return v;
}

auto Value::MakeAlternativeValue(std::string alt_name, std::string choice_name,
                                 const Value* argument) -> const Value* {
  auto* v = new Value();
  v->value = AlternativeValue({.alt_name = std::move(alt_name),
                               .choice_name = std::move(choice_name),
                               .argument = argument});
  return v;
}

auto Value::MakeAlternativeConstructorValue(std::string alt_name,
                                            std::string choice_name)
    -> const Value* {
  auto* v = new Value();
  v->value = AlternativeConstructorValue(
      {.alt_name = std::move(alt_name), .choice_name = std::move(choice_name)});
  return v;
}

// Return a first-class continuation represented a fragment
// of the stack.
auto Value::MakeContinuationValue(std::vector<Frame*> stack) -> Value* {
  auto* v = new Value();
  v->value = ContinuationValue({.stack = std::move(stack)});
  return v;
}

auto Value::MakeBindingPlaceholderValue(std::optional<std::string> name,
                                        const Value* type) -> const Value* {
  auto* v = new Value();
  v->value = BindingPlaceholderValue({.name = std::move(name), .type = type});
  return v;
}

auto Value::MakeIntType() -> const Value* {
  auto* v = new Value();
  v->value = IntType();
  return v;
}

auto Value::MakeBoolType() -> const Value* {
  auto* v = new Value();
  v->value = BoolType();
  return v;
}

auto Value::MakeTypeType() -> const Value* {
  auto* v = new Value();
  v->value = TypeType();
  return v;
}

// Return a Continuation type.
auto Value::MakeContinuationType() -> const Value* {
  auto* v = new Value();
  v->value = ContinuationType();
  return v;
}

auto Value::MakeAutoType() -> const Value* {
  auto* v = new Value();
  v->value = AutoType();
  return v;
}

auto Value::MakeFunctionType(std::vector<GenericBinding> deduced_params,
                             const Value* param, const Value* ret)
    -> const Value* {
  auto* v = new Value();
  v->value = FunctionType(
      {.deduced = std::move(deduced_params), .param = param, .ret = ret});
  return v;
}

auto Value::MakePointerType(const Value* type) -> const Value* {
  auto* v = new Value();
  v->value = PointerType({.type = type});
  return v;
}

auto Value::MakeStructType(std::string name, VarValues fields,
                           VarValues methods) -> const Value* {
  auto* v = new Value();
  v->value = StructType({.name = std::move(name),
                         .fields = std::move(fields),
                         .methods = std::move(methods)});
  return v;
}

auto Value::MakeUnitTypeVal() -> const Value* {
  auto* v = new Value();
  v->value = TupleValue({.elements = {}});
  return v;
}

auto Value::MakeChoiceType(std::string name, VarValues alts) -> const Value* {
  auto* v = new Value();
  v->value =
      ChoiceType({.name = std::move(name), .alternatives = std::move(alts)});
  return v;
}

auto Value::MakeVariableType(std::string name) -> const Value* {
  auto* v = new Value();
  v->value = VariableType({.name = std::move(name)});
  return v;
}

namespace {

auto GetMember(const Value* v, const std::string& f, int line_num)
    -> const Value* {
  switch (v->tag()) {
    case ValKind::StructValue: {
      const Value* field =
          v->GetStructValue().inits->GetTupleValue().FindField(f);
      if (field == nullptr) {
        llvm::errs() << "runtime error, member " << f << " not in " << *v
                     << "\n";
        exit(-1);
      }
      return field;
    }
    case ValKind::TupleValue: {
      const Value* field = v->GetTupleValue().FindField(f);
      if (field == nullptr) {
        llvm::errs() << "field " << f << " not in " << *v << "\n";
        exit(-1);
      }
      return field;
    }
    case ValKind::ChoiceType: {
      if (FindInVarValues(f, v->GetChoiceType().alternatives) == nullptr) {
        llvm::errs() << "alternative " << f << " not in " << *v << "\n";
        exit(-1);
      }
      return Value::MakeAlternativeConstructorValue(f, v->GetChoiceType().name);
    }
    default:
      llvm::errs() << "field access not allowed for value " << *v << "\n";
      exit(-1);
  }
}

}  // namespace

auto Value::GetField(const FieldPath& path, int line_num) const
    -> const Value* {
  const Value* value = this;
  for (const std::string& field : path.components) {
    value = GetMember(value, field, line_num);
  }
  return value;
}

namespace {

auto SetFieldImpl(const Value* value,
                  std::vector<std::string>::const_iterator path_begin,
                  std::vector<std::string>::const_iterator path_end,
                  const Value* field_value, int line_num) -> const Value* {
  if (path_begin == path_end) {
    return field_value;
  }
  switch (value->tag()) {
    case ValKind::StructValue: {
      return SetFieldImpl(value->GetStructValue().inits, path_begin, path_end,
                          field_value, line_num);
    }
    case ValKind::TupleValue: {
      std::vector<TupleElement> elements = value->GetTupleValue().elements;
      auto it = std::find_if(elements.begin(), elements.end(),
                             [path_begin](const TupleElement& element) {
                               return element.name == *path_begin;
                             });
      if (it == elements.end()) {
        llvm::errs() << "field " << *path_begin << " not in " << *value << "\n";
        exit(-1);
      }
      it->value = SetFieldImpl(it->value, path_begin + 1, path_end, field_value,
                               line_num);
      return Value::MakeTupleValue(elements);
    }
    default:
      llvm::errs() << "field access not allowed for value " << *value << "\n";
      exit(-1);
  }
}

}  // namespace

auto Value::SetField(const FieldPath& path, const Value* field_value,
                     int line_num) const -> const Value* {
  return SetFieldImpl(this, path.components.begin(), path.components.end(),
                      field_value, line_num);
}

void Value::Print(llvm::raw_ostream& out) const {
  switch (tag()) {
    case ValKind::AlternativeConstructorValue: {
      out << GetAlternativeConstructorValue().choice_name << "."
          << GetAlternativeConstructorValue().alt_name;
      break;
    }
    case ValKind::BindingPlaceholderValue: {
      const BindingPlaceholderValue& placeholder = GetBindingPlaceholderValue();
      if (placeholder.name.has_value()) {
        out << *placeholder.name;
      } else {
        out << "_";
      }
      out << ": " << *placeholder.type;
      break;
    }
    case ValKind::AlternativeValue: {
      out << "alt " << GetAlternativeValue().choice_name << "."
          << GetAlternativeValue().alt_name << " "
          << *GetAlternativeValue().argument;
      break;
    }
    case ValKind::StructValue: {
      out << GetStructValue().type->GetStructType().name
          << *GetStructValue().inits;
      break;
    }
    case ValKind::TupleValue: {
      out << "(";
      llvm::ListSeparator sep;
      for (const TupleElement& element : GetTupleValue().elements) {
        out << sep << element.name << " = " << *element.value;
      }
      out << ")";
      break;
    }
    case ValKind::IntValue:
      out << GetIntValue();
      break;
    case ValKind::BoolValue:
      out << (GetBoolValue() ? "true" : "false");
      break;
    case ValKind::FunctionValue:
      out << "fun<" << GetFunctionValue().name << ">";
      break;
    case ValKind::PointerValue:
      out << "ptr<" << GetPointerValue() << ">";
      break;
    case ValKind::BoolType:
      out << "Bool";
      break;
    case ValKind::IntType:
      out << "Int";
      break;
    case ValKind::TypeType:
      out << "Type";
      break;
    case ValKind::AutoType:
      out << "auto";
      break;
    case ValKind::ContinuationType:
      out << "Continuation";
      break;
    case ValKind::PointerType:
      out << *GetPointerType().type << "*";
      break;
    case ValKind::FunctionType:
      out << "fn ";
      if (GetFunctionType().deduced.size() > 0) {
        out << "[";
        unsigned int i = 0;
        for (const auto& deduced : GetFunctionType().deduced) {
          if (i != 0) {
            out << ", ";
          }
          out << deduced.name << ":! " << *deduced.type;
          ++i;
        }
        out << "]";
      }
      out << *GetFunctionType().param << " -> " << *GetFunctionType().ret;
      break;
    case ValKind::StructType:
      out << "struct " << GetStructType().name;
      break;
    case ValKind::ChoiceType:
      out << "choice " << GetChoiceType().name;
      break;
    case ValKind::VariableType:
      out << GetVariableType().name;
      break;
    case ValKind::ContinuationValue:
      out << "continuation";
      // TODO: Find a way to print useful information about the continuation
      // without creating a dependency cycle.
      break;
  }
}

auto CopyVal(const Value* val, int line_num) -> const Value* {
  switch (val->tag()) {
    case ValKind::TupleValue: {
      std::vector<TupleElement> elements;
      for (const TupleElement& element : val->GetTupleValue().elements) {
        elements.push_back(
            {.name = element.name, .value = CopyVal(element.value, line_num)});
      }
      return Value::MakeTupleValue(std::move(elements));
    }
    case ValKind::AlternativeValue: {
      const Value* arg = CopyVal(val->GetAlternativeValue().argument, line_num);
      return Value::MakeAlternativeValue(val->GetAlternativeValue().alt_name,
                                         val->GetAlternativeValue().choice_name,
                                         arg);
    }
    case ValKind::StructValue: {
      const Value* inits = CopyVal(val->GetStructValue().inits, line_num);
      return Value::MakeStructValue(val->GetStructValue().type, inits);
    }
    case ValKind::IntValue:
      return Value::MakeIntValue(val->GetIntValue());
    case ValKind::BoolValue:
      return Value::MakeBoolValue(val->GetBoolValue());
    case ValKind::FunctionValue:
      return Value::MakeFunctionValue(val->GetFunctionValue().name,
                                      val->GetFunctionValue().param,
                                      val->GetFunctionValue().body);
    case ValKind::PointerValue:
      return Value::MakePointerValue(val->GetPointerValue());
    case ValKind::ContinuationValue:
      // Copying a continuation is "shallow".
      return val;
    case ValKind::FunctionType:
      return Value::MakeFunctionType(
          val->GetFunctionType().deduced,
          CopyVal(val->GetFunctionType().param, line_num),
          CopyVal(val->GetFunctionType().ret, line_num));

    case ValKind::PointerType:
      return Value::MakePointerType(
          CopyVal(val->GetPointerType().type, line_num));
    case ValKind::IntType:
      return Value::MakeIntType();
    case ValKind::BoolType:
      return Value::MakeBoolType();
    case ValKind::TypeType:
      return Value::MakeTypeType();
    case ValKind::AutoType:
      return Value::MakeAutoType();
    case ValKind::ContinuationType:
      return Value::MakeContinuationType();
    case ValKind::VariableType:
    case ValKind::StructType:
    case ValKind::ChoiceType:
    case ValKind::BindingPlaceholderValue:
    case ValKind::AlternativeConstructorValue:
      // TODO: These should be copied so that they don't get destructed.
      return val;
  }
}

auto TypeEqual(const Value* t1, const Value* t2) -> bool {
  if (t1->tag() != t2->tag()) {
    return false;
  }
  switch (t1->tag()) {
    case ValKind::PointerType:
      return TypeEqual(t1->GetPointerType().type, t2->GetPointerType().type);
    case ValKind::FunctionType:
      return TypeEqual(t1->GetFunctionType().param,
                       t2->GetFunctionType().param) and
             TypeEqual(t1->GetFunctionType().ret, t2->GetFunctionType().ret);
    case ValKind::StructType:
      return t1->GetStructType().name == t2->GetStructType().name;
    case ValKind::ChoiceType:
      return t1->GetChoiceType().name == t2->GetChoiceType().name;
    case ValKind::TupleValue: {
      if (t1->GetTupleValue().elements.size() !=
          t2->GetTupleValue().elements.size()) {
        return false;
      }
      for (size_t i = 0; i < t1->GetTupleValue().elements.size(); ++i) {
        if (t1->GetTupleValue().elements[i].name !=
            t2->GetTupleValue().elements[i].name) {
          return false;
        }
        if (not TypeEqual(t1->GetTupleValue().elements[i].value,
                          t2->GetTupleValue().elements[i].value)) {
          return false;
        }
      }
      return true;
    }
    case ValKind::IntType:
    case ValKind::BoolType:
    case ValKind::ContinuationType:
    case ValKind::TypeType:
      return true;
    case ValKind::VariableType:
      return t1->GetVariableType().name == t2->GetVariableType().name;
    default:
      llvm::errs() << "TypeEqual used to compare non-type values\n"
                   << *t1 << "\n"
                   << *t2 << "\n";
      exit(-1);
  }
}

// Returns true if all the fields of the two tuples contain equal values
// and returns false otherwise.
static auto FieldsValueEqual(const std::vector<TupleElement>& ts1,
                             const std::vector<TupleElement>& ts2, int line_num)
    -> bool {
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
    if (not ValueEqual(element.value, iter->value, line_num)) {
      return false;
    }
  }
  return true;
}

// Returns true if the two values are equal and returns false otherwise.
//
// This function implements the `==` operator of Carbon.
auto ValueEqual(const Value* v1, const Value* v2, int line_num) -> bool {
  if (v1->tag() != v2->tag()) {
    return false;
  }
  switch (v1->tag()) {
    case ValKind::IntValue:
      return v1->GetIntValue() == v2->GetIntValue();
    case ValKind::BoolValue:
      return v1->GetBoolValue() == v2->GetBoolValue();
    case ValKind::PointerValue:
      return v1->GetPointerValue() == v2->GetPointerValue();
    case ValKind::FunctionValue:
      return v1->GetFunctionValue().body == v2->GetFunctionValue().body;
    case ValKind::TupleValue:
      return FieldsValueEqual(v1->GetTupleValue().elements,
                              v2->GetTupleValue().elements, line_num);
    default:
    case ValKind::IntType:
    case ValKind::BoolType:
    case ValKind::TypeType:
    case ValKind::FunctionType:
    case ValKind::PointerType:
    case ValKind::AutoType:
    case ValKind::StructType:
    case ValKind::ChoiceType:
    case ValKind::ContinuationType:
      return TypeEqual(v1, v2);
    case ValKind::StructValue:
    case ValKind::AlternativeValue:
    case ValKind::BindingPlaceholderValue:
    case ValKind::AlternativeConstructorValue:
    case ValKind::ContinuationValue:
      llvm::errs() << "ValueEqual does not support this kind of value.\n";
      exit(-1);
  }
}

}  // namespace Carbon
