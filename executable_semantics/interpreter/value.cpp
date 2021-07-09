// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/value.h"

#include <algorithm>
#include <iostream>

#include "common/check.h"
#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

int Value::GetIntValue() const {
  CHECK(tag == ValKind::IntValue);
  return u.integer;
}

bool Value::GetBoolValue() const {
  CHECK(tag == ValKind::BoolValue);
  return u.boolean;
}

FunctionValue Value::GetFunctionValue() const {
  CHECK(tag == ValKind::FunctionValue);
  return u.fun;
}

StructValue Value::GetStructValue() const {
  CHECK(tag == ValKind::StructValue);
  return u.struct_val;
}

AlternativeConstructorValue Value::GetAlternativeConstructorValue() const {
  CHECK(tag == ValKind::AlternativeConstructorValue);
  return u.alt_cons;
}

AlternativeValue Value::GetAlternativeValue() const {
  CHECK(tag == ValKind::AlternativeValue);
  return u.alt;
}

TupleValue Value::GetTupleValue() const {
  CHECK(tag == ValKind::TupleValue);
  return u.tuple;
}

Address Value::GetPointerValue() const {
  CHECK(tag == ValKind::PointerValue);
  return u.ptr;
}

std::string* Value::GetVariableType() const {
  CHECK(tag == ValKind::VarTV);
  return u.var_type;
}

BindingPlaceholderValue Value::GetBindingPlaceholderValue() const {
  CHECK(tag == ValKind::BindingPlaceholderValue);
  return u.var_pat;
}

FunctionType Value::GetFunctionType() const {
  CHECK(tag == ValKind::FunctionType);
  return u.fun_type;
}

PointerType Value::GetPointerType() const {
  CHECK(tag == ValKind::PointerType);
  return u.ptr_type;
}

StructType Value::GetStructType() const {
  CHECK(tag == ValKind::StructType);
  return u.struct_type;
}

ChoiceType Value::GetChoiceType() const {
  CHECK(tag == ValKind::ChoiceType);
  return u.choice_type;
}

ContinuationValue Value::GetContinuationValue() const {
  CHECK(tag == ValKind::ContinuationValue);
  return u.continuation;
}

auto FindInVarValues(const std::string& field, VarValues* inits)
    -> const Value* {
  for (auto& i : *inits) {
    if (i.first == field) {
      return i.second;
    }
  }
  return nullptr;
}

auto FieldsEqual(VarValues* ts1, VarValues* ts2) -> bool {
  if (ts1->size() == ts2->size()) {
    for (auto& iter1 : *ts1) {
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

auto FindTupleField(const std::string& name, const Value* tuple)
    -> std::optional<Address> {
  CHECK(tuple->tag == ValKind::TupleValue);
  for (const TupleElement& element : *tuple->GetTupleValue().elements) {
    if (element.name == name) {
      return element.address;
    }
  }
  return std::nullopt;
}

auto Value::MakeIntValue(int i) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::IntValue;
  v->u.integer = i;
  return v;
}

auto Value::MakeBoolValue(bool b) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::BoolValue;
  v->u.boolean = b;
  return v;
}

auto Value::MakeFunctionValue(std::string name, const Value* param,
                              const Statement* body) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::FunctionValue;
  v->u.fun.name = new std::string(std::move(name));
  v->u.fun.param = param;
  v->u.fun.body = body;
  return v;
}

auto Value::MakePointerValue(Address addr) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::PointerValue;
  v->u.ptr = addr;
  return v;
}

auto Value::MakeStructValue(const Value* type, const Value* inits)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::StructValue;
  v->u.struct_val.type = type;
  v->u.struct_val.inits = inits;
  return v;
}

auto Value::MakeTupleValue(std::vector<TupleElement>* elements)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TupleValue;
  v->u.tuple.elements = elements;
  return v;
}

auto Value::MakeAlternativeValue(std::string alt_name, std::string choice_name,
                                 Address argument) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::AlternativeValue;
  v->u.alt.alt_name = new std::string(std::move(alt_name));
  v->u.alt.choice_name = new std::string(std::move(choice_name));
  v->u.alt.argument = argument;
  return v;
}

auto Value::MakeAlternativeConstructorValue(std::string alt_name,
                                            std::string choice_name)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::AlternativeConstructorValue;
  v->u.alt.alt_name = new std::string(std::move(alt_name));
  v->u.alt.choice_name = new std::string(std::move(choice_name));
  return v;
}

// Return a first-class continuation represented a fragment
// of the stack.
auto Value::MakeContinuationValue(std::vector<Frame*> stack) -> Value* {
  auto* v = new Value();
  v->tag = ValKind::ContinuationValue;
  v->u.continuation.stack = new std::vector<Frame*>(stack);
  return v;
}

auto Value::MakeBindingPlaceholderValue(std::string name, const Value* type)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::BindingPlaceholderValue;
  v->u.var_pat.name = new std::string(std::move(name));
  v->u.var_pat.type = type;
  return v;
}

auto Value::MakeVarTypeVal(std::string name) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::VarTV;
  v->u.var_type = new std::string(std::move(name));
  return v;
}

auto Value::MakeIntType() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::IntType;
  return v;
}

auto Value::MakeBoolType() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::BoolType;
  return v;
}

auto Value::MakeTypeType() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TypeType;
  return v;
}

// Return a Continuation type.
auto Value::MakeContinuationType() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::ContinuationType;
  return v;
}

auto Value::MakeAutoType() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::AutoType;
  return v;
}

auto Value::MakeFunctionType(const Value* param, const Value* ret)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::FunctionType;
  v->u.fun_type.param = param;
  v->u.fun_type.ret = ret;
  return v;
}

auto Value::MakePointerType(const Value* type) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::PointerType;
  v->u.ptr_type.type = type;
  return v;
}

auto Value::MakeStructType(std::string name, VarValues* fields,
                           VarValues* methods) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::StructType;
  v->u.struct_type.name = new std::string(std::move(name));
  v->u.struct_type.fields = fields;
  v->u.struct_type.methods = methods;
  return v;
}

auto Value::MakeUnitTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TupleValue;
  v->u.tuple.elements = new std::vector<TupleElement>();
  return v;
}

auto Value::MakeChoiceType(
    std::string name, std::list<std::pair<std::string, const Value*>>* alts)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::ChoiceType;
  // Transitional leak: when we get rid of all pointers, this will disappear.
  v->u.choice_type.name = new std::string(name);
  v->u.choice_type.alternatives = alts;
  return v;
}

auto PrintValue(const Value* val, std::ostream& out) -> void {
  switch (val->tag) {
    case ValKind::AlternativeConstructorValue: {
      out << *val->GetAlternativeConstructorValue().choice_name << "."
          << *val->GetAlternativeConstructorValue().alt_name;
      break;
    }
    case ValKind::BindingPlaceholderValue: {
      PrintValue(val->GetBindingPlaceholderValue().type, out);
      out << ": " << *val->GetBindingPlaceholderValue().name;
      break;
    }
    case ValKind::AlternativeValue: {
      out << "alt " << *val->GetAlternativeValue().choice_name << "."
          << *val->GetAlternativeValue().alt_name << " ";
      state->heap.PrintAddress(val->GetAlternativeValue().argument, out);
      break;
    }
    case ValKind::StructValue: {
      out << *val->GetStructValue().type->GetStructType().name;
      PrintValue(val->GetStructValue().inits, out);
      break;
    }
    case ValKind::TupleValue: {
      out << "(";
      bool add_commas = false;
      for (const TupleElement& element : *val->GetTupleValue().elements) {
        if (add_commas) {
          out << ", ";
        } else {
          add_commas = true;
        }

        out << element.name << " = ";
        state->heap.PrintAddress(element.address, out);
      }
      out << ")";
      break;
    }
    case ValKind::IntValue:
      out << val->GetIntValue();
      break;
    case ValKind::BoolValue:
      out << std::boolalpha << val->GetBoolValue();
      break;
    case ValKind::FunctionValue:
      out << "fun<" << *val->GetFunctionValue().name << ">";
      break;
    case ValKind::PointerValue:
      out << "ptr<" << val->GetPointerValue() << ">";
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
      PrintValue(val->GetPointerType().type, out);
      out << "*";
      break;
    case ValKind::FunctionType:
      out << "fn ";
      PrintValue(val->GetFunctionType().param, out);
      out << " -> ";
      PrintValue(val->GetFunctionType().ret, out);
      break;
    case ValKind::VarTV:
      out << *val->GetVariableType();
      break;
    case ValKind::StructType:
      out << "struct " << *val->GetStructType().name;
      break;
    case ValKind::ChoiceType:
      out << "choice " << *val->GetChoiceType().name;
      break;
    case ValKind::ContinuationValue:
      out << "continuation[[";
      for (Frame* frame : *val->GetContinuationValue().stack) {
        PrintFrame(frame, out);
        out << " :: ";
      }
      out << "]]";
      break;
  }
}

auto TypeEqual(const Value* t1, const Value* t2) -> bool {
  if (t1->tag != t2->tag) {
    return false;
  }
  switch (t1->tag) {
    case ValKind::VarTV:
      return *t1->GetVariableType() == *t2->GetVariableType();
    case ValKind::PointerType:
      return TypeEqual(t1->GetPointerType().type, t2->GetPointerType().type);
    case ValKind::FunctionType:
      return TypeEqual(t1->GetFunctionType().param,
                       t2->GetFunctionType().param) &&
             TypeEqual(t1->GetFunctionType().ret, t2->GetFunctionType().ret);
    case ValKind::StructType:
      return *t1->GetStructType().name == *t2->GetStructType().name;
    case ValKind::ChoiceType:
      return *t1->GetChoiceType().name == *t2->GetChoiceType().name;
    case ValKind::TupleValue: {
      if (t1->GetTupleValue().elements->size() !=
          t2->GetTupleValue().elements->size()) {
        return false;
      }
      for (size_t i = 0; i < t1->GetTupleValue().elements->size(); ++i) {
        if ((*t1->GetTupleValue().elements)[i].name !=
            (*t2->GetTupleValue().elements)[i].name) {
          return false;
        }
        if (!TypeEqual(
                state->heap.Read((*t1->GetTupleValue().elements)[i].address, 0),
                state->heap.Read((*t2->GetTupleValue().elements)[i].address,
                                 0))) {
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
    default:
      std::cerr << "TypeEqual used to compare non-type values" << std::endl;
      PrintValue(t1, std::cerr);
      std::cerr << std::endl;
      PrintValue(t2, std::cerr);
      exit(-1);
  }
}

// Returns true if all the fields of the two tuples contain equal values
// and returns false otherwise.
static auto FieldsValueEqual(std::vector<TupleElement>* ts1,
                             std::vector<TupleElement>* ts2, int line_num)
    -> bool {
  if (ts1->size() != ts2->size()) {
    return false;
  }
  for (const TupleElement& element : *ts1) {
    auto iter = std::find_if(
        ts2->begin(), ts2->end(),
        [&](const TupleElement& e2) { return e2.name == element.name; });
    if (iter == ts2->end()) {
      return false;
    }
    if (!ValueEqual(state->heap.Read(element.address, line_num),
                    state->heap.Read(iter->address, line_num), line_num)) {
      return false;
    }
  }
  return true;
}

// Returns true if the two values are equal and returns false otherwise.
//
// This function implements the `==` operator of Carbon.
auto ValueEqual(const Value* v1, const Value* v2, int line_num) -> bool {
  if (v1->tag != v2->tag) {
    return false;
  }
  switch (v1->tag) {
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
    case ValKind::VarTV:
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
      std::cerr << "ValueEqual does not support this kind of value."
                << std::endl;
      exit(-1);
  }
}

auto ToInteger(const Value* v) -> int {
  switch (v->tag) {
    case ValKind::IntValue:
      return v->GetIntValue();
    default:
      std::cerr << "expected an integer, not ";
      PrintValue(v, std::cerr);
      exit(-1);
  }
}

}  // namespace Carbon
