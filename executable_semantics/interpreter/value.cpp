// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/value.h"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

int Value::GetInteger() const {
  assert(tag == ValKind::IntV);
  return u.integer;
}

bool Value::GetBoolean() const {
  assert(tag == ValKind::BoolV);
  return u.boolean;
}

Function Value::GetFunction() const {
  assert(tag == ValKind::FunV);
  return u.fun;
}

StructConstructor Value::GetStruct() const {
  assert(tag == ValKind::StructV);
  return u.struct_val;
}

AlternativeConstructor Value::GetAlternativeConstructor() const {
  assert(tag == ValKind::AltConsV);
  return u.alt_cons;
}

Alternative Value::GetAlternative() const {
  assert(tag == ValKind::AltV);
  return u.alt;
}

TupleValue Value::GetTuple() const {
  assert(tag == ValKind::TupleV);
  return u.tuple;
}

Address Value::GetPointer() const {
  assert(tag == ValKind::PtrV);
  return u.ptr;
}

std::string* Value::GetVariableType() const {
  assert(tag == ValKind::VarTV);
  return u.var_type;
}

VariablePatternValue Value::GetVariablePattern() const {
  assert(tag == ValKind::VarPatV);
  return u.var_pat;
}

FunctionTypeValue Value::GetFunctionType() const {
  assert(tag == ValKind::FunctionTV);
  return u.fun_type;
}

PointerType Value::GetPointerType() const {
  assert(tag == ValKind::PointerTV);
  return u.ptr_type;
}

StructType Value::GetStructType() const {
  assert(tag == ValKind::StructTV);
  return u.struct_type;
}

ChoiceType Value::GetChoiceType() const {
  assert(tag == ValKind::ChoiceTV);
  return u.choice_type;
}

ContinuationValue Value::GetContinuation() const {
  assert(tag == ValKind::ContinuationV);
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
  assert(tuple->tag == ValKind::TupleV);
  for (const TupleElement& element : *tuple->GetTuple().elements) {
    if (element.name == name) {
      return element.address;
    }
  }
  return std::nullopt;
}

auto Value::MakeIntVal(int i) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::IntV;
  v->u.integer = i;
  return v;
}

auto Value::MakeBoolVal(bool b) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::BoolV;
  v->u.boolean = b;
  return v;
}

auto Value::MakeFunVal(std::string name, const Value* param,
                       const Statement* body) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::FunV;
  v->u.fun.name = new std::string(std::move(name));
  v->u.fun.param = param;
  v->u.fun.body = body;
  return v;
}

auto Value::MakePtrVal(Address addr) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::PtrV;
  v->u.ptr = addr;
  return v;
}

auto Value::MakeStructVal(const Value* type, const Value* inits)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::StructV;
  v->u.struct_val.type = type;
  v->u.struct_val.inits = inits;
  return v;
}

auto Value::MakeTupleVal(std::vector<TupleElement>* elements) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TupleV;
  v->u.tuple.elements = elements;
  return v;
}

auto Value::MakeAltVal(std::string alt_name, std::string choice_name,
                       Address argument) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::AltV;
  v->u.alt.alt_name = new std::string(std::move(alt_name));
  v->u.alt.choice_name = new std::string(std::move(choice_name));
  v->u.alt.argument = argument;
  return v;
}

auto Value::MakeAltCons(std::string alt_name, std::string choice_name)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::AltConsV;
  v->u.alt.alt_name = new std::string(std::move(alt_name));
  v->u.alt.choice_name = new std::string(std::move(choice_name));
  return v;
}

// Return a first-class continuation represented a fragment
// of the stack.
auto Value::MakeContinuation(std::vector<Frame*> stack) -> Value* {
  auto* v = new Value();
  v->tag = ValKind::ContinuationV;
  v->u.continuation.stack = new std::vector<Frame*>(stack);
  return v;
}

auto Value::MakeVarPatVal(std::string name, const Value* type) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::VarPatV;
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

auto Value::MakeIntTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::IntTV;
  return v;
}

auto Value::MakeBoolTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::BoolTV;
  return v;
}

auto Value::MakeTypeTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TypeTV;
  return v;
}

// Return a Continuation type.
auto Value::MakeContinuationTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::ContinuationTV;
  return v;
}

auto Value::MakeAutoTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::AutoTV;
  return v;
}

auto Value::MakeFunTypeVal(const Value* param, const Value* ret)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::FunctionTV;
  v->u.fun_type.param = param;
  v->u.fun_type.ret = ret;
  return v;
}

auto Value::MakePtrTypeVal(const Value* type) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::PointerTV;
  v->u.ptr_type.type = type;
  return v;
}

auto Value::MakeStructTypeVal(std::string name, VarValues* fields,
                              VarValues* methods) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::StructTV;
  v->u.struct_type.name = new std::string(std::move(name));
  v->u.struct_type.fields = fields;
  v->u.struct_type.methods = methods;
  return v;
}

auto Value::MakeUnitTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TupleV;
  v->u.tuple.elements = new std::vector<TupleElement>();
  return v;
}

auto Value::MakeChoiceTypeVal(
    std::string name, std::list<std::pair<std::string, const Value*>>* alts)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::ChoiceTV;
  // Transitional leak: when we get rid of all pointers, this will disappear.
  v->u.choice_type.name = new std::string(name);
  v->u.choice_type.alternatives = alts;
  return v;
}

auto PrintValue(const Value* val, std::ostream& out) -> void {
  switch (val->tag) {
    case ValKind::AltConsV: {
      out << *val->GetAlternativeConstructor().choice_name << "."
          << *val->GetAlternativeConstructor().alt_name;
      break;
    }
    case ValKind::VarPatV: {
      PrintValue(val->GetVariablePattern().type, out);
      out << ": " << *val->GetVariablePattern().name;
      break;
    }
    case ValKind::AltV: {
      out << "alt " << *val->GetAlternative().choice_name << "."
          << *val->GetAlternative().alt_name << " ";
      state->heap.PrintAddress(val->GetAlternative().argument, out);
      break;
    }
    case ValKind::StructV: {
      out << *val->GetStruct().type->GetStructType().name;
      PrintValue(val->GetStruct().inits, out);
      break;
    }
    case ValKind::TupleV: {
      out << "(";
      bool add_commas = false;
      for (const TupleElement& element : *val->GetTuple().elements) {
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
    case ValKind::IntV:
      out << val->GetInteger();
      break;
    case ValKind::BoolV:
      out << std::boolalpha << val->GetBoolean();
      break;
    case ValKind::FunV:
      out << "fun<" << *val->GetFunction().name << ">";
      break;
    case ValKind::PtrV:
      out << "ptr<" << val->GetPointer() << ">";
      break;
    case ValKind::BoolTV:
      out << "Bool";
      break;
    case ValKind::IntTV:
      out << "Int";
      break;
    case ValKind::TypeTV:
      out << "Type";
      break;
    case ValKind::AutoTV:
      out << "auto";
      break;
    case ValKind::ContinuationTV:
      out << "Continuation";
      break;
    case ValKind::PointerTV:
      PrintValue(val->GetPointerType().type, out);
      out << "*";
      break;
    case ValKind::FunctionTV:
      out << "fn ";
      PrintValue(val->GetFunctionType().param, out);
      out << " -> ";
      PrintValue(val->GetFunctionType().ret, out);
      break;
    case ValKind::VarTV:
      out << *val->GetVariableType();
      break;
    case ValKind::StructTV:
      out << "struct " << *val->GetStructType().name;
      break;
    case ValKind::ChoiceTV:
      out << "choice " << *val->GetChoiceType().name;
      break;
    case ValKind::ContinuationV:
      out << "continuation[[";
      for (Frame* frame : *val->GetContinuation().stack) {
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
    case ValKind::PointerTV:
      return TypeEqual(t1->GetPointerType().type, t2->GetPointerType().type);
    case ValKind::FunctionTV:
      return TypeEqual(t1->GetFunctionType().param,
                       t2->GetFunctionType().param) &&
             TypeEqual(t1->GetFunctionType().ret, t2->GetFunctionType().ret);
    case ValKind::StructTV:
      return *t1->GetStructType().name == *t2->GetStructType().name;
    case ValKind::ChoiceTV:
      return *t1->GetChoiceType().name == *t2->GetChoiceType().name;
    case ValKind::TupleV: {
      if (t1->GetTuple().elements->size() != t2->GetTuple().elements->size()) {
        return false;
      }
      for (size_t i = 0; i < t1->GetTuple().elements->size(); ++i) {
        if ((*t1->GetTuple().elements)[i].name !=
            (*t2->GetTuple().elements)[i].name) {
          return false;
        }
        if (!TypeEqual(
                state->heap.Read((*t1->GetTuple().elements)[i].address, 0),
                state->heap.Read((*t2->GetTuple().elements)[i].address, 0))) {
          return false;
        }
      }
      return true;
    }
    case ValKind::IntTV:
    case ValKind::BoolTV:
    case ValKind::ContinuationTV:
    case ValKind::TypeTV:
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
    case ValKind::IntV:
      return v1->GetInteger() == v2->GetInteger();
    case ValKind::BoolV:
      return v1->GetBoolean() == v2->GetBoolean();
    case ValKind::PtrV:
      return v1->GetPointer() == v2->GetPointer();
    case ValKind::FunV:
      return v1->GetFunction().body == v2->GetFunction().body;
    case ValKind::TupleV:
      return FieldsValueEqual(v1->GetTuple().elements, v2->GetTuple().elements,
                              line_num);
    default:
    case ValKind::VarTV:
    case ValKind::IntTV:
    case ValKind::BoolTV:
    case ValKind::TypeTV:
    case ValKind::FunctionTV:
    case ValKind::PointerTV:
    case ValKind::AutoTV:
    case ValKind::StructTV:
    case ValKind::ChoiceTV:
    case ValKind::ContinuationTV:
      return TypeEqual(v1, v2);
    case ValKind::StructV:
    case ValKind::AltV:
    case ValKind::VarPatV:
    case ValKind::AltConsV:
    case ValKind::ContinuationV:
      std::cerr << "ValueEqual does not support this kind of value."
                << std::endl;
      exit(-1);
  }
}

auto ToInteger(const Value* v) -> int {
  switch (v->tag) {
    case ValKind::IntV:
      return v->GetInteger();
    default:
      std::cerr << "expected an integer, not ";
      PrintValue(v, std::cerr);
      exit(-1);
  }
}

}  // namespace Carbon
