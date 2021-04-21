// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/value.h"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

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
  for (const auto& i : *tuple->u.tuple.elts) {
    if (i.first == name) {
      return i.second;
    }
  }
  return std::nullopt;
}

auto MakeIntVal(int i) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::IntV;
  v->u.integer = i;
  return v;
}

auto MakeBoolVal(bool b) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::BoolV;
  v->u.boolean = b;
  return v;
}

auto MakeFunVal(std::string name, const Value* param, const Statement* body)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::FunV;
  v->u.fun.name = new std::string(std::move(name));
  v->u.fun.param = param;
  v->u.fun.body = body;
  return v;
}

auto MakePtrVal(Address addr) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::PtrV;
  v->u.ptr = addr;
  return v;
}

auto MakeStructVal(const Value* type, const Value* inits) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::StructV;
  v->u.struct_val.type = type;
  v->u.struct_val.inits = inits;
  return v;
}

auto MakeTupleVal(std::vector<std::pair<std::string, Address>>* elts)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TupleV;
  v->u.tuple.elts = elts;
  return v;
}

auto MakeAltVal(std::string alt_name, std::string choice_name, Address argument)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::AltV;
  v->u.alt.alt_name = new std::string(std::move(alt_name));
  v->u.alt.choice_name = new std::string(std::move(choice_name));
  v->u.alt.argument = argument;
  return v;
}

auto MakeAltCons(std::string alt_name, std::string choice_name)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::AltConsV;
  v->u.alt.alt_name = new std::string(std::move(alt_name));
  v->u.alt.choice_name = new std::string(std::move(choice_name));
  return v;
}

// Return a first-class continuation represented a fragment
// of the stack.
auto MakeContinuation(std::vector<Frame*> stack) -> Value* {
  auto* v = new Value();
  v->tag = ValKind::ContinuationV;
  v->u.continuation.stack = new std::vector<Frame*>(stack);
  return v;
}

auto MakeVarPatVal(std::string name, const Value* type) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::VarPatV;
  v->u.var_pat.name = new std::string(std::move(name));
  v->u.var_pat.type = type;
  return v;
}

auto MakeVarTypeVal(std::string name) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::VarTV;
  v->u.var_type = new std::string(std::move(name));
  return v;
}

auto MakeIntTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::IntTV;
  return v;
}

auto MakeBoolTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::BoolTV;
  return v;
}

auto MakeTypeTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TypeTV;
  return v;
}

// Return a Continuation type.
auto MakeContinuationTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::ContinuationTV;
  return v;
}

auto MakeAutoTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::AutoTV;
  return v;
}

auto MakeFunTypeVal(const Value* param, const Value* ret) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::FunctionTV;
  v->u.fun_type.param = param;
  v->u.fun_type.ret = ret;
  return v;
}

auto MakePtrTypeVal(const Value* type) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::PointerTV;
  v->u.ptr_type.type = type;
  return v;
}

auto MakeStructTypeVal(std::string name, VarValues* fields, VarValues* methods)
    -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::StructTV;
  v->u.struct_type.name = new std::string(std::move(name));
  v->u.struct_type.fields = fields;
  v->u.struct_type.methods = methods;
  return v;
}

auto MakeVoidTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TupleV;
  v->u.tuple.elts = new std::vector<std::pair<std::string, Address>>();
  return v;
}

auto MakeChoiceTypeVal(std::string name,
                       std::list<std::pair<std::string, const Value*>>* alts)
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
      out << *val->u.alt_cons.choice_name << "." << *val->u.alt_cons.alt_name;
      break;
    }
    case ValKind::VarPatV: {
      PrintValue(val->u.var_pat.type, out);
      out << ": " << *val->u.var_pat.name;
      break;
    }
    case ValKind::AltV: {
      out << "alt " << *val->u.alt.choice_name << "." << *val->u.alt.alt_name
          << " ";
      state->PrintAddress(val->u.alt.argument, out);
      break;
    }
    case ValKind::StructV: {
      out << *val->u.struct_val.type->u.struct_type.name;
      PrintValue(val->u.struct_val.inits, out);
      break;
    }
    case ValKind::TupleV: {
      out << "(";
      bool add_commas = false;
      for (const auto& elt : *val->u.tuple.elts) {
        if (add_commas) {
          out << ", ";
        } else {
          add_commas = true;
        }

        out << elt.first << " = ";
        state->PrintAddress(elt.second, out);
        out << "@" << elt.second;
      }
      out << ")";
      break;
    }
    case ValKind::IntV:
      out << val->u.integer;
      break;
    case ValKind::BoolV:
      out << std::boolalpha << val->u.boolean;
      break;
    case ValKind::FunV:
      out << "fun<" << *val->u.fun.name << ">";
      break;
    case ValKind::PtrV:
      out << "ptr<" << val->u.ptr << ">";
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
      out << "Ptr(";
      PrintValue(val->u.ptr_type.type, out);
      out << ")";
      break;
    case ValKind::FunctionTV:
      out << "fn ";
      PrintValue(val->u.fun_type.param, out);
      out << " -> ";
      PrintValue(val->u.fun_type.ret, out);
      break;
    case ValKind::VarTV:
      out << *val->u.var_type;
      break;
    case ValKind::StructTV:
      out << "struct " << *val->u.struct_type.name;
      break;
    case ValKind::ChoiceTV:
      out << "choice " << *val->u.choice_type.name;
      break;
    case ValKind::ContinuationV:
      out << "continuation[[";
      for (Frame* frame : *val->u.continuation.stack) {
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
      return *t1->u.var_type == *t2->u.var_type;
    case ValKind::PointerTV:
      return TypeEqual(t1->u.ptr_type.type, t2->u.ptr_type.type);
    case ValKind::FunctionTV:
      return TypeEqual(t1->u.fun_type.param, t2->u.fun_type.param) &&
             TypeEqual(t1->u.fun_type.ret, t2->u.fun_type.ret);
    case ValKind::StructTV:
      return *t1->u.struct_type.name == *t2->u.struct_type.name;
    case ValKind::ChoiceTV:
      return *t1->u.choice_type.name == *t2->u.choice_type.name;
    case ValKind::TupleV: {
      if (t1->u.tuple.elts->size() != t2->u.tuple.elts->size()) {
        return false;
      }
      for (size_t i = 0; i < t1->u.tuple.elts->size(); ++i) {
        std::optional<Address> t2_field =
            FindTupleField((*t1->u.tuple.elts)[i].first, t2);
        if (t2_field == std::nullopt) {
          return false;
        }
        if (!TypeEqual(state->ReadFromMemory((*t1->u.tuple.elts)[i].second, 0),
                       state->ReadFromMemory(*t2_field, 0))) {
          return false;
        }
      }
      return true;
    }
    case ValKind::IntTV:
    case ValKind::BoolTV:
    case ValKind::ContinuationTV:
      return true;
    default:
      std::cerr << "TypeEqual used to compare non-type values" << std::endl;
      exit(-1);
  }
}

// Returns true if all the fields of the two tuples contain equal values
// and returns false otherwise.
static auto FieldsValueEqual(VarAddresses* ts1, VarAddresses* ts2, int line_num)
    -> bool {
  if (ts1->size() != ts2->size()) {
    return false;
  }
  for (const auto& [name, address] : *ts1) {
    auto iter =
        std::find_if(ts2->begin(), ts2->end(),
                     [name = name](const auto& p) { return p.first == name; });
    if (iter == ts2->end()) {
      return false;
    }
    if (!ValueEqual(state->ReadFromMemory(address, line_num),
                    state->ReadFromMemory(iter->second, line_num), line_num)) {
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
      return v1->u.integer == v2->u.integer;
    case ValKind::BoolV:
      return v1->u.boolean == v2->u.boolean;
    case ValKind::PtrV:
      return v1->u.ptr == v2->u.ptr;
    case ValKind::FunV:
      return v1->u.fun.body == v2->u.fun.body;
    case ValKind::TupleV:
      return FieldsValueEqual(v1->u.tuple.elts, v2->u.tuple.elts, line_num);
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
      return v->u.integer;
    default:
      std::cerr << "expected an integer, not ";
      PrintValue(v, std::cerr);
      exit(-1);
  }
}

}  // namespace Carbon
