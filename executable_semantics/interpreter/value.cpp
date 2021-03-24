// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/value.h"

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

auto MakeFunVal(std::string name, const Value* param, Statement* body)
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

auto MakeTupleTypeVal(VarValues* fields) -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TupleTV;
  v->u.tuple_type.fields = fields;
  return v;
}

auto MakeVoidTypeVal() -> const Value* {
  auto* v = new Value();
  v->tag = ValKind::TupleTV;
  v->u.tuple_type.fields = new VarValues();
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

void PrintValue(const Value* val, std::ostream& out) {
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
      PrintValue(state->heap[val->u.alt.argument], out);
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
        PrintValue(state->heap[elt.second], out);
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
    case ValKind::TupleTV: {
      out << "Tuple(";
      bool add_commas = false;
      for (const auto& elt : *val->u.tuple_type.fields) {
        if (add_commas) {
          out << ", ";
        } else {
          add_commas = true;
        }

        out << elt.first << " = ";
        PrintValue(elt.second, out);
      }
      out << ")";
      break;
    }
    case ValKind::StructTV:
      out << "struct " << *val->u.struct_type.name;
      break;
    case ValKind::ChoiceTV:
      out << "choice " << *val->u.choice_type.name;
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
    case ValKind::TupleTV:
      return FieldsEqual(t1->u.tuple_type.fields, t2->u.tuple_type.fields);
    case ValKind::IntTV:
    case ValKind::BoolTV:
      return true;
    default:
      return false;
  }
}

static auto FieldsValueEqual(VarValues* ts1, VarValues* ts2, int line_num)
    -> bool {
  if (ts1->size() != ts2->size()) {
    return false;
  }
  for (auto& iter1 : *ts1) {
    auto t2 = FindInVarValues(iter1.first, ts2);
    if (t2 == nullptr) {
      return false;
    }
    if (!ValueEqual(iter1.second, t2, line_num)) {
      return false;
    }
  }
  return true;
}

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
      CheckAlive(v1->u.ptr, line_num);
      CheckAlive(v2->u.ptr, line_num);
      return v1->u.ptr == v2->u.ptr;
    case ValKind::FunV:
      return v1->u.fun.body == v2->u.fun.body;
    case ValKind::TupleV:
      return FieldsValueEqual(v1->u.tuple_type.fields, v2->u.tuple_type.fields,
                              line_num);
    default:
      return TypeEqual(v1, v2);
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

void CheckAlive(Address address, int line_num) {
  if (!state->alive[address]) {
    std::cerr << line_num << ": undefined behavior: access to dead value ";
    PrintValue(state->heap[address], std::cerr);
    std::cerr << std::endl;
    exit(-1);
  }
}

}  // namespace Carbon
