// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/value.h"

#include <algorithm>

#include "common/check.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error_builders.h"
#include "executable_semantics/interpreter/action.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

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

static auto GetMember(Nonnull<Arena*> arena, Nonnull<const Value*> v,
                      const FieldPath::Component& field,
                      SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Value*>> {
  const std::string& f = field.name();

  if (field.witness().has_value()) {
    Nonnull<const Witness*> witness = *field.witness();
    switch (witness->kind()) {
      case Value::Kind::Witness: {
        if (std::optional<Nonnull<const Declaration*>> mem_decl =
                FindMember(f, witness->declaration().members());
            mem_decl.has_value()) {
          const auto& fun_decl = cast<FunctionDeclaration>(**mem_decl);
          if (fun_decl.is_method()) {
            return arena->New<BoundMethodValue>(
                &fun_decl, v, witness->type_args(), witness->witnesses());
          } else {
            // Class function.
            auto fun = cast<FunctionValue>(*fun_decl.constant_value());
            return arena->New<FunctionValue>(&fun->declaration(),
                                             witness->type_args(),
                                             witness->witnesses());
          }
        } else {
          return CompilationError(source_loc)
                 << "member " << f << " not in " << *witness;
        }
      }
      default:
        FATAL() << "expected Witness, not " << *witness;
    }
  }
  switch (v->kind()) {
    case Value::Kind::StructValue: {
      std::optional<Nonnull<const Value*>> field =
          cast<StructValue>(*v).FindField(f);
      if (field == std::nullopt) {
        return RuntimeError(source_loc) << "member " << f << " not in " << *v;
      }
      return *field;
    }
    case Value::Kind::NominalClassValue: {
      const auto& object = cast<NominalClassValue>(*v);
      // Look for a field
      std::optional<Nonnull<const Value*>> field =
          cast<StructValue>(object.inits()).FindField(f);
      if (field.has_value()) {
        return *field;
      } else {
        // Look for a method in the object's class
        const auto& class_type = cast<NominalClassType>(object.type());
        std::optional<Nonnull<const FunctionValue*>> func =
            class_type.FindFunction(f);
        if (func == std::nullopt) {
          return RuntimeError(source_loc) << "member " << f << " not in " << *v
                                          << " or its " << class_type;
        } else if ((*func)->declaration().is_method()) {
          // Found a method. Turn it into a bound method.
          const FunctionValue& m = cast<FunctionValue>(**func);
          return arena->New<BoundMethodValue>(&m.declaration(), &object,
                                              class_type.type_args(),
                                              class_type.witnesses());
        } else {
          // Found a class function
          return arena->New<FunctionValue>(&(*func)->declaration(),
                                           class_type.type_args(),
                                           class_type.witnesses());
        }
      }
    }
    case Value::Kind::ChoiceType: {
      const auto& choice = cast<ChoiceType>(*v);
      if (!choice.FindAlternative(f)) {
        return RuntimeError(source_loc)
               << "alternative " << f << " not in " << *v;
      }
      return arena->New<AlternativeConstructorValue>(f, choice.name());
    }
    case Value::Kind::NominalClassType: {
      // Access a class function.
      const NominalClassType& class_type = cast<NominalClassType>(*v);
      std::optional<Nonnull<const FunctionValue*>> fun =
          class_type.FindFunction(f);
      if (fun == std::nullopt) {
        return RuntimeError(source_loc)
               << "class function " << f << " not in " << *v;
      }
      return arena->New<FunctionValue>(&(*fun)->declaration(),
                                       class_type.type_args(),
                                       class_type.witnesses());
    }
    default:
      FATAL() << "field access not allowed for value " << *v;
  }
}

auto Value::GetField(Nonnull<Arena*> arena, const FieldPath& path,
                     SourceLocation source_loc) const
    -> ErrorOr<Nonnull<const Value*>> {
  Nonnull<const Value*> value(this);
  for (const FieldPath::Component& field : path.components_) {
    ASSIGN_OR_RETURN(value, GetMember(arena, value, field, source_loc));
  }
  return value;
}

static auto SetFieldImpl(
    Nonnull<Arena*> arena, Nonnull<const Value*> value,
    std::vector<FieldPath::Component>::const_iterator path_begin,
    std::vector<FieldPath::Component>::const_iterator path_end,
    Nonnull<const Value*> field_value, SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Value*>> {
  if (path_begin == path_end) {
    return field_value;
  }
  switch (value->kind()) {
    case Value::Kind::StructValue: {
      std::vector<NamedValue> elements = cast<StructValue>(*value).elements();
      auto it = std::find_if(elements.begin(), elements.end(),
                             [path_begin](const NamedValue& element) {
                               return element.name == (*path_begin).name();
                             });
      if (it == elements.end()) {
        return RuntimeError(source_loc)
               << "field " << (*path_begin).name() << " not in " << *value;
      }
      ASSIGN_OR_RETURN(it->value,
                       SetFieldImpl(arena, it->value, path_begin + 1, path_end,
                                    field_value, source_loc));
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
      int index = std::stoi((*path_begin).name());
      if (index < 0 || static_cast<size_t>(index) >= elements.size()) {
        return RuntimeError(source_loc) << "index " << (*path_begin).name()
                                        << " out of range in " << *value;
      }
      ASSIGN_OR_RETURN(elements[index],
                       SetFieldImpl(arena, elements[index], path_begin + 1,
                                    path_end, field_value, source_loc));
      return arena->New<TupleValue>(elements);
    }
    default:
      FATAL() << "field access not allowed for value " << *value;
  }
}

auto Value::SetField(Nonnull<Arena*> arena, const FieldPath& path,
                     Nonnull<const Value*> field_value,
                     SourceLocation source_loc) const
    -> ErrorOr<Nonnull<const Value*>> {
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
      out << "Placeholder<";
      if (placeholder.value_node().has_value()) {
        out << (*placeholder.value_node());
      } else {
        out << "_";
      }
      out << ">";
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
      out << cast<NominalClassType>(s.type()).declaration().name() << s.inits();
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
    case Value::Kind::BoundMethodValue:
      out << "bound_method<"
          << cast<BoundMethodValue>(*this).declaration().name() << ">";
      break;
    case Value::Kind::PointerValue:
      out << "ptr<" << cast<PointerValue>(*this).address() << ">";
      break;
    case Value::Kind::LValue:
      out << "lval<" << cast<LValue>(*this).address() << ">";
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
      if (!fn_type.deduced().empty()) {
        out << "[";
        unsigned int i = 0;
        for (Nonnull<const GenericBinding*> deduced : fn_type.deduced()) {
          if (i != 0) {
            out << ", ";
          }
          out << deduced->name() << ":! " << deduced->type();
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
    case Value::Kind::NominalClassType: {
      const auto& class_type = cast<NominalClassType>(*this);
      out << "class " << class_type.declaration().name();
      if (!class_type.type_args().empty()) {
        out << "(";
        llvm::ListSeparator sep;
        for (const auto& [bind, val] : class_type.type_args()) {
          out << sep << bind->name() << " = " << *val;
        }
        out << ")";
      }
      if (!class_type.impls().empty()) {
        out << " impls ";
        llvm::ListSeparator sep;
        for (const auto& [impl_bind, impl] : class_type.impls()) {
          out << sep << *impl;
        }
      }
      if (!class_type.witnesses().empty()) {
        out << " witnesses ";
        llvm::ListSeparator sep;
        for (const auto& [impl_bind, witness] : class_type.witnesses()) {
          out << sep << *witness;
        }
      }
      break;
    }
    case Value::Kind::InterfaceType: {
      const auto& iface_type = cast<InterfaceType>(*this);
      out << "interface " << iface_type.declaration().name();
      break;
    }
    case Value::Kind::Witness: {
      const auto& witness = cast<Witness>(*this);
      out << "witness " << *witness.declaration().impl_type() << " as "
          << witness.declaration().interface();
      break;
    }
    case Value::Kind::ChoiceType:
      out << "choice " << cast<ChoiceType>(*this).name();
      break;
    case Value::Kind::VariableType:
      out << cast<VariableType>(*this).binding();
      break;
    case Value::Kind::ContinuationValue: {
      out << cast<ContinuationValue>(*this).stack();
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
    case Value::Kind::TypeOfClassType:
      out << "typeof("
          << cast<TypeOfClassType>(*this).class_type().declaration().name()
          << ")";
      break;
    case Value::Kind::TypeOfInterfaceType:
      out << "typeof("
          << cast<TypeOfInterfaceType>(*this)
                 .interface_type()
                 .declaration()
                 .name()
          << ")";
      break;
    case Value::Kind::TypeOfChoiceType:
      out << "typeof(" << cast<TypeOfChoiceType>(*this).choice_type().name()
          << ")";
      break;
    case Value::Kind::StaticArrayType: {
      const auto& array_type = cast<StaticArrayType>(*this);
      out << "[" << array_type.element_type() << "; " << array_type.size()
          << "]";
      break;
    }
  }
}

ContinuationValue::StackFragment::~StackFragment() {
  CHECK(reversed_todo_.empty())
      << "All StackFragments must be empty before the Carbon program ends.";
}

void ContinuationValue::StackFragment::StoreReversed(
    std::vector<std::unique_ptr<Action>> reversed_todo) {
  CHECK(reversed_todo_.empty());
  reversed_todo_ = std::move(reversed_todo);
}

void ContinuationValue::StackFragment::RestoreTo(
    Stack<std::unique_ptr<Action>>& todo) {
  while (!reversed_todo_.empty()) {
    todo.Push(std::move(reversed_todo_.back()));
    reversed_todo_.pop_back();
  }
}

void ContinuationValue::StackFragment::Clear() {
  // We destroy the underlying Actions explicitly to ensure they're
  // destroyed in the correct order.
  for (auto& action : reversed_todo_) {
    action.reset();
  }
  reversed_todo_.clear();
}

void ContinuationValue::StackFragment::Print(llvm::raw_ostream& out) const {
  out << "{";
  llvm::ListSeparator sep(" :: ");
  for (const std::unique_ptr<Action>& action : reversed_todo_) {
    out << sep << *action;
  }
  out << "}";
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
      if (cast<NominalClassType>(*t1).declaration().name() !=
          cast<NominalClassType>(*t2).declaration().name()) {
        return false;
      }
      for (const auto& [ty_var1, ty1] :
           cast<NominalClassType>(*t1).type_args()) {
        if (!TypeEqual(ty1,
                       cast<NominalClassType>(*t2).type_args().at(ty_var1))) {
          return false;
        }
      }
      return true;
    case Value::Kind::InterfaceType:
      return cast<InterfaceType>(*t1).declaration().name() ==
             cast<InterfaceType>(*t2).declaration().name();
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
      return &cast<VariableType>(*t1).binding() ==
             &cast<VariableType>(*t2).binding();
    case Value::Kind::TypeOfClassType:
      return TypeEqual(&cast<TypeOfClassType>(*t1).class_type(),
                       &cast<TypeOfClassType>(*t2).class_type());
    case Value::Kind::TypeOfInterfaceType:
      return TypeEqual(&cast<TypeOfInterfaceType>(*t1).interface_type(),
                       &cast<TypeOfInterfaceType>(*t2).interface_type());
    case Value::Kind::TypeOfChoiceType:
      return TypeEqual(&cast<TypeOfChoiceType>(*t1).choice_type(),
                       &cast<TypeOfChoiceType>(*t2).choice_type());
    case Value::Kind::StaticArrayType: {
      const auto& array1 = cast<StaticArrayType>(*t1);
      const auto& array2 = cast<StaticArrayType>(*t2);
      return TypeEqual(&array1.element_type(), &array2.element_type()) &&
             array1.size() == array2.size();
    }
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::StringValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::ContinuationValue:
      FATAL() << "TypeEqual used to compare non-type values\n"
              << *t1 << "\n"
              << *t2;
    case Value::Kind::Witness:
      FATAL() << "TypeEqual: unexpected Witness";
      break;
    case Value::Kind::AutoType:
      FATAL() << "TypeEqual: unexpected AutoType";
      break;
  }
}

// Returns true if the two values are equal and returns false otherwise.
//
// This function implements the `==` operator of Carbon.
auto ValueEqual(Nonnull<const Value*> v1, Nonnull<const Value*> v2) -> bool {
  if (v1->kind() != v2->kind()) {
    return false;
  }
  switch (v1->kind()) {
    case Value::Kind::IntValue:
      return cast<IntValue>(*v1).value() == cast<IntValue>(*v2).value();
    case Value::Kind::BoolValue:
      return cast<BoolValue>(*v1).value() == cast<BoolValue>(*v2).value();
    case Value::Kind::FunctionValue: {
      std::optional<Nonnull<const Statement*>> body1 =
          cast<FunctionValue>(*v1).declaration().body();
      std::optional<Nonnull<const Statement*>> body2 =
          cast<FunctionValue>(*v2).declaration().body();
      return body1.has_value() == body2.has_value() &&
             (!body1.has_value() || *body1 == *body2);
    }
    case Value::Kind::BoundMethodValue: {
      const auto& m1 = cast<BoundMethodValue>(*v1);
      const auto& m2 = cast<BoundMethodValue>(*v2);
      std::optional<Nonnull<const Statement*>> body1 = m1.declaration().body();
      std::optional<Nonnull<const Statement*>> body2 = m2.declaration().body();
      return ValueEqual(m1.receiver(), m2.receiver()) &&
             body1.has_value() == body2.has_value() &&
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
        if (!ValueEqual(elements1[i], elements2[i])) {
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
                        struct_v2.elements()[i].value)) {
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
    case Value::Kind::InterfaceType:
    case Value::Kind::Witness:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::StaticArrayType:
      return TypeEqual(v1, v2);
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
      // TODO: support pointer comparisons once we have a clearer distinction
      // between pointers and lvalues.
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

auto NominalClassType::FindFunction(const std::string& name) const
    -> std::optional<Nonnull<const FunctionValue*>> {
  for (const auto& member : declaration().members()) {
    switch (member->kind()) {
      case DeclarationKind::FunctionDeclaration: {
        const auto& fun = cast<FunctionDeclaration>(*member);
        if (fun.name() == name) {
          return &cast<FunctionValue>(**fun.constant_value());
        }
        break;
      }
      default:
        break;
    }
  }
  return std::nullopt;
}

auto FindMember(const std::string& name,
                llvm::ArrayRef<Nonnull<Declaration*>> members)
    -> std::optional<Nonnull<const Declaration*>> {
  for (Nonnull<const Declaration*> member : members) {
    if (std::optional<std::string> mem_name = GetName(*member);
        mem_name.has_value()) {
      if (*mem_name == name) {
        return member;
      }
    }
  }
  return std::nullopt;
}

void ImplBinding::Print(llvm::raw_ostream& out) const {
  out << "impl binding " << *type_var_ << " as " << *iface_;
}

void ImplBinding::PrintID(llvm::raw_ostream& out) const {
  out << *type_var_ << " as " << *iface_;
}

}  // namespace Carbon
