// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPE_ITERATOR_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPE_ITERATOR_H_

#include <concepts>
#include <utility>
#include <variant>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Iterates over the the types in a given instruction. For the purposes of the
// iterator, a facet value and an interface can also be treated like a type and
// be iterated over.
//
// To use, call `Add()` one or more times to set up the types, facets or
// interfaces to be iterated over. Then call `Next()` until it returns
// `Step::Done`.
//
// Note that this iterator looks through `FacetValue`s walking the type referred
// to by it instead, since `FacetValue` is an internal representation that
// converts to the original type whenever a type is needed so that its full API
// surface is not lost.
class TypeIterator {
 public:
  // The result of each iteration step.
  class Step;

  // `sem_ir` must not be null.
  explicit TypeIterator(File* sem_ir) : sem_ir_(sem_ir) {}

  // Add a type value or facet value to be iterated over.
  //
  // The iterator will visit things in the reverse order that they are added.
  auto Add(SemIR::InstId inst_id) -> void {
    auto type_id = sem_ir_->insts().Get(inst_id).type_id();
    CARBON_CHECK(sem_ir_->types().IsFacetType(type_id));
    PushInstId(inst_id);
  }

  // Add an interface to be iterated over.
  //
  // The iterator will visit things in the reverse order that they are added.
  auto Add(SemIR::SpecificInterface interface) -> void { Push(interface); }

  // Iterates and returns the next `Step`. Returns `Step::Done` when complete.
  auto Next() -> Step;

 private:
  // A work item to mark the end of an aggregate type's scope.
  struct EndType {};
  // A work item to mark a symbolic type.
  struct SymbolicType {
    SemIR::TypeId facet_type_id;
  };
  // A work item to mark a concrete non-type value.
  struct ConcreteNonTypeValue {
    SemIR::InstId inst_id;
  };
  // A work item to mark a symbolic non-type value.
  struct SymbolicNonTypeValue {
    SemIR::InstId inst_id;
  };
  // A work item to mark the name of a struct field.
  struct StructFieldName {
    SemIR::NameId name_id;
  };

  using WorkItem =
      std::variant<SemIR::TypeId, SymbolicType, ConcreteNonTypeValue,
                   SymbolicNonTypeValue, StructFieldName,
                   SemIR::SpecificInterface, EndType>;

  // Get the TypeId for an instruction that is not a facet value, otherwise
  // return SymbolicType to indicate the instruction is a symbolic facet value.
  //
  // If the instruction is not a type value, the return is TypeId::None.
  //
  // We reuse the `SymbolicType` work item here to give a nice return type.
  auto TryGetInstIdAsTypeId(SemIR::InstId inst_id) const
      -> std::variant<SemIR::TypeId, SymbolicType>;

  // Get the instructions in the specific's instruction block as an ArrayRef.
  auto GetSpecificArgs(SemIR::SpecificId specific_id) const
      -> llvm::ArrayRef<SemIR::InstId>;

  // Push all arguments from the array into the work queue.
  auto PushArgs(llvm::ArrayRef<SemIR::InstId> args) -> void;

  // Push an instruction's type value into the work queue, or a marker if the
  // instruction has a symbolic value.
  auto PushInstId(SemIR::InstId inst_id) -> void;

  // Push the next step into the work queue.
  auto Push(WorkItem item) -> void;

  File* sem_ir_;
  llvm::SmallVector<WorkItem> work_list_;
};

class TypeIterator::Step {
 public:
  // ===========================================================================
  // Results that enter a scope where the following results are related, until
  // the iterator returns `End`.

  // Followed by generic parameters.
  struct ClassStart {
    SemIR::ClassId class_id;
    SemIR::TypeId type_id;
  };
  // Followed by its fields.
  struct StructStart {
    SemIR::TypeId type_id;
  };
  // Followed by its members.
  struct TupleStart {
    SemIR::TypeId type_id;
  };
  // Followed by generic parameters.
  struct InterfaceStart {
    SemIR::InterfaceId interface_id;
  };
  // Followed by the bit width.
  struct IntStart {
    SemIR::TypeId type_id;
  };
  // Followed by the type and bound.
  struct ArrayStart {
    SemIR::TypeId type_id;
  };
  // Simple wrapped types, followed by the inner type.
  struct ConstStart {};
  struct MaybeUnformedStart {};
  struct PartialStart {};
  struct PointerStart {};

  // ===========================================================================
  // Results that would enter a scope, but there are known to be no related
  // results in that scope, so this means there won't be an `End` paired with
  // them.

  struct ClassStartOnly : public ClassStart {};
  struct StructStartOnly : public StructStart {};
  struct TupleStartOnly : public TupleStart {};
  struct InterfaceStartOnly : public InterfaceStart {};

  // ===========================================================================
  // Individual result values, which appear on their own or inside some scope
  // that begin with `StartWithEnd`.

  // A type value.
  struct ConcreteType {
    SemIR::TypeId type_id;
  };
  // A symbolic type value, constrained by `facet_type_id`.
  struct SymbolicType {
    // Either a FacetType or the TypeType singleton.
    SemIR::TypeId facet_type_id;
  };
  // A symbolic template type value.
  struct TemplateType {};
  // A concrete non-type value, which can be found as a generic parameter for a
  // type.
  struct ConcreteValue {
    // An instruction that evaluates to the constant value.
    SemIR::InstId inst_id;
  };
  // A symbolic non-type value, which can be found as a generic parameter for a
  // type.
  struct SymbolicValue {
    // An instruction that evaluates to the constant value.
    SemIR::InstId inst_id;
  };
  // A struct field name. The field names contribute to the type of the struct.
  struct StructFieldName {
    SemIR::NameId name_id;
  };

  // ===========================================================================
  // Results that report a state change in iteration.

  // Closes the scope of a `*Start` step.
  struct End {};
  // Iteration is complete.
  struct Done {};
  // Iteration found an error.
  struct Error {};

  // Each step is one of these.
  using Any =
      std::variant<ConcreteType, SymbolicType, TemplateType, ConcreteValue,
                   SymbolicValue, StructFieldName, ClassStartOnly,
                   StructStartOnly, TupleStartOnly, InterfaceStartOnly,
                   ClassStart, StructStart, TupleStart, InterfaceStart,
                   IntStart, ArrayStart, ConstStart, MaybeUnformedStart,
                   PartialStart, PointerStart, End, Done, Error>;

  template <typename T>
  auto Is() const -> bool {
    return std::holds_alternative<T>(any);
  }

  // Converts from any individual step.
  //
  // This is a template to allow implicit conversion directly from step values
  // that can go inside `Any` to `Step` (without having to make the `Any`
  // explicitly first).
  template <typename T>
    requires std::constructible_from<Any, T>
  explicit(false) Step(T any) : any(any) {}

  Any any;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPE_ITERATOR_H_
