// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CONSTANT_H_
#define CARBON_TOOLCHAIN_SEM_IR_CONSTANT_H_

#include "common/map.h"
#include "toolchain/base/yaml.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::SemIR {

// The kinds of symbolic bindings that a constant might depend on. These are
// ordered from least to most dependent, so that the dependence of an operation
// can typically be computed by taking the maximum of the dependences of its
// operands.
enum class ConstantDependence : uint8_t {
  // This constant's value is known concretely, and does not depend on any
  // symbolic binding.
  None,
  // The only symbolic binding that this constant depends on is `.Self`.
  PeriodSelf,
  // The only symbolic bindings that this constant depends on are checked
  // generic bindings.
  Checked,
  // This symbolic binding depends on a template-dependent value, such as a
  // template parameter.
  Template,
};

// Information about a symbolic constant value. These are indexed by
// `ConstantId`s for which `is_symbolic` is true.
//
// A constant value is defined by the canonical ID of a fully-evaluated inst,
// called a "constant inst", which may depend on the canonical IDs of other
// constant insts. "Canonical" here means that it is chosen such that equal
// constants will have equal canonical IDs. This is typically achieved by
// deduplication in `ConstantStore`, but certain kinds of constant insts are
// canonicalized in other ways.
//
// That constant inst ID fully defines the constant value in itself, but for
// symbolic constant values we sometimes need efficient access to metadata about
// the mapping between the constant and corresponding constants in specifics of
// its enclosing generic. As a result, the ID of a concrete constant directly
// encodes the ID of the constant inst, but the ID of a symbolic constant is an
// index into a table of `SymbolicConstant` entries containing that metadata, as
// well as the constant inst ID.
//
// The price of this optimization is that the constant value's ID depends on the
// enclosing generic, which isn't semantically relevant unless we're
// specifically operating on the generic -> specific mapping. As a result, every
// symbolic constant is represented by two `SymbolicConstant`s, with separate
// IDs: one with that additional metadata, and one without it. The form with
// additional metadata is called an "attached constant", and the form without it
// is an "unattached constant". Note that constants in separate generics may be
// represented by the same unattached constant. In general, only one of these
// IDs is correct to use in a given situation; `ConstantValueStore` can be used
// to map between them if necessary.
//
// Equivalently, you can think of an unattached constant as being implicitly
// parameterized by the `bind_symbolic_name` constant insts that it depends on,
// whereas an attached constant explicitly binds them to parameters of the
// enclosing generic. It's the difference between "`Vector(T)` where `T` is some
// value of type `type`" and "`Vector(T)` where `T` is the `T:! type` parameter
// of this particular enclosing generic".
//
// TODO: consider instead keeping this metadata in a separate hash map keyed by
// a `GenericId`/`ConstantId` pair, so that each constant has a single
// `ConstantId`, rather than separate attached and unattached IDs.
struct SymbolicConstant : Printable<SymbolicConstant> {
  // The canonical ID of the inst that defines this constant.
  InstId inst_id;
  // The generic that this constant is attached to, or `None` if this is an
  // unattached constant.
  GenericId generic_id;
  // The index of this constant within the generic's eval block, if this is an
  // attached constant. For a given specific of that generic, this is also the
  // index of this constant's value in the value block of that specific. If
  // this constant is unattached, `index` will be `None`.
  GenericInstIndex index;
  // The kind of dependence this symbolic constant exhibits. Should never be
  // `None`.
  ConstantDependence dependence;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{inst: " << inst_id << ", generic: " << generic_id
        << ", index: " << index << ", kind: ";
    switch (dependence) {
      case ConstantDependence::None:
        out << "<error: concrete>";
        break;
      case ConstantDependence::PeriodSelf:
        out << "self";
        break;
      case ConstantDependence::Checked:
        out << "checked";
        break;
      case ConstantDependence::Template:
        out << "template";
        break;
    }
    out << "}";
  }
};

// Provides a ValueStore wrapper for tracking the constant values of
// instructions.
class ConstantValueStore {
 public:
  explicit ConstantValueStore(ConstantId default_value)
      : default_(default_value) {}

  // Returns the constant value of the requested instruction, which is default_
  // if unallocated. Always returns an unattached constant.
  auto Get(InstId inst_id) const -> ConstantId {
    auto const_id = GetAttached(inst_id);
    return const_id.has_value() ? GetUnattachedConstant(const_id) : const_id;
  }

  // Returns the constant value of the requested instruction, which is default_
  // if unallocated. This may be an attached constant.
  auto GetAttached(InstId inst_id) const -> ConstantId {
    CARBON_DCHECK(inst_id.index >= 0);
    return static_cast<size_t>(inst_id.index) >= values_.size()
               ? default_
               : values_.Get(inst_id);
  }

  // Sets the constant value of the given instruction, or sets that it is known
  // to not be a constant.
  auto Set(InstId inst_id, ConstantId const_id) -> void {
    CARBON_DCHECK(inst_id.index >= 0);
    if (static_cast<size_t>(inst_id.index) >= values_.size()) {
      values_.Resize(inst_id.index + 1, default_);
    }
    values_.Get(inst_id) = const_id;
  }

  // Gets the ID of the underlying constant inst for the given constant. Returns
  // `None` if the constant ID is non-constant. Requires `const_id.has_value()`.
  auto GetInstId(ConstantId const_id) const -> InstId {
    if (const_id.is_concrete()) {
      return const_id.concrete_inst_id();
    }
    if (const_id.is_symbolic()) {
      return GetSymbolicConstant(const_id).inst_id;
    }
    return InstId::None;
  }

  // Gets the ID of the underlying constant inst for the given constant. Returns
  // `None` if the constant ID is non-constant or `None`.
  auto GetInstIdIfValid(ConstantId const_id) const -> InstId {
    return const_id.has_value() ? GetInstId(const_id) : InstId::None;
  }

  // Given an instruction, returns the unique constant instruction that is
  // equivalent to it. Returns `None` for a non-constant instruction.
  auto GetConstantInstId(InstId inst_id) const -> InstId {
    return GetInstId(GetAttached(inst_id));
  }

  // Given a symbolic constant, returns the unattached form of that constant.
  // For any other constant ID, returns the ID unchanged.
  auto GetUnattachedConstant(ConstantId const_id) const -> ConstantId {
    if (const_id.is_symbolic()) {
      return GetAttached(GetSymbolicConstant(const_id).inst_id);
    }
    return const_id;
  }

  auto AddSymbolicConstant(SymbolicConstant constant) -> ConstantId {
    return ConstantId::ForSymbolicConstantId(symbolic_constants_.Add(constant));
  }

  auto GetSymbolicConstant(ConstantId const_id) -> SymbolicConstant& {
    return symbolic_constants_.Get(const_id.symbolic_id());
  }

  auto GetSymbolicConstant(ConstantId const_id) const
      -> const SymbolicConstant& {
    return symbolic_constants_.Get(const_id.symbolic_id());
  }

  // Get the dependence of the given constant.
  auto GetDependence(ConstantId const_id) const -> ConstantDependence {
    return const_id.is_symbolic() ? GetSymbolicConstant(const_id).dependence
                                  : ConstantDependence::None;
  }

  // Returns true for symbolic constants other than those that are only symbolic
  // because they depend on `.Self`.
  auto DependsOnGenericParameter(ConstantId const_id) const -> bool {
    return GetDependence(const_id) > ConstantDependence::PeriodSelf;
  }

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "symbolic_constants_"),
                      symbolic_constants_);
  }

  // Makes an iterable range over pairs of the instruction id and constant value
  // id for each value in the store.
  auto enumerate() const -> auto { return values_.enumerate(); }

  // Outputs assigned constant values, and all symbolic constants.
  auto OutputYaml(bool include_singletons) const -> Yaml::OutputMapping {
    return Yaml::OutputMapping([&, include_singletons](
                                   Yaml::OutputMapping::Map map) {
      map.Add("values", Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                for (auto [id, value] : values_.enumerate()) {
                  if (!include_singletons && IsSingletonInstId(id)) {
                    continue;
                  }
                  if (!value.has_value() || value.is_constant()) {
                    map.Add(PrintToString(id), Yaml::OutputScalar(value));
                  }
                }
              }));
      map.Add("symbolic_constants", symbolic_constants_.OutputYaml());
    });
  }

 private:
  const ConstantId default_;

  // A mapping from `InstId::index` to the corresponding constant value. This is
  // expected to be sparse, and may be smaller than the list of instructions if
  // there are trailing non-constant instructions.
  //
  // Set inline size to 0 because these will typically be too large for the
  // stack, while this does make File smaller.
  ValueStore<InstId, ConstantId> values_;

  // A mapping from a symbolic constant ID index to information about the
  // symbolic constant. For a concrete constant, the only information that we
  // track is the instruction ID, which is stored directly within the
  // `ConstantId`. For a symbolic constant, we also track information about
  // where the constant was used, which is stored here.
  ValueStore<ConstantId::SymbolicId, SymbolicConstant> symbolic_constants_;
};

// Given a constant ID, returns an instruction that has that constant value.
// For an unattached constant, the returned instruction is the instruction that
// defines the constant; for an attached constant, this is the instruction in
// the eval block that computes the constant value in each specific.
//
// Returns InstId::None if the ConstantId is None or NotConstant.
auto GetInstWithConstantValue(const SemIR::File& file,
                              SemIR::ConstantId const_id) -> SemIR::InstId;

// Provides storage for instructions representing deduplicated global constants.
class ConstantStore {
 public:
  explicit ConstantStore(File* sem_ir) : sem_ir_(sem_ir) {}

  // Adds a new constant instruction, or gets the existing constant with this
  // value. Returns the ID of the constant.
  //
  // This updates `sem_ir->insts()` and `sem_ir->constant_values()` if the
  // constant is new.
  auto GetOrAdd(Inst inst, ConstantDependence dependence) -> ConstantId;

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "map_"), map_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "constants_"), constants_);
  }

  // Returns a copy of the constant IDs as a vector, in an arbitrary but
  // stable order. This should not be used anywhere performance-sensitive.
  auto array_ref() const -> llvm::ArrayRef<InstId> { return constants_; }

  auto size() const -> int { return constants_.size(); }

 private:
  File* const sem_ir_;
  Map<Inst, ConstantId> map_;
  llvm::SmallVector<InstId, 0> constants_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CONSTANT_H_
