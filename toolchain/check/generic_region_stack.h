// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_GENERIC_REGION_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_GENERIC_REGION_STACK_H_

#include "llvm/ADT/BitmaskEnum.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// A stack of enclosing regions that might be declaring or defining a generic
// entity. In such a region, we track the generic constructs that are used, such
// as symbolic constants and types, and instructions that depend on a template
// parameter.
//
// TODO: For now we're just tracking symbolic constants.
//
// We split a generic into two regions -- declaration and definition -- because
// these are in general introduced separately, and substituted into separately.
// For example, for `class C(T:! type, N:! T) { var x: T; }`, a use such as
// `C(i32, 0)*` substitutes into just the declaration, whereas a use such as
// `var x: C(i32, 0) = {.x = 0};` also substitutes into the definition.
class GenericRegionStack {
 public:
  // Ways in which an instruction can depend on a generic parameter.
  enum class DependencyKind : int8_t {
    None = 0x0,
    // The type of the instruction depends on a checked generic parameter.
    SymbolicType = 0x1,
    // The constant value of the instruction depends on a checked generic
    // parameter.
    SymbolicConstant = 0x2,
    Template = 0x4,
    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/Template)
  };

  // An instruction that depends on a generic parameter in some way.
  struct DependentInst {
    SemIR::InstId inst_id;
    DependencyKind kind;
  };

  // Pushes a region that might be declaring or defining a generic.
  auto Push() -> void;

  // Pops a generic region.
  auto Pop() -> void;

  // Adds an instruction to the list of instructions in the current region that
  // in some way depend on a generic parameter.
  auto AddDependentInst(DependentInst inst) -> void;

  // Returns the list of dependent instructions in the current generic region.
  auto PeekDependentInsts() -> llvm::ArrayRef<DependentInst>;

 private:
  // Information about an enclosing generic region that has been pushed onto the
  // stack.
  struct RegionInfo {
    // The size of `dependent_insts_` at the start of this region. Equivalently,
    // this is the first index within `dependent_insts_` that belongs to this
    // region or a region nested within it.
    int32_t first_dependent_inst;
  };

  // The current set of enclosing generic regions.
  llvm::SmallVector<RegionInfo> regions_;

  // List of symbolic constants used in any of the enclosing generic regions. We
  // keep a single vector rather than one vector per region in order to minimize
  // heap allocations.
  llvm::SmallVector<DependentInst> dependent_insts_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_GENERIC_REGION_STACK_H_
