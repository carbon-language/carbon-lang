// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_TYPE_H_
#define CARBON_TOOLCHAIN_LOWER_TYPE_H_

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Lower {

class FileContext;

// Information about how a function is called in SemIR, used as input to
// build a FunctionTypeInfo.
struct FunctionInContext {
  FileContext* context;
  SemIR::FunctionId function_id;
  SemIR::SpecificId specific_id;
};

// How a lowered function returns its result.
enum class ReturnKind : uint8_t {
  // The function returns `void`, either because it has no return type or
  // because the returned value has no representation.
  None,
  // The function is the entry point, which returns an `i32` exit code even
  // though it has no declared return type.
  EntryPointInt32,
  // The function returns the result by copy, as an LLVM return value.
  ByCopy,
  // The function returns a pointer to the result, as an LLVM return value.
  ByReference,
  // The function initializes the result in a caller-provided `sret` parameter
  // and returns `void`.
  InPlace,
};

// Information used to build a FunctionInfo in FileContext.
struct FunctionTypeInfo {
  // The type of the lowered function.
  llvm::FunctionType* type;

  // The debug info type of the lowered function.
  llvm::DISubroutineType* di_type;

  // The indices of the `Call` parameter patterns that correspond to parameters
  // of the LLVM IR function, in the order of the LLVM IR parameter list.
  llvm::SmallVector<SemIR::CallParamIndex> lowered_param_indices;

  // The indices of any `Call` param patterns that aren't present in
  // lowered_param_indices.
  llvm::SmallVector<SemIR::CallParamIndex> unused_param_indices;

  // The names of the lowered `Call` parameters, in the same order as
  // `lowered_param_indices`.
  llvm::SmallVector<SemIR::NameId> param_name_ids;

  // When `return_param_id` is not `None`, the corresponding lowered parameter
  // should be given an `sret` attribute with this type.
  llvm::Type* sret_type = nullptr;

  // How the lowered function returns its result.
  ReturnKind return_kind = ReturnKind::None;

  // Whether the function type information is inexact, because some component
  // type was incomplete.
  bool inexact;
};

// Builds and returns a FunctionTypeInfo from the accumulated information in the
// given functions.
auto BuildFunctionTypeInfo(llvm::ArrayRef<FunctionInContext> functions)
    -> FunctionTypeInfo;

struct LoweredTypes {
  llvm::Type* llvm_ir_type;
  llvm::DIType* llvm_di_type;
};

// Builds the `llvm::Type` and `llvm::DIType` for the given instruction.
auto BuildType(FileContext& context, SemIR::InstId inst_id) -> LoweredTypes;

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_TYPE_H_
