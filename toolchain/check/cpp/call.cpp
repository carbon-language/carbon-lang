// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/call.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/call.h"
#include "toolchain/check/cpp/operators.h"
#include "toolchain/check/cpp/overload_resolution.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns whether the function is an imported C++ operator member function.
static auto IsCppOperatorMethod(Context& context, SemIR::FunctionId function_id)
    -> bool {
  SemIR::ClangDeclId clang_decl_id =
      context.functions().Get(function_id).clang_decl_id;
  return clang_decl_id.has_value() &&
         IsCppOperatorMethodDecl(
             context.clang_decls().Get(clang_decl_id).key.decl);
}

auto PerformCallToCppFunction(Context& context, SemIR::LocId loc_id,
                              SemIR::CppOverloadSetId overload_set_id,
                              SemIR::InstId self_id,
                              llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId {
  SemIR::InstId callee_id = PerformCppOverloadResolution(
      context, loc_id, overload_set_id, self_id, arg_ids);
  SemIR::Callee callee = GetCallee(context.sem_ir(), callee_id);
  CARBON_KIND_SWITCH(callee) {
    case CARBON_KIND(SemIR::CalleeError _): {
      return SemIR::ErrorInst::InstId;
    }
    case CARBON_KIND(SemIR::CalleeFunction fn): {
      CARBON_CHECK(!fn.self_id.has_value());
      if (self_id.has_value()) {
        // Preserve the `self` argument from the original callee.
        fn.self_id = self_id;
      } else if (IsCppOperatorMethod(context, fn.function_id)) {
        // Adjust `self` and args for C++ overloaded operator methods.
        fn.self_id = arg_ids.consume_front();
      }
      return PerformCallToFunction(context, loc_id, callee_id, fn, arg_ids);
    }
    case CARBON_KIND(SemIR::CalleeCppOverloadSet _): {
      CARBON_FATAL("overloads can't be recursive");
    }
    case CARBON_KIND(SemIR::CalleeNonFunction _): {
      CARBON_FATAL("overloads should produce functions");
    }
  }
}

}  // namespace Carbon::Check
