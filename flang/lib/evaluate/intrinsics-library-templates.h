// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_EVALUATE_INTRINSICS_LIBRARY_TEMPLATES_H_
#define FORTRAN_EVALUATE_INTRINSICS_LIBRARY_TEMPLATES_H_

// This header defines the actual implementation of the templatized member
// function of the structures defined in intrinsics-library.h. It should only be
// included if these member functions are used, else intrinsics-library.h is
// sufficient. This is to avoid circular dependencies. The below implementation
// cannot be defined in .cc file because it would be too cumbersome to decide
// which version should be instantiated in a generic way.

#include "host.h"
#include "intrinsics-library.h"
#include "type.h"
#include "../common/template.h"

#include <cfenv>
#include <tuple>
#include <type_traits>

namespace Fortran::evaluate {

// Define meaningful types for the runtime
// TODO: add the support for void and descriptor
using RuntimeTypes = evaluate::AllIntrinsicTypes;

template<typename T, typename... TT> struct IndexInTupleHelper {};
template<typename T, typename... TT>
struct IndexInTupleHelper<T, std::tuple<TT...>> {
  static constexpr TypeCode value{common::TypeIndex<T, TT...>};
};

static_assert(
    std::tuple_size_v<RuntimeTypes> < std::numeric_limits<TypeCode>::max(),
    "TypeCode is too small");
template<typename T>
inline constexpr TypeCode typeCodeOf{
    IndexInTupleHelper<T, RuntimeTypes>::value};

template<TypeCode n>
using RuntimeTypeOf = typename std::tuple_element_t<n, RuntimeTypes>;

template<typename TA, PassBy Pass>
using HostArgType = std::conditional_t<Pass == PassBy::Ref,
    std::add_lvalue_reference_t<std::add_const_t<host::HostType<TA>>>,
    host::HostType<TA>>;

template<typename TR, typename... ArgInfo>
using HostFuncPointer = FuncPointer<host::HostType<TR>,
    HostArgType<typename ArgInfo::Type, ArgInfo::pass>...>;

// Helper class to handle host runtime traps, status flag and errno
class HostFloatingPointEnvironment {
public:
  void SetUpHostFloatingPointEnvironment(FoldingContext &);
  void CheckAndRestoreFloatingPointEnvironment(FoldingContext &);

private:
  std::fenv_t originalFenv_;
  std::fenv_t currentFenv_;
};

// Callable factory
template<typename TR, typename... ArgInfo> struct CallableHostWrapper {
  static Scalar<TR> scalarCallable(FoldingContext &context,
      HostFuncPointer<TR, ArgInfo...> func,
      const Scalar<typename ArgInfo::Type> &... x) {
    if constexpr (host::HostTypeExists<TR, typename ArgInfo::Type...>()) {
      HostFloatingPointEnvironment hostFPE;
      hostFPE.SetUpHostFloatingPointEnvironment(context);
      host::HostType<TR> res{
          func(host::CastFortranToHost<typename ArgInfo::Type>(x)...)};
      hostFPE.CheckAndRestoreFloatingPointEnvironment(context);
      return host::CastHostToFortran<TR>(res);
    } else {
      common::die("Internal error: Host does not supports this function types."
                  "This should not have been called for folding");
      return Scalar<TR>{};  // unreachable
    }
  }
  static constexpr inline auto MakeScalarCallable() { return &scalarCallable; }
};

template<typename TR, typename... ArgInfo>
IntrinsicProcedureRuntimeDescription::IntrinsicProcedureRuntimeDescription(
    const Signature<TR, ArgInfo...> &signature, bool isElemental)
  : name{signature.name}, returnType{typeCodeOf<TR>},
    argumentsType{typeCodeOf<typename ArgInfo::Type>...},
    argumentsPassedBy{ArgInfo::pass...}, isElemental{isElemental},
    callable{reinterpret_cast<FuncPointer<void *>>(
        CallableHostWrapper<TR, ArgInfo...>::MakeScalarCallable())} {}

template<typename HostTA> static constexpr inline PassBy PassByMethod() {
  if constexpr (std::is_pointer_v<std::decay_t<HostTA>> ||
      std::is_lvalue_reference_v<HostTA>) {
    return PassBy::Ref;
  }
  return PassBy::Val;
}

template<typename HostTA>
using ArgInfoFromHostType =
    ArgumentInfo<host::FortranType<std::remove_pointer_t<std::decay_t<HostTA>>>,
        PassByMethod<HostTA>()>;

template<typename HostTR, typename... HostTA>
using SignatureFromHostFuncPointer =
    Signature<host::FortranType<HostTR>, ArgInfoFromHostType<HostTA>...>;

template<typename HostTR, typename... HostTA>
HostRuntimeIntrinsicProcedure::HostRuntimeIntrinsicProcedure(
    const std::string &name, FuncPointer<HostTR, HostTA...> func,
    bool isElemental)
  : IntrinsicProcedureRuntimeDescription(
        SignatureFromHostFuncPointer<HostTR, HostTA...>{name}, isElemental),
    handle{reinterpret_cast<FuncPointer<void *>>(func)} {}

template<template<typename> typename ConstantContainer, typename TR,
    typename... TA>
std::optional<HostProcedureWrapper<ConstantContainer, TR, TA...>>
HostIntrinsicProceduresLibrary::GetHostProcedureWrapper(
    const std::string &name) {
  if constexpr (host::HostTypeExists<TR, TA...>()) {
    auto rteProcRange{procedures.equal_range(name)};
    const TypeCode resTypeCode{typeCodeOf<TR>};
    const std::vector<TypeCode> argTypes{typeCodeOf<TA>...};
    const size_t nargs{argTypes.size()};
    for (auto iter{rteProcRange.first}; iter != rteProcRange.second; ++iter) {
      if (nargs == iter->second.argumentsType.size() &&
          resTypeCode == iter->second.returnType &&
          (!std::is_same_v<ConstantContainer<TR>, Scalar<TR>> ||
              iter->second.isElemental)) {
        bool match{true};
        int pos{0};
        for (auto const &type : argTypes) {
          if (type != iter->second.argumentsType[pos++]) {
            match = false;
            break;
          }
        }
        if (match) {
          return {HostProcedureWrapper<ConstantContainer, TR, TA...>{
              [=](FoldingContext &context,
                  const ConstantContainer<TA> &... args) {
                auto callable{reinterpret_cast<
                    FuncPointer<ConstantContainer<TR>, FoldingContext &,
                        FuncPointer<void *>, const ConstantContainer<TA> &...>>(
                    iter->second.callable)};
                return callable(context, iter->second.handle, args...);
              }}};
        }
      }
    }
  }
  return std::nullopt;
}

template<typename TR, typename... ArgInfo>
TargetRuntimeIntrinsicProcedure::TargetRuntimeIntrinsicProcedure(
    const Signature<TR, ArgInfo...> &signature, const std::string &symbolName,
    bool isElemental)
  : IntrinsicProcedureRuntimeDescription{signature, isElemental},
    symbol{symbolName} {}

}
#endif  // FORTRAN_EVALUATE_INTRINSICS_LIBRARY_TEMPLATES_H_
