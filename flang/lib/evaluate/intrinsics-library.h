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

#ifndef FORTRAN_EVALUATE_INTRINSICS_LIBRARY_H_
#define FORTRAN_EVALUATE_INTRINSICS_LIBRARY_H_

// Defines structures to be used in F18 when dealing with the intrinsic
// procedures runtime. It abstracts both:
//  - the target intrinsic procedure runtime to be used for code generation
//  - the host intrinsic runtime to be used for constant folding purposes.
// To avoid unnecessary header circular dependencies, the actual implementation
// of the templatized member function are defined in
// intrinsics-library-templates.h The header at hand is meant to be included by
// files that need to define intrinsic runtime data structure but that do not
// use them directly. To actually use the runtime data structures,
// intrinsics-library-templates.h must be included Note that
// intrinsics-library-templates.h includes the header at hand.

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace Fortran::evaluate {
class FoldingContext;

using TypeCode = unsigned char;

template<typename TR, typename... TA> using FuncPointer = TR (*)(TA...);

enum class PassBy { Ref, Val };
template<typename TA, PassBy Pass = PassBy::Ref> struct ArgumentInfo {
  using Type = TA;
  static constexpr PassBy pass{Pass};
};

template<typename TR, typename... ArgInfo> struct Signature {
  // Note valid template argument are of form
  //<TR, ArgumentInfo<TA, PassBy>...> where TA and TR belong to RuntimeTypes.
  // RuntimeTypes is a type union defined in intrinsics-library-templates.h to
  // avoid circular dependencies. Argument of type void cannot be passed by
  // value. So far TR cannot be a pointer.
  const std::string name;
};

struct IntrinsicProcedureRuntimeDescription {
  const std::string name;
  const TypeCode returnType;
  const std::vector<TypeCode> argumentsType;
  const std::vector<PassBy> argumentsPassedBy;
  const bool isElemental;
  const FuncPointer<void *> callable;
  // callable only usable by HostRuntimeIntrinsicProcedure but need to be
  // created in case TargetRuntimeIntrinsicProcedure is dynamically loaded
  // because creating it dynamically would be too complex

  // Construct from description using host independent types (RuntimeTypes)
  template<typename TR, typename... ArgInfo>
  IntrinsicProcedureRuntimeDescription(
      const Signature<TR, ArgInfo...> &signature, bool isElemental = false);
};

// TargetRuntimeIntrinsicProcedure holds target runtime information
// for an intrinsics procedure.
struct TargetRuntimeIntrinsicProcedure : IntrinsicProcedureRuntimeDescription {
  // Construct from description using host independent types (RuntimeTypes)
  // Note: passing ref/val also needs to be passed by template to build
  // the callable
  template<typename TR, typename... ArgInfo>
  TargetRuntimeIntrinsicProcedure(const Signature<TR, ArgInfo...> &signature,
      const std::string &symbolName, bool isElemental = false);
  const std::string symbol;
};

struct TargetIntrinsicProceduresLibrary {
  TargetIntrinsicProceduresLibrary(const std::string &name) : name{name} {}
  void AddProcedure(TargetRuntimeIntrinsicProcedure &&sym) {
    const std::string name{sym.name};
    procedures.insert(std::make_pair(name, std::move(sym)));
  }
  const std::string name;
  std::multimap<std::string, const TargetRuntimeIntrinsicProcedure> procedures;
};

// HostRuntimeIntrinsicProcedure allows host runtime function to be called for
// constant folding.
struct HostRuntimeIntrinsicProcedure : IntrinsicProcedureRuntimeDescription {
  // Construct from runtime pointer with host types (float, double....)
  template<typename HostTR, typename... HostTA>
  HostRuntimeIntrinsicProcedure(const std::string &name,
      FuncPointer<HostTR, HostTA...> func, bool isElemental = false);
  HostRuntimeIntrinsicProcedure(
      const IntrinsicProcedureRuntimeDescription &rteProc,
      FuncPointer<void *> handle)
    : IntrinsicProcedureRuntimeDescription{rteProc}, handle{handle} {}
  const FuncPointer<void *> handle;
};

// Defines a wrapper type that indirects calls to host runtime functions.
// Valid ConstantContainer are Scalar (only for elementals) and Constant.
template<template<typename> typename ConstantContainer, typename TR,
    typename... TA>
using HostProcedureWrapper = std::function<ConstantContainer<TR>(
    FoldingContext &, ConstantContainer<TA>...)>;

// HostIntrinsicProceduresLibrary is a data structure that holds
// HostRuntimeIntrinsicProcedure elements. It is meant for constant folding.
// When queried for an intrinsic procedure, it can return a callable object that
// implements this intrinsic if a host runtime function pointer for this
// intrinsic was added to its data structure. It can also dynamically load
// function pointer from a TargetIntrinsicProceduresLibrary if the related
// library is available on the host.
struct HostIntrinsicProceduresLibrary {
  void AddProcedure(HostRuntimeIntrinsicProcedure &&sym) {
    const std::string name{sym.name};
    procedures.insert(std::make_pair(name, std::move(sym)));
  }
  bool HasEquivalentProcedure(
      const IntrinsicProcedureRuntimeDescription &sym) const;
  HostIntrinsicProceduresLibrary() { DefaultInit(); }
  ~HostIntrinsicProceduresLibrary();
  void DefaultInit();  // Try loading libpgmath functions and then load
                       // functions from <cmath> and <complex>
  void LoadTargetIntrinsicProceduresLibrary(
      const TargetIntrinsicProceduresLibrary &lib);
  template<template<typename> typename ConstantContainer, typename TR,
      typename... TA>
  std::optional<HostProcedureWrapper<ConstantContainer, TR, TA...>>
  GetHostProcedureWrapper(const std::string &name);
  std::multimap<std::string, const HostRuntimeIntrinsicProcedure> procedures;
  std::map<std::string, void *>
      dynamicallyLoadedLibraries;  // keep the handles for dlclose
};

}
#endif  // FORTRAN_EVALUATE_INTRINSICS_LIBRARY_H_
