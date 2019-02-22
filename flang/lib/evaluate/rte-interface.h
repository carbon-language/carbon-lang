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

#ifndef FORTRAN_EVALUATE_RTE_INTERFACE_H_
#define FORTRAN_EVALUATE_RTE_INTERFACE_H_

// Defines the structure that must be used in F18 when dealing with the
// runtime, either with the target runtime or with the host runtime for
// folding purposes
// To avoid unnecessary header circular dependencies, the actual implementation
// of the templatized member function are defined in rte.h
// The header at hand must be included in order to add the rte interface data
// structure as a member of some structure.
// To actually use the rte interface, rte.h must be included. Note that rte.h
// includes the header at hand.

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace Fortran::evaluate {
class FoldingContext;
namespace rte {

using TypeCode = unsigned char;

template<typename TR, typename... TA> using FuncPointer = TR (*)(TA...);

enum class PassBy { Ref, Val };
template<typename TA, PassBy Pass = PassBy::Ref> struct ArgumentInfo {
  using Type = TA;
  static constexpr PassBy pass{Pass};
};

template<typename TR, typename... ArgInfo> struct Signature {
  // Note valid template argument are of form
  //<TR, ArgumentInfo<TA, PassBy>...> where TA and TR belong to RteTypes.
  // RteTypes is a type union defined in rte.h to avoid circular dependencies.
  // Argument of type void cannot be passed by value
  // So far TR cannot be a pointer.
  const std::string name;
};

struct RteProcedureSymbol {
  const std::string name;
  const TypeCode returnType;
  const std::vector<TypeCode> argumentsType;
  const std::vector<PassBy> argumentsPassedBy;
  const bool isElemental;
  const void *callable;
  // callable only usable by HostRteProcedureSymbol but need to be created in
  // case TargetRteProcedureSymbol is dynamically loaded because creating it
  // dynamically would be too complex

  // Construct from description using host independent types (RteTypes)
  template<typename TR, typename... ArgInfo>
  RteProcedureSymbol(
      const Signature<TR, ArgInfo...> &signature, bool isElemental = false);
};

// For target rte library info
struct TargetRteProcedureSymbol : RteProcedureSymbol {
  // Construct from description using host independent types (RteTypes)
  // Note: passing ref/val also needs to be passed by template to build
  // the callable
  template<typename TR, typename... ArgInfo>
  TargetRteProcedureSymbol(const Signature<TR, ArgInfo...> &signature,
      const std::string &symbolName, bool isElemental = false);
  const std::string symbol;
};

struct TargetRteLibrary {
  TargetRteLibrary(const std::string &name) : name{name} {}
  void AddProcedure(TargetRteProcedureSymbol &&sym) {
    const std::string name{sym.name};
    procedures.insert(std::make_pair(name, std::move(sym)));
  }
  const std::string name;
  std::multimap<std::string, const TargetRteProcedureSymbol> procedures;
};

// To use host runtime for folding
struct HostRteProcedureSymbol : RteProcedureSymbol {
  // Construct from runtime pointer with host types (float, double....)
  template<typename HostTR, typename... HostTA>
  HostRteProcedureSymbol(const std::string &name,
      FuncPointer<HostTR, HostTA...> func, bool isElemental = false);
  HostRteProcedureSymbol(const RteProcedureSymbol &rteProc, const void *handle)
    : RteProcedureSymbol{rteProc}, handle{handle} {}
  const void *handle;
};

// valid ConstantContainer are Scalar (only for elementals) and Constant
template<template<typename> typename ConstantContainer, typename TR,
    typename... TA>
using HostProcedureWrapper = std::function<ConstantContainer<TR>(
    FoldingContext &, ConstantContainer<TA>...)>;

struct HostRte {
  void AddProcedure(HostRteProcedureSymbol &&sym) {
    const std::string name{sym.name};
    procedures.insert(std::make_pair(name, std::move(sym)));
  }
  bool HasEquivalentProcedure(const RteProcedureSymbol &sym) const;
  HostRte() { DefaultInit(); }
  ~HostRte();
  void DefaultInit();  // load functions from <cmath> and <complex>
  void LoadTargetRteLibrary(const TargetRteLibrary &lib);  // TODO
  template<template<typename> typename ConstantContainer, typename TR,
      typename... TA>
  std::optional<HostProcedureWrapper<ConstantContainer, TR, TA...>>
  GetHostProcedureWrapper(const std::string &name);
  // some data structure of HostRteProcedureSymbol
  std::multimap<std::string, const HostRteProcedureSymbol> procedures;
  std::map<std::string, void *>
      dynamicallyLoadedLibraries;  // keep the handles for dlclose
};

}
}
#endif  // FORTRAN_EVALUATE_RTE_INTERFACE_H_
