// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_CONVERTER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_CONVERTER_H_

#include "llvm/ADT/Any.h"
#include "toolchain/diagnostics/diagnostic.h"

namespace Carbon {

// An interface that can convert some representation of a location into a
// diagnostic location.
template <typename LocT>
class DiagnosticConverter {
 public:
  // Callback type used to report context messages from ConvertLoc.
  // Note that the first parameter type is DiagnosticLoc rather than
  // LocT, because ConvertLoc must not recurse.
  using ContextFnT = llvm::function_ref<void(
      DiagnosticLoc, const Internal::DiagnosticBase<>&)>;

  virtual ~DiagnosticConverter() = default;

  // Converts a LocT to a DiagnosticLoc. ConvertLoc may invoke
  // context_fn to provide context messages.
  virtual auto ConvertLoc(LocT loc, ContextFnT context_fn) const
      -> DiagnosticLoc = 0;

  // Converts arg types as needed. Not all uses require conversion, so the
  // default returns the argument unchanged.
  virtual auto ConvertArg(llvm::Any arg) const -> llvm::Any { return arg; }
};

// Used by types to indicate a diagnostic type conversion that results in the
// provided StorageType. For example, to convert NameId to a std::string, we
// write:
//
// struct NameId {
//   using DiagnosticType = DiagnosticTypeInfo<std::string>;
// };
template <typename StorageTypeT>
struct DiagnosticTypeInfo {
  using StorageType = StorageTypeT;
};

namespace Internal {

// Determines whether there's a DiagnosticType member on Arg.
// Used by DiagnosticEmitter.
template <typename Arg>
concept HasDiagnosticType = requires { typename Arg::DiagnosticType; };

// The default implementation with no conversion.
template <typename Arg>
struct DiagnosticTypeForArg : public DiagnosticTypeInfo<Arg> {};

// Exposes a custom conversion for an argument type.
template <typename Arg>
  requires HasDiagnosticType<Arg>
struct DiagnosticTypeForArg<Arg> : public Arg::DiagnosticType {};

}  // namespace Internal

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_CONVERTER_H_
