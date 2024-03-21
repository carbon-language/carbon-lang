// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_CONVERTER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_CONVERTER_H_

#include <cstdint>

#include "common/check.h"
#include "llvm/ADT/Any.h"
#include "toolchain/diagnostics/diagnostic.h"

namespace Carbon {

// Known diagnostic type conversions. These are enumerated because `llvm::Any`
// doesn't expose the contained type; instead, we infer it from a given
// diagnostic.
enum class DiagnosticTypeConversion : int8_t {
  None,
  NameId,
  TypeId,
};

// An interface that can convert some representation of a location into a
// diagnostic location.
template <typename LocationT>
class DiagnosticConverter {
 public:
  virtual ~DiagnosticConverter() = default;

  virtual auto ConvertLocation(LocationT loc) const -> DiagnosticLocation = 0;

  // Converts arg types as needed. Not all uses support conversion, so the
  // default simply errors.
  virtual auto ConvertArg(DiagnosticTypeConversion conversion,
                          llvm::Any /*arg*/) const -> llvm::Any {
    CARBON_FATAL() << "Unexpected call to ConvertArg: "
                   << static_cast<int8_t>(conversion);
  }
};

// Used by types to indicate a DiagnosticTypeConversion that results in the
// provided StorageType. For example, to convert NameId to a std::string, we
// write:
//
// struct NameId {
//   using DiagnosticType =
//       DiagnosticTypeInfo<std::string, DiagnosticTypeConversion::NameId>;
// };
template <typename StorageTypeT, DiagnosticTypeConversion ConversionV>
struct DiagnosticTypeInfo {
  using StorageType = StorageTypeT;
  static constexpr DiagnosticTypeConversion Conversion = ConversionV;
};

namespace Internal {

// Determines whether there's a DiagnosticType member on Arg.
// Used by DiagnosticEmitter.
template <typename Arg>
concept HasDiagnosticType =
    requires { std::type_identity<typename Arg::DiagnosticType>(); };

// The default implementation with no conversion.
template <typename Arg, typename /*Unused*/ = void>
struct DiagnosticTypeForArg
    : public DiagnosticTypeInfo<Arg, DiagnosticTypeConversion::None> {};

// Exposes a custom conversion for an argument type.
template <typename Arg>
  requires HasDiagnosticType<Arg>
struct DiagnosticTypeForArg<Arg> : public Arg::DiagnosticType {};

}  // namespace Internal

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_CONVERTER_H_
