//===- LookasideRTDyldMM - Redirect symbol lookup via a functor -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//   Defines an adapter for RuntimeDyldMM that allows lookups for external
// symbols to go via a functor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LOOKASIDERTDYLDMM_H
#define LLVM_EXECUTIONENGINE_ORC_LOOKASIDERTDYLDMM_H

#include "llvm/ADT/STLExtras.h"
#include <memory>
#include <vector>

namespace llvm {

/// @brief Defines an adapter for RuntimeDyldMM that allows lookups for external
///        symbols to go via a functor, before falling back to the lookup logic
///        provided by the underlying RuntimeDyldMM instance.
///
///   This class is useful for redirecting symbol lookup back to various layers
/// of a JIT component stack, e.g. to enable lazy module emission.
///
template <typename BaseRTDyldMM, typename ExternalLookupFtor,
          typename DylibLookupFtor>
class LookasideRTDyldMM : public BaseRTDyldMM {
public:
  /// @brief Create a LookasideRTDyldMM intance.
  LookasideRTDyldMM(ExternalLookupFtor ExternalLookup,
                    DylibLookupFtor DylibLookup)
      : ExternalLookup(std::move(ExternalLookup)),
        DylibLookup(std::move(DylibLookup)) {}

  /// @brief Look up the given symbol address, first via the functor this
  ///        instance was created with, then (if the symbol isn't found)
  ///        via the underlying RuntimeDyldMM.
  uint64_t getSymbolAddress(const std::string &Name) override {
    if (uint64_t Addr = ExternalLookup(Name))
      return Addr;
    return BaseRTDyldMM::getSymbolAddress(Name);
  }

  uint64_t getSymbolAddressInLogicalDylib(const std::string &Name) override {
    if (uint64_t Addr = DylibLookup(Name))
      return Addr;
    return BaseRTDyldMM::getSymbolAddressInLogicalDylib(Name);
  };

  /// @brief Get a reference to the ExternalLookup functor.
  ExternalLookupFtor &getExternalLookup() { return ExternalLookup; }

  /// @brief Get a const-reference to the ExternalLookup functor.
  const ExternalLookupFtor &getExternalLookup() const { return ExternalLookup; }

  /// @brief Get a reference to the DylibLookup functor.
  DylibLookupFtor &getDylibLookup() { return DylibLookup; }

  /// @brief Get a const-reference to the DylibLookup functor.
  const DylibLookupFtor &getDylibLookup() const { return DylibLookup; }

private:
  ExternalLookupFtor ExternalLookup;
  DylibLookupFtor DylibLookup;
};

/// @brief Create a LookasideRTDyldMM from a base memory manager type, an
///        external lookup functor, and a dylib lookup functor.
template <typename BaseRTDyldMM, typename ExternalLookupFtor,
          typename DylibLookupFtor>
std::unique_ptr<
    LookasideRTDyldMM<BaseRTDyldMM, ExternalLookupFtor, DylibLookupFtor>>
createLookasideRTDyldMM(ExternalLookupFtor &&ExternalLookup,
                        DylibLookupFtor &&DylibLookup) {
  typedef LookasideRTDyldMM<BaseRTDyldMM, ExternalLookupFtor, DylibLookupFtor>
      ThisLookasideMM;
  return llvm::make_unique<ThisLookasideMM>(
      std::forward<ExternalLookupFtor>(ExternalLookup),
      std::forward<DylibLookupFtor>(DylibLookup));
}
}

#endif // LLVM_EXECUTIONENGINE_ORC_LOOKASIDERTDYLDMM_H
