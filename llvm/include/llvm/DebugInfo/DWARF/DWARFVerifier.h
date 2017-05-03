//===- DWARFVerifier.h ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFVERIFIER_H
#define LLVM_DEBUGINFO_DWARF_DWARFVERIFIER_H

#include <cstdint>

namespace llvm {
class raw_ostream;
class DWARFContext;
class DWARFUnit;

/// A class that verifies DWARF debug information given a DWARF Context.
class DWARFVerifier {
  raw_ostream &OS;
  DWARFContext &DCtx;
  uint32_t NumDebugInfoErrors;
  uint32_t NumDebugLineErrors;

public:
  DWARFVerifier(raw_ostream &S, DWARFContext &D)
      : OS(S), DCtx(D), NumDebugInfoErrors(0), NumDebugLineErrors(0) {}
  /// Verify the information in the .debug_info section.
  ///
  /// Any errors are reported to the stream that was this object was
  /// constructed with.
  ///
  /// @return True if the .debug_info verifies successfully, false otherwise.
  bool handleDebugInfo();

  /// Verify the information in the .debug_line section.
  ///
  /// Any errors are reported to the stream that was this object was
  /// constructed with.
  ///
  /// @return True if the .debug_line verifies successfully, false otherwise.
  bool handleDebugLine();
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFCONTEXT_H
