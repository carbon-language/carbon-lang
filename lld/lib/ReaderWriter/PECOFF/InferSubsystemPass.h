//===- lib/ReaderWriter/PECOFF/InferSubsystemPass.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PE_COFF_INFER_SUBSYSTEM_PASS_H
#define LLD_READER_WRITER_PE_COFF_INFER_SUBSYSTEM_PASS_H

#include "Atoms.h"
#include "lld/Core/Pass.h"
#include <vector>

namespace lld {
namespace pecoff {

// Infers subsystem from entry point function name.
class InferSubsystemPass : public lld::Pass {
public:
  InferSubsystemPass(PECOFFLinkingContext &ctx) : _ctx(ctx) {}

  std::error_code perform(SimpleFile &file) override {
    if (_ctx.getSubsystem() != WindowsSubsystem::IMAGE_SUBSYSTEM_UNKNOWN)
      return std::error_code();

    if (_ctx.isDll()) {
      _ctx.setSubsystem(WindowsSubsystem::IMAGE_SUBSYSTEM_WINDOWS_GUI);
      return std::error_code();
    }

    // Scan the resolved symbols to infer the subsystem.
    const std::string wWinMain = _ctx.decorateSymbol("wWinMainCRTStartup");
    const std::string wWinMainAt = _ctx.decorateSymbol("wWinMainCRTStartup@");
    const std::string winMain = _ctx.decorateSymbol("WinMainCRTStartup");
    const std::string winMainAt = _ctx.decorateSymbol("WinMainCRTStartup@");
    const std::string wmain = _ctx.decorateSymbol("wmainCRTStartup");
    const std::string wmainAt = _ctx.decorateSymbol("wmainCRTStartup@");
    const std::string main = _ctx.decorateSymbol("mainCRTStartup");
    const std::string mainAt = _ctx.decorateSymbol("mainCRTStartup@");

    for (const DefinedAtom *atom : file.definedAtoms()) {
      if (atom->name() == wWinMain || atom->name().startswith(wWinMainAt) ||
          atom->name() == winMain || atom->name().startswith(winMainAt)) {
        _ctx.setSubsystem(WindowsSubsystem::IMAGE_SUBSYSTEM_WINDOWS_GUI);
        return std::error_code();
      }
      if (atom->name() == wmain || atom->name().startswith(wmainAt) ||
          atom->name() == main || atom->name().startswith(mainAt)) {
        _ctx.setSubsystem(WindowsSubsystem::IMAGE_SUBSYSTEM_WINDOWS_CUI);
        return std::error_code();
      }
    }
    llvm::report_fatal_error("Failed to infer subsystem");

    return std::error_code();
  }

private:
  PECOFFLinkingContext &_ctx;
};

} // namespace pecoff
} // namespace lld

#endif
