//===- lib/ReaderWriter/PECOFF/SetSubsystemPass.h -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PE_COFF_SET_SUBSYSTEM_PASS_H
#define LLD_READER_WRITER_PE_COFF_SET_SUBSYSTEM_PASS_H

#include "lld/ReaderWriter/PECOFFLinkingContext.h"

using llvm::COFF::WindowsSubsystem;

namespace lld {
namespace pecoff {

/// If "main" or "wmain" is defined, /subsystem:console is the default. If
/// "WinMain" or "wWinMain" is defined, /subsystem:windows is the default.
class SetSubsystemPass : public lld::Pass {
public:
  SetSubsystemPass(PECOFFLinkingContext &ctx) : _ctx(ctx) {}

  void perform(std::unique_ptr<MutableFile> &file) override {
    if (_ctx.getSubsystem() != WindowsSubsystem::IMAGE_SUBSYSTEM_UNKNOWN)
      return;
    StringRef main = _ctx.decorateSymbol("main");
    StringRef wmain = _ctx.decorateSymbol("wmain");
    StringRef winmain = _ctx.decorateSymbol("WinMain");
    StringRef wwinmain = _ctx.decorateSymbol("wWinMain");
    for (auto *atom : file->defined()) {
      StringRef s = atom->name();
      if (s == main || s == wmain) {
        _ctx.setSubsystem(WindowsSubsystem::IMAGE_SUBSYSTEM_WINDOWS_CUI);
        return;
      }
      if (s == winmain || s == wwinmain) {
        _ctx.setSubsystem(WindowsSubsystem::IMAGE_SUBSYSTEM_WINDOWS_GUI);
        return;
      }
    }
    if (_ctx.isDll()) {
      _ctx.setSubsystem(WindowsSubsystem::IMAGE_SUBSYSTEM_WINDOWS_GUI);
      return;
    }
    llvm_unreachable("Failed to infer the subsystem.");
  }

private:
  PECOFFLinkingContext &_ctx;
};

} // namespace pecoff
} // namespace lld

#endif
