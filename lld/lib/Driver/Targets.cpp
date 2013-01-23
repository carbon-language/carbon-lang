//===- lib/Driver/Targets.cpp - Linker Targets ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Concrete instances of the Target interface.
///
//===----------------------------------------------------------------------===//

#include "lld/Driver/Target.h"

#include "lld/Core/LinkerOptions.h"
#include "lld/Core/TargetInfo.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

using namespace lld;
using namespace std::placeholders;

class ELFTarget : public Target {
public:
  ELFTarget(std::unique_ptr<ELFTargetInfo> ti)
      : Target(std::unique_ptr<TargetInfo>(ti.get())),
        _elfTargetInfo(*ti.release()),
        _readerELF(createReaderELF(
            *_targetInfo, std::bind(&ELFTarget::getReader, this, _1))),
        _readerYAML(createReaderYAML(*_targetInfo)),
        _writer(createWriterELF(_elfTargetInfo)),
        _writerYAML(createWriterYAML(*_targetInfo)) {}

  virtual ErrorOr<lld::Reader&> getReader(const LinkerInput &input) {
    auto kind = input.getKind();
    if (!kind)
      return error_code(kind);

    if (*kind == InputKind::YAML)
      return *_readerYAML;

    if (*kind == InputKind::Object)
      return *_readerELF;

    return llvm::make_error_code(llvm::errc::invalid_argument);
  }

  virtual ErrorOr<lld::Writer&> getWriter() {
    return _targetInfo->getLinkerOptions()._outputYAML ? *_writerYAML
                                                       : *_writer;
  }

protected:
  const ELFTargetInfo &_elfTargetInfo;
  std::unique_ptr<lld::Reader> _readerELF, _readerYAML;
  std::unique_ptr<lld::Writer> _writer, _writerYAML;
};

class X86LinuxTarget LLVM_FINAL : public ELFTarget {
public:
  X86LinuxTarget(std::unique_ptr<ELFTargetInfo> ti)
      : ELFTarget(std::move(ti)) {}
};

class HexagonTarget LLVM_FINAL : public ELFTarget {
public:
  HexagonTarget(std::unique_ptr<ELFTargetInfo> ti)
      : ELFTarget(std::move(ti)) {}
};

std::unique_ptr<Target> Target::create(const LinkerOptions &lo) {
  llvm::Triple t(lo._target);
  // Create a TargetInfo.
  std::unique_ptr<ELFTargetInfo> ti(ELFTargetInfo::create(lo));

  // Create the Target
  if (t.getOS() == llvm::Triple::Linux && (t.getArch() == llvm::Triple::x86 ||
                                           t.getArch() == llvm::Triple::x86_64))
    return std::unique_ptr<Target>(new X86LinuxTarget(std::move(ti)));
  else if (t.getArch() == llvm::Triple::hexagon) 
    return std::unique_ptr<Target>(new HexagonTarget(std::move(ti)));
  return std::unique_ptr<Target>();
}
