//===- lld/ReaderWriter/MachOTargetInfo.h ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_TARGET_INFO_H
#define LLD_READER_WRITER_MACHO_TARGET_INFO_H

#include "lld/Core/TargetInfo.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/Support/ErrorHandling.h"

namespace lld {

namespace mach_o {
  class KindHandler;  // defined in lib. this header is in include.
}

class MachOTargetInfo : public TargetInfo {
public:
  MachOTargetInfo();
  ~MachOTargetInfo();

  virtual void addPasses(PassManager &pm) const;
  virtual ErrorOr<Reference::Kind> relocKindFromString(StringRef str) const;
  virtual ErrorOr<std::string> stringFromRelocKind(Reference::Kind kind) const;
  virtual bool validate(raw_ostream &diagnostics);
  
  virtual error_code parseFile(std::unique_ptr<MemoryBuffer> &mb,
                        std::vector<std::unique_ptr<File>> &result) const;

  uint32_t getCPUType() const;
  uint32_t getCPUSubType() const;

  bool addEntryPointLoadCommand() const;
  bool addUnixThreadLoadCommand() const;
  bool outputTypeHasEntry() const;

  virtual uint64_t pageZeroSize() const {
    return _pageZeroSize;
  }
  
  mach_o::KindHandler &kindHandler() const;
  
  uint32_t outputFileType() const { return _outputFileType; }
  
  enum Arch {
    arch_unknown,
    arch_x86,
    arch_x86_64,
    arch_armv6,
    arch_armv7,
    arch_armv7s,
  };
  
  enum class OS {
    macOSX,
    iOS,
    iOS_simulator
  };

  Arch arch() const { return _arch; }

  void setOutputFileType(uint32_t type) { _outputFileType = type; }
  void setArch(Arch arch) { _arch = arch; }
  bool setOS(OS os, StringRef minOSVersion);

private:
  virtual Writer &writer() const;

  /// 32-bit packed encoding of "X.Y.Z" where nibbles are xxxx.yy.zz.  
  struct PackedVersion {
    PackedVersion(StringRef);
    static bool parse(StringRef, PackedVersion&);
    bool operator<(const PackedVersion&) const;
    bool operator>=(const PackedVersion&) const;
    bool operator==(const PackedVersion&) const;
  private:
    PackedVersion(uint32_t v) : _value(v) { }

    uint32_t    _value;
  };

  bool minOS(StringRef mac, StringRef iOS) const;

  uint32_t        _outputFileType; // e.g MH_EXECUTE
  bool            _outputFileTypeStatic; // Disambiguate static vs dynamic prog
  Arch            _arch;
  OS              _os;
  PackedVersion   _osMinVersion;
  uint64_t        _pageZeroSize;
  mutable std::unique_ptr<mach_o::KindHandler>  _kindHandler;
  mutable std::unique_ptr<Reader>               _machoReader;
  mutable std::unique_ptr<Reader>               _yamlReader;
  mutable std::unique_ptr<Writer>               _writer;
};

} // end namespace lld

#endif
