//===- lld/Core/TargetInfo.h - Linker Target Info Interface ---------------===//
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
/// Interface for target specific information to be used be readers, writers,
/// and the resolver.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_TARGET_INFO_H
#define LLD_CORE_TARGET_INFO_H

#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace llvm {
  class Triple;
}

namespace lld {
class LinkerInput;
struct LinkerOptions;
class PassManager;
class Reader;
class Writer;

class TargetInfo {
protected:
  TargetInfo(const LinkerOptions &lo) : _options(lo) {}

public:
  virtual ~TargetInfo();

  const LinkerOptions &getLinkerOptions() const { return _options; }

  llvm::Triple getTriple() const;
  virtual bool is64Bits() const;
  virtual bool isLittleEndian() const;

  virtual uint64_t getPageSize() const = 0;

  virtual StringRef getEntry() const;

  virtual void addPasses(PassManager &pm) const {}

  /// \brief Get a reference to a Reader for the given input.
  ///
  /// Will always return the same object for the same input.
  virtual ErrorOr<Reader &> getReader(const LinkerInput &input) const = 0;

  /// \brief Get the writer.
  virtual ErrorOr<Writer &> getWriter() const = 0;

  // TODO: Split out to TargetRelocationInfo.
  virtual ErrorOr<int32_t> relocKindFromString(StringRef str) const {
    int32_t val;
    if (str.getAsInteger(10, val))
      return llvm::make_error_code(llvm::errc::invalid_argument);
    return val;
  }

  virtual ErrorOr<std::string> stringFromRelocKind(int32_t kind) const {
    std::string s;
    llvm::raw_string_ostream str(s);
    str << kind;
    str.flush();
    return s;
  }

protected:
  const LinkerOptions &_options;
};
} // end namespace lld

#endif
