//===- Error.h ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_ERROR_H
#define LLD_COFF_ERROR_H

#include <string>
#include <system_error>
#include "llvm/Support/ErrorHandling.h"

namespace lld {
namespace coff {

enum class LLDError {
  InvalidOption = 1,
  InvalidFile,
  BrokenFile,
  DuplicateSymbols,
};

class LLDErrorCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "lld"; }

  std::string message(int EV) const override {
    switch (static_cast<LLDError>(EV)) {
    case LLDError::InvalidOption:
      return "Invalid option";
    case LLDError::InvalidFile:
      return "Invalid file";
    case LLDError::BrokenFile:
      return "Broken file";
    case LLDError::DuplicateSymbols:
      return "Duplicate symbols";
    }
    llvm_unreachable("unknown error");
  }
};

inline std::error_code make_error_code(LLDError Err) {
  static LLDErrorCategory C;
  return std::error_code(static_cast<int>(Err), C);
}

} // namespace coff
} // namespace lld

#endif
