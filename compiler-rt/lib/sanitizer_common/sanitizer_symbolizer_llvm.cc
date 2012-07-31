//===-- sanitizer_symbolizer_llvm.cc --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a wrapper around llvm::DIContext, moved to separate file to
// include LLVM headers in a single place in sanitizer library. If macro
// SANITIZER_USES_LLVM_LIBS is not defined, then sanitizer runtime
// will not include LLVM headers and will not require static
// LLVM libraries to link with.
// In this case, the symbolizer will just return zeroes instead of
// valid file/line info.
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_symbolizer.h"

#ifdef SANITIZER_USES_LLVM_LIBS
# ifndef __STDC_LIMIT_MACROS
#  define __STDC_LIMIT_MACROS 1
# endif
# ifndef __STDC_CONSTANT_MACROS
#  define __STDC_CONSTANT_MACROS 1
# endif
# include "llvm/ADT/StringRef.h"
# include "llvm/DebugInfo/DIContext.h"

namespace __sanitizer {

static llvm::StringRef ToStringRef(const DWARFSection &section) {
  return llvm::StringRef(section.data, section.size);
}

class DWARFContext : public llvm::DIContext {};

DWARFContext *getDWARFContext(DWARFSection debug_info,
                              DWARFSection debug_abbrev,
                              DWARFSection debug_aranges,
                              DWARFSection debug_line,
                              DWARFSection debug_str) {
  return (DWARFContext*)llvm::DIContext::getDWARFContext(
      true, ToStringRef(debug_info), ToStringRef(debug_abbrev),
      llvm::StringRef(),  // don't use .debug_aranges for now.
      ToStringRef(debug_line), ToStringRef(debug_str));
}

void getLineInfoFromContext(DWARFContext *context, AddressInfo *info) {
  CHECK(context);
  uint32_t flags = llvm::DILineInfoSpecifier::FileLineInfo |
                   llvm::DILineInfoSpecifier::AbsoluteFilePath |
                   llvm::DILineInfoSpecifier::FunctionName;
  llvm::DILineInfo line_info = context->getLineInfoForAddress(
      info->module_offset, flags);

  const char *function = line_info.getFunctionName();
  CHECK(function);
  if (0 != internal_strcmp("<invalid>", function))
    info->function = internal_strdup(function);
  else
    info->function = 0;

  const char *file = line_info.getFileName();
  CHECK(file);
  if (0 != internal_strcmp("<invalid>", file))
    info->file = internal_strdup(file);
  else
    info->file = 0;

  info->line = line_info.getLine();
  info->column = line_info.getColumn();
}

}  // namespace __sanitizer

#else  // SANITIZER_USES_LLVM_LIBS
namespace __sanitizer {

class DWARFContext {};

DWARFContext *getDWARFContext(DWARFSection debug_info,
                              DWARFSection debug_abbrev,
                              DWARFSection debug_aranges,
                              DWARFSection debug_line,
                              DWARFSection debug_str) {
  return 0;
}

void getLineInfoFromContext(DWARFContext *context, AddressInfo *info) {
  (void)context;
  info->function = 0;
  info->file = 0;
  info->line = 0;
  info->column = 0;
}

}  // namespace __sanitizer
#endif  // SANITIZER_USES_LLVM_LIBS
