//===-- sanitizer_flags.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_flags.h"

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_list.h"
#include "sanitizer_flag_parser.h"

namespace __sanitizer {

CommonFlags common_flags_dont_use;

struct FlagDescription {
  const char *name;
  const char *description;
  FlagDescription *next;
};

IntrusiveList<FlagDescription> flag_descriptions;

// If set, the tool will install its own SEGV signal handler by default.
#ifndef SANITIZER_NEEDS_SEGV
# define SANITIZER_NEEDS_SEGV 1
#endif

void CommonFlags::SetDefaults() {
#define COMMON_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "sanitizer_flags.inc"
#undef COMMON_FLAG
}

void CommonFlags::CopyFrom(const CommonFlags &other) {
  internal_memcpy(this, &other, sizeof(*this));
}

class FlagHandlerInclude : public FlagHandlerBase {
  static const uptr kMaxIncludeSize = 1 << 15;
  FlagParser *parser_;

 public:
  explicit FlagHandlerInclude(FlagParser *parser) : parser_(parser) {}
  bool Parse(const char *value) {
    char *data;
    uptr data_mapped_size;
    int err;
    uptr len =
      ReadFileToBuffer(value, &data, &data_mapped_size,
                       Max(kMaxIncludeSize, GetPageSizeCached()), &err);
    if (!len) {
      Printf("Failed to read options from '%s': error %d\n", value, err);
      return false;
    }
    parser_->ParseString(data);
    UnmapOrDie(data, data_mapped_size);
    return true;
  }
};

void RegisterIncludeFlag(FlagParser *parser, CommonFlags *cf) {
  FlagHandlerInclude *fh_include =
      new (FlagParser::Alloc) FlagHandlerInclude(parser);  // NOLINT
  parser->RegisterHandler("include", fh_include,
                          "read more options from the given file");
}

void RegisterCommonFlags(FlagParser *parser, CommonFlags *cf) {
#define COMMON_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &cf->Name);
#include "sanitizer_flags.inc"
#undef COMMON_FLAG

  RegisterIncludeFlag(parser, cf);
}

}  // namespace __sanitizer
