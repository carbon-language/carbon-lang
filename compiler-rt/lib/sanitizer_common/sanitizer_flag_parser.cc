//===-- sanitizer_flag_parser.cc ------------------------------------------===//
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

#include "sanitizer_flag_parser.h"

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_flags.h"
#include "sanitizer_flag_parser.h"
#include "sanitizer_allocator_internal.h"

namespace __sanitizer {

void FlagParser::PrintFlagDescriptions() {
  Printf("Available flags for %s:\n", SanitizerToolName);
  for (int i = 0; i < n_flags_; ++i)
    Printf("\t%s\n\t\t- %s\n", flags_[i].name, flags_[i].desc);
}

void FlagParser::fatal_error(const char *err) {
  Printf("ERROR: %s\n", err);
  Die();
}

bool FlagParser::is_space(char c) {
  return c == ' ' || c == ',' || c == ':' || c == '\n' || c == '\t' ||
         c == '\r';
}

void FlagParser::skip_whitespace() {
  while (is_space(buf_[pos_])) ++pos_;
}

void FlagParser::parse_flag() {
  uptr name_start = pos_;
  while (buf_[pos_] != 0 && buf_[pos_] != '=' && !is_space(buf_[pos_])) ++pos_;
  if (buf_[pos_] != '=') fatal_error("expected '='");
  char *name = internal_strndup(buf_ + name_start, pos_ - name_start);

  uptr value_start = ++pos_;
  char *value;
  if (buf_[pos_] == '\'' || buf_[pos_] == '"') {
    char quote = buf_[pos_++];
    while (buf_[pos_] != 0 && buf_[pos_] != quote) ++pos_;
    if (buf_[pos_] == 0) fatal_error("unterminated string");
    value = internal_strndup(buf_ + value_start + 1, pos_ - value_start - 1);
    ++pos_; // consume the closing quote
  } else {
    while (buf_[pos_] != 0 && !is_space(buf_[pos_])) ++pos_;
    if (buf_[pos_] != 0 && !is_space(buf_[pos_]))
      fatal_error("expected separator or eol");
    value = internal_strndup(buf_ + value_start, pos_ - value_start);
  }

  bool res = run_handler(name, value);
  if (!res) {
    Printf("Flag parsing failed.");
    Die();
  }
  InternalFree(name);
  InternalFree(value);
}

void FlagParser::parse_flags() {
  while (true) {
    skip_whitespace();
    if (buf_[pos_] == 0) break;
    parse_flag();
  }

  // Do a sanity check for certain flags.
  if (common_flags_dont_use.malloc_context_size < 1)
    common_flags_dont_use.malloc_context_size = 1;
}

void FlagParser::ParseString(const char *s) {
  if (!s) return;
  // Backup current parser state to allow nested ParseString() calls.
  const char *old_buf_ = buf_;
  uptr old_pos_ = pos_;
  buf_ = s;
  pos_ = 0;

  parse_flags();

  buf_ = old_buf_;
  pos_ = old_pos_;
}

bool FlagParser::run_handler(const char *name, const char *value) {
  for (int i = 0; i < n_flags_; ++i) {
    if (internal_strcmp(name, flags_[i].name) == 0)
      return flags_[i].handler->Parse(value);
  }
  Printf("ERROR: Unknown flag: '%s'\n", name);
  return false;
}

void FlagParser::RegisterHandler(const char *name, FlagHandlerBase *handler,
                                 const char *desc) {
  CHECK(n_flags_ < kMaxFlags);
  flags_[n_flags_].name = name;
  flags_[n_flags_].desc = desc;
  flags_[n_flags_].handler = handler;
  ++n_flags_;
}

FlagParser::FlagParser() : n_flags_(0), buf_(nullptr), pos_(0) {
  flags_ = (Flag *)InternalAlloc(sizeof(Flag) * kMaxFlags);
}

FlagParser::~FlagParser() {
  for (int i = 0; i < n_flags_; ++i)
    InternalFree(flags_[i].handler);
  InternalFree(flags_);
}

}  // namespace __sanitizer
