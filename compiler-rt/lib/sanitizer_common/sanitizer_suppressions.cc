//===-- sanitizer_suppressions.cc -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Suppression parsing/matching code shared between TSan and LSan.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_suppressions.h"

#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"

namespace __sanitizer {

static const char *const kTypeStrings[SuppressionTypeCount] = {
    "none",   "race", "mutex",           "thread",
    "signal", "leak", "called_from_lib", "deadlock"};

bool TemplateMatch(char *templ, const char *str) {
  if (str == 0 || str[0] == 0)
    return false;
  bool start = false;
  if (templ && templ[0] == '^') {
    start = true;
    templ++;
  }
  bool asterisk = false;
  while (templ && templ[0]) {
    if (templ[0] == '*') {
      templ++;
      start = false;
      asterisk = true;
      continue;
    }
    if (templ[0] == '$')
      return str[0] == 0 || asterisk;
    if (str[0] == 0)
      return false;
    char *tpos = (char*)internal_strchr(templ, '*');
    char *tpos1 = (char*)internal_strchr(templ, '$');
    if (tpos == 0 || (tpos1 && tpos1 < tpos))
      tpos = tpos1;
    if (tpos != 0)
      tpos[0] = 0;
    const char *str0 = str;
    const char *spos = internal_strstr(str, templ);
    str = spos + internal_strlen(templ);
    templ = tpos;
    if (tpos)
      tpos[0] = tpos == tpos1 ? '$' : '*';
    if (spos == 0)
      return false;
    if (start && spos != str0)
      return false;
    start = false;
    asterisk = false;
  }
  return true;
}

bool SuppressionContext::Match(const char *str, SuppressionType type,
                               Suppression **s) {
  can_parse_ = false;
  uptr i;
  for (i = 0; i < suppressions_.size(); i++)
    if (type == suppressions_[i].type &&
        TemplateMatch(suppressions_[i].templ, str))
      break;
  if (i == suppressions_.size()) return false;
  *s = &suppressions_[i];
  return true;
}

static const char *StripPrefix(const char *str, const char *prefix) {
  while (str && *str == *prefix) {
    str++;
    prefix++;
  }
  if (!*prefix)
    return str;
  return 0;
}

void SuppressionContext::Parse(const char *str) {
  // Context must not mutate once Match has been called.
  CHECK(can_parse_);
  const char *line = str;
  while (line) {
    while (line[0] == ' ' || line[0] == '\t')
      line++;
    const char *end = internal_strchr(line, '\n');
    if (end == 0)
      end = line + internal_strlen(line);
    if (line != end && line[0] != '#') {
      const char *end2 = end;
      while (line != end2 && (end2[-1] == ' ' || end2[-1] == '\t'))
        end2--;
      int type;
      for (type = 0; type < SuppressionTypeCount; type++) {
        const char *next_char = StripPrefix(line, kTypeStrings[type]);
        if (next_char && *next_char == ':') {
          line = ++next_char;
          break;
        }
      }
      if (type == SuppressionTypeCount) {
        Printf("%s: failed to parse suppressions\n", SanitizerToolName);
        Die();
      }
      Suppression s;
      s.type = static_cast<SuppressionType>(type);
      s.templ = (char*)InternalAlloc(end2 - line + 1);
      internal_memcpy(s.templ, line, end2 - line);
      s.templ[end2 - line] = 0;
      s.hit_count = 0;
      s.weight = 0;
      suppressions_.push_back(s);
    }
    if (end[0] == 0)
      break;
    line = end + 1;
  }
}

uptr SuppressionContext::SuppressionCount() const {
  return suppressions_.size();
}

const Suppression *SuppressionContext::SuppressionAt(uptr i) const {
  CHECK_LT(i, suppressions_.size());
  return &suppressions_[i];
}

void SuppressionContext::GetMatched(
    InternalMmapVector<Suppression *> *matched) {
  for (uptr i = 0; i < suppressions_.size(); i++)
    if (suppressions_[i].hit_count)
      matched->push_back(&suppressions_[i]);
}

const char *SuppressionTypeString(SuppressionType t) {
  CHECK(t < SuppressionTypeCount);
  return kTypeStrings[t];
}

}  // namespace __sanitizer
