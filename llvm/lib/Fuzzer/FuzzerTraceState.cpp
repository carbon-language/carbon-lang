//===- FuzzerTraceState.cpp - Trace-based fuzzer mutator ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Data tracing.
//===----------------------------------------------------------------------===//

#include "FuzzerDictionary.h"
#include "FuzzerIO.h"
#include "FuzzerInternal.h"
#include "FuzzerMutate.h"
#include "FuzzerTracePC.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <thread>

namespace fuzzer {

// Declared as static globals for faster checks inside the hooks.
static bool RecordingMemmem = false;

int ScopedDoingMyOwnMemOrStr::DoingMyOwnMemOrStr;

class TraceState {
public:
  TraceState(MutationDispatcher &MD, const FuzzingOptions &Options,
             const Fuzzer *F)
      : MD(MD), Options(Options), F(F) {}

  void StartTraceRecording() {
    if (!Options.UseMemmem)
      return;
    RecordingMemmem = true;
    InterestingWords.clear();
    MD.ClearAutoDictionary();
  }

  void StopTraceRecording() {
    if (!RecordingMemmem)
      return;
    for (auto &W : InterestingWords)
      MD.AddWordToAutoDictionary({W});
  }

  void AddInterestingWord(const uint8_t *Data, size_t Size) {
    if (!RecordingMemmem || !F->InFuzzingThread()) return;
    if (Size <= 1) return;
    Size = std::min(Size, Word::GetMaxSize());
    Word W(Data, Size);
    InterestingWords.insert(W);
  }

 private:

  // TODO: std::set is too inefficient, need to have a custom DS here.
  std::set<Word> InterestingWords;
  MutationDispatcher &MD;
  const FuzzingOptions Options;
  const Fuzzer *F;
};

static TraceState *TS;

void Fuzzer::StartTraceRecording() {
  if (!TS) return;
  TS->StartTraceRecording();
}

void Fuzzer::StopTraceRecording() {
  if (!TS) return;
  TS->StopTraceRecording();
}

void Fuzzer::InitializeTraceState() {
  if (!Options.UseMemmem) return;
  TS = new TraceState(MD, Options, this);
}

}  // namespace fuzzer

using fuzzer::TS;

extern "C" {

ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_strstr(void *called_pc, const char *s1,
                                  const char *s2, char *result) {
  if (fuzzer::ScopedDoingMyOwnMemOrStr::DoingMyOwnMemOrStr) return;
  TS->AddInterestingWord(reinterpret_cast<const uint8_t *>(s2), strlen(s2));
}

ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_strcasestr(void *called_pc, const char *s1,
                                      const char *s2, char *result) {
  if (fuzzer::ScopedDoingMyOwnMemOrStr::DoingMyOwnMemOrStr) return;
  TS->AddInterestingWord(reinterpret_cast<const uint8_t *>(s2), strlen(s2));
}

ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_memmem(void *called_pc, const void *s1, size_t len1,
                                  const void *s2, size_t len2, void *result) {
  if (fuzzer::ScopedDoingMyOwnMemOrStr::DoingMyOwnMemOrStr) return;
  TS->AddInterestingWord(reinterpret_cast<const uint8_t *>(s2), len2);
}

}  // extern "C"
