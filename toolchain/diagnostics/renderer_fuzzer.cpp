// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzzes the diagnostic renderer directly, on `Diagnostic` values a compiler
// would never build: degenerate locations, spans past the end of a line,
// overlapping ranges, control bytes and invalid UTF-8 in every text field, and
// every combination of capabilities. The renderer's contract is that none of
// this is ever the reason a compiler dies while reporting a problem, so what is
// checked is that every form of every shape renders without a crash. Whether
// what it draws fits the width is not checked here: an unbreakable token or a
// path legitimately overhangs, and only the line art must fit, which the
// terminal buffer CHECKs on its own.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "common/check.h"
#include "common/terminal/capabilities.h"
#include "common/terminal/output_buffer_ref.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/renderer.h"

namespace Carbon::Diagnostics {
namespace {

// A cursor over the fuzzer input that hands out bounded values, returning zero
// once the bytes run out so that decoding always terminates.
class Bytes {
 public:
  Bytes(const unsigned char* data, size_t size) : data_(data), size_(size) {}

  // Returns the next byte, or 0 when the input is exhausted.
  auto Next() -> uint8_t { return at_ < size_ ? data_[at_++] : 0; }

  // Returns the next value in `[0, bound)`, or 0 when `bound` is 0.
  auto Below(int bound) -> int { return bound > 0 ? Next() % bound : 0; }

  // Returns a signed value spanning a range that includes the degenerate
  // location values (-1) the renderer must tolerate.
  auto Small() -> int32_t { return static_cast<int32_t>(Next()) - 2; }

  // Returns a run of the raw input bytes, so text fields carry whatever the
  // fuzzer chose -- control bytes, invalid UTF-8, and embedded escapes
  // included. The reference is into the input and lives as long as it does.
  auto Text() -> llvm::StringRef {
    size_t len = Next();
    if (at_ + len > size_) {
      len = size_ - at_;
    }
    llvm::StringRef result(reinterpret_cast<const char*>(data_ + at_), len);
    at_ += len;
    return result;
  }

 private:
  const unsigned char* data_;
  size_t size_;
  size_t at_ = 0;
};

// Builds a `Loc` from the fuzzer, sharing one buffer for `line`/`file_text` so
// the renderer's pointer-containment paths are exercised rather than always
// failing their guards.
static auto MakeLoc(Bytes& bytes, llvm::StringRef file_text) -> Loc {
  llvm::StringRef line = file_text;
  // Sometimes point `line` into `file_text` (a real slice), sometimes not.
  if (!file_text.empty() && (bytes.Next() & 1)) {
    size_t start = bytes.Next() % file_text.size();
    line = file_text.substr(start, bytes.Next());
  }
  return {.filename = bytes.Text(),
          .line = line,
          .file_text = file_text,
          .line_number = bytes.Small(),
          .column_number = bytes.Small(),
          .length = bytes.Small()};
}

// A format function that ignores the format and returns fuzzer-chosen text, so
// a part's rendered words are whatever the input holds.
static auto SaysText(std::string text) -> FormatFn {
  return [text = std::move(text)](llvm::StringLiteral /*format*/,
                                  llvm::ArrayRef<llvm::Any> /*args*/) {
    return text;
  };
}

}  // namespace

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  Bytes bytes(data, size);

  Terminal::Capabilities capabilities;
  capabilities.color_mode = static_cast<Terminal::ColorMode>(bytes.Below(4));
  capabilities.charset = static_cast<Terminal::Charset>(bytes.Below(2));
  capabilities.background = static_cast<Terminal::Background>(bytes.Below(2));
  capabilities.is_terminal = bytes.Next() & 1;
  // Sometimes a width, always positive as the field promises. The range spans
  // the compact-form threshold and small values a grid could divide badly.
  if (bytes.Next() & 1) {
    capabilities.columns = 1 + bytes.Below(200);
  }

  // The source both the message and the labels point into. One shared buffer
  // makes elision between two spans reachable.
  llvm::StringRef file_text = bytes.Text();

  bool include_kind = bytes.Next() & 1;
  auto level = static_cast<Level>(bytes.Below(2));

  Message message = {.kind = Kind::TestDiagnostic,
                     .level = level,
                     .loc = MakeLoc(bytes, file_text),
                     .format = "",
                     .format_fn = SaysText(bytes.Text().str())};

  // A context leads the diagnostic in place of the message, so this reaches the
  // shapes that only exist when one does.
  llvm::SmallVector<Context, 0> contexts;
  int context_count = bytes.Below(3);
  for (int i = 0; i < context_count; ++i) {
    contexts.push_back({.loc = MakeLoc(bytes, file_text),
                        .name = "TestContext",
                        .format = "",
                        .format_fn = SaysText(bytes.Text().str())});
  }

  llvm::SmallVector<Label, 0> labels;
  int label_count = bytes.Below(8);
  for (int i = 0; i < label_count; ++i) {
    auto category = static_cast<LabelCategory>(bytes.Below(2));
    llvm::StringRef words = bytes.Text();
    labels.push_back(
        {.category = category,
         .loc = MakeLoc(bytes, file_text),
         .name = "TestLabel",
         .format = "",
         .format_fn = words.empty() ? FormatFn() : SaysText(words.str())});
  }

  Diagnostic diagnostic = {.level = level,
                           .message = std::move(message),
                           .contexts = std::move(contexts),
                           .labels = std::move(labels)};

  Renderer renderer(capabilities);
  renderer.set_include_kind(include_kind);

  // The contract is that no shape of diagnostic, however malformed, is ever the
  // reason a compiler dies while reporting a problem. Every form has to hold to
  // it, so both are drawn; a crash in either is a crash in the error path.
  for (bool snippets : {true, false}) {
    renderer.set_snippets(snippets);
    llvm::SmallString<256> rendered;
    Terminal::OutputBufferRef out(rendered);
    renderer.Render(out, diagnostic);
  }

  return 0;
}

}  // namespace Carbon::Diagnostics
