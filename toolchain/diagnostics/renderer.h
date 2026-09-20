// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_RENDERER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_RENDERER_H_

#include <cstdint>
#include <string>

#include "common/ostream.h"
#include "common/terminal/capabilities.h"
#include "common/terminal/output_buffer_ref.h"
#include "llvm/ADT/ArrayRef.h"
#include "toolchain/diagnostics/diagnostic.h"

namespace Carbon::Diagnostics {

namespace Internal {
struct PreparedPart;
}  // namespace Internal

// Draws diagnostics for a terminal.
//
// A diagnostic renders as its headline, and then a frame holding everything
// that explains it: which file, the source there, and the text of each label
// against the code it marks.
//
//   error: 1 argument passed to function expecting 0 arguments
//          ╭─┤ foo.carbon:19:3
//          │
//        1 │ fn Run0() {}
//          · ──────┬─────
//          ·       ╰──┤ calling function declared here
//          ┆
//   ->  19 │   Run0(1);
//          ·   ───┬───
//          ·      ╰──┤ 1 argument passed here
//          ╰────────────
//
// With snippets off, each part of the diagnostic is one line, led by its
// location and the extent of its range:
//
//   foo.carbon:19:3-9: error: 1 argument passed to function expecting 0
//   foo.carbon:1:1-12: note: calling function declared here
//
// See /toolchain/docs/diagnostics_rendering.md for the full form, the palette,
// and why it looks like this rather than like something else.
//
// The rendering is decided entirely by the capabilities passed in, so the same
// diagnostic renders the same way every time for a given terminal, and the
// default capabilities render plain ASCII. What a stream's capabilities
// actually are is `Terminal::Capabilities::Detect`'s to answer: notably it
// takes the character set from the locale whether or not a terminal is
// attached, since the locale is what says how bytes will be decoded.
class Renderer {
 public:
  // Renders for a terminal with `capabilities`. The default describes a stream
  // with no terminal behind it: no color, no line drawing, and no width.
  // Layout still fits one -- the width code is formatted to, plus the gutter
  // -- so the rendering is decided by the diagnostic rather than by the
  // environment the stream ends up in.
  explicit Renderer(const Terminal::Capabilities& capabilities = {})
      : capabilities_(capabilities) {}

  // Renders `diagnostic`, appending the bytes that draw it to `out`. Each row
  // ends in a newline, and no style is left set at the end.
  auto Render(Terminal::OutputBufferRef out, const Diagnostic& diagnostic) const
      -> void;

  // Sets whether each message names the diagnostic kind that produced it. This
  // is for tests, which match against the kind to prove they cover it.
  auto set_include_kind(bool value) -> void { include_kind_ = value; }

  // Sets whether diagnostics show snippets of the source they point at. With
  // them off, every diagnostic takes the compact form: one line per part, led
  // by its location and the extent of its range, positioned against nothing,
  // which is what a build log and a golden file want.
  auto set_snippets(bool value) -> void { snippets_ = value; }

  auto set_capabilities(const Terminal::Capabilities& capabilities) -> void {
    capabilities_ = capabilities;
  }

 private:
  // Renders one line per part of the diagnostic, led by its location.
  auto RenderCompact(Terminal::OutputBufferRef out,
                     llvm::ArrayRef<Internal::PreparedPart> parts) const
      -> void;

  // Renders the framed form into the geometry `Render` chose.
  auto RenderFramed(Terminal::OutputBufferRef out,
                    llvm::MutableArrayRef<Internal::PreparedPart> parts,
                    int headline, Level level, int frame_x, int content_x,
                    int source_columns, int columns) const -> void;

  Terminal::Capabilities capabilities_;
  bool include_kind_ = false;
  bool snippets_ = true;
};

// Returns `loc` as `<filename>:<line>:<column>`, with the parts that are
// unknown dropped from the right, and empty when even the filename is unknown.
auto FormatLocation(const Loc& loc) -> std::string;

// Writes the source snippet for `loc` to `out`, indented by `indent` columns
// and with no styling.
//
// This and `FormatLocation` are for crash handlers, which interleave a location
// and its source with text of their own rather than emitting a diagnostic. They
// share their layout with `Renderer` so that there is one implementation of it.
auto PrintSnippet(llvm::raw_ostream& out, const Loc& loc, int indent) -> void;

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_RENDERER_H_
