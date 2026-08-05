// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <array>

#include "absl/random/random.h"
#include "common/terminal/buffer.h"
#include "common/terminal/capabilities.h"
#include "common/terminal/color.h"
#include "common/terminal/style.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon::Terminal {
namespace {

static auto RandomColor(absl::BitGen& bitgen) -> Color {
  return {absl::Uniform<uint8_t>(bitgen), absl::Uniform<uint8_t>(bitgen),
          absl::Uniform<uint8_t>(bitgen)};
}

// Benchmarks style transitions with a store-to-load dependency on the rendered
// buffer, so each iteration waits on the one before it.
static void BM_StyleTransition(benchmark::State& state, ColorMode mode) {
  // A large pool of styles keeps branch prediction from learning the innards.
  constexpr int PoolSize = 1024;
  std::array<Style, PoolSize> styles;

  absl::BitGen bitgen;

  // Generate a pool of styles. All styles have the same set of attributes
  // enabled (bold, italic, foreground color, background color, underline
  // color, and underline style), but with different random RGB values. This is
  // the case where no reset is needed and only colors change.
  for (int i = 0; i < PoolSize; ++i) {
    styles[i] = Style()
                    .Bold()
                    .Italic()
                    .Foreground(RandomColor(bitgen))
                    .Background(RandomColor(bitgen))
                    .Underline(UnderlineShape::Curly)
                    .UnderlineColor(RandomColor(bitgen));
  }

  // The style transitions accumulate in a reused buffer, for a small but
  // stable per-iteration overhead.
  llvm::SmallString<1024> str;

  int current_idx = 0;
  for (auto _ : state) {
    int next_idx = (current_idx + 1) % PoolSize;
    styles[current_idx].AppendTransitionTo(str, styles[next_idx], mode);
    // Reading the string's terminator makes each iteration wait on the store
    // the one before it made, and blocks the optimizer from guessing the
    // value.
    uint8_t last_byte = str.c_str()[str.size()];
    benchmark::DoNotOptimize(last_byte);
    current_idx = (next_idx + last_byte) % PoolSize;
    str.clear();
  }
}
BENCHMARK_CAPTURE(BM_StyleTransition, NoColor, ColorMode::NoColor);
BENCHMARK_CAPTURE(BM_StyleTransition, Ansi16, ColorMode::Ansi16);
BENCHMARK_CAPTURE(BM_StyleTransition, Ansi256, ColorMode::Ansi256);
BENCHMARK_CAPTURE(BM_StyleTransition, Truecolor, ColorMode::Truecolor);

// Benchmarks `Buffer::Render` for terminal-sized screens using a
// data-dependency feedback loop where the next buffer index depends on the
// bytes written in the previous iteration.
static void BM_BufferRender(benchmark::State& state, ColorMode mode) {
  const int width = state.range(0);
  const int height = state.range(1);

  // Given the significantly larger body of work, a much smaller pool suffices
  // without branch prediction skewing results.
  constexpr int PoolSize = 16;
  llvm::SmallVector<Buffer, PoolSize> buffers;

  absl::BitGen bitgen;

  // Generate a pool of buffers. To ensure workload consistency but without
  // being identical, cell (x, y) in all buffers have:
  // - The same style attributes enabled (fg, bg, bold, italic etc.).
  // - Different random color and character values.
  // - A different shape (a box of varying aspect ratio starting at (1, 1) with
  //   constant perimeter of 60 cells).
  for (int i = 0; i < PoolSize; ++i) {
    Buffer buffer(width, Charset::Utf8);

    auto get_style = [&](int x, int y) {
      if ((x + y) % 3 == 0) {
        return Style().Foreground(RandomColor(bitgen)).Bold();
      }
      if ((x + y) % 3 == 1) {
        return Style().Background(RandomColor(bitgen)).Italic();
      }
      return Style()
          .Underline(UnderlineShape::Single)
          .UnderlineColor(RandomColor(bitgen));
    };

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        buffer.DrawCodePoint(x, y, U'A' + absl::Uniform(bitgen, 0, 26),
                             get_style(x, y));
      }
    }

    int box_height = 6 + i;
    buffer.DrawBox(1, 1, 32 - box_height, box_height, get_style(1, 1));

    buffers.push_back(std::move(buffer));
  }

  // The rendered output accumulates in a reused buffer, for a small but
  // stable per-iteration overhead.
  llvm::SmallString<1 << 16> str;

  int current_idx = 0;
  for (auto _ : state) {
    buffers[current_idx].Render(str, mode);
    // Reading the string's terminator makes each iteration wait on the store
    // the one before it made, and blocks the optimizer from guessing the
    // value.
    uint8_t last_byte = str.c_str()[str.size()];
    benchmark::DoNotOptimize(last_byte);
    current_idx = (current_idx + 1 + last_byte) % PoolSize;
    str.clear();
  }
}
BENCHMARK_CAPTURE(BM_BufferRender, NoColor, ColorMode::NoColor)
    ->Args({80, 24})
    ->Args({120, 40});
BENCHMARK_CAPTURE(BM_BufferRender, Ansi16, ColorMode::Ansi16)
    ->Args({80, 24})
    ->Args({120, 40});
BENCHMARK_CAPTURE(BM_BufferRender, Ansi256, ColorMode::Ansi256)
    ->Args({80, 24})
    ->Args({120, 40});
BENCHMARK_CAPTURE(BM_BufferRender, Truecolor, ColorMode::Truecolor)
    ->Args({80, 24})
    ->Args({120, 40});

// A line of source of the sort a diagnostic quotes, in the two forms that
// matter for column measurement. The second has a double-width character and a
// combining mark, spelled out because the precomposed form is a single code
// point and wouldn't exercise marks at all.
static constexpr llvm::StringLiteral AsciiSource =
    "auto Foo(i32 x) -> i32 { return x * 42; }";
static constexpr llvm::StringLiteral UnicodeSource =
    "var 中文: String = \"he\xcc\x81llo\";";

// Benchmarks drawing text, which is where column measurement is paid. The
// three cases cover the regimes it runs in: no UTF-8 processing at all, UTF-8
// processing over text that turns out to be ASCII, and UTF-8 processing over
// text that isn't.
static void BM_DrawText(benchmark::State& state, Charset charset,
                        llvm::StringRef text) {
  constexpr int Width = 120;
  Buffer buffer(Width, charset);

  int row = 0;
  for (auto _ : state) {
    row = buffer.DrawText(0, row, text, Style()).y + 1;
    benchmark::DoNotOptimize(row);
    // Reuse a bounded band of rows so this measures drawing rather than the
    // buffer's growth.
    if (row > 64) {
      row = 0;
    }
  }
}
BENCHMARK_CAPTURE(BM_DrawText, AsciiCharset, Charset::Ascii, AsciiSource);
BENCHMARK_CAPTURE(BM_DrawText, Utf8Charset, Charset::Utf8, AsciiSource);
BENCHMARK_CAPTURE(BM_DrawText, Utf8CharsetWithUnicode, Charset::Utf8,
                  UnicodeSource);

// The size of the boxes the line art benchmarks draw, chosen so that one is
// about as large as the frame around a quoted snippet.
constexpr int BoxWidth = 40;
constexpr int BoxHeight = 12;

// Benchmarks drawing line art, which is where junction bookkeeping is paid.
// Every cell of a box is a separate glyph decision, and boxes are drawn over
// whatever was there before, so this covers clearing as well as drawing.
static void BM_DrawBox(benchmark::State& state, Charset charset) {
  constexpr int Width = 120;
  constexpr int Rows = 64;
  Buffer buffer(Width, charset);
  Style style = Style().Foreground(AnsiColor::Blue);

  // Boxes are drawn over a band of rows wider than one box, so that they land
  // on a mix of blank cells and cells already holding line art.
  int y = 0;
  for (auto _ : state) {
    buffer.DrawBox(0, y, BoxWidth, BoxHeight, style);
    y = (y + BoxHeight) % (Rows - BoxHeight);
    benchmark::DoNotOptimize(y);
  }
}
BENCHMARK_CAPTURE(BM_DrawBox, AsciiCharset, Charset::Ascii);
BENCHMARK_CAPTURE(BM_DrawBox, Utf8Charset, Charset::Utf8);

// Benchmarks rendering line art, which is the non-ASCII rendering that comes
// up in practice: the text a diagnostic quotes is nearly always ASCII, while
// the frames and connectors around it are box-drawing characters that each
// encode to three bytes.
static void BM_RenderLineArt(benchmark::State& state, ColorMode mode) {
  constexpr int Width = 120;
  constexpr int PoolSize = 16;
  llvm::SmallVector<Buffer, PoolSize> buffers;

  absl::BitGen bitgen;

  for (int i = 0; i < PoolSize; ++i) {
    Buffer buffer(Width, Charset::Utf8);
    // Overlapping boxes offset by a row and two columns each, so the rendered
    // rows carry line art and junctions wherever the edges cross.
    for (int box = 0; box < 4; ++box) {
      buffer.DrawBox(box * 2, box, BoxWidth + 2 * box, BoxHeight,
                     Style().Foreground(RandomColor(bitgen)));
    }
    buffers.push_back(std::move(buffer));
  }

  llvm::SmallString<1 << 14> str;

  int current_idx = 0;
  for (auto _ : state) {
    buffers[current_idx].Render(str, mode);
    // Reading the string's terminator makes each iteration wait on the store
    // the one before it made, and blocks the optimizer from guessing the
    // value.
    uint8_t last_byte = str.c_str()[str.size()];
    benchmark::DoNotOptimize(last_byte);
    current_idx = (current_idx + 1 + last_byte) % PoolSize;
    str.clear();
  }
}
BENCHMARK_CAPTURE(BM_RenderLineArt, NoColor, ColorMode::NoColor);
BENCHMARK_CAPTURE(BM_RenderLineArt, Truecolor, ColorMode::Truecolor);

}  // namespace
}  // namespace Carbon::Terminal
