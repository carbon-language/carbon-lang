// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_TERMINAL_CAPABILITIES_H_
#define CARBON_COMMON_TERMINAL_CAPABILITIES_H_

#include <cstdint>
#include <optional>

#include "common/filesystem.h"
#include "common/terminal/color.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Terminal {

// The encoding the terminal decodes output with.
//
// This decides far more than which characters can be drawn: rendering has to
// count the columns a run of bytes will occupy, and that count only follows
// from the code points those bytes encode if the terminal agrees about the
// encoding. Disagreeing misaligns the entire line rather than drawing a single
// character wrong.
//
// No other encoding is modeled. A terminal decoding something else, an ISO 8859
// part for example, is treated as `Ascii`. That is correct output for any of
// them, as they all encode printable ASCII as itself, and rendering in one
// natively would mean carrying its conversion and column-width tables to gain
// nothing but nicer line drawing, which `Ascii` already has a fallback for. So
// `Utf8` is used only where the environment says outright that the terminal
// decodes UTF-8, and `Ascii` does no UTF-8 processing at all.
enum class Charset : int8_t {
  // Every byte is one column, and lines are drawn from `-`, `|`, and `+`.
  //
  // Bytes outside printable ASCII are replaced rather than passed through,
  // because a terminal decoding some single-byte encoding will render them as
  // something, and there is no way to know what.
  Ascii,
  // Bytes are decoded as UTF-8, giving double-width characters two columns and
  // combining marks none, and lines are drawn with box-drawing characters.
  Utf8,
};

// Whether to use one of the terminal features detection decides about, where
// an explicit request overrides what detection would conclude.
//
// This is the tri-state that a `--color=never` style flag parses into. It says
// nothing about which feature is being requested; `Preferences` holds one of
// these per feature.
enum class Preference : int8_t {
  // Decide from the environment and the stream.
  Auto,
  // Never use the feature, whatever the environment says.
  Never,
  // Use the feature even when the stream isn't a terminal. This is what a
  // caller wants when piping into a pager, capturing output for later replay,
  // or writing a test.
  Always,
};

// What the terminal draws its text on.
//
// Nothing about the rendering depends on the exact color, only on which side of
// the middle it sits: a color chosen to read on one is hard to read on the
// other.
enum class Background : int8_t {
  Dark,
  Light,
};

// An explicit statement of what the terminal draws its text on, where `Auto`
// leaves it to detection.
//
// This is a tri-state like `Preference`, but its two settings name the answer
// rather than turning a feature on and off, so it is an enum of its own.
enum class BackgroundPreference : int8_t {
  Auto,
  Dark,
  Light,
};

// An explicit preference for each feature detection decides about, normally
// parsed from command line flags.
struct Preferences {
  Preference color = Preference::Auto;
  Preference utf8 = Preference::Auto;
  BackgroundPreference background = BackgroundPreference::Auto;
};

// The environment variables that control whether and how color is used.
//
// An unset variable and one set to the empty string mean the same thing
// throughout: no opinion. Values point into the process environment and are
// invalidated by anything that modifies it.
struct ColorEnvironment {
  // Reads the variables from the process environment.
  static auto FromProcess() -> ColorEnvironment;

  llvm::StringRef no_color;
  llvm::StringRef clicolor_force;
  llvm::StringRef force_color;
  llvm::StringRef clicolor;
  llvm::StringRef colorterm;
  llvm::StringRef term_program;
  llvm::StringRef term;
};

// Returns the color mode to render with.
//
// Whether to use color at all is decided first, from highest priority to
// lowest:
//
// - An explicit `Never` or `Always` preference.
// - `NO_COLOR` set to any non-empty value disables color: see
//   https://no-color.org.
// - `FORCE_COLOR=0` disables color, following the Node convention.
// - Any other non-empty `FORCE_COLOR`, or a non-empty `CLICOLOR_FORCE` other
//   than `0`, enables color even when the stream isn't a terminal.
// - `CLICOLOR=0` disables color.
// - `TERM=dumb` disables color, being an explicit statement that escape
//   sequences won't render.
// - Otherwise color is used only when the stream is a terminal and something
//   identifies that terminal: `TERM` set to anything else, or `COLORTERM` or
//   `TERM_PROGRAM` set at all. The latter two stand on their own rather than
//   refining `TERM`, because the emulator sets them itself and they are
//   specifically about color, while `TERM` is left unset by anything not
//   launched from a shell.
//
// How much color to use is then guessed from `FORCE_COLOR`'s level,
// `COLORTERM`, `TERM_PROGRAM`, and `TERM`, falling back to `Ansi16` when color
// is called for but nothing says how much of it works. Apart from
// `FORCE_COLOR`, which is a request rather than a description, none of these
// enable color on their own, so a rich `COLORTERM` inherited by a redirected
// stream can't put escape sequences into it.
//
// Depth comes from enumerating known terminals rather than from terminfo,
// which trades a list to maintain here for not depending on databases that are
// routinely absent from the containers and CI images this runs in. An
// unrecognized terminal gets the conservative answer.
//
// This is separated from `Capabilities::Detect` so that the policy can be
// tested without touching the process environment.
auto ChooseColorMode(Preference preference, const ColorEnvironment& env,
                     bool is_terminal) -> ColorMode;

// Returns the encoding to render with, where `locale` is the value of the
// first set variable among `LC_ALL`, `LC_CTYPE`, and `LANG`.
//
// Only a locale that names UTF-8 gets `Utf8`. Guessing wrong in that direction
// costs alignment on every line that isn't pure ASCII, while guessing wrong
// the other way only makes output plainer.
auto ChooseCharset(Preference preference, llvm::StringRef locale) -> Charset;

// Returns what the terminal draws its text on, where `colorfgbg` is the value
// of `COLORFGBG`.
//
// That variable is the only thing a process can read without talking to the
// terminal. `rxvt` and its derivatives set it, as do a few others, to the
// foreground and background palette indices separated by `;` -- sometimes with
// a third field between them -- so the background is the last of them. An index
// of 0 through 6 or 8 is a dark one, 7 and 9 through 15 a light one, and
// anything outside that range says nothing.
//
// It is missing far more often than it is present, and stale when the user
// changes their theme without restarting, so anything it doesn't answer is
// treated as dark. Guessing wrong that way costs contrast; guessing wrong the
// other way puts light text on a light background.
auto ChooseBackground(BackgroundPreference preference,
                      llvm::StringRef colorfgbg) -> Background;

// The width to lay out for when nothing says how wide the output is.
//
// Layout always has a width to fit, because the alternative is output laid out
// as if nothing bounded it, which a terminal then wraps at column zero --
// breaking every indent and gutter it was given, and in the middle of whatever
// word it lands on. The cost of guessing is asymmetric: a viewer wider than
// this sees slack on the right, while one narrower sees the wrapping done
// twice, ours and then its own.
//
// Eighty is the traditional terminal width, and narrower ones are rare enough
// that fitting them would cost more in wasted width everywhere else.
inline constexpr int DefaultColumns = 80;

// The columns between tab stops, absent anything saying otherwise.
//
// Eight is the interval terminfo records as `it#8` for all but a handful of
// legacy entries. Nothing measures a terminal's stops, so unlike its width this
// stands in for no measurement: `Capabilities` carries it as a plain value
// rather than as one a caller can tell apart from an absence.
inline constexpr int DefaultTabWidth = 8;

// What the terminal behind a stream can render, and how wide it is.
//
// Detect this once per stream at startup and pass it down; the fields come from
// environment queries and system calls that shouldn't be repeated per
// diagnostic.
struct Capabilities {
  // Detects the capabilities of the terminal behind `file`, honoring
  // `preferences`.
  //
  // Detection reads the descriptor directly, because `isatty` and `TIOCGWINSZ`
  // are what answer the question and no stream abstraction exposes them.
  // LLVM's `raw_ostream::has_colors()` is not a substitute for the enablement
  // rule above, which recognizes terminals its `TERM` list doesn't.
  //
  // An `OSC 11` query would ask the terminal what it draws on, which is the
  // only accurate answer and what `vim`, `delta`, and `bat` do. It isn't one a
  // non-interactive tool can use: the reply has to be waited for, and drawing
  // with it only when it arrives in time would leave the colors depending on
  // that. Reading it also consumes whatever was typed ahead for the next shell
  // command, along with the input a compile may be taking from stdin.
  // `COLORFGBG` is the passive stand-in that costs none of this.
  static auto Detect(Filesystem::WriteFileRef file,
                     Preferences preferences = {}) -> Capabilities;

  // The richest color escapes the terminal is believed to understand.
  ColorMode color_mode = ColorMode::NoColor;

  // The encoding the terminal decodes output with.
  Charset charset = Charset::Ascii;

  // What the terminal draws its text on.
  Background background = Background::Dark;

  // Whether the stream is attached to a terminal at all. Note that color can
  // still be in use when this is false, if the environment forces it.
  bool is_terminal = false;

  // The terminal's width, or none when nothing says how wide the output is.
  // Positive whenever it is set, so layout can divide by it freely.
  //
  // This says what was measured, and nothing is invented to fill it in: an
  // absence is a real answer about a pipe nobody described. Laying out still
  // needs a width, and `DefaultColumns` is what a layout falls back to, so
  // whether this is set decides whether output is fitted to the terminal in
  // front of it or to a width chosen to be safe wherever it ends up.
  std::optional<int> columns;

  // The columns between the terminal's tab stops, which is what a tab in text
  // advances to the next of.
  //
  // TODO: Nothing sets this away from `DefaultTabWidth`. A terminal's stops are
  // mutable at runtime -- `hts` sets one and `tbc` clears them -- so the only
  // report of the live ones is `DECRQPSR`, which few emulators outside `xterm`
  // answer, or a `DSR-CPR` round trip after writing a tab, which nearly all do.
  // Either means putting the descriptor in raw mode and reading a reply with a
  // timeout. Add it when a terminal that disagrees with eight is worth that.
  int tab_width = DefaultTabWidth;
};

}  // namespace Carbon::Terminal

#endif  // CARBON_COMMON_TERMINAL_CAPABILITIES_H_
