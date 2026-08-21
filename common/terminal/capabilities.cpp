// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/capabilities.h"

#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdlib>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

namespace Carbon::Terminal {

// Returns the value of `name` in the process environment, empty when unset.
static auto GetEnv(const char* name) -> llvm::StringRef {
  const char* value = std::getenv(name);
  return value ? llvm::StringRef(value) : llvm::StringRef();
}

auto ColorEnvironment::FromProcess() -> ColorEnvironment {
  return {.no_color = GetEnv("NO_COLOR"),
          .clicolor_force = GetEnv("CLICOLOR_FORCE"),
          .force_color = GetEnv("FORCE_COLOR"),
          .clicolor = GetEnv("CLICOLOR"),
          .colorterm = GetEnv("COLORTERM"),
          .term_program = GetEnv("TERM_PROGRAM"),
          .term = GetEnv("TERM")};
}

// Returns whether the environment and the stream call for color, ignoring any
// explicit preference. See `ChooseColorMode` for the precedence this
// implements and where it comes from.
static auto EnvironmentEnablesColor(const ColorEnvironment& env,
                                    bool is_terminal) -> bool {
  if (!env.no_color.empty()) {
    return false;
  }
  // The forcing variables use `0` to decline to force, and `FORCE_COLOR` takes
  // it further as a request to disable.
  if (env.force_color == "0") {
    return false;
  }
  if (!env.force_color.empty() ||
      (!env.clicolor_force.empty() && env.clicolor_force != "0")) {
    return true;
  }
  if (env.clicolor == "0") {
    return false;
  }
  if (!is_terminal) {
    return false;
  }

  // `dumb` says outright that escape sequences won't render, which outranks
  // anything below claiming they will.
  if (env.term == "dumb") {
    return false;
  }

  // Any of these identifies the terminal as something that renders escapes. A
  // terminal that none of them describe can't be assumed to.
  //
  // `COLORTERM` and `TERM_PROGRAM` stand on their own rather than refining
  // `TERM`: `TERM` names a terminfo entry, while these name the emulator and
  // the color it handles.
  return !env.term.empty() || !env.colorterm.empty() ||
         !env.term_program.empty();
}

// Returns the richest color escapes the terminal is believed to accept.
//
// Every signal here is a heuristic: there is no way to ask a terminal what it
// supports without writing to it and parsing a reply, which would be far too
// invasive for a compiler. Guessing too high garbles color on a terminal that
// can't keep up, and guessing too low only makes output plainer, so unknown
// terminals get the conservative answer.
static auto DetectColorDepth(const ColorEnvironment& env) -> ColorMode {
  // `FORCE_COLOR`'s levels name a depth outright.
  if (auto mode = llvm::StringSwitch<std::optional<ColorMode>>(env.force_color)
                      .Case("1", ColorMode::Ansi16)
                      .Case("2", ColorMode::Ansi256)
                      .Case("3", ColorMode::Truecolor)
                      .Default(std::nullopt)) {
    return *mode;
  }

  // The convention documented at
  // https://github.com/termstandard/colors#checking-for-colorterm.
  if (env.colorterm == "truecolor" || env.colorterm == "24bit") {
    return ColorMode::Truecolor;
  }

  // `TERM_PROGRAM` identifies the emulator regardless of how `TERM` is set,
  // which matters because several of these ship a conservative `TERM` while
  // rendering far more than it claims.
  {
    if (auto mode =
            llvm::StringSwitch<std::optional<ColorMode>>(env.term_program)
                .Case("vscode", ColorMode::Truecolor)
                .Case("iTerm.app", ColorMode::Truecolor)
                .Case("WarpTerminal", ColorMode::Truecolor)
                .Case("Hyper", ColorMode::Truecolor)
                .Case("Tabby", ColorMode::Truecolor)
                .Case("Terminus", ColorMode::Truecolor)
                // Apple's Terminal renders only the 256-color palette.
                .Case("Apple_Terminal", ColorMode::Ansi256)
                .Default(std::nullopt)) {
      return *mode;
    }
  }

  // The enumerated terminals that stand in for a terminfo lookup.
  {
    if (auto mode = llvm::StringSwitch<std::optional<ColorMode>>(env.term)
                        .Case("xterm-kitty", ColorMode::Truecolor)
                        .Case("alacritty", ColorMode::Truecolor)
                        .Case("wezterm", ColorMode::Truecolor)
                        .Case("ghostty", ColorMode::Truecolor)
                        .StartsWith("foot", ColorMode::Truecolor)
                        .StartsWith("contour", ColorMode::Truecolor)
                        .StartsWith("vte", ColorMode::Truecolor)
                        .EndsWith("-direct", ColorMode::Truecolor)
                        .EndsWith("-truecolor", ColorMode::Truecolor)
                        .EndsWith("-256color", ColorMode::Ansi256)
                        .EndsWith("-256", ColorMode::Ansi256)
                        .Default(std::nullopt)) {
      return *mode;
    }
  }

  // Color is called for, but nothing said how much of it works.
  return ColorMode::Ansi16;
}

auto ChooseColorMode(Preference preference, const ColorEnvironment& env,
                     bool is_terminal) -> ColorMode {
  switch (preference) {
    case Preference::Never:
      return ColorMode::NoColor;
    case Preference::Always:
      break;
    case Preference::Auto:
      if (!EnvironmentEnablesColor(env, is_terminal)) {
        return ColorMode::NoColor;
      }
      break;
  }
  return DetectColorDepth(env);
}

auto ChooseCharset(Preference preference, llvm::StringRef locale) -> Charset {
  switch (preference) {
    case Preference::Never:
      return Charset::Ascii;
    case Preference::Always:
      return Charset::Utf8;
    case Preference::Auto:
      break;
  }

  // Locale names spell the encoding several ways: `en_US.UTF-8`, `C.utf8`, and
  // bare `UTF-8` all appear in the wild.
  return locale.contains_insensitive("utf-8") ||
                 locale.contains_insensitive("utf8")
             ? Charset::Utf8
             : Charset::Ascii;
}

auto ChooseBackground(BackgroundPreference preference,
                      llvm::StringRef colorfgbg) -> Background {
  switch (preference) {
    case BackgroundPreference::Dark:
      return Background::Dark;
    case BackgroundPreference::Light:
      return Background::Light;
    case BackgroundPreference::Auto:
      break;
  }

  // The background is the last field, since some terminals write a third one
  // between the foreground and it.
  llvm::StringRef background = colorfgbg.rsplit(';').second;
  unsigned index = 0;
  if (!llvm::to_integer(background, index) || index > 15) {
    // Anything else, including the `default` some terminals write and the
    // variable being unset, says nothing.
    return Background::Dark;
  }
  // The first eight palette entries are the dark half, except that the eighth
  // is white and the ninth is the dark gray that follows it.
  return (index <= 6 || index == 8) ? Background::Dark : Background::Light;
}

// Returns the locale that determines the terminal's character encoding,
// following the precedence POSIX defines for `LC_CTYPE`.
static auto GetLocale() -> llvm::StringRef {
  for (const char* name : {"LC_ALL", "LC_CTYPE", "LANG"}) {
    if (llvm::StringRef value = GetEnv(name); !value.empty()) {
      return value;
    }
  }
  return "";
}

// Returns the terminal's width in columns, or nullopt when there is nothing to
// ask.
//
// `COLUMNS` comes first: when it is exported, the user has deliberately
// overridden the real width.
static auto GetColumns(int fd) -> std::optional<int> {
  int columns = 0;
  if (llvm::to_integer(GetEnv("COLUMNS"), columns) && columns > 0) {
    return columns;
  }

  struct winsize size = {};
  if (ioctl(fd, TIOCGWINSZ, &size) == 0 && size.ws_col > 0) {
    return size.ws_col;
  }
  return std::nullopt;
}

auto Capabilities::Detect(Filesystem::WriteFileRef file,
                          Preferences preferences) -> Capabilities {
  int fd = file.unix_fd();

  Capabilities capabilities;
  capabilities.is_terminal = isatty(fd) != 0;
  capabilities.color_mode =
      ChooseColorMode(preferences.color, ColorEnvironment::FromProcess(),
                      capabilities.is_terminal);
  capabilities.charset = ChooseCharset(preferences.utf8, GetLocale());
  capabilities.background =
      ChooseBackground(preferences.background, GetEnv("COLORFGBG"));
  capabilities.columns = GetColumns(fd);

  return capabilities;
}

}  // namespace Carbon::Terminal
