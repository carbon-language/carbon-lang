// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/capabilities.h"

#include <gtest/gtest.h>

#include <utility>

#include "common/filesystem.h"

namespace Carbon::Terminal {
namespace {

// Detection policy is a pure function of the environment and whether the
// stream is a terminal, so these tests build the environment directly instead
// of mutating the process environment, which would leak between tests and race
// with anything else running.
auto OnTerminal(const ColorEnvironment& env) -> ColorMode {
  return ChooseColorMode(Preference::Auto, env, /*is_terminal=*/true);
}

auto OffTerminal(const ColorEnvironment& env) -> ColorMode {
  return ChooseColorMode(Preference::Auto, env, /*is_terminal=*/false);
}

// A terminal that supports color, for tests varying one other variable.
auto ColorTerminal() -> ColorEnvironment { return {.term = "xterm-256color"}; }

TEST(CapabilitiesTest, ColorNeedsATerminal) {
  EXPECT_EQ(OnTerminal(ColorTerminal()), ColorMode::Ansi256);

  // Writing to a file or a pipe must stay plain, or every redirected build log
  // fills with escape sequences.
  EXPECT_EQ(OffTerminal(ColorTerminal()), ColorMode::NoColor);

  // A terminal that nothing says anything about can't be assumed to render
  // escapes.
  EXPECT_EQ(OnTerminal({}), ColorMode::NoColor);
  EXPECT_EQ(OnTerminal({.term = ""}), ColorMode::NoColor);
  EXPECT_EQ(OnTerminal({.term = "dumb"}), ColorMode::NoColor);
}

TEST(CapabilitiesTest, ColorFromTheEmulatorWithoutTerm) {
  // `TERM` is unset for anything not launched from a shell, but the emulator
  // sets `COLORTERM` and `TERM_PROGRAM` itself, and both are specifically
  // about color. Either one identifies the terminal on its own, at whatever
  // depth it names.
  EXPECT_EQ(OnTerminal({.colorterm = "truecolor"}), ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.colorterm = "yes"}), ColorMode::Ansi16);
  EXPECT_EQ(OnTerminal({.term_program = "vscode"}), ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term_program = "unknown"}), ColorMode::Ansi16);

  // An empty value says nothing at all.
  EXPECT_EQ(OnTerminal({.colorterm = "", .term_program = ""}),
            ColorMode::NoColor);

  // `dumb` outranks them: it states outright that escapes won't render.
  EXPECT_EQ(OnTerminal({.colorterm = "truecolor", .term = "dumb"}),
            ColorMode::NoColor);

  // And none of them enable color off a terminal.
  EXPECT_EQ(OffTerminal({.colorterm = "truecolor"}), ColorMode::NoColor);
  EXPECT_EQ(OffTerminal({.term_program = "vscode"}), ColorMode::NoColor);
}

TEST(CapabilitiesTest, ExplicitPreferenceWins) {
  ColorEnvironment forcing = {.force_color = "3", .term = "xterm-256color"};
  EXPECT_EQ(ChooseColorMode(Preference::Never, forcing, /*is_terminal=*/true),
            ColorMode::NoColor);

  ColorEnvironment disabling = {.no_color = "1", .term = "dumb"};
  EXPECT_EQ(ChooseColorMode(Preference::Always, disabling,
                            /*is_terminal=*/false),
            ColorMode::Ansi16);

  // Forcing color on without any hint of what the terminal handles gets the
  // depth every color terminal supports.
  EXPECT_EQ(ChooseColorMode(Preference::Always, {}, /*is_terminal=*/false),
            ColorMode::Ansi16);
  EXPECT_EQ(ChooseColorMode(Preference::Always, ColorTerminal(),
                            /*is_terminal=*/false),
            ColorMode::Ansi256);
}

TEST(CapabilitiesTest, NoColor) {
  // https://no-color.org: any non-empty value disables color, whatever it is.
  EXPECT_EQ(OnTerminal({.no_color = "1", .term = "xterm-256color"}),
            ColorMode::NoColor);
  EXPECT_EQ(OnTerminal({.no_color = "0", .term = "xterm-256color"}),
            ColorMode::NoColor);

  // Being set to the empty string carries no meaning, so it must not disable
  // color: an empty variable inherited from a wrapper script would otherwise
  // silently turn color off everywhere.
  EXPECT_EQ(OnTerminal({.no_color = "", .term = "xterm-256color"}),
            ColorMode::Ansi256);

  // It outranks the forcing variables.
  EXPECT_EQ(OnTerminal({.no_color = "1", .force_color = "3"}),
            ColorMode::NoColor);
  EXPECT_EQ(OnTerminal({.no_color = "1", .clicolor_force = "1"}),
            ColorMode::NoColor);
}

TEST(CapabilitiesTest, ForceColor) {
  // Color even without a terminal, at the depth the level names.
  EXPECT_EQ(OffTerminal({.force_color = "1"}), ColorMode::Ansi16);
  EXPECT_EQ(OffTerminal({.force_color = "2"}), ColorMode::Ansi256);
  EXPECT_EQ(OffTerminal({.force_color = "3"}), ColorMode::Truecolor);

  // The level overrides what the terminal claims.
  EXPECT_EQ(OnTerminal({.force_color = "1", .colorterm = "truecolor"}),
            ColorMode::Ansi16);

  // Any other non-empty value enables color without naming a depth.
  EXPECT_EQ(OffTerminal({.force_color = "true"}), ColorMode::Ansi16);
  EXPECT_EQ(OffTerminal({.force_color = "true", .term = "xterm-256color"}),
            ColorMode::Ansi256);

  // Zero disables color outright, even on a capable terminal.
  EXPECT_EQ(OnTerminal({.force_color = "0", .term = "xterm-256color"}),
            ColorMode::NoColor);
}

TEST(CapabilitiesTest, EmptyValuesCarryNoOpinion) {
  // An empty variable means the same as an unset one throughout, so a wrapper
  // script that exports one without a value changes nothing.
  EXPECT_EQ(OffTerminal({.force_color = ""}), ColorMode::NoColor);
  EXPECT_EQ(OffTerminal({.clicolor_force = ""}), ColorMode::NoColor);
  EXPECT_EQ(OnTerminal({.no_color = "", .term = "xterm-256color"}),
            ColorMode::Ansi256);
  EXPECT_EQ(OnTerminal({.clicolor = "", .term = "xterm-256color"}),
            ColorMode::Ansi256);
  EXPECT_EQ(OnTerminal({.force_color = "", .term = "xterm-256color"}),
            ColorMode::Ansi256);
}

TEST(CapabilitiesTest, CliColor) {
  // The BSD convention: `CLICOLOR_FORCE` enables color off a terminal, and
  // `CLICOLOR=0` disables it on one.
  EXPECT_EQ(OffTerminal({.clicolor_force = "1"}), ColorMode::Ansi16);
  EXPECT_EQ(OffTerminal({.clicolor_force = "1", .term = "xterm-256color"}),
            ColorMode::Ansi256);
  // `0` means "don't force", not "disable", so a terminal still gets color.
  EXPECT_EQ(OnTerminal({.clicolor_force = "0", .term = "xterm-256color"}),
            ColorMode::Ansi256);
  EXPECT_EQ(OffTerminal({.clicolor_force = "0"}), ColorMode::NoColor);

  EXPECT_EQ(OnTerminal({.clicolor = "0", .term = "xterm-256color"}),
            ColorMode::NoColor);
  EXPECT_EQ(OnTerminal({.clicolor = "1", .term = "xterm-256color"}),
            ColorMode::Ansi256);

  // Forcing beats disabling.
  EXPECT_EQ(OnTerminal({.clicolor_force = "1", .clicolor = "0"}),
            ColorMode::Ansi16);
}

TEST(CapabilitiesTest, ColorDepthFromColorterm) {
  EXPECT_EQ(OnTerminal({.colorterm = "truecolor", .term = "xterm"}),
            ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.colorterm = "24bit", .term = "xterm"}),
            ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.colorterm = "", .term = "xterm"}), ColorMode::Ansi16);

  // `COLORTERM` can't enable color off a terminal, so a rich value inherited
  // by a redirected stream can't smuggle escapes into it.
  EXPECT_EQ(OffTerminal({.colorterm = "truecolor", .term = "xterm"}),
            ColorMode::NoColor);
}

TEST(CapabilitiesTest, ColorDepthFromTermProgram) {
  EXPECT_EQ(OnTerminal({.term_program = "vscode", .term = "xterm"}),
            ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term_program = "iTerm.app", .term = "xterm"}),
            ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term_program = "WarpTerminal", .term = "xterm"}),
            ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term_program = "Hyper", .term = "xterm"}),
            ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term_program = "Tabby", .term = "xterm"}),
            ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term_program = "Terminus", .term = "xterm"}),
            ColorMode::Truecolor);
  // Apple's Terminal renders only the 256-color palette.
  EXPECT_EQ(OnTerminal({.term_program = "Apple_Terminal", .term = "xterm"}),
            ColorMode::Ansi256);
  EXPECT_EQ(OnTerminal({.term_program = "unknown", .term = "xterm-256color"}),
            ColorMode::Ansi256);
}

TEST(CapabilitiesTest, ColorDepthFromTerm) {
  EXPECT_EQ(OnTerminal({.term = "xterm-kitty"}), ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term = "alacritty"}), ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term = "wezterm"}), ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term = "foot-extra"}), ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term = "xterm-direct"}), ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term = "ghostty"}), ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term = "contour-latest"}), ColorMode::Truecolor);
  EXPECT_EQ(OnTerminal({.term = "xterm-truecolor"}), ColorMode::Truecolor);

  EXPECT_EQ(OnTerminal({.term = "xterm-256color"}), ColorMode::Ansi256);
  EXPECT_EQ(OnTerminal({.term = "screen-256color"}), ColorMode::Ansi256);
  EXPECT_EQ(OnTerminal({.term = "putty-256"}), ColorMode::Ansi256);

  // A terminal matching both a truecolor and a 256-color pattern takes the
  // richer one, so the order these are tried in is load-bearing.
  EXPECT_EQ(OnTerminal({.term = "vte-256color"}), ColorMode::Truecolor);

  // Known to render color, but with nothing saying how much.
  EXPECT_EQ(OnTerminal({.term = "xterm"}), ColorMode::Ansi16);
  EXPECT_EQ(OnTerminal({.term = "linux"}), ColorMode::Ansi16);
}

TEST(CapabilitiesTest, Charset) {
  EXPECT_EQ(ChooseCharset(Preference::Auto, "en_US.UTF-8"), Charset::Utf8);
  EXPECT_EQ(ChooseCharset(Preference::Auto, "C.utf8"), Charset::Utf8);
  EXPECT_EQ(ChooseCharset(Preference::Auto, "en_US.utf-8"), Charset::Utf8);

  // Drawing box characters into a terminal decoding something else turns them
  // into several bytes of mojibake and destroys the alignment they were for.
  EXPECT_EQ(ChooseCharset(Preference::Auto, "C"), Charset::Ascii);
  EXPECT_EQ(ChooseCharset(Preference::Auto, "POSIX"), Charset::Ascii);
  EXPECT_EQ(ChooseCharset(Preference::Auto, "en_US.ISO-8859-1"),
            Charset::Ascii);
  EXPECT_EQ(ChooseCharset(Preference::Auto, ""), Charset::Ascii);

  EXPECT_EQ(ChooseCharset(Preference::Never, "en_US.UTF-8"), Charset::Ascii);
  EXPECT_EQ(ChooseCharset(Preference::Always, "C"), Charset::Utf8);
}

TEST(CapabilitiesTest, BackgroundFromColorFgBg) {
  // `fg;bg`, which is what `rxvt` and its derivatives write.
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "15;0"),
            Background::Dark);
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "0;15"),
            Background::Light);
  // The eighth entry is white and the ninth the dark gray after it, so the
  // halves are not simply the low and high eight.
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "0;7"),
            Background::Light);
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "15;8"),
            Background::Dark);
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "15;6"),
            Background::Dark);
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "0;9"),
            Background::Light);
  // Some terminals write a third field between the two.
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "15;default;0"),
            Background::Dark);
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "0;default;15"),
            Background::Light);
}

TEST(CapabilitiesTest, BackgroundWithNothingToGoOn) {
  // Unset, unparsable, and out of range all say nothing, and what nothing
  // gets is the assumption that costs contrast rather than legibility.
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, ""), Background::Dark);
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "default;default"),
            Background::Dark);
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "0;99"),
            Background::Dark);
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "15"),
            Background::Dark);
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Auto, "15;"),
            Background::Dark);
}

TEST(CapabilitiesTest, BackgroundPreferenceWins) {
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Dark, "0;15"),
            Background::Dark);
  EXPECT_EQ(ChooseBackground(BackgroundPreference::Light, "15;0"),
            Background::Light);
}

TEST(CapabilitiesTest, Defaults) {
  // The defaults describe a plain-text sink, which is what a file or a pipe
  // gets and what tests should use unless exercising something richer.
  Capabilities capabilities;
  EXPECT_EQ(capabilities.color_mode, ColorMode::NoColor);
  EXPECT_EQ(capabilities.charset, Charset::Ascii);
  EXPECT_EQ(capabilities.background, Background::Dark);
  EXPECT_FALSE(capabilities.is_terminal);
  EXPECT_FALSE(capabilities.columns.has_value());
}

TEST(CapabilitiesTest, Detect) {
  // Detection reads the process environment and the descriptor it is handed, so
  // only what neither can change is pinned here. What the policy decides from
  // given inputs is tested above, against `ChooseColorMode` and `ChooseCharset`
  // directly.
  //
  // It detects against a file rather than the process's own streams: those are
  // a pipe under the test runner but a terminal under a debugger, and an
  // exported `FORCE_COLOR` turns color on for either.
  auto dir = Filesystem::MakeTmpDir();
  ASSERT_TRUE(dir.ok()) << dir.error();
  auto file = dir->OpenWriteOnly("out", Filesystem::CreationOptions::CreateNew);
  ASSERT_TRUE(file.ok()) << file.error();

  Capabilities capabilities = Capabilities::Detect(*file);
  // A file is never a terminal.
  EXPECT_FALSE(capabilities.is_terminal);
  // `COLUMNS` reaches detection from the environment, so whether a width is
  // found depends on it, but one that is found is usable.
  if (capabilities.columns) {
    EXPECT_GT(*capabilities.columns, 0);
  }
  EXPECT_GT(capabilities.tab_width, 0);

  // A preference decides on its own, whatever the environment holds. Color
  // forced on picks a depth from the environment, so only that it is on can be
  // pinned here.
  EXPECT_EQ(Capabilities::Detect(
                *file, {.color = Preference::Never, .utf8 = Preference::Never})
                .color_mode,
            ColorMode::NoColor);
  EXPECT_NE(
      Capabilities::Detect(*file, {.color = Preference::Always}).color_mode,
      ColorMode::NoColor);
  EXPECT_EQ(Capabilities::Detect(*file, {.utf8 = Preference::Always}).charset,
            Charset::Utf8);
  EXPECT_EQ(Capabilities::Detect(*file, {.utf8 = Preference::Never}).charset,
            Charset::Ascii);
  EXPECT_EQ(
      Capabilities::Detect(*file, {.background = BackgroundPreference::Light})
          .background,
      Background::Light);
  EXPECT_EQ(
      Capabilities::Detect(*file, {.background = BackgroundPreference::Dark})
          .background,
      Background::Dark);

  (*std::move(file)).Close().Check();
}

}  // namespace
}  // namespace Carbon::Terminal
