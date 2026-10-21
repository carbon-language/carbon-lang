// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/style.h"

#include <array>

#include "common/check.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon::Terminal {

// Returns the SGR parameter selecting `shape`, or an empty string for `None`.
//
// The shaped underlines are colon-separated subparameters of the plain
// underline code. Terminals that predate them are also the ones limited to the
// 16 ANSI colors, and they mishandle the subparameters rather than ignoring
// them, so in that mode every shape degrades to a plain underline.
static auto UnderlineSgrParam(UnderlineShape shape, ColorMode mode)
    -> llvm::StringRef {
  if (mode == ColorMode::Ansi16 && shape != UnderlineShape::None) {
    return "4";
  }
  switch (shape) {
    case UnderlineShape::None:
      return "";
    case UnderlineShape::Single:
      return "4";
    case UnderlineShape::Double:
      return "4:2";
    case UnderlineShape::Curly:
      return "4:3";
    case UnderlineShape::Dotted:
      return "4:4";
    case UnderlineShape::Dashed:
      return "4:5";
  }
}

auto Style::NeedsResetFrom(const Style& from) const -> bool {
  auto drops = [](bool from_set, bool to_set) { return from_set && !to_set; };
  return drops(from.bold_, bold_) || drops(from.dim_, dim_) ||
         drops(from.italic_, italic_) || drops(from.reverse_, reverse_) ||
         drops(from.strikethrough_, strikethrough_) ||
         drops(from.underline(), underline()) ||
         drops(from.foreground_.is_set(), foreground_.is_set()) ||
         drops(from.background_.is_set(), background_.is_set()) ||
         drops(from.underline_color_.is_set(), underline_color_.is_set());
}

auto Style::AppendDiff(OutputBufferRef out, ColorMode mode,
                       const Style& from) const -> void {
  CARBON_CHECK(!NeedsResetFrom(from),
               "Cannot reach this style from `from` without a reset.");

  // Attributes combine into a single SGR sequence, in ascending code order.
  // Every write goes through `add` so the bound is checked in one place.
  std::array<llvm::StringRef, 6> params;
  int param_count = 0;
  auto add = [&](llvm::StringRef param) {
    CARBON_CHECK(param_count < static_cast<int>(params.size()),
                 "More SGR parameters than the {0} there is room for.",
                 params.size());
    params[param_count++] = param;
  };
  auto add_if = [&](bool from_set, bool to_set, llvm::StringRef param) {
    if (to_set && !from_set) {
      add(param);
    }
  };
  add_if(from.bold_, bold_, "1");
  add_if(from.dim_, dim_, "2");
  add_if(from.italic_, italic_, "3");
  llvm::StringRef underline_param = UnderlineSgrParam(underline_shape_, mode);
  if (underline_param != UnderlineSgrParam(from.underline_shape_, mode)) {
    // Turning an underline off needs a reset, which is the caller's to do, so
    // reaching here with nothing to select would emit an empty parameter.
    CARBON_CHECK(!underline_param.empty(),
                 "Removing an underline cannot be done with a diff.");
    add(underline_param);
  }
  add_if(from.reverse_, reverse_, "7");
  add_if(from.strikethrough_, strikethrough_, "9");

  if (param_count > 0) {
    out.Append("\x1b[", params[0]);
    for (int i = 1; i < param_count; ++i) {
      out.Append(";", params[i]);
    }
    out.Append("m");
  }

  if (foreground_.is_set() && foreground_ != from.foreground_) {
    foreground_.AppendEscape(out, mode, ColorTarget::Foreground);
  }
  if (background_.is_set() && background_ != from.background_) {
    background_.AppendEscape(out, mode, ColorTarget::Background);
  }
  if (underline_color_.is_set() && underline_color_ != from.underline_color_) {
    underline_color_.AppendEscape(out, mode, ColorTarget::Underline);
  }
}

auto Style::AppendColorTransitionTo(OutputBufferRef out, const Style& target,
                                    ColorMode mode) const -> void {
  CARBON_CHECK(mode != ColorMode::NoColor,
               "Color transitions are only reached when color is in use.");
  if (*this == target) {
    return;
  }

  if (target.NeedsResetFrom(*this)) {
    out.Append(ResetEscape);
    target.AppendDiff(out, mode, Style());
    return;
  }
  target.AppendDiff(out, mode, *this);
}

// Returns the name of `shape`, for printing a style.
static auto UnderlineShapeName(UnderlineShape shape) -> llvm::StringRef {
  switch (shape) {
    case UnderlineShape::None:
      return "None";
    case UnderlineShape::Single:
      return "Single";
    case UnderlineShape::Double:
      return "Double";
    case UnderlineShape::Curly:
      return "Curly";
    case UnderlineShape::Dotted:
      return "Dotted";
    case UnderlineShape::Dashed:
      return "Dashed";
  }
}

auto Style::Print(llvm::raw_ostream& out) const -> void {
  out << "Style(";
  llvm::ListSeparator sep;
  if (bold_) {
    out << sep << "bold";
  }
  if (dim_) {
    out << sep << "dim";
  }
  if (italic_) {
    out << sep << "italic";
  }
  if (reverse_) {
    out << sep << "reverse";
  }
  if (strikethrough_) {
    out << sep << "strikethrough";
  }
  if (underline()) {
    out << sep << "underline=" << UnderlineShapeName(underline_shape_);
  }
  if (foreground_.is_set()) {
    out << sep << "foreground=" << foreground_;
  }
  if (background_.is_set()) {
    out << sep << "background=" << background_;
  }
  if (underline_color_.is_set()) {
    out << sep << "underline_color=" << underline_color_;
  }
  out << ")";
}

}  // namespace Carbon::Terminal
