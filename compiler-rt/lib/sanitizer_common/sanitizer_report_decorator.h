//===-- sanitizer_report_decorator.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tags to decorate the sanitizer reports.
// Currently supported tags:
//   * None.
//   * ANSI color sequences.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_ALLOCATOR_H
#define SANITIZER_ALLOCATOR_H

namespace __sanitizer {
class AnsiColorDecorator {
 public:
  explicit AnsiColorDecorator(bool use_ansi_colors) : ansi_(use_ansi_colors) { }
  const char *Black()        { return ansi_ ? "\033[1m\033[30m" : ""; }
  const char *Red()          { return ansi_ ? "\033[1m\033[31m" : ""; }
  const char *Green()        { return ansi_ ? "\033[1m\033[32m" : ""; }
  const char *Yellow()       { return ansi_ ? "\033[1m\033[33m" : ""; }
  const char *Blue()         { return ansi_ ? "\033[1m\033[34m" : ""; }
  const char *Magenta()      { return ansi_ ? "\033[1m\033[35m" : ""; }
  const char *Cyan()         { return ansi_ ? "\033[1m\033[36m" : ""; }
  const char *White()        { return ansi_ ? "\033[1m\033[37m" : ""; }
  const char *Default()      { return ansi_ ? "\033[1m\033[0m"  : ""; }
 private:
  bool ansi_;
};
}  // namespace __sanitizer
#endif  // SANITIZER_ALLOCATOR_H
