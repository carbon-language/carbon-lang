// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

namespace a {} // original
namespace a {} // ext
inline namespace b {} // inline original
inline namespace b {} // inline ext
inline namespace {} // inline unnamed
