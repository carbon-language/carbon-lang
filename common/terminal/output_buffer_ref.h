// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_TERMINAL_OUTPUT_BUFFER_REF_H_
#define CARBON_COMMON_TERMINAL_OUTPUT_BUFFER_REF_H_

#include <array>
#include <concepts>
#include <cstdint>
#include <cstring>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Terminal {

// A reference to the buffer a terminal rendering is assembled into.
//
// This owns nothing. It refers to a buffer the caller holds, which must outlive
// it, and converts implicitly from one so that rendering goes into storage the
// caller already has.
//
// Rendering assembles bytes here rather than streaming them: a stream call per
// literal and per number costs measurably more than handing over a finished
// sequence, and code that wants a stream prints the buffer once it is complete.
//
// Appending is shaped around what terminal output is made of, which is a great
// many short escape sequences, each a handful of literal bytes around a number
// that never exceeds 255. Taking a whole sequence at a time grows the buffer
// once per sequence rather than once per byte, and that difference is much of
// what rendering costs.
class OutputBufferRef {
 public:
  // Implicit, so that call sites pass the buffer they already hold rather than
  // naming this type.
  //
  // NOLINTNEXTLINE(google-explicit-constructor)
  OutputBufferRef(llvm::SmallVectorImpl<char>& bytes) : bytes_(&bytes) {}

  // Appends `pieces`, each of which is either text, appended as it is, or a
  // `uint8_t`, appended in decimal.
  //
  // No other type is accepted, so the two can never be taken for each other,
  // and nothing needs one: the literal bytes of an escape sequence are always
  // text, and every number one carries is a channel value, a palette index, or
  // an SGR code, none of which exceed 255.
  //
  // No piece may point into the buffer, which appending can reallocate.
  template <typename... PieceT>
  auto Append(const PieceT&... pieces) -> void {
    if constexpr (sizeof...(pieces) == 1) {
      // A lone piece has nothing to assemble, and the buffer's own append is
      // already the single growth and single copy this is after.
      (AppendPiece(pieces), ...);
    } else {
      // Growing to the bound before writing keeps how far the buffer grows
      // independent of the piece values, so computing one can't hold that up.
      // Only the trim afterwards depends on how many digits a number took.
      size_t begin = bytes_->size();
      bytes_->resize_for_overwrite(begin + (AppendedSize(pieces) + ... + 0));
      char* data = bytes_->data();
      char* cursor = data + begin;
      ((cursor = WritePiece(cursor, pieces)), ...);
      bytes_->truncate(cursor - data);
    }
  }

 private:
  // The room a number needs: its three digits, plus one more because it is
  // written as a single four-byte store whose last byte is discarded.
  static constexpr size_t NumberBytes = 4;

  // The decimal text of a number, and how many digits it took. The digits are
  // at the front and the length in the byte after them, so a whole entry is one
  // store and the length says how far of it to keep.
  struct NumberText {
    std::array<char, NumberBytes - 1> digits;
    uint8_t length;
  };
  static_assert(sizeof(NumberText) == NumberBytes,
                "A number is written by storing a whole entry at once.");

  // The text of every value a number piece can hold. A kilobyte of table, in
  // exchange for a lookup where computing the digits would branch on the value
  // three times.
  static constexpr std::array<NumberText, 256> NumberTexts = [] {
    std::array<NumberText, 256> texts = {};
    for (int value = 0; value < 256; ++value) {
      NumberText& text = texts[value];
      text.length = 1 + (value >= 10) + (value >= 100);
      int rest = value;
      for (int digit = text.length; digit > 0; --digit) {
        text.digits[digit - 1] = static_cast<char>('0' + rest % 10);
        rest /= 10;
      }
    }
    return texts;
  }();

  // Returns the most bytes a piece can append. A number contributes the bound
  // above rather than the digits it will take, so the bound for a sequence
  // doesn't depend on any of the values in it.
  template <size_t N>
  static constexpr auto AppendedSize(const char (& /*piece*/)[N]) -> size_t {
    return N - 1;
  }
  static constexpr auto AppendedSize(llvm::StringRef piece) -> size_t {
    return piece.size();
  }
  template <std::same_as<uint8_t> T>
  static constexpr auto AppendedSize(T /*piece*/) -> size_t {
    return NumberBytes;
  }

  // Writes a piece at `out` and returns the position past it. There must be
  // `AppendedSize(piece)` bytes of room, as nothing here checks.
  template <size_t N>
  static auto WritePiece(char* out, const char (&piece)[N]) -> char* {
    std::memcpy(out, piece, N - 1);
    return out + N - 1;
  }
  static auto WritePiece(char* out, llvm::StringRef piece) -> char* {
    // An empty `StringRef` may hold a null pointer, which `memcpy` doesn't
    // accept even for an empty copy.
    if (!piece.empty()) {
      std::memcpy(out, piece.data(), piece.size());
    }
    return out + piece.size();
  }
  template <std::same_as<uint8_t> T>
  static auto WritePiece(char* out, T piece) -> char* {
    // One load and one store, with no branch on the value. Escape sequences
    // carry color channels and palette indices, which are spread across the
    // whole range, so a branch per digit is one the processor can't predict,
    // and there are four numbers in a truecolor escape. The store always covers
    // four bytes, which is why a number reserves that many, and the cursor
    // advances only over the digits that count.
    const NumberText& text = NumberTexts[piece];
    std::memcpy(out, &text, sizeof(text));
    return out + text.length;
  }

  // Appends a piece on its own, growing the buffer to fit it.
  template <size_t N>
  auto AppendPiece(const char (&piece)[N]) -> void {
    bytes_->append(piece, piece + N - 1);
  }
  auto AppendPiece(llvm::StringRef piece) -> void {
    bytes_->append(piece.begin(), piece.end());
  }
  template <std::same_as<uint8_t> T>
  auto AppendPiece(T piece) -> void {
    std::array<char, NumberBytes> digits;
    bytes_->append(digits.data(), WritePiece(digits.data(), piece));
  }

  llvm::SmallVectorImpl<char>* bytes_;
};

}  // namespace Carbon::Terminal

#endif  // CARBON_COMMON_TERMINAL_OUTPUT_BUFFER_REF_H_
