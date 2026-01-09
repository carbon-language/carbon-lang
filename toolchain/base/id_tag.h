// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_ID_TAG_H_
#define CARBON_TOOLCHAIN_BASE_ID_TAG_H_

#include <stdint.h>

#include <limits>

#include "common/check.h"
#include "llvm/Support/MathExtras.h"

namespace Carbon {

// A tagged Id. It is used to add a tag into the unused bits of the id, in order
// to verify ids are used in the correct context. The tag should look like an
// Id, which is a non-negative number.
struct IdTag {
  IdTag() = default;

  explicit IdTag(int32_t tag, int32_t initial_reserved_ids)
      :  // Shift down by 1 to get out of the high bit to avoid using any
         // negative ids, since they have special uses. Shift down by another 1
         // to free up the second highest bit for a marker to indicate whether
         // the index is tagged (& needs to be untagged) or not. Add one to the
         // index so it's not zero-based, to make it a bit less likely this
         // doesn't collide with anything else (though with the
         // second-highest-bit-tagging this might not be needed).
        id_tag_(llvm::reverseBits((((tag + 1) << 1) | 1) << 1)),
        initial_reserved_ids_(initial_reserved_ids) {
    CARBON_CHECK(
        tag != -1,
        "IdTag should be default constructed if no tagging id is available.");
  }

  auto Apply(int32_t index) const -> int32_t {
    CARBON_DCHECK(index >= 0, "{0}", index);
    if (index < initial_reserved_ids_) {
      return index;
    }
    // TODO: Assert that id_tag_ doesn't have the second highest bit set.
    auto tagged_index = index ^ id_tag_;
    CARBON_DCHECK(tagged_index >= 0, "{0}", tagged_index);
    return tagged_index;
  }

  auto Remove(int32_t tagged_index) const -> int32_t {
    CARBON_DCHECK(tagged_index >= 0, "{0}", tagged_index);
    if (!HasTag(tagged_index)) {
      CARBON_DCHECK(tagged_index < initial_reserved_ids_,
                    "This untagged index is outside the initial reserved ids "
                    "and should have been tagged.");
      return tagged_index;
    }
    auto index = tagged_index ^ id_tag_;
    CARBON_DCHECK(index >= initial_reserved_ids_,
                  "When removing tagging bits, found an index that "
                  "shouldn't've been tagged in the first place.");
    return index;
  }

  // Gets the value unique to this IdTag instance that is added to indices in
  // Apply, and removed in Remove.
  auto GetContainerTag() const -> int32_t {
    return (llvm::reverseBits(id_tag_) >> 2) - 1;
  }

  // Returns whether `tagged_index` has an IdTag applied to it, from this IdTag
  // instance or any other one.
  static auto HasTag(int32_t tagged_index) -> bool {
    return (llvm::reverseBits(2) & tagged_index) != 0;
  }

  template <class TagT>
  struct TagAndIndex {
    int32_t tag;
    int32_t index;
  };

  template <typename TagT>
  static auto DecomposeWithBestEffort(int32_t tagged_index)
      -> TagAndIndex<TagT> {
    if (tagged_index < 0) {
      // TODO: This should return TagT::None, but we need a fallback TagT other
      // than `int32_t`.
      return {TagT{-1}, tagged_index};
    }
    if (!HasTag(tagged_index)) {
      // TODO: This should return TagT::None, but we need a fallback TagT other
      // than `int32_t`.
      return {TagT{-1}, tagged_index};
    }
    int length = 0;
    int location = 0;
    for (int i = 0; i != 32; ++i) {
      int current_run = 0;
      int location_of_current_run = i;
      while (i != 32 && (tagged_index & (1 << i)) == 0) {
        ++current_run;
        ++i;
      }
      if (current_run != 0) {
        --i;
      }
      if (current_run > length) {
        length = current_run;
        location = location_of_current_run;
      }
    }
    if (length < 8) {
      // TODO: This should return TagT::None, but we need a fallback TagT other
      // than `int32_t`.
      return {TagT{-1}, tagged_index};
    }
    auto index_mask = llvm::maskTrailingOnes<uint32_t>(location);
    auto tag = (llvm::reverseBits(tagged_index & ~index_mask) >> 2) - 1;
    auto index = tagged_index & index_mask;
    return {.tag = TagT{static_cast<int32_t>(tag)},
            .index = static_cast<int32_t>(index)};
  }

 private:
  int32_t id_tag_ = 0;
  int32_t initial_reserved_ids_ = std::numeric_limits<int32_t>::max();
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_ID_TAG_H_
