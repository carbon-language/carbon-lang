// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_ID_TAG_H_
#define CARBON_TOOLCHAIN_BASE_ID_TAG_H_

#include <stdint.h>

#include <limits>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/Support/MathExtras.h"

namespace Carbon {

// A sentinel type to construct an IdTag without tagging.
struct Untagged : Printable<Untagged> {
  auto Print(llvm::raw_ostream& out) const -> void { out << "<untagged>"; }
};

// A wrapper type used as the template argument to IdTag, in order to mark the
// tag type as such.
template <typename TagIdT>
struct Tag {};

template <typename T>
struct GetTagIdType {
  static_assert(false, "IdTag with TagT that is neither Untagged nor Tag");
};
template <>
struct GetTagIdType<Untagged> {
  using TagIdType = Untagged;
};
template <typename TagIdT>
struct GetTagIdType<Tag<TagIdT>> {
  using TagIdType = TagIdT;
};

// Tests if an `IdTag` type is untagged.
template <typename IdTagT>
concept IdTagIsUntagged = std::same_as<typename IdTagT::TagIdType, Untagged>;

// A tagged Id. It is used to add a tag into the unused bits of the id, in order
// to verify ids are used in the correct context. The tag type must be `Tag` or
// `Untagged`.
template <typename IdT, typename TagT>
struct IdTag {
  using IdType = IdT;
  using TagIdType = GetTagIdType<TagT>::TagIdType;

  IdTag()
    requires(IdTagIsUntagged<IdTag>)
  = default;

  IdTag(TagIdType tag, int32_t initial_reserved_ids)
    requires(!IdTagIsUntagged<IdTag>)
      :  // Shift down by 1 to get out of the high bit to avoid using any
         // negative ids, since they have special uses. Shift down by another 1
         // to free up the second highest bit for a marker to indicate whether
         // the index is tagged (& needs to be untagged) or not. Add one to the
         // index so it's not zero-based, to make it a bit less likely this
         // doesn't collide with anything else (though with the
         // second-highest-bit-tagging this might not be needed).
        tag_(llvm::reverseBits((((tag.index + 1) << 1) | 1) << 1)),
        initial_reserved_ids_(initial_reserved_ids) {}

  auto Apply(int32_t index) const -> IdT {
    CARBON_DCHECK(index >= 0, "{0}", index);
    if (index < initial_reserved_ids_) {
      return IdT(index);
    }
    // TODO: Assert that tag_ doesn't have the second highest bit set.
    auto tagged_index = index ^ tag_;
    CARBON_DCHECK(tagged_index >= 0, "{0}", tagged_index);
    return IdT(tagged_index);
  }

  auto Remove(IdT id) const -> int32_t {
    CARBON_DCHECK(id.index >= 0, "{0}", id);
    if (!HasTag(id.index)) {
      CARBON_DCHECK(id.index < initial_reserved_ids_,
                    "This untagged index is outside the initial reserved ids "
                    "and should have been tagged.");
      return id.index;
    }
    auto untagged_index = id.index ^ tag_;
    CARBON_DCHECK(untagged_index >= initial_reserved_ids_,
                  "When removing tagging bits, found an index that "
                  "shouldn't've been tagged in the first place.");
    return untagged_index;
  }

  // Gets the value unique to this IdTag instance that is added to indices in
  // Apply, and removed in Remove.
  auto GetContainerTag() const -> TagIdType {
    if constexpr (IdTagIsUntagged<IdTag>) {
      return TagIdType();
    } else {
      return TagIdType((llvm::reverseBits(tag_) >> 2) - 1);
    }
  }

  // Returns whether `tagged_index` has an IdTag applied to it, from this IdTag
  // instance or any other one.
  static auto HasTag(int32_t tagged_index) -> bool {
    return (llvm::reverseBits(2) & tagged_index) != 0;
  }

  struct TagAndIndex {
    TagIdType tag;
    int32_t index;
  };

  static auto DecomposeWithBestEffort(IdT id) -> TagAndIndex {
    if constexpr (IdTagIsUntagged<IdTag>) {
      return {TagIdType(), id.index};
    } else {
      if (!id.has_value()) {
        return {TagIdType::None, id.index};
      }
      if (!HasTag(id.index)) {
        return {TagIdType::None, id.index};
      }
      int length = 0;
      int location = 0;
      for (int i = 0; i != 32; ++i) {
        int current_run = 0;
        int location_of_current_run = i;
        while (i != 32 && (id.index & (1 << i)) == 0) {
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
        return {TagIdType::None, id.index};
      }
      auto index_mask = llvm::maskTrailingOnes<uint32_t>(location);
      auto tag = (llvm::reverseBits(id.index & ~index_mask) >> 2) - 1;
      auto index = id.index & index_mask;
      return {.tag = TagIdType(static_cast<int32_t>(tag)),
              .index = static_cast<int32_t>(index)};
    }
  }

  // Converts an IdTag to be used for a different ID type. This is only valid
  // when the id indices are interchangeable, as they will have the same tag and
  // the same reserved ids.
  template <typename OtherIdT>
    requires(!IdTagIsUntagged<IdTag>)
  auto ToEquivalentIdType() -> IdTag<OtherIdT, Tag<TagIdType>> {
    return {GetContainerTag(), initial_reserved_ids_};
  }

 private:
  int32_t tag_ = 0;
  int32_t initial_reserved_ids_ = std::numeric_limits<int32_t>::max();
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_ID_TAG_H_
