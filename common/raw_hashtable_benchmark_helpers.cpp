// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/raw_hashtable_benchmark_helpers.h"

#include <cstddef>
#include <forward_list>

namespace Carbon::RawHashtable {

// A local shuffle implementation built on Abseil to improve performance in
// debug builds.
template <typename T>
static auto Shuffle(llvm::MutableArrayRef<T> data, absl::BitGen& gen) {
  for (ssize_t i : llvm::seq<ssize_t>(0, data.size() - 1)) {
    ssize_t j = absl::Uniform<ssize_t>(gen, 0, data.size() - i);
    if (j != 0) {
      std::swap(data[i], data[i + j]);
    }
  }
}

constexpr ssize_t NumChars = 64;
static_assert(llvm::isPowerOf2_64(NumChars));

// For benchmarking, we use short strings in a fixed distribution with common
// characters. Real-world strings aren't uniform across ASCII or Unicode, etc.
// And for *micro*-benchmarking we want to focus on the map overhead with short,
// fast keys.
static auto MakeChars() -> llvm::OwningArrayRef<char> {
  llvm::OwningArrayRef<char> characters(NumChars);

  // Start with `-` and `_`, and then add `a` - `z`, `A` - `Z`, and `0` - `9`.
  characters[0] = '-';
  characters[1] = '_';
  ssize_t i = 2;
  for (auto range :
       {llvm::seq_inclusive('a', 'z'), llvm::seq_inclusive('A', 'Z'),
        llvm::seq_inclusive('0', '9')}) {
    for (char c : range) {
      characters[i] = c;
      ++i;
    }
  }
  CARBON_CHECK(i == NumChars,
               "Expected exactly {0} characters, got {1} instead!", NumChars,
               i);
  return characters;
}

constexpr ssize_t NumFourCharStrs = NumChars * NumChars * NumChars * NumChars;
static_assert(llvm::isPowerOf2_64(NumFourCharStrs));

// Compute every 4-character string in a shuffled array. This is a little memory
// intense -- 64 MiB -- but ends up being much cheaper by letting us reliably
// select a unique 4-character sequence to avoid collisions.
static auto MakeFourCharStrs(llvm::ArrayRef<char> characters, absl::BitGen& gen)
    -> llvm::OwningArrayRef<std::array<char, 4>> {
  constexpr ssize_t NumCharsMask = NumChars - 1;
  constexpr ssize_t NumCharsShift = llvm::CTLog2<NumChars>();
  llvm::OwningArrayRef<std::array<char, 4>> four_char_strs(NumFourCharStrs);
  for (auto [i, str] : llvm::enumerate(four_char_strs)) {
    str[0] = characters[i & NumCharsMask];
    i >>= NumCharsShift;
    str[1] = characters[i & NumCharsMask];
    i >>= NumCharsShift;
    str[2] = characters[i & NumCharsMask];
    i >>= NumCharsShift;
    CARBON_CHECK((i & ~NumCharsMask) == 0);
    str[3] = characters[i];
  }
  Shuffle(four_char_strs, gen);
  return four_char_strs;
}

constexpr ssize_t NumRandomChars = static_cast<ssize_t>(64) * 1024;

// Create a pool of random characters to sample from rather than computing this
// for every string which is very slow in debug builds. We also pad this pool
// with the max length so we can pull the full length from the end to simplify
// the logic when wrapping around the pool.
static auto MakeRandomChars(llvm::ArrayRef<char> characters, int max_length,
                            absl::BitGen& gen) -> llvm::OwningArrayRef<char> {
  llvm::OwningArrayRef<char> random_chars(NumRandomChars + max_length);
  for (char& c : random_chars) {
    c = characters[absl::Uniform<ssize_t>(gen, 0, NumChars)];
  }
  return random_chars;
}

// Make a small vector of pointers into a single allocation of raw strings. The
// allocated memory is expected to leak and must be transitively referenced by a
// global. Each string has `length` size (which must be >= 4), and there are
// `key_count` keys in the result. Each key is filled from the `random_chars`
// until the last 4 characters. The last four characters of each string will be
// taken sequentially from `four_char_strs` from some random start position to
// ensure no duplicate keys are produced.
static auto MakeRawStrKeys(ssize_t length, ssize_t key_count,
                           llvm::ArrayRef<std::array<char, 4>> four_char_strs,
                           llvm::ArrayRef<char> random_chars, absl::BitGen& gen)
    -> llvm::SmallVector<const char*> {
  llvm::SmallVector<const char*> raw_keys;
  CARBON_CHECK(length >= 4);
  ssize_t prefix_length = length - 4;

  // Select a random start for indexing our four character strings.
  ssize_t four_char_index = absl::Uniform<ssize_t>(gen, 0, NumFourCharStrs);

  // Select a random start for the prefix random characters.
  ssize_t random_chars_index = absl::Uniform<ssize_t>(gen, 0, NumRandomChars);

  // Do a single memory allocation for all the keys of this length to
  // avoid an excessive number of small and fragmented allocations. This
  // memory is intentionally leaked as the keys are global and will
  // themselves will point into it.
  char* key_text = new char[key_count * length];

  // Reserve all the key space since we know how many we'll need.
  raw_keys.reserve(key_count);
  for ([[gnu::unused]] ssize_t i : llvm::seq<ssize_t>(0, key_count)) {
    memcpy(key_text, random_chars.data() + random_chars_index, prefix_length);
    random_chars_index += prefix_length;
    random_chars_index &= NumRandomChars - 1;
    // Set the last four characters with this entry in the shuffled
    // sequence.
    memcpy(key_text + prefix_length, four_char_strs[four_char_index].data(), 4);
    // Step through the shuffled sequence. We start at a random position,
    // so we need to wrap around the end.
    ++four_char_index;
    four_char_index &= NumFourCharStrs - 1;

    // And finally save the start pointer as one of our raw keys.
    raw_keys.push_back(key_text);
    key_text += length;
  }
  return raw_keys;
}

// Build up a large collection of random and unique string keys. This is
// actually a relatively expensive operation due to needing to build all the
// random string text. As a consequence, the initializer of this global is
// somewhat performance tuned to ensure benchmarks don't take an excessive
// amount of time to run or use an excessive amount of memory.
static absl::NoDestructor<llvm::OwningArrayRef<llvm::StringRef>> raw_str_keys{
    [] {
      llvm::OwningArrayRef<llvm::StringRef> keys(MaxNumKeys);
      absl::BitGen gen;

      std::array length_buckets = {
          4, 4, 4, 4, 5, 5, 5, 5, 7, 7, 10, 10, 15, 25, 40, 80,
      };
      static_assert((MaxNumKeys % length_buckets.size()) == 0);
      CARBON_CHECK(llvm::is_sorted(length_buckets));

      // For each distinct length bucket, we build a vector of raw keys.
      std::forward_list<llvm::SmallVector<const char*>> raw_keys_storage;
      // And a parallel array to the length buckets with the raw keys of that
      // length.
      std::array<llvm::SmallVector<const char*>*, length_buckets.size()>
          raw_keys_buckets;

      llvm::OwningArrayRef<char> characters = MakeChars();
      llvm::OwningArrayRef<std::array<char, 4>> four_char_strs =
          MakeFourCharStrs(characters, gen);
      llvm::OwningArrayRef<char> random_chars = MakeRandomChars(
          characters, /*max_length=*/length_buckets.back(), gen);

      ssize_t prev_length = -1;
      for (auto [length_index, length] : llvm::enumerate(length_buckets)) {
        // We can detect repetitions in length as they are sorted.
        if (length == prev_length) {
          raw_keys_buckets[length_index] = raw_keys_buckets[length_index - 1];
          continue;
        }
        prev_length = length;

        // We want to compute all the keys of this length that we'll need.
        ssize_t key_count = (MaxNumKeys / length_buckets.size()) *
                            llvm::count(length_buckets, length);

        raw_keys_buckets[length_index] =
            &raw_keys_storage.emplace_front(MakeRawStrKeys(
                length, key_count, four_char_strs, random_chars, gen));
      }

      // Now build the actual key array from our intermediate storage by
      // round-robin extracting from the length buckets.
      for (auto [index, key] : llvm::enumerate(keys)) {
        ssize_t bucket = index % length_buckets.size();
        ssize_t length = length_buckets[bucket];
        // We pop a raw key from the list of them associated with this bucket.
        const char* raw_key = raw_keys_buckets[bucket]->pop_back_val();
        // And build our key from that.
        key = llvm::StringRef(raw_key, length);
      }
      // Check that in fact we popped every raw key into our main keys.
      for (const auto& raw_keys : raw_keys_storage) {
        CARBON_CHECK(raw_keys.empty());
      }
      return keys;
    }()};

static absl::NoDestructor<llvm::OwningArrayRef<int*>> raw_ptr_keys{[] {
  llvm::OwningArrayRef<int*> keys(MaxNumKeys);
  for (auto [index, key] : llvm::enumerate(keys)) {
    // We leak these pointers -- this is a static initializer executed once.
    key = new int(static_cast<int>(index));
  }
  return keys;
}()};

static absl::NoDestructor<llvm::OwningArrayRef<int>> raw_int_keys{[] {
  llvm::OwningArrayRef<int> keys(MaxNumKeys);
  for (auto [index, key] : llvm::enumerate(keys)) {
    key = index + 1;
  }
  return keys;
}()};

namespace {

// Allow generically dispatching over the specific key types we build and
// support.
template <typename T>
auto GetRawKeys() -> llvm::ArrayRef<T> {
  if constexpr (std::is_same_v<T, llvm::StringRef>) {
    return *raw_str_keys;
  } else if constexpr (std::is_pointer_v<T>) {
    return *raw_ptr_keys;
  } else {
    return *raw_int_keys;
  }
}

template <typename T>
static absl::NoDestructor<
    std::map<std::pair<ssize_t, ssize_t>, llvm::OwningArrayRef<T>>>
    lookup_keys_storage;

// Given a particular table keys size and lookup keys size, provide an array ref
// to a shuffled set of lookup keys.
//
// Because different table sizes pull from different sub-ranges of our raw keys,
// we need to compute a distinct set of random keys in the table to use for
// lookups depending on the table size. And we also want to have an even
// distribution of key *sizes* throughout the lookup keys, and so we can't
// compute a single lookup keys array of the maximum size. Instead we need to
// compute a distinct special set of lookup keys for each pair of table and
// lookup size, and then shuffle that specific set into a random sequence that
// is returned. This function memoizes this sequence for each pair of sizes.
template <typename T>
auto GetShuffledLookupKeys(ssize_t table_keys_size, ssize_t lookup_keys_size)
    -> llvm::ArrayRef<T> {
  // The raw keys aren't shuffled and round-robin through the sizes. We want to
  // keep the total size of lookup keys used exactly the same across runs. So
  // for a given size we always take the leading sequence from the raw keys for
  // that size, duplicating as needed to get the desired lookup sequence size,
  // and then shuffle the keys in that sequence to end up with a random sequence
  // of keys. We store each of these shuffled sequences in a map to avoid
  // repeatedly computing them.
  llvm::OwningArrayRef<T>& lookup_keys =
      (*lookup_keys_storage<T>)[{table_keys_size, lookup_keys_size}];
  if (lookup_keys.empty()) {
    lookup_keys = llvm::OwningArrayRef<T>(lookup_keys_size);
    auto raw_keys = GetRawKeys<T>();
    for (auto [index, key] : llvm::enumerate(lookup_keys)) {
      key = raw_keys[index % table_keys_size];
    }
    absl::BitGen gen;
    Shuffle(lookup_keys, gen);
  }
  CARBON_CHECK(static_cast<ssize_t>(lookup_keys.size()) == lookup_keys_size);

  return lookup_keys;
}

}  // namespace

template <typename T>
auto GetKeysAndMissKeys(ssize_t table_keys_size)
    -> std::pair<llvm::ArrayRef<T>, llvm::ArrayRef<T>> {
  CARBON_CHECK(table_keys_size <= MaxNumKeys);
  // The raw keys aren't shuffled and round-robin through the sizes. Take the
  // tail of this sequence and shuffle it to form a random set of miss keys with
  // a consistent total size.
  static absl::NoDestructor<llvm::OwningArrayRef<T>> miss_keys{[] {
    llvm::OwningArrayRef<T> keys;
    keys = GetRawKeys<T>().take_back(NumOtherKeys);
    CARBON_CHECK(keys.size() == NumOtherKeys);
    absl::BitGen gen;
    Shuffle(keys, gen);
    return keys;
  }()};

  return {GetRawKeys<T>().slice(0, table_keys_size), *miss_keys};
}
template auto GetKeysAndMissKeys<int>(ssize_t size)
    -> std::pair<llvm::ArrayRef<int>, llvm::ArrayRef<int>>;
template auto GetKeysAndMissKeys<int*>(ssize_t size)
    -> std::pair<llvm::ArrayRef<int*>, llvm::ArrayRef<int*>>;
template auto GetKeysAndMissKeys<llvm::StringRef>(ssize_t size)
    -> std::pair<llvm::ArrayRef<llvm::StringRef>,
                 llvm::ArrayRef<llvm::StringRef>>;

template <typename T>
auto GetKeysAndHitKeys(ssize_t table_keys_size, ssize_t lookup_keys_size)
    -> std::pair<llvm::ArrayRef<T>, llvm::ArrayRef<T>> {
  CARBON_CHECK(table_keys_size <= MaxNumKeys);
  CARBON_CHECK(lookup_keys_size <= MaxNumKeys);
  return {GetRawKeys<T>().slice(0, table_keys_size),
          GetShuffledLookupKeys<T>(table_keys_size, lookup_keys_size)};
}
template auto GetKeysAndHitKeys<int>(ssize_t size, ssize_t lookup_keys_size)
    -> std::pair<llvm::ArrayRef<int>, llvm::ArrayRef<int>>;
template auto GetKeysAndHitKeys<int*>(ssize_t size, ssize_t lookup_keys_size)
    -> std::pair<llvm::ArrayRef<int*>, llvm::ArrayRef<int*>>;
template auto GetKeysAndHitKeys<llvm::StringRef>(ssize_t size,
                                                 ssize_t lookup_keys_size)
    -> std::pair<llvm::ArrayRef<llvm::StringRef>,
                 llvm::ArrayRef<llvm::StringRef>>;

template <typename T>
auto DumpHashStatistics(llvm::ArrayRef<T> keys) -> void {
  if (keys.size() < GroupSize) {
    return;
  }

  // The hash table load factor is 7/8ths, so we want to add 1/7th of our
  // current size, subtract one, and pick the next power of two to get the power
  // of two where 7/8ths is greater than or equal to the incoming key size.
  ssize_t expected_size =
      llvm::NextPowerOf2(keys.size() + (keys.size() / 7) - 1);

  constexpr int GroupShift = llvm::CTLog2<GroupSize>();

  size_t mask = ComputeProbeMaskFromSize(expected_size);
  uint64_t salt = ComputeSeed();
  auto get_hash_index = [mask, salt](auto x) -> ssize_t {
    auto [hash_index, _] = HashValue(x, salt).template ExtractIndexAndTag<7>();
    return (hash_index & mask) >> GroupShift;
  };

  std::vector<std::vector<int>> grouped_key_indices(expected_size >>
                                                    GroupShift);
  for (auto [i, k] : llvm::enumerate(keys)) {
    ssize_t hash_index = get_hash_index(k);
    CARBON_CHECK(hash_index < (expected_size >> GroupShift), "{0}", hash_index);
    grouped_key_indices[hash_index].push_back(i);
  }
  ssize_t max_group_index =
      llvm::max_element(grouped_key_indices,
                        [](const auto& lhs, const auto& rhs) {
                          return lhs.size() < rhs.size();
                        }) -
      grouped_key_indices.begin();

  // If the max number of collisions on the index is less than or equal to the
  // group size, there shouldn't be any necessary probing (outside of deletion)
  // and so this isn't interesting, skip printing.
  if (grouped_key_indices[max_group_index].size() <= GroupSize) {
    return;
  }

  llvm::errs() << "keys: " << keys.size()
               << "  groups: " << grouped_key_indices.size() << "\n"
               << "max group index: " << llvm::formatv("{0x8}", max_group_index)
               << "  collisions: "
               << grouped_key_indices[max_group_index].size() << "\n";

  for (auto i : llvm::ArrayRef(grouped_key_indices[max_group_index])
                    .take_front(2 * GroupSize)) {
    auto k = keys[i];
    auto hash = static_cast<uint64_t>(HashValue(k, salt));
    llvm::errs() << "  key: " << k
                 << "  salt: " << llvm::formatv("{0:x16}", salt)
                 << "  hash: " << llvm::formatv("{0:x16}", hash) << "\n";
  }
}
template auto DumpHashStatistics(llvm::ArrayRef<int> keys) -> void;
template auto DumpHashStatistics(llvm::ArrayRef<int*> keys) -> void;
template auto DumpHashStatistics(llvm::ArrayRef<llvm::StringRef> keys) -> void;

}  // namespace Carbon::RawHashtable
