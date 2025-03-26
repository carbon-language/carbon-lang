// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/source_gen.h"

#include <numeric>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Testing {

auto SourceGen::Global() -> SourceGen& {
  static SourceGen global_gen;
  return global_gen;
}

SourceGen::SourceGen(Language language) : language_(language) {}

// Heuristic numbers used in synthesizing various identifier sequences.
constexpr static int MinClassNameLength = 5;
constexpr static int MinMemberNameLength = 4;

// The shuffled state used to generate some number of classes.
//
// This state encodes everything used to generate class definitions. The state
// will be consumed until empty.
//
// Detailed comments for out-of-line methods are on their definitions.
class SourceGen::ClassGenState {
 public:
  ClassGenState(SourceGen& gen, int num_classes,
                const ClassParams& class_params,
                const TypeUseParams& type_use_params);

  auto public_function_param_counts() -> llvm::SmallVectorImpl<int>& {
    return public_function_param_counts_;
  }
  auto public_method_param_counts() -> llvm::SmallVectorImpl<int>& {
    return public_method_param_counts_;
  }
  auto private_function_param_counts() -> llvm::SmallVectorImpl<int>& {
    return private_function_param_counts_;
  }
  auto private_method_param_counts() -> llvm::SmallVectorImpl<int>& {
    return private_method_param_counts_;
  }

  auto class_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return class_names_;
  }
  auto member_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return member_names_;
  }
  auto param_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return param_names_;
  }

  auto type_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return type_names_;
  }

  auto AddValidTypeName(llvm::StringRef type_name) -> void {
    valid_type_names_.Insert(type_name);
  }

  auto GetValidTypeName() -> llvm::StringRef;

 private:
  auto BuildClassAndTypeNames(SourceGen& gen, int num_classes, int num_types,
                              const TypeUseParams& type_use_params) -> void;

  llvm::SmallVector<int> public_function_param_counts_;
  llvm::SmallVector<int> public_method_param_counts_;
  llvm::SmallVector<int> private_function_param_counts_;
  llvm::SmallVector<int> private_method_param_counts_;

  llvm::SmallVector<llvm::StringRef> class_names_;
  llvm::SmallVector<llvm::StringRef> member_names_;
  llvm::SmallVector<llvm::StringRef> param_names_;

  llvm::SmallVector<llvm::StringRef> type_names_;
  Set<llvm::StringRef> valid_type_names_;
  int last_type_name_index_ = 0;
};

// A helper to sum elements of a range.
template <typename T>
static auto Sum(const T& range) -> int {
  return std::accumulate(range.begin(), range.end(), 0);
}

// Given a number of class definitions and the params with which to generate
// them, builds the state that will be used while generating that many classes.
//
// We build the state first and across all the class definitions that will be
// generated so that we can distribute random components across all the
// definitions.
SourceGen::ClassGenState::ClassGenState(SourceGen& gen, int num_classes,
                                        const ClassParams& class_params,
                                        const TypeUseParams& type_use_params) {
  public_function_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.public_function_decls, 0,
                          class_params.public_function_decl_params.max_params);
  public_method_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.public_method_decls, 0,
                          class_params.public_method_decl_params.max_params);
  private_function_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.private_function_decls, 0,
                          class_params.private_function_decl_params.max_params);
  private_method_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.private_method_decls, 0,
                          class_params.private_method_decl_params.max_params);

  int num_members =
      num_classes *
      (class_params.public_function_decls + class_params.public_method_decls +
       class_params.private_function_decls + class_params.private_method_decls +
       class_params.private_field_decls);
  member_names_ = gen.GetShuffledIdentifiers(
      num_members, /*min_length=*/MinMemberNameLength);
  int num_params =
      Sum(public_function_param_counts_) + Sum(public_method_param_counts_) +
      Sum(private_function_param_counts_) + Sum(private_method_param_counts_);
  param_names_ = gen.GetShuffledIdentifiers(num_params);

  BuildClassAndTypeNames(gen, num_classes, num_members + num_params,
                         type_use_params);
}

auto SourceGen::ClassGenState::GetValidTypeName() -> llvm::StringRef {
  // Check that we don't completely wrap the type names by tracking where we
  // started.
  int initial_last_type_name_index = last_type_name_index_;

  // Now search the type names, starting from the last used index, to find the
  // first valid name.
  for (;;) {
    if (last_type_name_index_ == 0) {
      last_type_name_index_ = type_names_.size();
    }
    --last_type_name_index_;
    llvm::StringRef& type_name = type_names_[last_type_name_index_];
    if (valid_type_names_.Contains(type_name)) {
      // Found a valid type name, swap it with the back and pop that off.
      std::swap(type_names_.back(), type_name);
      return type_names_.pop_back_val();
    }

    CARBON_CHECK(last_type_name_index_ != initial_last_type_name_index,
                 "Failed to find a valid type name with {0} candidates, an "
                 "initial index of {1}, and with {2} classes left to emit!",
                 type_names_.size(), initial_last_type_name_index,
                 class_names_.size());
  }
}

// Build both the class names this file will declare and a list of type
// references to use throughout those classes.
//
// We combine a list of fixed types in the `type_use_params` with the list of
// class names that will be defined to form the spelling of all the referenced
// types. The `type_use_params` provides weights for each fixed type as well as
// an overall weight for referencing class names that are being declared. We
// build a set of type references so that its histogram will roughly match these
// weights.
//
// For each of the fixed types, `type_use_params` provides a spelling for both
// Carbon and C++.
//
// We distribute our references to declared class names evenly to the extent
// possible.
//
// Before all the references are formed, the class names are kept their original
// unshuffled order. This ensures that any uneven sampling of names is done
// deterministically. At the end, we randomly shuffle the sequences of both the
// declared class names and type references to provide an unpredictable order in
// the generated output.
auto SourceGen::ClassGenState::BuildClassAndTypeNames(
    SourceGen& gen, int num_classes, int num_types,
    const TypeUseParams& type_use_params) -> void {
  // Initially get the sequence of class names without shuffling so we can
  // compute our type name pool from them prior to any shuffling.
  class_names_ =
      gen.GetUniqueIdentifiers(num_classes, /*min_length=*/MinClassNameLength);

  type_names_.reserve(num_types);

  // Compute the sum of weights and pre-process the fixed types.
  int type_weight_sum = type_use_params.declared_types_weight;
  for (const auto& fixed_type_weight : type_use_params.fixed_type_weights) {
    type_weight_sum += fixed_type_weight.weight;
    // Add all the fixed type spellings as immediately valid.
    valid_type_names_.Insert(gen.IsCpp() ? fixed_type_weight.cpp_spelling
                                         : fixed_type_weight.carbon_spelling);
  }

  // Compute the number of declared types used. We expect to have a decent
  // number of repeated names, so we repeatedly append the entire sequence of
  // class names until there is some remainder of names needed.
  int num_declared_types =
      num_types * type_use_params.declared_types_weight / type_weight_sum;
  for ([[maybe_unused]] auto _ : llvm::seq(num_declared_types / num_classes)) {
    llvm::append_range(type_names_, class_names_);
  }
  // Now append the remainder number of class names. This is where the class
  // names being un-shuffled is essential. We're going to have one extra
  // reference to some fraction of the class names and we want that to be a
  // stable subset.
  type_names_.append(class_names_.begin(),
                     class_names_.begin() + (num_declared_types % num_classes));
  CARBON_CHECK(static_cast<int>(type_names_.size()) == num_declared_types);

  // Use each fixed type weight to append the expected number of copies of that
  // type. This isn't exact however, and is designed to stop short.
  for (const auto& fixed_type_weight : type_use_params.fixed_type_weights) {
    int num_fixed_type = num_types * fixed_type_weight.weight / type_weight_sum;
    type_names_.append(num_fixed_type, gen.IsCpp()
                                           ? fixed_type_weight.cpp_spelling
                                           : fixed_type_weight.carbon_spelling);
  }

  // If we need a tail of types to hit the exact number, simply round-robin
  // through the fixed types without any weighting. With reasonably large
  // numbers of types this won't distort the distribution in an interesting way
  // and is simpler than trying to scale the distribution down.
  while (static_cast<int>(type_names_.size()) < num_types) {
    for (const auto& fixed_type_weight :
         llvm::ArrayRef(type_use_params.fixed_type_weights)
             .take_front(num_types - type_names_.size())) {
      type_names_.push_back(gen.IsCpp() ? fixed_type_weight.cpp_spelling
                                        : fixed_type_weight.carbon_spelling);
    }
  }
  CARBON_CHECK(static_cast<int>(type_names_.size()) == num_types);
  last_type_name_index_ = num_types;

  // Now shuffle both the class names and the type names.
  std::shuffle(class_names_.begin(), class_names_.end(), gen.rng_);
  std::shuffle(type_names_.begin(), type_names_.end(), gen.rng_);
}

// Some heuristic numbers used when formatting generated code. These heuristics
// are loosely based on what we expect to make Carbon code readable, and might
// not fit as well in C++, but we use the same heuristics across languages for
// simplicity and to make the output in different languages more directly
// comparable.
constexpr static int NumSingleLineFunctionParams = 3;
constexpr static int NumSingleLineMethodParams = 2;
constexpr static int MaxParamsPerLine = 4;

static auto EstimateAvgFunctionDeclLines(SourceGen::FunctionDeclParams params)
    -> double {
  // Currently model a uniform distribution [0, max] parameters. Assume a line
  // break before the first parameter for >3 and after every 4th.
  int param_lines = 0;
  for (int num_params : llvm::seq_inclusive(0, params.max_params)) {
    if (num_params > NumSingleLineFunctionParams) {
      param_lines += (num_params + MaxParamsPerLine - 1) / MaxParamsPerLine;
    }
  }
  return 1.0 + static_cast<double>(param_lines) / (params.max_params + 1);
}

static auto EstimateAvgMethodDeclLines(SourceGen::MethodDeclParams params)
    -> double {
  // Currently model a uniform distribution [0, max] parameters. Assume a line
  // break before the first parameter for >2 and after every 4th.
  int param_lines = 0;
  for (int num_params : llvm::seq_inclusive(0, params.max_params)) {
    if (num_params > NumSingleLineMethodParams) {
      param_lines += (num_params + MaxParamsPerLine - 1) / MaxParamsPerLine;
    }
  }
  return 1.0 + static_cast<double>(param_lines) / (params.max_params + 1);
}

// Note that this should match the heuristics used when formatting.
// TODO: See top-level TODO about line estimates and formatting.
static auto EstimateAvgClassDefLines(SourceGen::ClassParams params) -> double {
  // Comment line, and class open line.
  double avg = 2.0;

  // One comment line and blank line per function, plus the function lines.
  avg +=
      (2.0 + EstimateAvgFunctionDeclLines(params.public_function_decl_params)) *
      params.public_function_decls;
  avg += (2.0 + EstimateAvgMethodDeclLines(params.public_method_decl_params)) *
         params.public_method_decls;
  avg += (2.0 +
          EstimateAvgFunctionDeclLines(params.private_function_decl_params)) *
         params.private_function_decls;
  avg += (2.0 + EstimateAvgMethodDeclLines(params.private_method_decl_params)) *
         params.private_method_decls;

  // A blank line and all the fields (if any).
  if (params.private_field_decls > 0) {
    avg += 1.0 + params.private_field_decls;
  }

  // No need to account for the class close line, we have an extra blank line
  // count for the last of the above.
  return avg;
}

auto SourceGen::GenApiFileDenseDecls(int target_lines,
                                     const DenseDeclParams& params)
    -> std::string {
  RawStringOstream source;

  // Figure out how many classes fit in our target lines, each separated by a
  // blank line. We need to account the comment lines below to start the file.
  // Note that we want a blank line after our file comment block, so every class
  // needs a blank line.
  constexpr int NumFileCommentLines = 4;
  double avg_class_lines = EstimateAvgClassDefLines(params.class_params);
  CARBON_CHECK(target_lines > NumFileCommentLines + avg_class_lines,
               "Not enough target lines to generate a single class!");
  int num_classes = static_cast<double>(target_lines - NumFileCommentLines) /
                    (avg_class_lines + 1);
  int expected_lines =
      NumFileCommentLines + num_classes * (avg_class_lines + 1);

  source << "// Generated " << (!IsCpp() ? "Carbon" : "C++")
         << " source file.\n";
  source << llvm::formatv(
                "// {0} target lines: {1} classes, {2} expected lines",
                target_lines, num_classes, expected_lines)
         << "\n";
  source << "//\n// Generating as an API file with dense declarations.\n";

  // Carbon uses an implicitly imported prelude to get builtin types, but C++
  // requires header files so include those.
  if (IsCpp()) {
    source << "\n";
    // Header for specific integer types like `std::int64_t`.
    source << "#include <cstdint>\n";
    // Header for `std::pair`.
    source << "#include <utility>\n";
  }

  auto class_gen_state = ClassGenState(*this, num_classes, params.class_params,
                                       params.type_use_params);
  for ([[maybe_unused]] auto _ : llvm::seq(num_classes)) {
    source << "\n";
    GenerateClassDef(params.class_params, class_gen_state, source);
  }

  // Make sure we consumed all the state.
  CARBON_CHECK(class_gen_state.public_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.public_method_param_counts().empty());
  CARBON_CHECK(class_gen_state.private_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.private_method_param_counts().empty());
  CARBON_CHECK(class_gen_state.class_names().empty());
  CARBON_CHECK(class_gen_state.type_names().empty());

  return source.TakeStr();
}

auto SourceGen::GetShuffledIdentifiers(int number, int min_length,
                                       int max_length, bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> idents =
      GetIdentifiers(number, min_length, max_length, uniform);
  std::shuffle(idents.begin(), idents.end(), rng_);
  return idents;
}

auto SourceGen::GetShuffledUniqueIdentifiers(int number, int min_length,
                                             int max_length, bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length >= 4,
               "Cannot trivially guarantee enough distinct, unique identifiers "
               "for lengths <= 3");
  llvm::SmallVector<llvm::StringRef> idents =
      GetUniqueIdentifiers(number, min_length, max_length, uniform);
  std::shuffle(idents.begin(), idents.end(), rng_);
  return idents;
}

auto SourceGen::GetIdentifiers(int number, int min_length, int max_length,
                               bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> idents = GetIdentifiersImpl(
      number, min_length, max_length, uniform,
      [this](int length, int length_count,
             llvm::SmallVectorImpl<llvm::StringRef>& dest) {
        llvm::append_range(dest,
                           GetSingleLengthIdentifiers(length, length_count));
      });

  return idents;
}

auto SourceGen::GetUniqueIdentifiers(int number, int min_length, int max_length,
                                     bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length >= 4,
               "Cannot trivially guarantee enough distinct, unique identifiers "
               "for lengths <= 3");
  llvm::SmallVector<llvm::StringRef> idents =
      GetIdentifiersImpl(number, min_length, max_length, uniform,
                         [this](int length, int length_count,
                                llvm::SmallVectorImpl<llvm::StringRef>& dest) {
                           AppendUniqueIdentifiers(length, length_count, dest);
                         });

  return idents;
}

auto SourceGen::GetSingleLengthIdentifiers(int length, int number)
    -> llvm::ArrayRef<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef>& idents =
      identifiers_by_length_.Insert(length, {}).value();

  if (static_cast<int>(idents.size()) < number) {
    idents.reserve(number);
    for ([[maybe_unused]] auto _ : llvm::seq<int>(idents.size(), number)) {
      auto ident_storage =
          llvm::MutableArrayRef(reinterpret_cast<char*>(storage_.Allocate(
                                    /*Size=*/length, /*Alignment=*/1)),
                                length);
      GenerateRandomIdentifier(ident_storage);
      llvm::StringRef new_id(ident_storage.data(), length);
      idents.push_back(new_id);
    }
    CARBON_CHECK(static_cast<int>(idents.size()) == number);
  }
  return llvm::ArrayRef(idents).slice(0, number);
}

static auto IdentifierStartChars() -> llvm::ArrayRef<char> {
  static llvm::SmallVector<char> chars = [] {
    llvm::SmallVector<char> chars;
    for (char c : llvm::seq_inclusive('A', 'Z')) {
      chars.push_back(c);
    }
    for (char c : llvm::seq_inclusive('a', 'z')) {
      chars.push_back(c);
    }
    return chars;
  }();
  return chars;
}

static auto IdentifierChars() -> llvm::ArrayRef<char> {
  static llvm::SmallVector<char> chars = [] {
    llvm::ArrayRef<char> start_chars = IdentifierStartChars();
    llvm::SmallVector<char> chars(start_chars.begin(), start_chars.end());
    chars.push_back('_');
    for (char c : llvm::seq_inclusive('0', '9')) {
      chars.push_back(c);
    }
    return chars;
  }();
  return chars;
}

constexpr static llvm::StringRef NonCarbonCppKeywords[] = {
    "asm", "do",  "double", "float",    "int", "long", "new", "signed",
    "std", "try", "unix",   "unsigned", "xor", "NAN",  "M_E", "M_PI",
};

// Returns a random identifier string of the specified length.
//
// Ensures this is a valid identifier, avoiding any overlapping syntaxes or
// keywords both in Carbon and C++.
//
// This routine is somewhat expensive and so is useful to cache and reduce the
// frequency of calls. However, each time it is called it computes a completely
// new random identifier and so can be useful to eventually find a distinct
// identifier when needed.
auto SourceGen::GenerateRandomIdentifier(
    llvm::MutableArrayRef<char> dest_storage) -> void {
  llvm::ArrayRef<char> start_chars = IdentifierStartChars();
  llvm::ArrayRef<char> chars = IdentifierChars();

  llvm::StringRef ident(dest_storage.data(), dest_storage.size());
  do {
    dest_storage[0] =
        start_chars[absl::Uniform<int>(rng_, 0, start_chars.size())];
    for (int i : llvm::seq<int>(1, dest_storage.size())) {
      dest_storage[i] = chars[absl::Uniform<int>(rng_, 0, chars.size())];
    }
  } while (
      // TODO: Clean up and simplify this code. With some small refactorings and
      // post-processing we should be able to make this both easier to read and
      // less inefficient.
      llvm::any_of(
          Lex::TokenKind::KeywordTokens,
          [ident](auto token) { return ident == token.fixed_spelling(); }) ||
      llvm::is_contained(NonCarbonCppKeywords, ident) ||
      ident.ends_with("_t") || ident.ends_with("_MIN") ||
      ident.ends_with("_MAX") || ident.ends_with("_C") ||
      (llvm::is_contained({'i', 'u', 'f'}, ident[0]) &&
       llvm::all_of(ident.substr(1),
                    [](const char c) { return llvm::isDigit(c); })));
}

// Appends a number of unique, random identifiers with a particular length to
// the provided destination vector.
//
// Uses, and when necessary grows, a cached sequence of random identifiers with
// the specified length. Because these are cached, this is efficient to call
// repeatedly, but will not produce a different sequence of identifiers.
auto SourceGen::AppendUniqueIdentifiers(
    int length, int number, llvm::SmallVectorImpl<llvm::StringRef>& dest)
    -> void {
  auto& [count, unique_idents] =
      unique_identifiers_by_length_.Insert(length, {}).value();

  // See if we need to grow our pool of unique identifiers with the requested
  // length.
  if (count < number) {
    // We'll need to insert exactly the requested new unique identifiers. All
    // our other inserts will find an existing entry.
    unique_idents.GrowForInsertCount(count - number);

    // Generate the needed number of identifiers.
    for ([[maybe_unused]] auto _ : llvm::seq<int>(count, number)) {
      // Allocate stable storage for the identifier so we can form stable
      // `StringRef`s to it.
      auto ident_storage =
          llvm::MutableArrayRef(reinterpret_cast<char*>(storage_.Allocate(
                                    /*Size=*/length, /*Alignment=*/1)),
                                length);
      // Repeatedly generate novel identifiers of this length until we find a
      // new unique one.
      for (;;) {
        GenerateRandomIdentifier(ident_storage);
        auto result =
            unique_idents.Insert(llvm::StringRef(ident_storage.data(), length));
        if (result.is_inserted()) {
          break;
        }
      }
    }
    count = number;
  }
  // Append all the identifiers directly out of the set. We make no guarantees
  // about the relative order so we just use the non-deterministic order of the
  // set and avoid additional storage.
  //
  // TODO: It's awkward the `ForEach` here can't early-exit. This just walks the
  // whole set which is harmless if inefficient. We should add early exiting
  // the loop support to `Set` and update this code.
  unique_idents.ForEach([&](llvm::StringRef ident) {
    if (number > 0) {
      dest.push_back(ident);
      --number;
    }
  });
  CARBON_CHECK(number == 0);
}

// An array of the counts that should be used for each identifier length to
// produce our desired distribution.
//
// Note that the zero-based index corresponds to a 1-based length, so the count
// for identifiers of length 1 is at index 0.
static constexpr std::array<int, 64> IdentifierLengthCounts = [] {
  std::array<int, 64> ident_length_counts;
  // For non-uniform distribution, we simulate a distribution roughly based on
  // the observed histogram of identifier lengths, but smoothed a bit and
  // reduced to small counts so that we cycle through all the lengths
  // reasonably quickly. We want sampling of even 10% of NumTokens from this
  // in a round-robin form to not be skewed overly much. This still inherently
  // compresses the long tail as we'd rather have coverage even though it
  // distorts the distribution a bit.
  //
  // The distribution here comes from a script that analyzes source code run
  // over a few directories of LLVM. The script renders a visual ascii-art
  // histogram along with the data for each bucket, and that output is
  // included in comments above each bucket size below to help visualize the
  // rough shape we're aiming for.
  //
  // 1 characters   [3976]  ███████████████████████████████▊
  ident_length_counts[0] = 40;
  // 2 characters   [3724]  █████████████████████████████▊
  ident_length_counts[1] = 40;
  // 3 characters   [4173]  █████████████████████████████████▍
  ident_length_counts[2] = 40;
  // 4 characters   [5000]  ████████████████████████████████████████
  ident_length_counts[3] = 50;
  // 5 characters   [1568]  ████████████▌
  ident_length_counts[4] = 20;
  // 6 characters   [2226]  █████████████████▊
  ident_length_counts[5] = 20;
  // 7 characters   [2380]  ███████████████████
  ident_length_counts[6] = 20;
  // 8 characters   [1786]  ██████████████▎
  ident_length_counts[7] = 18;
  // 9 characters   [1397]  ███████████▏
  ident_length_counts[8] = 12;
  // 10 characters  [ 739]  █████▉
  ident_length_counts[9] = 12;
  // 11 characters  [ 779]  ██████▎
  ident_length_counts[10] = 12;
  // 12 characters  [1344]  ██████████▊
  ident_length_counts[11] = 12;
  // 13 characters  [ 498]  ████
  ident_length_counts[12] = 5;
  // 14 characters  [ 284]  ██▎
  ident_length_counts[13] = 3;
  // 15 characters  [ 172]  █▍
  // 16 characters  [ 278]  ██▎
  // 17 characters  [ 191]  █▌
  // 18 characters  [ 207]  █▋
  for (int i = 14; i < 18; ++i) {
    ident_length_counts[i] = 2;
  }
  // 19 - 63 characters are all <100 but non-zero, and we map them to 1 for
  // coverage despite slightly over weighting the tail.
  for (int i = 18; i < 64; ++i) {
    ident_length_counts[i] = 1;
  }
  return ident_length_counts;
}();

// A template function that implements the common logic of `GetIdentifiers` and
// `GetUniqueIdentifiers`. Most parameters correspond to the parameters of those
// functions. Additionally, an `AppendFunc` callable is provided to implement
// the appending operation.
//
// The main functionality provided here is collecting the correct number of
// identifiers from each of the lengths in the range [min_length, max_length]
// and either in our default representative distribution or a uniform
// distribution.
auto SourceGen::GetIdentifiersImpl(int number, int min_length, int max_length,
                                   bool uniform,
                                   llvm::function_ref<AppendFn> append)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length <= max_length);
  CARBON_CHECK(
      uniform || max_length <= 64,
      "Cannot produce a meaningful non-uniform distribution of lengths longer "
      "than 64 as those are exceedingly rare in our observed data sets.");

  llvm::SmallVector<llvm::StringRef> idents;
  idents.reserve(number);

  // First, compute the total weight of the distribution so we know how many
  // identifiers we'll get each time we collect from it.
  int num_lengths = max_length - min_length + 1;
  auto length_counts =
      llvm::ArrayRef(IdentifierLengthCounts).slice(min_length - 1, num_lengths);
  int count_sum = uniform ? num_lengths : Sum(length_counts);
  CARBON_CHECK(count_sum >= 1);

  int number_rem = number % count_sum;

  // Finally, walk through each length in the distribution.
  for (int length : llvm::seq_inclusive(min_length, max_length)) {
    // Scale how many identifiers we want of this length if computing a
    // non-uniform distribution. For uniform, we always take one.
    int scale = uniform ? 1 : IdentifierLengthCounts[length - 1];

    // Now we can compute how many identifiers of this length to request.
    int length_count = (number / count_sum) * scale;
    if (number_rem > 0) {
      int rem_adjustment = std::min(scale, number_rem);
      length_count += rem_adjustment;
      number_rem -= rem_adjustment;
    }
    append(length, length_count, idents);
  }
  CARBON_CHECK(number_rem == 0, "Unexpected number remaining: {0}", number_rem);
  CARBON_CHECK(static_cast<int>(idents.size()) == number,
               "Ended up with {0} identifiers instead of the requested {1}",
               idents.size(), number);

  return idents;
}

// Returns a shuffled sequence of integers in the range [min, max].
//
// The order of the returned integers is random, but each integer in the range
// appears the same number of times in the result, with the number of
// appearances rounded up for lower numbers and rounded down for higher numbers
// in order to exactly produce `number` results.
auto SourceGen::GetShuffledInts(int number, int min, int max)
    -> llvm::SmallVector<int> {
  llvm::SmallVector<int> ints;
  ints.reserve(number);

  // Evenly distribute to each value between min and max.
  int num_values = max - min + 1;
  for (int i : llvm::seq_inclusive(min, max)) {
    int i_count = number / num_values;
    i_count += i < (min + (number % num_values));
    ints.append(i_count, i);
  }
  CARBON_CHECK(static_cast<int>(ints.size()) == number);

  std::shuffle(ints.begin(), ints.end(), rng_);
  return ints;
}

// A helper to pop series of unique identifiers off a sequence of random
// identifiers that may have duplicates.
//
// This is particularly designed to work with the sequences of non-unique
// identifiers produced by `GetShuffledIdentifiers` with the important property
// that while popping off unique identifiers found in the shuffled list, we
// don't change the distribution of identifier lengths.
//
// The uniqueness is only per-instance of the class, and so an instance can be
// used to extract a series of names that share a scope.
//
// It works by scanning the sequence to extract each unique identifier found,
// swapping it to the back and popping it off the list. This does shuffle the
// order, but it isn't expected to do so in an interesting way.
//
// It also provides a fallback path in case there are no unique identifiers left
// which computes fresh, random identifiers with the same length as the next one
// in the sequence until a unique one is found.
//
// For simplicity of the fallback path, the lifetime of the identifiers produced
// is bound to the lifetime of the popper instance, and not the generator as a
// whole. If this is ever a problematic constraint, we can start copying
// fallback identifiers into the generator's storage.
class SourceGen::UniqueIdentifierPopper {
 public:
  explicit UniqueIdentifierPopper(SourceGen& gen,
                                  llvm::SmallVectorImpl<llvm::StringRef>& data)
      : gen_(&gen), data_(&data), it_(data_->rbegin()) {}

  // Pop the next unique identifier that can be found in the data, or synthesize
  // one with a valid length. Always consumes exactly one identifier from the
  // data.
  //
  // Note that the lifetime of the underlying identifier is that of the popper
  // and not the underlying data.
  auto Pop() -> llvm::StringRef {
    for (auto end = data_->rend(); it_ != end; ++it_) {
      auto insert = set_.Insert(*it_);
      if (!insert.is_inserted()) {
        continue;
      }

      if (it_ != data_->rbegin()) {
        std::swap(*data_->rbegin(), *it_);
      }
      CARBON_CHECK(insert.key() == data_->back());
      return data_->pop_back_val();
    }

    // Out of unique elements. Overwrite the back, preserving its length,
    // generating a new identifiers until we find a unique one and return that.
    // This ensures we continue to consume the structure and produce the same
    // size identifiers even in the fallback.
    int length = data_->pop_back_val().size();
    auto fallback_ident_storage =
        llvm::MutableArrayRef(reinterpret_cast<char*>(gen_->storage_.Allocate(
                                  /*Size=*/length, /*Alignment=*/1)),
                              length);
    for (;;) {
      gen_->GenerateRandomIdentifier(fallback_ident_storage);
      auto fallback_id = llvm::StringRef(fallback_ident_storage.data(), length);
      if (set_.Insert(fallback_id).is_inserted()) {
        return fallback_id;
      }
    }
  }

 private:
  SourceGen* gen_;
  llvm::SmallVectorImpl<llvm::StringRef>* data_;
  llvm::SmallVectorImpl<llvm::StringRef>::reverse_iterator it_;
  Set<llvm::StringRef> set_;
};

// Generates a function declaration and writes it to the provided stream.
//
// The declaration can be configured with a function name, private modifier,
// whether it is a method, the parameter count, an how indented it is.
//
// This is also provided a collection of identifiers to consume as parameter
// names -- it will use a unique popper to extract unique parameter names from
// this collection.
auto SourceGen::GenerateFunctionDecl(
    llvm::StringRef name, bool is_private, bool is_method, int param_count,
    llvm::StringRef indent, llvm::SmallVectorImpl<llvm::StringRef>& param_names,
    llvm::function_ref<auto()->llvm::StringRef> get_type_name,
    llvm::raw_ostream& os) -> void {
  os << indent << "// TODO: make better comment text\n";
  if (!IsCpp()) {
    os << indent << (is_private ? "private " : "") << "fn " << name;

    if (is_method) {
      os << "[self: Self]";
    }
  } else {
    os << indent;
    if (!is_method) {
      os << "static ";
    }
    os << "auto " << name;
  }

  os << "(";

  if (param_count >
      (is_method ? NumSingleLineMethodParams : NumSingleLineFunctionParams)) {
    os << "\n" << indent << "    ";
  }
  UniqueIdentifierPopper unique_param_names(*this, param_names);
  for (int i : llvm::seq(param_count)) {
    if (i > 0) {
      if ((i % MaxParamsPerLine) == 0) {
        os << ",\n" << indent << "    ";
      } else {
        os << ", ";
      }
    }
    if (!IsCpp()) {
      os << unique_param_names.Pop() << ": " << get_type_name();
    } else {
      os << get_type_name() << " " << unique_param_names.Pop();
    }
  }
  os << ")";

  os << " -> " << get_type_name();
  os << ";\n";
}

// Generate a class definition and write it to the provided stream.
//
// The structure of the definition is guided by the `params` provided, and it
// consumes the provided state.
auto SourceGen::GenerateClassDef(const ClassParams& params,
                                 ClassGenState& state, llvm::raw_ostream& os)
    -> void {
  llvm::StringRef name = state.class_names().pop_back_val();
  os << "// TODO: make better comment text\n";
  os << "class " << name << " {\n";
  if (IsCpp()) {
    os << " public:\n";
  }

  // Field types can't be the class we're currently declaring. We enforce this
  // by collecting them before inserting that type into the valid set.
  llvm::SmallVector<llvm::StringRef> field_type_names;
  field_type_names.reserve(params.private_field_decls);
  for ([[maybe_unused]] auto _ : llvm::seq(params.private_field_decls)) {
    field_type_names.push_back(state.GetValidTypeName());
  }

  // Mark this class as now a valid type now that field type names have been
  // collected. We can reference this class from functions and methods within
  // the definition.
  state.AddValidTypeName(name);

  UniqueIdentifierPopper unique_member_names(*this, state.member_names());
  llvm::ListSeparator line_sep("\n");
  for ([[maybe_unused]] auto _ : llvm::seq(params.public_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(
        unique_member_names.Pop(), /*is_private=*/false,
        /*is_method=*/false,
        state.public_function_param_counts().pop_back_val(),
        /*indent=*/"  ", state.param_names(),
        [&] { return state.GetValidTypeName(); }, os);
  }
  for ([[maybe_unused]] auto _ : llvm::seq(params.public_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(
        unique_member_names.Pop(), /*is_private=*/false,
        /*is_method=*/true, state.public_method_param_counts().pop_back_val(),
        /*indent=*/"  ", state.param_names(),
        [&] { return state.GetValidTypeName(); }, os);
  }

  if (IsCpp()) {
    os << "\n private:\n";
    // Reset the separator.
    line_sep = llvm::ListSeparator("\n");
  }

  for ([[maybe_unused]] auto _ : llvm::seq(params.private_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(
        unique_member_names.Pop(), /*is_private=*/true,
        /*is_method=*/false,
        state.private_function_param_counts().pop_back_val(),
        /*indent=*/"  ", state.param_names(),
        [&] { return state.GetValidTypeName(); }, os);
  }
  for ([[maybe_unused]] auto _ : llvm::seq(params.private_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(
        unique_member_names.Pop(), /*is_private=*/true,
        /*is_method=*/true, state.private_method_param_counts().pop_back_val(),
        /*indent=*/"  ", state.param_names(),
        [&] { return state.GetValidTypeName(); }, os);
  }
  os << line_sep;
  for (llvm::StringRef type_name : field_type_names) {
    if (!IsCpp()) {
      os << "  private var " << unique_member_names.Pop() << ": " << type_name
         << ";\n";
    } else {
      os << "  " << type_name << " " << unique_member_names.Pop() << ";\n";
    }
  }
  os << "}" << (IsCpp() ? ";" : "") << "\n";
}

}  // namespace Carbon::Testing
