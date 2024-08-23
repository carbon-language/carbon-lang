// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/source_gen.h"

#include <numeric>

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

auto SourceGen::GenAPIFileDenseDecls(int target_lines, DenseDeclParams params)
    -> std::string {
  std::string source;
  llvm::raw_string_ostream os(source);

  // Figure out how many classes fit in our target lines, each separated by a
  // blank line. We need to account the comment lines below to start the file.
  // Note that we want a blank line after our file comment block, so every class
  // needs a blank line.
  constexpr int NumFileCommentLines = 4;
  double avg_class_lines = EstimateAvgClassDefLines(params.class_params);
  CARBON_CHECK(target_lines > NumFileCommentLines + avg_class_lines)
      << "Not enough target lines to generate a single class!";
  int num_classes = static_cast<double>(target_lines - NumFileCommentLines) /
                    (avg_class_lines + 1);
  int expected_lines =
      NumFileCommentLines + num_classes * (avg_class_lines + 1);

  os << "// Generated " << (!IsCpp() ? "Carbon" : "C++") << " source file.\n";
  os << llvm::formatv("// {0} target lines: {1} classes, {2} expected lines",
                      target_lines, num_classes, expected_lines)
     << "\n";
  os << "//\n// Generating as an API file with dense declarations.\n";

  auto class_gen_state = GetClassGenState(num_classes, params.class_params);
  for ([[maybe_unused]] int _ : llvm::seq(num_classes)) {
    os << "\n";
    GenerateClassDef(params.class_params, class_gen_state, os);
  }

  // Make sure we consumed all the state.
  CARBON_CHECK(class_gen_state.public_function_param_counts.empty());
  CARBON_CHECK(class_gen_state.public_method_param_counts.empty());
  CARBON_CHECK(class_gen_state.private_function_param_counts.empty());
  CARBON_CHECK(class_gen_state.private_method_param_counts.empty());
  CARBON_CHECK(class_gen_state.class_names.empty());

  return source;
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
  CARBON_CHECK(min_length >= 4)
      << "Cannot trivially guarantee enough distinct, unique identifiers for "
         "lengths <= 3";
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
        auto length_idents = GetSingleLengthIdentifiers(length, length_count);
        dest.append(length_idents.begin(), length_idents.end());
      });

  return idents;
}

auto SourceGen::GetUniqueIdentifiers(int number, int min_length, int max_length,
                                     bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length >= 4)
      << "Cannot trivially guarantee enough distinct, unique identifiers for "
         "lengths <= 3";
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
    for ([[maybe_unused]] int _ : llvm::seq<int>(idents.size(), number)) {
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
    "asm", "do",     "double", "float", "int",      "long",
    "new", "signed", "try",    "unix",  "unsigned", "xor",
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
    for ([[maybe_unused]] int _ : llvm::seq<int>(count, number)) {
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

// A helper to sum elements of a range.
template <typename T>
static auto Sum(const T& range) -> int {
  return std::accumulate(range.begin(), range.end(), 0);
}

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
  CARBON_CHECK(uniform || max_length <= 64)
      << "Cannot produce a meaningful non-uniform distribution of lengths "
         "longer than 64 as those are exceedingly rare in our observed data "
         "sets.";

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
  CARBON_CHECK(number_rem == 0)
      << "Unexpected number remaining: " << number_rem;
  CARBON_CHECK(static_cast<int>(idents.size()) == number)
      << "Ended up with " << idents.size()
      << " identifiers instead of the requested " << number;

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

// Given a number of class definitions and the params with which to generate
// them, builds the state that will be used while generating that many classes.
//
// We build the state first and across all the class definitions that will be
// generated so that we can distribute random components across all the
// definitions.
auto SourceGen::GetClassGenState(int number, ClassParams params)
    -> ClassGenState {
  ClassGenState state;
  state.public_function_param_counts =
      GetShuffledInts(number * params.public_function_decls, 0,
                      params.public_function_decl_params.max_params);
  state.public_method_param_counts =
      GetShuffledInts(number * params.public_method_decls, 0,
                      params.public_method_decl_params.max_params);
  state.private_function_param_counts =
      GetShuffledInts(number * params.private_function_decls, 0,
                      params.private_function_decl_params.max_params);
  state.private_method_param_counts =
      GetShuffledInts(number * params.private_method_decls, 0,
                      params.private_method_decl_params.max_params);

  state.class_names = GetShuffledUniqueIdentifiers(number, /*min_length=*/5);
  int num_members =
      number * (params.public_function_decls + params.public_method_decls +
                params.private_function_decls + params.private_method_decls +
                params.private_field_decls);
  state.member_names = GetShuffledIdentifiers(num_members, /*min_length=*/4);
  int num_params = Sum(state.public_function_param_counts) +
                   Sum(state.public_method_param_counts) +
                   Sum(state.private_function_param_counts) +
                   Sum(state.private_method_param_counts);
  state.param_names = GetShuffledIdentifiers(num_params);
  return state;
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
      os << unique_param_names.Pop() << ": i32";
    } else {
      os << "int " << unique_param_names.Pop();
    }
  }

  os << ")" << (IsCpp() ? " -> void" : "") << ";\n";
}

// Generate a class definition and write it to the provided stream.
//
// The structure of the definition is guided by the `params` provided, and it
// consumes the provided state.
auto SourceGen::GenerateClassDef(const ClassParams& params,
                                 ClassGenState& state, llvm::raw_ostream& os)
    -> void {
  os << "// TODO: make better comment text\n";
  os << "class " << state.class_names.pop_back_val() << " {\n";
  if (IsCpp()) {
    os << " public:\n";
  }

  UniqueIdentifierPopper unique_member_names(*this, state.member_names);
  llvm::ListSeparator line_sep("\n");
  for ([[maybe_unused]] int _ : llvm::seq(params.public_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(unique_member_names.Pop(), /*is_private=*/false,
                         /*is_method=*/false,
                         state.public_function_param_counts.pop_back_val(),
                         /*indent=*/"  ", state.param_names, os);
  }
  for ([[maybe_unused]] int _ : llvm::seq(params.public_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(unique_member_names.Pop(), /*is_private=*/false,
                         /*is_method=*/true,
                         state.public_method_param_counts.pop_back_val(),
                         /*indent=*/"  ", state.param_names, os);
  }

  if (IsCpp()) {
    os << "\n private:\n";
    // Reset the separator.
    line_sep = llvm::ListSeparator("\n");
  }

  for ([[maybe_unused]] int _ : llvm::seq(params.private_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(unique_member_names.Pop(), /*is_private=*/true,
                         /*is_method=*/false,
                         state.private_function_param_counts.pop_back_val(),
                         /*indent=*/"  ", state.param_names, os);
  }
  for ([[maybe_unused]] int _ : llvm::seq(params.private_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(unique_member_names.Pop(), /*is_private=*/true,
                         /*is_method=*/true,
                         state.private_method_param_counts.pop_back_val(),
                         /*indent=*/"  ", state.param_names, os);
  }
  os << line_sep;
  for ([[maybe_unused]] int _ : llvm::seq(params.private_field_decls)) {
    if (!IsCpp()) {
      os << "  private var " << unique_member_names.Pop() << ": i32;\n";
    } else {
      os << "  int " << unique_member_names.Pop() << ";\n";
    }
  }
  os << "}" << (IsCpp() ? ";" : "") << "\n";
}

}  // namespace Carbon::Testing
