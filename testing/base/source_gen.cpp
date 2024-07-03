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
static auto EstimateAvgClassLines(SourceGen::ClassParams params) -> double {
  // Blank line, comment line, and class open line.
  double avg = 3.0;

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

  double avg_class_lines = EstimateAvgClassLines(params.class_params);
  int num_classes = static_cast<double>(target_lines) / avg_class_lines;
  int expected_lines = num_classes * avg_class_lines;
  os << "// Generated " << (!IsCpp() ? "Carbon" : "C++") << " source file.\n";
  os << llvm::formatv("// {0} target lines: {1} classes, {2} expected lines",
                      target_lines, num_classes, expected_lines)
     << "\n";
  os << "//\n// Generating as an API file with dense declarations.\n\n";

  auto class_gen_state = GetClassGenState(num_classes, params.class_params);
  llvm::ListSeparator line_sep("\n");
  for ([[maybe_unused]] int i : llvm::seq(num_classes)) {
    os << line_sep;
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

auto SourceGen::GetShuffledIds(int number, int min_length, int max_length,
                               bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> ids =
      GetIds(number, min_length, max_length, uniform);
  std::shuffle(ids.begin(), ids.end(), rng);
  return ids;
}

auto SourceGen::GetShuffledUniqueIds(int number, int min_length, int max_length,
                                     bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length >= 4)
      << "Cannot trivially guarantee enough distinct, unique identifiers for "
         "lengths <= 3";
  llvm::SmallVector<llvm::StringRef> ids =
      GetUniqueIds(number, min_length, max_length, uniform);
  std::shuffle(ids.begin(), ids.end(), rng);
  return ids;
}

static constexpr std::array<int, 64> IdLengthCounts = [] {
  std::array<int, 64> id_length_counts;
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
  id_length_counts[0] = 40;
  // 2 characters   [3724]  █████████████████████████████▊
  id_length_counts[1] = 40;
  // 3 characters   [4173]  █████████████████████████████████▍
  id_length_counts[2] = 40;
  // 4 characters   [5000]  ████████████████████████████████████████
  id_length_counts[3] = 50;
  // 5 characters   [1568]  ████████████▌
  id_length_counts[4] = 20;
  // 6 characters   [2226]  █████████████████▊
  id_length_counts[5] = 20;
  // 7 characters   [2380]  ███████████████████
  id_length_counts[6] = 20;
  // 8 characters   [1786]  ██████████████▎
  id_length_counts[7] = 18;
  // 9 characters   [1397]  ███████████▏
  id_length_counts[8] = 12;
  // 10 characters  [ 739]  █████▉
  id_length_counts[9] = 12;
  // 11 characters  [ 779]  ██████▎
  id_length_counts[10] = 12;
  // 12 characters  [1344]  ██████████▊
  id_length_counts[11] = 12;
  // 13 characters  [ 498]  ████
  id_length_counts[12] = 5;
  // 14 characters  [ 284]  ██▎
  id_length_counts[13] = 3;
  // 15 characters  [ 172]  █▍
  // 16 characters  [ 278]  ██▎
  // 17 characters  [ 191]  █▌
  // 18 characters  [ 207]  █▋
  for (int i = 14; i < 18; ++i) {
    id_length_counts[i] = 2;
  }
  // 19 - 63 characters are all <100 but non-zero, and we map them to 1 for
  // coverage despite slightly over weighting the tail.
  for (int i = 18; i < 64; ++i) {
    id_length_counts[i] = 1;
  }
  return id_length_counts;
}();

template <typename T>
static auto Sum(const T& range) -> int {
  return std::accumulate(range.begin(), range.end(), 0);
}

// Note that this template must be defined prior to its use below.
template <typename AppendIds>
auto SourceGen::GetIdsImpl(int number, int min_length, int max_length,
                           bool uniform, AppendIds append_ids)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length <= max_length);
  CARBON_CHECK(uniform || max_length <= 64)
      << "Cannot produce a meaningful non-uniform distribution of lengths "
         "longer than 64 as those are exceedingly rare in our observed data "
         "sets.";

  llvm::SmallVector<llvm::StringRef> ids;
  ids.reserve(number);

  // First, compute how many identifiers of each size we'll need.
  int count_sum =
      uniform ? (max_length - min_length) + 1
              : Sum(llvm::ArrayRef(IdLengthCounts)
                        .slice(min_length - 1, max_length - min_length + 1));
  CARBON_CHECK(count_sum >= 1);
  int number_rem = number % count_sum;
  for (int length : llvm::seq_inclusive(min_length, max_length)) {
    // Scale this length if non-uniform.
    int scale = uniform ? 1 : IdLengthCounts[length - 1];
    int length_count = (number / count_sum) * scale;
    if (number_rem > 0) {
      length_count += std::min(scale, number_rem);
      number_rem -= scale;
    }
    append_ids(length, length_count, ids);
  }
  CARBON_CHECK(static_cast<int>(ids.size()) == number);

  return ids;
}

auto SourceGen::GetIds(int number, int min_length, int max_length, bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> ids =
      GetIdsImpl(number, min_length, max_length, uniform,
                 [this](int length, int length_count,
                        llvm::SmallVectorImpl<llvm::StringRef>& dest) {
                   auto length_ids = GetSingleLengthIds(length, length_count);
                   dest.append(length_ids.begin(), length_ids.end());
                 });

  return ids;
}

auto SourceGen::GetUniqueIds(int number, int min_length, int max_length,
                             bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length >= 4)
      << "Cannot trivially guarantee enough distinct, unique identifiers for "
         "lengths <= 3";
  llvm::SmallVector<llvm::StringRef> ids =
      GetIdsImpl(number, min_length, max_length, uniform,
                 [this](int length, int length_count,
                        llvm::SmallVectorImpl<llvm::StringRef>& dest) {
                   AppendUniqueIdentifiers(length, length_count, dest);
                 });

  return ids;
}

auto SourceGen::GetSingleLengthIds(int length, int number)
    -> llvm::ArrayRef<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef>& ids =
      ids_by_length.Insert(length, {}).value();

  if (static_cast<int>(ids.size()) < number) {
    ids.reserve(number);
    for ([[maybe_unused]] int i : llvm::seq<int>(ids.size(), number)) {
      char* id_storage = reinterpret_cast<char*>(
          storage.Allocate(/*Size=*/length, /*Alignment=*/1));
      std::string new_id_tmp = GenerateRandomIdentifier(length);
      memcpy(id_storage, new_id_tmp.data(), length);
      llvm::StringRef new_id(id_storage, length);
      ids.push_back(new_id);
    }
    CARBON_CHECK(static_cast<int>(ids.size()) == number);
  }
  return llvm::ArrayRef(ids).slice(0, number);
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

// Generates a random identifier string of the specified length using the
// provided RNG BitGen.
auto SourceGen::GenerateRandomIdentifier(int length) -> std::string {
  llvm::ArrayRef<char> start_chars = IdentifierStartChars();
  llvm::ArrayRef<char> chars = IdentifierChars();

  std::string id_result;
  llvm::raw_string_ostream os(id_result);
  llvm::StringRef id;
  do {
    // Erase any prior attempts to find an identifier.
    id_result.clear();
    os << start_chars[absl::Uniform<int>(rng, 0, start_chars.size())];
    for (int j : llvm::seq(1, length)) {
      static_cast<void>(j);
      os << chars[absl::Uniform<int>(rng, 0, chars.size())];
    }
    // Check if we ended up forming an integer type literal or a keyword, and
    // try again.
    id = llvm::StringRef(id_result);
  } while (
      // TODO: Clean up and simplify this code. With some small refactorings and
      // post-processing we should be able to make this both easier to read and
      // less inefficient.
      llvm::any_of(Lex::TokenKind::KeywordTokens,
                   [id](auto token) { return id == token.fixed_spelling(); }) ||
      llvm::is_contained(NonCarbonCppKeywords, id) ||
      (llvm::is_contained({'i', 'u', 'f'}, id[0]) &&
       llvm::all_of(id.substr(1),
                    [](const char c) { return llvm::isDigit(c); })));
  CARBON_CHECK(id == id_result);
  CARBON_CHECK(static_cast<int>(id.size()) == length);
  return id_result;
}

auto SourceGen::AppendUniqueIdentifiers(
    int length, int number, llvm::SmallVectorImpl<llvm::StringRef>& dest)
    -> void {
  auto& [count, unique_ids] = unique_ids_by_length.Insert(length, {}).value();

  if (count < number) {
    unique_ids.GrowForInsertCount(count - number);
    for ([[maybe_unused]] int i : llvm::seq<int>(count, number)) {
      char* id_storage = reinterpret_cast<char*>(
          storage.Allocate(/*Size=*/length, /*Alignment=*/1));
      for (;;) {
        std::string new_id_tmp = GenerateRandomIdentifier(length);
        memcpy(id_storage, new_id_tmp.data(), length);
        llvm::StringRef new_id(id_storage, length);
        auto result = unique_ids.Insert(new_id);
        if (result.is_inserted()) {
          break;
        }
      }
    }
    count = number;
  }
  unique_ids.ForEach([&](llvm::StringRef id) {
    if (number > 0) {
      dest.push_back(id);
      --number;
    }
  });
  CARBON_CHECK(number == 0);
}

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

  std::shuffle(ints.begin(), ints.end(), rng);
  return ints;
}

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

  state.class_names = GetShuffledUniqueIds(number, /*min_length=*/5);
  int num_members =
      number * (params.public_function_decls + params.public_method_decls +
                params.private_function_decls + params.private_method_decls +
                params.private_field_decls);
  state.member_names = GetShuffledIds(num_members, /*min_length=*/4);
  int num_params = Sum(state.public_function_param_counts) +
                   Sum(state.public_method_param_counts) +
                   Sum(state.private_function_param_counts) +
                   Sum(state.private_method_param_counts);
  state.param_names = GetShuffledIds(num_params);
  return state;
}

class SourceGen::UniqueIdPopper {
 public:
  explicit UniqueIdPopper(SourceGen& gen,
                          llvm::SmallVectorImpl<llvm::StringRef>& data)
      : gen_(&gen), data_(&data), it_(data_->rbegin()) {}

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

    // Out of unique elements. Pop the back and use its length to generate new
    // identifiers until we find a unique one and return that. This ensures we
    // continue to consume the structure and produce the same size identifiers
    // even in the fallback. Note that fallback identifiers only live to the end
    // of the popper.
    int length = data_->pop_back_val().size();
    fallback_ids_.push_back("");
    std::string& fallback_id = fallback_ids_.back();
    for (;;) {
      fallback_id = gen_->GenerateRandomIdentifier(length);
      if (set_.Insert(llvm::StringRef(fallback_id)).is_inserted()) {
        return fallback_id;
      }
    }
  }

 private:
  SourceGen* gen_;
  llvm::SmallVectorImpl<llvm::StringRef>* data_;
  llvm::SmallVectorImpl<llvm::StringRef>::reverse_iterator it_;
  llvm::SmallVector<std::string> fallback_ids_;
  Set<llvm::StringRef> set_;
};

auto SourceGen::GenerateFunctionDecl(
    llvm::StringRef name, bool is_private, bool is_method,
    llvm::SmallVectorImpl<int>& param_counts,
    llvm::SmallVectorImpl<llvm::StringRef>& param_names, llvm::raw_ostream& os,
    llvm::StringRef indent) -> void {
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

  int param_count = param_counts.pop_back_val();
  if (param_count >
      (is_method ? NumSingleLineMethodParams : NumSingleLineFunctionParams)) {
    os << "\n" << indent << "    ";
  }
  UniqueIdPopper unique_param_names(*this, param_names);
  for (int i : llvm::seq(0, param_count)) {
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
auto SourceGen::GenerateClassDef(const ClassParams& params,
                                 ClassGenState& state, llvm::raw_ostream& os)
    -> void {
  os << "// TODO: make better comment text\n";
  os << "class " << state.class_names.pop_back_val() << " {\n";
  if (IsCpp()) {
    os << " public:\n";
  }

  UniqueIdPopper unique_member_names(*this, state.member_names);
  llvm::ListSeparator line_sep("\n");
  for ([[maybe_unused]] int i : llvm::seq(0, params.public_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(
        unique_member_names.Pop(), /*is_private=*/false, /*is_method=*/false,
        state.public_function_param_counts, state.param_names, os, "  ");
  }
  for ([[maybe_unused]] int i : llvm::seq(0, params.public_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(unique_member_names.Pop(), /*is_private=*/false,
                         /*is_method=*/true, state.public_method_param_counts,
                         state.param_names, os, "  ");
  }

  if (IsCpp()) {
    os << "\n private:\n";
    // Reset the separator.
    line_sep = llvm::ListSeparator("\n");
  }

  for ([[maybe_unused]] int i : llvm::seq(0, params.private_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(
        unique_member_names.Pop(), /*is_private=*/true, /*is_method=*/false,
        state.private_function_param_counts, state.param_names, os, "  ");
  }
  for ([[maybe_unused]] int i : llvm::seq(0, params.private_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(unique_member_names.Pop(), /*is_private=*/true,
                         /*is_method=*/true, state.private_method_param_counts,
                         state.param_names, os, "  ");
  }
  os << line_sep;
  for ([[maybe_unused]] int i : llvm::seq(0, params.private_field_decls)) {
    if (!IsCpp()) {
      os << "  private var " << unique_member_names.Pop() << ": i32;\n";
    } else {
      os << "  int " << unique_member_names.Pop() << ";\n";
    }
  }
  os << "}" << (IsCpp() ? ";" : "") << "\n";
}

}  // namespace Carbon::Testing
