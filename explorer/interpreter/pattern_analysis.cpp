// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/pattern_analysis.h"

#include <set>

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace Carbon {

auto AbstractPattern::kind() const -> Kind {
  if (auto* pattern = value_.dyn_cast<const Pattern*>()) {
    return Compound;
  }
  if (auto* value = value_.dyn_cast<const Value*>()) {
    if (isa<TupleValue, AlternativeValue, BoolValue>(value)) {
      return Compound;
    }
    return Primitive;
  }
  CARBON_CHECK(value_.is<const WildcardTag*>());
  return Wildcard;
}

auto AbstractPattern::discriminator() const -> std::string_view {
  CARBON_CHECK(kind() == Compound);
  if (auto* pattern = value_.dyn_cast<const Pattern*>()) {
    if (auto* alt_pattern = dyn_cast<AlternativePattern>(pattern)) {
      return alt_pattern->alternative_name();
    }
  } else if (auto* value = value_.dyn_cast<const Value*>()) {
    if (auto* alt = dyn_cast<AlternativeValue>(value)) {
      return alt->alt_name();
    } else if (auto* bool_val = dyn_cast<BoolValue>(value)) {
      return bool_val->value() ? "true" : "false";
    }
  }
  return {};
}

auto AbstractPattern::elements_size() const -> int {
  if (auto* pattern = value_.dyn_cast<const Pattern*>()) {
    if (auto* tuple_pattern = dyn_cast<TuplePattern>(pattern)) {
      return tuple_pattern->fields().size();
    } else if (isa<AlternativePattern>(pattern)) {
      return 1;
    }
  } else if (auto* value = value_.dyn_cast<const Value*>()) {
    if (auto* tuple = dyn_cast<TupleValue>(value)) {
      return tuple->elements().size();
    } else if (auto* alt = dyn_cast<AlternativeValue>(value)) {
      return 1;
    }
  }
  return 0;
}

void AbstractPattern::AppendElementsTo(
    std::vector<AbstractPattern>& out) const {
  if (auto* pattern = value_.dyn_cast<const Pattern*>()) {
    if (auto* tuple_pattern = dyn_cast<TuplePattern>(pattern)) {
      auto fields = tuple_pattern->fields();
      out.insert(out.end(), fields.begin(), fields.end());
    } else if (auto* alt_pattern = dyn_cast<AlternativePattern>(pattern)) {
      out.push_back(&alt_pattern->arguments());
    }
  } else if (auto* value = value_.dyn_cast<const Value*>()) {
    if (auto* tuple = dyn_cast<TupleValue>(value)) {
      auto* tuple_type = cast<TupleValue>(type_);
      CARBON_CHECK(tuple->elements().size() == tuple_type->elements().size());
      for (size_t i = 0; i != tuple->elements().size(); ++i) {
        out.push_back(
            AbstractPattern(tuple->elements()[i], tuple_type->elements()[i]));
      }
    } else if (auto* alt = dyn_cast<AlternativeValue>(value)) {
      out.push_back(AbstractPattern(
          &alt->argument(),
          *cast<ChoiceType>(type_)->FindAlternative(alt->alt_name())));
    }
  }
}

auto AbstractPattern::value() const -> const Value& {
  CARBON_CHECK(kind() == Primitive);
  return *value_.get<const Value*>();
}

auto AbstractPattern::type() const -> const Value& {
  CARBON_CHECK(kind() != Wildcard);
  return *type_;
}

void AbstractPattern::Set(Nonnull<const Pattern*> pattern) {
  type_ = &pattern->static_type();
  switch (pattern->kind()) {
    case PatternKind::AddrPattern:
    case PatternKind::AutoPattern:
    case PatternKind::BindingPattern:
    case PatternKind::GenericBinding:
      value_ = static_cast<const WildcardTag*>(nullptr);
      break;

    case PatternKind::TuplePattern:
    case PatternKind::AlternativePattern:
      value_ = pattern;
      break;

    case PatternKind::ExpressionPattern:
      value_ = &pattern->value();
      break;

    case PatternKind::VarPattern:
      Set(&cast<VarPattern>(pattern)->pattern());
      break;
  }
}

auto PatternMatrix::IsUseful(llvm::ArrayRef<AbstractPattern> pattern,
                             int max_exponential_depth) const -> bool {
  if (matrix_.empty()) {
    return true;
  }

  CARBON_CHECK(pattern.size() == matrix_[0].size());
  if (matrix_[0].empty()) {
    return false;
  }

  switch (pattern[0].kind()) {
    case AbstractPattern::Wildcard: {
      auto discrim = FirstColumnDiscriminators();
      // Check if we hit the depth limit. If so, we act as if the
      // constructors present in this position are not exhaustive, that is,
      // as if the type we're matching has some other constructor not
      // corresponding to anything written in the pattern in this position.
      // This can lead us to conclude that a pattern is useful if it is not,
      // and that a set of patterns is not exhaustive when it is.
      int new_depth =
          max_exponential_depth - (discrim.found.size() > 1 ? 1 : 0);
      if (!discrim.any_missing && new_depth >= 0) {
        for (auto found : discrim.found) {
          if (Specialize(found).IsUseful(*SpecializeRow(pattern, found),
                                         new_depth)) {
            return true;
          }
        }
        return false;
      }
      return Default().IsUseful(pattern.slice(1), max_exponential_depth);
    }

    case AbstractPattern::Compound: {
      DiscriminatorInfo discrim = {.discriminator = pattern[0].discriminator(),
                                   .size = pattern[0].elements_size()};
      return Specialize(discrim).IsUseful(*SpecializeRow(pattern, discrim),
                                          max_exponential_depth);
    }

    case AbstractPattern::Primitive: {
      return Specialize(pattern[0].value())
          .IsUseful(pattern.slice(1), max_exponential_depth);
    }
  }
}

auto PatternMatrix::FirstColumnDiscriminators() const -> DiscriminatorSet {
  std::set<std::string_view> discrims;
  std::optional<int> num_discrims;
  std::optional<int> elem_size;

  for (auto& row : matrix_) {
    CARBON_CHECK(!row.empty());
    switch (row[0].kind()) {
      case AbstractPattern::Wildcard:
        continue;
      case AbstractPattern::Compound: {
        const Value& type = row[0].type();
        if (auto* tuple = dyn_cast<TupleValue>(&type)) {
          // If we find a tuple match, we've found all constructors (there's
          // only one!) and none were missing.
          return {
              .found = {{.discriminator = {},
                         .size = static_cast<int>(tuple->elements().size())}},
              .any_missing = false};
        } else if (auto* choice = dyn_cast<ChoiceType>(&type)) {
          num_discrims = choice->declaration().alternatives().size();
          elem_size = 1;
        } else if (isa<BoolType>(type)) {
          // `bool` behaves like a choice type with two alternativs,
          // and with no nested patterns for either of them.
          num_discrims = 2;
          elem_size = 0;
        } else {
          llvm_unreachable("unexpected compound type");
        }
        discrims.insert(row[0].discriminator());
        break;
      }
      case AbstractPattern::Primitive: {
        // We assume that primitive value matches are always incomplete, even
        // for types like `i8` where a covering match might be possible.
        return {.found = {}, .any_missing = true};
      }
    }
  }

  if (!num_discrims || *num_discrims != static_cast<int>(discrims.size())) {
    return {.found = {}, .any_missing = true};
  }

  DiscriminatorSet result = {.found = {}, .any_missing = false};
  result.found.reserve(discrims.size());
  for (auto s : discrims) {
    result.found.push_back({.discriminator = s, .size = *elem_size});
  }
  return result;
}

auto PatternMatrix::SpecializeRow(llvm::ArrayRef<AbstractPattern> row,
                                  DiscriminatorInfo discriminator)
    -> std::optional<std::vector<AbstractPattern>> {
  CARBON_CHECK(!row.empty());
  std::vector<AbstractPattern> new_row;
  switch (row[0].kind()) {
    case AbstractPattern::Wildcard:
      new_row.reserve(discriminator.size + row.size() - 1);
      new_row.insert(new_row.end(), discriminator.size,
                     AbstractPattern::MakeWildcard());
      break;
    case AbstractPattern::Compound: {
      if (row[0].discriminator() != discriminator.discriminator) {
        return std::nullopt;
      }
      CARBON_CHECK(static_cast<int>(row[0].elements_size()) ==
                   discriminator.size);
      new_row.reserve(discriminator.size + row.size() - 1);
      row[0].AppendElementsTo(new_row);
      break;
    }
    case AbstractPattern::Primitive:
      // These cases should be rejected by the type checker.
      llvm_unreachable("matched primitive against compound");
  }
  new_row.insert(new_row.end(), row.begin() + 1, row.end());
  return std::move(new_row);
}

auto PatternMatrix::Specialize(DiscriminatorInfo discriminator) const
    -> PatternMatrix {
  PatternMatrix specialized;
  for (auto& row : matrix_) {
    // TODO: If we add support for "or" patterns, specialization might
    // produce multiple rows here.
    if (auto new_row = SpecializeRow(row, discriminator)) {
      specialized.Add(std::move(new_row.value()));
    }
  }
  return specialized;
}

// Specialize the pattern matrix for the case where the first value is known
// to be `value`, and is not matched.
auto PatternMatrix::Specialize(const Value& value) const -> PatternMatrix {
  PatternMatrix specialized;
  for (auto& row : matrix_) {
    CARBON_CHECK(!row.empty());
    switch (row[0].kind()) {
      case AbstractPattern::Wildcard:
        break;
      case AbstractPattern::Compound:
        llvm_unreachable("matched compound against primitive");
      case AbstractPattern::Primitive:
        // TODO: Use an equality context here?
        if (!ValueEqual(&row[0].value(), &value, std::nullopt)) {
          continue;
        }
        break;
    }
    specialized.Add(std::vector<AbstractPattern>(row.begin() + 1, row.end()));
  }
  return specialized;
}

// Specialize the pattern matrix for the case where the first value uses a
// discriminator matching none of the non-wildcard patterns.
auto PatternMatrix::Default() const -> PatternMatrix {
  PatternMatrix default_matrix;
  for (auto& row : matrix_) {
    CARBON_CHECK(!row.empty());
    switch (row[0].kind()) {
      case AbstractPattern::Wildcard:
        default_matrix.Add(
            std::vector<AbstractPattern>(row.begin() + 1, row.end()));
        break;
      case AbstractPattern::Compound:
      case AbstractPattern::Primitive:
        break;
    }
  }
  return default_matrix;
}

}  // namespace Carbon
