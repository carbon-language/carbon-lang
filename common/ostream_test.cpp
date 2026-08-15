// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/ostream.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <compare>
#include <concepts>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAre;

// A child that defaults its comparisons with member declarations.
struct Point : Printable<Point> {
  int x;
  int y;

  constexpr Point(int x, int y) : x(x), y(y) {}

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "(" << x << ", " << y << ")";
  }

  auto operator<=>(const Point& rhs) const = default;
};

// A child that defaults its comparisons with friend declarations, and whose
// comparisons are neither trivial nor `noexcept`.
struct Label : Printable<Label> {
  std::string text;

  explicit Label(std::string text) : text(std::move(text)) {}

  auto Print(llvm::raw_ostream& out) const -> void { out << text; }

  friend auto operator<=>(const Label& lhs, const Label& rhs) = default;
};

// A child that defaults equality without providing any ordering.
struct Id : Printable<Id> {
  int value;

  constexpr explicit Id(int value) : value(value) {}

  auto Print(llvm::raw_ostream& out) const -> void { out << "#" << value; }

  auto operator==(const Id& rhs) const -> bool = default;
};

// A child whose defaulted comparison is only a partial ordering.
struct Measure : Printable<Measure> {
  double value;

  constexpr explicit Measure(double value) : value(value) {}

  auto Print(llvm::raw_ostream& out) const -> void { out << value; }

  auto operator<=>(const Measure& rhs) const = default;
};

// A child that requests a weaker ordering than its members provide.
struct Version : Printable<Version> {
  int major;
  int minor;

  constexpr Version(int major, int minor) : major(major), minor(minor) {}

  auto Print(llvm::raw_ostream& out) const -> void {
    out << major << "." << minor;
  }

  auto operator<=>(const Version& rhs) const -> std::weak_ordering = default;
};

// A child that doesn't want to be compared at all.
struct Opaque : Printable<Opaque> {
  int value;

  constexpr explicit Opaque(int value) : value(value) {}

  auto Print(llvm::raw_ostream& out) const -> void { out << value; }
};

// A child that compares through an implicit conversion rather than through
// operators of its own, the way `EnumBase` children do.
class Level : public Printable<Level> {
 public:
  enum RawLevel { Low, High };

  constexpr explicit Level(RawLevel value) : value_(value) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  explicit(false) constexpr operator RawLevel() const { return value_; }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << (value_ == Low ? "low" : "high");
  }

 private:
  RawLevel value_;
};

TEST(PrintableTest, Printing) {
  RawStringOstream raw_out;
  raw_out << Point(1, 2) << " " << Label("label");
  EXPECT_EQ(raw_out.TakeStr(), "(1, 2) label");

  std::ostringstream standard_out;
  standard_out << Point(1, 2) << " " << Label("label");
  EXPECT_EQ(standard_out.str(), "(1, 2) label");

  EXPECT_EQ(PrintToString(Point(1, 2)), "(1, 2)");
}

TEST(PrintableTest, DefaultedEquality) {
  EXPECT_EQ(Point(1, 2), Point(1, 2));
  EXPECT_NE(Point(1, 2), Point(1, 3));
  EXPECT_NE(Point(1, 2), Point(2, 2));

  static_assert(Point(1, 2) == Point(1, 2));
  static_assert(Point(1, 2) != Point(1, 3));

  // The base class comparisons don't make defaulted comparisons throwing.
  Point point(1, 2);
  static_assert(noexcept(point == point));
}

TEST(PrintableTest, DefaultedOrdering) {
  // Ordering is lexicographic in declaration order, with the empty base class
  // contributing nothing.
  EXPECT_LT(Point(1, 2), Point(1, 3));
  EXPECT_LT(Point(1, 9), Point(2, 0));
  EXPECT_LE(Point(1, 2), Point(1, 2));
  EXPECT_GT(Point(2, 0), Point(1, 9));
  EXPECT_GE(Point(1, 2), Point(1, 2));

  EXPECT_EQ(Point(1, 2) <=> Point(1, 3), std::strong_ordering::less);
  EXPECT_EQ(Point(1, 2) <=> Point(1, 2), std::strong_ordering::equal);
  EXPECT_EQ(Point(1, 3) <=> Point(1, 2), std::strong_ordering::greater);

  static_assert(std::totally_ordered<Point>);
  static_assert(std::same_as<std::compare_three_way_result_t<Point>,
                             std::strong_ordering>);
  static_assert(Point(1, 2) < Point(1, 3));

  Point point(1, 2);
  static_assert(noexcept(point <=> point));
}

TEST(PrintableTest, DefaultedFriendComparison) {
  EXPECT_EQ(Label("a"), Label("a"));
  EXPECT_NE(Label("a"), Label("b"));
  EXPECT_LT(Label("a"), Label("b"));
  EXPECT_EQ(Label("a") <=> Label("b"), std::strong_ordering::less);

  static_assert(std::totally_ordered<Label>);
}

TEST(PrintableTest, DefaultedEqualityWithoutOrdering) {
  EXPECT_EQ(Id(1), Id(1));
  EXPECT_NE(Id(1), Id(2));

  static_assert(std::equality_comparable<Id>);
  static_assert(!std::totally_ordered<Id>);
  static_assert(!std::three_way_comparable<Id>);
}

TEST(PrintableTest, DefaultedComparisonCategories) {
  static_assert(std::same_as<std::compare_three_way_result_t<Measure>,
                             std::partial_ordering>);
  static_assert(std::same_as<std::compare_three_way_result_t<Version>,
                             std::weak_ordering>);

  EXPECT_LT(Measure(1.0), Measure(2.0));
  EXPECT_EQ(Measure(1.0) <=> Measure(2.0), std::partial_ordering::less);

  // The base class comparing equal must not make unordered values ordered.
  Measure nan(std::numeric_limits<double>::quiet_NaN());
  EXPECT_EQ(nan <=> Measure(1.0), std::partial_ordering::unordered);
  EXPECT_NE(nan, nan);

  EXPECT_LT(Version(1, 0), Version(1, 1));
  EXPECT_EQ(Version(1, 0) <=> Version(1, 1), std::weak_ordering::less);
}

TEST(PrintableTest, NoComparisonWithoutDefaulting) {
  // Inheriting from `Printable` must not by itself make a type comparable, and
  // in particular must not make distinct values compare equal.
  static_assert(!std::equality_comparable<Opaque>);
  static_assert(!std::totally_ordered<Opaque>);
  static_assert(!std::three_way_comparable<Opaque>);

  EXPECT_EQ(PrintToString(Opaque(1)), "1");
}

TEST(PrintableTest, ComparisonThroughConversion) {
  // The base class comparisons must not displace comparisons that a child
  // provides through a conversion.
  EXPECT_TRUE(Level(Level::Low) == Level(Level::Low));
  EXPECT_FALSE(Level(Level::Low) == Level(Level::High));
  EXPECT_TRUE(Level(Level::Low) < Level(Level::High));
  EXPECT_TRUE(Level(Level::High) == Level::High);
}

TEST(PrintableTest, ComparisonsUsableGenerically) {
  llvm::SmallVector<Point> points = {Point(2, 1), Point(1, 2), Point(1, 1)};
  llvm::sort(points);
  EXPECT_THAT(points, ElementsAre(Point(1, 1), Point(1, 2), Point(2, 1)));

  EXPECT_EQ(llvm::find(points, Point(1, 2)), points.begin() + 1);
}

TEST(PrintableTest, EmptyBaseClass) {
  // The comparison support must not add any state to children.
  static_assert(sizeof(Point) == 2 * sizeof(int));
  static_assert(sizeof(Id) == sizeof(int));
  static_assert(std::is_empty_v<Printable<Point>>);
}

}  // namespace
}  // namespace Carbon::Testing
