#ifndef FORTRAN_POSITION_H_
#define FORTRAN_POSITION_H_

// Represents a position in a source file.
// TODO: Support multiple source files for inclusion and support contextual
// positions for macro expansions.

#include <ostream>

namespace Fortran {

class Position {
 public:
  constexpr Position() {}
  constexpr Position(const Position &) = default;
  constexpr Position(Position &&) = default;
  constexpr Position(int ln, int col) : lineNumber_{ln}, column_{col} {}
  constexpr Position &operator=(const Position &) = default;
  constexpr Position &operator=(Position &&) = default;

  constexpr int lineNumber() const { return lineNumber_; }
  constexpr int column() const { return column_; }
  Position &set_lineNumber(int line) { lineNumber_ = line; return *this; }
  Position &set_column(int column) { column_ = column; return *this; }

  constexpr bool operator<(const Position &that) const {
    return lineNumber_ < that.lineNumber_ ||
           (lineNumber_ == that.lineNumber_ &&
            column_ < that.column_);
  }

  constexpr bool operator<=(const Position &that) const {
    return lineNumber_ < that.lineNumber_ ||
           (lineNumber_ == that.lineNumber_ &&
            column_ <= that.column_);
  }

  constexpr bool operator==(const Position &that) const {
    return lineNumber_ == that.lineNumber_ &&
           column_ == that.column_;
  }

  constexpr bool operator!=(const Position &that) const {
    return !operator==(that);
  }

  constexpr bool operator>(const Position &that) const {
    return !operator<=(that);
  }

  constexpr bool operator>=(const Position &that) const {
    return !operator<(that);
  }

  void AdvanceColumn() {
    ++column_;
  }

  void TabAdvanceColumn() {
    column_ = ((column_ + 7) & -8) + 1;
  }

  void AdvanceLine() {
    ++lineNumber_;
    column_ = 1;
  }

 private:
  int lineNumber_{1};
  int column_{1};
};

std::ostream &operator<<(std::ostream &, const Position &);
}  // namespace Fortran
#endif  // FORTRAN_POSITION_H_
