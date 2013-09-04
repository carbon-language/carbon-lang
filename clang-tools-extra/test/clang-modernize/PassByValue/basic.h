#ifndef BASIC_H
#define BASIC_H

// POD types are trivially move constructible
struct Movable {
  int a, b, c;
};

struct NotMovable {
  NotMovable() = default;
  NotMovable(const NotMovable &) = default;
  NotMovable(NotMovable &&) = delete;
  int a, b, c;
};

// The test runs the migrator without header modifications enabled for this
// header making the constructor parameter M unmodifiable.
struct UnmodifiableClass {
  UnmodifiableClass(const Movable &M);
  Movable M;
};

#endif // BASIC_H
