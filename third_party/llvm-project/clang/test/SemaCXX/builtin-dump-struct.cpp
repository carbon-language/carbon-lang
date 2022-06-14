// RUN: %clang_cc1 -std=c++20 -verify %s

namespace std {
  typedef decltype(sizeof(int)) size_t;

  template <class E> struct initializer_list {
    const E *data;
    size_t size;

    constexpr initializer_list(const E *data, size_t size)
        : data(data), size(size) {}
    constexpr initializer_list() : data(), size() {}

    constexpr const E *begin() const { return data; }
    constexpr const E *end() const { return data + size; }
  };
}

struct ConstexprString {
  constexpr ConstexprString() : ConstexprString("") {}
  constexpr ConstexprString(const char *p, std::size_t size) : data(new char[size+1]) {
    __builtin_memcpy(data, p, size);
    data[size] = '\0';
  }
  constexpr ConstexprString(const char *p) : ConstexprString(p, __builtin_strlen(p)) {}
  constexpr explicit ConstexprString(const char *p, const char *q) : data(nullptr) {
    auto p_size = __builtin_strlen(p);
    auto q_size = __builtin_strlen(q);
    data = new char[p_size + q_size + 1];
    __builtin_memcpy(data, p, p_size);
    __builtin_memcpy(data + p_size, q, q_size + 1);
  }
  constexpr ConstexprString(const ConstexprString &o) : ConstexprString(o.data) {}
  constexpr ConstexprString(ConstexprString &&o) : data(o.data) { o.data = nullptr; }
  constexpr ConstexprString &operator=(const ConstexprString &o) {
    return *this = ConstexprString(o);
  }
  constexpr ConstexprString &operator=(ConstexprString &&o) {
    delete[] data;
    data = o.data;
    o.data = nullptr;
    return *this;
  }
  constexpr ~ConstexprString() { delete[] data; }
  char *data;

  friend constexpr ConstexprString operator+(const ConstexprString &a, const ConstexprString &b) {
    return ConstexprString(a.data, b.data);
  }
  friend constexpr ConstexprString &operator+=(ConstexprString &a, const ConstexprString &b) {
    return a = a + b;
  }
  friend constexpr bool operator==(const ConstexprString &a, const ConstexprString &b) {
    return __builtin_strcmp(a.data, b.data) == 0;
  }
};

template<typename... T> constexpr void Format(ConstexprString &out, const char *fmt, T... args);

struct Arg {
  template<typename T, int (*)[__is_integral(T) ? 1 : -1] = nullptr>
  constexpr Arg(T value) {
    bool negative = false;
    if (value < 0) {
      value = -value;
      negative = true;
    }
    while (value > 0) {
      char str[2] = {char('0' + value % 10), '\0'};
      s = ConstexprString(str) + s;
      value /= 10;
    }
    if (negative)
      s = "-" + s;
  }
  template<typename T, int (*)[__is_class(T) ? 1 : -1] = nullptr>
  constexpr Arg(const T &value) {
    __builtin_dump_struct(&value, Format, s);
  }
  constexpr Arg(const char *s) : s(s) {}
  constexpr Arg(const ConstexprString *s) : s("\"" + *s + "\"") {}
  template<typename T, int (*)[__is_integral(T) ? 1 : -1] = nullptr>
  constexpr Arg(const T *p) : s("reference to " + Arg(*p).s) {}
  ConstexprString s;
};

template<typename... T> constexpr void Format(ConstexprString &out, const char *fmt, T... args) { // #Format
  Arg formatted_args[] = {args...};
  int i = 0;
  while (const char *percent = __builtin_strchr(fmt, '%')) {
    if (percent[1] == '%') continue;
    if (percent != fmt && percent[-1] == '*') --percent;
    out += ConstexprString(fmt, percent - fmt);
    out += formatted_args[i++].s;

    // Skip past format specifier until we hit a conversion specifier.
    fmt = percent;
    while (!__builtin_strchr("diouxXeEfFgGcsp", *fmt)) ++fmt;
    // Skip the conversion specifier too. TODO: Check it's the right one.
    ++fmt;
  }
  out += ConstexprString(fmt);
}

template<typename T> constexpr ConstexprString ToString(const T &t) { return Arg(t).s; }

struct A {
  int x, y, z : 3;
  int : 4;
  ConstexprString s;
};
struct B : A {
  int p, q;
  struct {
    int anon1, anon2;
  };
  union {
    int anon3;
  };
  struct {
    int m;
  } c;
  int &&r;
};

#if PRINT_OUTPUT
#include <stdio.h>
int main() {
  puts(ToString(B{1, 2, 3, "hello", 4, 5, 6, 7, 8, 9, 10}).data);
}
#else
static_assert(ToString(B{1, 2, 3, "hello", 4, 5, 6, 7, 8, 9, 10}) == &R"(
B {
  A {
    int x = 1
    int y = 2
    int z : 3 = 3
    ConstexprString s = "hello"
  }
  int p = 4
  int q = 5
  int anon1 = 6
  int anon2 = 7
  int anon3 = 8
  struct (unnamed) c = {
    int m = 9
  }
  int && r = reference to 10
}
)"[1]);

void errors(B b) {
  __builtin_dump_struct(); // expected-error {{too few arguments to function call, expected 2, have 0}}
  __builtin_dump_struct(1); // expected-error {{too few arguments to function call, expected 2, have 1}}
  __builtin_dump_struct(1, 2); // expected-error {{expected pointer to struct as 1st argument to '__builtin_dump_struct', found 'int'}}
  __builtin_dump_struct(&b, 2); // expected-error {{expected a callable expression as 2nd argument to '__builtin_dump_struct', found 'int'}}
  __builtin_dump_struct(&b, Format, 0); // expected-error {{no matching function for call to 'Format'}}
                                        // expected-note@-1 {{in call to printing function with arguments '(0, "%s", "B")' while dumping struct}}
                                        // expected-note@#Format {{no known conversion from 'int' to 'ConstexprString &' for 1st argument}}
}
#endif
