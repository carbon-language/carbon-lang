// RUN: %check_clang_tidy %s readability-string-compare %t

namespace std {
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T = std::char_traits<C>, typename A = std::allocator<C>>
class basic_string {
public:
  basic_string();
  basic_string(const C *, unsigned int size);
  int compare(const basic_string<char> &str) const;
  int compare(const C *) const;
  int compare(int, int, const basic_string<char> &str) const;
  bool empty();
};
bool operator==(const basic_string<char> &lhs, const basic_string<char> &rhs);
bool operator!=(const basic_string<char> &lhs, const basic_string<char> &rhs);
bool operator==(const basic_string<char> &lhs, const char *&rhs);
typedef basic_string<char> string;
}

void func(bool b);

std::string comp() {
  std::string str("a", 1);
  return str;
}

void Test() {
  std::string str1("a", 1);
  std::string str2("b", 1);

  if (str1.compare(str2)) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings; use the string equality operator instead [readability-string-compare]
  if (!str1.compare(str2)) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:8: warning: do not use 'compare' to test equality of strings; use the string equality operator instead [readability-string-compare]
  if (str1.compare(str2) == 0) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
  // CHECK-FIXES: if (str1 == str2) {
  if (str1.compare(str2) != 0) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
  // CHECK-FIXES: if (str1 != str2) {
  if (str1.compare("foo") == 0) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
  // CHECK-FIXES: if (str1 == "foo") {
  if (0 == str1.compare(str2)) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
  // CHECK-FIXES: if (str2 == str1) {
  if (0 != str1.compare(str2)) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
  // CHECK-FIXES: if (str2 != str1) {
  func(str1.compare(str2));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: do not use 'compare' to test equality of strings;
  if (str2.empty() || str1.compare(str2) != 0) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:23: warning: do not use 'compare' to test equality of strings;
  // CHECK-FIXES: if (str2.empty() || str1 != str2) {
  std::string *str3 = &str1;
  if (str3->compare(str2)) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
  if (str3->compare(str2) == 0) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
  // CHECK-FIXES: if (*str3 == str2) {
  if (str2.compare(*str3) == 0) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
  // CHECK-FIXES: if (str2 == *str3) {
  if (comp().compare(str1) == 0) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
  // CHECK-FIXES: if (comp() == str1) {
  if (str1.compare(comp()) == 0) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
  // CHECK-FIXES: if (str1 == comp()) {
  if (str1.compare(comp())) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings;
}

void Valid() {
  std::string str1("a", 1);
  std::string str2("b", 1);
  if (str1 == str2) {
  }
  if (str1 != str2) {
  }
  if (str1.compare(str2) == str1.compare(str2)) {
  }
  if (0 == 0) {
  }
  if (str1.compare(str2) > 0) {
  }
  if (str1.compare(1, 3, str2)) {
  }
  if (str1.compare(str2) > 0) {
  }
  if (str1.compare(str2) < 0) {
  }
  if (str1.compare(str2) == 2) {
  }
  if (str1.compare(str2) == -3) {
  }
  if (str1.compare(str2) == 1) {
  }
  if (str1.compare(str2) == -1) {
  }
}
