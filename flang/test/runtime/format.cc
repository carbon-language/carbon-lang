// Test basic FORMAT string traversal
#include "../runtime/format.h"
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <list>
#include <string>

using namespace Fortran::runtime;
using namespace std::literals::string_literals;

static int failures{0};
using Results = std::list<std::string>;
static Results results;

static void handleCharacterLiteral(const char *s, std::size_t len) {
  std::string str{s, len};
  results.push_back("'"s + str + '\'');
}

static void handleSlash() { results.emplace_back("/"); }

static void handleAbsolutePosition(int n) {
  results.push_back("T"s + std::to_string(n));
}

static void handleRelativePosition(int n) {
  if (n < 0) {
    results.push_back("TL"s + std::to_string(-n));
  } else {
    results.push_back(std::to_string(n) + 'X');
  }
}

static void Report(const DataEdit &edit) {
  std::string str{edit.descriptor};
  if (edit.repeat != 1) {
    str = std::to_string(edit.repeat) + '*' + str;
  }
  if (edit.variation) {
    str += edit.variation;
  }
  str += std::to_string(edit.width);
  if (edit.digits) {
    str += "."s + std::to_string(*edit.digits);
  }
  if (edit.expoDigits) {
    str += "E"s + std::to_string(*edit.expoDigits);
  }
  // modes?
  results.push_back(str);
}

// Override the Crash() in the runtime library
void Terminator::Crash(const char *message, ...) {
  std::va_list ap;
  va_start(ap, message);
  char buffer[1000];
  std::vsnprintf(buffer, sizeof buffer, message, ap);
  va_end(ap);
  throw std::string{buffer};
}

static void Check(Results &expect) {
  if (expect != results) {
    std::cerr << "expected:";
    for (const std::string &s : expect) {
      std::cerr << ' ' << s;
    }
    std::cerr << "\ngot:";
    for (const std::string &s : results) {
      std::cerr << ' ' << s;
    }
    std::cerr << '\n';
    ++failures;
  }
  expect.clear();
  results.clear();
}

static void Test(FormatContext &context, int n, const char *format,
    Results &&expect, int repeat = 1) {
  MutableModes modes;
  FormatControl control{context, format, std::strlen(format), modes};
  try {
    for (int j{0}; j < n; ++j) {
      DataEdit edit;
      control.GetNext(edit, repeat);
      Report(edit);
    }
    control.FinishOutput();
  } catch (const std::string &crash) {
    results.push_back("Crash:"s + crash);
  }
  Check(expect);
}

int main() {
  Terminator terminator{"source", 1};
  FormatContext context{terminator, &handleCharacterLiteral, nullptr, nullptr,
      &handleSlash, &handleAbsolutePosition, &handleRelativePosition};
  Test(context, 1, "('PI=',F9.7)", Results{"'PI='", "F9.7"});
  Test(context, 1, "(3HPI=F9.7)", Results{"'PI='", "F9.7"});
  Test(context, 1, "(3HPI=/F9.7)", Results{"'PI='", "/", "F9.7"});
  Test(context, 2, "('PI=',F9.7)", Results{"'PI='", "F9.7", "'PI='", "F9.7"});
  Test(context, 2, "(2('PI=',F9.7),'done')",
      Results{"'PI='", "F9.7", "'PI='", "F9.7", "'done'"});
  Test(context, 2, "(3('PI=',F9.7,:),'tooFar')",
      Results{"'PI='", "F9.7", "'PI='", "F9.7"});
  Test(context, 2, "(*('PI=',F9.7,:),'tooFar')",
      Results{"'PI='", "F9.7", "'PI='", "F9.7"});
  Test(context, 1, "(3F9.7)", Results{"2*F9.7"}, 2);
  return failures > 0;
}
