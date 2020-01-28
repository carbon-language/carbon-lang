// Basic tests of I/O API

#include "../../runtime/io-api.h"
#include <cstring>
#include <iostream>

using namespace Fortran::runtime::io;

static int failures{0};

int main() {
  char buffer[32];
  const char *format1{"(12HHELLO, WORLD)"};
  auto cookie{IONAME(BeginInternalFormattedOutput)(buffer, sizeof buffer, format1, std::strlen(format1))};
  if (auto status{IONAME(EndIoStatement)(cookie)}) {
    std::cerr << "format1 failed, status " << static_cast<int>(status) << '\n';
    ++failures;
  }
  std::string got1{buffer, sizeof buffer};
  std::string expect1{"HELLO, WORLD"};
  expect1.resize(got1.length(), ' ');
  if (got1 != expect1) {
    std::cerr << "format1 failed, got '" << got1 << "', expected '" << expect1 << "'\n";
    ++failures;
  }

  if (failures == 0) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL " << failures << " tests\n";
  }
  return failures > 0;
}
