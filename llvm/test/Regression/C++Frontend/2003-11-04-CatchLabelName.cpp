#include <string>

void bar();

void test() {
  try {
    bar();
  } catch (std::string) {}
}
