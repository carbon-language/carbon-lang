#include "USRFindingAction.h"
#include "gtest/gtest.h"
#include "clang/Tooling/Tooling.h"
#include <stdio.h>
#include <set>
#include <map>
#include <vector>

namespace clang {
namespace rename {
namespace test {

// Determines if the symbol group invariants hold. To recap, those invariants
// are:
//  (1) All symbols in the same symbol group share the same USR.
//  (2) Two symbols from two different groups do not share the same USR.
static void testOffsetGroups(const char *Code,
                             const std::vector<std::vector<unsigned>> Groups) {
  std::set<std::string> AllUSRs, CurrUSR;

  for (const auto &Group : Groups) {
    // Groups the invariants do not hold then the value of USR is also invalid,
    // but at that point the test has already failed and USR ceases to be
    // useful.
    std::string USR;
    for (const auto &Offset : Group) {
      USRFindingAction Action(Offset);
      auto Factory = tooling::newFrontendActionFactory(&Action);
      EXPECT_TRUE(tooling::runToolOnCode(Factory->create(), Code));
      const auto &USRs = Action.getUSRs();
      EXPECT_EQ(1u, USRs.size());
      USR = USRs[0];
      CurrUSR.insert(USR);
    }
    EXPECT_EQ(1u, CurrUSR.size());
    CurrUSR.clear();
    AllUSRs.insert(USR);
  }

  EXPECT_EQ(Groups.size(), AllUSRs.size());
}

#if !(defined(_MSC_VER) && _MSC_VER < 1800)
TEST(USRLocFinding, FindsVarUSR) {
  const char VarTest[] = "\n\
namespace A {\n\
int foo;\n\
}\n\
int foo;\n\
int bar = foo;\n\
int baz = A::foo;\n\
void fun1() {\n\
  struct {\n\
    int foo;\n\
  } b = { 100 };\n\
  int foo = 100;\n\
  baz = foo;\n\
  {\n\
    extern int foo;\n\
    baz = foo;\n\
    foo = A::foo + baz;\n\
    A::foo = b.foo;\n\
  }\n\
 foo = b.foo;\n\
}\n";
  std::vector<std::vector<unsigned>> VarTestOffsets = {
    { 19, 63, 205, 223 },
    { 30, 45, 172, 187 },
    { 129, 148, 242 },
  };

  testOffsetGroups(VarTest, VarTestOffsets);
}
#endif

}
}
}
