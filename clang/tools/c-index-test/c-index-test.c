
#include "clang-c/Index.h"

/*
 * First sign of life:-)
 */
int main(int argc, char **argv) {
  CXIndex Idx = clang_createIndex();
  CXTranslationUnit TU = clang_createTranslationUnit(Idx, argv[1]);
  clang_loadTranslationUnit(TU, 0);
  return 1;
}
