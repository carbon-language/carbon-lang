#include "char-set.h"

namespace Fortran {
namespace parser {

std::string SetOfCharsToString(SetOfChars set) {
  std::string result;
  for (char ch{' '}; set != 0; ++ch) {
    if (IsCharInSet(set, ch)) {
      set -= SingletonChar(ch);
      result += ch;
    }
  }
  return result;
}

}  // namespace parser
}  // namespace Fortran
