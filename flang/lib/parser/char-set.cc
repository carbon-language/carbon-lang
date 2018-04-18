#include "char-set.h"

namespace Fortran {
namespace parser {

std::string SetOfCharsToString(SetOfChars set) {
  int code{0};
  std::string result;
  for (SetOfChars bit{1}; set != 0; bit = bit + bit, ++code) {
    if ((set & bit) != 0) {
      set &= ~bit;
      result += SixBitDecoding(code);
    }
  }
  return result;
}

}  // namespace parser
}  // namespace Fortran
