#include "char-set.h"

namespace Fortran {
namespace parser {

std::string SetOfChars::ToString() const {
  std::string result;
  SetOfChars set{*this};
  for (char ch{' '}; !set.empty(); ++ch) {
    if (set.Has(ch)) {
      set = set.Difference(ch);
      result += ch;
    }
  }
  return result;
}

}  // namespace parser
}  // namespace Fortran
