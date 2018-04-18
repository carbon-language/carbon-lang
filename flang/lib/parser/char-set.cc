#include "char-set.h"

namespace Fortran {
namespace parser {

std::string SetOfChars::ToString() const {
  std::string result;
  std::uint64_t set{bits_};
  for (char ch{' '}; set != 0; ++ch) {
    if (IsCharInSet(set, ch)) {
      set -= SetOfChars{ch}.bits_;
      result += ch;
    }
  }
  return result;
}

}  // namespace parser
}  // namespace Fortran
