

#include "llvm/Support/StringExtras.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/DerivedTypes.h"

// Can we treat the specified array as a string?  Only if it is an array of
// ubytes or non-negative sbytes.
//
bool isStringCompatible(ConstPoolArray *CPA) {
  const Type *ETy = cast<ArrayType>(CPA->getType())->getElementType();
  if (ETy == Type::UByteTy) return true;
  if (ETy != Type::SByteTy) return false;

  for (unsigned i = 0; i < CPA->getNumOperands(); ++i)
    if (cast<ConstPoolSInt>(CPA->getOperand(i))->getValue() < 0)
      return false;

  return true;
}

// toOctal - Convert the low order bits of X into an octal letter
static inline char toOctal(int X) {
  return (X&7)+'0';
}

// getAsCString - Return the specified array as a C compatible string, only if
// the predicate isStringCompatible is true.
//
string getAsCString(ConstPoolArray *CPA) {
  if (isStringCompatible(CPA)) {
    string Result;
    const Type *ETy = cast<ArrayType>(CPA->getType())->getElementType();
    Result = "\"";
    for (unsigned i = 0; i < CPA->getNumOperands(); ++i) {
      unsigned char C = (ETy == Type::SByteTy) ?
        (unsigned char)cast<ConstPoolSInt>(CPA->getOperand(i))->getValue() :
        (unsigned char)cast<ConstPoolUInt>(CPA->getOperand(i))->getValue();

      if (isprint(C)) {
        Result += C;
      } else {
        switch(C) {
        case '\a': Result += "\\a"; break;
        case '\b': Result += "\\b"; break;
        case '\f': Result += "\\f"; break;
        case '\n': Result += "\\n"; break;
        case '\r': Result += "\\r"; break;
        case '\t': Result += "\\t"; break;
        case '\v': Result += "\\v"; break;
        default:
          Result += '\\';
          Result += toOctal(C >> 6);
          Result += toOctal(C >> 3);
          Result += toOctal(C >> 0);
          break;
        }
      }
    }
    Result += "\"";

    return Result;
  } else {
    return CPA->getStrValue();
  }
}
