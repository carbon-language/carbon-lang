//===--- HexagonConstExtInfo.h - Provides constant extender information ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions that extract constant extender
// information for a specified instruction from the HexagonConstExtInfo table.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONCONSTEXT_H
#define HEXAGONCONSTEXT_H
namespace llvm {
namespace HexagonConstExt {
  typedef struct {
    const char * Name;
    const short CExtOpNum;
    const int MinValue;
    const int MaxValue;
    const int NonExtOpcode;
  } HexagonConstExtInfo;

#include "HexagonCExtTable.h"

/// HexagonConstExt - This namespace holds the constant extension related
/// information.

  bool isOperandExtended(unsigned short Opcode, unsigned short OperandNum);
  unsigned short getCExtOpNum(unsigned short Opcode);
  int getMinValue(unsigned short Opcode);
  int getMaxValue(unsigned short Opcode);
  bool NonExtEquivalentExists (unsigned short Opcode);
  int getNonExtOpcode (unsigned short Opcode);
}

}
#endif
