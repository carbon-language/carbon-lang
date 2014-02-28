//===-- X86AsmParserCommon.h - Common functions for X86AsmParser ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef X86_ASM_PARSER_COMMON_H
#define X86_ASM_PARSER_COMMON_H

namespace llvm {

inline bool isImmSExti16i8Value(uint64_t Value) {
  return ((                                  Value <= 0x000000000000007FULL)||
          (0x000000000000FF80ULL <= Value && Value <= 0x000000000000FFFFULL)||
          (0xFFFFFFFFFFFFFF80ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
}

inline bool isImmSExti32i8Value(uint64_t Value) {
  return ((                                  Value <= 0x000000000000007FULL)||
          (0x00000000FFFFFF80ULL <= Value && Value <= 0x00000000FFFFFFFFULL)||
          (0xFFFFFFFFFFFFFF80ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
}

inline bool isImmZExtu32u8Value(uint64_t Value) {
    return (Value <= 0x00000000000000FFULL);
}

inline bool isImmSExti64i8Value(uint64_t Value) {
  return ((                                  Value <= 0x000000000000007FULL)||
          (0xFFFFFFFFFFFFFF80ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
}

inline bool isImmSExti64i32Value(uint64_t Value) {
  return ((                                  Value <= 0x000000007FFFFFFFULL)||
          (0xFFFFFFFF80000000ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
}

} // End of namespace llvm

#endif // X86_ASM_PARSER_COMMON_H
