/* ===-- inttypes.h - stub SDK header for compiler-rt -----------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===-----------------------------------------------------------------------===
 *
 * This is a stub SDK header file. This file is not part of the interface of
 * this library nor an official version of the appropriate SDK header. It is
 * intended only to stub the features of this header required by compiler-rt.
 *
 * ===-----------------------------------------------------------------------===
 */

#ifndef __INTTYPES_H__
#define __INTTYPES_H__

#if __WORDSIZE == 64
#define __INTTYPE_PRI64__ "l"
#else
#define __INTTYPE_PRI64__ "ll"
#endif

#define PRId8  "hhd"
#define PRId16 "hd"
#define PRId32 "d"
#define PRId64 __INTTYPE_PRI64__ "d"

#define PRIi8  "hhi"
#define PRIi16 "hi"
#define PRIi32 "i"
#define PRIi64 __INTTYPE_PRI64__ "i"

#define PRIo8  "hho"
#define PRIo16 "ho"
#define PRIo32 "o"
#define PRIo64 __INTTYPE_PRI64__ "o"

#define PRIu8  "hhu"
#define PRIu16 "hu"
#define PRIu32 "u"
#define PRIu64 __INTTYPE_PRI64__ "u"

#define PRIx8  "hhx"
#define PRIx16 "hx"
#define PRIx32 "x"
#define PRIx64 __INTTYPE_PRI64__ "x"

#define PRIX8  "hhX"
#define PRIX16 "hX"
#define PRIX32 "X"
#define PRIX64 __INTTYPE_PRI64__ "X"

#define SCNd8  "hhd"
#define SCNd16 "hd"
#define SCNd32 "d"
#define SCNd64 __INTTYPE_PRI64__ "d"

#define SCNi8  "hhi"
#define SCNi16 "hi"
#define SCNi32 "i"
#define SCNi64 __INTTYPE_PRI64__ "i"

#define SCNo8  "hho"
#define SCNo16 "ho"
#define SCNo32 "o"
#define SCNo64 __INTTYPE_PRI64__ "o"

#define SCNu8  "hhu"
#define SCNu16 "hu"
#define SCNu32 "u"
#define SCNu64 __INTTYPE_PRI64__ "u"

#define SCNx8  "hhx"
#define SCNx16 "hx"
#define SCNx32 "x"
#define SCNx64 __INTTYPE_PRI64__ "x"

#define SCNX8  "hhX"
#define SCNX16 "hX"
#define SCNX32 "X"
#define SCNX64 __INTTYPE_PRI64__ "X"

#endif  /* __INTTYPES_H__ */
