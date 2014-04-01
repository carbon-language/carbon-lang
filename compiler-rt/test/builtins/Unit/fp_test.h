//===--------------------------- fp_test.h - ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines shared functions for the test.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <limits.h>
#include <string.h>

enum EXPECTED_RESULT {
    LESS_0, LESS_EQUAL_0, EQUAL_0, GREATER_0, GREATER_EQUAL_0, NEQUAL_0
};

static inline float fromRep32(uint32_t x)
{
    float ret;
    memcpy(&ret, &x, 4);
    return ret;
}

static inline double fromRep64(uint64_t x)
{
    double ret;
    memcpy(&ret, &x, 8);
    return ret;
}

static inline long double fromRep128(uint64_t hi, uint64_t lo)
{
    __uint128_t x = ((__uint128_t)hi << 64) + lo;
    long double ret;
    memcpy(&ret, &x, 16);
    return ret;
}

static inline uint32_t toRep32(float x)
{
    uint32_t ret;
    memcpy(&ret, &x, 4);
    return ret;
}

static inline uint64_t toRep64(double x)
{
    uint64_t ret;
    memcpy(&ret, &x, 8);
    return ret;
}

static inline __uint128_t toRep128(long double x)
{
    __uint128_t ret;
    memcpy(&ret, &x, 16);
    return ret;
}

static inline int compareResultF(float result,
                                 uint32_t expected)
{
    uint32_t rep = toRep32(result);

    if (rep == expected){
        return 0;
    }
    // test other posible NaN representation(signal NaN)
    else if (expected == 0x7fc00000U){
        if ((rep & 0x7f800000U) == 0x7f800000U &&
            (rep & 0x7fffffU) > 0){
            return 0;
        }
    }
    return 1;
}

static inline int compareResultD(double result,
                                 uint64_t expected)
{
    uint64_t rep = toRep64(result);

    if (rep == expected){
        return 0;
    }
    // test other posible NaN representation(signal NaN)
    else if (expected == 0x7ff8000000000000UL){
        if ((rep & 0x7ff0000000000000UL) == 0x7ff0000000000000UL &&
            (rep & 0xfffffffffffffUL) > 0){
            return 0;
        }
    }
    return 1;
}

// return 0 if equal
// use two 64-bit integers intead of one 128-bit integer
// because 128-bit integer constant can't be assigned directly
static inline int compareResultLD(long double result,
                                  uint64_t expectedHi,
                                  uint64_t expectedLo)
{
    __uint128_t rep = toRep128(result);
    uint64_t hi = rep >> 64;
    uint64_t lo = rep;

    if (hi == expectedHi && lo == expectedLo){
        return 0;
    }
    // test other posible NaN representation(signal NaN)
    else if (expectedHi == 0x7fff800000000000UL && expectedLo == 0x0UL){
        if ((hi & 0x7fff000000000000UL) == 0x7fff000000000000UL &&
            ((hi & 0xffffffffffffUL) > 0 || lo > 0)){
            return 0;
        }
    }
    return 1;
}

static inline int compareResultCMP(int result,
                                   enum EXPECTED_RESULT expected)
{
    switch(expected){
        case LESS_0:
            if (result < 0)
                return 0;
            break;
        case LESS_EQUAL_0:
            if (result <= 0)
                return 0;
            break;
        case EQUAL_0:
            if (result == 0)
                return 0;
            break;
        case NEQUAL_0:
            if (result != 0)
                return 0;
            break;
        case GREATER_EQUAL_0:
            if (result >= 0)
                return 0;
            break;
        case GREATER_0:
            if (result > 0)
                return 0;
            break;
        default:
            return 1;
    }
    return 1;
}

static inline char *expectedStr(enum EXPECTED_RESULT expected)
{
    switch(expected){
        case LESS_0:
            return "<0";
        case LESS_EQUAL_0:
            return "<=0";
        case EQUAL_0:
            return "=0";
        case NEQUAL_0:
            return "!=0";
        case GREATER_EQUAL_0:
            return ">=0";
        case GREATER_0:
            return ">0";
        default:
            return "";
    }
    return "";
}

static inline float makeQNaN32()
{
    return fromRep32(0x7fc00000U);
}

static inline double makeQNaN64()
{
    return fromRep64(0x7ff8000000000000UL);
}

static inline long double makeQNaN128()
{
    return fromRep128(0x7fff800000000000UL, 0x0UL);
}

static inline float makeNaN32(uint32_t rand)
{
    return fromRep32(0x7f800000U | (rand & 0x7fffffU));
}

static inline double makeNaN64(uint64_t rand)
{
    return fromRep64(0x7ff0000000000000UL | (rand & 0xfffffffffffffUL));
}

static inline long double makeNaN128(uint64_t rand)
{
    return fromRep128(0x7fff000000000000UL | (rand & 0xffffffffffffUL), 0x0UL);
}

static inline float makeInf32()
{
    return fromRep32(0x7f800000U);
}

static inline double makeInf64()
{
    return fromRep64(0x7ff0000000000000UL);
}

static inline long double makeInf128()
{
    return fromRep128(0x7fff000000000000UL, 0x0UL);
}
