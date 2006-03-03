//===- ValueTypes.td - ValueType definitions ---------------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Value types - These values correspond to the register types defined in the
// ValueTypes.h file.  If you update anything here, you must update it there as
// well!
//
//===----------------------------------------------------------------------===//

class ValueType<int size, int value> {
  string Namespace = "MVT";
  int Size = size;
  int Value = value;
}

def OtherVT: ValueType<0  ,  0>;   // "Other" value
def i1     : ValueType<1  ,  1>;   // One bit boolean value
def i8     : ValueType<8  ,  2>;   // 8-bit integer value
def i16    : ValueType<16 ,  3>;   // 16-bit integer value
def i32    : ValueType<32 ,  4>;   // 32-bit integer value
def i64    : ValueType<64 ,  5>;   // 64-bit integer value
def i128   : ValueType<128,  6>;   // 128-bit integer value
def f32    : ValueType<32 ,  7>;   // 32-bit floating point value
def f64    : ValueType<64 ,  8>;   // 64-bit floating point value
def f80    : ValueType<80 ,  9>;   // 80-bit floating point value
def f128   : ValueType<128, 10>;   // 128-bit floating point value
def FlagVT : ValueType<0  , 11>;   // Condition code or machine flag
def isVoid : ValueType<0  , 12>;   // Produces no value
def Vector : ValueType<0  , 13>;   // Abstract vector value
def v8i8   : ValueType<64 , 14>;   //  8 x i8  vector value
def v4i16  : ValueType<64 , 15>;   //  4 x i16 vector value
def v2i32  : ValueType<64 , 16>;   //  2 x i32 vector value
def v16i8  : ValueType<128, 17>;   // 16 x i8  vector value
def v8i16  : ValueType<128, 18>;   //  8 x i16 vector value
def v4i32  : ValueType<128, 19>;   //  4 x i32 vector value
def v2i64  : ValueType<128, 20>;   //  2 x i64 vector value
def v2f32  : ValueType<64,  21>;   //  2 x f32 vector value
def v4f32  : ValueType<128, 22>;   //  4 x f32 vector value
def v2f64  : ValueType<128, 23>;   //  2 x f64 vector value
