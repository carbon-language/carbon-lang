//===- OperationKinds.h - Operation enums -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the different kinds of operations that can be
// performed by various expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_OPERATION_KINDS_H
#define LLVM_CLANG_AST_OPERATION_KINDS_H

namespace clang {
  
/// CastKind - the kind of cast this represents.
enum CastKind {
  /// CK_Unknown - Unknown cast kind.
  /// FIXME: The goal is to get rid of this and make all casts have a
  /// kind so that the AST client doesn't have to try to figure out what's
  /// going on.
  CK_Unknown,

  /// CK_BitCast - Used for reinterpret_cast.
  CK_BitCast,

  /// CK_LValueBitCast - Used for reinterpret_cast of expressions to
  /// a reference type.
  CK_LValueBitCast,
    
  /// CK_NoOp - Used for const_cast.
  CK_NoOp,

  /// CK_BaseToDerived - Base to derived class casts.
  CK_BaseToDerived,

  /// CK_DerivedToBase - Derived to base class casts.
  CK_DerivedToBase,

  /// CK_UncheckedDerivedToBase - Derived to base class casts that
  /// assume that the derived pointer is not null.
  CK_UncheckedDerivedToBase,

  /// CK_Dynamic - Dynamic cast.
  CK_Dynamic,

  /// CK_ToUnion - Cast to union (GCC extension).
  CK_ToUnion,

  /// CK_ArrayToPointerDecay - Array to pointer decay.
  CK_ArrayToPointerDecay,

  /// CK_FunctionToPointerDecay - Function to pointer decay.
  CK_FunctionToPointerDecay,

  /// CK_NullToPointer - Null pointer to pointer.
  CK_NullToPointer,

  /// CK_NullToMemberPointer - Null pointer to member pointer.
  CK_NullToMemberPointer,

  /// CK_BaseToDerivedMemberPointer - Member pointer in base class to
  /// member pointer in derived class.
  CK_BaseToDerivedMemberPointer,

  /// CK_DerivedToBaseMemberPointer - Member pointer in derived class to
  /// member pointer in base class.
  CK_DerivedToBaseMemberPointer,
    
  /// CK_UserDefinedConversion - Conversion using a user defined type
  /// conversion function.
  CK_UserDefinedConversion,

  /// CK_ConstructorConversion - Conversion by constructor
  CK_ConstructorConversion,
    
  /// CK_IntegralToPointer - Integral to pointer
  CK_IntegralToPointer,
    
  /// CK_PointerToIntegral - Pointer to integral
  CK_PointerToIntegral,
    
  /// CK_ToVoid - Cast to void.
  CK_ToVoid,
    
  /// CK_VectorSplat - Casting from an integer/floating type to an extended
  /// vector type with the same element type as the src type. Splats the 
  /// src expression into the destination expression.
  CK_VectorSplat,
    
  /// CK_IntegralCast - Casting between integral types of different size.
  CK_IntegralCast,

  /// CK_IntegralToFloating - Integral to floating point.
  CK_IntegralToFloating,
    
  /// CK_FloatingToIntegral - Floating point to integral.
  CK_FloatingToIntegral,
    
  /// CK_FloatingCast - Casting between floating types of different size.
  CK_FloatingCast,
    
  /// CK_MemberPointerToBoolean - Member pointer to boolean
  CK_MemberPointerToBoolean,

  /// CK_AnyPointerToObjCPointerCast - Casting any pointer to objective-c 
  /// pointer
  CK_AnyPointerToObjCPointerCast,

  /// CK_AnyPointerToBlockPointerCast - Casting any pointer to block 
  /// pointer
  CK_AnyPointerToBlockPointerCast,

  /// \brief Converting between two Objective-C object types, which
  /// can occur when performing reference binding to an Objective-C
  /// object.
  CK_ObjCObjectLValueCast
};


enum BinaryOperatorKind {
  // Operators listed in order of precedence.
  // Note that additions to this should also update the StmtVisitor class.
  BO_PtrMemD, BO_PtrMemI,       // [C++ 5.5] Pointer-to-member operators.
  BO_Mul, BO_Div, BO_Rem,       // [C99 6.5.5] Multiplicative operators.
  BO_Add, BO_Sub,               // [C99 6.5.6] Additive operators.
  BO_Shl, BO_Shr,               // [C99 6.5.7] Bitwise shift operators.
  BO_LT, BO_GT, BO_LE, BO_GE,   // [C99 6.5.8] Relational operators.
  BO_EQ, BO_NE,                 // [C99 6.5.9] Equality operators.
  BO_And,                       // [C99 6.5.10] Bitwise AND operator.
  BO_Xor,                       // [C99 6.5.11] Bitwise XOR operator.
  BO_Or,                        // [C99 6.5.12] Bitwise OR operator.
  BO_LAnd,                      // [C99 6.5.13] Logical AND operator.
  BO_LOr,                       // [C99 6.5.14] Logical OR operator.
  BO_Assign, BO_MulAssign,      // [C99 6.5.16] Assignment operators.
  BO_DivAssign, BO_RemAssign,
  BO_AddAssign, BO_SubAssign,
  BO_ShlAssign, BO_ShrAssign,
  BO_AndAssign, BO_XorAssign,
  BO_OrAssign,
  BO_Comma                      // [C99 6.5.17] Comma operator.
};

enum UnaryOperatorKind {
  // Note that additions to this should also update the StmtVisitor class.
  UO_PostInc, UO_PostDec, // [C99 6.5.2.4] Postfix increment and decrement
  UO_PreInc, UO_PreDec,   // [C99 6.5.3.1] Prefix increment and decrement
  UO_AddrOf, UO_Deref,    // [C99 6.5.3.2] Address and indirection
  UO_Plus, UO_Minus,      // [C99 6.5.3.3] Unary arithmetic
  UO_Not, UO_LNot,        // [C99 6.5.3.3] Unary arithmetic
  UO_Real, UO_Imag,       // "__real expr"/"__imag expr" Extension.
  UO_Extension            // __extension__ marker.
};

}

#endif
