//===- PCHBitCodes.h - Enum values for the PCH bitcode format ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines Bitcode enum values for Clang precompiled header files.
//
// The enum values defined in this file should be considered permanent.  If
// new features are added, they should have values added at the end of the
// respective lists.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_FRONTEND_PCHBITCODES_H
#define LLVM_CLANG_FRONTEND_PCHBITCODES_H

#include "llvm/Bitcode/BitCodes.h"
#include "llvm/Support/DataTypes.h"

namespace clang {
  namespace pch {
    /// \brief An ID number that refers to a declaration in a PCH file.
    ///
    /// The ID numbers of types are consecutive (in order of
    /// discovery) and start at 2. 0 is reserved for NULL, and 1 is
    /// reserved for the translation unit declaration.
    typedef uint32_t DeclID;

    /// \brief An ID number that refers to a type in a PCH file.
    ///
    /// The ID of a type is partitioned into two parts: the lower
    /// three bits are used to store the const/volatile/restrict
    /// qualifiers (as with QualType) and the upper bits provide a
    /// type index. The type index values are partitioned into two
    /// sets. The values below NUM_PREDEF_TYPE_IDs are predefined type
    /// IDs (based on the PREDEF_TYPE_*_ID constants), with 0 as a
    /// placeholder for "no type". Values from NUM_PREDEF_TYPE_IDs are
    /// other types that have serialized representations.
    typedef uint32_t TypeID;

    /// \brief Describes the various kinds of blocks that occur within
    /// a PCH file.
    enum BlockIDs {
      /// \brief The PCH block, which acts as a container around the
      /// full PCH block.
      PCH_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID,

      /// \brief The block containing information about the source
      /// manager.
      SOURCE_MANAGER_BLOCK_ID,

      /// \brief The block containing information about the
      /// preprocessor.
      PREPROCESSOR_BLOCK_ID,

      /// \brief The block containing the definitions of all of the
      /// types used within the PCH file.
      TYPES_BLOCK_ID,

      /// \brief The block containing the definitions of all of the
      /// declarations stored in the PCH file.
      DECLS_BLOCK_ID
    };

    /// \brief Record types that occur within the PCH block itself.
    enum PCHRecordTypes {
      /// \brief Offset of each type within the types block.
      ///
      /// The TYPE_OFFSET constant describes the record that occurs
      /// within the block identified by TYPE_OFFSETS_BLOCK_ID within
      /// the PCH file. The record itself is an array of offsets that
      /// point into the types block (identified by TYPES_BLOCK_ID in
      /// the PCH file). The index into the array is based on the ID
      /// of a type. For a given type ID @c T, the lower three bits of
      /// @c T are its qualifiers (const, volatile, restrict), as in
      /// the QualType class. The upper bits, after being shifted and
      /// subtracting NUM_PREDEF_TYPE_IDS, are used to index into the
      /// TYPE_OFFSET block to determine the offset of that type's
      /// corresponding record within the TYPES_BLOCK_ID block.
      TYPE_OFFSET = 1,
      
      /// \brief Record code for the offsets of each decl.
      ///
      /// The DECL_OFFSET constant describes the record that occurs
      /// within the block identifier by DECL_OFFSETS_BLOCK_ID within
      /// the PCH file. The record itself is an array of offsets that
      /// point into the declarations block (identified by
      /// DECLS_BLOCK_ID). The declaration ID is an index into this
      /// record, after subtracting one to account for the use of
      /// declaration ID 0 for a NULL declaration pointer. Index 0 is
      /// reserved for the translation unit declaration.
      DECL_OFFSET = 2,

      /// \brief Record code for the language options table.
      ///
      /// The record with this code contains the contents of the
      /// LangOptions structure. We serialize the entire contents of
      /// the structure, and let the reader decide which options are
      /// actually important to check.
      LANGUAGE_OPTIONS = 3,

      /// \brief Record code for the target triple used to build the
      /// PCH file.
      TARGET_TRIPLE = 4
    };

    /// \brief Record types used within a source manager block.
    enum SourceManagerRecordTypes {
      /// \brief Describes a source location entry (SLocEntry) for a
      /// file.
      SM_SLOC_FILE_ENTRY = 1,
      /// \brief Describes a source location entry (SLocEntry) for a
      /// buffer.
      SM_SLOC_BUFFER_ENTRY = 2,
      /// \brief Describes a blob that contains the data for a buffer
      /// entry. This kind of record always directly follows a
      /// SM_SLOC_BUFFER_ENTRY record.
      SM_SLOC_BUFFER_BLOB = 3,
      /// \brief Describes a source location entry (SLocEntry) for a
      /// macro instantiation.
      SM_SLOC_INSTANTIATION_ENTRY = 4
    };
    
    /// \brief Record types used within a preprocessor block.
    enum PreprocessorRecordTypes {
      // The macros in the PP section are a PP_MACRO_* instance followed by a
      // list of PP_TOKEN instances for each token in the definition.

      // FIXME: TEMP HACK UNTIL WE HAVE IDENTIFIER INFO IDs.
      PP_MACRO_NAME = 4,

      /// \brief An object-like macro definition.
      /// [PP_MACRO_OBJECT_LIKE, IdentInfoID, SLoc, IsUsed]
      PP_MACRO_OBJECT_LIKE = 1,

      /// \brief A function-like macro definition.
      /// [PP_MACRO_FUNCTION_LIKE, <ObjectLikeStuff>, IsC99Varargs, IsGNUVarars,
      ///  NumArgs, ArgIdentInfoID* ]
      PP_MACRO_FUNCTION_LIKE = 2,
      
      /// \brief Describes one token.
      /// [PPTOKEN, SLoc, Length, IdentInfoID, Kind, Flags]
      PP_TOKEN = 3
    };
    

    /// \defgroup PCHAST Precompiled header AST constants
    ///
    /// The constants in this group describe various components of the
    /// abstract syntax tree within a precompiled header.
    ///
    /// @{

    /// \brief Predefined type IDs.
    ///
    /// These type IDs correspond to predefined types in the AST
    /// context, such as built-in types (int) and special place-holder
    /// types (the <overload> and <dependent> type markers). Such
    /// types are never actually serialized, since they will be built
    /// by the AST context when it is created.
    enum PredefinedTypeIDs {
      /// \brief The NULL type.
      PREDEF_TYPE_NULL_ID       = 0,
      /// \brief The void type.
      PREDEF_TYPE_VOID_ID       = 1,
      /// \brief The 'bool' or '_Bool' type.
      PREDEF_TYPE_BOOL_ID       = 2,
      /// \brief The 'char' type, when it is unsigned.
      PREDEF_TYPE_CHAR_U_ID     = 3,
      /// \brief The 'unsigned char' type.
      PREDEF_TYPE_UCHAR_ID      = 4,
      /// \brief The 'unsigned short' type.
      PREDEF_TYPE_USHORT_ID     = 5,
      /// \brief The 'unsigned int' type.
      PREDEF_TYPE_UINT_ID       = 6,
      /// \brief The 'unsigned long' type.
      PREDEF_TYPE_ULONG_ID      = 7,
      /// \brief The 'unsigned long long' type.
      PREDEF_TYPE_ULONGLONG_ID  = 8,
      /// \brief The 'char' type, when it is signed.
      PREDEF_TYPE_CHAR_S_ID     = 9,
      /// \brief The 'signed char' type.
      PREDEF_TYPE_SCHAR_ID      = 10,
      /// \brief The C++ 'wchar_t' type.
      PREDEF_TYPE_WCHAR_ID      = 11,
      /// \brief The (signed) 'short' type.
      PREDEF_TYPE_SHORT_ID      = 12,
      /// \brief The (signed) 'int' type.
      PREDEF_TYPE_INT_ID        = 13,
      /// \brief The (signed) 'long' type.
      PREDEF_TYPE_LONG_ID       = 14,
      /// \brief The (signed) 'long long' type.
      PREDEF_TYPE_LONGLONG_ID   = 15,
      /// \brief The 'float' type.
      PREDEF_TYPE_FLOAT_ID      = 16,
      /// \brief The 'double' type.
      PREDEF_TYPE_DOUBLE_ID     = 17,
      /// \brief The 'long double' type.
      PREDEF_TYPE_LONGDOUBLE_ID = 18,
      /// \brief The placeholder type for overloaded function sets.
      PREDEF_TYPE_OVERLOAD_ID   = 19,
      /// \brief The placeholder type for dependent types.
      PREDEF_TYPE_DEPENDENT_ID  = 20
    };

    /// \brief The number of predefined type IDs that are reserved for
    /// the PREDEF_TYPE_* constants.
    ///
    /// Type IDs for non-predefined types will start at
    /// NUM_PREDEF_TYPE_IDs.
    const unsigned NUM_PREDEF_TYPE_IDS = 100;

    /// \brief Record codes for each kind of type.
    ///
    /// These constants describe the records that can occur within a
    /// block identified by TYPES_BLOCK_ID in the PCH file. Each
    /// constant describes a record for a specific type class in the
    /// AST.
    enum TypeCode {
      /// \brief An ExtQualType record.
      TYPE_EXT_QUAL                 = 1,
      /// \brief A FixedWidthIntType record.
      TYPE_FIXED_WIDTH_INT          = 2,
      /// \brief A ComplexType record.
      TYPE_COMPLEX                  = 3,
      /// \brief A PointerType record.
      TYPE_POINTER                  = 4,
      /// \brief A BlockPointerType record.
      TYPE_BLOCK_POINTER            = 5,
      /// \brief An LValueReferenceType record.
      TYPE_LVALUE_REFERENCE         = 6,
      /// \brief An RValueReferenceType record.
      TYPE_RVALUE_REFERENCE         = 7,
      /// \brief A MemberPointerType record.
      TYPE_MEMBER_POINTER           = 8,
      /// \brief A ConstantArrayType record.
      TYPE_CONSTANT_ARRAY           = 9,
      /// \brief An IncompleteArrayType record.
      TYPE_INCOMPLETE_ARRAY         = 10,
      /// \brief A VariableArrayType record.
      TYPE_VARIABLE_ARRAY           = 11,
      /// \brief A VectorType record.
      TYPE_VECTOR                   = 12,
      /// \brief An ExtVectorType record.
      TYPE_EXT_VECTOR               = 13,
      /// \brief A FunctionNoProtoType record.
      TYPE_FUNCTION_NO_PROTO        = 14,
      /// \brief A FunctionProtoType record.
      TYPE_FUNCTION_PROTO           = 15,
      /// \brief A TypedefType record.
      TYPE_TYPEDEF                  = 16,
      /// \brief A TypeOfExprType record.
      TYPE_TYPEOF_EXPR              = 17,
      /// \brief A TypeOfType record.
      TYPE_TYPEOF                   = 18,
      /// \brief A RecordType record.
      TYPE_RECORD                   = 19,
      /// \brief An EnumType record.
      TYPE_ENUM                     = 20,
      /// \brief An ObjCInterfaceType record.
      TYPE_OBJC_INTERFACE           = 21,
      /// \brief An ObjCQualifiedInterfaceType record.
      TYPE_OBJC_QUALIFIED_INTERFACE = 22,
      /// \brief An ObjCQualifiedIdType record.
      TYPE_OBJC_QUALIFIED_ID        = 23,
      /// \brief An ObjCQualifiedClassType record.
      TYPE_OBJC_QUALIFIED_CLASS     = 24
    };

    /// \brief Record code for the offsets of each type.
    ///

    /// \brief Record codes for each kind of declaration.
    ///
    /// These constants describe the records that can occur within a
    /// declarations block (identified by DECLS_BLOCK_ID). Each
    /// constant describes a record for a specific declaration class
    /// in the AST.
    enum DeclCode {
      /// \brief A TranslationUnitDecl record.
      DECL_TRANSLATION_UNIT = 1,
      /// \brief A TypedefDecl record.
      DECL_TYPEDEF,
      /// \brief A VarDecl record.
      DECL_VAR,
      /// \brief A record that stores the set of declarations that are
      /// lexically stored within a given DeclContext.
      ///
      /// The record itself is an array of declaration IDs, in the
      /// order in which those declarations were added to the
      /// declaration context. This data is used when iterating over
      /// the contents of a DeclContext, e.g., via
      /// DeclContext::decls_begin()/DeclContext::decls_end().
      DECL_CONTEXT_LEXICAL,
      /// \brief A record that stores the set of declarations that are
      /// visible from a given DeclContext.
      ///
      /// The record itself stores a set of mappings, each of which
      /// associates a declaration name with one or more declaration
      /// IDs. This data is used when performing qualified name lookup
      /// into a DeclContext via DeclContext::lookup.
      DECL_CONTEXT_VISIBLE
    };
    /// @}
  }
} // end namespace clang

#endif
