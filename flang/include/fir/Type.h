// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FIR_TYPE_H
#define FIR_TYPE_H

#include "mlir/IR/Types.h"
#include <variant>

namespace llvm {
class StringRef;
template<typename> class ArrayRef;
class hash_code;
}

namespace fir {

class FIROpsDialect;

using KindTy = int;

namespace detail {
struct FIRBoxTypeStorage;
struct FIRBoxCharTypeStorage;
struct FIRBoxProcTypeStorage;
struct FIRCharacterTypeStorage;
struct FIRCplxTypeStorage;
struct FIRDimsTypeStorage;
struct FIRFieldTypeStorage;
struct FIRHeapTypeStorage;
struct FIRIntTypeStorage;
struct FIRLogicalTypeStorage;
struct FIRPointerTypeStorage;
struct FIRRealTypeStorage;
struct FIRRecordTypeStorage;
struct FIRReferenceTypeStorage;
struct FIRSequenceTypeStorage;
struct FIRTypeDescTypeStorage;
}

enum FIRTypeKind {
  // The enum starts at the range reserved for this dialect.
  FIR_TYPE = mlir::Type::FIRST_FIR_TYPE,
  FIR_BOX,
  FIR_BOXCHAR,
  FIR_BOXPROC,
  FIR_CHARACTER,  // intrinsic
  FIR_COMPLEX,  // intrinsic
  FIR_DERIVED,  // derived
  FIR_DIMS,
  FIR_FIELD,
  FIR_HEAP,
  FIR_INT,  // intrinsic
  FIR_LOGICAL,  // intrinsic
  FIR_POINTER,
  FIR_REAL,  // intrinsic
  FIR_REFERENCE,
  FIR_SEQUENCE,
  FIR_TYPEDESC,
};

template<class... Ts> struct Typecase : Ts... { using Ts::operator()...; };
template<class... Ts> Typecase(Ts...)->Typecase<Ts...>;

bool isa_fir_type(mlir::Type);
bool isa_std_type(mlir::Type t);
bool isa_fir_or_std_type(mlir::Type t);

// Intrinsic types

class CharacterType : public mlir::Type::TypeBase<CharacterType, mlir::Type,
                          detail::FIRCharacterTypeStorage> {
public:
  using Base::Base;
  static CharacterType get(mlir::MLIRContext *ctxt, KindTy kind);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_CHARACTER;
  }

  int getSizeInBits() const;
  KindTy getFKind() const;
};

class IntType : public mlir::Type::TypeBase<IntType, mlir::Type,
                        detail::FIRIntTypeStorage> {
public:
  using Base::Base;
  static IntType get(mlir::MLIRContext *ctxt, KindTy kind);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_INT; }

  int getSizeInBits() const;
  KindTy getFKind() const;
};

class LogicalType : public mlir::Type::TypeBase<LogicalType, mlir::Type,
                        detail::FIRLogicalTypeStorage> {
public:
  using Base::Base;
  static LogicalType get(mlir::MLIRContext *ctxt, KindTy kind);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_LOGICAL; }

  int getSizeInBits() const;
  KindTy getFKind() const;
};

class RealType : public mlir::Type::TypeBase<RealType, mlir::Type,
                     detail::FIRRealTypeStorage> {
public:
  using Base::Base;
  static RealType get(mlir::MLIRContext *ctxt, KindTy kind);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_REAL; }

  int getSizeInBits() const;
  KindTy getFKind() const;
};

class CplxType : public mlir::Type::TypeBase<CplxType, mlir::Type,
                        detail::FIRCplxTypeStorage> {
public:
  using Base::Base;
  static CplxType get(mlir::MLIRContext *ctxt, KindTy kind);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_COMPLEX; }

  int getSizeInBits() const;
  KindTy getFKind() const;
};

// FIR support types

class BoxType : public mlir::Type::TypeBase<BoxType, mlir::Type,
                    detail::FIRBoxTypeStorage> {
public:
  using Base::Base;
  static BoxType get(mlir::Type eleTy);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_BOX; }
  mlir::Type getEleTy() const;
};

class BoxCharType : public mlir::Type::TypeBase<BoxCharType, mlir::Type,
                        detail::FIRBoxCharTypeStorage> {
public:
  using Base::Base;
  static BoxCharType get(CharacterType charTy);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_BOXCHAR; }
  CharacterType getEleTy() const;
};

class BoxProcType : public mlir::Type::TypeBase<BoxProcType, mlir::Type,
                        detail::FIRBoxProcTypeStorage> {
public:
  using Base::Base;
  static BoxProcType get(mlir::Type eleTy);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_BOXPROC; }
  mlir::Type getEleTy() const;
};

class DimsType : public mlir::Type::TypeBase<DimsType, mlir::Type,
                     detail::FIRDimsTypeStorage> {
public:
  using Base::Base;
  static DimsType get(mlir::MLIRContext *ctx, unsigned rank);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_DIMS; }

  /// returns -1 if the rank is unknown
  int getRank() const;
};

class FieldType : public mlir::Type::TypeBase<FieldType, mlir::Type,
                      detail::FIRFieldTypeStorage> {
public:
  using Base::Base;
  static FieldType get(mlir::MLIRContext *ctxt, KindTy _ = 0);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_FIELD; }
};

class HeapType : public mlir::Type::TypeBase<HeapType, mlir::Type,
                     detail::FIRHeapTypeStorage> {
public:
  using Base::Base;
  static HeapType get(mlir::Type elementType);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_HEAP; }

  mlir::Type getEleTy() const;
};

class PointerType : public mlir::Type::TypeBase<PointerType, mlir::Type,
                        detail::FIRPointerTypeStorage> {
public:
  using Base::Base;
  static PointerType get(mlir::Type elementType);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_POINTER; }

  mlir::Type getEleTy() const;
};

class ReferenceType : public mlir::Type::TypeBase<ReferenceType, mlir::Type,
                          detail::FIRReferenceTypeStorage> {
public:
  using Base::Base;
  static ReferenceType get(mlir::Type elementType);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_REFERENCE;
  }

  mlir::Type getEleTy() const;
};

class SequenceType : public mlir::Type::TypeBase<SequenceType, mlir::Type,
                         detail::FIRSequenceTypeStorage> {
public:
  using Base::Base;
  struct Unknown {};
  using BoundInfo = int64_t;
  using Extent = std::variant<Unknown, BoundInfo>;
  using Bounds = std::vector<Extent>;
  using Shape = std::variant<Unknown, Bounds>;

  mlir::Type getEleTy() const;
  Shape getShape() const;

  static SequenceType get(const Shape &shape, mlir::Type elementType);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_SEQUENCE;
  }
};

bool operator==(const SequenceType::Shape &, const SequenceType::Shape &);
llvm::hash_code hash_value(const SequenceType::Extent &);
llvm::hash_code hash_value(const SequenceType::Shape &);


class TypeDescType : public mlir::Type::TypeBase<TypeDescType, mlir::Type,
                         detail::FIRTypeDescTypeStorage> {
public:
  using Base::Base;
  static TypeDescType get(mlir::Type ofType);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_TYPEDESC;
  }
  mlir::Type getOfTy() const;
};

// Derived types

class RecordType : public mlir::Type::TypeBase<RecordType, mlir::Type,
                       detail::FIRRecordTypeStorage> {
public:
  using Base::Base;
  using TypePair = std::pair<std::string, mlir::Type>;
  using TypeList = std::vector<TypePair>;
  using LenPPair = std::pair<std::string, mlir::Type>;
  using LenPList = std::vector<TypePair>;

  llvm::StringRef getName();
  TypeList getTypeList();
  LenPList getLenParamList();

  static RecordType get(mlir::MLIRContext *ctxt, llvm::StringRef name,
      llvm::ArrayRef<LenPPair> lenPList = {},
      llvm::ArrayRef<TypePair> typeList = {});
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_DERIVED; }
};

mlir::Type parseFirType(
    FIROpsDialect *dialect, llvm::StringRef rawData, mlir::Location loc);

}  // fir

#endif  // FIR_TYPE_H
