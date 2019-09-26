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

namespace llvm {
class StringRef;
template <typename>
class ArrayRef;
class hash_code;
} // namespace llvm

namespace fir {

class FIROpsDialect;

using KindTy = int;

namespace detail {
struct BoxTypeStorage;
struct BoxCharTypeStorage;
struct BoxProcTypeStorage;
struct CharacterTypeStorage;
struct CplxTypeStorage;
struct DimsTypeStorage;
struct FieldTypeStorage;
struct HeapTypeStorage;
struct IntTypeStorage;
struct LogicalTypeStorage;
struct PointerTypeStorage;
struct RealTypeStorage;
struct RecordTypeStorage;
struct ReferenceTypeStorage;
struct SequenceTypeStorage;
struct TypeDescTypeStorage;
} // namespace detail

enum TypeKind {
  // The enum starts at the range reserved for this dialect.
  FIR_TYPE = mlir::Type::FIRST_FIR_TYPE,
  FIR_BOX,
  FIR_BOXCHAR,
  FIR_BOXPROC,
  FIR_CHARACTER, // intrinsic
  FIR_COMPLEX,   // intrinsic
  FIR_DERIVED,   // derived
  FIR_DIMS,
  FIR_FIELD,
  FIR_HEAP,
  FIR_INT,     // intrinsic
  FIR_LOGICAL, // intrinsic
  FIR_POINTER,
  FIR_REAL, // intrinsic
  FIR_REFERENCE,
  FIR_SEQUENCE,
  FIR_TYPEDESC,
};

bool isa_fir_type(mlir::Type);
bool isa_std_type(mlir::Type t);
bool isa_fir_or_std_type(mlir::Type t);

template <typename A, unsigned Id>
struct IntrinsicTypeMixin {
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() { return Id; }
};

class CharacterType
    : public mlir::Type::TypeBase<CharacterType, mlir::Type,
                                  detail::CharacterTypeStorage>,
      public IntrinsicTypeMixin<CharacterType, TypeKind::FIR_CHARACTER> {
public:
  using Base::Base;
  static CharacterType get(mlir::MLIRContext *ctxt, KindTy kind);
  int getSizeInBits() const;
  KindTy getFKind() const { return getSizeInBits() / 8; }
};

class IntType
    : public mlir::Type::TypeBase<IntType, mlir::Type, detail::IntTypeStorage>,
      public IntrinsicTypeMixin<IntType, TypeKind::FIR_INT> {
public:
  using Base::Base;
  static IntType get(mlir::MLIRContext *ctxt, KindTy kind);
  int getSizeInBits() const;
  KindTy getFKind() const { return getSizeInBits() / 8; }
};

class LogicalType
    : public mlir::Type::TypeBase<LogicalType, mlir::Type,
                                  detail::LogicalTypeStorage>,
      public IntrinsicTypeMixin<LogicalType, TypeKind::FIR_LOGICAL> {
public:
  using Base::Base;
  static LogicalType get(mlir::MLIRContext *ctxt, KindTy kind);
  int getSizeInBits() const;
  KindTy getFKind() const { return getSizeInBits() / 8; }
};

class RealType : public mlir::Type::TypeBase<RealType, mlir::Type,
                                             detail::RealTypeStorage>,
                 public IntrinsicTypeMixin<RealType, TypeKind::FIR_REAL> {
public:
  using Base::Base;
  static RealType get(mlir::MLIRContext *ctxt, KindTy kind);
  int getSizeInBits() const;
  KindTy getFKind() const { return getSizeInBits() / 8; }
};

class CplxType : public mlir::Type::TypeBase<CplxType, mlir::Type,
                                             detail::CplxTypeStorage>,
                 public IntrinsicTypeMixin<CplxType, TypeKind::FIR_COMPLEX> {
public:
  using Base::Base;
  static CplxType get(mlir::MLIRContext *ctxt, KindTy kind);
  int getSizeInBits() const;
  KindTy getFKind() const;
};

// FIR support types

class BoxType
    : public mlir::Type::TypeBase<BoxType, mlir::Type, detail::BoxTypeStorage> {
public:
  using Base::Base;
  static BoxType get(mlir::Type eleTy);
  static bool kindof(unsigned kind) { return kind == TypeKind::FIR_BOX; }
  mlir::Type getEleTy() const;
};

class BoxCharType : public mlir::Type::TypeBase<BoxCharType, mlir::Type,
                                                detail::BoxCharTypeStorage> {
public:
  using Base::Base;
  static BoxCharType get(mlir::MLIRContext *ctxt, KindTy kind);
  static bool kindof(unsigned kind) { return kind == TypeKind::FIR_BOXCHAR; }
  CharacterType getEleTy() const;
};

class BoxProcType : public mlir::Type::TypeBase<BoxProcType, mlir::Type,
                                                detail::BoxProcTypeStorage> {
public:
  using Base::Base;
  static BoxProcType get(mlir::Type eleTy);
  static bool kindof(unsigned kind) { return kind == TypeKind::FIR_BOXPROC; }
  mlir::Type getEleTy() const;
};

class DimsType : public mlir::Type::TypeBase<DimsType, mlir::Type,
                                             detail::DimsTypeStorage> {
public:
  using Base::Base;
  static DimsType get(mlir::MLIRContext *ctx, unsigned rank);
  static bool kindof(unsigned kind) { return kind == TypeKind::FIR_DIMS; }

  /// returns -1 if the rank is unknown
  int getRank() const;
};

class FieldType : public mlir::Type::TypeBase<FieldType, mlir::Type,
                                              detail::FieldTypeStorage> {
public:
  using Base::Base;
  static FieldType get(mlir::MLIRContext *ctxt, KindTy _ = 0);
  static bool kindof(unsigned kind) { return kind == TypeKind::FIR_FIELD; }
};

class HeapType : public mlir::Type::TypeBase<HeapType, mlir::Type,
                                             detail::HeapTypeStorage> {
public:
  using Base::Base;
  static HeapType get(mlir::Type elementType);
  static bool kindof(unsigned kind) { return kind == TypeKind::FIR_HEAP; }

  mlir::Type getEleTy() const;
};

class PointerType : public mlir::Type::TypeBase<PointerType, mlir::Type,
                                                detail::PointerTypeStorage> {
public:
  using Base::Base;
  static PointerType get(mlir::Type elementType);
  static bool kindof(unsigned kind) { return kind == TypeKind::FIR_POINTER; }

  mlir::Type getEleTy() const;
};

class ReferenceType
    : public mlir::Type::TypeBase<ReferenceType, mlir::Type,
                                  detail::ReferenceTypeStorage> {
public:
  using Base::Base;
  static ReferenceType get(mlir::Type elementType);
  static bool kindof(unsigned kind) { return kind == TypeKind::FIR_REFERENCE; }

  mlir::Type getEleTy() const;
};

class SequenceType : public mlir::Type::TypeBase<SequenceType, mlir::Type,
                                                 detail::SequenceTypeStorage> {
public:
  using Base::Base;
  using BoundInfo = int64_t;
  struct Extent {
    bool known;
    BoundInfo bound;
    explicit Extent(bool k, BoundInfo b) : known(k), bound(b) {}
  };
  using Bounds = std::vector<Extent>;
  struct Shape {
    bool known;
    Bounds bounds;
    Shape() : known(false) {}
    Shape(const Bounds &b) : known(true), bounds(b) {}
  };

  mlir::Type getEleTy() const;
  Shape getShape() const;

  static SequenceType get(const Shape &shape, mlir::Type elementType);
  static bool kindof(unsigned kind) { return kind == TypeKind::FIR_SEQUENCE; }
};

bool operator==(const SequenceType::Shape &, const SequenceType::Shape &);
llvm::hash_code hash_value(const SequenceType::Extent &);
llvm::hash_code hash_value(const SequenceType::Shape &);

class TypeDescType : public mlir::Type::TypeBase<TypeDescType, mlir::Type,
                                                 detail::TypeDescTypeStorage> {
public:
  using Base::Base;
  static TypeDescType get(mlir::Type ofType);
  static bool kindof(unsigned kind) { return kind == TypeKind::FIR_TYPEDESC; }
  mlir::Type getOfTy() const;
};

// Derived types

class RecordType : public mlir::Type::TypeBase<RecordType, mlir::Type,
                                               detail::RecordTypeStorage> {
public:
  using Base::Base;
  using TypePair = std::pair<std::string, mlir::Type>;
  using TypeList = std::vector<TypePair>;

  llvm::StringRef getName();
  TypeList getTypeList();
  TypeList getLenParamList();

  static RecordType get(mlir::MLIRContext *ctxt, llvm::StringRef name,
                        llvm::ArrayRef<TypePair> lenPList = {},
                        llvm::ArrayRef<TypePair> typeList = {});
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() { return TypeKind::FIR_DERIVED; }
};

mlir::Type parseFirType(FIROpsDialect *dialect, llvm::StringRef rawData,
                        mlir::Location loc);

} // namespace fir

#endif // FIR_TYPE_H
