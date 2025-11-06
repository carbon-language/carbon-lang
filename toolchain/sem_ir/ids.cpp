// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/ids.h"

#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/NativeFormatting.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/sem_ir/singleton_insts.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

auto InstId::Print(llvm::raw_ostream& out) const -> void {
  if (IsSingletonInstId(*this)) {
    out << Label << "(" << SingletonInstKinds[index] << ")";
  } else {
    IdBase::Print(out);
  }
}

auto ConstantId::Print(llvm::raw_ostream& out, bool disambiguate) const
    -> void {
  if (!has_value()) {
    IdBase::Print(out);
    return;
  }
  if (is_concrete()) {
    if (disambiguate) {
      out << "concrete_constant(";
    }
    out << concrete_inst_id();
    if (disambiguate) {
      out << ")";
    }
  } else if (is_symbolic()) {
    out << symbolic_id();
  } else {
    CARBON_CHECK(!is_constant());
    out << "runtime";
  }
}

auto CheckIRId::Print(llvm::raw_ostream& out) const -> void {
  if (*this == Cpp) {
    out << Label << "(Cpp)";
  } else {
    IdBase::Print(out);
  }
}

auto GenericInstIndex::Print(llvm::raw_ostream& out) const -> void {
  out << "generic_inst";
  if (has_value()) {
    out << (region() == Declaration ? "_in_decl" : "_in_def") << index();
  } else {
    out << "<none>";
  }
}

auto ImportIRId::Print(llvm::raw_ostream& out) const -> void {
  if (*this == ApiForImpl) {
    out << Label << "(ApiForImpl)";
  } else if (*this == Cpp) {
    out << Label << "(Cpp)";
  } else {
    IdBase::Print(out);
  }
}

auto BoolValue::Print(llvm::raw_ostream& out) const -> void {
  if (*this == False) {
    out << "false";
  } else if (*this == True) {
    out << "true";
  } else {
    CARBON_FATAL("Invalid bool value {0}", index);
  }
}

auto CharId::Print(llvm::raw_ostream& out) const -> void {
  // TODO: If we switch to C++23, `std::format("`{:?}`")` might be a better
  // choice.
  out << "U+";
  llvm::write_hex(out, index, llvm::HexPrintStyle::Upper, 4);
}

auto IntKind::Print(llvm::raw_ostream& out) const -> void {
  if (*this == Unsigned) {
    out << "unsigned";
  } else if (*this == Signed) {
    out << "signed";
  } else {
    CARBON_FATAL("Invalid int kind value {0}", index);
  }
}

static auto FloatKindToStringLiteral(FloatKind kind) -> llvm::StringLiteral {
  switch (kind.index) {
    case FloatKind::None.index:
      return "<none>";
    case FloatKind::Binary16.index:
      return "f16";
    case FloatKind::Binary32.index:
      return "f32";
    case FloatKind::Binary64.index:
      return "f64";
    case FloatKind::Binary128.index:
      return "f128";
    case FloatKind::BFloat16.index:
      return "f16_brain";
    case FloatKind::X87Float80.index:
      return "f80_x87";
    case FloatKind::PPCFloat128.index:
      return "f128_ppc";
    default:
      return "<invalid>";
  }
}

auto FloatKind::Print(llvm::raw_ostream& out) const -> void {
  out << FloatKindToStringLiteral(*this);
}

auto FloatKind::Semantics() const -> const llvm::fltSemantics& {
  switch (this->index) {
    case Binary16.index:
      return llvm::APFloat::IEEEhalf();
    case Binary32.index:
      return llvm::APFloat::IEEEsingle();
    case Binary64.index:
      return llvm::APFloat::IEEEdouble();
    case Binary128.index:
      return llvm::APFloat::IEEEquad();
    case BFloat16.index:
      return llvm::APFloat::BFloat();
    case X87Float80.index:
      return llvm::APFloat::x87DoubleExtended();
    case PPCFloat128.index:
      return llvm::APFloat::PPCDoubleDouble();
    default:
      CARBON_FATAL("Unexpected float kind {0}", *this);
  }
}

// Double-check the special value mapping and constexpr evaluation.
static_assert(NameId::SpecialNameId::Vptr == *NameId::Vptr.AsSpecialNameId());

auto NameId::ForIdentifier(IdentifierId id) -> NameId {
  if (id.index >= 0) {
    return NameId(id.index);
  } else if (!id.has_value()) {
    return NameId::None;
  } else {
    CARBON_FATAL("Unexpected identifier ID {0}", id);
  }
}

auto NameId::ForPackageName(PackageNameId id) -> NameId {
  if (auto identifier_id = id.AsIdentifierId(); identifier_id.has_value()) {
    return ForIdentifier(identifier_id);
  } else if (id == PackageNameId::Core) {
    return NameId::Core;
  } else if (id == PackageNameId::Cpp) {
    return NameId::Cpp;
  } else if (!id.has_value()) {
    return NameId::None;
  } else {
    CARBON_FATAL("Unexpected package ID {0}", id);
  }
}

auto NameId::Print(llvm::raw_ostream& out) const -> void {
  if (!has_value() || index >= 0) {
    IdBase::Print(out);
    return;
  }
  out << Label << "(";
  auto special_name_id = AsSpecialNameId();
  CARBON_CHECK(special_name_id, "Unknown index {0}", index);

  switch (*special_name_id) {
#define CARBON_SPECIAL_NAME_ID_FOR_PRINT(Name) \
  case SpecialNameId::Name:                    \
    out << #Name;                              \
    break;
    CARBON_SPECIAL_NAME_ID(CARBON_SPECIAL_NAME_ID_FOR_PRINT)
#undef CARBON_SPECIAL_NAME_ID_FOR_PRINT
  }
  out << ")";
}

auto InstBlockId::Print(llvm::raw_ostream& out) const -> void {
  if (*this == Unreachable) {
    out << "unreachable";
  } else if (*this == Empty) {
    out << Label << "_empty";
  } else if (*this == Exports) {
    out << "exports";
  } else if (*this == Imports) {
    out << "imports";
  } else if (*this == GlobalInit) {
    out << "global_init";
  } else {
    IdBase::Print(out);
  }
}

auto TypeId::Print(llvm::raw_ostream& out) const -> void {
  out << Label << "(";
  if (*this == TypeType::TypeId) {
    out << "TypeType";
  } else if (*this == ErrorInst::TypeId) {
    out << "Error";
  } else {
    AsConstantId().Print(out, /*disambiguate=*/false);
  }
  out << ")";
}

auto LibraryNameId::ForStringLiteralValueId(StringLiteralValueId id)
    -> LibraryNameId {
  CARBON_CHECK(id.index >= NoneIndex, "Unexpected library name ID {0}", id);
  if (id == StringLiteralValueId::None) {
    // Prior to SemIR, we use `None` to indicate `default`.
    return LibraryNameId::Default;
  } else {
    return LibraryNameId(id.index);
  }
}

auto LibraryNameId::Print(llvm::raw_ostream& out) const -> void {
  if (*this == Default) {
    out << Label << "Default";
  } else if (*this == Error) {
    out << Label << "<error>";
  } else {
    IdBase::Print(out);
  }
}

auto LocId::Print(llvm::raw_ostream& out) const -> void {
  switch (kind()) {
    case Kind::None:
      IdBase::Print(out);
      break;
    case Kind::ImportIRInstId:
      out << Label << "_" << import_ir_inst_id();
      break;
    case Kind::InstId:
      out << Label << "_" << inst_id();
      break;
    case Kind::NodeId:
      out << Label << "_" << node_id();
      if (is_desugared()) {
        out << "_desugared";
      }
      break;
  }
}

}  // namespace Carbon::SemIR
