// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/ids.h"

#include "toolchain/sem_ir/singleton_insts.h"

namespace Carbon::SemIR {

auto InstId::Print(llvm::raw_ostream& out) const -> void {
  out << "inst";
  if (!is_valid()) {
    IdBase::Print(out);
  } else if (is_builtin()) {
    out << builtin_inst_kind();
  } else {
    // Use the `+` as a small reminder that this is a delta, rather than an
    // absolute index.
    out << "+" << index - SingletonInstKinds.size();
  }
}

auto ConstantId::Print(llvm::raw_ostream& out, bool disambiguate) const
    -> void {
  if (!is_valid()) {
    IdBase::Print(out);
  } else if (is_template()) {
    if (disambiguate) {
      out << "templateConstant(";
    }
    out << template_inst_id();
    if (disambiguate) {
      out << ")";
    }
  } else if (is_symbolic()) {
    out << "symbolicConstant" << symbolic_index();
  } else {
    out << "runtime";
  }
}

auto RuntimeParamIndex::Print(llvm::raw_ostream& out) const -> void {
  out << "runtime_param";
  if (*this == Unknown) {
    out << "<unknown>";
  } else {
    IndexBase::Print(out);
  }
}

auto GenericInstIndex::Print(llvm::raw_ostream& out) const -> void {
  out << "genericInst";
  if (is_valid()) {
    out << (region() == Declaration ? "InDecl" : "InDef") << index();
  } else {
    out << "<invalid>";
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

auto IntKind::Print(llvm::raw_ostream& out) const -> void {
  if (*this == Unsigned) {
    out << "unsigned";
  } else if (*this == Signed) {
    out << "signed";
  } else {
    CARBON_FATAL("Invalid int kind value {0}", index);
  }
}

auto NameId::ForIdentifier(IdentifierId id) -> NameId {
  if (id.index >= 0) {
    return NameId(id.index);
  } else if (!id.is_valid()) {
    return NameId::Invalid;
  } else {
    CARBON_FATAL("Unexpected identifier ID {0}", id);
  }
}

auto NameId::Print(llvm::raw_ostream& out) const -> void {
  out << "name";
  if (*this == SelfValue) {
    out << "SelfValue";
  } else if (*this == SelfType) {
    out << "SelfType";
  } else if (*this == PeriodSelf) {
    out << "PeriodSelf";
  } else if (*this == ReturnSlot) {
    out << "ReturnSlot";
  } else if (*this == PackageNamespace) {
    out << "PackageNamespace";
  } else if (*this == Base) {
    out << "Base";
  } else {
    CARBON_CHECK(!is_valid() || index >= 0, "Unknown index {0}", index);
    IdBase::Print(out);
  }
}

auto InstBlockId::Print(llvm::raw_ostream& out) const -> void {
  if (*this == Unreachable) {
    out << "unreachable";
  } else if (*this == Empty) {
    out << "empty";
  } else if (*this == Exports) {
    out << "exports";
  } else if (*this == ImportRefs) {
    out << "import_refs";
  } else if (*this == GlobalInit) {
    out << "global_init";
  } else {
    out << "block";
    IdBase::Print(out);
  }
}

auto TypeId::Print(llvm::raw_ostream& out) const -> void {
  out << "type";
  if (*this == TypeType) {
    out << "TypeType";
  } else if (*this == AutoType) {
    out << "AutoType";
  } else if (*this == Error) {
    out << "Error";
  } else {
    out << "(";
    AsConstantId().Print(out, /*disambiguate=*/false);
    out << ")";
  }
}

auto LibraryNameId::ForStringLiteralValueId(StringLiteralValueId id)
    -> LibraryNameId {
  CARBON_CHECK(id.index >= InvalidIndex, "Unexpected library name ID {0}", id);
  if (id == StringLiteralValueId::Invalid) {
    // Prior to SemIR, we use invalid to indicate `default`.
    return LibraryNameId::Default;
  } else {
    return LibraryNameId(id.index);
  }
}

auto LibraryNameId::Print(llvm::raw_ostream& out) const -> void {
  out << "libraryName";
  if (*this == Default) {
    out << "Default";
  } else if (*this == Error) {
    out << "<error>";
  } else {
    IdBase::Print(out);
  }
}

auto LocId::Print(llvm::raw_ostream& out) const -> void {
  out << "loc_";
  if (is_node_id() || !is_valid()) {
    out << node_id();
  } else {
    out << import_ir_inst_id();
  }
}

}  // namespace Carbon::SemIR
