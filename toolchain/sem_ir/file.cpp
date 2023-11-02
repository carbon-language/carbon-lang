// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/file.h"

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::SemIR {

auto ValueRepresentation::Print(llvm::raw_ostream& out) const -> void {
  out << "{kind: ";
  switch (kind) {
    case Unknown:
      out << "unknown";
      break;
    case None:
      out << "none";
      break;
    case Copy:
      out << "copy";
      break;
    case Pointer:
      out << "pointer";
      break;
    case Custom:
      out << "custom";
      break;
  }
  out << ", type: " << type_id << "}";
}

auto TypeInfo::Print(llvm::raw_ostream& out) const -> void {
  out << "{inst: " << inst_id << ", value_rep: " << value_representation << "}";
}

File::File(SharedValueStores& value_stores)
    : value_stores_(&value_stores),
      filename_("<builtins>"),
      // Builtins are always the first IR, even when self-referential.
      cross_reference_irs_({this}),
      type_blocks_(allocator_),
      inst_blocks_(allocator_) {
  // Default entry for InstBlockId::Empty.
  inst_blocks_.AddDefaultValue();

  insts_.Reserve(BuiltinKind::ValidCount);

  // Error uses a self-referential type so that it's not accidentally treated as
  // a normal type. Every other builtin is a type, including the
  // self-referential TypeType.
#define CARBON_SEM_IR_BUILTIN_KIND(Name, ...)                         \
  insts_.AddInNoBlock(Builtin{BuiltinKind::Name == BuiltinKind::Error \
                                  ? TypeId::Error                     \
                                  : TypeId::TypeType,                 \
                              BuiltinKind::Name});
#include "toolchain/sem_ir/builtin_kind.def"

  CARBON_CHECK(insts_.size() == BuiltinKind::ValidCount)
      << "Builtins should produce " << BuiltinKind::ValidCount
      << " insts, actual: " << insts_.size();
}

File::File(SharedValueStores& value_stores, std::string filename,
           const File* builtins)
    : value_stores_(&value_stores),
      filename_(std::move(filename)),
      // Builtins are always the first IR.
      cross_reference_irs_({builtins}),
      type_blocks_(allocator_),
      inst_blocks_(allocator_) {
  CARBON_CHECK(builtins != nullptr);
  CARBON_CHECK(builtins->cross_reference_irs_[0] == builtins)
      << "Not called with builtins!";

  // Default entry for InstBlockId::Empty.
  inst_blocks_.AddDefaultValue();

  // Copy builtins over.
  insts_.Reserve(BuiltinKind::ValidCount);
  static constexpr auto BuiltinIR = CrossReferenceIRId(0);
  for (auto [i, inst] : llvm::enumerate(builtins->insts_.array_ref())) {
    // We can reuse builtin type IDs because they're special-cased values.
    insts_.AddInNoBlock(
        CrossReference{inst.type_id(), BuiltinIR, SemIR::InstId(i)});
  }
}

auto File::Verify() const -> ErrorOr<Success> {
  // Invariants don't necessarily hold for invalid IR.
  if (has_errors_) {
    return Success();
  }

  // Check that every code block has a terminator sequence that appears at the
  // end of the block.
  for (const Function& function : functions_.array_ref()) {
    for (InstBlockId block_id : function.body_block_ids) {
      TerminatorKind prior_kind = TerminatorKind::NotTerminator;
      for (InstId inst_id : inst_blocks().Get(block_id)) {
        TerminatorKind inst_kind =
            insts().Get(inst_id).kind().terminator_kind();
        if (prior_kind == TerminatorKind::Terminator) {
          return Error(llvm::formatv("Inst {0} in block {1} follows terminator",
                                     inst_id, block_id));
        }
        if (prior_kind > inst_kind) {
          return Error(
              llvm::formatv("Non-terminator inst {0} in block {1} follows "
                            "terminator sequence",
                            inst_id, block_id));
        }
        prior_kind = inst_kind;
      }
      if (prior_kind != TerminatorKind::Terminator) {
        return Error(llvm::formatv("No terminator in block {0}", block_id));
      }
    }
  }

  // TODO: Check that an instruction only references other instructions that are
  // either global or that dominate it.
  return Success();
}

auto File::OutputYaml(bool include_builtins) const -> Yaml::OutputMapping {
  return Yaml::OutputMapping([this,
                              include_builtins](Yaml::OutputMapping::Map map) {
    map.Add("filename", filename_);
    map.Add("sem_ir", Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
              map.Add("cross_reference_irs_size",
                      Yaml::OutputScalar(cross_reference_irs_.size()));
              map.Add("functions", functions_.OutputYaml());
              map.Add("classes", classes_.OutputYaml());
              map.Add("types", types_.OutputYaml());
              map.Add("type_blocks", type_blocks_.OutputYaml());
              map.Add("insts",
                      Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                        int start =
                            include_builtins ? 0 : BuiltinKind::ValidCount;
                        for (int i : llvm::seq(start, insts_.size())) {
                          auto id = InstId(i);
                          map.Add(PrintToString(id),
                                  Yaml::OutputScalar(insts_.Get(id)));
                        }
                      }));
              map.Add("inst_blocks", inst_blocks_.OutputYaml());
            }));
  });
}

// Map an instruction kind representing a type into an integer describing the
// precedence of that type's syntax. Higher numbers correspond to higher
// precedence.
static auto GetTypePrecedence(InstKind kind) -> int {
  // clang warns on unhandled enum values; clang-tidy is incorrect here.
  // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
  switch (kind) {
    case ArrayType::Kind:
    case Builtin::Kind:
    case ClassType::Kind:
    case NameReference::Kind:
    case StructType::Kind:
    case TupleType::Kind:
    case UnboundFieldType::Kind:
      return 0;
    case ConstType::Kind:
      return -1;
    case PointerType::Kind:
      return -2;

    case CrossReference::Kind:
      // TODO: Once we support stringification of cross-references, we'll need
      // to determine the precedence of the target of the cross-reference. For
      // now, all cross-references refer to builtin types from the prelude.
      return 0;

    case AddressOf::Kind:
    case ArrayIndex::Kind:
    case ArrayInit::Kind:
    case Assign::Kind:
    case BinaryOperatorAdd::Kind:
    case BindName::Kind:
    case BindValue::Kind:
    case BlockArg::Kind:
    case BoolLiteral::Kind:
    case BoundMethod::Kind:
    case Branch::Kind:
    case BranchIf::Kind:
    case BranchWithArg::Kind:
    case Call::Kind:
    case ClassDeclaration::Kind:
    case ClassFieldAccess::Kind:
    case ClassInit::Kind:
    case Dereference::Kind:
    case Field::Kind:
    case FunctionDeclaration::Kind:
    case InitializeFrom::Kind:
    case IntegerLiteral::Kind:
    case Namespace::Kind:
    case NoOp::Kind:
    case Parameter::Kind:
    case RealLiteral::Kind:
    case Return::Kind:
    case ReturnExpression::Kind:
    case SelfParameter::Kind:
    case SpliceBlock::Kind:
    case StringLiteral::Kind:
    case StructAccess::Kind:
    case StructTypeField::Kind:
    case StructLiteral::Kind:
    case StructInit::Kind:
    case StructValue::Kind:
    case Temporary::Kind:
    case TemporaryStorage::Kind:
    case TupleAccess::Kind:
    case TupleIndex::Kind:
    case TupleLiteral::Kind:
    case TupleInit::Kind:
    case TupleValue::Kind:
    case UnaryOperatorNot::Kind:
    case ValueAsReference::Kind:
    case ValueOfInitializer::Kind:
    case VarStorage::Kind:
      CARBON_FATAL() << "GetTypePrecedence for non-type inst kind " << kind;
  }
}

auto File::StringifyType(TypeId type_id, bool in_type_context) const
    -> std::string {
  return StringifyTypeExpression(GetTypeAllowBuiltinTypes(type_id),
                                 in_type_context);
}

auto File::StringifyTypeExpression(InstId outer_inst_id,
                                   bool in_type_context) const -> std::string {
  std::string str;
  llvm::raw_string_ostream out(str);

  struct Step {
    // The instruction to print.
    InstId inst_id;
    // The index into inst_id to print. Not used by all types.
    int index = 0;

    auto Next() const -> Step {
      return {.inst_id = inst_id, .index = index + 1};
    }
  };
  llvm::SmallVector<Step> steps = {{.inst_id = outer_inst_id}};

  while (!steps.empty()) {
    auto step = steps.pop_back_val();
    if (!step.inst_id.is_valid()) {
      out << "<invalid type>";
      continue;
    }

    // Builtins have designated labels.
    if (step.inst_id.index < BuiltinKind::ValidCount) {
      out << BuiltinKind::FromInt(step.inst_id.index).label();
      continue;
    }

    auto inst = insts().Get(step.inst_id);
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (inst.kind()) {
      case ArrayType::Kind: {
        auto array = inst.As<ArrayType>();
        if (step.index == 0) {
          out << "[";
          steps.push_back(step.Next());
          steps.push_back(
              {.inst_id = GetTypeAllowBuiltinTypes(array.element_type_id)});
        } else if (step.index == 1) {
          out << "; " << GetArrayBoundValue(array.bound_id) << "]";
        }
        break;
      }
      case ClassType::Kind: {
        auto class_name_id =
            classes().Get(inst.As<ClassType>().class_id).name_id;
        out << identifiers().Get(class_name_id);
        break;
      }
      case ConstType::Kind: {
        if (step.index == 0) {
          out << "const ";

          // Add parentheses if required.
          auto inner_type_inst_id =
              GetTypeAllowBuiltinTypes(inst.As<ConstType>().inner_id);
          if (GetTypePrecedence(insts().Get(inner_type_inst_id).kind()) <
              GetTypePrecedence(inst.kind())) {
            out << "(";
            steps.push_back(step.Next());
          }

          steps.push_back({.inst_id = inner_type_inst_id});
        } else if (step.index == 1) {
          out << ")";
        }
        break;
      }
      case NameReference::Kind: {
        out << identifiers().Get(inst.As<NameReference>().name_id);
        break;
      }
      case PointerType::Kind: {
        if (step.index == 0) {
          steps.push_back(step.Next());
          steps.push_back({.inst_id = GetTypeAllowBuiltinTypes(
                               inst.As<PointerType>().pointee_id)});
        } else if (step.index == 1) {
          out << "*";
        }
        break;
      }
      case StructType::Kind: {
        auto refs = inst_blocks().Get(inst.As<StructType>().fields_id);
        if (refs.empty()) {
          out << "{}";
          break;
        } else if (step.index == 0) {
          out << "{";
        } else if (step.index < static_cast<int>(refs.size())) {
          out << ", ";
        } else {
          out << "}";
          break;
        }

        steps.push_back(step.Next());
        steps.push_back({.inst_id = refs[step.index]});
        break;
      }
      case StructTypeField::Kind: {
        auto field = inst.As<StructTypeField>();
        out << "." << identifiers().Get(field.name_id) << ": ";
        steps.push_back(
            {.inst_id = GetTypeAllowBuiltinTypes(field.field_type_id)});
        break;
      }
      case TupleType::Kind: {
        auto refs = type_blocks().Get(inst.As<TupleType>().elements_id);
        if (refs.empty()) {
          out << "()";
          break;
        } else if (step.index == 0) {
          out << "(";
        } else if (step.index < static_cast<int>(refs.size())) {
          out << ", ";
        } else {
          // A tuple of one element has a comma to disambiguate from an
          // expression.
          if (step.index == 1) {
            out << ",";
          }
          out << ")";
          break;
        }
        steps.push_back(step.Next());
        steps.push_back(
            {.inst_id = GetTypeAllowBuiltinTypes(refs[step.index])});
        break;
      }
      case UnboundFieldType::Kind: {
        if (step.index == 0) {
          out << "<unbound field of class ";
          steps.push_back(step.Next());
          steps.push_back({.inst_id = GetTypeAllowBuiltinTypes(
                               inst.As<UnboundFieldType>().class_type_id)});
        } else {
          out << ">";
        }
        break;
      }
      case AddressOf::Kind:
      case ArrayIndex::Kind:
      case ArrayInit::Kind:
      case Assign::Kind:
      case BinaryOperatorAdd::Kind:
      case BindName::Kind:
      case BindValue::Kind:
      case BlockArg::Kind:
      case BoolLiteral::Kind:
      case BoundMethod::Kind:
      case Branch::Kind:
      case BranchIf::Kind:
      case BranchWithArg::Kind:
      case Builtin::Kind:
      case Call::Kind:
      case ClassDeclaration::Kind:
      case ClassFieldAccess::Kind:
      case ClassInit::Kind:
      case CrossReference::Kind:
      case Dereference::Kind:
      case Field::Kind:
      case FunctionDeclaration::Kind:
      case InitializeFrom::Kind:
      case IntegerLiteral::Kind:
      case Namespace::Kind:
      case NoOp::Kind:
      case Parameter::Kind:
      case RealLiteral::Kind:
      case Return::Kind:
      case ReturnExpression::Kind:
      case SelfParameter::Kind:
      case SpliceBlock::Kind:
      case StringLiteral::Kind:
      case StructAccess::Kind:
      case StructLiteral::Kind:
      case StructInit::Kind:
      case StructValue::Kind:
      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case TupleAccess::Kind:
      case TupleIndex::Kind:
      case TupleLiteral::Kind:
      case TupleInit::Kind:
      case TupleValue::Kind:
      case UnaryOperatorNot::Kind:
      case ValueAsReference::Kind:
      case ValueOfInitializer::Kind:
      case VarStorage::Kind:
        // We don't need to handle stringification for instructions that don't
        // show up in errors, but make it clear what's going on so that it's
        // clearer when stringification is needed.
        out << "<cannot stringify " << step.inst_id << ">";
        break;
    }
  }

  // For `{}` or any tuple type, we've printed a non-type expression, so add a
  // conversion to type `type` if it's not implied by the context.
  if (!in_type_context) {
    auto outer_inst = insts().Get(outer_inst_id);
    if (outer_inst.Is<TupleType>() ||
        (outer_inst.Is<StructType>() &&
         inst_blocks().Get(outer_inst.As<StructType>().fields_id).empty())) {
      out << " as type";
    }
  }

  return str;
}

auto GetExpressionCategory(const File& file, InstId inst_id)
    -> ExpressionCategory {
  const File* ir = &file;

  // The overall expression category if the current instruction is a value
  // expression.
  ExpressionCategory value_category = ExpressionCategory::Value;

  while (true) {
    auto inst = ir->insts().Get(inst_id);
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (inst.kind()) {
      case Assign::Kind:
      case Branch::Kind:
      case BranchIf::Kind:
      case BranchWithArg::Kind:
      case ClassDeclaration::Kind:
      case Field::Kind:
      case FunctionDeclaration::Kind:
      case Namespace::Kind:
      case NoOp::Kind:
      case Return::Kind:
      case ReturnExpression::Kind:
      case StructTypeField::Kind:
        return ExpressionCategory::NotExpression;

      case CrossReference::Kind: {
        auto xref = inst.As<CrossReference>();
        ir = &ir->GetCrossReferenceIR(xref.ir_id);
        inst_id = xref.inst_id;
        continue;
      }

      case NameReference::Kind: {
        inst_id = inst.As<NameReference>().value_id;
        continue;
      }

      case AddressOf::Kind:
      case ArrayType::Kind:
      case BinaryOperatorAdd::Kind:
      case BindValue::Kind:
      case BlockArg::Kind:
      case BoolLiteral::Kind:
      case BoundMethod::Kind:
      case ClassType::Kind:
      case ConstType::Kind:
      case IntegerLiteral::Kind:
      case Parameter::Kind:
      case PointerType::Kind:
      case RealLiteral::Kind:
      case SelfParameter::Kind:
      case StringLiteral::Kind:
      case StructValue::Kind:
      case StructType::Kind:
      case TupleValue::Kind:
      case TupleType::Kind:
      case UnaryOperatorNot::Kind:
      case UnboundFieldType::Kind:
      case ValueOfInitializer::Kind:
        return value_category;

      case Builtin::Kind: {
        if (inst.As<Builtin>().builtin_kind == BuiltinKind::Error) {
          return ExpressionCategory::Error;
        }
        return value_category;
      }

      case BindName::Kind: {
        inst_id = inst.As<BindName>().value_id;
        continue;
      }

      case ArrayIndex::Kind: {
        inst_id = inst.As<ArrayIndex>().array_id;
        continue;
      }

      case ClassFieldAccess::Kind: {
        inst_id = inst.As<ClassFieldAccess>().base_id;
        // A value of class type is a pointer to an object representation.
        // Therefore, if the base is a value, the result is an ephemeral
        // reference.
        value_category = ExpressionCategory::EphemeralReference;
        continue;
      }

      case StructAccess::Kind: {
        inst_id = inst.As<StructAccess>().struct_id;
        continue;
      }

      case TupleAccess::Kind: {
        inst_id = inst.As<TupleAccess>().tuple_id;
        continue;
      }

      case TupleIndex::Kind: {
        inst_id = inst.As<TupleIndex>().tuple_id;
        continue;
      }

      case SpliceBlock::Kind: {
        inst_id = inst.As<SpliceBlock>().result_id;
        continue;
      }

      case StructLiteral::Kind:
      case TupleLiteral::Kind:
        return ExpressionCategory::Mixed;

      case ArrayInit::Kind:
      case Call::Kind:
      case InitializeFrom::Kind:
      case ClassInit::Kind:
      case StructInit::Kind:
      case TupleInit::Kind:
        return ExpressionCategory::Initializing;

      case Dereference::Kind:
      case VarStorage::Kind:
        return ExpressionCategory::DurableReference;

      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case ValueAsReference::Kind:
        return ExpressionCategory::EphemeralReference;
    }
  }
}

auto GetInitializingRepresentation(const File& file, TypeId type_id)
    -> InitializingRepresentation {
  auto value_rep = GetValueRepresentation(file, type_id);
  switch (value_rep.kind) {
    case ValueRepresentation::None:
      return {.kind = InitializingRepresentation::None};

    case ValueRepresentation::Copy:
      // TODO: Use in-place initialization for types that have non-trivial
      // destructive move.
      return {.kind = InitializingRepresentation::ByCopy};

    case ValueRepresentation::Pointer:
    case ValueRepresentation::Custom:
      return {.kind = InitializingRepresentation::InPlace};

    case ValueRepresentation::Unknown:
      CARBON_FATAL()
          << "Attempting to perform initialization of incomplete type";
  }
}

}  // namespace Carbon::SemIR
