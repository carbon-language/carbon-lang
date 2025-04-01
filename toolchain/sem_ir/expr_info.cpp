// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/expr_info.h"

#include "common/check.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

auto GetExprCategory(const File& file, InstId inst_id) -> ExprCategory {
  const File* ir = &file;

  // The overall expression category if the current instruction is a value
  // expression.
  ExprCategory value_category = ExprCategory::Value;

  while (true) {
    auto untyped_inst = ir->insts().Get(inst_id);
    CARBON_KIND_SWITCH(untyped_inst) {
      case AdaptDecl::Kind:
      case AddrPattern::Kind:
      case Assign::Kind:
      case BaseDecl::Kind:
      case BindingPattern::Kind:
      case Branch::Kind:
      case BranchIf::Kind:
      case BranchWithArg::Kind:
      case FieldDecl::Kind:
      case FunctionDecl::Kind:
      case ImplDecl::Kind:
      case NameBindingDecl::Kind:
      case Namespace::Kind:
      case OutParamPattern::Kind:
      case RefParamPattern::Kind:
      case RequirementEquivalent::Kind:
      case RequirementImpls::Kind:
      case RequirementRewrite::Kind:
      case Return::Kind:
      case ReturnSlotPattern::Kind:
      case ReturnExpr::Kind:
      case TuplePattern::Kind:
      case VarPattern::Kind:
      case Vtable::Kind:
        return ExprCategory::NotExpr;

      case ImportRefUnloaded::Kind:
      case ImportRefLoaded::Kind: {
        auto import_ir_inst = ir->import_ir_insts().Get(
            untyped_inst.As<SemIR::AnyImportRef>().import_ir_inst_id);
        ir = ir->import_irs().Get(import_ir_inst.ir_id).sem_ir;
        inst_id = import_ir_inst.inst_id;
        continue;
      }

      case CARBON_KIND(AsCompatible inst): {
        inst_id = inst.source_id;
        continue;
      }

      case CARBON_KIND(BindAlias inst): {
        inst_id = inst.value_id;
        continue;
      }
      case CARBON_KIND(ExportDecl inst): {
        inst_id = inst.value_id;
        continue;
      }
      case CARBON_KIND(NameRef inst): {
        inst_id = inst.value_id;
        continue;
      }

      case CARBON_KIND(Converted inst): {
        inst_id = inst.result_id;
        continue;
      }

      case CARBON_KIND(SpecificConstant inst): {
        inst_id = inst.inst_id;
        continue;
      }

      case AccessMemberAction::Kind:
      case AccessOptionalMemberAction::Kind:
      case AddrOf::Kind:
      case ArrayType::Kind:
      case AssociatedConstantDecl::Kind:
      case AssociatedEntity::Kind:
      case AssociatedEntityType::Kind:
      case AutoType::Kind:
      case BindSymbolicName::Kind:
      case BindValue::Kind:
      case BlockArg::Kind:
      case BoolLiteral::Kind:
      case BoolType::Kind:
      case BoundMethod::Kind:
      case BoundMethodType::Kind:
      case ClassDecl::Kind:
      case ClassType::Kind:
      case CompleteTypeWitness::Kind:
      case ConstType::Kind:
      case ConvertToValueAction::Kind:
      case FacetAccessType::Kind:
      case FacetAccessWitness::Kind:
      case FacetType::Kind:
      case FacetValue::Kind:
      case FloatLiteral::Kind:
      case FloatType::Kind:
      case FunctionType::Kind:
      case FunctionTypeWithSelfType::Kind:
      case GenericClassType::Kind:
      case GenericInterfaceType::Kind:
      case LookupImplWitness::Kind:
      case ImplWitness::Kind:
      case ImplWitnessAccess::Kind:
      case ImportCppDecl::Kind:
      case ImportDecl::Kind:
      case InstType::Kind:
      case InstValue::Kind:
      case IntLiteralType::Kind:
      case IntType::Kind:
      case IntValue::Kind:
      case InterfaceDecl::Kind:
      case LegacyFloatType::Kind:
      case NamespaceType::Kind:
      case PointerType::Kind:
      case RefineTypeAction::Kind:
      case RequireCompleteType::Kind:
      case SpecificFunction::Kind:
      case SpecificFunctionType::Kind:
      case SpecificImplFunction::Kind:
      case StringLiteral::Kind:
      case StringType::Kind:
      case StructType::Kind:
      case StructValue::Kind:
      case SymbolicBindingPattern::Kind:
      case TupleType::Kind:
      case TupleValue::Kind:
      case TypeOfInst::Kind:
      case TypeType::Kind:
      case UnaryOperatorNot::Kind:
      case UnboundElementType::Kind:
      case ValueOfInitializer::Kind:
      case ValueParam::Kind:
      case ValueParamPattern::Kind:
      case VtableType::Kind:
      case WhereExpr::Kind:
      case WitnessType::Kind:
        return value_category;

      case ErrorInst::Kind:
        return ExprCategory::Error;

      case CARBON_KIND(BindName inst): {
        // TODO: Don't rely on value_id for expression category, since it may
        // not be valid yet. This workaround only works because we don't support
        // `var` in function signatures yet.
        if (!inst.value_id.has_value()) {
          return value_category;
        }
        inst_id = inst.value_id;
        continue;
      }

      case CARBON_KIND(ArrayIndex inst): {
        inst_id = inst.array_id;
        continue;
      }

      case VtablePtr::Kind:
        return ExprCategory::EphemeralRef;

      case CARBON_KIND(ClassElementAccess inst): {
        inst_id = inst.base_id;
        // A value of class type is a pointer to an object representation.
        // Therefore, if the base is a value, the result is an ephemeral
        // reference.
        value_category = ExprCategory::EphemeralRef;
        continue;
      }

      case CARBON_KIND(StructAccess inst): {
        inst_id = inst.struct_id;
        continue;
      }

      case CARBON_KIND(TupleAccess inst): {
        inst_id = inst.tuple_id;
        continue;
      }

      case CARBON_KIND(SpliceBlock inst): {
        inst_id = inst.result_id;
        continue;
      }

      case SpliceInst::Kind:
        // TODO: Add ExprCategory::Dependent.
        return value_category;

      case StructLiteral::Kind:
      case TupleLiteral::Kind:
        return ExprCategory::Mixed;

      case ArrayInit::Kind:
      case Call::Kind:
      case InitializeFrom::Kind:
      case ClassInit::Kind:
      case StructInit::Kind:
      case TupleInit::Kind:
        return ExprCategory::Initializing;

      case Deref::Kind:
      case VarStorage::Kind:
      case ReturnSlot::Kind:
        return ExprCategory::DurableRef;

      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case ValueAsRef::Kind:
        return ExprCategory::EphemeralRef;

      case OutParam::Kind:
      case RefParam::Kind:
        // TODO: Consider introducing a separate category for OutParam:
        // unlike other DurableRefs, it permits initialization.
        return ExprCategory::DurableRef;
    }
  }
}

auto FindReturnSlotArgForInitializer(const File& sem_ir, InstId init_id)
    -> InstId {
  while (true) {
    Inst init_untyped = sem_ir.insts().Get(init_id);
    CARBON_KIND_SWITCH(init_untyped) {
      case CARBON_KIND(AsCompatible init): {
        init_id = init.source_id;
        continue;
      }
      case CARBON_KIND(Converted init): {
        init_id = init.result_id;
        continue;
      }
      case CARBON_KIND(ArrayInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(ClassInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(StructInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(TupleInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(InitializeFrom init): {
        return init.dest_id;
      }
      case CARBON_KIND(Call call): {
        if (!ReturnTypeInfo::ForType(sem_ir, call.type_id).has_return_slot()) {
          return InstId::None;
        }
        if (!call.args_id.has_value()) {
          // Argument initialization failed, so we have no return slot.
          return InstId::None;
        }
        return sem_ir.inst_blocks().Get(call.args_id).back();
      }
      default:
        CARBON_FATAL("Initialization from unexpected inst {0}", init_untyped);
    }
  }
}

}  // namespace Carbon::SemIR
