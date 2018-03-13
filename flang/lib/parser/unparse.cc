// Generates Fortran from the content of a parse tree, using the
// traversal templates in parse-tree-visitor.h.

#include "unparse.h"
#include "characters.h"
#include "idioms.h"
#include "indirection.h"
#include "parse-tree-visitor.h"
#include "parse-tree.h"
#include <algorithm>

namespace Fortran {
namespace parser {

class UnparseVisitor {
public:
  UnparseVisitor(std::ostream &out, int indentationAmount, Encoding encoding)
    : out_{out}, indentationAmount_{indentationAmount}, encoding_{encoding} {}

  // Default actions: just traverse the children
  template<typename T> bool Pre(const T &x) { return true; }
  template<typename T> void Post(const T &) {}

  // Emit simple types as-is.
  bool Pre(const std::string &x) {
    Put(x);
    return false;
  }
  bool Pre(int x) {
    Put(std::to_string(x));
    return false;
  }
  bool Pre(std::uint64_t x) {
    Put(std::to_string(x));
    return false;
  }
  bool Pre(std::int64_t x) {
    Put(std::to_string(x));
    return false;
  }
  bool Pre(char x) {
    Put(x);
    return false;
  }

  // Statement labels and ends of lines
  template<typename T> bool Pre(const Statement<T> &x) {
    Walk(x.label, " ");
    return true;
  }
  template<typename T> void Post(const Statement<T> &) { Put('\n'); }

  // The special-case formatting functions for these productions are
  // ordered to correspond roughly to their order of appearance in
  // the Fortran 2018 standard (and parse-tree.h).

  void Post(const ProgramUnit &x) {  // R502, R503
    out_ << '\n';  // blank line after each ProgramUnit
  }
  bool Pre(const DefinedOperator::IntrinsicOperator &x) {  // R608
    switch (x) {
    case DefinedOperator::IntrinsicOperator::Power: Put("**"); break;
    case DefinedOperator::IntrinsicOperator::Multiply: Put('*'); break;
    case DefinedOperator::IntrinsicOperator::Divide: Put('/'); break;
    case DefinedOperator::IntrinsicOperator::Add: Put('+'); break;
    case DefinedOperator::IntrinsicOperator::Subtract: Put('-'); break;
    case DefinedOperator::IntrinsicOperator::Concat: Put("//"); break;
    case DefinedOperator::IntrinsicOperator::LT: Put('<'); break;
    case DefinedOperator::IntrinsicOperator::LE: Put("<="); break;
    case DefinedOperator::IntrinsicOperator::EQ: Put("=="); break;
    case DefinedOperator::IntrinsicOperator::NE: Put("/="); break;
    case DefinedOperator::IntrinsicOperator::GE: Put(">="); break;
    case DefinedOperator::IntrinsicOperator::GT: Put('>'); break;
    default:
      PutEnum(static_cast<int>(x), DefinedOperator::IntrinsicOperatorAsString);
    }
    return false;
  }
  void Post(const Star &) { Put('*'); }  // R701 &c.
  void Post(const TypeParamValue::Deferred &) { Put(':'); }  // R701
  bool Pre(const DeclarationTypeSpec::Type &x) {  // R703
    Put("TYPE("), Walk(x.derived), Put(')');
    return false;
  }
  bool Pre(const DeclarationTypeSpec::Class &x) {
    Put("CLASS("), Walk(x.derived), Put(')');
    return false;
  }
  void Post(const DeclarationTypeSpec::ClassStar &) { Put("CLASS(*)"); }
  void Post(const DeclarationTypeSpec::TypeStar &) { Put("TYPE(*)"); }
  bool Pre(const DeclarationTypeSpec::Record &x) {
    Put("RECORD /"), Walk(x.v), Put('/');
    return false;
  }
  bool Pre(const IntrinsicTypeSpec::Real &x) {  // R704
    Put("REAL");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Complex &x) {
    Put("COMPLEX");
    return true;
  }
  void Post(const IntrinsicTypeSpec::DoublePrecision &) {
    Put("DOUBLE PRECISION");
  }
  bool Pre(const IntrinsicTypeSpec::Character &x) {
    Put("CHARACTER");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Logical &x) {
    Put("LOGICAL");
    return true;
  }
  void Post(const IntrinsicTypeSpec::DoubleComplex &) { Put("DOUBLE COMPLEX"); }
  bool Pre(const IntrinsicTypeSpec::NCharacter &x) {
    Put("NCHARACTER");
    return true;
  }
  bool Pre(const IntegerTypeSpec &x) {  // R705
    Put("INTEGER");
    return true;
  }
  bool Pre(const KindSelector &x) {  // R706
    std::visit(
        visitors{[&](const ScalarIntConstantExpr &y) {
                   Put("(KIND="), Walk(y), Put(')');
                 },
            [&](const KindSelector::StarSize &y) { Put('*'), Walk(y.v); }},
        x.u);
    return false;
  }
  bool Pre(const SignedIntLiteralConstant &x) {  // R707
    Walk(std::get<std::int64_t>(x.t));
    Walk("_", std::get<std::optional<KindParam>>(x.t));
    return false;
  }
  bool Pre(const IntLiteralConstant &x) {  // R708
    Walk(std::get<std::uint64_t>(x.t));
    Walk("_", std::get<std::optional<KindParam>>(x.t));
    return false;
  }
  bool Pre(const Sign &x) {  // R712
    Put(x == Sign::Negative ? '-' : '+');
    return false;
  }
  bool Pre(const RealLiteralConstant &x) {  // R714, R715
    Put(x.intPart), Put('.'), Put(x.fraction), Walk(x.exponent);
    Walk("_", x.kind);
    return false;
  }
  bool Pre(const ComplexLiteralConstant &x) {  // R718 - R720
    Put('('), Walk(x.t, ","), Put(')');
    return false;
  }
  bool Pre(const CharSelector::LengthAndKind &x) {  // R721
    Put("(KIND="), Walk(x.kind), Walk(", LEN=", x.length), Put(')');
    return false;
  }
  bool Pre(const LengthSelector &x) {  // R722
    std::visit(visitors{[&](const TypeParamValue &y) {
                          Put("(LEN="), Walk(y), Put(')');
                        },
                   [&](const CharLength &y) { Put('*'), Walk(y); }},
        x.u);
    return false;
  }
  bool Pre(const CharLength &x) {  // R723
    std::visit(
        visitors{[&](const TypeParamValue &y) { Put('('), Walk(y), Put(')'); },
            [&](const std::int64_t &y) { Walk(y); }},
        x.u);
    return false;
  }
  bool Pre(const CharLiteralConstant &x) {  // R724
    if (const auto &k = std::get<std::optional<KindParam>>(x.t)) {
      if (std::holds_alternative<KindParam::Kanji>(k->u)) {
        Put("NC");
      } else {
        Walk(*k), Put('_');
      }
    }
    PutQuoted(std::get<std::string>(x.t));
    return false;
  }
  bool Pre(const HollerithLiteralConstant &x) {
    std::optional<size_t> chars{CountCharacters(x.v.data(), x.v.size(),
        encoding_ == Encoding::EUC_JP ? EUC_JPCharacterBytes
                                      : UTF8CharacterBytes)};
    if (chars.has_value()) {
      Pre(*chars);
    } else {
      Pre(x.v.size());
    }
    Put('H');
    return true;
  }
  bool Pre(const LogicalLiteralConstant &x) {  // R725
    Put(x.v ? ".TRUE." : ".FALSE.");
    return false;
  }
  bool Pre(const DerivedTypeStmt &x) {  // R727
    Put("TYPE"), Walk(", ", std::get<std::list<TypeAttrSpec>>(x.t), ", ");
    Put(" :: "), Put(std::get<Name>(x.t));
    Walk("(", std::get<std::list<Name>>(x.t), ", ", ")");
    Indent();
    return false;
  }
  bool Pre(const Abstract &x) {  // R728, &c.
    Put("ABSTRACT");
    return false;
  }
  bool Pre(const TypeAttrSpec::BindC &x) {
    Put("BIND(C)");
    return false;
  }
  bool Pre(const TypeAttrSpec::Extends &x) {
    Put("EXTENDS("), Walk(x.v), Put(')');
    return false;
  }
  void Post(const EndTypeStmt &) {  // R730
    Outdent();
    Put("END TYPE");
  }
  bool Pre(const SequenceStmt &x) {  // R731
    Put("SEQUENCE");
    return false;
  }
  bool Pre(const TypeParamDefStmt &x) {  // R732
    Walk(std::get<IntegerTypeSpec>(x.t));
    Put(", "), Walk(std::get<TypeParamDefStmt::KindOrLen>(x.t));
    Put(" :: "), Walk(std::get<std::list<TypeParamDecl>>(x.t), ", ");
    return false;
  }
  bool Pre(const TypeParamDecl &x) {  // R733
    Put(std::get<Name>(x.t));
    Walk("=", std::get<std::optional<ScalarIntConstantExpr>>(x.t));
    return false;
  }
  bool Pre(const DataComponentDefStmt &x) {  // R737
    const auto &dts = std::get<DeclarationTypeSpec>(x.t);
    const auto &attrs = std::get<std::list<ComponentAttrSpec>>(x.t);
    const auto &decls = std::get<std::list<ComponentDecl>>(x.t);
    Walk(dts), Walk(", ", attrs, ", ");
    if (!attrs.empty() ||
        (!std::holds_alternative<DeclarationTypeSpec::Record>(dts.u) &&
            std::none_of(
                decls.begin(), decls.end(), [](const ComponentDecl &d) {
                  const auto &init =
                      std::get<std::optional<Initialization>>(d.t);
                  return init.has_value() &&
                      std::holds_alternative<
                          std::list<Indirection<DataStmtValue>>>(init->u);
                }))) {
      Put(" ::");
    }
    Put(' '), Walk(decls, ", ");
    return false;
  }
  bool Pre(const Allocatable &x) {  // R738
    Put("ALLOCATABLE");
    return false;
  }
  bool Pre(const Pointer &x) {
    Put("POINTER");
    return false;
  }
  bool Pre(const Contiguous &x) {
    Put("CONTIGUOUS");
    return false;
  }
  bool Pre(const ComponentAttrSpec &x) {
    std::visit(visitors{[&](const CoarraySpec &) { Put("CODIMENSION["); },
                   [&](const ComponentArraySpec &) { Put("DIMENSION("); },
                   [&](const auto &) {}},
        x.u);
    return true;
  }
  void Post(const ComponentAttrSpec &x) {
    std::visit(visitors{[&](const CoarraySpec &) { Put(']'); },
                   [&](const ComponentArraySpec &) { Put(')'); },
                   [&](const auto &) {}},
        x.u);
  }
  bool Pre(const ComponentDecl &x) {  // R739
    Walk(std::get<ObjectName>(x.t));
    Walk("(", std::get<std::optional<ComponentArraySpec>>(x.t), ")");
    Walk("[", std::get<std::optional<CoarraySpec>>(x.t), "]");
    Walk("*", std::get<std::optional<CharLength>>(x.t));
    Walk(std::get<std::optional<Initialization>>(x.t));
    return false;
  }
  bool Pre(const ComponentArraySpec &x) {  // R740
    std::visit(
        visitors{[&](const std::list<ExplicitShapeSpec> &y) { Walk(y, ","); },
            [&](const DeferredShapeSpecList &y) { Walk(y); }},
        x.u);
    return false;
  }
  bool Pre(const ProcComponentDefStmt &x) {  // R741
    Put("PROCEDURE(");
    Walk(std::get<std::optional<ProcInterface>>(x.t)), Put(')');
    Walk(", ", std::get<std::list<ProcComponentAttrSpec>>(x.t), ", ");
    Put(" :: "), Walk(std::get<std::list<ProcDecl>>(x.t), ", ");
    return false;
  }
  bool Pre(const NoPass &x) {  // R742
    Put("NOPASS");
    return false;
  }
  bool Pre(const Pass &x) {
    Put("PASS"), Walk("(", x.v, ")");
    return false;
  }
  bool Pre(const Initialization &x) {  // R743 & R805
    std::visit(visitors{[&](const ConstantExpr &y) { Put(" = "), Walk(y); },
                   [&](const NullInit &y) { Put(" => "), Walk(y); },
                   [&](const InitialDataTarget &y) { Put(" => "), Walk(y); },
                   [&](const std::list<Indirection<DataStmtValue>> &y) {
                     Walk("/", y, ", ", "/");
                   }},
        x.u);
    return false;
  }
  bool Pre(const PrivateStmt &x) {  // R745
    Put("PRIVATE");
    return false;
  }
  bool Pre(const TypeBoundProcedureStmt::WithoutInterface &x) {  // R749
    Put("PROCEDURE"), Walk(", ", x.attributes, ", ");
    Put(" :: "), Walk(x.declarations);
    return false;
  }
  bool Pre(const TypeBoundProcedureStmt::WithInterface &x) {
    Put("PROCEDURE("), Walk(x.interfaceName), Put("), ");
    Walk(x.attributes);
    Put(" :: "), Walk(x.bindingNames);
    return false;
  }
  bool Pre(const TypeBoundProcDecl &x) {  // R750
    Walk(std::get<Name>(x.t));
    Walk(" => ", std::get<std::optional<Name>>(x.t));
    return false;
  }
  bool Pre(const TypeBoundGenericStmt &x) {  // R751
    Put("GENERIC"), Walk(", ", std::get<std::optional<AccessSpec>>(x.t));
    Put(" :: "), Walk(std::get<Indirection<GenericSpec>>(x.t));
    Put(" => "), Walk(std::get<std::list<Name>>(x.t), ", ");
    return false;
  }
  void Post(const BindAttr::Deferred &) { Put("DEFERRED"); }  // R752
  void Post(const BindAttr::Non_Overridable &) { Put("NON_OVERRIDABLE"); }
  void Post(const FinalProcedureStmt &) { Put("FINAL :: "); }  // R753
  bool Pre(const DerivedTypeSpec &x) {  // R754
    Walk(std::get<Name>(x.t));
    Walk("(", std::get<std::list<TypeParamSpec>>(x.t), ",", ")");
    return false;
  }
  bool Pre(const TypeParamSpec &x) {  // R755
    Walk(std::get<std::optional<Keyword>>(x.t), "=");
    Walk(std::get<TypeParamValue>(x.t));
    return false;
  }
  bool Pre(const StructureConstructor &x) {  // R756
    Walk(std::get<DerivedTypeSpec>(x.t));
    Put('('), Walk(std::get<std::list<ComponentSpec>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const ComponentSpec &x) {  // R757
    Walk(std::get<std::optional<Keyword>>(x.t), "=");
    Walk(std::get<ComponentDataSource>(x.t));
    return false;
  }
  bool Pre(const EnumDefStmt &) {  // R760
    Put("ENUM, BIND(C)");
    Indent();
    return false;
  }
  bool Pre(const EnumeratorDefStmt &) {  // R761
    Put("ENUMERATOR :: ");
    return true;
  }
  bool Pre(const Enumerator &x) {  // R762
    Walk(std::get<NamedConstant>(x.t));
    Walk(" = ", std::get<std::optional<ScalarIntConstantExpr>>(x.t));
    return false;
  }
  void Post(const EndEnumStmt &) {  // R763
    Outdent();
    Put("END ENUM");
  }
  bool Pre(const BOZLiteralConstant &x) {  // R764 - R767
    Put("Z'");
    out_ << std::hex << x.v << std::dec << '\'';
    return false;
  }
  bool Pre(const AcValue::Triplet &x) {  // R773
    Walk(std::get<0>(x.t)), Put(':'), Walk(std::get<1>(x.t));
    Walk(":", std::get<std::optional<ScalarIntExpr>>(x.t));
    return false;
  }
  bool Pre(const ArrayConstructor &x) {  // R769
    Put('['), Walk(x.v), Put(']');
    return false;
  }
  bool Pre(const AcSpec &x) {  // R770
    Walk(x.type, "::"), Walk(x.values, ", ");
    return false;
  }
  template<typename A> bool Pre(const LoopBounds<A> &x) {
    Walk(x.name), Put('='), Walk(x.lower), Put(','), Walk(x.upper);
    Walk(",", x.step);
    return false;
  }
  bool Pre(const AcImpliedDoControl &x) {  // R775
    Walk(std::get<std::optional<IntegerTypeSpec>>(x.t), "::");
    Walk(std::get<LoopBounds<ScalarIntExpr>>(x.t));
    return false;
  }

  bool Pre(const TypeDeclarationStmt &x) {  // R801
    const auto &dts = std::get<DeclarationTypeSpec>(x.t);
    const auto &attrs = std::get<std::list<AttrSpec>>(x.t);
    const auto &decls = std::get<std::list<EntityDecl>>(x.t);
    Walk(dts), Walk(", ", attrs, ", ");
    if (!attrs.empty() ||
        (!std::holds_alternative<DeclarationTypeSpec::Record>(dts.u) &&
            std::none_of(decls.begin(), decls.end(), [](const EntityDecl &d) {
              const auto &init = std::get<std::optional<Initialization>>(d.t);
              return init.has_value() &&
                  std::holds_alternative<std::list<Indirection<DataStmtValue>>>(
                      init->u);
            }))) {
      Put(" ::");
    }
    Put(' '), Walk(std::get<std::list<EntityDecl>>(x.t), ", ");
    return false;
  }
  bool Pre(const AttrSpec &x) {  // R802
    std::visit(visitors{[&](const CoarraySpec &y) { Put("CODIMENSION["); },
                   [&](const ArraySpec &y) { Put("DIMENSION("); },
                   [&](const auto &) {}},
        x.u);
    return true;
  }
  void Post(const AttrSpec &x) {
    std::visit(visitors{[&](const CoarraySpec &y) { Put(']'); },
                   [&](const ArraySpec &y) { Put(')'); }, [&](const auto &) {}},
        x.u);
  }
  bool Pre(const EntityDecl &x) {  // R803
    Walk(std::get<ObjectName>(x.t));
    Walk("(", std::get<std::optional<ArraySpec>>(x.t), ")");
    Walk("[", std::get<std::optional<CoarraySpec>>(x.t), "]");
    Walk("*", std::get<std::optional<CharLength>>(x.t));
    Walk(std::get<std::optional<Initialization>>(x.t));
    return false;
  }
  bool Pre(const NullInit &x) {  // R806
    Put("NULL()");
    return false;
  }
  bool Pre(const LanguageBindingSpec &x) {  // R808 & R1528
    Put("BIND(C"), Walk(", NAME=", x.v), Put(')');
    return false;
  }
  bool Pre(const CoarraySpec &x) {  // R809
    std::visit(visitors{[&](const DeferredCoshapeSpecList &y) { Walk(y); },
                   [&](const ExplicitCoshapeSpec &y) { Walk(y); }},
        x.u);
    return false;
  }
  bool Post(const DeferredCoshapeSpecList &x) {  // R810
    for (auto j = x.v; j > 0; --j) {
      Put(':');
      if (j > 1) {
        Put(',');
      }
    }
    return false;
  }
  bool Pre(const ExplicitCoshapeSpec &x) {  // R811
    Walk(std::get<std::list<ExplicitShapeSpec>>(x.t), ",", ",");
    Walk(std::get<std::optional<SpecificationExpr>>(x.t), ":"), Put('*');
    return false;
  }
  bool Pre(const ExplicitShapeSpec &x) {  // R812 - R813 & R816 - R818
    Walk(std::get<std::optional<SpecificationExpr>>(x.t), ":");
    Walk(std::get<SpecificationExpr>(x.t));
    return false;
  }
  bool Pre(const ArraySpec &x) {  // R815
    std::visit(
        visitors{[&](const std::list<ExplicitShapeSpec> &y) { Walk(y, ","); },
            [&](const std::list<AssumedShapeSpec> &y) { Walk(y, ","); },
            [&](const DeferredShapeSpecList &y) { Walk(y); },
            [&](const AssumedSizeSpec &y) { Walk(y); },
            [&](const ImpliedShapeSpec &y) { Walk(y); },
            [&](const AssumedRankSpec &y) { Walk(y); }},
        x.u);
    return false;
  }
  void Post(const AssumedShapeSpec &) { Put(':'); }  // R819
  bool Post(const DeferredShapeSpecList &x) {  // R820
    for (auto j = x.v; j > 0; --j) {
      Put(':');
      if (j > 1) {
        Put(',');
      }
    }
    return false;
  }
  bool Pre(const AssumedImpliedSpec &x) {  // R821
    Walk(x.v, ":");
    Put('*');
    return false;
  }
  bool Pre(const AssumedSizeSpec &x) {  // R822
    Walk(std::get<std::list<ExplicitShapeSpec>>(x.t), ",", ",");
    Walk(std::get<AssumedImpliedSpec>(x.t));
    return false;
  }
  bool Pre(const ImpliedShapeSpec &x) {  // R823
    Walk(x.v, ",");
    return false;
  }
  void Post(const AssumedRankSpec &) { Put(".."); }  // R825
  void Post(const Asynchronous &) { Put("ASYNCHRONOUS"); }
  void Post(const External &) { Put("EXTERNAL"); }
  void Post(const Intrinsic &) { Put("INTRINSIC"); }
  void Post(const Optional &) { Put("OPTIONAL"); }
  void Post(const Parameter &) { Put("PARAMETER"); }
  void Post(const Protected &) { Put("PROTECTED"); }
  void Post(const Save &) { Put("SAVE"); }
  void Post(const Target &) { Put("TARGET"); }
  void Post(const Value &) { Put("VALUE"); }
  void Post(const Volatile &) { Put("VOLATILE"); }
  bool Pre(const IntentSpec &x) {  // R826
    Put("INTENT("), Walk(x.v), Put(")");
    return false;
  }
  bool Pre(const AccessStmt &x) {  // R827
    Walk(std::get<AccessSpec>(x.t));
    Walk(" :: ", std::get<std::list<AccessId>>(x.t), ", ");
    return false;
  }
  bool Pre(const AllocatableStmt &x) {  // R829
    Put("ALLOCATABLE :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ObjectDecl &x) {  // R830 & R860
    Walk(std::get<ObjectName>(x.t));
    Walk("(", std::get<std::optional<ArraySpec>>(x.t), ")");
    Walk("[", std::get<std::optional<CoarraySpec>>(x.t), "]");
    return false;
  }
  bool Pre(const AsynchronousStmt &x) {  // R831
    Put("ASYNCHRONOUS :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const BindStmt &x) {  // R832
    Walk(x.t, " :: ");
    return false;
  }
  bool Pre(const BindEntity &x) {  // R833
    bool isCommon{std::get<BindEntity::Kind>(x.t) == BindEntity::Kind::Common};
    const char *slash{isCommon ? "/" : ""};
    Put(slash), Walk(std::get<Name>(x.t)), Put(slash);
    return false;
  }
  bool Pre(const CodimensionStmt &x) {  // R834
    Put("CODIMENSION :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const CodimensionDecl &x) {  // R835
    Walk(std::get<Name>(x.t));
    Put('['), Walk(std::get<CoarraySpec>(x.t)), Put(']');
    return false;
  }
  bool Pre(const ContiguousStmt &x) {  // R836
    Put("CONTIGUOUS :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const DataStmt &) {  // R837
    Put("DATA ");
    return true;
  }
  bool Pre(const DataStmtSet &x) {  // R838
    Walk(std::get<std::list<DataStmtObject>>(x.t), ", ");
    Put('/'), Walk(std::get<std::list<DataStmtValue>>(x.t), ", "), Put('/');
    return false;
  }
  bool Pre(const DataImpliedDo &x) {  // R840, R842
    Put("("), Walk(std::get<std::list<DataIDoObject>>(x.t), ", "), Put(',');
    Walk(std::get<std::optional<IntegerTypeSpec>>(x.t), "::");
    Walk(std::get<LoopBounds<ScalarIntConstantExpr>>(x.t)), Put(')');
    return false;
  }
  bool Pre(const DataStmtValue &x) {  // R843
    Walk(std::get<std::optional<DataStmtRepeat>>(x.t), "*");
    Walk(std::get<DataStmtConstant>(x.t));
    return false;
  }
  bool Pre(const DimensionStmt &x) {  // R848
    Put("DIMENSION :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const DimensionStmt::Declaration &x) {
    Walk(std::get<Name>(x.t));
    Put('('), Walk(std::get<ArraySpec>(x.t)), Put(')');
    return false;
  }
  bool Pre(const IntentStmt &x) {  // R849
    Walk(x.t, " :: ");
    return false;
  }
  bool Pre(const OptionalStmt &x) {  // R850
    Put("OPTIONAL :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ParameterStmt &x) {  // R851
    Put("PARAMETER("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const NamedConstantDef &x) {  // R852
    Walk(x.t, "=");
    return false;
  }
  bool Pre(const PointerStmt &x) {  // R853
    Put("POINTER :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ProtectedStmt &x) {  // R855
    Put("PROTECTED :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const SaveStmt &x) {  // R856
    Put("SAVE"), Walk(" :: ", x.v, ", ");
    return false;
  }
  bool Pre(const SavedEntity &x) {  // R857, R858
    bool isCommon{
        std::get<SavedEntity::Kind>(x.t) == SavedEntity::Kind::Common};
    const char *slash{isCommon ? "/" : ""};
    Put(slash), Walk(std::get<Name>(x.t)), Put(slash);
    return false;
  }
  bool Pre(const TargetStmt &x) {  // R859
    Put("TARGET :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ValueStmt &x) {  // R861
    Put("VALUE :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const VolatileStmt &x) {  // R862
    Put("VOLATILE :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ImplicitStmt &x) {  // R863
    Put("IMPLICIT ");
    std::visit(
        visitors{[&](const std::list<ImplicitSpec> &y) { Walk(y, ", "); },
            [&](const std::list<ImplicitStmt::ImplicitNoneNameSpec> &y) {
              Put("NONE"), Walk(" (", y, ", ", ")");
            }},
        x.u);
    return false;
  }
  bool Pre(const ImplicitSpec &x) {  // R864
    Walk(std::get<DeclarationTypeSpec>(x.t));
    Put('('), Walk(std::get<std::list<LetterSpec>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const LetterSpec &x) {  // R865
    Put(std::get<char>(x.t)), Walk("-", std::get<std::optional<char>>(x.t));
    return false;
  }
  bool Pre(const ImportStmt &x) {  // R867
    Put("IMPORT");
    switch (x.kind) {
    case ImportStmt::Kind::Default:
      Put(" :: ");
      Walk(x.names);
      break;
    case ImportStmt::Kind::Only:
      Put(", ONLY: ");
      Walk(x.names);
      break;
    case ImportStmt::Kind::None: Put(", NONE"); break;
    case ImportStmt::Kind::All: Put(", ALL"); break;
    default: CRASH_NO_CASE;
    }
    return false;
  }
  bool Pre(const NamelistStmt &x) {  // R868
    Put("NAMELIST"), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const NamelistStmt::Group &x) {
    Put('/'), Put(std::get<Name>(x.t)), Put('/');
    Walk(std::get<std::list<Name>>(x.t), ", ");
    return false;
  }
  bool Pre(const EquivalenceStmt &x) {  // R870, R871
    Put("EQUIVALENCE");
    const char *separator{" "};
    for (const std::list<EquivalenceObject> &y : x.v) {
      Put(separator), Put('('), Walk(y), Put(')');
      separator = ", ";
    }
    return false;
  }
  bool Pre(const CommonStmt &x) {  // R873
    Put("COMMON ");
    Walk("/", std::get<std::optional<std::optional<Name>>>(x.t), "/");
    Walk(std::get<std::list<CommonBlockObject>>(x.t), ", ");
    Walk(", ", std::get<std::list<CommonStmt::Block>>(x.t), ", ");
    return false;
  }
  bool Pre(const CommonBlockObject &x) {  // R874
    Walk(std::get<Name>(x.t));
    Walk("(", std::get<std::optional<ArraySpec>>(x.t), ")");
    return false;
  }
  bool Pre(const CommonStmt::Block &x) {
    Walk("/", std::get<std::optional<Name>>(x.t), "/");
    Walk(std::get<std::list<CommonBlockObject>>(x.t));
    return false;
  }

  bool Pre(const Substring &x) {  // R908, R909
    Walk(std::get<DataReference>(x.t));
    Put('('), Walk(std::get<SubstringRange>(x.t)), Put(')');
    return false;
  }
  bool Pre(const CharLiteralConstantSubstring &x) {
    Walk(std::get<CharLiteralConstant>(x.t));
    Put('('), Walk(std::get<SubstringRange>(x.t)), Put(')');
    return false;
  }
  bool Pre(const SubstringRange &x) {  // R910
    Walk(x.t, ":");
    return false;
  }
  bool Pre(const PartRef &x) {  // R912
    Walk(x.name);
    Walk("(", x.subscripts, ",", ")");
    Walk(x.imageSelector);
    return false;
  }
  bool Pre(const StructureComponent &x) {  // R913
    Walk(x.base), Put(percentOrDot_), Walk(x.component);
    return false;
  }
  bool Pre(const ArrayElement &x) {  // R917
    Walk(x.base);
    Put('('), Walk(x.subscripts, ","), Put(')');
    return false;
  }
  bool Pre(const SubscriptTriplet &x) {  // R921
    Walk(std::get<0>(x.t)), Put(':'), Walk(std::get<1>(x.t));
    Walk(":", std::get<2>(x.t));
    return false;
  }
  bool Pre(const ImageSelector &x) {  // R924
    Put('['), Walk(std::get<std::list<Cosubscript>>(x.t), ",");
    Walk(",", std::get<std::list<ImageSelectorSpec>>(x.t), ","), Put(']');
    return false;
  }
  bool Pre(const ImageSelectorSpec::Stat &) {
    Put("STAT=");
    return true;
  }
  bool Pre(const ImageSelectorSpec::Team &) {
    Put("TEAM=");
    return true;
  }
  bool Pre(const ImageSelectorSpec::Team_Number &) {
    Put("TEAM_NUMBER=");
    return true;
  }
  bool Pre(const AllocateStmt &x) {  // R927
    Put("ALLOCATE("), Walk(std::get<std::optional<TypeSpec>>(x.t), "::");
    Walk(std::get<std::list<Allocation>>(x.t), ", ");
    Walk(", ", std::get<std::list<AllocOpt>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const AllocOpt &x) {  // R928, R931
    std::visit(visitors{[&](const AllocOpt::Mold &) { Put("MOLD="); },
                   [&](const AllocOpt::Source &) { Put("SOURCE="); },
                   [&](const StatOrErrmsg &y) {}},
        x.u);
    return true;
  }
  bool Pre(const Allocation &x) {  // R932
    Walk(std::get<AllocateObject>(x.t));
    Walk("(", std::get<std::list<AllocateShapeSpec>>(x.t), ",", ")");
    Walk("[", std::get<std::optional<AllocateCoarraySpec>>(x.t), "]");
    return false;
  }
  bool Pre(const AllocateShapeSpec &x) {  // R934 & R938
    Walk(std::get<std::optional<BoundExpr>>(x.t), ":");
    Walk(std::get<BoundExpr>(x.t));
    return false;
  }
  bool Pre(const AllocateCoarraySpec &x) {  // R937
    Walk(std::get<std::list<AllocateCoshapeSpec>>(x.t), ",", ",");
    Walk(std::get<std::optional<BoundExpr>>(x.t), ":"), Put('*');
    return false;
  }
  bool Pre(const NullifyStmt &x) {  // R939
    Put("NULLIFY("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const DeallocateStmt &x) {  // R941
    Put("DEALLOCATE("), Walk(std::get<std::list<AllocateObject>>(x.t), ", ");
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const StatOrErrmsg &x) {  // R942 & R1165
    std::visit(visitors{[&](const StatVariable &) { Put("STAT="); },
                   [&](const MsgVariable &) { Put("ERRMSG="); }},
        x.u);
    return true;
  }

  // R1001 - R1022
  bool Pre(const Expr::Parentheses &x) {
    Put('('), Walk(x.v), Put(')');
    return false;
  }
  bool Pre(const Expr::UnaryPlus &x) {
    Put("+");
    return true;
  }
  bool Pre(const Expr::Negate &x) {
    Put("-");
    return true;
  }
  bool Pre(const Expr::NOT &x) {
    Put(".NOT.");
    return true;
  }
  bool Pre(const Expr::PercentLoc &x) {
    Put("%LOC("), Walk(x.v), Put(')');
    return false;
  }
  bool Pre(const Expr::DefinedUnary &x) {
    Put('.'), Walk(x.t, ". ");
    return false;
  }
  bool Pre(const Expr::Power &x) {
    Walk(x.t, "**");
    return false;
  }
  bool Pre(const Expr::Multiply &x) {
    Walk(x.t, "*");
    return false;
  }
  bool Pre(const Expr::Divide &x) {
    Walk(x.t, "/");
    return false;
  }
  bool Pre(const Expr::Add &x) {
    Walk(x.t, "+");
    return false;
  }
  bool Pre(const Expr::Subtract &x) {
    Walk(x.t, "-");
    return false;
  }
  bool Pre(const Expr::Concat &x) {
    Walk(x.t, "//");
    return false;
  }
  bool Pre(const Expr::LT &x) {
    Walk(x.t, "<");
    return false;
  }
  bool Pre(const Expr::LE &x) {
    Walk(x.t, "<=");
    return false;
  }
  bool Pre(const Expr::EQ &x) {
    Walk(x.t, "==");
    return false;
  }
  bool Pre(const Expr::NE &x) {
    Walk(x.t, "/=");
    return false;
  }
  bool Pre(const Expr::GE &x) {
    Walk(x.t, ">=");
    return false;
  }
  bool Pre(const Expr::GT &x) {
    Walk(x.t, ">");
    return false;
  }
  bool Pre(const Expr::AND &x) {
    Walk(x.t, ".AND.");
    return false;
  }
  bool Pre(const Expr::OR &x) {
    Walk(x.t, ".OR.");
    return false;
  }
  bool Pre(const Expr::EQV &x) {
    Walk(x.t, ".EQV.");
    return false;
  }
  bool Pre(const Expr::NEQV &x) {
    Walk(x.t, ".NEQV.");
    return false;
  }
  bool Pre(const Expr::ComplexConstructor &x) {
    Put('('), Walk(x.t, ","), Put(')');
    return false;
  }
  bool Pre(const Expr::DefinedBinary &x) {
    Walk(std::get<1>(x.t));  // left
    Walk(std::get<DefinedOpName>(x.t));
    Walk(std::get<2>(x.t));  // right
    return false;
  }
  bool Pre(const DefinedOpName &x) {  // R1003, R1023, R1414, & R1415
    Put('.'), Put(x.v), Put('.');
    return false;
  }
  bool Pre(const AssignmentStmt &x) {  // R1032
    Walk(x.t, " = ");
    return false;
  }
  bool Pre(const PointerAssignmentStmt &x) {  // R1033, R1034, R1038
    Walk(std::get<Variable>(x.t));
    std::visit(
        visitors{[&](const std::list<BoundsRemapping> &y) {
                   Put('('), Walk(y), Put(')');
                 },
            [&](const std::list<BoundsSpec> &y) { Walk("(", y, ", ", ")"); }},
        std::get<PointerAssignmentStmt::Bounds>(x.t).u);
    Put(" => "), Walk(std::get<Expr>(x.t));
    return false;
  }
  void Post(const BoundsSpec &) {  // R1035
    Put(':');
  }
  bool Pre(const BoundsRemapping &x) {  // R1036
    Walk(x.t, ":");
    return false;
  }
  bool Pre(const ProcComponentRef &x) {  // R1039
    Walk(std::get<Scalar<Variable>>(x.t)), Put(percentOrDot_);
    Walk(std::get<Name>(x.t));
    return false;
  }
  bool Pre(const WhereStmt &x) {  // R1041, R1045, R1046
    Put("WHERE ("), Walk(x.t, ") ");
    return false;
  }
  bool Pre(const WhereConstructStmt &x) {  // R1043
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Put("WHERE ("), Walk(std::get<LogicalExpr>(x.t)), Put(')');
    Indent();
    return false;
  }
  bool Pre(const MaskedElsewhereStmt &x) {  // R1047
    Outdent();
    Put("ELSEWHERE ("), Walk(std::get<LogicalExpr>(x.t)), Put(')');
    Walk(" ", std::get<std::optional<Name>>(x.t));
    Indent();
    return false;
  }
  bool Pre(const ElsewhereStmt &x) {  // R1048
    Outdent(), Put("ELSEWHERE"), Walk(" ", x.v), Indent();
    return false;
  }
  bool Pre(const EndWhereStmt &x) {  // R1049
    Outdent(), Put("END WHERE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ForallConstructStmt &x) {  // R1051
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Put("FORALL"), Walk(std::get<Indirection<ConcurrentHeader>>(x.t));
    Indent();
    return false;
  }
  bool Pre(const EndForallStmt &x) {  // R1054
    Outdent(), Put("END FORALL"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ForallStmt &) {  // R1055
    Put("FORALL");
    return true;
  }

  bool Pre(const AssociateStmt &x) {  // R1103
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Put("ASSOCIATE ("), Walk(std::get<std::list<Association>>(x.t), ", ");
    Put(')'), Indent();
    return false;
  }
  bool Pre(const Association &x) {  // R1104
    Walk(x.t, " => ");
    return false;
  }
  bool Pre(const EndAssociateStmt &x) {  // R1106
    Outdent(), Put("END ASSOCIATE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const BlockStmt &x) {  // R1108
    Walk(x.v, ": "), Put("BLOCK"), Indent();
    return false;
  }
  bool Pre(const EndBlockStmt &x) {  // R1110
    Outdent(), Put("END BLOCK"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ChangeTeamStmt &x) {  // R1112
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Put("CHANGE TEAM ("), Walk(std::get<TeamVariable>(x.t));
    Walk(", ", std::get<std::list<CoarrayAssociation>>(x.t), ", ");
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
    Indent();
    return false;
  }
  bool Pre(const CoarrayAssociation &x) {  // R1113
    Walk(x.t, " => ");
    return false;
  }
  bool Pre(const EndChangeTeamStmt &x) {  // R1114
    Outdent(), Put("END TEAM (");
    Walk(std::get<std::list<StatOrErrmsg>>(x.t), ", ");
    Put(')'), Walk(" ", std::get<std::optional<Name>>(x.t));
    return false;
  }
  bool Pre(const CriticalStmt &x) {  // R1117
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Put("CRITICAL ("), Walk(std::get<std::list<StatOrErrmsg>>(x.t), ", ");
    Put(')'), Indent();
    return false;
  }
  bool Pre(const EndCriticalStmt &x) {  // R1118
    Outdent(), Put("END CRITICAL"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const DoConstruct &x) {  // R1119, R1120
    Walk(std::get<Statement<NonLabelDoStmt>>(x.t));
    Indent(), Walk(std::get<Block>(x.t), ""), Outdent();
    Walk(std::get<Statement<EndDoStmt>>(x.t));
    return false;
  }
  bool Pre(const LabelDoStmt &x) {  // R1121
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Put("DO "), Walk(std::get<Label>(x.t));
    Walk(" ", std::get<std::optional<LoopControl>>(x.t));
    return false;
  }
  bool Pre(const NonLabelDoStmt &x) {  // R1122
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Put("DO "), Walk(std::get<std::optional<LoopControl>>(x.t));
    return false;
  }
  bool Pre(const LoopControl &x) {  // R1123
    std::visit(visitors{[&](const ScalarLogicalExpr &y) {
                          Put("WHILE ("), Walk(y), Put(')');
                        },
                   [&](const auto &y) { Walk(y); }},
        x.u);
    return false;
  }
  bool Pre(const ConcurrentHeader &x) {  // R1125
    Put('('), Walk(std::get<std::optional<IntegerTypeSpec>>(x.t), "::");
    Walk(std::get<std::list<ConcurrentControl>>(x.t), ", ");
    Walk(", ", std::get<std::optional<ScalarLogicalExpr>>(x.t)), Put(')');
    return false;
  }
  bool Pre(const ConcurrentControl &x) {  // R1126 - R1128
    Walk(std::get<Name>(x.t)), Put('='), Walk(std::get<1>(x.t));
    Put(':'), Walk(std::get<2>(x.t));
    Walk(":", std::get<std::optional<ScalarIntExpr>>(x.t));
    return false;
  }
  bool Pre(const LoopControl::Concurrent &x) {  // R1129
    Put("CONCURRENT");
    return true;
  }
  bool Pre(const LocalitySpec::Local &x) {
    Put("LOCAL("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const LocalitySpec::LocalInit &x) {
    Put("LOCAL INIT("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const LocalitySpec::Shared &x) {
    Put("SHARED("), Walk(x.v, ", "), Put(')');
    return false;
  }
  void Post(const LocalitySpec::DefaultNone &x) { Put("DEFAULT(NONE)"); }
  bool Pre(const EndDoStmt &x) {  // R1132
    Put("END DO"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const CycleStmt &x) {  // R1133
    Put("CYCLE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const IfThenStmt &x) {  // R1135
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Put("IF ("), Walk(std::get<ScalarLogicalExpr>(x.t)), Put(") THEN");
    Indent();
    return false;
  }
  bool Pre(const ElseIfStmt &x) {  // R1136
    Outdent(), Put("ELSE IF ("), Walk(std::get<ScalarLogicalExpr>(x.t));
    Put(") THEN"), Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
    return false;
  }
  bool Pre(const ElseStmt &x) {  // R1137
    Outdent(), Put("ELSE"), Walk(" ", x.v), Indent();
    return false;
  }
  bool Pre(const EndIfStmt &x) {  // R1138
    Outdent(), Put("END IF"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const IfStmt &x) {  // R1139
    Put("IF ("), Walk(x.t, ") ");
    return false;
  }
  bool Pre(const SelectCaseStmt &x) {  // R1141, R1144
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Put("SELECT CASE ("), Walk(std::get<Scalar<Expr>>(x.t)), Put(')'), Indent();
    return false;
  }
  bool Pre(const CaseStmt &x) {  // R1142
    Outdent(), Put("CASE "), Walk(std::get<CaseSelector>(x.t));
    Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
    return false;
  }
  bool Pre(const EndSelectStmt &x) {  // R1143 & R1151 & R1155
    Outdent(), Put("END SELECT"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const CaseSelector &x) {  // R1145
    std::visit(visitors{[&](const std::list<CaseValueRange> &y) {
                          Put('('), Walk(y), Put(')');
                        },
                   [&](const Default &) { Put("DEFAULT"); }},
        x.u);
    return false;
  }
  bool Pre(const CaseValueRange::Range &x) {  // R1146
    Walk(x.lower), Put(':'), Walk(x.upper);
    return false;
  }
  bool Pre(const SelectRankStmt &x) {  // R1149
    Walk(std::get<0>(x.t), ": ");
    Put("SELECT RANK ("), Walk(std::get<1>(x.t), " => ");
    Walk(std::get<Selector>(x.t)), Put(')'), Indent();
    return false;
  }
  bool Pre(const SelectRankCaseStmt &x) {  // R1150
    Outdent(), Put("RANK ");
    std::visit(visitors{[&](const ScalarIntConstantExpr &y) {
                          Put('('), Walk(y), Put(')');
                        },
                   [&](const Star &) { Put("(*)"); },
                   [&](const Default &) { Put("DEFAULT"); }},
        std::get<SelectRankCaseStmt::Rank>(x.t).u);
    Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
    return false;
  }
  bool Pre(const SelectTypeStmt &x) {  // R1153
    Walk(std::get<0>(x.t), ": ");
    Put("SELECT TYPE ("), Walk(std::get<1>(x.t), " => ");
    Walk(std::get<Selector>(x.t)), Put(')'), Indent();
    return false;
  }
  bool Pre(const TypeGuardStmt &x) {  // R1154
    Outdent(), Walk(std::get<TypeGuardStmt::Guard>(x.t));
    Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
    return false;
  }
  bool Pre(const TypeGuardStmt::Guard &x) {
    std::visit(visitors{[&](const TypeSpec &y) {
                          Put("TYPE IS ("), Walk(y), Put(')');
                        },
                   [&](const DerivedTypeSpec &y) {
                     Put("CLASS IS ("), Walk(y), Put(')');
                   },
                   [&](const Default &) { Put("CLASS DEFAULT"); }},
        x.u);
    return false;
  }
  bool Pre(const ExitStmt &x) {  // R1156
    Put("EXIT"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const GotoStmt &x) {  // R1157
    Put("GO TO ");
    return true;
  }
  bool Pre(const ComputedGotoStmt &x) {  // R1158
    Put("GO TO ("), Walk(x.t, "), ");
    return false;
  }
  bool Pre(const ContinueStmt &x) {  // R1159
    Put("CONTINUE");
    return false;
  }
  bool Pre(const StopStmt &x) {  // R1160, R1161
    if (std::get<StopStmt::Kind>(x.t) == StopStmt::Kind::ErrorStop) {
      Put("ERROR ");
    }
    Put("STOP"), Walk(" ", std::get<std::optional<StopCode>>(x.t));
    Walk(", QUIET=", std::get<std::optional<ScalarLogicalExpr>>(x.t));
    return false;
  }
  bool Pre(const FailImageStmt &x) {  // R1163
    Put("FAIL IMAGE");
    return false;
  }
  bool Pre(const SyncAllStmt &x) {  // R1164
    Put("SYNC ALL ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const SyncImagesStmt &x) {  // R1166
    Put("SYNC IMAGES ("), Walk(std::get<SyncImagesStmt::ImageSet>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const SyncMemoryStmt &x) {  // R1168
    Put("SYNC MEMORY ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const SyncTeamStmt &x) {  // R1169
    Put("SYNC TEAM ("), Walk(std::get<TeamVariable>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const EventPostStmt &x) {  // R1170
    Put("EVENT POST ("), Walk(std::get<EventVariable>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const EventWaitStmt::EventWaitSpec &x) {  // R1173, R1174
    std::visit(
        visitors{[&](const ScalarIntExpr &x) { Put("UNTIL_COUNT="), Walk(x); },
            [&](const StatOrErrmsg &y) {}},
        x.u);
    return true;
  }
  bool Pre(const EventWaitStmt &x) {  // R1170
    Put("EVENT WAIT ("), Walk(std::get<EventVariable>(x.t));
    Walk(", ", std::get<std::list<EventWaitStmt::EventWaitSpec>>(x.t), ", ");
    Put(')');
    return false;
  }
  bool Pre(const FormTeamStmt &x) {  // R1175
    Put("FORM TEAM ("), Walk(std::get<ScalarIntExpr>(x.t));
    Put(','), Walk(std::get<TeamVariable>(x.t));
    Walk(", ", std::get<std::list<FormTeamStmt::FormTeamSpec>>(x.t), ", ");
    Put(')');
    return false;
  }
  bool Pre(const FormTeamStmt::FormTeamSpec &x) {  // R1176, R1177
    std::visit(
        visitors{[&](const ScalarIntExpr &x) { Put("NEW_INDEX="), Walk(x); },
            [&](const StatOrErrmsg &y) {}},
        x.u);
    return true;
  }
  bool Pre(const LockStmt &x) {  // R1178
    Put("LOCK ("), Walk(std::get<LockVariable>(x.t));
    Walk(", ", std::get<std::list<LockStmt::LockStat>>(x.t), ", ");
    Put(')');
    return false;
  }
  bool Pre(const LockStmt::LockStat &x) {  // R1179
    std::visit(visitors{[&](const ScalarLogicalVariable &x) {
                          Put("ACQUIRED_LOCK="), Walk(x);
                        },
                   [&](const StatOrErrmsg &y) {}},
        x.u);
    return true;
  }
  bool Pre(const UnlockStmt &x) {  // R1180
    Put("UNLOCK ("), Walk(std::get<LockVariable>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", ");
    Put(')');
    return false;
  }

  bool Pre(const OpenStmt &x) {  // R1204
    Put("OPEN ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const ConnectSpec &x) {  // R1205
    return std::visit(visitors{[&](const FileUnitNumber &) {
                                 Put("UNIT=");
                                 return true;
                               },
                          [&](const FileNameExpr &) {
                            Put("FILE=");
                            return true;
                          },
                          [&](const ConnectSpec::CharExpr &y) {
                            Walk(y.t, "=");
                            return false;
                          },
                          [&](const MsgVariable &) {
                            Put("IOMSG=");
                            return true;
                          },
                          [&](const StatVariable &) {
                            Put("IOSTAT=");
                            return true;
                          },
                          [&](const ConnectSpec::Recl &) {
                            Put("RECL=");
                            return true;
                          },
                          [&](const ConnectSpec::Newunit &) {
                            Put("NEWUNIT=");
                            return true;
                          },
                          [&](const ErrLabel &) {
                            Put("ERR=");
                            return true;
                          },
                          [&](const StatusExpr &) {
                            Put("STATUS=");
                            return true;
                          }},
        x.u);
  }
  bool Pre(const CloseStmt &x) {  // R1208
    Put("CLOSE ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const CloseStmt::CloseSpec &x) {  // R1209
    std::visit(visitors{[&](const FileUnitNumber &) { Put("UNIT="); },
                   [&](const StatVariable &) { Put("IOSTAT="); },
                   [&](const MsgVariable &) { Put("IOMSG="); },
                   [&](const ErrLabel &) { Put("ERR="); },
                   [&](const StatusExpr &) { Put("STATUS="); }},
        x.u);
    return true;
  }
  bool Pre(const ReadStmt &x) {  // R1210
    Put("READ ");
    if (x.iounit) {
      Put('('), Walk(x.iounit);
      if (x.format) {
        Put(", "), Walk(x.format);
      }
      Put(')');
    } else if (x.format) {
      Walk(x.format);
      if (!x.items.empty()) {
        Put(", ");
      }
    } else {
      Put('('), Walk(x.controls), Put(')');
    }
    Walk(" ", x.items, ", ");
    return false;
  }
  bool Pre(const WriteStmt &x) {  // R1211
    Put("WRITE (");
    if (x.iounit) {
      Walk(x.iounit);
      if (x.format) {
        Put(", "), Walk(x.format);
      }
      Walk(", ", x.controls);
    } else {
      Walk(x.controls);
    }
    Put(')'), Walk(" ", x.items, ", ");
    return false;
  }
  bool Pre(const PrintStmt &x) {  // R1212
    Put("PRINT "), Walk(std::get<Format>(x.t));
    Walk(", ", std::get<std::list<OutputItem>>(x.t), ", ");
    return false;
  }
  bool Pre(const IoControlSpec &x) {  // R1213
    return std::visit(visitors{[&](const IoUnit &) {
                                 Put("UNIT=");
                                 return true;
                               },
                          [&](const Format &) {
                            Put("FMT=");
                            return true;
                          },
                          [&](const Name &) {
                            Put("NML=");
                            return true;
                          },
                          [&](const IoControlSpec::CharExpr &y) {
                            Walk(y.t, "=");
                            return false;
                          },
                          [&](const IoControlSpec::Asynchronous &) {
                            Put("ASYNCHRONOUS=");
                            return true;
                          },
                          [&](const EndLabel &) {
                            Put("END=");
                            return true;
                          },
                          [&](const EorLabel &) {
                            Put("EOR=");
                            return true;
                          },
                          [&](const ErrLabel &) {
                            Put("ERR=");
                            return true;
                          },
                          [&](const IdVariable &) {
                            Put("ID=");
                            return true;
                          },
                          [&](const MsgVariable &) {
                            Put("IOMSG=");
                            return true;
                          },
                          [&](const StatVariable &) {
                            Put("IOSTAT=");
                            return true;
                          },
                          [&](const IoControlSpec::Pos &) {
                            Put("POS=");
                            return true;
                          },
                          [&](const IoControlSpec::Rec &) {
                            Put("REC=");
                            return true;
                          },
                          [&](const IoControlSpec::Size &) {
                            Put("SIZE=");
                            return true;
                          }},
        x.u);
  }
  bool Pre(const InputImpliedDo &x) {  // R1218
    Put('('), Walk(std::get<std::list<InputItem>>(x.t), ", "), Put(", ");
    Walk(std::get<IoImpliedDoControl>(x.t)), Put(')');
    return false;
  }
  bool Pre(const OutputImpliedDo &x) {  // R1219
    Put('('), Walk(std::get<std::list<OutputItem>>(x.t), ", "), Put(", ");
    Walk(std::get<IoImpliedDoControl>(x.t)), Put(')');
    return false;
  }
  bool Pre(const WaitStmt &x) {  // R1222
    Put("WAIT ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const WaitSpec &x) {  // R1223
    std::visit(visitors{[&](const FileUnitNumber &) { Put("UNIT="); },
                   [&](const EndLabel &) { Put("END="); },
                   [&](const EorLabel &) { Put("EOR="); },
                   [&](const ErrLabel &) { Put("ERR="); },
                   [&](const IdExpr &) { Put("ID="); },
                   [&](const MsgVariable &) { Put("IOMSG="); },
                   [&](const StatVariable &) { Put("IOSTAT="); }},
        x.u);
    return true;
  }
  bool Pre(const BackspaceStmt &x) {  // R1224
    Put("BACKSPACE ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const EndfileStmt &x) {  // R1225
    Put("ENDFILE ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const RewindStmt &x) {  // R1226
    Put("REWIND ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const FlushStmt &x) {  // R1228
    Put("FLUSH ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const InquireStmt &x) {  // R1230
    Put("INQUIRE (");
    std::visit(
        visitors{[&](const InquireStmt::Iolength &y) {
                   Put("IOLENGTH="), Walk(y.t, ") ");
                 },
            [&](const std::list<InquireSpec> &y) { Walk(y, ", "), Put(')'); }},
        x.u);
    return false;
  }
  bool Pre(const InquireSpec &x) {  // R1231
    return std::visit(visitors{[&](const FileUnitNumber &) {
                                 Put("UNIT=");
                                 return true;
                               },
                          [&](const FileNameExpr &) {
                            Put("FILE=");
                            return true;
                          },
                          [&](const InquireSpec::CharVar &y) {
                            Walk(y.t, "=");
                            return false;
                          },
                          [&](const InquireSpec::IntVar &y) {
                            Walk(y.t, "=");
                            return false;
                          },
                          [&](const InquireSpec::LogVar &y) {
                            Walk(y.t, "=");
                            return false;
                          },
                          [&](const IdExpr &) {
                            Put("ID=");
                            return true;
                          },
                          [&](const ErrLabel &) {
                            Put("ERR=");
                            return true;
                          }},
        x.u);
  }

  bool Pre(const FormatStmt &) {  // R1301
    Put("FORMAT");
    return true;
  }
  bool Pre(const format::FormatSpecification &x) {  // R1302, R1303, R1305
    Put('('), Walk("", x.items, ",", x.unlimitedItems.empty() ? "" : ",");
    Walk("*(", x.unlimitedItems, ",", ")"), Put(')');
    return false;
  }
  bool Pre(const format::FormatItem &x) {  // R1304, R1306, R1321
    if (x.repeatCount.has_value()) {
      Walk(*x.repeatCount);
    }
    std::visit(visitors{[&](const std::string &y) { PutQuoted(y); },
                   [&](const std::list<format::FormatItem> &y) {
                     Walk("(", y, ",", ")");
                   },
                   [&](const auto &y) { Walk(y); }},
        x.u);
    return false;
  }
  bool Pre(const format::IntrinsicTypeDataEditDesc &x) {  // R1307(1/2) - R1311
    switch (x.kind) {
#define FMT(x) \
  case format::IntrinsicTypeDataEditDesc::Kind::x: Put(#x); break
      FMT(I);
      FMT(B);
      FMT(O);
      FMT(Z);
      FMT(F);
      FMT(E);
      FMT(EN);
      FMT(ES);
      FMT(EX);
      FMT(G);
      FMT(L);
      FMT(A);
      FMT(D);
#undef FMT
    default: CRASH_NO_CASE;
    }
    Walk(x.width), Walk(".", x.digits), Walk("E", x.exponentWidth);
    return false;
  }
  bool Pre(const format::DerivedTypeDataEditDesc &x) {  // R1307(2/2), R1312
    Put("DT");
    if (!x.type.empty()) {
      Put('"'), Put(x.type), Put('"');
    }
    Walk("(", x.parameters, ",", ")");
    return false;
  }
  bool Pre(const format::ControlEditDesc &x) {  // R1313, R1315-R1320
    switch (x.kind) {
    case format::ControlEditDesc::Kind::T:
      Put('T');
      Walk(x.count);
      break;
    case format::ControlEditDesc::Kind::TL:
      Put("TL");
      Walk(x.count);
      break;
    case format::ControlEditDesc::Kind::TR:
      Put("TR");
      Walk(x.count);
      break;
    case format::ControlEditDesc::Kind::X:
      if (x.count != 1) {
        Walk(x.count);
      }
      Put('X');
      break;
    case format::ControlEditDesc::Kind::Slash:
      if (x.count != 1) {
        Walk(x.count);
      }
      Put('/');
      break;
    case format::ControlEditDesc::Kind::Colon: Put(':'); break;
    case format::ControlEditDesc::Kind::P:
      Walk(x.count);
      Put('P');
      break;
#define FMT(x) \
  case format::ControlEditDesc::Kind::x: Put(#x); break
      FMT(SS);
      FMT(SP);
      FMT(S);
      FMT(BN);
      FMT(BZ);
      FMT(RU);
      FMT(RD);
      FMT(RZ);
      FMT(RN);
      FMT(RC);
      FMT(RP);
      FMT(DC);
      FMT(DP);
#undef FMT
    default: CRASH_NO_CASE;
    }
    return false;
  }

  bool Pre(const MainProgram &x) {  // R1401
    if (!std::get<std::optional<Statement<ProgramStmt>>>(x.t)) {
      Indent();
    }
    return true;
  }
  bool Pre(const ProgramStmt &x) {  // R1402
    Put("PROGRAM "), Indent();
    return true;
  }
  bool Pre(const EndProgramStmt &x) {  // R1403
    Outdent(), Put("END PROGRAM"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ModuleStmt &) {  // R1405
    Put("MODULE "), Indent();
    return true;
  }
  bool Pre(const EndModuleStmt &x) {  // R1406
    Outdent(), Put("END MODULE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const UseStmt &x) {  // R1409
    Put("USE"), Walk(", ", x.nature), Put(" :: "), Walk(x.moduleName);
    std::visit(
        visitors{[&](const std::list<Rename> &y) { Walk(", ", y, ", "); },
            [&](const std::list<Only> &y) { Walk(", ONLY: ", y, ", "); }},
        x.u);
    return false;
  }
  bool Pre(const Rename &x) {  // R1411
    std::visit(visitors{[&](const Rename::Names &y) { Walk(y.t, " => "); },
                   [&](const Rename::Operators &y) {
                     Put('.'), Walk(y.t, ". => ."), Put('.');
                   }},
        x.u);
    return false;
  }
  bool Pre(const SubmoduleStmt &x) {  // R1417
    Put("SUBMODULE "), Indent();
    return true;
  }
  bool Pre(const ParentIdentifier &x) {  // R1418
    Walk(std::get<Name>(x.t)), Walk(":", std::get<std::optional<Name>>(x.t));
    return false;
  }
  bool Pre(const EndSubmoduleStmt &x) {  // R1419
    Outdent(), Put("END SUBMODULE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const BlockDataStmt &x) {  // R1421
    Put("BLOCK DATA"), Walk(" ", x.v), Indent();
    return false;
  }
  bool Pre(const EndBlockDataStmt &x) {  // R1422
    Outdent(), Put("END BLOCK DATA"), Walk(" ", x.v);
    return false;
  }

  bool Pre(const InterfaceStmt &x) {  // R1503
    std::visit(visitors{[&](const std::optional<GenericSpec> &y) {
                          Put("INTERFACE"), Walk(" ", y);
                        },
                   [&](const Abstract &) { Put("ABSTRACT INTERFACE"); }},
        x.u);
    Indent();
    return false;
  }
  bool Pre(const EndInterfaceStmt &x) {  // R1504
    Outdent(), Put("END INTERFACE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ProcedureStmt &x) {  // R1506
    if (std::get<ProcedureStmt::Kind>(x.t) ==
        ProcedureStmt::Kind::ModuleProcedure) {
      Put("MODULE ");
    }
    Put("PROCEDURE :: ");
    Walk(std::get<std::list<Name>>(x.t), ", ");
    return false;
  }
  bool Pre(const GenericSpec &x) {  // R1508, R1509
    std::visit(visitors{[&](const GenericSpec::Assignment &) {
                          Put("ASSIGNMENT (=)");
                        },
                   [&](const GenericSpec::ReadFormatted &) {
                     Put("READ (FORMATTED)");
                   },
                   [&](const GenericSpec::ReadUnformatted &) {
                     Put("READ (UNFORMATTED)");
                   },
                   [&](const GenericSpec::WriteFormatted &) {
                     Put("WRITE (FORMATTED)");
                   },
                   [&](const GenericSpec::WriteUnformatted &) {
                     Put("WRITE (UNFORMATTED)");
                   },
                   [&](const auto &y) {}},
        x.u);
    return true;
  }
  bool Pre(const GenericStmt &x) {  // R1510
    Put("GENERIC"), Walk(", ", std::get<std::optional<AccessSpec>>(x.t));
    Put(" :: "), Walk(std::get<GenericSpec>(x.t)), Put(" => ");
    Walk(std::get<std::list<Name>>(x.t), ", ");
    return false;
  }
  bool Pre(const ExternalStmt &x) {  // R1511
    Put("EXTERNAL :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ProcedureDeclarationStmt &x) {  // R1512
    Put("PROCEDURE ("), Walk(std::get<std::optional<ProcInterface>>(x.t));
    Put(')'), Walk(", ", std::get<std::list<ProcAttrSpec>>(x.t), ", ");
    Put(" :: "), Walk(std::get<std::list<ProcDecl>>(x.t), ", ");
    return false;
  }
  bool Pre(const ProcDecl &x) {  // R1515
    Walk(std::get<Name>(x.t));
    Walk(" => ", std::get<std::optional<ProcPointerInit>>(x.t));
    return false;
  }
  bool Pre(const IntrinsicStmt &x) {  // R1519
    Put("INTRINSIC :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const FunctionReference &x) {  // R1520
    Walk(std::get<ProcedureDesignator>(x.v.t));
    Put('('), Walk(std::get<std::list<ActualArgSpec>>(x.v.t), ", "), Put(')');
    return false;
  }
  bool Pre(const CallStmt &x) {  // R1521
    Put("CALL "), Walk(std::get<ProcedureDesignator>(x.v.t));
    Walk(" (", std::get<std::list<ActualArgSpec>>(x.v.t), ", ", ")");
    return false;
  }
  bool Pre(const ActualArgSpec &x) {  // R1523
    Walk(std::get<std::optional<Keyword>>(x.t), "=");
    Walk(std::get<ActualArg>(x.t));
    return false;
  }
  bool Pre(const ActualArg::PercentRef &x) {  // R1524
    Put("%REF("), Walk(x.v), Put(')');
    return false;
  }
  bool Pre(const ActualArg::PercentVal &x) {
    Put("%VAL("), Walk(x.v), Put(')');
    return false;
  }
  bool Pre(const AltReturnSpec &) {  // R1525
    Put("*");
    return true;
  }
  bool Pre(const FunctionStmt &x) {  // R1530
    Walk("", std::get<std::list<PrefixSpec>>(x.t), " ", " ");
    Put("FUNCTION "), Walk(std::get<Name>(x.t)), Put(" (");
    Walk(std::get<std::list<Name>>(x.t), ", "), Put(')');
    Walk(" ", std::get<std::optional<Suffix>>(x.t)), Indent();
    return false;
  }
  bool Pre(const Suffix &x) {  // R1532
    if (x.resultName) {
      Put("RESULT ("), Walk(x.resultName), Put(')');
      Walk(" ", x.binding);
    } else {
      Walk(x.binding);
    }
    return false;
  }
  bool Pre(const EndFunctionStmt &x) {  // R1533
    Outdent(), Put("END FUNCTION"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const SubroutineStmt &x) {  // R1535
    Walk("", std::get<std::list<PrefixSpec>>(x.t), " ", " ");
    Put("SUBROUTINE "), Walk(std::get<Name>(x.t));
    Walk(" (", std::get<std::list<DummyArg>>(x.t), ", ", ")");
    Walk(" ", std::get<std::optional<LanguageBindingSpec>>(x.t));
    Indent();
    return false;
  }
  bool Pre(const EndSubroutineStmt &x) {  // R1537
    Outdent(), Put("END SUBROUTINE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const MpSubprogramStmt &) {  // R1539
    Put("MODULE PROCEDURE "), Indent();
    return true;
  }
  bool Pre(const EndMpSubprogramStmt &x) {  // R1540
    Outdent(), Put("END PROCEDURE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const EntryStmt &x) {  // R1541
    Put("ENTRY "), Walk(std::get<Name>(x.t));
    Walk(" (", std::get<std::list<DummyArg>>(x.t), ", ", ")");
    Walk(" ", std::get<std::optional<Suffix>>(x.t));
    return false;
  }
  bool Pre(const ReturnStmt &x) {  // R1542
    Put("RETURN"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ContainsStmt &x) {  // R1543
    Outdent();
    Put("CONTAINS");
    Indent();
    return false;
  }
  bool Pre(const StmtFunctionStmt &x) {  // R1544
    Walk(std::get<Name>(x.t)), Put('(');
    Walk(std::get<std::list<Name>>(x.t), ", "), Put(") = ");
    Walk(std::get<Scalar<Expr>>(x.t));
    return false;
  }

  // Extensions and deprecated constructs
  bool Pre(const BasedPointerStmt &x) {
    Put("POINTER ("), Walk(std::get<0>(x.t)), Put(", "), Walk(std::get<1>(x.t));
    Walk("(", std::get<std::optional<ArraySpec>>(x.t), ")"), Put(')');
    return false;
  }
  bool Pre(const StructureStmt &x) {
    Put("STRUCTURE ");
    if (std::get<bool>(x.t)) {  // slashes around name
      Put('/'), Walk(std::get<Name>(x.t)), Put('/');
      Walk(" ", std::get<std::list<EntityDecl>>(x.t), ", ");
    } else {
      CHECK(std::get<std::list<EntityDecl>>(x.t).empty());
      Walk(std::get<Name>(x.t));
    }
    Indent();
    percentOrDot_ = '.';  // TODO: this is so lame
    return false;
  }
  void Post(const Union::UnionStmt &) { Put("UNION"), Indent(); }
  void Post(const Union::EndUnionStmt &) { Outdent(), Put("END UNION"); }
  void Post(const Map::MapStmt &) { Put("MAP"), Indent(); }
  void Post(const Map::EndMapStmt &) { Outdent(), Put("END MAP"); }
  void Post(const StructureDef::EndStructureStmt &) {
    Outdent(), Put("END STRUCTURE");
  }
  bool Pre(const OldParameterStmt &x) {
    Put("PARAMETER "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ArithmeticIfStmt &x) {
    Put("IF ("), Walk(std::get<Expr>(x.t)), Put(") ");
    Walk(std::get<1>(x.t)), Put(", ");
    Walk(std::get<2>(x.t)), Put(", ");
    Walk(std::get<3>(x.t));
    return false;
  }
  bool Pre(const AssignStmt &x) {
    Put("ASSIGN "), Walk(std::get<Label>(x.t));
    Put(" TO "), Walk(std::get<Name>(x.t));
    return false;
  }
  bool Pre(const AssignedGotoStmt &x) {
    Put("GO TO "), Walk(std::get<Name>(x.t));
    Walk(", (", std::get<std::list<Label>>(x.t), ", ", ")");
    return false;
  }
  bool Pre(const PauseStmt &x) {
    Put("PAUSE"), Walk(" ", x.v);
    return false;
  }

#define WALK_NESTED_ENUM(ENUMTYPE) \
  bool Pre(const ENUMTYPE &x) { \
    PutEnum(static_cast<int>(x), ENUMTYPE##AsString); \
    return false; \
  }
  WALK_NESTED_ENUM(AccessSpec::Kind)  // R807
  WALK_NESTED_ENUM(TypeParamDefStmt::KindOrLen)  // R734
  WALK_NESTED_ENUM(IntentSpec::Intent)  // R826
  WALK_NESTED_ENUM(ImplicitStmt::ImplicitNoneNameSpec)  // R866
  WALK_NESTED_ENUM(ConnectSpec::CharExpr::Kind)  // R1205
  WALK_NESTED_ENUM(IoControlSpec::CharExpr::Kind)
  WALK_NESTED_ENUM(InquireSpec::CharVar::Kind)
  WALK_NESTED_ENUM(InquireSpec::IntVar::Kind)
  WALK_NESTED_ENUM(InquireSpec::LogVar::Kind)
  WALK_NESTED_ENUM(ProcedureStmt::Kind)  // R1506
  WALK_NESTED_ENUM(UseStmt::ModuleNature)  // R1410
#undef WALK_NESTED_ENUM

  void Done() const { CHECK(indent_ == 0); }

private:
  void Put(char);
  void Put(const char *);
  void Put(const std::string &);
  void PutUpperCase(const std::string &);
  void PutQuoted(const std::string &);
  void PutEnum(int, const char *);
  void Indent() { indent_ += indentationAmount_; }
  void Outdent() {
    CHECK(indent_ >= indentationAmount_);
    indent_ -= indentationAmount_;
  }

  // Call back to the traversal framework.
  template<typename T> void Walk(const T &x) {
    Fortran::parser::Walk(x, *this);
  }

  // Traverse a std::optional<> value.  Emit a prefix and/or a suffix string
  // only when it contains a value.
  template<typename A>
  void Walk(
      const char *prefix, const std::optional<A> &x, const char *suffix = "") {
    if (x.has_value()) {
      Put(prefix), Walk(*x), Put(suffix);
    }
  }
  template<typename A>
  void Walk(const std::optional<A> &x, const char *suffix = "") {
    return Walk("", x, suffix);
  }

  // Traverse a std::list<>.  Separate the elements with an optional string.
  // Emit a prefix and/or a suffix string only when the list is not empty.
  template<typename A>
  void Walk(const char *prefix, const std::list<A> &list,
      const char *comma = ", ", const char *suffix = "") {
    if (!list.empty()) {
      const char *str{prefix};
      for (const auto &x : list) {
        Put(str), Walk(x);
        str = comma;
      }
      Put(suffix);
    }
  }
  template<typename A>
  void Walk(const std::list<A> &list, const char *comma = ", ",
      const char *suffix = "") {
    return Walk("", list, comma, suffix);
  }

  // Traverse a std::tuple<>, with an optional separator.
  template<size_t J = 0, typename T>
  void WalkTupleElements(const T &tuple, const char *separator) {
    if constexpr (J < std::tuple_size_v<T>) {
      if (J > 0) {
        Put(separator);
      }
      Walk(std::get<J>(tuple));
      WalkTupleElements<J + 1>(tuple, separator);
    }
  }
  template<typename... A>
  void Walk(const std::tuple<A...> &tuple, const char *separator = "") {
    WalkTupleElements(tuple, separator);
  }

  std::ostream &out_;
  int indent_{0};
  const int indentationAmount_{1};
  int column_{1};
  const int maxColumns_{80};
  char percentOrDot_{'%'};
  Encoding encoding_{Encoding::UTF8};
};

void UnparseVisitor::Put(char ch) {
  if (column_ <= 1) {
    if (ch == '\n') {
      return;
    }
    for (int j{0}; j < indent_; ++j) {
      out_ << ' ';
    }
    column_ = indent_ + 2;
  } else if (ch == '\n') {
    column_ = 1;
  } else if (++column_ >= maxColumns_) {
    out_ << "&\n";
    for (int j{0}; j < indent_; ++j) {
      out_ << ' ';
    }
    out_ << '&';
    column_ = indent_ + 3;
  }
  out_ << ch;
}

void UnparseVisitor::Put(const char *str) {
  for (; *str != '\0'; ++str) {
    Put(*str);
  }
}

void UnparseVisitor::Put(const std::string &str) {
  for (char ch : str) {
    Put(ch);
  }
}

void UnparseVisitor::PutUpperCase(const std::string &str) {
  for (char ch : str) {
    Put(ToUpperCaseLetter(ch));
  }
}

void UnparseVisitor::PutQuoted(const std::string &str) {
  Put('"');
  const auto emit = [&](char ch) { Put(ch); };
  for (char ch : str) {
    EmitQuotedChar(ch, emit, emit);
  }
  Put('"');
}

void UnparseVisitor::PutEnum(int n, const char *enumNames) {
  const char *p{enumNames};
  for (; n > 0; --n, ++p) {
    for (; *p && *p != ','; ++p) {
    }
  }
  while (*p == ' ') {
    ++p;
  }
  CHECK(*p != '\0');
  for (; *p && *p != ' ' && *p != ','; ++p) {
    Put(ToUpperCaseLetter(*p));
  }
}

void Unparse(std::ostream &out, const Program &program, Encoding encoding) {
  UnparseVisitor visitor{out, 1, encoding};
  Walk(program, visitor);
  visitor.Done();
}
}  // namespace parser
}  // namespace Fortran
