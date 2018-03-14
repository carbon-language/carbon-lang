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
  UnparseVisitor(std::ostream &out, int indentationAmount, Encoding encoding,
      bool capitalize)
    : out_{out}, indentationAmount_{indentationAmount}, encoding_{encoding},
      capitalizeKeywords_{capitalize} {}

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
    Word("TYPE("), Walk(x.derived), Put(')');
    return false;
  }
  bool Pre(const DeclarationTypeSpec::Class &x) {
    Word("CLASS("), Walk(x.derived), Put(')');
    return false;
  }
  void Post(const DeclarationTypeSpec::ClassStar &) { Word("CLASS(*)"); }
  void Post(const DeclarationTypeSpec::TypeStar &) { Word("TYPE(*)"); }
  bool Pre(const DeclarationTypeSpec::Record &x) {
    Word("RECORD/"), Walk(x.v), Put('/');
    return false;
  }
  bool Pre(const IntrinsicTypeSpec::Real &x) {  // R704
    Word("REAL");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Complex &x) {
    Word("COMPLEX");
    return true;
  }
  void Post(const IntrinsicTypeSpec::DoublePrecision &) {
    Word("DOUBLE PRECISION");
  }
  bool Pre(const IntrinsicTypeSpec::Character &x) {
    Word("CHARACTER");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Logical &x) {
    Word("LOGICAL");
    return true;
  }
  void Post(const IntrinsicTypeSpec::DoubleComplex &) {
    Word("DOUBLE COMPLEX");
  }
  bool Pre(const IntrinsicTypeSpec::NCharacter &x) {
    Word("NCHARACTER");
    return true;
  }
  bool Pre(const IntegerTypeSpec &x) {  // R705
    Word("INTEGER");
    return true;
  }
  bool Pre(const KindSelector &x) {  // R706
    std::visit(
        visitors{[&](const ScalarIntConstantExpr &y) {
                   Put('('), Word("KIND="), Walk(y), Put(')');
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
    Put('('), Word("KIND="), Walk(x.kind);
    Walk(", LEN=", x.length), Put(')');
    return false;
  }
  bool Pre(const LengthSelector &x) {  // R722
    std::visit(visitors{[&](const TypeParamValue &y) {
                          Put('('), Word("LEN="), Walk(y), Put(')');
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
        Word("NC");
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
    Word("TYPE"), Walk(", ", std::get<std::list<TypeAttrSpec>>(x.t), ", ");
    Put(" :: "), Put(std::get<Name>(x.t));
    Walk("(", std::get<std::list<Name>>(x.t), ", ", ")");
    Indent();
    return false;
  }
  bool Pre(const Abstract &x) {  // R728, &c.
    Word("ABSTRACT");
    return false;
  }
  bool Pre(const TypeAttrSpec::BindC &x) {
    Word("BIND(C)");
    return false;
  }
  bool Pre(const TypeAttrSpec::Extends &x) {
    Word("EXTENDS("), Walk(x.v), Put(')');
    return false;
  }
  bool Pre(const EndTypeStmt &x) {  // R730
    Outdent(), Word("END TYPE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const SequenceStmt &x) {  // R731
    Word("SEQUENCE");
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
    Word("ALLOCATABLE");
    return false;
  }
  bool Pre(const Pointer &x) {
    Word("POINTER");
    return false;
  }
  bool Pre(const Contiguous &x) {
    Word("CONTIGUOUS");
    return false;
  }
  bool Pre(const ComponentAttrSpec &x) {
    std::visit(visitors{[&](const CoarraySpec &) { Word("CODIMENSION["); },
                   [&](const ComponentArraySpec &) { Word("DIMENSION("); },
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
    Word("PROCEDURE(");
    Walk(std::get<std::optional<ProcInterface>>(x.t)), Put(')');
    Walk(", ", std::get<std::list<ProcComponentAttrSpec>>(x.t), ", ");
    Put(" :: "), Walk(std::get<std::list<ProcDecl>>(x.t), ", ");
    return false;
  }
  bool Pre(const NoPass &x) {  // R742
    Word("NOPASS");
    return false;
  }
  bool Pre(const Pass &x) {
    Word("PASS"), Walk("(", x.v, ")");
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
    Word("PRIVATE");
    return false;
  }
  bool Pre(const TypeBoundProcedureStmt::WithoutInterface &x) {  // R749
    Word("PROCEDURE"), Walk(", ", x.attributes, ", ");
    Put(" :: "), Walk(x.declarations);
    return false;
  }
  bool Pre(const TypeBoundProcedureStmt::WithInterface &x) {
    Word("PROCEDURE("), Walk(x.interfaceName), Put("), ");
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
    Word("GENERIC"), Walk(", ", std::get<std::optional<AccessSpec>>(x.t));
    Put(" :: "), Walk(std::get<Indirection<GenericSpec>>(x.t));
    Put(" => "), Walk(std::get<std::list<Name>>(x.t), ", ");
    return false;
  }
  void Post(const BindAttr::Deferred &) { Word("DEFERRED"); }  // R752
  void Post(const BindAttr::Non_Overridable &) { Word("NON_OVERRIDABLE"); }
  void Post(const FinalProcedureStmt &) {  // R753
    Word("FINAL :: ");
  }
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
    Word("ENUM "), Word("BIND(C)"), Indent();
    return false;
  }
  bool Pre(const EnumeratorDefStmt &) {  // R761
    Word("ENUMERATOR :: ");
    return true;
  }
  bool Pre(const Enumerator &x) {  // R762
    Walk(std::get<NamedConstant>(x.t));
    Walk(" = ", std::get<std::optional<ScalarIntConstantExpr>>(x.t));
    return false;
  }
  void Post(const EndEnumStmt &) {  // R763
    Outdent();
    Word("END ENUM");
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
  bool Pre(const AcImpliedDo &x) {  // R774
    Put('('), Walk(std::get<std::list<AcValue>>(x.t), ", ");
    Put(", "), Walk(std::get<AcImpliedDoControl>(x.t)), Put(')');
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
    std::visit(visitors{[&](const CoarraySpec &y) { Word("CODIMENSION["); },
                   [&](const ArraySpec &y) { Word("DIMENSION("); },
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
    Word("NULL()");
    return false;
  }
  bool Pre(const LanguageBindingSpec &x) {  // R808 & R1528
    Word("BIND(C"), Walk(", NAME=", x.v), Put(')');
    return false;
  }
  bool Pre(const CoarraySpec &x) {  // R809
    std::visit(visitors{[&](const DeferredCoshapeSpecList &y) { Walk(y); },
                   [&](const ExplicitCoshapeSpec &y) { Walk(y); }},
        x.u);
    return false;
  }
  bool Pre(const DeferredCoshapeSpecList &x) {  // R810
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
  bool Pre(const DeferredShapeSpecList &x) {  // R820
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
  void Post(const Asynchronous &) { Word("ASYNCHRONOUS"); }
  void Post(const External &) { Word("EXTERNAL"); }
  void Post(const Intrinsic &) { Word("INTRINSIC"); }
  void Post(const Optional &) { Word("OPTIONAL"); }
  void Post(const Parameter &) { Word("PARAMETER"); }
  void Post(const Protected &) { Word("PROTECTED"); }
  void Post(const Save &) { Word("SAVE"); }
  void Post(const Target &) { Word("TARGET"); }
  void Post(const Value &) { Word("VALUE"); }
  void Post(const Volatile &) { Word("VOLATILE"); }
  bool Pre(const IntentSpec &x) {  // R826
    Word("INTENT("), Walk(x.v), Put(")");
    return false;
  }
  bool Pre(const AccessStmt &x) {  // R827
    Walk(std::get<AccessSpec>(x.t));
    Walk(" :: ", std::get<std::list<AccessId>>(x.t), ", ");
    return false;
  }
  bool Pre(const AllocatableStmt &x) {  // R829
    Word("ALLOCATABLE :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ObjectDecl &x) {  // R830 & R860
    Walk(std::get<ObjectName>(x.t));
    Walk("(", std::get<std::optional<ArraySpec>>(x.t), ")");
    Walk("[", std::get<std::optional<CoarraySpec>>(x.t), "]");
    return false;
  }
  bool Pre(const AsynchronousStmt &x) {  // R831
    Word("ASYNCHRONOUS :: "), Walk(x.v, ", ");
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
    Word("CODIMENSION :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const CodimensionDecl &x) {  // R835
    Walk(std::get<Name>(x.t));
    Put('['), Walk(std::get<CoarraySpec>(x.t)), Put(']');
    return false;
  }
  bool Pre(const ContiguousStmt &x) {  // R836
    Word("CONTIGUOUS :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const DataStmt &x) {  // R837
    Word("DATA "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const DataStmtSet &x) {  // R838
    Walk(std::get<std::list<DataStmtObject>>(x.t), ", ");
    Put('/'), Walk(std::get<std::list<DataStmtValue>>(x.t), ", "), Put('/');
    return false;
  }
  bool Pre(const DataImpliedDo &x) {  // R840, R842
    Put('('), Walk(std::get<std::list<DataIDoObject>>(x.t), ", "), Put(',');
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
    Word("DIMENSION :: "), Walk(x.v, ", ");
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
    Word("OPTIONAL :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ParameterStmt &x) {  // R851
    Word("PARAMETER("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const NamedConstantDef &x) {  // R852
    Walk(x.t, "=");
    return false;
  }
  bool Pre(const PointerStmt &x) {  // R853
    Word("POINTER :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ProtectedStmt &x) {  // R855
    Word("PROTECTED :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const SaveStmt &x) {  // R856
    Word("SAVE"), Walk(" :: ", x.v, ", ");
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
    Word("TARGET :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ValueStmt &x) {  // R861
    Word("VALUE :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const VolatileStmt &x) {  // R862
    Word("VOLATILE :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ImplicitStmt &x) {  // R863
    Word("IMPLICIT ");
    std::visit(
        visitors{[&](const std::list<ImplicitSpec> &y) { Walk(y, ", "); },
            [&](const std::list<ImplicitStmt::ImplicitNoneNameSpec> &y) {
              Word("NONE"), Walk(" (", y, ", ", ")");
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
    Word("IMPORT");
    switch (x.kind) {
    case ImportStmt::Kind::Default:
      Put(" :: ");
      Walk(x.names);
      break;
    case ImportStmt::Kind::Only:
      Put(", "), Word("ONLY: ");
      Walk(x.names);
      break;
    case ImportStmt::Kind::None: Word(", NONE"); break;
    case ImportStmt::Kind::All: Word(", ALL"); break;
    default: CRASH_NO_CASE;
    }
    return false;
  }
  bool Pre(const NamelistStmt &x) {  // R868
    Word("NAMELIST"), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const NamelistStmt::Group &x) {
    Put('/'), Put(std::get<Name>(x.t)), Put('/');
    Walk(std::get<std::list<Name>>(x.t), ", ");
    return false;
  }
  bool Pre(const EquivalenceStmt &x) {  // R870, R871
    Word("EQUIVALENCE");
    const char *separator{" "};
    for (const std::list<EquivalenceObject> &y : x.v) {
      Put(separator), Put('('), Walk(y), Put(')');
      separator = ", ";
    }
    return false;
  }
  bool Pre(const CommonStmt &x) {  // R873
    Word("COMMON ");
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
    Word("STAT=");
    return true;
  }
  bool Pre(const ImageSelectorSpec::Team &) {
    Word("TEAM=");
    return true;
  }
  bool Pre(const ImageSelectorSpec::Team_Number &) {
    Word("TEAM_NUMBER=");
    return true;
  }
  bool Pre(const AllocateStmt &x) {  // R927
    Word("ALLOCATE(");
    Walk(std::get<std::optional<TypeSpec>>(x.t), "::");
    Walk(std::get<std::list<Allocation>>(x.t), ", ");
    Walk(", ", std::get<std::list<AllocOpt>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const AllocOpt &x) {  // R928, R931
    std::visit(visitors{[&](const AllocOpt::Mold &) { Word("MOLD="); },
                   [&](const AllocOpt::Source &) { Word("SOURCE="); },
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
    Word("NULLIFY("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const DeallocateStmt &x) {  // R941
    Word("DEALLOCATE(");
    Walk(std::get<std::list<AllocateObject>>(x.t), ", ");
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const StatOrErrmsg &x) {  // R942 & R1165
    std::visit(visitors{[&](const StatVariable &) { Word("STAT="); },
                   [&](const MsgVariable &) { Word("ERRMSG="); }},
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
    Word(".NOT.");
    return true;
  }
  bool Pre(const Expr::PercentLoc &x) {
    Word("%LOC("), Walk(x.v), Put(')');
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
    Word("WHERE ("), Walk(x.t, ") ");
    return false;
  }
  bool Pre(const WhereConstructStmt &x) {  // R1043
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("WHERE ("), Walk(std::get<LogicalExpr>(x.t)), Put(')');
    Indent();
    return false;
  }
  bool Pre(const MaskedElsewhereStmt &x) {  // R1047
    Outdent();
    Word("ELSEWHERE ("), Walk(std::get<LogicalExpr>(x.t)), Put(')');
    Walk(" ", std::get<std::optional<Name>>(x.t));
    Indent();
    return false;
  }
  bool Pre(const ElsewhereStmt &x) {  // R1048
    Outdent(), Word("ELSEWHERE"), Walk(" ", x.v), Indent();
    return false;
  }
  bool Pre(const EndWhereStmt &x) {  // R1049
    Outdent(), Word("END WHERE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ForallConstructStmt &x) {  // R1051
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("FORALL"), Walk(std::get<Indirection<ConcurrentHeader>>(x.t));
    Indent();
    return false;
  }
  bool Pre(const EndForallStmt &x) {  // R1054
    Outdent(), Word("END FORALL"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ForallStmt &) {  // R1055
    Word("FORALL");
    return true;
  }

  bool Pre(const AssociateStmt &x) {  // R1103
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("ASSOCIATE (");
    Walk(std::get<std::list<Association>>(x.t), ", "), Put(')'), Indent();
    return false;
  }
  bool Pre(const Association &x) {  // R1104
    Walk(x.t, " => ");
    return false;
  }
  bool Pre(const EndAssociateStmt &x) {  // R1106
    Outdent(), Word("END ASSOCIATE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const BlockStmt &x) {  // R1108
    Walk(x.v, ": "), Word("BLOCK"), Indent();
    return false;
  }
  bool Pre(const EndBlockStmt &x) {  // R1110
    Outdent(), Word("END BLOCK"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ChangeTeamStmt &x) {  // R1112
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("CHANGE TEAM ("), Walk(std::get<TeamVariable>(x.t));
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
    Outdent(), Word("END TEAM (");
    Walk(std::get<std::list<StatOrErrmsg>>(x.t), ", ");
    Put(')'), Walk(" ", std::get<std::optional<Name>>(x.t));
    return false;
  }
  bool Pre(const CriticalStmt &x) {  // R1117
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("CRITICAL ("), Walk(std::get<std::list<StatOrErrmsg>>(x.t), ", ");
    Put(')'), Indent();
    return false;
  }
  bool Pre(const EndCriticalStmt &x) {  // R1118
    Outdent(), Word("END CRITICAL"), Walk(" ", x.v);
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
    Word("DO "), Walk(std::get<Label>(x.t));
    Walk(" ", std::get<std::optional<LoopControl>>(x.t));
    return false;
  }
  bool Pre(const NonLabelDoStmt &x) {  // R1122
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("DO "), Walk(std::get<std::optional<LoopControl>>(x.t));
    return false;
  }
  bool Pre(const LoopControl &x) {  // R1123
    std::visit(visitors{[&](const ScalarLogicalExpr &y) {
                          Word("WHILE ("), Walk(y), Put(')');
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
    Word("CONCURRENT");
    return true;
  }
  bool Pre(const LocalitySpec::Local &x) {
    Word("LOCAL("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const LocalitySpec::LocalInit &x) {
    Word("LOCAL INIT("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const LocalitySpec::Shared &x) {
    Word("SHARED("), Walk(x.v, ", "), Put(')');
    return false;
  }
  void Post(const LocalitySpec::DefaultNone &x) { Word("DEFAULT(NONE)"); }
  bool Pre(const EndDoStmt &x) {  // R1132
    Word("END DO"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const CycleStmt &x) {  // R1133
    Word("CYCLE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const IfThenStmt &x) {  // R1135
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("IF ("), Walk(std::get<ScalarLogicalExpr>(x.t));
    Put(") "), Word("THEN"), Indent();
    return false;
  }
  bool Pre(const ElseIfStmt &x) {  // R1136
    Outdent(), Word("ELSE IF (");
    Walk(std::get<ScalarLogicalExpr>(x.t)), Put(") "), Word("THEN");
    Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
    return false;
  }
  bool Pre(const ElseStmt &x) {  // R1137
    Outdent(), Word("ELSE"), Walk(" ", x.v), Indent();
    return false;
  }
  bool Pre(const EndIfStmt &x) {  // R1138
    Outdent(), Word("END IF"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const IfStmt &x) {  // R1139
    Word("IF ("), Walk(x.t, ") ");
    return false;
  }
  bool Pre(const SelectCaseStmt &x) {  // R1141, R1144
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("SELECT CASE (");
    Walk(std::get<Scalar<Expr>>(x.t)), Put(')'), Indent();
    return false;
  }
  bool Pre(const CaseStmt &x) {  // R1142
    Outdent(), Word("CASE "), Walk(std::get<CaseSelector>(x.t));
    Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
    return false;
  }
  bool Pre(const EndSelectStmt &x) {  // R1143 & R1151 & R1155
    Outdent(), Word("END SELECT"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const CaseSelector &x) {  // R1145
    std::visit(visitors{[&](const std::list<CaseValueRange> &y) {
                          Put('('), Walk(y), Put(')');
                        },
                   [&](const Default &) { Word("DEFAULT"); }},
        x.u);
    return false;
  }
  bool Pre(const CaseValueRange::Range &x) {  // R1146
    Walk(x.lower), Put(':'), Walk(x.upper);
    return false;
  }
  bool Pre(const SelectRankStmt &x) {  // R1149
    Walk(std::get<0>(x.t), ": ");
    Word("SELECT RANK ("), Walk(std::get<1>(x.t), " => ");
    Walk(std::get<Selector>(x.t)), Put(')'), Indent();
    return false;
  }
  bool Pre(const SelectRankCaseStmt &x) {  // R1150
    Outdent(), Word("RANK ");
    std::visit(visitors{[&](const ScalarIntConstantExpr &y) {
                          Put('('), Walk(y), Put(')');
                        },
                   [&](const Star &) { Put("(*)"); },
                   [&](const Default &) { Word("DEFAULT"); }},
        std::get<SelectRankCaseStmt::Rank>(x.t).u);
    Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
    return false;
  }
  bool Pre(const SelectTypeStmt &x) {  // R1153
    Walk(std::get<0>(x.t), ": ");
    Word("SELECT TYPE ("), Walk(std::get<1>(x.t), " => ");
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
                          Word("TYPE IS ("), Walk(y), Put(')');
                        },
                   [&](const DerivedTypeSpec &y) {
                     Word("CLASS IS ("), Walk(y), Put(')');
                   },
                   [&](const Default &) { Word("CLASS DEFAULT"); }},
        x.u);
    return false;
  }
  bool Pre(const ExitStmt &x) {  // R1156
    Word("EXIT"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const GotoStmt &x) {  // R1157
    Word("GO TO ");
    return true;
  }
  bool Pre(const ComputedGotoStmt &x) {  // R1158
    Word("GO TO ("), Walk(x.t, "), ");
    return false;
  }
  bool Pre(const ContinueStmt &x) {  // R1159
    Word("CONTINUE");
    return false;
  }
  bool Pre(const StopStmt &x) {  // R1160, R1161
    if (std::get<StopStmt::Kind>(x.t) == StopStmt::Kind::ErrorStop) {
      Word("ERROR ");
    }
    Word("STOP"), Walk(" ", std::get<std::optional<StopCode>>(x.t));
    Walk(", QUIET=", std::get<std::optional<ScalarLogicalExpr>>(x.t));
    return false;
  }
  bool Pre(const FailImageStmt &x) {  // R1163
    Word("FAIL IMAGE");
    return false;
  }
  bool Pre(const SyncAllStmt &x) {  // R1164
    Word("SYNC ALL ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const SyncImagesStmt &x) {  // R1166
    Word("SYNC IMAGES (");
    Walk(std::get<SyncImagesStmt::ImageSet>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const SyncMemoryStmt &x) {  // R1168
    Word("SYNC MEMORY ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const SyncTeamStmt &x) {  // R1169
    Word("SYNC TEAM ("), Walk(std::get<TeamVariable>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const EventPostStmt &x) {  // R1170
    Word("EVENT POST ("), Walk(std::get<EventVariable>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
    return false;
  }
  bool Pre(const EventWaitStmt::EventWaitSpec &x) {  // R1173, R1174
    std::visit(
        visitors{[&](const ScalarIntExpr &x) { Word("UNTIL_COUNT="), Walk(x); },
            [&](const StatOrErrmsg &y) {}},
        x.u);
    return true;
  }
  bool Pre(const EventWaitStmt &x) {  // R1170
    Word("EVENT WAIT ("), Walk(std::get<EventVariable>(x.t));
    Walk(", ", std::get<std::list<EventWaitStmt::EventWaitSpec>>(x.t), ", ");
    Put(')');
    return false;
  }
  bool Pre(const FormTeamStmt &x) {  // R1175
    Word("FORM TEAM ("), Walk(std::get<ScalarIntExpr>(x.t));
    Put(','), Walk(std::get<TeamVariable>(x.t));
    Walk(", ", std::get<std::list<FormTeamStmt::FormTeamSpec>>(x.t), ", ");
    Put(')');
    return false;
  }
  bool Pre(const FormTeamStmt::FormTeamSpec &x) {  // R1176, R1177
    std::visit(
        visitors{[&](const ScalarIntExpr &x) { Word("NEW_INDEX="), Walk(x); },
            [&](const StatOrErrmsg &y) {}},
        x.u);
    return true;
  }
  bool Pre(const LockStmt &x) {  // R1178
    Word("LOCK ("), Walk(std::get<LockVariable>(x.t));
    Walk(", ", std::get<std::list<LockStmt::LockStat>>(x.t), ", ");
    Put(')');
    return false;
  }
  bool Pre(const LockStmt::LockStat &x) {  // R1179
    std::visit(visitors{[&](const ScalarLogicalVariable &x) {
                          Word("ACQUIRED_LOCK="), Walk(x);
                        },
                   [&](const StatOrErrmsg &y) {}},
        x.u);
    return true;
  }
  bool Pre(const UnlockStmt &x) {  // R1180
    Word("UNLOCK ("), Walk(std::get<LockVariable>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", ");
    Put(')');
    return false;
  }

  bool Pre(const OpenStmt &x) {  // R1204
    Word("OPEN ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const ConnectSpec &x) {  // R1205
    return std::visit(visitors{[&](const FileUnitNumber &) {
                                 Word("UNIT=");
                                 return true;
                               },
                          [&](const FileNameExpr &) {
                            Word("FILE=");
                            return true;
                          },
                          [&](const ConnectSpec::CharExpr &y) {
                            Walk(y.t, "=");
                            return false;
                          },
                          [&](const MsgVariable &) {
                            Word("IOMSG=");
                            return true;
                          },
                          [&](const StatVariable &) {
                            Word("IOSTAT=");
                            return true;
                          },
                          [&](const ConnectSpec::Recl &) {
                            Word("RECL=");
                            return true;
                          },
                          [&](const ConnectSpec::Newunit &) {
                            Word("NEWUNIT=");
                            return true;
                          },
                          [&](const ErrLabel &) {
                            Word("ERR=");
                            return true;
                          },
                          [&](const StatusExpr &) {
                            Word("STATUS=");
                            return true;
                          }},
        x.u);
  }
  bool Pre(const CloseStmt &x) {  // R1208
    Word("CLOSE ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const CloseStmt::CloseSpec &x) {  // R1209
    std::visit(visitors{[&](const FileUnitNumber &) { Word("UNIT="); },
                   [&](const StatVariable &) { Word("IOSTAT="); },
                   [&](const MsgVariable &) { Word("IOMSG="); },
                   [&](const ErrLabel &) { Word("ERR="); },
                   [&](const StatusExpr &) { Word("STATUS="); }},
        x.u);
    return true;
  }
  bool Pre(const ReadStmt &x) {  // R1210
    Word("READ ");
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
    Word("WRITE (");
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
    Word("PRINT "), Walk(std::get<Format>(x.t));
    Walk(", ", std::get<std::list<OutputItem>>(x.t), ", ");
    return false;
  }
  bool Pre(const IoControlSpec &x) {  // R1213
    return std::visit(visitors{[&](const IoUnit &) {
                                 Word("UNIT=");
                                 return true;
                               },
                          [&](const Format &) {
                            Word("FMT=");
                            return true;
                          },
                          [&](const Name &) {
                            Word("NML=");
                            return true;
                          },
                          [&](const IoControlSpec::CharExpr &y) {
                            Walk(y.t, "=");
                            return false;
                          },
                          [&](const IoControlSpec::Asynchronous &) {
                            Word("ASYNCHRONOUS=");
                            return true;
                          },
                          [&](const EndLabel &) {
                            Word("END=");
                            return true;
                          },
                          [&](const EorLabel &) {
                            Word("EOR=");
                            return true;
                          },
                          [&](const ErrLabel &) {
                            Word("ERR=");
                            return true;
                          },
                          [&](const IdVariable &) {
                            Word("ID=");
                            return true;
                          },
                          [&](const MsgVariable &) {
                            Word("IOMSG=");
                            return true;
                          },
                          [&](const StatVariable &) {
                            Word("IOSTAT=");
                            return true;
                          },
                          [&](const IoControlSpec::Pos &) {
                            Word("POS=");
                            return true;
                          },
                          [&](const IoControlSpec::Rec &) {
                            Word("REC=");
                            return true;
                          },
                          [&](const IoControlSpec::Size &) {
                            Word("SIZE=");
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
    Word("WAIT ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const WaitSpec &x) {  // R1223
    std::visit(visitors{[&](const FileUnitNumber &) { Word("UNIT="); },
                   [&](const EndLabel &) { Word("END="); },
                   [&](const EorLabel &) { Word("EOR="); },
                   [&](const ErrLabel &) { Word("ERR="); },
                   [&](const IdExpr &) { Word("ID="); },
                   [&](const MsgVariable &) { Word("IOMSG="); },
                   [&](const StatVariable &) { Word("IOSTAT="); }},
        x.u);
    return true;
  }
  bool Pre(const BackspaceStmt &x) {  // R1224
    Word("BACKSPACE ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const EndfileStmt &x) {  // R1225
    Word("ENDFILE ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const RewindStmt &x) {  // R1226
    Word("REWIND ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const FlushStmt &x) {  // R1228
    Word("FLUSH ("), Walk(x.v, ", "), Put(')');
    return false;
  }
  bool Pre(const InquireStmt &x) {  // R1230
    Word("INQUIRE (");
    std::visit(
        visitors{[&](const InquireStmt::Iolength &y) {
                   Word("IOLENGTH="), Walk(y.t, ") ");
                 },
            [&](const std::list<InquireSpec> &y) { Walk(y, ", "), Put(')'); }},
        x.u);
    return false;
  }
  bool Pre(const InquireSpec &x) {  // R1231
    return std::visit(visitors{[&](const FileUnitNumber &) {
                                 Word("UNIT=");
                                 return true;
                               },
                          [&](const FileNameExpr &) {
                            Word("FILE=");
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
                            Word("ID=");
                            return true;
                          },
                          [&](const ErrLabel &) {
                            Word("ERR=");
                            return true;
                          }},
        x.u);
  }

  bool Pre(const FormatStmt &) {  // R1301
    Word("FORMAT");
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
    Word("DT");
    if (!x.type.empty()) {
      Put('"'), Put(x.type), Put('"');
    }
    Walk("(", x.parameters, ",", ")");
    return false;
  }
  bool Pre(const format::ControlEditDesc &x) {  // R1313, R1315-R1320
    switch (x.kind) {
    case format::ControlEditDesc::Kind::T:
      Word("T");
      Walk(x.count);
      break;
    case format::ControlEditDesc::Kind::TL:
      Word("TL");
      Walk(x.count);
      break;
    case format::ControlEditDesc::Kind::TR:
      Word("TR");
      Walk(x.count);
      break;
    case format::ControlEditDesc::Kind::X:
      if (x.count != 1) {
        Walk(x.count);
      }
      Word("X");
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
      Word("P");
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
    Word("PROGRAM "), Indent();
    return true;
  }
  bool Pre(const EndProgramStmt &x) {  // R1403
    Outdent(), Word("END PROGRAM"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ModuleStmt &) {  // R1405
    Word("MODULE "), Indent();
    return true;
  }
  bool Pre(const EndModuleStmt &x) {  // R1406
    Outdent(), Word("END MODULE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const UseStmt &x) {  // R1409
    Word("USE"), Walk(", ", x.nature), Put(" :: "), Walk(x.moduleName);
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
    Word("SUBMODULE "), Indent();
    return true;
  }
  bool Pre(const ParentIdentifier &x) {  // R1418
    Walk(std::get<Name>(x.t)), Walk(":", std::get<std::optional<Name>>(x.t));
    return false;
  }
  bool Pre(const EndSubmoduleStmt &x) {  // R1419
    Outdent(), Word("END SUBMODULE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const BlockDataStmt &x) {  // R1421
    Word("BLOCK DATA"), Walk(" ", x.v), Indent();
    return false;
  }
  bool Pre(const EndBlockDataStmt &x) {  // R1422
    Outdent(), Word("END BLOCK DATA"), Walk(" ", x.v);
    return false;
  }

  bool Pre(const InterfaceStmt &x) {  // R1503
    std::visit(visitors{[&](const std::optional<GenericSpec> &y) {
                          Word("INTERFACE"), Walk(" ", y);
                        },
                   [&](const Abstract &) { Word("ABSTRACT INTERFACE"); }},
        x.u);
    Indent();
    return false;
  }
  bool Pre(const EndInterfaceStmt &x) {  // R1504
    Outdent(), Word("END INTERFACE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ProcedureStmt &x) {  // R1506
    if (std::get<ProcedureStmt::Kind>(x.t) ==
        ProcedureStmt::Kind::ModuleProcedure) {
      Word("MODULE ");
    }
    Word("PROCEDURE :: ");
    Walk(std::get<std::list<Name>>(x.t), ", ");
    return false;
  }
  bool Pre(const GenericSpec &x) {  // R1508, R1509
    std::visit(visitors{[&](const GenericSpec::Assignment &) {
                          Word("ASSIGNMENT(=)");
                        },
                   [&](const GenericSpec::ReadFormatted &) {
                     Word("READ(FORMATTED)");
                   },
                   [&](const GenericSpec::ReadUnformatted &) {
                     Word("READ(UNFORMATTED)");
                   },
                   [&](const GenericSpec::WriteFormatted &) {
                     Word("WRITE(FORMATTED)");
                   },
                   [&](const GenericSpec::WriteUnformatted &) {
                     Word("WRITE(UNFORMATTED)");
                   },
                   [&](const auto &y) {}},
        x.u);
    return true;
  }
  bool Pre(const GenericStmt &x) {  // R1510
    Word("GENERIC"), Walk(", ", std::get<std::optional<AccessSpec>>(x.t));
    Put(" :: "), Walk(std::get<GenericSpec>(x.t)), Put(" => ");
    Walk(std::get<std::list<Name>>(x.t), ", ");
    return false;
  }
  bool Pre(const ExternalStmt &x) {  // R1511
    Word("EXTERNAL :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ProcedureDeclarationStmt &x) {  // R1512
    Word("PROCEDURE ("), Walk(std::get<std::optional<ProcInterface>>(x.t));
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
    Word("INTRINSIC :: "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const FunctionReference &x) {  // R1520
    Walk(std::get<ProcedureDesignator>(x.v.t));
    Put('('), Walk(std::get<std::list<ActualArgSpec>>(x.v.t), ", "), Put(')');
    return false;
  }
  bool Pre(const CallStmt &x) {  // R1521
    Word("CALL "), Walk(std::get<ProcedureDesignator>(x.v.t));
    Walk(" (", std::get<std::list<ActualArgSpec>>(x.v.t), ", ", ")");
    return false;
  }
  bool Pre(const ActualArgSpec &x) {  // R1523
    Walk(std::get<std::optional<Keyword>>(x.t), "=");
    Walk(std::get<ActualArg>(x.t));
    return false;
  }
  bool Pre(const ActualArg::PercentRef &x) {  // R1524
    Word("%REF("), Walk(x.v), Put(')');
    return false;
  }
  bool Pre(const ActualArg::PercentVal &x) {
    Word("%VAL("), Walk(x.v), Put(')');
    return false;
  }
  bool Pre(const AltReturnSpec &) {  // R1525
    Put('*');
    return true;
  }
  bool Pre(const FunctionStmt &x) {  // R1530
    Walk("", std::get<std::list<PrefixSpec>>(x.t), " ", " ");
    Word("FUNCTION "), Walk(std::get<Name>(x.t)), Put(" (");
    Walk(std::get<std::list<Name>>(x.t), ", "), Put(')');
    Walk(" ", std::get<std::optional<Suffix>>(x.t)), Indent();
    return false;
  }
  bool Pre(const Suffix &x) {  // R1532
    if (x.resultName) {
      Word("RESULT("), Walk(x.resultName), Put(')');
      Walk(" ", x.binding);
    } else {
      Walk(x.binding);
    }
    return false;
  }
  bool Pre(const EndFunctionStmt &x) {  // R1533
    Outdent(), Word("END FUNCTION"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const SubroutineStmt &x) {  // R1535
    Walk("", std::get<std::list<PrefixSpec>>(x.t), " ", " ");
    Word("SUBROUTINE "), Walk(std::get<Name>(x.t));
    Walk(" (", std::get<std::list<DummyArg>>(x.t), ", ", ")");
    Walk(" ", std::get<std::optional<LanguageBindingSpec>>(x.t));
    Indent();
    return false;
  }
  bool Pre(const EndSubroutineStmt &x) {  // R1537
    Outdent(), Word("END SUBROUTINE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const MpSubprogramStmt &) {  // R1539
    Word("MODULE PROCEDURE "), Indent();
    return true;
  }
  bool Pre(const EndMpSubprogramStmt &x) {  // R1540
    Outdent(), Word("END PROCEDURE"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const EntryStmt &x) {  // R1541
    Word("ENTRY "), Walk(std::get<Name>(x.t));
    Walk(" (", std::get<std::list<DummyArg>>(x.t), ", ", ")");
    Walk(" ", std::get<std::optional<Suffix>>(x.t));
    return false;
  }
  bool Pre(const ReturnStmt &x) {  // R1542
    Word("RETURN"), Walk(" ", x.v);
    return false;
  }
  bool Pre(const ContainsStmt &x) {  // R1543
    Outdent();
    Word("CONTAINS");
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
    Word("POINTER ("), Walk(std::get<0>(x.t)), Put(", ");
    Walk(std::get<1>(x.t));
    Walk("(", std::get<std::optional<ArraySpec>>(x.t), ")"), Put(')');
    return false;
  }
  bool Pre(const StructureStmt &x) {
    Word("STRUCTURE ");
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
  void Post(const Union::UnionStmt &) { Word("UNION"), Indent(); }
  void Post(const Union::EndUnionStmt &) { Outdent(), Word("END UNION"); }
  void Post(const Map::MapStmt &) { Word("MAP"), Indent(); }
  void Post(const Map::EndMapStmt &) { Outdent(), Word("END MAP"); }
  void Post(const StructureDef::EndStructureStmt &) {
    Outdent(), Word("END STRUCTURE");
  }
  bool Pre(const OldParameterStmt &x) {
    Word("PARAMETER "), Walk(x.v, ", ");
    return false;
  }
  bool Pre(const ArithmeticIfStmt &x) {
    Word("IF ("), Walk(std::get<Expr>(x.t)), Put(") ");
    Walk(std::get<1>(x.t)), Put(", ");
    Walk(std::get<2>(x.t)), Put(", ");
    Walk(std::get<3>(x.t));
    return false;
  }
  bool Pre(const AssignStmt &x) {
    Word("ASSIGN "), Walk(std::get<Label>(x.t));
    Word(" TO "), Walk(std::get<Name>(x.t));
    return false;
  }
  bool Pre(const AssignedGotoStmt &x) {
    Word("GO TO "), Walk(std::get<Name>(x.t));
    Walk(", (", std::get<std::list<Label>>(x.t), ", ", ")");
    return false;
  }
  bool Pre(const PauseStmt &x) {
    Word("PAUSE"), Walk(" ", x.v);
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
  void PutKeywordLetter(char);
  void PutQuoted(const std::string &);
  void PutEnum(int, const char *);
  void Word(const char *);
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
      Word(prefix), Walk(*x), Word(suffix);
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
        Word(str), Walk(x);
        str = comma;
      }
      Word(suffix);
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
        Word(separator);
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
  bool capitalizeKeywords_{true};
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

void UnparseVisitor::PutKeywordLetter(char ch) {
  if (capitalizeKeywords_) {
    Put(ToUpperCaseLetter(ch));
  } else {
    Put(ToLowerCaseLetter(ch));
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
    PutKeywordLetter(*p);
  }
}

void UnparseVisitor::Word(const char *str) {
  for (; *str != '\0'; ++str) {
    PutKeywordLetter(*str);
  }
}

void Unparse(std::ostream &out, const Program &program, Encoding encoding,
    bool capitalizeKeywords) {
  UnparseVisitor visitor{out, 1, encoding, capitalizeKeywords};
  Walk(program, visitor);
  visitor.Done();
}
}  // namespace parser
}  // namespace Fortran
