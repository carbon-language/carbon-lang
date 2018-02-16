#include "../../lib/parser/grammar.h"
#include "../../lib/parser/idioms.h"
#include "../../lib/parser/indirection.h"
#include "../../lib/parser/message.h"
#include "../../lib/parser/parse-state.h"
#include "../../lib/parser/parse-tree.h"
#include "../../lib/parser/preprocessor.h"
#include "../../lib/parser/prescan.h"
#include "../../lib/parser/provenance.h"
#include "../../lib/parser/source.h"
#include "../../lib/parser/user-state.h"
#include "../../lib/semantics/attr.h"
#include "../../lib/semantics/type.h"
#include <cstdlib>
#include <iostream>
#include <list>
#include <optional>
#include <sstream>
#include <string>
#include <stddef.h>

using namespace Fortran;
using namespace parser;

static void visitProgramUnit(const ProgramUnit &unit);

int main(int argc, char *const argv[]) {
  if (argc != 2) {
    std::cerr << "Expected 1 source file, got " << (argc - 1) << "\n";
    return EXIT_FAILURE;
  }

  std::string path{argv[1]};
  AllSources allSources;
  std::stringstream error;
  const auto *sourceFile = allSources.Open(path, &error);
  if (!sourceFile) {
    std::cerr << error.str() << '\n';
    return 1;
  }

  ProvenanceRange range{allSources.AddIncludedFile(
      *sourceFile, ProvenanceRange{})};
  Messages messages{allSources};
  CookedSource cooked{&allSources};
  Preprocessor preprocessor{&allSources};
  bool prescanOk{Prescanner{&messages, &cooked, &preprocessor}.Prescan(range)};
  messages.Emit(std::cerr);
  if (!prescanOk) {
    return EXIT_FAILURE;
  }
  cooked.Marshal();
  ParseState state{cooked};
  UserState ustate;
  std::optional<Program> result{program.Parse(&state)};
  if (!result.has_value() || state.anyErrorRecovery()) {
    std::cerr << "parse FAILED\n";
    state.messages()->Emit(std::cerr);
    return EXIT_FAILURE;
  }
  for (const ProgramUnit &unit : result->v) {
    visitProgramUnit(unit);
  }
  return EXIT_SUCCESS;
}

static semantics::KindParamValue doKindSelector(
    const std::optional<KindSelector> &kind) {
  if (!kind) {
    return semantics::KindParamValue();
  } else {
    const LiteralConstant &lit =
        std::get<LiteralConstant>(kind->v.thing.thing.thing->u);
    const IntLiteralConstant &intlit = std::get<IntLiteralConstant>(lit.u);
    return semantics::KindParamValue(std::get<std::uint64_t>(intlit.t));
  }
}

static const semantics::IntrinsicTypeSpec *doIntrinsicTypeSpec(
    const IntrinsicTypeSpec &its) {
  using returnType = const semantics::IntrinsicTypeSpec *;
  return std::visit(
      visitors{
          [](const IntegerTypeSpec &x) -> returnType {
            return semantics::IntegerTypeSpec::Make(doKindSelector(x.v));
          },
          [](const IntrinsicTypeSpec::Logical &x) -> returnType {
            return semantics::LogicalTypeSpec::Make(doKindSelector(x.kind));
          },
          [](const IntrinsicTypeSpec::Real &x) -> returnType {
            return semantics::RealTypeSpec::Make(doKindSelector(x.kind));
          },
          [](const IntrinsicTypeSpec::Complex &x) -> returnType {
            return semantics::ComplexTypeSpec::Make(doKindSelector(x.kind));
          },
          [](const IntrinsicTypeSpec::DoublePrecision &x) -> returnType {
            return nullptr;  // TODO
          },
          [](const IntrinsicTypeSpec::Character &x) -> returnType {
            return nullptr;  // TODO
          },
          [](const IntrinsicTypeSpec::DoubleComplex &x) -> returnType {
            return nullptr;  // TODO
          },
          [](const IntrinsicTypeSpec::NCharacter &x) -> returnType {
            return nullptr;  // TODO
          },
      },
      its.u);
}

static semantics::DeclTypeSpec doDeclarationTypeSpec(
    const DeclarationTypeSpec &dts) {
  return std::visit(
      visitors{
          [](const DeclarationTypeSpec::ClassStar &) {
            return semantics::DeclTypeSpec::MakeClassStar();
          },
          [](const DeclarationTypeSpec::TypeStar &) {
            return semantics::DeclTypeSpec::MakeTypeStar();
          },
          [](const IntrinsicTypeSpec &x) {
            return semantics::DeclTypeSpec::MakeIntrinsic(
                doIntrinsicTypeSpec(x));
          },
          [](const auto &x) {
            // TODO
            return semantics::DeclTypeSpec::MakeTypeStar();
          },
      },
      dts.u);
}

// TODO: use for each AccessSpec
static semantics::Attr doAccessSpec(const AccessSpec &accessSpec) {
  switch (accessSpec.v) {
  case AccessSpec::Kind::Public: return semantics::Attr::PUBLIC;
  case AccessSpec::Kind::Private: return semantics::Attr::PRIVATE;
  default: CRASH_NO_CASE;
  }
}

#define TODO(x) \
  Fortran::parser::die( \
      "Not yet implemented at " __FILE__ "(%d): %s", __LINE__, x)

static semantics::Attrs doComponentAttrSpec(
    const std::list<ComponentAttrSpec> &cas) {
  semantics::Attrs attrs;
  for (const auto &attr : cas) {
    std::visit(
        visitors{
            [&](const AccessSpec &accessSpec) {
              attrs.Set(doAccessSpec(accessSpec));
            },
            [&](const CoarraySpec &) { TODO("CoarraySpec"); },
            [&](const ComponentArraySpec &) { TODO("ComponentArraySpec"); },
            [&](const Allocatable &) {
              attrs.Set(semantics::Attr::ALLOCATABLE);
            },
            [&](const Pointer &) { attrs.Set(semantics::Attr::POINTER); },
            [&](const Contiguous &) { attrs.Set(semantics::Attr::CONTIGUOUS); },
        },
        attr.u);
  }
  return attrs;
}

static void visitDataComponentDefStmt(const DataComponentDefStmt &stmt,
    semantics::DerivedTypeDefBuilder &builder) {
  const semantics::DeclTypeSpec type =
      doDeclarationTypeSpec(std::get<DeclarationTypeSpec>(stmt.t));
  const semantics::Attrs attrs =
      doComponentAttrSpec(std::get<std::list<ComponentAttrSpec>>(stmt.t));
  for (const auto &decl : std::get<std::list<ComponentDecl>>(stmt.t)) {
    const Name &name = std::get<Name>(decl.t);
    builder.dataComponent(semantics::DataComponentDef(type, name, attrs));
  }
}

// static semantics::DataComponentDef doDataComponentDefStmt(
//    const DataComponentDefStmt &x) {
//  const auto &cd = std::get<std::list<ComponentDecl>>(x.t);
//  const semantics::DeclTypeSpec type =
//  doDeclarationTypeSpec(std::get<DeclarationTypeSpec>(x.t)); const
//  semantics::Attrs attrs =
//      doComponentAttrSpec(std::get<std::list<ComponentAttrSpec>>(x.t));
//  const auto &decls = std::get<std::list<ComponentDecl>>(x.t);
//  //TODO: return a list
//  for (const auto &decl : decls) {
//    const Name &name = std::get<Name>(decl);
//    return DataComponentDef(type, name, attrs);
//  }
//}

static void visitDerivedTypeDef(const DerivedTypeDef &dtd) {
  const DerivedTypeStmt &dts =
      std::get<Statement<DerivedTypeStmt>>(dtd.t).statement;
  const Name &name = std::get<Name>(dts.t);
  semantics::DerivedTypeDefBuilder builder{name};
  for (const TypeAttrSpec &attr : std::get<std::list<TypeAttrSpec>>(dts.t)) {
    std::visit(
        visitors{
            [&](const TypeAttrSpec::Extends &extends) {
              builder.extends(extends.v);
            },
            [&](const TypeAttrSpec::BindC &) {
              builder.attr(semantics::Attr::BIND_C);
            },
            [&](const Abstract &) { builder.attr(semantics::Attr::ABSTRACT); },
            [&](const AccessSpec &accessSpec) {
              builder.attr(doAccessSpec(accessSpec));
            },
        },
        attr.u);
  }
  // TODO: const std::list<Name> &typeParamNames =
  // std::get<std::list<Name>>(dts.t);

  for (const auto &ps :
      std::get<std::list<Statement<PrivateOrSequence>>>(dtd.t)) {
    std::visit(
        visitors{
            [&](const PrivateStmt &) { builder.Private(); },
            [&](const SequenceStmt &) { builder.sequence(); },
        },
        ps.statement.u);
  }
  for (const auto &cds :
      std::get<std::list<Statement<ComponentDefStmt>>>(dtd.t)) {
    std::visit(
        visitors{
            [&](const DataComponentDefStmt &x) {
              visitDataComponentDefStmt(x, builder);
            },
            [&](const ProcComponentDefStmt &x) {
              TODO("ProcComponentDefStmt");
            },
        },
        cds.statement.u);
  }

  semantics::DerivedTypeDef derivedType{builder};
  std::cout << derivedType << "\n";
}

static void visitSpecificationConstruct(const SpecificationConstruct &sc) {
  std::visit(
      visitors{
          [](const Indirection<DerivedTypeDef> &dtd) {
            visitDerivedTypeDef(*dtd);
          },
          [](const Indirection<EnumDef> &x) { TODO("EnumDef"); },
          [](const Statement<Indirection<GenericStmt>> &x) {
            TODO("GenericStmt");
          },
          [](const Indirection<InterfaceBlock> &x) { TODO("InterfaceBlock"); },
          [](const Statement<Indirection<ParameterStmt>> &x) {
            TODO("ParameterStmt");
          },
          [](const Statement<Indirection<ProcedureDeclarationStmt>> &x) {
            TODO("ProcedureDeclarationStmt");
          },
          [](const Statement<OtherSpecificationStmt> &x) {
            TODO("OtherSpecificationStmt");
          },
          [](const Statement<Indirection<TypeDeclarationStmt>> &x) {
            TODO("TypeDeclarationStmt");
          },
          [](const Indirection<StructureDef> &x) { TODO("StructureDef"); },
      },
      sc.u);
}

static void visitDeclarationConstruct(const DeclarationConstruct &dc) {
  if (std::holds_alternative<SpecificationConstruct>(dc.u)) {
    visitSpecificationConstruct(std::get<SpecificationConstruct>(dc.u));
  }
}

static void visitSpecificationPart(const SpecificationPart &sp) {
  for (const DeclarationConstruct &dc :
      std::get<std::list<DeclarationConstruct>>(sp.t)) {
    visitDeclarationConstruct(dc);
  }
}

static void visitProgramUnit(const ProgramUnit &unit) {
  std::visit(
      visitors{
          [](const Indirection<MainProgram> &x) {
            visitSpecificationPart(std::get<SpecificationPart>(x->t));
          },
          [](const Indirection<FunctionSubprogram> &x) {
            visitSpecificationPart(std::get<SpecificationPart>(x->t));
          },
          [](const Indirection<SubroutineSubprogram> &x) {
            visitSpecificationPart(std::get<SpecificationPart>(x->t));
          },
          [](const Indirection<Module> &x) {
            visitSpecificationPart(std::get<SpecificationPart>(x->t));
          },
          [](const Indirection<Submodule> &x) {
            visitSpecificationPart(std::get<SpecificationPart>(x->t));
          },
          [](const auto &x) {},
      },
      unit.u);
}
