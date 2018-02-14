#include <cstdlib>
#include <iostream>
#include <list>
#include <optional>
#include <sstream>
#include <stddef.h>
#include <string>

#include "../../lib/parser/grammar.h"
#include "../../lib/parser/idioms.h"
#include "../../lib/parser/indirection.h"
#include "../../lib/parser/message.h"
#include "../../lib/parser/parse-state.h"
#include "../../lib/parser/parse-tree.h"
#include "../../lib/parser/position.h"
#include "../../lib/parser/source.h"
#include "../../lib/parser/user-state.h"
#include "../../lib/semantics/attr.h"
#include "../../lib/semantics/type.h"

using namespace Fortran;
using namespace parser;

static void visitProgramUnit(const ProgramUnit &unit);

int main(int argc, char *const argv[]) {
  if (argc != 2) {
    std::cerr << "Expected 1 source file, got " << (argc - 1) << "\n";
    return EXIT_FAILURE;
  }

  std::string path{argv[1]};
  Fortran::parser::SourceFile source;
  std::stringstream error;
  if (!source.Open(path, &error)) {
    std::cerr << error.str() << '\n';
    return EXIT_FAILURE;
  }

  const char *sourceContent{source.content()};
  size_t sourceBytes{source.bytes()};

  Fortran::parser::ParseState state{sourceContent, sourceBytes};
  state.PushContext("source file '"s + path + "'");
  Fortran::parser::UserState ustate;
  state.set_userState(&ustate);

  std::optional<Program> result;
  result = program.Parse(&state);
  if (!result.has_value()) {
    std::cerr << "parse FAILED " << state.position() << '\n'
              << *state.messages();
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
            return semantics::IntegerTypeSpec::make(doKindSelector(x.v));
          },
          [](const IntrinsicTypeSpec::Logical &x) -> returnType {
            return semantics::LogicalTypeSpec::make(doKindSelector(x.kind));
          },
          [](const IntrinsicTypeSpec::Real &x) -> returnType {
            return semantics::RealTypeSpec::make(doKindSelector(x.kind));
          },
          [](const IntrinsicTypeSpec::Complex &x) -> returnType {
            return semantics::ComplexTypeSpec::make(doKindSelector(x.kind));
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
            return semantics::DeclTypeSpec::makeClassStar();
          },
          [](const DeclarationTypeSpec::TypeStar &) {
            return semantics::DeclTypeSpec::makeTypeStar();
          },
          [](const IntrinsicTypeSpec &x) {
            return semantics::DeclTypeSpec::makeIntrinsic(
                doIntrinsicTypeSpec(x));
          },
          [](const auto &x) {
            // TODO
            return semantics::DeclTypeSpec::makeTypeStar();
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
              attrs.set(doAccessSpec(accessSpec));
            },
            [&](const CoarraySpec &) { TODO("CoarraySpec"); },
            [&](const ComponentArraySpec &) { TODO("ComponentArraySpec"); },
            [&](const Allocatable &) {
              attrs.set(semantics::Attr::ALLOCATABLE);
            },
            [&](const Pointer &) { attrs.set(semantics::Attr::POINTER); },
            [&](const Contiguous &) { attrs.set(semantics::Attr::CONTIGUOUS); },
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
              builder.extends(extends.name);
            },
            [&](const TypeAttrSpec::BindC &) {
              builder.attr(semantics::Attr::BIND_C);
            },
            [&](const Abstract &) { builder.attr(semantics::Attr::ABSTRACT); },
            [&](const AccessSpec &accessSpec) {
              switch (accessSpec.v) {
              case AccessSpec::Kind::Public:
                builder.attr(semantics::Attr::PUBLIC);
                break;
              case AccessSpec::Kind::Private:
                builder.attr(semantics::Attr::PRIVATE);
                break;
              default: CRASH_NO_CASE;
              }
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
  std::cout << "SpecificationConstruct\n";
  std::visit(
      visitors{
          [](const Indirection<DerivedTypeDef> &dtd) {
            visitDerivedTypeDef(*dtd);
          },
          [](const auto &x) -> void {
            std::cout << "something else in SpecificationConstruct\n";
          },
      },
      sc.u);
}

static void visitDeclarationConstruct(const DeclarationConstruct &dc) {
  std::cout << "DeclarationConstruct\n";
  std::visit(
      visitors{
          [](const SpecificationConstruct &sc) {
            visitSpecificationConstruct(sc);
          },
          [](const auto &x) -> void {
            std::cout << "something else in DeclarationConstruct\n";
          },
      },
      dc.u);
}

static void visitSpecificationPart(const SpecificationPart &sp) {
  std::cout << "SpecificationPart\n";
  for (const DeclarationConstruct &dc :
      std::get<std::list<DeclarationConstruct>>(sp.t)) {
    visitDeclarationConstruct(dc);
  }
}

static void visitMainProgram(const MainProgram &mp) {
  std::cout << "MainProgram\n";
  visitSpecificationPart(std::get<SpecificationPart>(mp.t));
}

static void visitFunctionSubprogram(const FunctionSubprogram &fs) {
  std::cout << "FunctionSubprogram\n";
}

static void visitProgramUnit(const ProgramUnit &unit) {
  std::visit(
      visitors{
          [](const Indirection<MainProgram> &mp) -> void {
            visitMainProgram(*mp);
          },
          [](const Indirection<FunctionSubprogram> &fs) -> void {
            visitFunctionSubprogram(*fs);
          },
          [](const auto &x) -> void { std::cout << "something else\n"; },
      },
      unit.u);
}
