#include "unparse.h"

#include "format-specification.h"
#include "idioms.h"
#include "indirection.h"
#include "parse-tree-visitor.h"
#include "parse-tree.h"

namespace Fortran {
namespace parser {

class UnparseVisitor {
public:
  // Create an UnparseVisitor that emits the Fortran to this ostream.
  UnparseVisitor(std::ostream &out, const char *indentation = " ")
    : out_{out}, indentation_{indentation} {}

  // Default action for a parse tree node is to visit children.
  template<typename T> bool Pre(const T &x) { return true; }

  template<typename T> void Post(const T &) {}

  template<typename T> void Post(const Statement<T> &x) { Endl(); }

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

  bool Pre(const ContainsStmt &x) {
    Outdent();
    Put("CONTAINS");
    Indent();
    return false;
  }
  bool Pre(const ContinueStmt &x) {
    Put("CONTINUE");
    return false;
  }
  bool Pre(const FailImageStmt &x) {
    Put("FAIL IMAGE");
    return false;
  }
  void Post(const ProgramUnit &x) {
    Put('\n');  // blank line after each ProgramUnit
  }
  bool Pre(const DefinedOpName &x) {
    Put('.');
    Put(x.v);
    Put('.');
    return false;
  }
  bool Pre(const ImportStmt &x) {
    Put("IMPORT");
    switch (x.kind) {
    case ImportStmt::Kind::Default:
      Put(" :: ");
      WalkList(x.names);
      break;
    case ImportStmt::Kind::Only: Put(", ONLY:"); break;
    case ImportStmt::Kind::None: Put(", NONE"); break;
    case ImportStmt::Kind::All: Put(", ALL"); break;
    default: CRASH_NO_CASE;
    }
    return false;
  }
  bool Pre(const NamelistStmt &x) {
    Put("NAMELIST");
    WalkList(x.v);
    return false;
  }
  bool Pre(const NamelistStmt::Group &x) {
    Put('/');
    Put(std::get<Name>(x.t));
    Put('/');
    WalkList(std::get<std::list<Name>>(x.t));
    return false;
  }
  bool Pre(const Star &x) {
    Put('*');
    return false;
  }
  bool Pre(const TypeParamValue::Deferred &x) {
    Put(':');
    return false;
  }
  bool Pre(const KindSelector &x) {
    Put('(');
    Walk(x.v);
    Put(')');
    return false;
  }
  bool Pre(const IntegerTypeSpec &x) {
    Put("INTEGER");
    return true;
  }
  bool Pre(const CharLength &x) {
    std::visit(
        visitors{
            [&](const TypeParamValue &y) {
              Put('(');
              Walk(y);
              Put(')');
            },
            [&](const int64_t &y) { Put(y); },
        },
        x.u);
    return false;
  }
  bool Pre(const LengthSelector &x) {
    std::visit(
        visitors{
            [&](const TypeParamValue &y) {
              Put('(');
              Walk(y);
              Put(')');
            },
            [&](const CharLength &y) {
              Put('*');
              Walk(y);
            },
        },
        x.u);
    return false;
  }
  bool Pre(const CharSelector::LengthAndKind &x) {
    Put('(');
    if (x.length) {
      Put("LEN=");
      Walk(*x.length);
      Put(", ");
    }
    Put("KIND=");
    Walk(x.kind);
    Put(')');
    return false;
  }

  // TODO: rest of parse-tree.h after CharSelector

  bool Pre(const ModuleStmt &x) {
    Put("MODULE ");
    Indent();
    return true;
  }
  bool Pre(const EndModuleStmt &x) {
    Outdent();
    Put("END MODULE");
    return true;
  }
  bool Pre(const MainProgram &x) {
    if (std::get<std::optional<Statement<Name>>>(x.t)) {
      Put("PROGRAM ");
      Indent();
    }
    return true;
  }
  bool Pre(const EndProgramStmt &x) {
    Outdent();
    Put("END PROGRAM");
    return false;
  }
  bool Pre(const TypeDeclarationStmt &x) {
    Walk(std::get<DeclarationTypeSpec>(x.t));
    WalkList(std::get<std::list<AttrSpec>>(x.t), true);
    Put(" :: ");
    WalkList(std::get<std::list<EntityDecl>>(x.t));
    return false;
  }
  bool Pre(const Abstract &x) {
    Put("ABSTRACT");
    return false;
  }
  bool Pre(const Allocatable &x) {
    Put("ALLOCATABLE");
    return false;
  }
  bool Pre(const AssignmentStmt &x) {
    WalkPair(x.t, " = ");
    return false;
  }
  bool Pre(const Expr::Add &x) {
    WalkPair(x.t, " + ");
    return false;
  }
  bool Pre(const Expr::Concat &x) {
    WalkPair(x.t, " // ");
    return false;
  }
  bool Pre(const Expr::Divide &x) {
    WalkPair(x.t, " / ");
    return false;
  }
  bool Pre(const Expr::EQ &x) {
    WalkPair(x.t, " .eq. ");
    return false;
  }
  bool Pre(const Expr::EQV &x) {
    WalkPair(x.t, " .eqv. ");
    return false;
  }
  bool Pre(const Expr::Multiply &x) {
    WalkPair(x.t, " * ");
    return false;
  }
  bool Pre(const Expr::Negate &x) {
    Put("-");
    return true;
  }
  bool Pre(const Expr::Parentheses &x) {
    Put("(");
    return true;
  }
  void Post(const Expr::Parentheses &x) { Put(")"); }
  bool Pre(const Initialization &x) {
    std::visit(
        visitors{[&](const ConstantExpr &y) { Put(" = "); },
            [&](const NullInit &) {}, [&](const auto &) { Put("TODO"); }},
        x.u);
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Character &x) {
    Put("CHARACTER");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Complex &x) {
    Put("COMPLEX");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::DoubleComplex &x) {
    Put("DOUBLE COMPLEX");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::DoublePrecision &x) {
    Put("DOUBLE PRECISION");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Logical &x) {
    Put("LOGICAL");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::NCharacter &x) {
    Put("NCHARACTER");
    return true;
  }
  bool Pre(const IntrinsicTypeSpec::Real &x) {
    Put("REAL");
    return true;
  }
  bool Pre(const KindParam &x) {
    Put("_");
    return true;
  }
  bool Pre(const KindParam::Kanji &x) {
    Put("Kanji???");
    return false;
  }
  bool Pre(const RealLiteralConstant &x) {
    Put(x.intPart);
    ;
    Put(".");
    Put(x.fraction);
    if (x.exponent) {
      Walk(*x.exponent);
    }
    Walk(x.kind);
    return false;
  }

  bool Pre(const DerivedTypeStmt &x) {
    Put("TYPE");
    WalkList(std::get<std::list<TypeAttrSpec>>(x.t), true);
    Put(" :: ");
    Put(std::get<Name>(x.t));
    const auto &params = std::get<std::list<Name>>(x.t);
    if (!params.empty()) {
      Put('(');
      WalkList(params);
      Put(')');
    }
    Indent();
    return false;
  }
  bool Pre(const EndTypeStmt &) {
    Outdent();
    Put("END TYPE");
    return false;
  }
  bool Pre(const TypeAttrSpec::BindC &x) {
    Put("BIND(C)");
    return false;
  }
  bool Pre(const TypeAttrSpec::Extends &x) {
    Put("EXTENDS(");
    return true;
  }
  void Post(const TypeAttrSpec::Extends &x) { Put(")"); }
  bool Pre(const AccessSpec &x) {
    if (x.v == AccessSpec::Kind::Public) {
      Put("PUBLIC");
    } else if (x.v == AccessSpec::Kind::Private) {
      Put("PRIVATE");
    } else {
      CHECK(false);
    }
    return false;
  }
  bool Pre(const SequenceStmt &x) {
    Put("SEQUENCE");
    return false;
  }
  bool Pre(const PrivateStmt &x) {
    Put("PRIVATE");
    return false;
  }
  bool Pre(const DataComponentDefStmt &x) {
    Walk(std::get<DeclarationTypeSpec>(x.t));
    WalkList(std::get<std::list<ComponentAttrSpec>>(x.t), true);
    Put(" :: ");
    WalkList(std::get<std::list<ComponentDecl>>(x.t));
    return false;
  }
  bool Pre(const ProcComponentDefStmt &x) {
    Put("PROCEDURE(");
    Walk(std::get<std::optional<ProcInterface>>(x.t));
    Put(')');
    WalkList(std::get<std::list<ProcComponentAttrSpec>>(x.t), true);
    Put(" :: ");
    WalkList(std::get<std::list<ProcDecl>>(x.t));
    return false;
  }
  bool Pre(const Pass &x) {
    Put("PASS");
    return false;
  }
  bool Pre(const NoPass &x) {
    Put("NOPASS");
    return false;
  }
  bool Pre(const Pointer &x) {
    Put("POINTER");
    return false;
  }
  bool Pre(const ProcPointerInit &x) {
    Put(" => ");
    return true;
  }
  bool Pre(const NullInit &x) {
    Put("NULL()");
    return false;
  }

private:
  std::ostream &out_;
  const char *const indentation_;
  int indent_{0};
  int col_{0};

  void Put(char x) { Put(std::string(1, x)); }

  void Put(const std::string &str) {
    int len = str.length();
    if (len == 0) {
      return;
    }
    if (col_ == 0) {
      for (int i = 0; i < indent_; ++i) {
        out_ << indentation_;
      }
    }
    out_ << str;
    if (str.back() == '\n') {
      col_ = 0;
    } else {
      col_ += len;
    }
  }
  void Endl() {
    if (col_ > 0) {
      out_ << '\n';
      col_ = 0;
    }
  }
  void Indent() { ++indent_; }
  void Outdent() { --indent_; }

  template<typename T> void Walk(const T &x) {
    Fortran::parser::Walk(x, *this);
  }

  // Walk the two elements of a pair, emitting separator between them.
  template<typename T1, typename T2>
  void WalkPair(const std::tuple<T1, T2> &pair, const std::string &separator) {
    Walk(std::get<0>(pair));
    Put(separator);
    Walk(std::get<1>(pair));
  }

  // Walk the elements of list, emitting ", " between them.
  // If atFront is true, emit ", " before the first element as well.
  template<typename T>
  void WalkList(const std::list<T> &list, bool atFront = false) {
    WalkList(list, ", ", atFront);
  }

  // Walk the elements of list, emitting separator between them.
  // If atFront is true, emit separator before the first element as well.
  template<typename T>
  void WalkList(const std::list<T> &list, const std::string &separator,
      bool atFront = false) {
    int n = atFront ? 1 : 0;
    for (const auto &elem : list) {
      if (n++ > 0) Put(separator);
      Walk(elem);
    }
  }
};

void Unparse(std::ostream &out, const Program &program) {
  UnparseVisitor visitor{out};
  Walk(program, visitor);
}

}  // namespace parser
}  // namespace Fortran
