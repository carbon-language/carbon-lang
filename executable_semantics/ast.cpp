// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ast.h"

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include "interp.h"

/***** Utilities *****/

char* input;

/***** Types *****/

auto MakeTypeType(int line_num) -> Expression* {
  auto* t = new Expression();
  t->tag = TypeT;
  t->line_num = line_num;
  return t;
}

auto MakeIntType(int line_num) -> Expression* {
  auto* t = new Expression();
  t->tag = IntT;
  t->line_num = line_num;
  return t;
}

auto MakeBoolType(int line_num) -> Expression* {
  auto* t = new Expression();
  t->tag = BoolT;
  t->line_num = line_num;
  return t;
}

auto MakeAutoType(int line_num) -> Expression* {
  auto* t = new Expression();
  t->tag = AutoT;
  t->line_num = line_num;
  return t;
}

auto MakeFunType(int line_num, Expression* param, Expression* ret)
    -> Expression* {
  auto* t = new Expression();
  t->tag = FunctionT;
  t->line_num = line_num;
  t->u.function_type.parameter = param;
  t->u.function_type.return_type = ret;
  return t;
}

void PrintString(std::string* s) { std::cout << *s; }

/***** Expressions *****/

auto MakeVar(int line_num, std::string var) -> Expression* {
  auto* v = new Expression();
  v->line_num = line_num;
  v->tag = Variable;
  v->u.variable.name = new std::string(std::move(var));
  return v;
}

auto MakeVarPat(int line_num, std::string var, Expression* type) -> Expression* {
  auto* v = new Expression();
  v->line_num = line_num;
  v->tag = PatternVariable;
  v->u.pattern_variable.name = new std::string(std::move(var));
  v->u.pattern_variable.type = type;
  return v;
}

auto MakeInt(int line_num, int i) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = Integer;
  e->u.integer = i;
  return e;
}

auto MakeBool(int line_num, bool b) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = Boolean;
  e->u.boolean = b;
  return e;
}

auto MakeOp(int line_num, enum Operator op, std::vector<Expression*>* args)
    -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = PrimitiveOp;
  e->u.primitive_op.operator_ = op;
  e->u.primitive_op.arguments = args;
  return e;
}

auto MakeUnOp(int line_num, enum Operator op, Expression* arg) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = PrimitiveOp;
  e->u.primitive_op.operator_ = op;
  auto* args = new std::vector<Expression*>();
  args->push_back(arg);
  e->u.primitive_op.arguments = args;
  return e;
}

auto MakeBinOp(int line_num, enum Operator op, Expression* arg1, Expression* arg2)
    -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = PrimitiveOp;
  e->u.primitive_op.operator_ = op;
  auto* args = new std::vector<Expression*>();
  args->push_back(arg1);
  args->push_back(arg2);
  e->u.primitive_op.arguments = args;
  return e;
}

auto MakeCall(int line_num, Expression* fun, Expression* arg) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = Call;
  e->u.call.function = fun;
  e->u.call.argument = arg;
  return e;
}

auto MakeGetField(int line_num, Expression* exp, std::string field)
    -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = GetField;
  e->u.get_field.aggregate = exp;
  e->u.get_field.field = new std::string(std::move(field));
  return e;
}

auto MakeTuple(int line_num,
               std::vector<std::pair<std::string, Expression*> >* args)
    -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = Tuple;
  int i = 0;
  for (auto& arg : *args) {
    if (arg.first == "") {
      arg.first = std::to_string(i);
      ++i;
    }
  }
  e->u.tuple.fields = args;
  return e;
}

auto MakeIndex(int line_num, Expression* exp, Expression* i) -> Expression* {
  auto* e = new Expression();
  e->line_num = line_num;
  e->tag = Index;
  e->u.index.aggregate = exp;
  e->u.index.offset = i;
  return e;
}

void PrintOp(Operator op) {
  switch (op) {
    case Neg:
      std::cout << "-";
      break;
    case Add:
      std::cout << "+";
      break;
    case Sub:
      std::cout << "-";
      break;
    case Not:
      std::cout << "!";
      break;
    case And:
      std::cout << "&&";
      break;
    case Or:
      std::cout << "||";
      break;
    case Eq:
      std::cout << "==";
      break;
  }
}

void PrintFields(std::vector<std::pair<std::string, Expression*> >* fields) {
  int i = 0;
  for (auto iter = fields->begin(); iter != fields->end(); ++iter, ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << iter->first;
    std::cout << " = ";
    PrintExp(iter->second);
  }
}

void PrintExp(Expression* e) {
  switch (e->tag) {
    case Index:
      PrintExp(e->u.index.aggregate);
      std::cout << "[";
      PrintExp(e->u.index.offset);
      std::cout << "]";
      break;
    case GetField:
      PrintExp(e->u.get_field.aggregate);
      std::cout << ".";
      std::cout << *e->u.get_field.field;
      break;
    case Tuple:
      std::cout << "(";
      PrintFields(e->u.tuple.fields);
      std::cout << ")";
      break;
    case Integer:
      std::cout << e->u.integer;
      break;
    case Boolean:
      std::cout << std::boolalpha;
      std::cout << e->u.boolean;
      break;
    case PrimitiveOp:
      std::cout << "(";
      if (e->u.primitive_op.arguments->size() == 0) {
        PrintOp(e->u.primitive_op.operator_);
      } else if (e->u.primitive_op.arguments->size() == 1) {
        PrintOp(e->u.primitive_op.operator_);
        std::cout << " ";
        auto iter = e->u.primitive_op.arguments->begin();
        PrintExp(*iter);
      } else if (e->u.primitive_op.arguments->size() == 2) {
        auto iter = e->u.primitive_op.arguments->begin();
        PrintExp(*iter);
        std::cout << " ";
        PrintOp(e->u.primitive_op.operator_);
        std::cout << " ";
        ++iter;
        PrintExp(*iter);
      }
      std::cout << ")";
      break;
    case Variable:
      std::cout << *e->u.variable.name;
      break;
    case PatternVariable:
      PrintExp(e->u.pattern_variable.type);
      std::cout << ": ";
      std::cout << *e->u.pattern_variable.name;
      break;
    case Call:
      PrintExp(e->u.call.function);
      if (e->u.call.argument->tag == Tuple) {
        PrintExp(e->u.call.argument);
      } else {
        std::cout << "(";
        PrintExp(e->u.call.argument);
        std::cout << ")";
      }
      break;
    case BoolT:
      std::cout << "Bool";
      break;
    case IntT:
      std::cout << "Int";
      break;
    case TypeT:
      std::cout << "Type";
      break;
    case AutoT:
      std::cout << "auto";
      break;
    case FunctionT:
      std::cout << "fn ";
      PrintExp(e->u.function_type.parameter);
      std::cout << " -> ";
      PrintExp(e->u.function_type.return_type);
      break;
  }
}

/***** Expression or Field List *****/

auto MakeExp(Expression* exp) -> ExpOrFieldList* {
  auto e = new ExpOrFieldList();
  e->tag = Exp;
  e->u.exp = exp;
  return e;
}

auto MakeFieldList(std::list<std::pair<std::string, Expression*> >* fields)
    -> ExpOrFieldList* {
  auto e = new ExpOrFieldList();
  e->tag = FieldList;
  e->u.fields = fields;
  return e;
}

auto MakeConstructorField(ExpOrFieldList* e1, ExpOrFieldList* e2)
    -> ExpOrFieldList* {
  auto fields = new std::list<std::pair<std::string, Expression*> >();
  switch (e1->tag) {
    case Exp:
      fields->push_back(std::make_pair("", e1->u.exp));
      break;
    case FieldList:
      for (auto& field : *e1->u.fields) {
        fields->push_back(field);
      }
      break;
  }
  switch (e2->tag) {
    case Exp:
      fields->push_back(std::make_pair("", e2->u.exp));
      break;
    case FieldList:
      for (auto& field : *e2->u.fields) {
        fields->push_back(field);
      }
      break;
  }
  return MakeFieldList(fields);
}

auto EnsureTuple(int line_num, Expression* e) -> Expression* {
  if (e->tag == Tuple) {
    return e;
  } else {
    auto vec = new std::vector<std::pair<std::string, Expression*> >();
    vec->push_back(std::make_pair("", e));
    return MakeTuple(line_num, vec);
  }
}

/***** Statements *****/

auto MakeExpStmt(int line_num, Expression* exp) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = ExpressionStatement;
  s->u.exp = exp;
  return s;
}

auto MakeAssign(int line_num, Expression* lhs, Expression* rhs) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = Assign;
  s->u.assign.lhs = lhs;
  s->u.assign.rhs = rhs;
  return s;
}

auto MakeVarDef(int line_num, Expression* pat, Expression* init) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = VariableDefinition;
  s->u.variable_definition.pat = pat;
  s->u.variable_definition.init = init;
  return s;
}

auto MakeIf(int line_num, Expression* cond, Statement* then_stmt,
            Statement* else_stmt) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = If;
  s->u.if_stmt.cond = cond;
  s->u.if_stmt.then_stmt = then_stmt;
  s->u.if_stmt.else_stmt = else_stmt;
  return s;
}

auto MakeWhile(int line_num, Expression* cond, Statement* body) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = While;
  s->u.while_stmt.cond = cond;
  s->u.while_stmt.body = body;
  return s;
}

auto MakeBreak(int line_num) -> Statement* {
  std::cout << "MakeBlock" << std::endl;
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = Break;
  return s;
}

auto MakeContinue(int line_num) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = Continue;
  return s;
}

auto MakeReturn(int line_num, Expression* e) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = Return;
  s->u.return_stmt = e;
  return s;
}

auto MakeSeq(int line_num, Statement* s1, Statement* s2) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = Sequence;
  s->u.sequence.stmt = s1;
  s->u.sequence.next = s2;
  return s;
}

auto MakeBlock(int line_num, Statement* stmt) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = Block;
  s->u.block.stmt = stmt;
  return s;
}

auto MakeMatch(int line_num, Expression* exp,
               std::list<std::pair<Expression*, Statement*> >* clauses)
    -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = Match;
  s->u.match_stmt.exp = exp;
  s->u.match_stmt.clauses = clauses;
  return s;
}

void PrintStatement(Statement* s, int depth) {
  if (!s) {
    return;
  }
  if (depth == 0) {
    std::cout << " ... ";
    return;
  }
  switch (s->tag) {
    case Match:
      std::cout << "match (";
      PrintExp(s->u.match_stmt.exp);
      std::cout << ") {";
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
        for (auto& clause : *s->u.match_stmt.clauses) {
          std::cout << "case ";
          PrintExp(clause.first);
          std::cout << " =>" << std::endl;
          PrintStatement(clause.second, depth - 1);
          std::cout << std::endl;
        }
      } else {
        std::cout << "...";
      }
      std::cout << "}";
      break;
    case While:
      std::cout << "while (";
      PrintExp(s->u.while_stmt.cond);
      std::cout << ")" << std::endl;
      PrintStatement(s->u.while_stmt.body, depth - 1);
      break;
    case Break:
      std::cout << "break;";
      break;
    case Continue:
      std::cout << "continue;";
      break;
    case VariableDefinition:
      std::cout << "var ";
      PrintExp(s->u.variable_definition.pat);
      std::cout << " = ";
      PrintExp(s->u.variable_definition.init);
      std::cout << ";";
      break;
    case ExpressionStatement:
      PrintExp(s->u.exp);
      std::cout << ";";
      break;
    case Assign:
      PrintExp(s->u.assign.lhs);
      std::cout << " = ";
      PrintExp(s->u.assign.rhs);
      std::cout << ";";
      break;
    case If:
      std::cout << "if (";
      PrintExp(s->u.if_stmt.cond);
      std::cout << ")" << std::endl;
      PrintStatement(s->u.if_stmt.then_stmt, depth - 1);
      std::cout << std::endl << "else" << std::endl;
      PrintStatement(s->u.if_stmt.else_stmt, depth - 1);
      break;
    case Return:
      std::cout << "return ";
      PrintExp(s->u.return_stmt);
      std::cout << ";";
      break;
    case Sequence:
      PrintStatement(s->u.sequence.stmt, depth);
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
      }
      PrintStatement(s->u.sequence.next, depth - 1);
      break;
    case Block:
      std::cout << "{" << std::endl;
      PrintStatement(s->u.block.stmt, depth - 1);
      std::cout << std::endl << "}" << std::endl;
  }
}

/***** Struct Members *****/

auto MakeField(int line_num, std::string name, Expression* type) -> Member* {
  auto m = new Member();
  m->line_num = line_num;
  m->tag = FieldMember;
  m->u.field.name = new std::string(std::move(name));
  m->u.field.type = type;
  return m;
}

/***** Declarations *****/

auto MakeFunDef(int line_num, std::string name, Expression* ret_type,
                Expression* param_pattern, Statement* body)
    -> struct FunctionDefinition* {
  auto* f = new struct FunctionDefinition();
  f->line_num = line_num;
  f->name = std::move(name);
  f->return_type = ret_type;
  f->param_pattern = param_pattern;
  f->body = body;
  return f;
}

auto MakeFunDecl(struct FunctionDefinition* f) -> Declaration* {
  auto* d = new Declaration();
  d->tag = FunctionDeclaration;
  d->u.fun_def = f;
  return d;
}

auto MakeStructDecl(int line_num, std::string name, std::list<Member*>* members)
    -> Declaration* {
  auto* d = new Declaration();
  d->tag = StructDeclaration;
  d->u.struct_def = new struct StructDefinition();
  d->u.struct_def->line_num = line_num;
  d->u.struct_def->name = new std::string(std::move(name));
  d->u.struct_def->members = members;
  return d;
}

auto MakeChoiceDecl(int line_num, std::string name,
                    std::list<std::pair<std::string, Expression*> >* alts)
    -> Declaration* {
  auto* d = new Declaration();
  d->tag = ChoiceDeclaration;
  d->u.choice_def.line_num = line_num;
  d->u.choice_def.name = new std::string(std::move(name));
  d->u.choice_def.alternatives = alts;
  return d;
}

void PrintParams(VarTypes* ps) {
  int i = 0;
  for (auto iter = ps->begin(); iter != ps->end(); ++iter, ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    PrintExp(iter->second);
    std::cout << ": ";
    std::cout << iter->first;
  }
}

void PrintVarDecls(VarTypes* ps) {
  int i = 0;
  for (auto iter = ps->begin(); iter != ps->end(); ++iter, ++i) {
    std::cout << "var ";
    std::cout << iter->first;
    std::cout << " : ";
    PrintExp(iter->second);
    std::cout << "; ";
  }
}

void PrintFunDefDepth(struct FunctionDefinition* f, int depth) {
  std::cout << "fn " << f->name << " ";
  PrintExp(f->param_pattern);
  std::cout << " -> ";
  PrintExp(f->return_type);
  if (f->body) {
    std::cout << " {" << std::endl;
    PrintStatement(f->body, depth);
    std::cout << std::endl << "}" << std::endl;
  } else {
    std::cout << ";" << std::endl;
  }
}

void PrintFunDef(struct FunctionDefinition* f) { PrintFunDefDepth(f, -1); }

void PrintMember(Member* m) {
  switch (m->tag) {
    case FieldMember:
      std::cout << "var " << *m->u.field.name << " : ";
      PrintExp(m->u.field.type);
      std::cout << ";" << std::endl;
      break;
  }
}

void PrintDecl(Declaration* d) {
  switch (d->tag) {
    case FunctionDeclaration:
      PrintFunDef(d->u.fun_def);
      break;
    case StructDeclaration:
      std::cout << "struct " << *d->u.struct_def->name << " {" << std::endl;
      for (auto& member : *d->u.struct_def->members) {
        PrintMember(member);
      }
      std::cout << "}" << std::endl;
      break;
    case ChoiceDeclaration:
      std::cout << "choice " << *d->u.choice_def.name << " {" << std::endl;
      for (auto& alternative : *d->u.choice_def.alternatives) {
        std::cout << "alt " << alternative.first << " ";
        PrintExp(alternative.second);
        std::cout << ";" << std::endl;
      }
      std::cout << "}" << std::endl;
      break;
  }
}

auto ReadFile(FILE* fp) -> char* {
  char* content = nullptr;
  if (fp) {
    fseek(fp, 0, SEEK_END);
    int64_t fsize = ftell(fp);
    rewind(fp);

    content = static_cast<char*>(malloc(sizeof(char) * fsize));
    fread(content, 1, fsize, fp);

    fclose(fp);
  }
  return content;
}
