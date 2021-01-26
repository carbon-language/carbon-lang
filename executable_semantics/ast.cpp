// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ast.h"

#include <stdio.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "interp.h"

using std::cout;
using std::endl;
using std::make_pair;

/***** Utilities *****/

char* input;

/***** Types *****/

Expression* MakeTypeType(int lineno) {
  Expression* t = new Expression();
  t->tag = TypeT;
  t->lineno = lineno;
  return t;
}

Expression* MakeIntType(int lineno) {
  Expression* t = new Expression();
  t->tag = IntT;
  t->lineno = lineno;
  return t;
}

Expression* MakeBoolType(int lineno) {
  Expression* t = new Expression();
  t->tag = BoolT;
  t->lineno = lineno;
  return t;
}

Expression* MakeAutoType(int lineno) {
  Expression* t = new Expression();
  t->tag = AutoT;
  t->lineno = lineno;
  return t;
}

Expression* MakeFunType(int lineno, Expression* param, Expression* ret) {
  Expression* t = new Expression();
  t->tag = FunctionT;
  t->lineno = lineno;
  t->u.function_type.parameter = param;
  t->u.function_type.return_type = ret;
  return t;
}

void print_string(std::string* s) { cout << *s; }

/***** Expressions *****/

Expression* MakeVar(int lineno, std::string var) {
  Expression* v = new Expression();
  v->lineno = lineno;
  v->tag = Variable;
  v->u.variable.name = new std::string(var);
  return v;
}

Expression* MakeVarPat(int lineno, std::string var, Expression* type) {
  Expression* v = new Expression();
  v->lineno = lineno;
  v->tag = PatternVariable;
  v->u.pattern_variable.name = new std::string(var);
  v->u.pattern_variable.type = type;
  return v;
}

Expression* MakeInt(int lineno, int i) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = Integer;
  e->u.integer = i;
  return e;
}

Expression* MakeBool(int lineno, bool b) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = Boolean;
  e->u.boolean = b;
  return e;
}

Expression* MakeOp(int lineno, enum Operator op,
                   std::vector<Expression*>* args) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = PrimitiveOp;
  e->u.primitive_op.operator_ = op;
  e->u.primitive_op.arguments = args;
  return e;
}

Expression* MakeUnOp(int lineno, enum Operator op, Expression* arg) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = PrimitiveOp;
  e->u.primitive_op.operator_ = op;
  std::vector<Expression*>* args = new std::vector<Expression*>();
  args->push_back(arg);
  e->u.primitive_op.arguments = args;
  return e;
}

Expression* MakeBinOp(int lineno, enum Operator op, Expression* arg1,
                      Expression* arg2) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = PrimitiveOp;
  e->u.primitive_op.operator_ = op;
  std::vector<Expression*>* args = new std::vector<Expression*>();
  args->push_back(arg1);
  args->push_back(arg2);
  e->u.primitive_op.arguments = args;
  return e;
}

Expression* MakeCall(int lineno, Expression* fun, Expression* arg) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = Call;
  e->u.call.function = fun;
  e->u.call.argument = arg;
  return e;
}

Expression* MakeGetField(int lineno, Expression* exp, std::string field) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = GetField;
  e->u.get_field.aggregate = exp;
  e->u.get_field.field = new std::string(field);
  return e;
}

Expression* MakeTuple(int lineno,
                      std::vector<std::pair<std::string, Expression*> >* args) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = Tuple;
  int i = 0;
  for (auto f = args->begin(); f != args->end(); ++f) {
    if (f->first == "") {
      f->first = std::to_string(i);
      ++i;
    }
  }
  e->u.tuple.fields = args;
  return e;
}

Expression* MakeIndex(int lineno, Expression* exp, Expression* i) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = Index;
  e->u.index.aggregate = exp;
  e->u.index.offset = i;
  return e;
}

void print_op(Operator op) {
  switch (op) {
    case Neg:
      cout << "-";
      break;
    case Add:
      cout << "+";
      break;
    case Sub:
      cout << "-";
      break;
    case Not:
      cout << "!";
      break;
    case And:
      cout << "&&";
      break;
    case Or:
      cout << "||";
      break;
    case Eq:
      cout << "==";
      break;
  }
}

void print_fields(std::vector<std::pair<std::string, Expression*> >* fields) {
  int i = 0;
  for (auto iter = fields->begin(); iter != fields->end(); ++iter, ++i) {
    if (i != 0)
      cout << ", ";
    cout << iter->first;
    cout << " = ";
    PrintExp(iter->second);
  }
}

void PrintExp(Expression* e) {
  switch (e->tag) {
    case Index:
      PrintExp(e->u.index.aggregate);
      cout << "[";
      PrintExp(e->u.index.offset);
      cout << "]";
      break;
    case GetField:
      PrintExp(e->u.get_field.aggregate);
      cout << ".";
      cout << *e->u.get_field.field;
      break;
    case Tuple:
      cout << "(";
      print_fields(e->u.tuple.fields);
      cout << ")";
      break;
    case Integer:
      cout << e->u.integer;
      break;
    case Boolean:
      cout << std::boolalpha;
      cout << e->u.boolean;
      break;
    case PrimitiveOp:
      cout << "(";
      if (e->u.primitive_op.arguments->size() == 0) {
        print_op(e->u.primitive_op.operator_);
      } else if (e->u.primitive_op.arguments->size() == 1) {
        print_op(e->u.primitive_op.operator_);
        cout << " ";
        auto iter = e->u.primitive_op.arguments->begin();
        PrintExp(*iter);
      } else if (e->u.primitive_op.arguments->size() == 2) {
        auto iter = e->u.primitive_op.arguments->begin();
        PrintExp(*iter);
        cout << " ";
        print_op(e->u.primitive_op.operator_);
        cout << " ";
        ++iter;
        PrintExp(*iter);
      }
      cout << ")";
      break;
    case Variable:
      cout << *e->u.variable.name;
      break;
    case PatternVariable:
      PrintExp(e->u.pattern_variable.type);
      cout << ": ";
      cout << *e->u.pattern_variable.name;
      break;
    case Call:
      PrintExp(e->u.call.function);
      if (e->u.call.argument->tag == Tuple) {
        PrintExp(e->u.call.argument);
      } else {
        cout << "(";
        PrintExp(e->u.call.argument);
        cout << ")";
      }
      break;
    case BoolT:
      cout << "Bool";
      break;
    case IntT:
      cout << "Int";
      break;
    case TypeT:
      cout << "Type";
      break;
    case AutoT:
      cout << "auto";
      break;
    case FunctionT:
      cout << "fn ";
      PrintExp(e->u.function_type.parameter);
      cout << " -> ";
      PrintExp(e->u.function_type.return_type);
      break;
  }
}

/***** Expression or Field List *****/

ExpOrFieldList* MakeExp(Expression* exp) {
  auto e = new ExpOrFieldList();
  e->tag = Exp;
  e->u.exp = exp;
  return e;
}

ExpOrFieldList* MakeFieldList(
    std::list<std::pair<std::string, Expression*> >* fields) {
  auto e = new ExpOrFieldList();
  e->tag = FieldList;
  e->u.fields = fields;
  return e;
}

ExpOrFieldList* MakeConstructorField(ExpOrFieldList* e1, ExpOrFieldList* e2) {
  auto fields = new std::list<std::pair<std::string, Expression*> >();
  switch (e1->tag) {
    case Exp:
      fields->push_back(make_pair("", e1->u.exp));
      break;
    case FieldList:
      for (auto i = e1->u.fields->begin(); i != e1->u.fields->end(); ++i) {
        fields->push_back(*i);
      }
      break;
  }
  switch (e2->tag) {
    case Exp:
      fields->push_back(make_pair("", e2->u.exp));
      break;
    case FieldList:
      for (auto i = e2->u.fields->begin(); i != e2->u.fields->end(); ++i) {
        fields->push_back(*i);
      }
      break;
  }
  return MakeFieldList(fields);
}

Expression* ensure_tuple(int lineno, Expression* e) {
  if (e->tag == Tuple) {
    return e;
  } else {
    auto vec = new std::vector<std::pair<std::string, Expression*> >();
    vec->push_back(make_pair("", e));
    return MakeTuple(lineno, vec);
  }
}

/***** Statements *****/

Statement* MakeExpStmt(int lineno, Expression* exp) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = ExpressionStatement;
  s->u.exp = exp;
  return s;
}

Statement* MakeAssign(int lineno, Expression* lhs, Expression* rhs) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Assign;
  s->u.assign.lhs = lhs;
  s->u.assign.rhs = rhs;
  return s;
}

Statement* MakeVarDef(int lineno, Expression* pat, Expression* init) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = VariableDefinition;
  s->u.variable_definition.pat = pat;
  s->u.variable_definition.init = init;
  return s;
}

Statement* MakeIf(int lineno, Expression* cond, Statement* then_stmt,
                  Statement* else_stmt) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = If;
  s->u.if_stmt.cond = cond;
  s->u.if_stmt.then_stmt = then_stmt;
  s->u.if_stmt.else_stmt = else_stmt;
  return s;
}

Statement* MakeWhile(int lineno, Expression* cond, Statement* body) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = While;
  s->u.while_stmt.cond = cond;
  s->u.while_stmt.body = body;
  return s;
}

Statement* MakeBreak(int lineno) {
  cout << "MakeBlock" << endl;
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Break;
  return s;
}

Statement* MakeContinue(int lineno) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Continue;
  return s;
}

Statement* MakeReturn(int lineno, Expression* e) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Return;
  s->u.return_stmt = e;
  return s;
}

Statement* MakeSeq(int lineno, Statement* s1, Statement* s2) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Sequence;
  s->u.sequence.stmt = s1;
  s->u.sequence.next = s2;
  return s;
}

Statement* MakeBlock(int lineno, Statement* stmt) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Block;
  s->u.block.stmt = stmt;
  return s;
}

Statement* MakeMatch(int lineno, Expression* exp,
                     std::list<std::pair<Expression*, Statement*> >* clauses) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Match;
  s->u.match_stmt.exp = exp;
  s->u.match_stmt.clauses = clauses;
  return s;
}

void PrintStatement(Statement* s, int depth) {
  if (!s)
    return;
  if (depth == 0) {
    cout << " ... ";
    return;
  }
  switch (s->tag) {
    case Match:
      cout << "match (";
      PrintExp(s->u.match_stmt.exp);
      cout << ") {";
      if (depth < 0 || depth > 1) {
        cout << endl;
        for (auto c = s->u.match_stmt.clauses->begin();
             c != s->u.match_stmt.clauses->end(); ++c) {
          cout << "case ";
          PrintExp(c->first);
          cout << " =>" << endl;
          PrintStatement(c->second, depth - 1);
          cout << endl;
        }
      } else {
        cout << "...";
      }
      cout << "}";
      break;
    case While:
      cout << "while (";
      PrintExp(s->u.while_stmt.cond);
      cout << ")" << endl;
      PrintStatement(s->u.while_stmt.body, depth - 1);
      break;
    case Break:
      cout << "break;";
      break;
    case Continue:
      cout << "continue;";
      break;
    case VariableDefinition:
      cout << "var ";
      PrintExp(s->u.variable_definition.pat);
      cout << " = ";
      PrintExp(s->u.variable_definition.init);
      cout << ";";
      break;
    case ExpressionStatement:
      PrintExp(s->u.exp);
      cout << ";";
      break;
    case Assign:
      PrintExp(s->u.assign.lhs);
      cout << " = ";
      PrintExp(s->u.assign.rhs);
      cout << ";";
      break;
    case If:
      cout << "if (";
      PrintExp(s->u.if_stmt.cond);
      cout << ")" << endl;
      PrintStatement(s->u.if_stmt.then_stmt, depth - 1);
      cout << endl << "else" << endl;
      PrintStatement(s->u.if_stmt.else_stmt, depth - 1);
      break;
    case Return:
      cout << "return ";
      PrintExp(s->u.return_stmt);
      cout << ";";
      break;
    case Sequence:
      PrintStatement(s->u.sequence.stmt, depth);
      if (depth < 0 || depth > 1)
        cout << endl;
      PrintStatement(s->u.sequence.next, depth - 1);
      break;
    case Block:
      cout << "{" << endl;
      PrintStatement(s->u.block.stmt, depth - 1);
      cout << endl << "}" << endl;
  }
}

/***** Struct Members *****/

Member* MakeField(int lineno, std::string name, Expression* type) {
  auto m = new Member();
  m->lineno = lineno;
  m->tag = FieldMember;
  m->u.field.name = new std::string(name);
  m->u.field.type = type;
  return m;
}

/***** Declarations *****/

struct FunctionDefinition* MakeFunDef(int lineno, std::string name,
                                      Expression* ret_type,
                                      Expression* param_pattern,
                                      Statement* body) {
  struct FunctionDefinition* f = new struct FunctionDefinition();
  f->lineno = lineno;
  f->name = name;
  f->return_type = ret_type;
  f->param_pattern = param_pattern;
  f->body = body;
  return f;
}

Declaration* MakeFunDecl(struct FunctionDefinition* f) {
  Declaration* d = new Declaration();
  d->tag = FunctionDeclaration;
  d->u.fun_def = f;
  return d;
}

Declaration* MakeStructDecl(int lineno, std::string name,
                            std::list<Member*>* members) {
  Declaration* d = new Declaration();
  d->tag = StructDeclaration;
  d->u.struct_def = new struct StructDefinition();
  d->u.struct_def->lineno = lineno;
  d->u.struct_def->name = new std::string(name);
  d->u.struct_def->members = members;
  return d;
}

Declaration* MakeChoiceDecl(
    int lineno, std::string name,
    std::list<std::pair<std::string, Expression*> >* alts) {
  Declaration* d = new Declaration();
  d->tag = ChoiceDeclaration;
  d->u.choice_def.lineno = lineno;
  d->u.choice_def.name = new std::string(name);
  d->u.choice_def.alternatives = alts;
  return d;
}

void PrintParams(VarTypes* ps) {
  int i = 0;
  for (auto iter = ps->begin(); iter != ps->end(); ++iter, ++i) {
    if (i != 0)
      cout << ", ";
    PrintExp(iter->second);
    cout << ": ";
    cout << iter->first;
  }
}

void PrintVarDecls(VarTypes* ps) {
  int i = 0;
  for (auto iter = ps->begin(); iter != ps->end(); ++iter, ++i) {
    cout << "var ";
    cout << iter->first;
    cout << " : ";
    PrintExp(iter->second);
    cout << "; ";
  }
}

void PrintFunDefDepth(struct FunctionDefinition* f, int depth) {
  cout << "fn " << f->name << " ";
  PrintExp(f->param_pattern);
  cout << " -> ";
  PrintExp(f->return_type);
  if (f->body) {
    cout << " {" << endl;
    PrintStatement(f->body, depth);
    cout << endl << "}" << endl;
  } else {
    cout << ";" << endl;
  }
}

void PrintFunDef(struct FunctionDefinition* f) { PrintFunDefDepth(f, -1); }

void PrintMember(Member* m) {
  switch (m->tag) {
    case FieldMember:
      cout << "var " << *m->u.field.name << " : ";
      PrintExp(m->u.field.type);
      cout << ";" << endl;
      break;
  }
}

void PrintDecl(Declaration* d) {
  switch (d->tag) {
    case FunctionDeclaration:
      PrintFunDef(d->u.fun_def);
      break;
    case StructDeclaration:
      cout << "struct " << *d->u.struct_def->name << " {" << endl;
      for (auto m = d->u.struct_def->members->begin();
           m != d->u.struct_def->members->end(); ++m) {
        PrintMember(*m);
      }
      cout << "}" << endl;
      break;
    case ChoiceDeclaration:
      cout << "choice " << *d->u.choice_def.name << " {" << endl;
      for (auto a = d->u.choice_def.alternatives->begin();
           a != d->u.choice_def.alternatives->end(); ++a) {
        cout << "alt " << a->first << " ";
        PrintExp(a->second);
        cout << ";" << endl;
      }
      cout << "}" << endl;
      break;
  }
}

char* ReadFile(FILE* fp) {
  char* fcontent = NULL;
  int fsize = 0;

  if (fp) {
    fseek(fp, 0, SEEK_END);
    fsize = ftell(fp);
    rewind(fp);

    fcontent = (char*)malloc(sizeof(char) * fsize);
    fread(fcontent, 1, fsize, fp);

    fclose(fp);
  }
  return fcontent;
}
