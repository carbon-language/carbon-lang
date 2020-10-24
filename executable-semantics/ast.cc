#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include "ast.h"
#include "interp.h"

using std::cout;
using std::endl;
using std::make_pair;

/***** Utilities *****/

char* input;

/***** Types *****/

Expression* make_type_type(int lineno) {
  Expression* t = new Expression();
  t->tag = TypeT;
  t->lineno = lineno;
  return t;
}

Expression* make_int_type(int lineno) {
  Expression* t = new Expression();
  t->tag = IntT;
  t->lineno = lineno;
  return t;
}

Expression* make_bool_type(int lineno) {
  Expression* t = new Expression();
  t->tag = BoolT;
  t->lineno = lineno;
  return t;
}

Expression* make_auto_type(int lineno) {
  Expression* t = new Expression();
  t->tag = AutoT;
  t->lineno = lineno;
  return t;
}

Expression* make_fun_type(int lineno, Expression* param, Expression* ret) {
  Expression* t = new Expression();
  t->tag = FunctionT;
  t->lineno = lineno;
  t->u.function_type.parameter = param;
  t->u.function_type.return_type = ret;
  return t;
}

void print_string(string* s) {
  cout << *s;
}

/***** Expressions *****/

Expression* make_var(int lineno, string var) {
  Expression* v = new Expression();
  v->lineno = lineno;
  v->tag = Variable;
  v->u.variable.name = new string(var);
  return v;
}

Expression* make_var_pat(int lineno, string var, Expression* type) {
  Expression* v = new Expression();
  v->lineno = lineno;
  v->tag = PatternVariable;
  v->u.pattern_variable.name = new string(var);
  v->u.pattern_variable.type = type;
  return v;
}

Expression* make_int(int lineno, int i) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = Integer;
  e->u.integer = i;
  return e;
}

Expression* make_bool(int lineno, bool b) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = Boolean;
  e->u.boolean = b;
  return e;
}

Expression* make_op(int lineno, enum Operator op, vector<Expression*>* args) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = PrimitiveOp;
  e->u.primitive_op.operator_ = op;
  e->u.primitive_op.arguments = args;
  return e;
}

Expression* make_unop(int lineno, enum Operator op, Expression* arg) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = PrimitiveOp;
  e->u.primitive_op.operator_ = op;
  vector<Expression*>* args = new vector<Expression*>();
  args->push_back(arg);
  e->u.primitive_op.arguments = args;
  return e;
}

Expression* make_binop(int lineno, enum Operator op, Expression* arg1, Expression* arg2) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = PrimitiveOp;
  e->u.primitive_op.operator_ = op;
  vector<Expression*>* args = new vector<Expression*>();
  args->push_back(arg1);
  args->push_back(arg2);
  e->u.primitive_op.arguments = args;
  return e;
}

Expression* make_call(int lineno, Expression* fun, Expression* arg) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = Call;
  e->u.call.function = fun;
  e->u.call.argument = arg;
  return e;
}

Expression* make_get_field(int lineno, Expression* exp, string field) {
  Expression* e = new Expression();
  e->lineno = lineno;
  e->tag = GetField;
  e->u.get_field.aggregate = exp;
  e->u.get_field.field = new string(field);
  return e;
}

Expression* make_tuple(int lineno, vector<pair<string,Expression*> >* args) {
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

Expression* make_index(int lineno, Expression* exp, Expression* i) {
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

void print_fields(vector<pair<string,Expression*> >* fields) {
  int i = 0;
  for (auto iter = fields->begin(); iter != fields->end(); ++iter, ++i) {
    if (i != 0)
      cout << ", ";
    cout << iter->first;
    cout << " = ";
    print_exp(iter->second);
  }
}

void print_exp(Expression* e) {
  switch (e->tag) {
  case Index:
    print_exp(e->u.index.aggregate);
    cout << "[";
    print_exp(e->u.index.offset);
    cout << "]";
    break;
  case GetField:
    print_exp(e->u.get_field.aggregate);
    cout << ".";
    cout << * e->u.get_field.field;
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
      print_exp(*iter);
    } else if (e->u.primitive_op.arguments->size() == 2) {
      auto iter = e->u.primitive_op.arguments->begin();
      print_exp(*iter);
      cout << " ";
      print_op(e->u.primitive_op.operator_);
      cout << " ";
      ++iter;
      print_exp(*iter);
    }
    cout << ")";
    break;
  case Variable:
    cout << * e->u.variable.name;
    break;
  case PatternVariable:
    print_exp(e->u.pattern_variable.type);
    cout << ": ";
    cout << * e->u.pattern_variable.name;
    break;
  case Call:
    print_exp(e->u.call.function);
    if (e->u.call.argument->tag == Tuple) {
      print_exp(e->u.call.argument);
    } else {
      cout << "(";
      print_exp(e->u.call.argument);
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
    print_exp(e->u.function_type.parameter);
    cout << " -> ";
    print_exp(e->u.function_type.return_type);
    break;
  }
}

/***** Expression or Field List *****/

ExpOrFieldList* make_exp(Expression* exp) {
  auto e = new ExpOrFieldList();
  e->tag = Exp;
  e->u.exp = exp;
  return e;
}

ExpOrFieldList* make_field_list(list<pair<string,Expression*> >* fields) {
  auto e = new ExpOrFieldList();
  e->tag = FieldList;
  e->u.fields = fields;
  return e;
}

ExpOrFieldList* cons_field(ExpOrFieldList* e1, ExpOrFieldList* e2) {
  auto fields = new list<pair<string,Expression*> >();
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
  return make_field_list(fields);
}

/***** Statements *****/

Statement* make_exp_stmt(int lineno, Expression* exp) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = ExpressionStatement;
  s->u.exp = exp;
  return s;
}

Statement* make_assign(int lineno, Expression* lhs, Expression* rhs) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Assign;
  s->u.assign.lhs = lhs;
  s->u.assign.rhs = rhs;
  return s;
}

Statement* make_var_def(int lineno, Expression* pat, Expression* init) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = VariableDefinition;
  s->u.variable_definition.pat = pat;
  s->u.variable_definition.init = init;
  return s;
}

Statement* make_if(int lineno, Expression* cond, Statement* thn, Statement* els)  {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = If;
  s->u.if_stmt.cond = cond;
  s->u.if_stmt.thn = thn;
  s->u.if_stmt.els = els;
  return s;
}

Statement* make_while(int lineno, Expression* cond, Statement* body) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = While;
  s->u.while_stmt.cond = cond;
  s->u.while_stmt.body = body;
  return s;
}

Statement* make_break(int lineno) {
  cout << "make_block" << endl;
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Break;
  return s;
}

Statement* make_continue(int lineno) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Continue;
  return s;
}

Statement* make_return(int lineno, Expression* e) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Return;
  s->u.return_stmt = e;
  return s;
}

Statement* make_seq(int lineno, Statement* s1, Statement* s2) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Sequence;
  s->u.sequence.stmt = s1;
  s->u.sequence.next = s2;
  return s;
}

Statement* make_block(int lineno, Statement* stmt) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Block;
  s->u.block.stmt = stmt;
  return s;
}

Statement* make_match(int lineno, Expression* exp, list< pair<Expression*,Statement*> >* clauses) {
  Statement* s = new Statement();
  s->lineno = lineno;
  s->tag = Match;
  s->u.match_stmt.exp = exp;
  s->u.match_stmt.clauses = clauses;
  return s;
}

void print_stmt(Statement* s, int depth) {
  if (! s)
    return;
  if (depth == 0) {
    cout << " ... ";
    return;
  }
  switch (s->tag) {
  case Match:
    cout << "match (";
    print_exp(s->u.match_stmt.exp);
    cout << ") {";
    if (depth < 0 || depth > 1) {
      cout << endl;
      for (auto c = s->u.match_stmt.clauses->begin();
           c != s->u.match_stmt.clauses->end(); ++c) {
        cout << "case ";
        print_exp(c->first);
        cout << " =>" << endl;
        print_stmt(c->second, depth - 1);
        cout << endl;
      }
    } else {
      cout << "...";
    }
    cout << "}";
    break;
  case While:
    cout << "while (";
    print_exp(s->u.while_stmt.cond);
    cout << ")" << endl;
    print_stmt(s->u.while_stmt.body, depth - 1);
    break;
  case Break:
    cout << "break;";
    break;
  case Continue:
    cout << "continue;";
    break;
  case VariableDefinition:
    cout << "var ";
    print_exp(s->u.variable_definition.pat);
    cout << " = ";
    print_exp(s->u.variable_definition.init);
    cout << ";";
    break;
  case ExpressionStatement:
    print_exp(s->u.exp);
    cout << ";";
    break;
  case Assign:
    print_exp(s->u.assign.lhs);
    cout << " = ";
    print_exp(s->u.assign.rhs);
    cout << ";";
    break;
  case If:
    cout << "if (";
    print_exp(s->u.if_stmt.cond);
    cout << ")" << endl;
    print_stmt(s->u.if_stmt.thn, depth - 1);
    cout << endl << "else" << endl;
    print_stmt(s->u.if_stmt.els, depth - 1);
    break;
  case Return:
    cout << "return ";
    print_exp(s->u.return_stmt);
    cout << ";";
    break;
  case Sequence:
    print_stmt(s->u.sequence.stmt, depth);
    if (depth < 0 || depth > 1)    
      cout << endl;
    print_stmt(s->u.sequence.next, depth - 1);
    break;
  case Block:
    cout << "{" << endl;
    print_stmt(s->u.block.stmt, depth - 1);
    cout << endl << "}" << endl;
  }
}

/***** Struct Members *****/

Member* make_field(int lineno, string name, Expression* type) {
  auto m = new Member();
  m->lineno = lineno;
  m->tag = FieldMember;
  m->u.field.name = new string(name);
  m->u.field.type = type;
  return m;
}

/***** Declarations *****/

struct FunctionDefinition*
make_fun_def(int lineno, string name, Expression* ret_type,
             Expression* param_pattern, Statement* body) {
  struct FunctionDefinition* f = new struct FunctionDefinition();
  f->lineno = lineno;
  f->name = name;
  f->return_type = ret_type;
  f->param_pattern = param_pattern;
  f->body = body;
  return f;
}

Declaration* make_fun_decl(struct FunctionDefinition* f) {
  Declaration* d = new Declaration();
  d->tag = FunctionDeclaration;
  d->u.fun_def = f;
  return d;
}

Declaration* make_struct_decl(int lineno, string name, list<Member*>* members) {
  Declaration* d = new Declaration();
  d->tag = StructDeclaration;
  d->u.struct_def = new struct StructDefinition();
  d->u.struct_def->lineno = lineno;
  d->u.struct_def->name = new string(name);
  d->u.struct_def->members = members;
  return d;
}

Declaration* make_choice_decl(int lineno, string name,
                       list<pair<string, Expression*> >* alts) {
  Declaration* d = new Declaration();
  d->tag = ChoiceDeclaration;
  d->u.choice_def.lineno = lineno;
  d->u.choice_def.name = new string(name);
  d->u.choice_def.alternatives = alts;
  return d;
}

void print_params(VarTypes* ps) {
  int i = 0;
  for (auto iter = ps->begin(); iter != ps->end(); ++iter, ++i) {
    if (i != 0)
      cout << ", ";
    print_exp(iter->second);
    cout << ": ";
    cout << iter->first;
  }
}

void print_var_decls(VarTypes* ps) {
  int i = 0;
  for (auto iter = ps->begin(); iter != ps->end(); ++iter, ++i) {
    cout << "var ";
    cout << iter->first;
    cout << " : ";
    print_exp(iter->second);
    cout << "; ";
  }
}

void print_fun_def_depth(struct FunctionDefinition* f, int depth) {
  cout << "fn " << f->name << " ";
  print_exp(f->param_pattern);
  cout << " -> ";
  print_exp(f->return_type);
  if (f->body) {
    cout << " {" << endl;
    print_stmt(f->body, depth);
    cout << endl << "}" << endl;
  } else {
    cout << ";" << endl;
  }
}

void print_fun_def(struct FunctionDefinition* f) {
  print_fun_def_depth(f, -1);
}

void print_member(Member* m) {
  switch (m->tag) {
  case FieldMember:
    cout << "var " << * m->u.field.name << " : ";
    print_exp(m->u.field.type);
    cout << ";" << endl;
    break;
  }
}

void print_decl(Declaration* d) {
  switch (d->tag) {
  case FunctionDeclaration:
    print_fun_def(d->u.fun_def);
    break;
  case StructDeclaration:
    cout << "struct " << * d->u.struct_def->name << " {" << endl;
    for (auto m = d->u.struct_def->members->begin();
         m != d->u.struct_def->members->end(); ++m) {
      print_member(*m);
    }
    cout << "}" << endl;
    break;
  case ChoiceDeclaration:
    cout << "choice " << * d->u.choice_def.name << " {" << endl;
    for (auto a = d->u.choice_def.alternatives->begin();
         a != d->u.choice_def.alternatives->end(); ++a) {
      cout << "alt " << a->first << " ";
      print_exp(a->second);
      cout << ";" << endl;
    }
    cout << "}" << endl;
    break;
  }
}


char *read_file(FILE* fp)
{
    char *fcontent = NULL;
    int fsize = 0;

    if(fp) {
        fseek(fp, 0, SEEK_END);
        fsize = ftell(fp);
        rewind(fp);

        fcontent = (char*) malloc(sizeof(char) * fsize);
        fread(fcontent, 1, fsize, fp);

        fclose(fp);
    }
    return fcontent;
}
